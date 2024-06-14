import torch
import torch.nn.functional as F
from safetensors.numpy import save_file, load_file
from omegaconf import OmegaConf
from transformers import AutoConfig
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import json
import os
from argparse import ArgumentParser
import contextlib
from tqdm import tqdm

#
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipelineLegacy,
    StableDiffusionInpaintPipeline,
    DDIMScheduler,
    AutoencoderKL,
)
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, DDIMScheduler
from diffusers import DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler

#
from models.pipeline_mimicbrush import MimicBrushPipeline
from models.ReferenceNet import ReferenceNet
from models.depth_guider import DepthGuider
from mimicbrush import MimicBrush_RefNet
from dataset.data_utils import *

val_configs = OmegaConf.load("./configs/inference.yaml")

# === import Depth Anything ===
import sys

sys.path.append("./depthanything")
from torchvision.transforms import Compose
from depthanything.fast_import import depth_anything_model
from depthanything.depth_anything.util.transform import (
    Resize,
    NormalizeImage,
    PrepareForNet,
)

transform = Compose(
    [
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ]
)
depth_anything_model.load_state_dict(torch.load(val_configs.model_path.depth_model))


# === load the checkpoint ===
base_model_path = val_configs.model_path.pretrained_imitativer_path
vae_model_path = val_configs.model_path.pretrained_vae_name_or_path
image_encoder_path = val_configs.model_path.image_encoder_path
ref_model_path = val_configs.model_path.pretrained_reference_path
mimicbrush_ckpt = val_configs.model_path.mimicbrush_ckpt_path
device = "cuda"


def pad_img_to_square(original_image, is_mask=False):
    width, height = original_image.size

    if height == width:
        return original_image

    if height > width:
        padding = (height - width) // 2
        new_size = (height, height)
    else:
        padding = (width - height) // 2
        new_size = (width, width)

    if is_mask:
        new_image = Image.new("RGB", new_size, "black")
    else:
        new_image = Image.new("RGB", new_size, "white")

    if height > width:
        new_image.paste(original_image, (padding, 0))
    else:
        new_image.paste(original_image, (0, padding))
    return new_image


def collage_region(low, high, mask):
    mask = (np.array(mask) > 128).astype(np.uint8)
    low = np.array(low).astype(np.uint8)
    low = (low * 0).astype(np.uint8)
    high = np.array(high).astype(np.uint8)
    mask_3 = mask
    collage = low * mask_3 + high * (1 - mask_3)
    collage = Image.fromarray(collage)
    return collage


def resize_image_keep_aspect_ratio(image, target_size=512):
    height, width = image.shape[:2]
    if height > width:
        new_height = target_size
        new_width = int(width * (target_size / height))
    else:
        new_width = target_size
        new_height = int(height * (target_size / width))
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image


def crop_padding_and_resize(ori_image, square_image):
    ori_height, ori_width, _ = ori_image.shape
    scale = max(ori_height / square_image.shape[0], ori_width / square_image.shape[1])
    resized_square_image = cv2.resize(
        square_image,
        (int(square_image.shape[1] * scale), int(square_image.shape[0] * scale)),
    )
    padding_size = max(
        resized_square_image.shape[0] - ori_height,
        resized_square_image.shape[1] - ori_width,
    )
    if ori_height < ori_width:
        top = padding_size // 2
        bottom = resized_square_image.shape[0] - (padding_size - top)
        cropped_image = resized_square_image[top:bottom, :, :]
    else:
        left = padding_size // 2
        right = resized_square_image.shape[1] - (padding_size - left)
        cropped_image = resized_square_image[:, left:right, :]
    return cropped_image


def get_args(args):
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args(args)


def get_data(data_path):
    # * columns should be ['bg_image_path', 'bg_mask_path', 'fg_image_path', 'filename']
    data = pd.read_csv(data_path)
    return data


def main():
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )

    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
    unet = UNet2DConditionModel.from_pretrained(
        base_model_path,
        subfolder="unet",
        in_channels=13,
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True,
    ).to(dtype=torch.float16)

    pipe = MimicBrushPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        unet=unet,
        feature_extractor=None,
        safety_checker=None,
    )

    depth_guider = DepthGuider()
    referencenet = ReferenceNet.from_pretrained(ref_model_path, subfolder="unet").to(
        dtype=torch.float16
    )
    mimicbrush_model = MimicBrush_RefNet(
        pipe,
        image_encoder_path,
        mimicbrush_ckpt,
        depth_anything_model,
        depth_guider,
        referencenet,
        device,
    )

    args = get_args(sys.argv[1:])
    data = get_data(args.data_path)

    with contextlib.redirect_stdout(None):
        for _, row in tqdm(data.iterrows()):
            bg_image_path = row["bg_image_path"]
            fg_image_path = row["fg_image_path"]
            bg_mask_path = row["bg_mask_path"]

            if pd.isna(bg_mask_path):
                print(f"No background mask found for image {row['filename']}")
                continue

            bg_image = cv2.imread(bg_image_path)
            bg_image_raw = bg_image.copy()
            bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)
            bg_image = resize_image_keep_aspect_ratio(bg_image)

            fg_image = cv2.imread(fg_image_path)
            fg_image = cv2.cvtColor(fg_image, cv2.COLOR_BGR2RGB)
            fg_image = resize_image_keep_aspect_ratio(fg_image)

            mask = cv2.imread(bg_mask_path)
            bg_mask_raw = mask.copy()

            mask = resize_image_keep_aspect_ratio(mask)
            mask = (mask > 128).astype(np.uint8)
            masked_bg_image = bg_image * (1 - mask)

            fg_image = Image.fromarray(fg_image.astype(np.uint8))
            fg_image = pad_img_to_square(fg_image)

            bg_image = Image.fromarray(bg_image.astype(np.uint8))
            bg_image = pad_img_to_square(bg_image)
            bg_image_low = bg_image

            bg_mask = mask[:, :, 0]
            bg_mask = (
                np.stack([bg_mask, bg_mask, bg_mask], axis=-1).astype(np.uint8) * 255
            )
            bg_mask = Image.fromarray(bg_mask)

            mask = pad_img_to_square(bg_mask, True)
            depth_image = bg_image.copy()
            bg_image = collage_region(bg_image_low, bg_image, mask)

            depth_image = np.array(depth_image)
            depth_image = transform({"image": depth_image})["image"]
            depth_image = torch.from_numpy(depth_image).unsqueeze(0) / 255

            shape_control_flag = 0
            if not shape_control_flag:
                depth_image = depth_image * 0
            pred, depth_pred = mimicbrush_model.generate(
                pil_image=fg_image,
                depth_image=depth_image,
                num_samples=1,
                num_inference_steps=50,
                seed=1,
                image=bg_image,
                mask_image=mask,
                strength=1.0,
                guidance_scale=5,
            )

            depth_pred = F.interpolate(
                depth_pred, size=(512, 512), mode="bilinear", align_corners=True
            )[0][0]
            depth_pred = depth_pred.detach().cpu().numpy().astype(np.uint8)
            depth_pred = cv2.applyColorMap(depth_pred, cv2.COLORMAP_INFERNO)[:, :, ::-1]
            depth_pred = Image.fromarray(depth_pred)

            pred = pred[0]
            pred = np.array(pred).astype(np.uint8)

            pred = crop_padding_and_resize(bg_image_raw, pred)

            mask_alpha = bg_mask_raw
            for i in range(10):
                mask_alpha = cv2.GaussianBlur(mask_alpha, (3, 3), 0)

            mask_alpha_norm = mask_alpha / 255
            pred = pred[:, :, ::-1] * mask_alpha_norm + bg_image_raw * (
                1 - mask_alpha_norm
            )

            save_path = os.path.join(
                args.output_dir, f"{row['filename']}_composite.png"
            )
            cv2.imwrite(save_path, pred)


if __name__ == "__main__":
    main()
