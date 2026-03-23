from pathlib import Path
import argparse
import logging

import torch
from torchvision import transforms
from torchvision.io import write_video
from tqdm import tqdm

from diffusers import (
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
)

from transformers import set_seed
from typing import Dict, Tuple, List
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from safetensors.torch import load_file

import json
import os
import cv2
from PIL import Image

from pathlib import Path
import imageio.v3 as iio
import glob

# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add path for finetune utils
import sys
sys.path.append(os.getcwd())
try:
    from finetune.utils.ref_utils import get_ref_frames_api, save_ref_frames_locally
except ImportError:
    logger.warning("Could not import finetune.utils.ref_utils. API features may disabled.")

# 0 ~ 1
to_tensor = transforms.ToTensor()
video_exts = ['.mp4', '.avi', '.mov', '.mkv']
fr_metrics = ['psnr', 'ssim', 'lpips', 'dists']


def no_grad(func):
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return wrapper


    return any(filename.lower().endswith(ext) for ext in video_exts)


def center_crop_to_aspect_ratio(tensor: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """
    Center sorts a tensor (C, H, W) to match the aspect ratio of target_h/target_w.
    """
    _, src_h, src_w = tensor.shape
    target_ar = target_w / target_h
    src_ar = src_w / src_h

    if abs(target_ar - src_ar) < 1e-3:
        return tensor

    if src_ar > target_ar:
        # Source is wider: crop width
        new_w = int(src_h * target_ar)
        start_w = (src_w - new_w) // 2
        return tensor[:, :, start_w : start_w + new_w]
    else:
        # Source is taller: crop height
        new_h = int(src_w / target_ar)
        start_h = (src_h - new_h) // 2
        return tensor[:, start_h : start_h + new_h, :]


def read_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(to_tensor(Image.fromarray(rgb)))
    cap.release()
    return torch.stack(frames)


def read_image_folder(folder_path):
    image_files = sorted([
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    frames = [to_tensor(Image.open(p).convert("RGB")) for p in image_files]
    return torch.stack(frames)


def load_sequence(path):
    # return a tensor of shape [F, C, H, W] // 0, 1
    if os.path.isdir(path):
        return read_image_folder(path)
    elif os.path.isfile(path):
        if is_video_file(path):
            return read_video_frames(path)
        elif path.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Treat image as a single-frame video
            img = to_tensor(Image.open(path).convert("RGB"))
            return img.unsqueeze(0)  # [1, C, H, W]
    raise ValueError(f"Unsupported input: {path}")

@no_grad
def compute_metrics(pred_frames, gt_frames, metrics_model, metric_accumulator, file_name):

    print(f"\n\n[{file_name}] Metrics:", end=" ")

    # Center crop GT to match pred resolution if misaligned
    if gt_frames is not None:
        pred_h, pred_w = pred_frames.shape[-2], pred_frames.shape[-1]
        gt_h, gt_w = gt_frames.shape[-2], gt_frames.shape[-1]
        if (pred_h, pred_w) != (gt_h, gt_w):
            crop_top = (gt_h - pred_h) // 2
            crop_left = (gt_w - pred_w) // 2
            gt_frames = gt_frames[:, :, crop_top:crop_top + pred_h, crop_left:crop_left + pred_w]
            print(f"[Align] GT {gt_h}x{gt_w} -> center crop to {pred_h}x{pred_w}", end=" ")

    for name, model in metrics_model.items():
        scores = []
        # Ensure lengths match
        min_len = min(pred_frames.shape[0], gt_frames.shape[0])
        for i in range(min_len):
            pred = pred_frames[i].unsqueeze(0)
            if gt_frames is not None:
                gt = gt_frames[i].unsqueeze(0)
            else:
                gt = None
                
            if name in fr_metrics and gt is not None:
                score = model(pred, gt).item()
            else:
                score = model(pred).item()
            scores.append(score)
        val = sum(scores) / len(scores)
        metric_accumulator[name].append(val)
        print(f"{name.upper()}={val:.4f}", end="  ")
    print()


def save_frames_as_png(video, output_dir, fps=8):
    video = video[0]  # Remove batch dimension
    video = video.permute(1, 2, 3, 0)  # [F, H, W, C]

    os.makedirs(output_dir, exist_ok=True)
    frames = (video * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    
    for i, frame in enumerate(frames):
        filename = os.path.join(output_dir, f"{i:03d}.png")
        Image.fromarray(frame).save(filename)


def save_video_with_imageio(video, output_path, fps=8, format='yuv444p'):
    video = video[0]
    video = video.permute(1, 2, 3, 0)

    frames = (video * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()

    if format == 'yuv444p':
        iio.imwrite(
            output_path,
            frames,
            fps=fps,
            codec='libx264',
            pixelformat='yuv444p',
            macro_block_size=None,
            ffmpeg_params=['-crf', '0'],
        )
    else:
        iio.imwrite(
            output_path,
            frames,
            fps=fps,
            codec='libx264',
            pixelformat='yuv420p',
            macro_block_size=None,
            ffmpeg_params=['-crf', '10'],
        )


def preprocess_video_match(
    video_path: Path | str,
    is_match: bool = False,
) -> torch.Tensor:
    if isinstance(video_path, str):
        video_path = Path(video_path)
    video_reader = decord.VideoReader(uri=video_path.as_posix())
    video_num_frames = len(video_reader)
    frames = video_reader.get_batch(list(range(video_num_frames)))
    F, H, W, C = frames.shape
    original_shape = (F, H, W, C)
    
    pad_f = 0
    pad_h = 0
    pad_w = 0

    if is_match:
        remainder = (F - 1) % 8
        if remainder != 0:
            last_frame = frames[-1:]
            pad_f = 8 - remainder
            repeated_frames = last_frame.repeat(pad_f, 1, 1, 1)
            frames = torch.cat([frames, repeated_frames], dim=0)

        pad_h = (4 - H % 4) % 4
        pad_w = (4 - W % 4) % 4
        if pad_h > 0 or pad_w > 0:
            # pad = (w_left, w_right, h_top, h_bottom)
            frames = torch.nn.functional.pad(frames, pad=(0, 0, 0, pad_w, 0, pad_h))  # pad right and bottom

    # to F, C, H, W
    return frames.float().permute(0, 3, 1, 2).contiguous(), pad_f, pad_h, pad_w, original_shape


def remove_padding_and_extra_frames(video, pad_F, pad_H, pad_W):
    if pad_F > 0:
        video = video[:, :, :-pad_F, :, :]
    if pad_H > 0:
        video = video[:, :, :, :-pad_H, :]
    if pad_W > 0:
        video = video[:, :, :, :, :-pad_W]
    
    return video


def make_temporal_chunks(F, chunk_len, overlap_t=8):
    if chunk_len == 0:
        return [(0, F)]

    effective_stride = chunk_len - overlap_t
    if effective_stride <= 0:
        raise ValueError("chunk_len must be greater than overlap")

    chunk_starts = list(range(0, F - overlap_t, effective_stride))
    if chunk_starts[-1] + chunk_len < F:
        chunk_starts.append(F - chunk_len)

    time_chunks = []
    for i, t_start in enumerate(chunk_starts):
        t_end = min(t_start + chunk_len, F)
        time_chunks.append((t_start, t_end))

    if len(time_chunks) >= 2 and time_chunks[-1][1] - time_chunks[-1][0] < chunk_len:
        last = time_chunks.pop()
        prev_start, _ = time_chunks[-1]
        time_chunks[-1] = (prev_start, last[1])

    return time_chunks


def make_spatial_tiles(H, W, tile_size_hw, overlap_hw=(32, 32)):
    tile_height, tile_width = tile_size_hw
    overlap_h, overlap_w = overlap_hw

    if tile_height == 0 or tile_width == 0:
        return [(0, H, 0, W)]

    tile_stride_h = tile_height - overlap_h
    tile_stride_w = tile_width - overlap_w

    if tile_stride_h <= 0 or tile_stride_w <= 0:
        raise ValueError("Tile size must be greater than overlap")

    h_tiles = list(range(0, H - overlap_h, tile_stride_h))
    if not h_tiles or h_tiles[-1] + tile_height < H:
        h_tiles.append(H - tile_height)
    
     # Merge last row if needed
    if len(h_tiles) >= 2 and h_tiles[-1] + tile_height > H:
        h_tiles.pop()

    w_tiles = list(range(0, W - overlap_w, tile_stride_w))
    if not w_tiles or w_tiles[-1] + tile_width < W:
        w_tiles.append(W - tile_width)
    
    # Merge last column if needed
    if len(w_tiles) >= 2 and w_tiles[-1] + tile_width > W:
        w_tiles.pop()

    spatial_tiles = []
    for h_start in h_tiles:
        h_end = min(h_start + tile_height, H)
        if h_end + tile_stride_h > H:
            h_end = H
        for w_start in w_tiles:
            w_end = min(w_start + tile_width, W)
            if w_end + tile_stride_w > W:
                w_end = W
            spatial_tiles.append((h_start, h_end, w_start, w_end))
    return spatial_tiles


def get_valid_tile_region(t_start, t_end, h_start, h_end, w_start, w_end,
                          video_shape, overlap_t, overlap_h, overlap_w):
    _, _, F, H, W = video_shape

    t_len = t_end - t_start
    h_len = h_end - h_start
    w_len = w_end - w_start

    valid_t_start = 0 if t_start == 0 else overlap_t // 2
    valid_t_end = t_len if t_end == F else t_len - overlap_t // 2
    valid_h_start = 0 if h_start == 0 else overlap_h // 2
    valid_h_end = h_len if h_end == H else h_len - overlap_h // 2
    valid_w_start = 0 if w_start == 0 else overlap_w // 2
    valid_w_end = w_len if w_end == W else w_len - overlap_w // 2

    out_t_start = t_start + valid_t_start
    out_t_end = t_start + valid_t_end
    out_h_start = h_start + valid_h_start
    out_h_end = h_start + valid_h_end
    out_w_start = w_start + valid_w_start
    out_w_end = w_start + valid_w_end

    return {
        "valid_t_start": valid_t_start, "valid_t_end": valid_t_end,
        "valid_h_start": valid_h_start, "valid_h_end": valid_h_end,
        "valid_w_start": valid_w_start, "valid_w_end": valid_w_end,
        "out_t_start": out_t_start, "out_t_end": out_t_end,
        "out_h_start": out_h_start, "out_h_end": out_h_end,
        "out_w_start": out_w_start, "out_w_end": out_w_end,
    }

# ==================== REF SPECIFIC LOGIC ====================

def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)

def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    transformer_config: Dict,
    vae_scale_factor_spatial: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:

    grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
    grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)

    p = transformer_config.patch_size
    p_t = transformer_config.patch_size_t

    base_size_width = transformer_config.sample_width // p
    base_size_height = transformer_config.sample_height // p

    if p_t is None:
        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
            device=device,
        )
    else:
        base_num_frames = (num_frames + p_t - 1) // p_t

        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(max(base_size_height, grid_height), max(base_size_width, grid_width)),
            device=device,
        )

    return freqs_cos, freqs_sin

@no_grad
def process_video_ref_i2v(
    pipe: CogVideoXImageToVideoPipeline,
    video: torch.Tensor,
    prompt: str = '',
    ref_frames: List[torch.Tensor] = [],
    ref_indices: List[int] = [],
    chunk_start_idx: int = 0,
    noise_step: int = 0,
    sr_noise_step: int = 399,
    empty_prompt_embedding: torch.Tensor = None,
    ref_guidance_scale: float = 1.0,
):
    # Decode video
    # video: [B, C, F, H, W]
    # pipe.vae.to(video.device, dtype=video.dtype)
    video = video.to(pipe.device, dtype=pipe.dtype)
    latent_dist = pipe.vae.encode(video).latent_dist
    lq_latent = latent_dist.sample() * pipe.vae.config.scaling_factor 
    # lq_latent: [B, 16, F_lat, H_lat, W_lat]
    
    batch_size, num_channels, num_frames, height, width = lq_latent.shape
    device = lq_latent.device
    dtype = lq_latent.dtype

    # Prepare Ref Latent
    full_ref_latent = torch.zeros_like(lq_latent)
    
    for i, idx in enumerate(ref_indices):
        if i >= len(ref_frames): break
        
        # Calculate local index in this chunk
        local_frame_idx = idx - chunk_start_idx
        
        # If idx is outside this chunk, skip
        # Note: video F is in pixels. latent F is F_pix / 4. 
        # local_frame_idx is in pixels.
        
        # Map pixel index to latent index
        target_lat_idx = local_frame_idx // 4
        
        if 0 <= target_lat_idx < num_frames:
             # This reference frame belongs to this latent chunk
             r_frame = ref_frames[i].to(device, dtype=dtype) # [C, H, W]
             
             # Chunk for VAE [1, C, 4, H, W]
             chunk = r_frame.unsqueeze(0).unsqueeze(2).repeat(1, 1, 4, 1, 1)
             
             lat_dist = pipe.vae.encode(chunk).latent_dist
             lat = lat_dist.sample() * pipe.vae.config.scaling_factor
             
             full_ref_latent[:, :, target_lat_idx, :, :] = lat[0, :, 0, :, :]
             
    # --- Dual-Pass / CFG Logic ---
    do_classifier_free_guidance = abs(ref_guidance_scale - 1.0) > 1e-3
    
    if do_classifier_free_guidance:
        # Cond
        input_latent_cond = torch.cat([lq_latent, full_ref_latent], dim=1)
        # Uncond
        uncond_ref_latent = torch.zeros_like(full_ref_latent)
        input_latent_uncond = torch.cat([lq_latent, uncond_ref_latent], dim=1)
        
        # Concatenate batch for parallel forward pass
        input_latent = torch.cat([input_latent_uncond, input_latent_cond], dim=0) # [2*B, C*2, F, H, W]
    else:
        input_latent = torch.cat([lq_latent, full_ref_latent], dim=1) # [B, 32, F, H, W]

    # Handle Patch Size T
    patch_size_t = pipe.transformer.config.patch_size_t
    ncopy = 0
    if patch_size_t is not None:
        ncopy = input_latent.shape[2] % patch_size_t
        first_frame = input_latent[:, :, :1, :, :]
        input_latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), input_latent], dim=2)

    # Encode Prompt
    if prompt == "" and empty_prompt_embedding is not None:
        prompt_embedding = empty_prompt_embedding.to(device, dtype=dtype)
        if prompt_embedding.shape[0] != batch_size:
            prompt_embedding = prompt_embedding.repeat(batch_size, 1, 1)
    else:
        prompt_token_ids = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipe.transformer.config.max_text_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_token_ids = prompt_token_ids.input_ids
        prompt_embedding = pipe.text_encoder(
            prompt_token_ids.to(device)
        )[0]
        _, seq_len, _ = prompt_embedding.shape
        prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=dtype)

    latents = input_latent.permute(0, 2, 1, 3, 4) # [B or 2B, F, C, H, W]
    
    # Expand prompt embedding for CFG
    if do_classifier_free_guidance:
         prompt_embedding = torch.cat([prompt_embedding, prompt_embedding], dim=0)

    # Add Noise
    if noise_step != 0:
         # Separating Lq part
         lq_part = latents[:, :, :16, :, :]
         ref_part = latents[:, :, 16:, :, :]
         
         noise = torch.randn_like(lq_part)
         add_timesteps = torch.full(
            (latents.shape[0],), # Batch size varies
            fill_value=noise_step,
            dtype=torch.long,
            device=device,
        )
         lq_part = pipe.scheduler.add_noise(lq_part.transpose(1, 2), noise.transpose(1, 2), add_timesteps).transpose(1, 2)
         latents = torch.cat([lq_part, ref_part], dim=2)
    
    timesteps = torch.full(
        (latents.shape[0],), # Batch size varies
        fill_value=sr_noise_step,
        dtype=torch.long,
        device=device,
    )

    # RoPE
    vae_scale_factor_spatial = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
    transformer_config = pipe.transformer.config
    rotary_emb = (
        prepare_rotary_positional_embeddings(
            height=height * vae_scale_factor_spatial,
            width=width * vae_scale_factor_spatial,
            num_frames=num_frames, # Use original num_frames (before cat) for PE logic? 
            # Wait, latents F dim is padded by ncopy.
            # PE logic usually handles effective F.
            # But let's check `latents.shape[1]`.
            # In trainer: `num_frames=latents.shape[1]`
            # So here: `num_frames=latents.shape[1]`
            transformer_config=transformer_config,
            vae_scale_factor_spatial=vae_scale_factor_spatial,
            device=device,
        )
        if pipe.transformer.config.use_rotary_positional_embeddings
        else None
    )
    
    # OFS
    ofs = None
    if pipe.transformer.config.ofs_embed_dim is not None:
         ofs = torch.full((latents.shape[0],), fill_value=2.0, device=device, dtype=dtype)

    # Predict
    predicted_noise = pipe.transformer(
        hidden_states=latents,
        encoder_hidden_states=prompt_embedding,
        timestep=timesteps,
        image_rotary_emb=rotary_emb,
        ofs=ofs,
        return_dict=False,
    )[0]
    
    # Denoise
    predicted_noise_slice = predicted_noise[:, :, :16, :, :].transpose(1, 2)
    lq_sample = latents[:, :, :16, :, :].transpose(1, 2)

    # Apply Guidance
    if do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_cond = predicted_noise_slice.chunk(2)
        predicted_noise_slice = noise_pred_uncond + ref_guidance_scale * (noise_pred_cond - noise_pred_uncond)
        
        # Split lq_sample and timesteps for scheduler step (take one half)
        lq_sample = lq_sample.chunk(2)[1] # Take cond part as base? Or uncond? Typically X_t is same.
        timesteps = timesteps.chunk(2)[0]

    latent_generate = pipe.scheduler.get_velocity(
        predicted_noise_slice, lq_sample, timesteps
    )

    if patch_size_t is not None and ncopy > 0:
        latent_generate = latent_generate[:, :, ncopy:, :, :]

    # Decode
    video_generate = pipe.vae.decode(latent_generate / pipe.vae.config.scaling_factor).sample
    video_generate = (video_generate * 0.5 + 0.5).clamp(0.0, 1.0)
    
    return video_generate


def main():
    parser = argparse.ArgumentParser(description="VSR using DOVE Ref I2V")

    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--input_json", type=str, default=None)
    parser.add_argument("--gt_dir", type=str, default=None)
    parser.add_argument("--eval_metrics", type=str, default='')
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="./results")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--upscale_mode", type=str, default="bilinear")
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--output_resolution", type=int, nargs=2, default=None, 
                        help="Target output resolution as H W (e.g., 720 1280 for 720p). Overrides --upscale.")
    parser.add_argument("--noise_step", type=int, default=0)
    parser.add_argument("--sr_noise_step", type=int, default=399)
    parser.add_argument("--is_cpu_offload", action="store_true")
    parser.add_argument("--is_vae_st", action="store_true")
    parser.add_argument("--png_save", action="store_true")
    parser.add_argument("--save_format", type=str, default="yuv444p")
    parser.add_argument("--tile_size_hw", type=int, nargs=2, default=(0, 0))
    parser.add_argument("--overlap_hw", type=int, nargs=2, default=(32, 32))
    parser.add_argument("--chunk_len", type=int, default=0)
    parser.add_argument("--overlap_t", type=int, default=8)
    
    # New Arguments
    # New Arguments
    parser.add_argument("--ref_mode", type=str, default="no_ref", choices=["no_ref", "gt", "api", "pisasr"])
    parser.add_argument("--ref_prompt_mode", type=str, default="fixed", choices=["fixed", "dynamic"], help="fixed: Use static prompt. dynamic: Use VLM analysis.")
    parser.add_argument("--ref_indices", type=int, nargs='*', default=None, help="Manually specify reference frame indices (0-based). Must have interval > 3.")
    parser.add_argument("--ref_guidance_scale", type=float, default=1.0, help="Classifier-Free Guidance scale for reference importance (default: 1.0).")
    parser.add_argument("--ref_api_cache_dir", type=str, default=None, help="Directory to cache API generated reference frames.")
    parser.add_argument("--ref_pisa_cache_dir", type=str, default=None, help="Directory to cache PiSA-SR generated reference frames.")
    parser.add_argument("--pisa_python_executable", type=str, default=None, help="Path to Python executable for PiSA-SR environment (e.g. /path/to/conda/env/bin/python)")
    parser.add_argument("--pisa_script_path", type=str, default=None, help="Path to PiSA-SR test_pisasr.py script")
    parser.add_argument("--pisa_sd_model_path", type=str, default=None, help="Path to PiSA-SR Stable Diffusion base model")
    parser.add_argument("--pisa_chkpt_path", type=str, default=None, help="Path to PiSA-SR pisa_sr.pkl weight")
    parser.add_argument("--pisa_gpu", type=str, default="0", help="GPU ID to run PiSA-SR on")
    
    args = parser.parse_args()

    # Setup
    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
        
    set_seed(args.seed)
    
    # Load Empty Prompt
    empty_prompt_embedding = None
    empty_prompt_path = Path("pretrained_models/prompt_embeddings/e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.safetensors")
    if empty_prompt_path.exists():
        empty_prompt_embedding = load_file(str(empty_prompt_path))["prompt_embedding"]
    
    # Load Video List
    video_files = []
    if os.path.isfile(args.input_dir):
        video_files.append(args.input_dir)
    else:
        for ext in video_exts:
            video_files.extend(glob.glob(os.path.join(args.input_dir, f'*{ext}')))
    video_files = sorted(video_files)
    
    if args.input_json:
        with open(args.input_json, 'r') as f:
            prompt_dict = json.load(f)
    else:
        prompt_dict = {}

    os.makedirs(args.output_path, exist_ok=True)
    
    # Load Pipeline
    print(f"Loading Model from {args.model_path}")
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(args.model_path, torch_dtype=dtype, low_cpu_mem_usage=True)
    
    if args.lora_path:
        print(f"Loading LoRA from {args.lora_path}")
        pipe.load_lora_weights(args.lora_path, adapter_name="dove_ref_i2v")
        pipe.fuse_lora(lora_scale=1.0)
        
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    
    if args.is_cpu_offload:
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to("cuda")
        
    if args.is_vae_st:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        
    # Metrics
    if args.eval_metrics:
        import pyiqa

        metrics_list = [m.strip().lower() for m in args.eval_metrics.split(',')]
        metrics_models = {}
        for name in metrics_list:
            try:
                metrics_models[name] = pyiqa.create_metric(name).to(pipe.device).eval()
            except:
                pass
        metric_accumulator = {name: [] for name in metrics_list}
    else:
        metrics_models = None
        metric_accumulator = None

    # Processing
    from finetune.utils.ref_utils import _select_indices
    
    for video_path in tqdm(video_files, desc="Processing videos"):
        video_name = os.path.basename(video_path)
        prompt = prompt_dict.get(video_name, "")
        
        # Read Video
        video, pad_f, pad_h, pad_w, original_shape = preprocess_video_match(video_path, is_match=True)
        H_orig, W_orig = video.shape[2], video.shape[3]
        
        # Determine GT Path
        gt_path = None
        if args.gt_dir:
            if os.path.isfile(args.gt_dir):
                gt_path = args.gt_dir
            else:
                gt_path = os.path.join(args.gt_dir, video_name)

        # Upscale Input
        if args.output_resolution is not None:
            target_h, target_w = args.output_resolution
            
            # Scale-and-Center-Crop: scale so both dims >= target, then crop center
            src_h, src_w = H_orig, W_orig
            scale_h = target_h / src_h
            scale_w = target_w / src_w
            scale_factor = max(scale_h, scale_w)  # Ensure both dims >= target
            
            scaled_h = int(src_h * scale_factor)
            scaled_w = int(src_w * scale_factor)
            
            print(f"Output Resolution Mode: {target_h}x{target_w}")
            print(f"  Source: {src_h}x{src_w} | Scale: {scale_factor:.4f} -> Scaled: {scaled_h}x{scaled_w}")
            
            # Step 1: Scale up
            video_up = torch.nn.functional.interpolate(
                video, 
                size=(scaled_h, scaled_w),
                mode=args.upscale_mode,
                align_corners=False
            )
            
            # Step 2: Center crop to target
            crop_top = (scaled_h - target_h) // 2
            crop_left = (scaled_w - target_w) // 2
            video_up = video_up[:, :, crop_top:crop_top + target_h, crop_left:crop_left + target_w]
            print(f"  Center crop: top={crop_top} left={crop_left} -> Final: {target_h}x{target_w}")
            
            # Step 3: Pad to VAE-compatible size (multiple of 8)
            pad_h_extra = (8 - target_h % 8) % 8
            pad_w_extra = (8 - target_w % 8) % 8
            if pad_h_extra > 0 or pad_w_extra > 0:
                video_up = torch.nn.functional.pad(video_up, (0, pad_w_extra, 0, pad_h_extra))
                print(f"  VAE pad: +{pad_h_extra}h +{pad_w_extra}w -> {target_h + pad_h_extra}x{target_w + pad_w_extra}")
            
            # Set effective upscale to 1 for downstream padding removal
            effective_upscale = 1
        else:
            video_up = torch.nn.functional.interpolate(
                video, 
                size=(H_orig * args.upscale, W_orig * args.upscale),
                mode=args.upscale_mode,
                align_corners=False
            )
            effective_upscale = args.upscale
        
        # Normalize to [-1, 1]
        video_up = (video_up / 255.0 * 2.0) - 1.0 # From [0, 255] Tensor (preprocess returns 0-255 float range? wait)
        # Check preprocess: `frames.float().permute...` frames are `to_tensor` (which is 0-1) * 255?
        # `preprocess_video_match`: `video_reader` returns uint8. `frames.float()`. 
        # `transforms.ToTensor()` is used in read_video_frames but NOT in preprocess_video_match.
        # preprocess_video_match uses decord which returns format.
        # decord returns [0, 255].
        # So video is [0, 255].
        # So conversion is correct.
        
        video_lr = video
        video = video_up.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous() # [B, C, F, H, W]
        # Wait, normalize video_up first.
        # video_up is [F, C, H, W]
        
        # Retrieve References
        ref_frames_list = []
        if args.ref_mode != "no_ref":
            if args.ref_indices is not None:
                # Manually specified
                ref_indices = sorted(list(set(args.ref_indices)))
                # Validate interval
                if len(ref_indices) > 1:
                    for i in range(len(ref_indices) - 1):
                        if ref_indices[i+1] - ref_indices[i] < 4:
                            raise ValueError(f"Reference frame interval must be > 3 (>= 4). Found interval {ref_indices[i+1] - ref_indices[i]} between {ref_indices[i]} and {ref_indices[i+1]}.")
                if not ref_indices:
                    print(f"Using manually specified indices: NONE (0 reference frames)")
                else:
                    print(f"Using manually specified indices: {ref_indices}")
            else:
                ref_indices = _select_indices(video.shape[2]) # Shape 2 is F now after permute
                print(f"Using auto-selected indices: {ref_indices}")
        else:
            ref_indices = []
        
        if args.ref_mode == "no_ref":
            print("Running in No-Ref mode (0 reference frames).")
            ref_frames_list = []
            ref_indices = []

        elif args.ref_mode == "gt":
             print("Fetching GT Frames...")
             # Assuming standard dataset structure or using this video as source
             saved = save_ref_frames_locally(
                video_path=video_path,
                output_dir=os.path.join(args.output_path, "ref_gt_cache", Path(video_name).stem),
                video_id=Path(video_name).stem,
                is_match=True, # Match padding logic
                specific_indices=ref_indices
             )
             # Reload selected frames
             for idx in ref_indices:
                 # Find file from list
                 found = False
                 for s_idx, s_path in saved:
                     if s_idx == idx:
                         img = Image.open(s_path).convert("RGB")
                         t_img = transforms.ToTensor()(img) # [0, 1]
                         t_img = t_img * 2.0 - 1.0 # [-1, 1]
                         
                         # Align GT ref frame to padded video size (pad instead of resize)
                         target_h, target_w = video.shape[-2], video.shape[-1]
                         if t_img.shape[-2:] != (target_h, target_w):
                             gt_h, gt_w = t_img.shape[-2], t_img.shape[-1]
                             orig_h_up = original_shape[1] * effective_upscale
                             orig_w_up = original_shape[2] * effective_upscale
                             if gt_h == orig_h_up and gt_w == orig_w_up:
                                 # Same base resolution — pad to match
                                 gt_pad_h = target_h - gt_h
                                 gt_pad_w = target_w - gt_w
                                 if gt_pad_h > 0 or gt_pad_w > 0:
                                     t_img = torch.nn.functional.pad(
                                         t_img, (0, gt_pad_w, 0, gt_pad_h),
                                         mode='replicate'
                                     )
                             else:
                                 # Different resolution — resize as fallback
                                 t_img = torch.nn.functional.interpolate(
                                     t_img.unsqueeze(0),
                                     size=(target_h, target_w),
                                     mode="bilinear",
                                     align_corners=False
                                 ).squeeze(0)
                         
                         ref_frames_list.append(t_img)
                         found = True
                         break
                 if not found:
                     # Fallback to current video frame if GT missing?
                     print(f"Warning: GT frame {idx} not found. Using LQ frame.")
                     ref_frames_list.append(video[0, :, idx]) 

        elif args.ref_mode == "pisasr":
             import tempfile
             import subprocess
             import shutil
             print("Generating PiSA-SR Frames...")
             
             if args.ref_pisa_cache_dir:
                 pisa_cache_dir = os.path.join(args.ref_pisa_cache_dir, Path(video_name).stem)
             else:
                 pisa_cache_dir = os.path.join(args.output_path, "ref_pisasr_cache", Path(video_name).stem)
             os.makedirs(pisa_cache_dir, exist_ok=True)
             
             for idx in ref_indices:
                 pisa_frame_path = os.path.join(pisa_cache_dir, f"{video_name}_frame_{idx:05d}.png")
                 
                 found = False
                 if not os.path.exists(pisa_frame_path):
                     print(f"Generating PiSA-SR reference for {video_name} frame {idx}...")
                     lr_frame = video_lr[idx].cpu().permute(1, 2, 0).numpy() # [H, W, C] in [0, 255]
                     with tempfile.TemporaryDirectory() as tmpdir:
                         lr_path = os.path.join(tmpdir, "input_frame.png")
                         lr_img = Image.fromarray(lr_frame.astype('uint8'))
                         lr_img.save(lr_path)
                         
                         out_dir = os.path.join(tmpdir, "out")
                         os.makedirs(out_dir, exist_ok=True)
                         
                         if not all([args.pisa_python_executable, args.pisa_script_path, args.pisa_sd_model_path, args.pisa_chkpt_path]):
                             raise ValueError("PiSA-SR mode requires --pisa_python_executable, --pisa_script_path, --pisa_sd_model_path, and --pisa_chkpt_path to be specified.")
                             
                         cmd = [
                             args.pisa_python_executable,
                             args.pisa_script_path,
                             "--input_image", lr_path,
                             "--output_dir", out_dir,
                             "--pretrained_model_path", args.pisa_sd_model_path,
                             "--pretrained_path", args.pisa_chkpt_path,
                             "--upscale", str(args.upscale),
                             "--align_method", "adain",
                             "--lambda_pix", "1.0",
                             "--lambda_sem", "1.0",
                         ]
                         env = os.environ.copy()
                         env["CUDA_VISIBLE_DEVICES"] = str(args.pisa_gpu)
                         
                         pisa_cwd = os.path.dirname(args.pisa_script_path)
                         try:
                             subprocess.run(cmd, env=env, check=True, capture_output=True, text=True, cwd=pisa_cwd)
                             out_img_path = os.path.join(out_dir, "input_frame.png")
                             if os.path.exists(out_img_path):
                                 shutil.copy(out_img_path, pisa_frame_path)
                                 print(f"PiSA-SR generated for {video_name} frame {idx}.")
                             else:
                                 print(f"Warning: PiSA-SR output missing for {video_name} frame {idx}!")
                         except subprocess.CalledProcessError as e:
                             print(f"PiSA-SR Subprocess failed (exit {e.returncode}): stderr={e.stderr[:500] if e.stderr else 'N/A'}")
                         except Exception as e:
                             print(f"PiSA-SR Subprocess error: {e}")
                 
                 if os.path.exists(pisa_frame_path):
                     img = Image.open(pisa_frame_path).convert("RGB")
                     t_img = transforms.ToTensor()(img) # [0, 1]
                     t_img = t_img * 2.0 - 1.0 # [-1, 1]
                     
                     target_h, target_w = video.shape[-2], video.shape[-1]
                     
                     orig_h, orig_w = t_img.shape[-2], t_img.shape[-1]
                     print(f"[PiSA-SR] Generated HD reference resolution: {orig_w}x{orig_h}")
                     print(f"[PiSA-SR] Target generated video resolution: {target_w}x{target_h}")
                     
                     if t_img.shape[-2:] != (target_h, target_w):
                         t_img = torch.nn.functional.interpolate(
                             t_img.unsqueeze(0),
                             size=(target_h, target_w),
                             mode="bilinear",
                             align_corners=False
                         ).squeeze(0)
                         
                     final_h, final_w = t_img.shape[-2], t_img.shape[-1]
                     print(f"[PiSA-SR] Resized reference resolution: {final_w}x{final_h}")
                         
                     ref_frames_list.append(t_img)
                     found = True
                 
                 if not found:
                     print(f"Warning: PiSA-SR frame {idx} not generated. Using LQ frame.")
                     ref_frames_list.append(video[0, :, idx])

        elif args.ref_mode == "api":
             print("Fetching API Frames...")
             # Need 0-1 input
             vid_01 = (video[0] + 1.0) / 2.0 # [C, F, H, W] or [F, C, H, W]
             # video[0] is [C, F, H, W]
             vid_01 = vid_01.permute(1, 0, 2, 3) # [F, C, H, W]
             
             if args.ref_api_cache_dir:
                 api_cache_base = args.ref_api_cache_dir
             else:
                 api_cache_base = os.path.join(args.output_path, "ref_api_cache")
                 
             target_h, target_w = video.shape[-2], video.shape[-1]
             max_dim = max(target_h, target_w)
             if max_dim <= 1536:
                 api_resolution = "1K"
             elif max_dim <= 3000:
                 api_resolution = "2K"
             else:
                 api_resolution = "4K"
                 
             api_res = get_ref_frames_api(
                output_dir=os.path.join(api_cache_base, Path(video_name).stem),
                video_tensor=vid_01,
                video_id=Path(video_name).stem,
                is_match=True,
                specific_indices=ref_indices,
                ref_prompt_mode=args.ref_prompt_mode,
                resolution=api_resolution
             )
             # Map api indices
             # We trust API returns correct list order or we search
             for idx in ref_indices:
                 found = False
                 for s_idx, s_tensor in api_res:
                      if s_idx == idx:
                          # Resize API result if needed 
                          # Resize API result if needed 
                          # 1. Match Aspect Ratio First (Center Crop)
                          target_h, target_w = video.shape[-2], video.shape[-1]
                          
                          orig_h, orig_w = s_tensor.shape[-2], s_tensor.shape[-1]
                          print(f"[API] Generated HD reference resolution: {orig_w}x{orig_h}")
                          print(f"[API] Target generated video resolution: {target_w}x{target_h}")
                          
                          s_tensor = center_crop_to_aspect_ratio(s_tensor, target_h, target_w)
                          
                          # 2. Resize
                          if s_tensor.shape[-2:] != (target_h, target_w):
                               s_tensor = torch.nn.functional.interpolate(
                                   s_tensor.unsqueeze(0),
                                   size=(target_h, target_w),
                                   mode="bicubic", # Use bicubic for better quality
                                   align_corners=False
                               ).squeeze(0)
                               
                          final_h, final_w = s_tensor.shape[-2], s_tensor.shape[-1]
                          print(f"[API] Resized reference resolution: {final_w}x{final_h}")
                          
                          ref_frames_list.append(s_tensor)
                          found = True
                          break
                 if not found:
                     ref_frames_list.append(video[0, :, idx])

        # Tiling Setup
        B, C, F, H, W = video.shape
        overlap_t = args.overlap_t if args.chunk_len > 0 else 0
        overlap_hw = args.overlap_hw if args.tile_size_hw != (0,0) else (0,0)
        
        time_chunks = make_temporal_chunks(F, args.chunk_len, overlap_t)
        spatial_tiles = make_spatial_tiles(H, W, args.tile_size_hw, overlap_hw)
        
        output_video = torch.zeros_like(video)
        write_count = torch.zeros_like(video, dtype=torch.int)

        print(f"Processing: F={F} H={H} W={W} | Chunks={len(time_chunks)} Tiles={len(spatial_tiles)}")

        for t_start, t_end in time_chunks:
            for h_start, h_end, w_start, w_end in spatial_tiles:
                video_chunk = video[:, :, t_start:t_end, h_start:h_end, w_start:w_end]
                
                # Check Refs for Spatial Tile (Optimization: Crop Refs)
                # We need to crop reference frames to match the current spatial tile
                current_ref_frames = []
                for rf in ref_frames_list:
                    # rf is [C, VideoH, VideoW]
                    rf_crop = rf[:, h_start:h_end, w_start:w_end]
                    current_ref_frames.append(rf_crop)

                _video_generate = process_video_ref_i2v(
                    pipe=pipe,
                    video=video_chunk,
                    prompt=prompt,
                    ref_frames=current_ref_frames,
                    ref_indices=ref_indices,
                    chunk_start_idx=t_start,
                    noise_step=args.noise_step,
                    sr_noise_step=args.sr_noise_step,
                    empty_prompt_embedding=empty_prompt_embedding,
                    ref_guidance_scale=args.ref_guidance_scale,
                )
                
                region = get_valid_tile_region(
                    t_start, t_end, h_start, h_end, w_start, w_end,
                    video.shape, overlap_t, overlap_hw[0], overlap_hw[1]
                )
                
                output_video[:, :, region["out_t_start"]:region["out_t_end"],
                                region["out_h_start"]:region["out_h_end"],
                                region["out_w_start"]:region["out_w_end"]] = \
                _video_generate[:, :, region["valid_t_start"]:region["valid_t_end"],
                                region["valid_h_start"]:region["valid_h_end"],
                                region["valid_w_start"]:region["valid_w_end"]]
                
                write_count[:, :, region["out_t_start"]:region["out_t_end"],
                                region["out_h_start"]:region["out_h_end"],
                                region["out_w_start"]:region["out_w_end"]] += 1

        video_generate = output_video
        
        # Save
        video_generate = remove_padding_and_extra_frames(video_generate, pad_f, pad_h*effective_upscale, pad_w*effective_upscale)
        file_name = os.path.basename(video_path)
        
        out_file_path = os.path.join(args.output_path, file_name)
        if args.png_save:
             save_frames_as_png(video_generate, out_file_path.rsplit('.', 1)[0], fps=args.fps)
        else:
             out_file_path = out_file_path.replace('.mkv', '.mp4')
             save_video_with_imageio(video_generate, out_file_path, fps=args.fps, format=args.save_format)
             
        # Metrics
        if metrics_models is not None:
             pred_frames = video_generate[0].permute(1, 0, 2, 3) 
             gt_frames = None
             if args.gt_dir:
                 if os.path.isfile(args.gt_dir):
                     gt_path = args.gt_dir
                 else:
                     gt_path = os.path.join(args.gt_dir, file_name)
                 
                 try:
                     gt_frames = load_sequence(gt_path)
                 except:
                     pass
             
             if gt_frames is not None:
                 compute_metrics(pred_frames, gt_frames, metrics_models, metric_accumulator, file_name)
             else:
                 print(f"Skipping metrics for {file_name}: GT not found or load failed.")

    if args.eval_metrics and metrics_models:
        print("\n\n" + "="*50)
        print("AVERAGE METRICS")
        print("="*50)
        
        final_metrics = {}
        for name, values in metric_accumulator.items():
            if len(values) > 0:
                avg = sum(values) / len(values)
                print(f"{name.upper()}: {avg:.4f}")
                final_metrics[name] = avg
        print("="*50 + "\n")
        
        # Save metric summary
        out_json_path = os.path.join(args.output_path, "all_metrics.json")
        with open(out_json_path, "w") as f:
            json.dump(final_metrics, f, indent=4)
        print(f"Metrics saved to {out_json_path}")

if __name__ == "__main__":
    main()
