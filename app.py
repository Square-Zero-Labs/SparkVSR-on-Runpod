from __future__ import annotations

import contextlib
import gc
import imageio
import os
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Iterable

import cv2
import gradio as gr
import torch


APP_ROOT = Path(os.environ.get("SPARKVSR_WORKSPACE_DIR", Path(__file__).resolve().parent)).resolve()
SUBTREE_ROOT = APP_ROOT / "SparkVSR-base"
INFERENCE_SCRIPT = SUBTREE_ROOT / "sparkvsr_inference_script.py"
MODEL_PATH = Path(os.environ.get("SPARKVSR_MODEL_PATH", APP_ROOT / "checkpoints" / "sparkvsr-s2" / "ckpt-500-sft")).resolve()
LOG_DIR = APP_ROOT / "logs"
OUTPUT_ROOT = APP_ROOT / "out"
INPUT_ROOT = APP_ROOT / "in"
MANUAL_REF_ROWS = 8
SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
DEFAULT_CHUNK_LEN = max(0, int(os.environ.get("SPARKVSR_CHUNK_LEN", "25")))
DEFAULT_OVERLAP_T = max(0, int(os.environ.get("SPARKVSR_OVERLAP_T", "8")))

if str(SUBTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBTREE_ROOT))

import sparkvsr_inference_script as spark_module
from finetune.utils.ref_utils import _select_indices, get_ref_frames_api, save_ref_frames_locally


PIPELINE_STATE = {
    "pipe": None,
    "cpu_offload": None,
    "empty_prompt_embedding": None,
}


def ensure_runtime_dirs() -> None:
    for path in (LOG_DIR, OUTPUT_ROOT, INPUT_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def slugify_stem(name: str) -> str:
    stem = Path(name).stem or "input"
    safe = "".join(char if char.isalnum() or char in ("-", "_") else "-" for char in stem)
    return safe.strip("-") or "input"


def parse_optional_resolution(raw_value: str) -> tuple[int, int] | None:
    value = (raw_value or "").strip().lower()
    if not value:
        return None
    normalized = value.replace(" ", "").replace("x", ",")
    parts = [part for part in normalized.split(",") if part]
    if len(parts) != 2:
        raise ValueError("Output resolution must look like `720x1280`.")
    height, width = (int(part) for part in parts)
    if height <= 0 or width <= 0:
        raise ValueError("Output resolution must be positive.")
    return height, width


def parse_ref_indices(raw_value: str) -> list[int]:
    value = (raw_value or "").strip()
    if not value:
        return []
    try:
        indices = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    except ValueError as exc:
        raise ValueError("Reference indices must be comma-separated integers.") from exc
    validate_reference_indices(indices)
    return indices


def validate_reference_indices(indices: Iterable[int]) -> None:
    indices = list(indices)
    if any(index < 0 for index in indices):
        raise ValueError("Reference indices must be zero or greater.")
    for previous, current in zip(indices, indices[1:]):
        if current - previous < 4:
            raise ValueError("Reference indices must be at least 4 frames apart.")


def resize_to_cover(image_bgr, width: int, height: int):
    src_h, src_w = image_bgr.shape[:2]
    scale = max(width / src_w, height / src_h)
    resized = cv2.resize(image_bgr, (max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))), interpolation=cv2.INTER_CUBIC)
    start_x = max(0, (resized.shape[1] - width) // 2)
    start_y = max(0, (resized.shape[0] - height) // 2)
    return resized[start_y:start_y + height, start_x:start_x + width]


def get_video_fps(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
    finally:
        cap.release()
    return int(round(fps)) if fps and fps > 1 else 16


def copy_or_convert_video(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() == ".mp4":
        shutil.copy2(source, destination)
        return

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(destination),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to normalize the input video: {completed.stderr.strip()}")


def collect_manual_references(values: list[object]) -> list[tuple[int, Path]]:
    references: list[tuple[int, Path]] = []
    for row_index in range(MANUAL_REF_ROWS):
        base = row_index * 3
        enabled = bool(values[base])
        frame_index = values[base + 1]
        image_value = values[base + 2]
        if not enabled and frame_index in (None, "") and not image_value:
            continue
        if not enabled:
            enabled = bool(image_value) or frame_index not in (None, "")
        if not enabled:
            continue
        if frame_index in (None, ""):
            raise ValueError(f"Manual reference row {row_index + 1} is missing a frame index.")
        if not image_value:
            raise ValueError(f"Manual reference row {row_index + 1} is missing an image.")
        try:
            index_value = int(frame_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Manual reference row {row_index + 1} has an invalid frame index.") from exc

        image_path = Path(image_value)
        if not image_path.exists():
            raise ValueError(f"Manual reference row {row_index + 1} points to a missing image file.")
        references.append((index_value, image_path))

    references.sort(key=lambda item: item[0])
    validate_reference_indices(index for index, _ in references)
    if len({index for index, _ in references}) != len(references):
        raise ValueError("Manual reference indices must be unique.")
    return references


def build_synthetic_gt_video(
    source_video: Path,
    destination_lq: Path,
    destination_gt: Path,
    references: list[tuple[int, Path]],
    upscale: int,
    output_resolution: tuple[int, int] | None,
) -> int:
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise ValueError("Could not open the uploaded input video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    fps_value = int(round(fps)) if fps and fps > 1 else 16
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_height, target_width = output_resolution if output_resolution else (height * upscale, width * upscale)

    destination_lq.parent.mkdir(parents=True, exist_ok=True)
    destination_gt.parent.mkdir(parents=True, exist_ok=True)
    copy_or_convert_video(source_video, destination_lq)

    prepared_refs: dict[int, object] = {}
    for frame_index, image_path in references:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read manual reference image `{image_path.name}`.")
        prepared_refs[frame_index] = resize_to_cover(image, target_width, target_height)

    writer = cv2.VideoWriter(
        str(destination_gt),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_value,
        (target_width, target_height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create the synthetic GT video for manual references.")

    try:
        frame_index = 0
        while True:
            success, frame = cap.read()
            if not success:
                break
            base_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
            writer.write(prepared_refs.get(frame_index, base_frame))
            frame_index += 1
    finally:
        writer.release()
        cap.release()

    return fps_value


def find_generated_video(output_dir: Path) -> Path:
    candidates = []
    for suffix in sorted(SUPPORTED_VIDEO_SUFFIXES):
        candidates.extend(sorted(output_dir.glob(f"*{suffix}")))
    if not candidates:
        raise RuntimeError("SparkVSR completed without producing an output video.")
    return candidates[0]


def mode_to_ref_mode(mode: str) -> str:
    if mode == "No Reference":
        return "no_ref"
    if mode == "API Reference":
        return "api"
    if mode == "Manual References":
        return "gt"
    raise ValueError(f"Unsupported mode: {mode}")


def format_duration(total_seconds: float) -> str:
    seconds = max(0, int(round(total_seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    parts: list[str] = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs or not parts:
        parts.append(f"{secs}s")
    return " ".join(parts)


def format_progress(current: int, total: int) -> str:
    if total <= 0:
        return f"{current}/0"
    percent = current / total * 100.0
    return f"{current}/{total} ({percent:.1f}%)"


def get_model_status_text() -> str:
    pipe = PIPELINE_STATE["pipe"]
    if pipe is None:
        return "Model status: unloaded"
    if PIPELINE_STATE["cpu_offload"]:
        return "Model status: loaded with CPU offload"
    return "Model status: loaded on GPU"


def clear_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unload_loaded_pipeline() -> str:
    pipe = PIPELINE_STATE["pipe"]
    if pipe is not None:
        try:
            pipe.to("cpu")
        except Exception:
            pass
    PIPELINE_STATE["pipe"] = None
    PIPELINE_STATE["cpu_offload"] = None
    PIPELINE_STATE["empty_prompt_embedding"] = None
    clear_cuda_memory()
    return get_model_status_text()


def resolve_dtype():
    return torch.bfloat16


def load_empty_prompt_embedding():
    empty_prompt_path = SUBTREE_ROOT / "pretrained_models" / "prompt_embeddings" / "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.safetensors"
    if empty_prompt_path.exists():
        return spark_module.load_file(str(empty_prompt_path))["prompt_embedding"]
    return None


def ensure_pipeline_loaded(cpu_offload: bool) -> tuple[object, object, bool]:
    current_pipe = PIPELINE_STATE["pipe"]
    current_mode = PIPELINE_STATE["cpu_offload"]
    reused = current_pipe is not None and current_mode == cpu_offload
    if reused:
        return current_pipe, PIPELINE_STATE["empty_prompt_embedding"], True

    if current_pipe is not None:
        unload_loaded_pipeline()

    pipe = spark_module.CogVideoXImageToVideoPipeline.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=resolve_dtype(),
        low_cpu_mem_usage=True,
    )
    pipe.scheduler = spark_module.CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    if cpu_offload:
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to("cuda")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    PIPELINE_STATE["pipe"] = pipe
    PIPELINE_STATE["cpu_offload"] = cpu_offload
    PIPELINE_STATE["empty_prompt_embedding"] = load_empty_prompt_embedding()
    return pipe, PIPELINE_STATE["empty_prompt_embedding"], False


def get_chunk_settings(total_frames: int) -> tuple[int, int]:
    if DEFAULT_CHUNK_LEN <= 0 or total_frames <= DEFAULT_CHUNK_LEN:
        return 0, 0
    overlap = min(DEFAULT_OVERLAP_T, DEFAULT_CHUNK_LEN - 1)
    return DEFAULT_CHUNK_LEN, overlap


def decord_to_torch(frames) -> torch.Tensor:
    if isinstance(frames, torch.Tensor):
        return frames
    if hasattr(frames, "asnumpy"):
        return torch.from_numpy(frames.asnumpy())
    if hasattr(frames, "numpy"):
        return torch.from_numpy(frames.numpy())
    return torch.as_tensor(frames)


def read_video_metadata(video_path: Path) -> dict[str, int | Path | object]:
    video_reader = spark_module.decord.VideoReader(uri=video_path.as_posix())
    total_frames = len(video_reader)
    first_frame = decord_to_torch(video_reader[0])
    height = int(first_frame.shape[0])
    width = int(first_frame.shape[1])
    channels = int(first_frame.shape[2])

    pad_f = 0
    remainder = (total_frames - 1) % 8
    if remainder != 0:
        pad_f = 8 - remainder

    pad_h = (4 - height % 4) % 4
    pad_w = (4 - width % 4) % 4

    return {
        "video_path": video_path,
        "video_reader": video_reader,
        "total_frames": total_frames,
        "padded_frames": total_frames + pad_f,
        "height": height,
        "width": width,
        "channels": channels,
        "pad_f": pad_f,
        "pad_h": pad_h,
        "pad_w": pad_w,
    }


def compute_output_geometry(metadata: dict[str, int | Path | object], upscale: int, output_resolution: tuple[int, int] | None) -> dict[str, int | float | None]:
    input_height = int(metadata["height"]) + int(metadata["pad_h"])
    input_width = int(metadata["width"]) + int(metadata["pad_w"])

    if output_resolution is not None:
        final_output_h, final_output_w = output_resolution
        scale_h = final_output_h / input_height
        scale_w = final_output_w / input_width
        scale_factor = max(scale_h, scale_w)
        scaled_h = int(input_height * scale_factor)
        scaled_w = int(input_width * scale_factor)
        crop_top = (scaled_h - final_output_h) // 2
        crop_left = (scaled_w - final_output_w) // 2
        output_pad_h = (8 - final_output_h % 8) % 8
        output_pad_w = (8 - final_output_w % 8) % 8
        process_output_h = final_output_h + output_pad_h
        process_output_w = final_output_w + output_pad_w
        effective_upscale = 1
    else:
        scale_factor = None
        scaled_h = None
        scaled_w = None
        crop_top = 0
        crop_left = 0
        output_pad_h = int(metadata["pad_h"]) * upscale
        output_pad_w = int(metadata["pad_w"]) * upscale
        process_output_h = input_height * upscale
        process_output_w = input_width * upscale
        final_output_h = int(metadata["height"]) * upscale
        final_output_w = int(metadata["width"]) * upscale
        effective_upscale = upscale

    return {
        "input_height_padded": input_height,
        "input_width_padded": input_width,
        "process_output_h": process_output_h,
        "process_output_w": process_output_w,
        "final_output_h": final_output_h,
        "final_output_w": final_output_w,
        "output_pad_h": output_pad_h,
        "output_pad_w": output_pad_w,
        "scale_factor": scale_factor,
        "scaled_h": scaled_h,
        "scaled_w": scaled_w,
        "crop_top": crop_top,
        "crop_left": crop_left,
        "effective_upscale": effective_upscale,
    }


def read_video_chunk(video_reader, start_frame: int, end_frame: int, total_frames: int, pad_h: int, pad_w: int) -> torch.Tensor:
    read_indices = [min(index, total_frames - 1) for index in range(start_frame, end_frame)]
    frames = decord_to_torch(video_reader.get_batch(read_indices))
    if pad_h > 0 or pad_w > 0:
        frames = torch.nn.functional.pad(frames, pad=(0, 0, 0, pad_w, 0, pad_h))
    return frames.float().permute(0, 3, 1, 2).contiguous()


def prepare_video_chunk_for_model(video_chunk: torch.Tensor, geometry: dict[str, int | float | None], upscale_mode: str) -> torch.Tensor:
    if geometry["scale_factor"] is not None:
        video_up = torch.nn.functional.interpolate(
            video_chunk,
            size=(int(geometry["scaled_h"]), int(geometry["scaled_w"])),
            mode=upscale_mode,
            align_corners=False,
        )
        crop_top = int(geometry["crop_top"])
        crop_left = int(geometry["crop_left"])
        final_h = int(geometry["final_output_h"])
        final_w = int(geometry["final_output_w"])
        video_up = video_up[:, :, crop_top:crop_top + final_h, crop_left:crop_left + final_w]
        if int(geometry["output_pad_h"]) > 0 or int(geometry["output_pad_w"]) > 0:
            video_up = torch.nn.functional.pad(
                video_up,
                (0, int(geometry["output_pad_w"]), 0, int(geometry["output_pad_h"])),
            )
    else:
        video_up = torch.nn.functional.interpolate(
            video_chunk,
            size=(int(geometry["process_output_h"]), int(geometry["process_output_w"])),
            mode=upscale_mode,
            align_corners=False,
        )

    video_up = (video_up / 255.0 * 2.0) - 1.0
    return video_up.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()


def prepare_reference_tensor(image_tensor: torch.Tensor, geometry: dict[str, int | float | None]) -> torch.Tensor:
    target_h = int(geometry["process_output_h"])
    target_w = int(geometry["process_output_w"])
    final_h = int(geometry["final_output_h"])
    final_w = int(geometry["final_output_w"])

    if image_tensor.shape[-2:] == (final_h, final_w) and (target_h != final_h or target_w != final_w):
        image_tensor = torch.nn.functional.pad(image_tensor, (0, target_w - final_w, 0, target_h - final_h), mode="replicate")
    elif image_tensor.shape[-2:] != (target_h, target_w):
        image_tensor = torch.nn.functional.interpolate(
            image_tensor.unsqueeze(0),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return image_tensor


def load_fallback_reference_frame(source_video: Path, frame_index: int, metadata: dict[str, int | Path | object], geometry: dict[str, int | float | None]) -> torch.Tensor:
    video_reader = metadata["video_reader"]
    raw_frame = decord_to_torch(video_reader[min(frame_index, int(metadata["total_frames"]) - 1)])
    raw_frame = raw_frame.unsqueeze(0)
    if int(metadata["pad_h"]) > 0 or int(metadata["pad_w"]) > 0:
        raw_frame = torch.nn.functional.pad(raw_frame, pad=(0, 0, 0, int(metadata["pad_w"]), 0, int(metadata["pad_h"])))
    raw_frame = raw_frame.float().permute(0, 3, 1, 2).contiguous()
    prepared = prepare_video_chunk_for_model(raw_frame, geometry, "bilinear")
    del source_video
    return prepared[0, :, 0]


def collect_reference_frames(
    ref_mode: str,
    source_video: Path,
    output_dir: Path,
    metadata: dict[str, int | Path | object],
    geometry: dict[str, int | float | None],
    explicit_indices: list[int],
):
    ref_frames_map: dict[int, torch.Tensor] = {}

    if ref_mode == "no_ref":
        return ref_frames_map, []

    total_frames_padded = int(metadata["padded_frames"])
    if explicit_indices:
        ref_indices = explicit_indices
    else:
        ref_indices = _select_indices(total_frames_padded)

    if ref_mode == "gt":
        saved = save_ref_frames_locally(
            video_path=str(source_video),
            output_dir=str(output_dir / "ref_gt_cache" / source_video.stem),
            video_id=source_video.stem,
            is_match=True,
            specific_indices=ref_indices,
        )
        for idx in ref_indices:
            found = False
            for saved_idx, saved_path in saved:
                if saved_idx != idx:
                    continue
                img = spark_module.Image.open(saved_path).convert("RGB")
                t_img = spark_module.transforms.ToTensor()(img)
                t_img = t_img * 2.0 - 1.0
                ref_frames_map[idx] = prepare_reference_tensor(t_img, geometry)
                found = True
                break
            if not found:
                ref_frames_map[idx] = load_fallback_reference_frame(source_video, idx, metadata, geometry)

    elif ref_mode == "api":
        api_cache_base = output_dir / "ref_api_cache"
        max_dim = max(int(geometry["process_output_h"]), int(geometry["process_output_w"]))
        if max_dim <= 1536:
            api_resolution = "1K"
        elif max_dim <= 3000:
            api_resolution = "2K"
        else:
            api_resolution = "4K"

        api_res = get_ref_frames_api(
            video_path=str(source_video),
            output_dir=str(api_cache_base / source_video.stem),
            video_id=source_video.stem,
            is_match=True,
            specific_indices=ref_indices,
            ref_prompt_mode="fixed",
            resolution=api_resolution,
        )
        for idx in ref_indices:
            found = False
            for saved_idx, saved_tensor in api_res:
                if saved_idx != idx:
                    continue
                final_h = int(geometry["final_output_h"])
                final_w = int(geometry["final_output_w"])
                saved_tensor = spark_module.center_crop_to_aspect_ratio(saved_tensor, final_h, final_w)
                ref_frames_map[idx] = prepare_reference_tensor(saved_tensor, geometry)
                found = True
                break
            if not found:
                ref_frames_map[idx] = load_fallback_reference_frame(source_video, idx, metadata, geometry)
    else:
        raise ValueError(f"Unsupported reference mode: {ref_mode}")

    return ref_frames_map, ref_indices


def write_chunk_frames(
    writer,
    video_chunk: torch.Tensor,
    valid_t_start: int,
    valid_t_end: int,
    geometry: dict[str, int | float | None],
    frames_remaining: int,
) -> int:
    if frames_remaining <= 0:
        return 0

    chunk = video_chunk[:, :, valid_t_start:valid_t_end, :, :]
    if int(geometry["output_pad_h"]) > 0:
        chunk = chunk[:, :, :, :-int(geometry["output_pad_h"]), :]
    if int(geometry["output_pad_w"]) > 0:
        chunk = chunk[:, :, :, :, :-int(geometry["output_pad_w"])]

    frames = chunk[0].permute(1, 2, 3, 0)
    if frames.shape[0] > frames_remaining:
        frames = frames[:frames_remaining]
    frames = (frames * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    for frame in frames:
        writer.append_data(frame)
    written = int(frames.shape[0])
    del chunk, frames
    return written


def run_single_video_job(
    pipe,
    empty_prompt_embedding,
    input_path: Path,
    output_dir: Path,
    fps: int,
    mode: str,
    upscale: int,
    output_resolution: tuple[int, int] | None,
    reference_guidance: float,
    explicit_indices: list[int],
):
    ref_mode = mode_to_ref_mode(mode)
    metadata = read_video_metadata(input_path)
    geometry = compute_output_geometry(metadata, upscale, output_resolution)
    ref_frames_map, ref_indices = collect_reference_frames(
        ref_mode=ref_mode,
        source_video=input_path,
        output_dir=output_dir,
        metadata=metadata,
        geometry=geometry,
        explicit_indices=explicit_indices,
    )

    chunk_len, overlap_t = get_chunk_settings(int(metadata["padded_frames"]))
    time_chunks = spark_module.make_temporal_chunks(int(metadata["padded_frames"]), chunk_len, overlap_t)
    out_file_path = output_dir / input_path.name.replace(".mkv", ".mp4")
    written_frames = 0
    total_real_frames = int(metadata["total_frames"])
    total_chunks = len(time_chunks)
    full_video_shape = (1, 3, int(metadata["padded_frames"]), int(geometry["process_output_h"]), int(geometry["process_output_w"]))

    chunk_frames = chunk_len or int(metadata["padded_frames"])
    summary_message = " ".join(
        [
            "Chunked processing:",
            f"real_frames={total_real_frames}",
            f"padded_frames={int(metadata['padded_frames'])}",
            f"chunk_frames={chunk_frames}",
            f"overlap_t={overlap_t}",
            f"chunks={total_chunks}",
            f"output={int(geometry['final_output_w'])}x{int(geometry['final_output_h'])}",
        ]
    )
    print(summary_message)
    yield {"type": "log", "message": summary_message}
    plan_message = f"Chunk plan: {total_chunks} chunks total, up to {chunk_frames} frames per chunk, {overlap_t} frame overlap."
    print(plan_message)
    yield {"type": "log", "message": plan_message}

    with imageio.get_writer(
        str(out_file_path),
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=None,
        ffmpeg_params=["-crf", "10"],
    ) as writer:
        for chunk_index, (t_start, t_end) in enumerate(time_chunks, start=1):
            current_ref_indices = [idx for idx in ref_indices if t_start <= idx < t_end]
            start_message = " ".join(
                [
                    f"Chunk {chunk_index}/{total_chunks} started.",
                    f"frames={t_start}:{t_end}",
                    f"size={t_end - t_start}",
                    f"refs={len(current_ref_indices)}",
                    f"written={format_progress(written_frames, total_real_frames)}",
                ]
            )
            print(start_message)
            yield {"type": "log", "message": start_message}

            chunk_started_at = time.monotonic()
            chunk_lr = read_video_chunk(
                metadata["video_reader"],
                t_start,
                t_end,
                int(metadata["total_frames"]),
                int(metadata["pad_h"]),
                int(metadata["pad_w"]),
            )
            video_chunk_model = prepare_video_chunk_for_model(chunk_lr, geometry, "bilinear")
            current_ref_frames = [ref_frames_map[idx] for idx in current_ref_indices]

            generated_chunk = spark_module.process_video_ref_i2v(
                pipe=pipe,
                video=video_chunk_model,
                prompt="",
                ref_frames=current_ref_frames,
                ref_indices=current_ref_indices,
                chunk_start_idx=t_start,
                noise_step=0,
                sr_noise_step=399,
                empty_prompt_embedding=empty_prompt_embedding,
                ref_guidance_scale=reference_guidance,
            )

            region = spark_module.get_valid_tile_region(
                t_start,
                t_end,
                0,
                int(geometry["process_output_h"]),
                0,
                int(geometry["process_output_w"]),
                full_video_shape,
                overlap_t,
                0,
                0,
            )
            remaining = total_real_frames - written_frames
            chunk_written = write_chunk_frames(
                writer=writer,
                video_chunk=generated_chunk,
                valid_t_start=region["valid_t_start"],
                valid_t_end=region["valid_t_end"],
                geometry=geometry,
                frames_remaining=remaining,
            )
            written_frames += chunk_written
            chunk_elapsed = format_duration(time.monotonic() - chunk_started_at)
            done_message = " ".join(
                [
                    f"Chunk {chunk_index}/{total_chunks} finished in {chunk_elapsed}.",
                    f"wrote={chunk_written}",
                    f"total={format_progress(written_frames, total_real_frames)}",
                ]
            )
            print(done_message)
            yield {"type": "log", "message": done_message}

            del chunk_lr, video_chunk_model, generated_chunk, current_ref_frames
            clear_cuda_memory()

    if written_frames != total_real_frames:
        raise RuntimeError(f"Frame count mismatch while writing output video: wrote {written_frames}, expected {total_real_frames}.")

    output_message = f"Output encoding complete: {format_progress(written_frames, total_real_frames)} written to `{out_file_path}`."
    print(output_message)
    yield {"type": "log", "message": output_message}
    yield {"type": "done", "output_path": out_file_path}


class TeeWriter:
    def __init__(self, handle):
        self.handle = handle

    def write(self, value):
        self.handle.write(value)
        self.handle.flush()
        return len(value)

    def flush(self):
        self.handle.flush()


def build_inference_command(
    input_path: Path,
    output_dir: Path,
    fps: int,
    mode: str,
    upscale: int,
    output_resolution: tuple[int, int] | None,
    reference_guidance: float,
    cpu_offload: bool,
    seed: int,
    api_indices: list[int],
    manual_indices: list[int],
) -> list[str]:
    command = [
        "python3",
        str(INFERENCE_SCRIPT.name),
        "--input_dir",
        str(input_path),
        "--model_path",
        str(MODEL_PATH),
        "--output_path",
        str(output_dir),
        "--fps",
        str(fps),
        "--dtype",
        "bfloat16",
        "--seed",
        str(seed),
        "--upscale",
        str(upscale),
        "--is_vae_st",
        "--save_format",
        "yuv420p",
        "--ref_mode",
        mode_to_ref_mode(mode),
        "--ref_prompt_mode",
        "fixed",
        "--ref_guidance_scale",
        str(reference_guidance),
    ]
    if cpu_offload:
        command.append("--is_cpu_offload")
    if output_resolution:
        command.extend(["--output_resolution", str(output_resolution[0]), str(output_resolution[1])])
    if mode == "API Reference" and api_indices:
        command.extend(["--ref_indices", *(str(index) for index in api_indices)])
    if mode == "Manual References":
        command.extend(["--ref_indices", *(str(index) for index in manual_indices)])
    return command


def run_inference(
    input_video,
    mode,
    upscale,
    output_resolution,
    reference_guidance,
    cpu_offload,
    explicit_indices,
    seed,
    *manual_row_values,
):
    ensure_runtime_dirs()
    log_text = ""
    current_log_path: Path | None = None
    job_started_at = time.monotonic()

    def emit(message: str, video=None, download_file=None, log_file=None):
        nonlocal log_text
        log_text = f"{log_text}{message.rstrip()}\n"
        return log_text, video, download_file, log_file, get_model_status_text()

    try:
        if not input_video:
            raise ValueError("Upload a video to start a SparkVSR job.")
        if not INFERENCE_SCRIPT.exists():
            raise FileNotFoundError("SparkVSR inference script is missing from the workspace.")
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"SparkVSR model directory is missing: {MODEL_PATH}")

        input_path = Path(input_video)
        if input_path.suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES:
            raise ValueError("Upload an `.mp4`, `.mov`, `.avi`, `.mkv`, or `.webm` file.")

        parsed_resolution = parse_optional_resolution(output_resolution)
        api_indices = parse_ref_indices(explicit_indices)
        manual_refs = collect_manual_references(list(manual_row_values))

        if mode == "API Reference" and not (os.environ.get("SPARKVSR_FAL_KEY") or os.environ.get("FAL_KEY")):
            raise ValueError("API Reference mode requires the `SPARKVSR_FAL_KEY` environment variable.")
        if mode == "Manual References" and not manual_refs:
            raise ValueError("Manual References mode requires at least one frame-index/image pair.")

        job_id = uuid.uuid4().hex[:12]
        safe_name = slugify_stem(input_path.name)
        job_dir = OUTPUT_ROOT / job_id
        output_dir = job_dir / "output"
        staging_dir = job_dir / "staging"
        job_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = job_dir / "job.log"
        current_log_path = log_path

        with log_path.open("w", encoding="utf-8") as log_file:
            original_input = job_dir / "input" / f"{safe_name}.mp4"
            original_input.parent.mkdir(parents=True, exist_ok=True)
            copy_or_convert_video(input_path, original_input)
            yield emit(f"Created job `{job_id}`.", None, None, str(log_path))
            yield emit(f"Prepared normalized input video at `{original_input}`.", None, None, str(log_path))

            pipe, empty_prompt_embedding, reused_pipeline = ensure_pipeline_loaded(bool(cpu_offload))
            if reused_pipeline:
                yield emit("Reusing loaded SparkVSR pipeline.", None, None, str(log_path))
            else:
                mode_text = "CPU offload" if cpu_offload else "GPU"
                yield emit(f"Loaded SparkVSR pipeline into memory ({mode_text} mode).", None, None, str(log_path))

            if mode == "Manual References":
                staged_input = staging_dir / "LQ-Video" / f"{safe_name}.mp4"
                staged_gt = staging_dir / "GT-Video" / f"{safe_name}.mp4"
                fps = build_synthetic_gt_video(
                    source_video=original_input,
                    destination_lq=staged_input,
                    destination_gt=staged_gt,
                    references=manual_refs,
                    upscale=int(upscale),
                    output_resolution=parsed_resolution,
                )
                command_input = staged_input
                manual_indices = [index for index, _ in manual_refs]
                yield emit("Prepared synthetic `GT-Video` staging for manual references.", None, None, str(log_path))
            else:
                fps = get_video_fps(original_input)
                command_input = original_input

            render_parts = build_inference_command(
                input_path=command_input,
                output_dir=output_dir,
                fps=fps,
                mode=mode,
                upscale=int(upscale),
                output_resolution=parsed_resolution,
                reference_guidance=float(reference_guidance),
                cpu_offload=bool(cpu_offload),
                seed=int(seed),
                api_indices=api_indices,
                manual_indices=[index for index, _ in manual_refs],
            )
            rendered_command = " ".join(shlex.quote(part) for part in render_parts)
            log_file.write(rendered_command + "\n")
            log_file.flush()
            yield emit(f"Launching SparkVSR:\n{rendered_command}", None, None, str(log_path))

            spark_module.set_seed(int(seed))
            tee = TeeWriter(log_file)
            with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                generated_video = None
                for event in run_single_video_job(
                    pipe=pipe,
                    empty_prompt_embedding=empty_prompt_embedding,
                    input_path=command_input,
                    output_dir=output_dir,
                    fps=fps,
                    mode=mode,
                    upscale=int(upscale),
                    output_resolution=parsed_resolution,
                    reference_guidance=float(reference_guidance),
                    explicit_indices=[index for index, _ in manual_refs] if mode == "Manual References" else api_indices,
                ):
                    if event["type"] == "log":
                        yield emit(event["message"], None, None, str(log_path))
                    elif event["type"] == "done":
                        generated_video = event["output_path"]
                if generated_video is None:
                    raise RuntimeError("SparkVSR finished without returning an output path.")
            log_file.flush()
            log_text = log_path.read_text(encoding="utf-8")

        elapsed = format_duration(time.monotonic() - job_started_at)
        yield emit(
            f"Finished successfully in {elapsed}. Output video: `{generated_video}`",
            str(generated_video),
            str(generated_video),
            str(log_path),
        )
    except Exception as exc:
        elapsed = format_duration(time.monotonic() - job_started_at)
        if current_log_path and current_log_path.exists():
            log_text = current_log_path.read_text(encoding="utf-8")
        yield emit(f"Error after {elapsed}: {exc}", None, None, str(current_log_path) if current_log_path else None)


def update_mode_visibility(mode: str):
    manual_visible = mode == "Manual References"
    api_visible = mode == "API Reference"
    return (
        gr.update(visible=manual_visible),
        gr.update(visible=api_visible),
    )


def unload_model():
    return get_model_status_text() if PIPELINE_STATE["pipe"] is None else unload_loaded_pipeline()


ensure_runtime_dirs()

with gr.Blocks(title="SparkVSR RunPod", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # SparkVSR on RunPod
        Upload a video, choose an inference mode, and run SparkVSR from this pod.
        `API Reference` requires `SPARKVSR_FAL_KEY`. `Manual References` accepts up to 8 frame-index/image pairs.
        """
    )
    model_status = gr.Markdown(get_model_status_text())

    with gr.Row():
        with gr.Column(scale=2):
            input_video = gr.File(label="Input Video", file_types=["video"])
            mode = gr.Radio(
                label="Mode",
                choices=["No Reference", "API Reference", "Manual References"],
                value="No Reference",
            )
            upscale = gr.Slider(label="Upscale Factor", minimum=1, maximum=4, step=1, value=4)
            output_resolution = gr.Textbox(label="Optional Output Resolution", placeholder="720x1280")
            reference_guidance = gr.Slider(label="Reference Guidance Scale", minimum=0.0, maximum=5.0, step=0.1, value=1.0)
            cpu_offload = gr.Checkbox(
                label="Low VRAM Mode (CPU Offload)",
                value=False,
                info="Slower, but can help larger jobs fit on smaller GPUs or avoid CUDA OOM errors.",
            )
            explicit_indices = gr.Textbox(
                label="Explicit Reference Indices",
                placeholder="0,16,32",
                info="Optional for API Reference mode. Leave blank to use upstream auto-selection.",
            )
            seed = gr.Number(label="Seed", value=42, precision=0)

        with gr.Column(scale=3):
            with gr.Group(visible=False) as api_hint_group:
                gr.Markdown("`API Reference` uses SparkVSR's upstream API-assisted reference path and requires `SPARKVSR_FAL_KEY` in the container environment.")

            with gr.Group(visible=False) as manual_group:
                gr.Markdown("Enable one row per manual reference, set the frame index, and upload the replacement image for that frame.")
                manual_components: list[gr.Component] = []
                for row in range(MANUAL_REF_ROWS):
                    with gr.Row():
                        enabled = gr.Checkbox(label=f"Use Row {row + 1}", value=False, min_width=120)
                        frame_index = gr.Number(label="Frame Index", precision=0, min_width=140)
                        image = gr.File(label="Reference Image", file_types=["image"], min_width=220)
                    manual_components.extend([enabled, frame_index, image])

    with gr.Row():
        run_button = gr.Button("Run SparkVSR", variant="primary")
        unload_button = gr.Button("Unload Model")

    with gr.Row():
        logs = gr.Textbox(label="Progress / Logs", lines=20)
    with gr.Row():
        output_video = gr.Video(label="Output Video")
        output_download = gr.DownloadButton(label="Download Output")
        log_file = gr.File(label="Job Log")

    mode.change(
        update_mode_visibility,
        inputs=[mode],
        outputs=[manual_group, api_hint_group],
    )
    unload_button.click(
        unload_model,
        outputs=[model_status],
    )
    run_button.click(
        run_inference,
        inputs=[input_video, mode, upscale, output_resolution, reference_guidance, cpu_offload, explicit_indices, seed, *manual_components],
        outputs=[logs, output_video, output_download, log_file, model_status],
    )

demo.queue(default_concurrency_limit=1)

if __name__ == "__main__":
    demo.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"), server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")))
