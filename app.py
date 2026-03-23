from __future__ import annotations

import contextlib
import gc
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


def prepare_video_tensor(video, upscale: int, output_resolution: tuple[int, int] | None, upscale_mode: str):
    h_orig, w_orig = video.shape[2], video.shape[3]

    if output_resolution is not None:
        target_h, target_w = output_resolution
        scale_h = target_h / h_orig
        scale_w = target_w / w_orig
        scale_factor = max(scale_h, scale_w)
        scaled_h = int(h_orig * scale_factor)
        scaled_w = int(w_orig * scale_factor)
        video_up = torch.nn.functional.interpolate(
            video,
            size=(scaled_h, scaled_w),
            mode=upscale_mode,
            align_corners=False,
        )
        crop_top = (scaled_h - target_h) // 2
        crop_left = (scaled_w - target_w) // 2
        video_up = video_up[:, :, crop_top:crop_top + target_h, crop_left:crop_left + target_w]
        pad_h_extra = (8 - target_h % 8) % 8
        pad_w_extra = (8 - target_w % 8) % 8
        if pad_h_extra > 0 or pad_w_extra > 0:
            video_up = torch.nn.functional.pad(video_up, (0, pad_w_extra, 0, pad_h_extra))
        effective_upscale = 1
    else:
        video_up = torch.nn.functional.interpolate(
            video,
            size=(h_orig * upscale, w_orig * upscale),
            mode=upscale_mode,
            align_corners=False,
        )
        effective_upscale = upscale

    video_up = (video_up / 255.0 * 2.0) - 1.0
    video_model = video_up.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()
    return video_model, effective_upscale


def collect_reference_frames(
    ref_mode: str,
    video_path: Path,
    output_dir: Path,
    video_tensor_model: torch.Tensor,
    video_lr: torch.Tensor,
    effective_upscale: int,
    explicit_indices: list[int],
    reference_guidance: float,
):
    del reference_guidance  # Guidance is applied later in the model call.
    video_name = video_path.name
    ref_frames_list = []

    if ref_mode == "no_ref":
        return ref_frames_list, []

    if explicit_indices:
        ref_indices = explicit_indices
    else:
        ref_indices = _select_indices(video_tensor_model.shape[2])

    if ref_mode == "gt":
        saved = save_ref_frames_locally(
            video_path=str(video_path),
            output_dir=str(output_dir / "ref_gt_cache" / video_path.stem),
            video_id=video_path.stem,
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
                target_h, target_w = video_tensor_model.shape[-2], video_tensor_model.shape[-1]
                if t_img.shape[-2:] != (target_h, target_w):
                    gt_h, gt_w = t_img.shape[-2], t_img.shape[-1]
                    orig_h_up = video_lr.shape[1] * effective_upscale
                    orig_w_up = video_lr.shape[2] * effective_upscale
                    if gt_h == orig_h_up and gt_w == orig_w_up:
                        gt_pad_h = target_h - gt_h
                        gt_pad_w = target_w - gt_w
                        if gt_pad_h > 0 or gt_pad_w > 0:
                            t_img = torch.nn.functional.pad(t_img, (0, gt_pad_w, 0, gt_pad_h), mode="replicate")
                    else:
                        t_img = torch.nn.functional.interpolate(
                            t_img.unsqueeze(0),
                            size=(target_h, target_w),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0)
                ref_frames_list.append(t_img)
                found = True
                break
            if not found:
                ref_frames_list.append(video_tensor_model[0, :, idx])

    elif ref_mode == "api":
        vid_01 = ((video_tensor_model[0] + 1.0) / 2.0).permute(1, 0, 2, 3)
        api_cache_base = output_dir / "ref_api_cache"
        target_h, target_w = video_tensor_model.shape[-2], video_tensor_model.shape[-1]
        max_dim = max(target_h, target_w)
        if max_dim <= 1536:
            api_resolution = "1K"
        elif max_dim <= 3000:
            api_resolution = "2K"
        else:
            api_resolution = "4K"

        api_res = get_ref_frames_api(
            output_dir=str(api_cache_base / video_path.stem),
            video_tensor=vid_01,
            video_id=video_path.stem,
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
                target_h, target_w = video_tensor_model.shape[-2], video_tensor_model.shape[-1]
                saved_tensor = spark_module.center_crop_to_aspect_ratio(saved_tensor, target_h, target_w)
                if saved_tensor.shape[-2:] != (target_h, target_w):
                    saved_tensor = torch.nn.functional.interpolate(
                        saved_tensor.unsqueeze(0),
                        size=(target_h, target_w),
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze(0)
                ref_frames_list.append(saved_tensor)
                found = True
                break
            if not found:
                ref_frames_list.append(video_tensor_model[0, :, idx])
    else:
        raise ValueError(f"Unsupported reference mode: {ref_mode}")

    return ref_frames_list, ref_indices


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
    video, pad_f, pad_h, pad_w, _original_shape = spark_module.preprocess_video_match(str(input_path), is_match=True)
    video_lr = video
    video_tensor_model, effective_upscale = prepare_video_tensor(video, upscale, output_resolution, "bilinear")

    ref_frames_list, ref_indices = collect_reference_frames(
        ref_mode=ref_mode,
        video_path=input_path,
        output_dir=output_dir,
        video_tensor_model=video_tensor_model,
        video_lr=video_lr,
        effective_upscale=effective_upscale,
        explicit_indices=explicit_indices,
        reference_guidance=reference_guidance,
    )

    _, _, num_frames, height, width = video_tensor_model.shape
    overlap_t = 0
    overlap_hw = (0, 0)
    time_chunks = spark_module.make_temporal_chunks(num_frames, 0, overlap_t)
    spatial_tiles = spark_module.make_spatial_tiles(height, width, (0, 0), overlap_hw)

    output_video = torch.zeros_like(video_tensor_model)
    for t_start, t_end in time_chunks:
        for h_start, h_end, w_start, w_end in spatial_tiles:
            video_chunk = video_tensor_model[:, :, t_start:t_end, h_start:h_end, w_start:w_end]
            current_ref_frames = [rf[:, h_start:h_end, w_start:w_end] for rf in ref_frames_list]
            generated_chunk = spark_module.process_video_ref_i2v(
                pipe=pipe,
                video=video_chunk,
                prompt="",
                ref_frames=current_ref_frames,
                ref_indices=ref_indices,
                chunk_start_idx=t_start,
                noise_step=0,
                sr_noise_step=399,
                empty_prompt_embedding=empty_prompt_embedding,
                ref_guidance_scale=reference_guidance,
            )
            region = spark_module.get_valid_tile_region(
                t_start,
                t_end,
                h_start,
                h_end,
                w_start,
                w_end,
                video_tensor_model.shape,
                overlap_t,
                overlap_hw[0],
                overlap_hw[1],
            )
            output_video[
                :,
                :,
                region["out_t_start"]:region["out_t_end"],
                region["out_h_start"]:region["out_h_end"],
                region["out_w_start"]:region["out_w_end"],
            ] = generated_chunk[
                :,
                :,
                region["valid_t_start"]:region["valid_t_end"],
                region["valid_h_start"]:region["valid_h_end"],
                region["valid_w_start"]:region["valid_w_end"],
            ]

    video_generate = spark_module.remove_padding_and_extra_frames(output_video, pad_f, pad_h * effective_upscale, pad_w * effective_upscale)
    out_file_path = output_dir / input_path.name.replace(".mkv", ".mp4")
    spark_module.save_video_with_imageio(video_generate, str(out_file_path), fps=fps, format="yuv420p")
    clear_cuda_memory()
    return out_file_path


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

    def emit(message: str, video=None, log_file=None):
        nonlocal log_text
        log_text = f"{log_text}{message.rstrip()}\n"
        return log_text, video, log_file, get_model_status_text()

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
            yield emit(f"Created job `{job_id}`.", None, str(log_path))
            yield emit(f"Prepared normalized input video at `{original_input}`.", None, str(log_path))

            pipe, empty_prompt_embedding, reused_pipeline = ensure_pipeline_loaded(bool(cpu_offload))
            if reused_pipeline:
                yield emit("Reusing loaded SparkVSR pipeline.", None, str(log_path))
            else:
                mode_text = "CPU offload" if cpu_offload else "GPU"
                yield emit(f"Loaded SparkVSR pipeline into memory ({mode_text} mode).", None, str(log_path))

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
                yield emit("Prepared synthetic `GT-Video` staging for manual references.", None, str(log_path))
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
            yield emit(f"Launching SparkVSR:\n{rendered_command}", None, str(log_path))

            spark_module.set_seed(int(seed))
            tee = TeeWriter(log_file)
            with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                generated_video = run_single_video_job(
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
                )
            log_file.flush()
            log_text = log_path.read_text(encoding="utf-8")

        elapsed = format_duration(time.monotonic() - job_started_at)
        yield emit(
            f"Finished successfully in {elapsed}. Output video: `{generated_video}`",
            str(generated_video),
            str(log_path),
        )
    except Exception as exc:
        elapsed = format_duration(time.monotonic() - job_started_at)
        if current_log_path and current_log_path.exists():
            log_text = current_log_path.read_text(encoding="utf-8")
        yield emit(f"Error after {elapsed}: {exc}", None, str(current_log_path) if current_log_path else None)


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
        outputs=[logs, output_video, log_file, model_status],
    )

demo.queue(default_concurrency_limit=1)

if __name__ == "__main__":
    demo.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"), server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")))
