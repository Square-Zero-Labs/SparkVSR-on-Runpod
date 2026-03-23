from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Iterable

import cv2
import gradio as gr


APP_ROOT = Path(os.environ.get("SPARKVSR_WORKSPACE_DIR", Path(__file__).resolve().parent)).resolve()
SUBTREE_ROOT = APP_ROOT / "SparkVSR-base"
INFERENCE_SCRIPT = SUBTREE_ROOT / "sparkvsr_inference_script.py"
MODEL_PATH = Path(os.environ.get("SPARKVSR_MODEL_PATH", APP_ROOT / "checkpoints" / "sparkvsr-s2" / "ckpt-500-sft")).resolve()
LOG_DIR = APP_ROOT / "logs"
OUTPUT_ROOT = APP_ROOT / "out"
INPUT_ROOT = APP_ROOT / "in"
MANUAL_REF_ROWS = 8
SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


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

    def emit(message: str, video=None, log_file=None):
        nonlocal log_text
        log_text = f"{log_text}{message.rstrip()}\n"
        return log_text, video, log_file

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
                manual_indices = []

            command = build_inference_command(
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
                manual_indices=manual_indices,
            )

            rendered_command = " ".join(shlex.quote(part) for part in command)
            log_file.write(rendered_command + "\n")
            log_file.flush()
            yield emit(f"Launching SparkVSR:\n{rendered_command}", None, str(log_path))

            process = subprocess.Popen(
                command,
                cwd=SUBTREE_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=os.environ.copy(),
            )

            assert process.stdout is not None
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
                yield emit(line, None, str(log_path))

            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(f"SparkVSR exited with status {return_code}.")

        generated_video = find_generated_video(output_dir)
        yield emit(f"Finished successfully. Output video: `{generated_video}`", str(generated_video), str(log_path))
    except Exception as exc:
        yield emit(f"Error: {exc}", None, str(current_log_path) if current_log_path else None)


def update_mode_visibility(mode: str):
    manual_visible = mode == "Manual References"
    api_visible = mode == "API Reference"
    return (
        gr.update(visible=manual_visible),
        gr.update(visible=api_visible),
    )


ensure_runtime_dirs()

with gr.Blocks(title="SparkVSR RunPod", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # SparkVSR on RunPod
        Upload a video, choose an inference mode, and run SparkVSR from this pod.
        `API Reference` requires `SPARKVSR_FAL_KEY`. `Manual References` accepts up to 8 frame-index/image pairs.
        """
    )

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

    run_button = gr.Button("Run SparkVSR", variant="primary")

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
    run_button.click(
        run_inference,
        inputs=[input_video, mode, upscale, output_resolution, reference_guidance, cpu_offload, explicit_indices, seed, *manual_components],
        outputs=[logs, output_video, log_file],
    )

demo.queue(default_concurrency_limit=1)

if __name__ == "__main__":
    demo.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"), server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")))
