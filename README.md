# SparkVSR RunPod Template

This repository packages [SparkVSR](https://github.com/taco-group/SparkVSR) for RunPod with a Gradio UI, nginx basic-auth, startup model caching, and GitHub Actions publishing to GHCR.

The upstream SparkVSR source is included as a git subtree in [`SparkVSR-base`](./SparkVSR-base). That subtree remains subject to the Apache 2.0 license defined upstream.

## What This Template Provides

- RunPod-ready Docker image based on `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- RTX 5090-friendly CUDA 12.8+ runtime
- Gradio app for:
  - blind upscaling (`no_ref`)
  - API-assisted keyframes (`api`)
  - manual reference keyframes (`gt` via synthetic `GT-Video` staging)
- nginx basic auth in front of Gradio
- startup download for the official SparkVSR Stage-2 checkpoint
- GitHub Actions workflow that builds and pushes to GHCR on `main`, `docker`, and `v*` tags

## Repository Layout

- `SparkVSR-base/`: upstream SparkVSR subtree
- `app.py`: Gradio application and inference wrapper
- `start-sparkvsr.sh`: RunPod startup and model bootstrap
- `restart-sparkvsr.sh`: helper to restart the app inside a running pod
- `runtime-requirements.txt`: curated inference-only Python dependencies
- `.github/workflows/docker-build.yml`: GHCR build/push workflow

## Build

```bash
docker build --platform=linux/amd64 -t sparkvsr-runpod:latest .
```

## Run Locally

```bash
docker run --gpus all -p 7862:7862 \
  -e SPARKVSR_USERNAME=admin \
  -e SPARKVSR_PASSWORD=sparkvsr \
  -e HF_TOKEN=your_hf_token \
  sparkvsr-runpod:latest
```

The container proxies nginx on port `7862` to Gradio on port `7860`.

## Deploy On RunPod

1. Expose port `7862`.
2. Set environment variables as needed:
   - `SPARKVSR_USERNAME` default `admin`
   - `SPARKVSR_PASSWORD` default `sparkvsr`
   - `HF_TOKEN` optional, but recommended for Hugging Face downloads
3. Launch on an RTX 5090 or A40 pod.

On first boot, the startup script restores the app into `/workspace/SparkVSR`, downloads the required models, configures nginx auth, and launches Gradio. Subsequent restarts reuse the cached model directories in `/workspace/SparkVSR`.

The template only downloads `JiongzeYu/SparkVSR`. That Hugging Face repo already contains the full diffusers pipeline needed by `CogVideoXImageToVideoPipeline`, so the separate `zai-org/CogVideoX1.5-5B-I2V` base-model download is intentionally not used here.

## UI Behavior

- `No Reference`: standard SparkVSR blind upscaling
- `API Reference`: uses SparkVSR's upstream API-assisted reference path and exposes a masked `FAL API Key` field plus an editable `API Prompt` field directly in the Gradio UI
- `Manual References`: lets the user provide frame-index/image pairs; the wrapper creates a synthetic `GT-Video` clip so SparkVSR can reuse its upstream `ref_mode=gt` flow

PiSA-SR is intentionally not exposed in this template.

## Chunk Tuning

The wrapper uses temporal chunking to keep RAM usage under control on long videos. Chunk controls now live in the Gradio UI on a per-job basis. The default UI values are:

- `Chunk Frames = 49`
- `Chunk Overlap = 8`

Practical tuning path:

- start with `49 / 8`
- if you need lower memory usage, try `41 / 8`, then `33 / 8`, then `25 / 8` or `17 / 8`
- if memory is still comfortable and you want fewer chunks, try `57 / 8`
- set `Chunk Frames = 0` and `Chunk Overlap = 0` to disable chunking entirely

Practical tradeoff:

- smaller chunk length: lower RAM and VRAM use, more overlap overhead, slower jobs
- larger chunk length: fewer chunks, less repeated work, faster jobs, higher memory use

Memory note:

- input resolution is a major memory driver; higher source width and height increase decoded frame tensor size and intermediate activations
- output resolution and upscale factor also matter heavily, especially on `x4` jobs
- video duration mostly affects runtime when chunking is enabled, not peak memory

## Logs And Restart

Inside the pod:

```bash
tail -f /workspace/SparkVSR/logs/sparkvsr.log
restart-sparkvsr
```

Per-job logs are stored under `/workspace/SparkVSR/out/<job-id>/job.log`.
