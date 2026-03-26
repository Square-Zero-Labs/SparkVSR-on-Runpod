# SparkVSR RunPod Template (Unofficial)

### _This template has been tested with an A40 and an RTX5090_

This template launches a Gradio UI for [SparkVSR](https://github.com/taco-group/SparkVSR) on Runpod. It is set up for video super-resolution with three workflows:

- `No Reference`: standard blind SparkVSR upscaling
- `API Reference`: calls FAL AI to generate keyframes
- `Manual References`: lets you upload your own frame-index/image pairs

## Quick Start

1. Deploy the template on RunPod.
2. Open the app on port `7862`.
3. Log in with the default credentials:
   - Username: `admin`
   - Password: `sparkvsr`
4. Upload a video, choose a mode, and click `Run SparkVSR`.

## Template Variables

| Variable            | Description                        | Default    |
| ------------------- | ---------------------------------- | ---------- |
| `SPARKVSR_USERNAME` | nginx / Gradio basic-auth username | `admin`    |
| `SPARKVSR_PASSWORD` | nginx / Gradio basic-auth password | `sparkvsr` |

## Chunking And Memory

The default chunking settings are:

- `Chunk Frames = 33`
- `Chunk Overlap = 8`

Practical guidance:

- If you hit memory limits, try `25 / 8` or `17 / 8`
- If memory is comfortable, try `41 / 8`
- Input resolution is one of the biggest memory drivers
- Output resolution and `x4` upscaling also increase memory use sharply

## Outputs And Logs

- Main app log: `/workspace/SparkVSR/logs/sparkvsr.log`
- Per-job logs: `/workspace/SparkVSR/out/<job-id>/job.log`
- Outputs are stored under `/workspace/SparkVSR/out`

Tail logs inside the pod:

```bash
tail -f /workspace/SparkVSR/logs/sparkvsr.log
```

Restart the app inside the pod:

```bash
restart-sparkvsr
```

## Notes

- PiSA-SR for image super resolution is not exposed in this template
- The template is optimized for SparkVSR Stage-2 inference, not training
- Recommended GPU class: RTX 5090 or A40

## Resources

- The [Dockerfile and code](https://github.com/Square-Zero-Labs/SparkVSR-on-Runpod) are public. If you encounter any problems, please open an issue in the repo.
- SparkVSR upstream repo: https://github.com/taco-group/SparkVSR
