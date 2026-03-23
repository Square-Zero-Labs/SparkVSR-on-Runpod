#!/bin/bash

# Default arguments
PRED_DIR="results/UDM10/api_ref_num1_indices0_rgs1.0"
GT_DIR="datasets/test/UDM10/GT-Video"
OUT_DIR="results/UDM10/api_ref_num1_indices0_rgs1.0"
DEVICE="cuda"
METRICS="psnr,ssim,lpips,dists,clipiqa,niqe,musiq,dover,ewarp,vbench,fastvqa"
# METRICS="clipiqa,niqe,musiq,dover,ewarp,vbench,fastvqa"
GPU_ID="6"
FILENAME="all_metrics_results.json"

# Function to display usage
usage() {
    echo "Usage: $0 --pred <pred_dir> [--gt <gt_dir>] [--out <out_dir>] [--device <device>] [--metrics <metrics_list>] [--gpu_id <id>] [--filename <name>]"
    echo "  --pred: Path to the directory containing predicted videos/images (Required)"
    echo "  --gt: Path to the directory containing Ground Truth videos/images (Optional, required for FR metrics like PSNR, SSIM)"
    echo "  --out: Output directory for results (Default: $OUT_DIR)"
    echo "  --device: Device to use (Default: $DEVICE)"
    echo "  --metrics: Comma-separated list of metrics (Default: $METRICS)"
    echo "  --gpu_id: GPU ID to use (Default: $GPU_ID)"
    echo "  --filename: Output JSON filename (Default: $FILENAME)"
    exit 1
}

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --pred) PRED_DIR="$2"; shift ;;
        --gt) GT_DIR="$2"; shift ;;
        --out) OUT_DIR="$2"; shift ;;
        --device) DEVICE="$2"; shift ;;
        --metrics) METRICS="$2"; shift ;;
        --gpu_id|-g) GPU_ID="$2"; shift ;;
        --filename) FILENAME="$2"; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# Check if PRED_DIR is set
if [ -z "$PRED_DIR" ]; then
    echo "Error: --pred argument is required."
    usage
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Optional: Avoid fragmentation if OOM occurs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting evaluation..."
echo "Prediction Directory: $PRED_DIR"
if [ ! -z "$GT_DIR" ]; then
    echo "Ground Truth Directory: $GT_DIR"
else
    echo "Ground Truth Directory: Not provided (Full-Reference metrics will be skipped)"
fi
echo "Output Directory: $OUT_DIR"
echo "Output Filename: $FILENAME"
echo "Metrics: $METRICS"
echo "Device: $DEVICE (GPU ID: $GPU_ID)"

# Ensure output directory exists
mkdir -p "$OUT_DIR"

# Run the python script
python finetune/scripts/eval_all_metrics.py \
    --pred "$PRED_DIR" \
    --gt "$GT_DIR" \
    --out "$OUT_DIR" \
    --metrics "$METRICS" \
    --device "$DEVICE" \
    --filename "$FILENAME"

echo "Evaluation finished."
