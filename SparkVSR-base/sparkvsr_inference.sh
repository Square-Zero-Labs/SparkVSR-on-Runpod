# Model Path (Update as needed)
# Stage 1
# MODEL_PATH="checkpoints/sparkvsr-s1/ckpt-10000-sft" 
# Stage 2
MODEL_PATH="checkpoints/sparkvsr-s2/ckpt-500-sft" 


# UDM10 (No-Ref)
CUDA_VISIBLE_DEVICES=0 python sparkvsr_inference_script.py \
    --input_dir datasets/test/UDM10/LQ-Video \
    --model_path $MODEL_PATH \
    --output_path results/UDM10/no_ref \
    --gt_dir datasets/test/UDM10/GT-Video \
    --is_vae_st \
    --ref_mode no_ref \
    --ref_prompt_mode fixed \
    --ref_guidance_scale 1.0 \
    --eval_metrics psnr,ssim,lpips,dists,clipiqa \
    --upscale 4

# UDM10 (API Mode)
CUDA_VISIBLE_DEVICES=0 python sparkvsr_inference_script.py \
    --input_dir datasets/test/UDM10/LQ-Video \
    --model_path $MODEL_PATH \
    --output_path results/UDM10/api_ref_num1_indices0_rgs1.0 \
    --gt_dir datasets/test/UDM10/GT-Video \
    --is_vae_st \
    --ref_mode api \
    --ref_prompt_mode fixed \
    --ref_guidance_scale 1.0 \
    --eval_metrics psnr,ssim,lpips,dists,clipiqa \
    --upscale 4 \
    --ref_indices 0


# UDM10 (PiSA Mode)
CUDA_VISIBLE_DEVICES=0 python sparkvsr_inference_script.py \
    --input_dir datasets/test/UDM10/LQ-Video \
    --model_path $MODEL_PATH \
    --output_path results/UDM10/pisa_ref_num1_indices0_rgs1.0 \
    --gt_dir datasets/test/UDM10/GT-Video \
    --is_vae_st \
    --ref_mode pisasr \
    --ref_guidance_scale 1.0 \
    --eval_metrics psnr,ssim,lpips,dists,clipiqa \
    --upscale 4 \
    --ref_indices 0 \
    --pisa_python_executable "path/to/your/pisasr/conda/env/bin/python" \
    --pisa_script_path "path/to/your/PiSA-SR/test_pisasr.py" \
    --pisa_sd_model_path "path/to/your/PiSA-SR/preset/models/stable-diffusion-2-1-base" \
    --pisa_chkpt_path "path/to/your/PiSA-SR/preset/models/pisa_sr.pkl" \
    --pisa_gpu "1"