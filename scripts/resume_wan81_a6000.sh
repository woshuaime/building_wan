#!/bin/bash

set -euo pipefail

ROOT="${BUILDING_WAN_ROOT:-/mnt/windowsE/chuanjun/building_wan}"
PYTHON="${WAN_PYTHON:-/home/shi/miniconda3/envs/scj/bin/python}"
DIFFSYNTH_ROOT="$ROOT/code/DiffSynth-Studio"
TRAIN_ENTRY="$ROOT/code/Python_3D_Scanner/training/wan_train_entry.py"
CACHE_ROOT="$ROOT/cache/wan21_t2v_13b_single_orbit_same_prompt_384x384_f81_train"
OUTPUT_ROOT="${WAN_RESUME_OUTPUT_ROOT:-$ROOT/checkpoints/building_wan_a6000_single_orbit_resumed}"
CHECKPOINT="${WAN_RESUME_FROM_CHECKPOINT:-$ROOT/checkpoints/building_wan_a6000_single_orbit/step-2400.safetensors}"
SKIP_BATCHES="${WAN_RESUME_SKIP_BATCHES:-2400}"
INITIAL_STEPS="${WAN_RESUME_INITIAL_STEPS:-2400}"
TRAIN_BATCH_SIZE="${WAN_TRAIN_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${WAN_GRADIENT_ACCUMULATION_STEPS:-2}"
DATASET_NUM_WORKERS="${WAN_DATASET_NUM_WORKERS:-0}"

DIT="$ROOT/models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"
T5="$ROOT/models/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors"
VAE="$ROOT/models/Wan-Series-Converted-Safetensors/Wan2.1_VAE.safetensors"
TOKENIZER="$ROOT/models/Wan2.1-T2V-1.3B/google/umt5-xxl"
MODEL_PATHS="[\"$DIT\", \"$T5\", \"$VAE\"]"
SHARED_TEXT_CONTEXT="$CACHE_ROOT/.shared-text-context-bf16.pt"

required_paths=("$PYTHON" "$DIFFSYNTH_ROOT" "$TRAIN_ENTRY" "$CACHE_ROOT" "$CHECKPOINT" "$SHARED_TEXT_CONTEXT" "$DIT" "$T5" "$VAE" "$TOKENIZER/tokenizer.json")
for path in "${required_paths[@]}"; do
    [[ -e "$path" ]] || { echo "Required path is missing: $path" >&2; exit 2; }
done

if ps -eo args= | grep -F "$TRAIN_ENTRY" | grep -v grep >/dev/null; then
    echo "A Wan training process is already running; refusing to start a second one." >&2
    exit 7
fi

if [[ -e "$OUTPUT_ROOT" ]] && find "$OUTPUT_ROOT" -maxdepth 1 -type f -name '*.safetensors' | grep -q .; then
    echo "Resume output already contains checkpoints: $OUTPUT_ROOT" >&2
    exit 6
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONPATH="$DIFFSYNTH_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT_ROOT"
cd "$ROOT/code/Python_3D_Scanner"

exec "$PYTHON" "$TRAIN_ENTRY" \
    --diffsynth-root "$DIFFSYNTH_ROOT" \
    --num-processes 1 \
    --backend nccl \
    --train-batch-size "$TRAIN_BATCH_SIZE" \
    --shared-text-context-path "$SHARED_TEXT_CONTEXT" \
    --dataset_base_path "$CACHE_ROOT" \
    --dataset_repeat 1 \
    --dataset_num_workers "$DATASET_NUM_WORKERS" \
    --model_paths "$MODEL_PATHS" \
    --tokenizer_path "$TOKENIZER" \
    --offload_models "$T5,$VAE" \
    --output_path "$OUTPUT_ROOT" \
    --task sft:train \
    --height 384 \
    --width 384 \
    --num_frames 81 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --num_epochs 8 \
    --save_steps 400 \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --resume_from_checkpoint "$CHECKPOINT" \
    --skip_train_batches "$SKIP_BATCHES" \
    --initial_train_steps "$INITIAL_STEPS" \
    --lora_base_model dit \
    --lora_target_modules q,k,v,o,ffn.0,ffn.2 \
    --lora_rank 16 \
    --remove_prefix_in_ckpt pipe.dit. \
    --use_gradient_checkpointing \
    --enable_csv_log
