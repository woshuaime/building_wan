#!/bin/bash

set -euo pipefail

ROOT=/home/share/CHUANJUN/building_wan
TRAIN_NAME=building_wan
PYTHON=/home/u2025171963/miniconda3/envs/mv2v/bin/python

DIFFSYNTH_ROOT="$ROOT/code/DiffSynth-Studio"
TRAIN_ENTRY="$ROOT/code/Python_3D_Scanner/training/wan_train_entry.py"

DATASET_ROOT="$ROOT/data/video_wan81"
METADATA="$ROOT/data/splits/building_v1/metadata_train.csv"
CACHE_ROOT="$ROOT/cache/wan21_t2v_13b_building_384x384_f81_train"
CACHE_FORMAT=compact-sft-shared-context-bf16-v2
CACHE_MARKER="$CACHE_ROOT/.$CACHE_FORMAT"
SHARED_TEXT_CONTEXT="$CACHE_ROOT/.shared-text-context-bf16.pt"
OUTPUT_ROOT="$ROOT/checkpoints/$TRAIN_NAME"

DIT="$ROOT/models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"
T5="$ROOT/models/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors"
VAE="$ROOT/models/Wan-Series-Converted-Safetensors/Wan2.1_VAE.safetensors"
TOKENIZER="$ROOT/models/Wan2.1-T2V-1.3B/google/umt5-xxl"
MODEL_PATHS="[\"$DIT\", \"$T5\", \"$VAE\"]"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONPATH="$DIFFSYNTH_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date --iso-8601=seconds)] Launching $TRAIN_NAME"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-manual}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

required_paths=(
    "$PYTHON"
    "$DIFFSYNTH_ROOT"
    "$TRAIN_ENTRY"
    "$DATASET_ROOT"
    "$METADATA"
    "$DIT"
    "$T5"
    "$VAE"
    "$TOKENIZER/tokenizer.json"
)
for path in "${required_paths[@]}"; do
    if [[ ! -e "$path" ]]; then
        echo "Required path is missing: $path" >&2
        exit 2
    fi
done

"$PYTHON" -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; assert torch.cuda.device_count() == 1, f'Expected exactly one visible GPU, found {torch.cuda.device_count()}'; print('CUDA GPU:', torch.cuda.get_device_name(0)); print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))"

expected_samples=$(($(wc -l < "$METADATA") - 1))
if [[ "$expected_samples" -ne 1912 ]]; then
    echo "Expected 1912 training samples, found $expected_samples in $METADATA" >&2
    exit 3
fi

if [[ -d "$CACHE_ROOT" ]]; then
    cached_samples=$(find "$CACHE_ROOT" -type f -name '*.pth' | wc -l)
else
    cached_samples=0
fi

if [[ "$cached_samples" -eq "$expected_samples" && -f "$CACHE_MARKER" && -f "$SHARED_TEXT_CONTEXT" ]]; then
    echo "[$(date --iso-8601=seconds)] Reusing complete $CACHE_FORMAT cache: $cached_samples files."
else
    if [[ "$cached_samples" -eq 0 ]]; then
        echo "[$(date --iso-8601=seconds)] Cache is empty; generating compact 81-frame cache."
    else
        echo "[$(date --iso-8601=seconds)] Rebuilding legacy or partial cache ($cached_samples/$expected_samples) as $CACHE_FORMAT."
    fi
    mkdir -p "$CACHE_ROOT"
    cd "$ROOT/code/Python_3D_Scanner"
    "$PYTHON" -m accelerate.commands.launch \
        --num_processes 1 \
        --num_machines 1 \
        --mixed_precision bf16 \
        --dynamo_backend no \
        "$TRAIN_ENTRY" \
        --diffsynth-root "$DIFFSYNTH_ROOT" \
        --num-processes 1 \
        --backend nccl \
        --compact-sft-cache \
        --shared-text-context-path "$SHARED_TEXT_CONTEXT" \
        --dataset_base_path "$DATASET_ROOT" \
        --dataset_metadata_path "$METADATA" \
        --dataset_repeat 1 \
        --dataset_num_workers 4 \
        --model_paths "$MODEL_PATHS" \
        --tokenizer_path "$TOKENIZER" \
        --offload_models "$DIT" \
        --output_path "$CACHE_ROOT" \
        --task sft:data_process \
        --height 384 \
        --width 384 \
        --num_frames 81 \
        --lora_base_model dit \
        --lora_target_modules q,k,v,o,ffn.0,ffn.2 \
        --lora_rank 16 \
        --use_gradient_checkpointing
fi

cached_samples=$(find "$CACHE_ROOT" -type f -name '*.pth' | wc -l)
if [[ "$cached_samples" -ne "$expected_samples" ]]; then
    echo "Cache validation failed: expected $expected_samples files, found $cached_samples" >&2
    exit 5
fi
if [[ ! -f "$SHARED_TEXT_CONTEXT" ]]; then
    echo "Cache validation failed: shared text context is missing: $SHARED_TEXT_CONTEXT" >&2
    exit 5
fi
touch "$CACHE_MARKER"

mkdir -p "$OUTPUT_ROOT"
existing_checkpoints=$(find "$OUTPUT_ROOT" -maxdepth 1 -type f -name '*.safetensors' | wc -l)
if [[ "$existing_checkpoints" -ne 0 ]]; then
    echo "Output already contains $existing_checkpoints checkpoint(s): $OUTPUT_ROOT" >&2
    echo "Refusing to mix a new run with an existing run." >&2
    exit 6
fi

echo "[$(date --iso-8601=seconds)] Cache ready; starting LoRA training."
cd "$ROOT/code/Python_3D_Scanner"

exec "$PYTHON" "$TRAIN_ENTRY" \
    --diffsynth-root "$DIFFSYNTH_ROOT" \
    --num-processes 1 \
    --backend nccl \
    --shared-text-context-path "$SHARED_TEXT_CONTEXT" \
    --dataset_base_path "$CACHE_ROOT" \
    --dataset_repeat 1 \
    --dataset_num_workers 2 \
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
    --gradient_accumulation_steps 1 \
    --lora_base_model dit \
    --lora_target_modules q,k,v,o,ffn.0,ffn.2 \
    --lora_rank 16 \
    --remove_prefix_in_ckpt pipe.dit. \
    --use_gradient_checkpointing \
    --enable_csv_log
