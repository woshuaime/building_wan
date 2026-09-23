#!/bin/bash

set -euo pipefail

ROOT=/home/share/CHUANJUN/building_wan
PYTHON=/home/u2025171963/miniconda3/envs/mv2v/bin/python
DIFFSYNTH_ROOT="$ROOT/code/DiffSynth-Studio"
VALIDATION_ENTRY="$ROOT/code/Python_3D_Scanner/training/validate_domain_lora.py"
ANALYZER="$ROOT/code/Python_3D_Scanner/training/analyze_validation_videos.py"
CONFIG="$ROOT/code/Python_3D_Scanner/training/configs/building_wan_a100_validation.json"
OUTPUT_DIR="$ROOT/outputs/validation/building_wan"

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export DIFFSYNTH_SKIP_DOWNLOAD=True
export PYTHONPATH="$DIFFSYNTH_ROOT${PYTHONPATH:+:$PYTHONPATH}"

required_paths=("$PYTHON" "$DIFFSYNTH_ROOT" "$VALIDATION_ENTRY" "$ANALYZER" "$CONFIG")
for path in "${required_paths[@]}"; do
    if [[ ! -e "$path" ]]; then
        echo "Required validation path is missing: $path" >&2
        exit 2
    fi
done

"$PYTHON" -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; assert torch.cuda.device_count() == 1, f'Expected exactly one visible GPU, found {torch.cuda.device_count()}'; print('Validation GPU:', torch.cuda.get_device_name(0))"

cd "$ROOT/code/Python_3D_Scanner"
"$PYTHON" "$VALIDATION_ENTRY" --config "$CONFIG" --run
"$PYTHON" "$ANALYZER" "$OUTPUT_DIR" --samples 5
