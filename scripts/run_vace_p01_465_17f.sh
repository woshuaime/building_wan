#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${WAN_PYTHON:-/home/shi/miniconda3/envs/scj/bin/python}"

if [[ $# -gt 0 ]]; then
    case "$1" in
        --help)
            printf '%s\n' 'Run once in the foreground: bash scripts/run_vace_p01_465_17f.sh' \
                'Read-only preflight: bash scripts/run_vace_p01_465_17f.sh --dry-run' \
                'No model downloads, training, automatic retries or frozen-parameter changes.'
            exit 0
            ;;
        --dry-run)
            [[ $# -eq 1 ]] || exit 2
            cd -- "$PROJECT_ROOT"
            exec "$PYTHON_BIN" -u -B scripts/vace_p01.py --dry-run --verify-models --check-loader
            ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
fi

cd -- "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export DIFFSYNTH_SKIP_DOWNLOAD=True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
OUTPUT_DIR="$PROJECT_ROOT/outputs/vace/P01_465_17f_baseline_run_$(date -u +%Y%m%dT%H%M%S)_$$"
printf '\nOne foreground VACE P01 experiment. Output directory:\n%s\n\n' "$OUTPUT_DIR"
exec "$PYTHON_BIN" -u -B scripts/vace_p01.py --run --output-dir "$OUTPUT_DIR"
