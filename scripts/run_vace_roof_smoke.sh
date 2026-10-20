#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${WAN_PYTHON:-/home/shi/miniconda3/envs/scj/bin/python}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export DIFFSYNTH_SKIP_DOWNLOAD=True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd -- "$PROJECT_ROOT"

[[ $# -le 1 ]] || { printf 'Use only --dry-run, --run, or --tmux.\n' >&2; exit 2; }
MODE="${1:---dry-run}"
case "$MODE" in
    --dry-run) exec "$PYTHON_BIN" -u -B scripts/vace_roof_smoke.py --dry-run ;;
    --run) exec "$PYTHON_BIN" -u -B scripts/vace_roof_smoke.py --run ;;
    --tmux)
        if [[ -n "${TMUX:-}" ]]; then
            printf 'Using current tmux. Reconnect: tmux attach-session -t %s\n' "$(tmux display-message -p '#S')"
            exec "$PYTHON_BIN" -u -B scripts/vace_roof_smoke.py --run
        fi
        command -v tmux >/dev/null || { printf 'tmux unavailable; use --run in a foreground terminal.\n' >&2; exit 1; }
        printf -v RUN_COMMAND 'bash %q --run' "$PROJECT_ROOT/scripts/run_vace_roof_smoke.sh"
        START_COMMAND="$RUN_COMMAND"'; status=$?; printf "\nVACE roof smoke exit code: %s\n" "$status"; exec bash'
        printf 'Reconnect: tmux attach-session -t vace_roof_smoke\n'
        exec tmux new-session -A -s vace_roof_smoke "$START_COMMAND"
        ;;
    --help) printf 'CPU only (default): --dry-run\nManual GPU start: --tmux (recoverable) or --run (foreground)\n' ;;
    *) printf 'Unknown argument: %s\n' "$MODE" >&2; exit 2 ;;
esac
