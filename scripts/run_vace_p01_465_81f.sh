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

cd -- "$PROJECT_ROOT"
if [[ $# -gt 0 ]]; then
    [[ $# -eq 1 ]] || exit 2
    case "$1" in
        --help) printf '%s\n' 'Foreground single 81F experiment, retaining only four files.' \
            'Use --tmux to start/attach a visible recoverable session, or reuse the current tmux session.' \
            'Use --dry-run for read-only checks without inference or output files.'; exit 0 ;;
        --dry-run) exec "$PYTHON_BIN" -u -B scripts/vace_p01_81f.py --dry-run ;;
        --tmux)
            if [[ -n "${TMUX:-}" ]]; then
                printf 'Using current tmux session. Reconnect with: tmux attach-session -t %s\n' "$(tmux display-message -p '#S')"
            elif command -v tmux >/dev/null; then
                printf -v START_COMMAND 'bash %q; status=$?; printf "\nLauncher exit code: %%s\n" "$status"; exec bash' \
                    "$PROJECT_ROOT/scripts/run_vace_p01_465_81f.sh"
                exec tmux new-session -A -s vace_p01_81f "$START_COMMAND"
            else
                printf '%s\n' 'tmux unavailable; using this foreground terminal without installing software.'
            fi
            ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
fi

exec "$PYTHON_BIN" -u -B scripts/vace_p01_81f.py --run
