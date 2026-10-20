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

MODE=""
RESUME_ARGS=()
for ARGUMENT in "$@"; do
    case "$ARGUMENT" in
        --help)
            [[ $# -eq 1 ]] || exit 2
            printf '%s\n' 'Reuse P01 and sequentially run only P02-P08; four files each under outputs/vace/test/01-08.' \
                '--dry-run: read-only checks; --prepare: checks and P01 copy only; --tmux: visible recoverable batch.' \
                '--resume: explicitly verify/reuse completed results and run only the missing suffix; combine with --dry-run or --tmux.'
            exit 0 ;;
        --resume) [[ ${#RESUME_ARGS[@]} -eq 0 ]] || exit 2; RESUME_ARGS=(--resume) ;;
        --dry-run|--prepare|--tmux) [[ -z "$MODE" ]] || exit 2; MODE="$ARGUMENT" ;;
        *) printf 'Unknown argument: %s\n' "$ARGUMENT" >&2; exit 2 ;;
    esac
done

case "$MODE" in
    --dry-run|--prepare) exec "$PYTHON_BIN" -u -B scripts/vace_building_test.py "$MODE" "${RESUME_ARGS[@]}" ;;
    --tmux)
        if [[ -n "${TMUX:-}" ]]; then
            printf 'Using current tmux session. Reconnect: tmux attach-session -t %s\n' "$(tmux display-message -p '#S')"
        elif command -v tmux >/dev/null; then
            printf -v RUN_COMMAND 'bash %q' "$PROJECT_ROOT/scripts/run_vace_building_test.sh"
            if [[ ${#RESUME_ARGS[@]} -gt 0 ]]; then RUN_COMMAND+=' --resume'; fi
            START_COMMAND="$RUN_COMMAND"'; status=$?; printf "\nBatch exit code: %s\n" "$status"; exec bash'
            exec tmux new-session -A -s vace_building_test "$START_COMMAND"
        else
            printf '%s\n' 'tmux unavailable; running in this foreground terminal without installing software.'
        fi
        ;;
esac

exec "$PYTHON_BIN" -u -B scripts/vace_building_test.py --run "${RESUME_ARGS[@]}"
