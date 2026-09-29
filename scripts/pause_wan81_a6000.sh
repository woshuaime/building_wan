#!/bin/bash
# Record the exact pause point of a running A6000 LoRA training, then optionally stop it.
#
#   bash scripts/pause_wan81_a6000.sh          # only record, training keeps running
#   bash scripts/pause_wan81_a6000.sh --stop   # record, then stop the training process
#
# The recorded JSON is what scripts/resume_from_pause.sh reads later.

set -euo pipefail

ROOT="${BUILDING_WAN_ROOT:-/mnt/windowsE/chuanjun/building_wan}"
LOG_DIR="$ROOT/logs"
STOP=0

if [[ "${1:-}" == "--stop" ]]; then
    STOP=1
elif [[ -n "${1:-}" ]]; then
    echo "Usage: $0 [--stop]" >&2
    exit 2
fi

cd "$ROOT"
mkdir -p "$LOG_DIR"

pid="$(pgrep -f wan_train_entry.py | head -n 1 || true)"
if [[ -z "$pid" ]]; then
    echo "没有正在运行的训练进程（wan_train_entry.py），无需暂停。" >&2
    exit 3
fi

cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
output_dir="$(sed -n 's/.*--output_path \([^ ]*\).*/\1/p' <<<"$cmdline")"
if [[ -z "$output_dir" || ! -d "$output_dir" ]]; then
    echo "无法从进程参数中解析输出目录: $output_dir" >&2
    exit 4
fi

checkpoint="$(ls -1 "$output_dir"/step-*.safetensors 2>/dev/null | sort -t- -k2 -n | tail -n 1 || true)"
if [[ -z "$checkpoint" ]]; then
    echo "输出目录里还没有任何权重，无法确定恢复点: $output_dir" >&2
    exit 5
fi

checkpoint_step="$(basename "$checkpoint" | sed -n 's/step-\([0-9]*\)\.safetensors/\1/p')"
current_step="$(tail -n 1 "$output_dir/loss.csv" | cut -d, -f1)"

timestamp="$(date +%Y%m%d_%H%M%S)"
state="$LOG_DIR/pause_state_$timestamp.json"
next_output="$ROOT/checkpoints/building_wan_a6000_single_orbit_all/building_wan_a6000_single_orbit_resumed_$timestamp"

python3 - "$output_dir" "$checkpoint" "$checkpoint_step" "$current_step" "$next_output" "$pid" "$state" <<'PY'
import json
import sys
from pathlib import Path

output_dir, checkpoint, checkpoint_step, current_step, next_output, pid, state_path = sys.argv[1:8]
output_dir = Path(output_dir)

args = json.loads((output_dir / "training_args.json").read_text(encoding="utf-8"))
run_skip = int(args.get("skip_train_batches", 0))
run_initial = int(args.get("initial_train_steps", 0))
run_batch = int(args.get("project_train_batch_size", 1))
grad_accum = int(args.get("gradient_accumulation_steps", 1))

checkpoint_step = int(checkpoint_step)
current_step = int(current_step)

# Dataloader position the next run must skip to, counted in samples of the
# current run's batch size. This is independent of the step number printed in
# the logs, so it is passed explicitly rather than derived from the step.
resume_position = run_skip + (checkpoint_step - run_initial) * run_batch

payload = {
    "recorded_at": __import__("datetime").datetime.now().astimezone().isoformat(),
    "training_pid": int(pid),
    "output_dir": str(output_dir),
    "last_checkpoint": checkpoint,
    "last_checkpoint_step": checkpoint_step,
    "current_step_when_paused": current_step,
    "lost_steps": current_step - checkpoint_step,
    "resume": {
        "WAN_RESUME_FROM_CHECKPOINT": checkpoint,
        "WAN_RESUME_CHECKPOINT_STEP": checkpoint_step,
        "WAN_RESUME_CHECKPOINT_BATCH_SIZE": run_batch,
        "WAN_TRAIN_BATCH_SIZE": run_batch,
        "WAN_GRADIENT_ACCUMULATION_STEPS": grad_accum,
        "WAN_RESUME_INITIAL_STEPS": checkpoint_step,
        "WAN_RESUME_SKIP_BATCHES": resume_position,
    },
    "next_output_dir": next_output,
    "note": (
        f"从 step-{checkpoint_step} 恢复，step-{checkpoint_step}→{current_step} "
        f"这 {current_step - checkpoint_step} 步会重跑。"
    ),
}

Path(state_path).write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)

print(f"暂停点已记录: {state_path}")
print(f"  当前步数       : {current_step}")
print(f"  最后权重       : {checkpoint}")
print(f"  恢复起点       : step-{checkpoint_step}")
print(f"  会重跑的步数   : {current_step - checkpoint_step}")
print(f"  下次输出目录   : {next_output}")
PY

if (( STOP )); then
    echo
    echo "正在停止训练进程 $pid ..."
    kill -TERM "$pid"
    for _ in $(seq 1 30); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "训练进程已退出。"
            break
        fi
        sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
        echo "进程未响应 SIGTERM，改用 SIGKILL。"
        kill -KILL "$pid"
    fi
    echo
    echo "暂停完成。对比跑完后用下面这条命令继续训练:"
    echo "  bash scripts/resume_from_pause.sh --start"
else
    echo
    echo "仅记录，训练仍在运行。要真正停止请加 --stop。"
fi
