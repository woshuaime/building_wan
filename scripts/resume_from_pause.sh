#!/bin/bash
# Resume training from the point recorded by scripts/pause_wan81_a6000.sh.
#
#   bash scripts/resume_from_pause.sh                 # show what would run
#   bash scripts/resume_from_pause.sh --start         # actually start training
#   bash scripts/resume_from_pause.sh --state logs/pause_state_X.json --start
#
# Training is refused while another wan_train_entry.py process is alive.

set -euo pipefail

ROOT="${BUILDING_WAN_ROOT:-/mnt/windowsE/chuanjun/building_wan}"
PYTHON="${WAN_PYTHON:-/home/shi/miniconda3/envs/scj/bin/python}"

STATE=""
START=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --state) STATE="$2"; shift 2 ;;
        --start) START=1; shift ;;
        *) echo "Usage: $0 [--state FILE] [--start]" >&2; exit 2 ;;
    esac
done

cd "$ROOT"

if [[ -z "$STATE" ]]; then
    STATE="$(ls -1t logs/pause_state_*.json 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "$STATE" || ! -f "$STATE" ]]; then
    echo "找不到暂停记录。先运行: bash scripts/pause_wan81_a6000.sh" >&2
    exit 3
fi

echo "使用暂停记录: $STATE"
read -r CKPT OUTDIR SKIP INITIAL BATCH ACCUM CHECKPOINT_STEP < <(
    "$PYTHON" - "$STATE" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
resume = payload["resume"]
print(
    resume["WAN_RESUME_FROM_CHECKPOINT"],
    payload["next_output_dir"],
    resume["WAN_RESUME_SKIP_BATCHES"],
    resume["WAN_RESUME_INITIAL_STEPS"],
    resume["WAN_RESUME_CHECKPOINT_BATCH_SIZE"],
    resume["WAN_GRADIENT_ACCUMULATION_STEPS"],
    resume["WAN_RESUME_CHECKPOINT_STEP"],
)
PY
)

echo "  权重          : $CKPT"
echo "  步号起点      : step-$INITIAL"
echo "  跳过样本数    : $SKIP"
echo "  物理 batch    : $BATCH  梯度累积: $ACCUM"
echo "  新输出目录    : $OUTDIR"
echo

[[ -f "$CKPT" ]] || { echo "权重文件不存在: $CKPT" >&2; exit 4; }

if ps -eo args= | grep -F "wan_train_entry.py" | grep -v grep >/dev/null; then
    echo "已有训练进程在运行，拒绝重复启动。" >&2
    exit 7
fi

if (( ! START )); then
    echo "演练模式，未启动训练。确认无误后加 --start。"
    exit 0
fi

if [[ -e "$OUTDIR" ]] && find "$OUTDIR" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    echo "输出目录非空，拒绝启动: $OUTDIR" >&2
    exit 6
fi

LOG="$ROOT/logs/a6000_resume_from_pause_$(date +%Y%m%d_%H%M%S).log"

WAN_RESUME_OUTPUT_ROOT="$OUTDIR" \
WAN_RESUME_FROM_CHECKPOINT="$CKPT" \
WAN_RESUME_CHECKPOINT_STEP="$CHECKPOINT_STEP" \
WAN_RESUME_CHECKPOINT_BATCH_SIZE="$BATCH" \
WAN_RESUME_INITIAL_STEPS="$INITIAL" \
WAN_RESUME_SKIP_BATCHES="$SKIP" \
WAN_TRAIN_BATCH_SIZE="$BATCH" \
WAN_GRADIENT_ACCUMULATION_STEPS="$ACCUM" \
WAN_DATASET_NUM_WORKERS=0 \
setsid nohup bash "$ROOT/scripts/resume_wan81_a6000.sh" >"$LOG" 2>&1 &

sleep 5
echo "训练已启动。"
echo "  日志: $LOG"
echo "  查看: tail -c 500 \"$LOG\" | tr '\\r' '\\n' | tail -n 5"
