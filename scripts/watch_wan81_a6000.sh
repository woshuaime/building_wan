#!/bin/bash
# Live training progress monitor.
#
# Finds the output directory from the running process, so restarting training
# never requires editing this file.
#
#   bash scripts/watch_wan81_a6000.sh

cd "${BUILDING_WAN_ROOT:-/mnt/windowsE/chuanjun/building_wan}"

TOTAL=1992        # samples per pass over the dataset (batch size 1)
TOTAL_EPOCHS=8    # --num_epochs

START=$(date +%s)
STARTSTEP=0
RATE_STEP=0
RATE_TIME=$(date +%s)

while true; do
  clear
  date

  OUT=$(ps -eo args= 2>/dev/null | grep wan_train_entry.py | grep -v grep \
        | grep -o -- '--output_path [^ ]*' | awk '{print $2}' | head -n 1)

  if [[ -z "$OUT" || ! -d "$OUT" ]]; then
    echo "!!! 训练进程已退出 !!!"
    echo "最后目录: ${OUT:-无}"
    break
  fi

  CSV="$OUT/loss.csv"
  ARGS="$OUT/training_args.json"
  if [[ ! -f "$CSV" || ! -f "$ARGS" ]]; then
    echo "等待文件生成 ... ($OUT)"
    sleep 10
    continue
  fi

  # BASE: step number this run continues counting from.
  # SKIP: samples already consumed before this run started training.
  read -r BASE SKIP < <(python3 -c "
import json
a = json.load(open('$ARGS'))
print(a['initial_train_steps'], a['skip_train_batches'])
")

  S=$(tail -n 1 "$CSV" | cut -d, -f1)
  L=$(tail -n 1 "$CSV" | cut -d, -f3)

  # Samples trained since the very first run started at step 0.
  CUM=$(( SKIP + S - BASE ))
  PASS=$(( CUM / TOTAL ))        # finished passes over the dataset
  POS=$(( CUM % TOTAL ))         # position inside the current pass

  # This run stops when the 8-epoch schedule is exhausted.
  TRAIN_TOTAL=$(( TOTAL * TOTAL_EPOCHS - SKIP ))
  DONE_RUN=$(( S - BASE ))
  REMAIN=$(( TRAIN_TOTAL - DONE_RUN ))

  BARW=30
  W=$(( POS * BARW / TOTAL ))
  BAR=$(printf '█%.0s' $(seq 1 $W 2>/dev/null))
  REST=$(printf '░%.0s' $(seq 1 $(( BARW - W )) 2>/dev/null))

  # ETA from a speed average over at least 3 minutes, to avoid jumping values.
  NOW=$(date +%s)
  SPAN=$(( NOW - RATE_TIME ))
  if (( SPAN >= 180 )) && (( S > RATE_STEP )); then
    SPD=$(( SPAN / (S - RATE_STEP) ))          # seconds per step
    RATE_STEP=$S
    RATE_TIME=$NOW
    LAST_SPD=$SPD
  fi
  if (( REMAIN > 0 )) && [[ -n "${LAST_SPD:-}" ]]; then
    ETA=$(( REMAIN * LAST_SPD ))
    ETA_TXT="$(( ETA / 3600 ))小时$(( (ETA % 3600) / 60 ))分"
  else
    ETA_TXT="计算中（需累计约 3 分钟速度样本）"
  fi

  echo "目录: $(basename "$OUT")"
  echo "步数: $S   （从 step-$BASE 起，本次已跑 $DONE_RUN 步）"
  echo
  echo "累计训练量: 已完成 $PASS 遍 + $POS/$TOTAL  [$BAR$REST] $(( POS * 100 / TOTAL ))%"
  echo "本次剩余:   $REMAIN 步，约 $ETA_TXT"
  echo
  echo "最近 loss: $L"

  NEWEST=$(ls -1t "$OUT"/step-*.safetensors 2>/dev/null | head -n 1)
  echo "权重: $(ls -1 "$OUT"/step-*.safetensors 2>/dev/null | wc -l) 个"
  [[ -n "$NEWEST" ]] && echo "最新: $(basename "$NEWEST")  ($(date -r "$NEWEST" +%H:%M))"

  NEXT_AT=$(( (DONE_RUN / 400 + 1) * 400 + BASE ))
  [[ "$REMAIN" -gt 0 ]] && echo "下一个存档: step-$NEXT_AT"
  echo "──────────────────────────────────────"

  STARTSTEP=$S
  sleep 15
done