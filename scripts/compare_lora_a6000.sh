#!/bin/bash
# Compare five LoRA checkpoints (plus the base model) with one shared config.
#
#   bash scripts/compare_lora_a6000.sh              # dry run, writes the config only
#   bash scripts/compare_lora_a6000.sh --run        # also generate the six videos
#   bash scripts/compare_lora_a6000.sh --run --date 0928
#
# Output goes to outputs/validation/<date>/ in the project root.
# Training must not be running: the validator loads the full Wan pipeline.

set -euo pipefail

ROOT="${BUILDING_WAN_ROOT:-/mnt/windowsE/chuanjun/building_wan}"
PYTHON="${WAN_PYTHON:-/home/shi/miniconda3/envs/scj/bin/python}"

RUN=0
DATE="$(date +%-m.%-d)"
OVERWRITE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run) RUN=1; shift ;;
        --date) DATE="$2"; shift 2 ;;
        --overwrite) OVERWRITE=1; shift ;;
        *) echo "Usage: $0 [--run] [--date MMDD] [--overwrite]" >&2; exit 2 ;;
    esac
done

cd "$ROOT"

FIRST="checkpoints/building_wan_a6000_single_orbit_all/building_wan_a6000_single_orbit"
SECOND_DIR="$("$PYTHON" - <<'PY'
import glob, os
cands = [
    d for d in glob.glob("checkpoints/building_wan_a6000_single_orbit_all/building_wan_a6000_single_orbit_resumed_epoch4_*")
    if os.path.isdir(d) and not d.endswith(".lock")
]
print(max(cands, key=os.path.getmtime) if cands else "")
PY
)"

if [[ -z "$SECOND_DIR" ]]; then
    echo "找不到今天的续训目录。" >&2
    exit 3
fi

OUTDIR="$ROOT/outputs/validation/$DATE"
mkdir -p "$OUTDIR"

CONFIG="$OUTDIR/validation_config.json"

"$PYTHON" - "$ROOT" "$FIRST" "$SECOND_DIR" "$OUTDIR" "$CONFIG" <<'PY'
import csv
import json
import sys
from pathlib import Path

root, first_dir, second_dir, outdir, config_path = sys.argv[1:6]
root = Path(root)
first_dir = root / first_dir
second_dir = root / second_dir

# Five checkpoints spanning the whole training history. Labels are explicit so
# the same step number from different runs can never be confused.
plan = [
    ("base", None),
    ("B-step800", first_dir / "step-800.safetensors"),
    ("B-step2400", first_dir / "step-2400.safetensors"),
    ("A-step3200", second_dir / "step-3200.safetensors"),
    ("A-step3600", second_dir / "step-3600.safetensors"),
]

checkpoints = []
for label, path in plan:
    if path is not None and not path.is_file():
        raise SystemExit(f"权重不存在: {path}")
    checkpoints.append({"label": label, "path": None if path is None else str(path)})

split_root = root / "configs/single_orbit_same_prompt_v1"
val_metadata = split_root / "metadata_val.csv"
manifest = json.loads((split_root / "split_manifest.json").read_text(encoding="utf-8"))

with val_metadata.open(encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))
if not rows:
    raise SystemExit("验证集清单为空")
val_count = len(rows)
prompt = rows[0]["prompt"].strip()
if val_count != manifest["counts"]["val"]:
    raise SystemExit(f"验证集数量不符: {val_count} != {manifest['counts']['val']}")

config = {
    "experiment_name": f"compare_lora_{Path(outdir).name}",
    "runtime": {
        "python": sys.executable,
        "diffsynth_root": str(root / "code/DiffSynth-Studio"),
        "gpu": 0,
        "model_paths": [
            str(root / "models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"),
            str(root / "models/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors"),
            str(root / "models/Wan-Series-Converted-Safetensors/Wan2.1_VAE.safetensors"),
        ],
        "tokenizer_path": str(root / "models/Wan2.1-T2V-1.3B/google/umt5-xxl"),
    },
    "dataset": {
        "root": str(root / "data/single_orbit_videos"),
        "metadata": str(val_metadata),
        "expected_count": val_count,
    },
    "training": {
        "output_dir": str(second_dir),
        "height": 384,
        "width": 384,
        "num_frames": 81,
    },
    "validation": {
        "output_dir": str(outdir),
        "checkpoints": checkpoints,
        "prompt": prompt,
        "negative_prompt": (
            "people, trees, cars, text, watermark, cluttered background, "
            "camera shake, flicker, deformed geometry, duplicated building, "
            "cropped object, low quality, blurry"
        ),
        "seed": int(manifest["seed"]),
        "height": 384,
        "width": 384,
        "num_frames": 81,
        "num_inference_steps": 30,
        "cfg_scale": 5.0,
        "fps": 18,
        "quality": 5,
        "lora_alpha": 1.0,
        "vram_reserved_gb": 2.0,
        "enable_vram_management": True,
    },
}

Path(config_path).write_text(
    json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(f"对比配置已写入: {config_path}")
print(f"对比项 ({len(checkpoints)} 个):")
for item in checkpoints:
    name = "基础模型（不加 LoRA）" if item["path"] is None else Path(item["path"]).name
    print(f"  {item['label']:12s} {name}")
print(f"输出目录: {outdir}")
PY

echo
if ps -eo args= | grep -F "wan_train_entry.py" | grep -v grep >/dev/null; then
    echo "警告：训练进程还在运行。验证会加载完整 Wan 管线，可能与训练抢显存。"
    echo "      建议先暂停训练：bash scripts/pause_wan81_a6000.sh --stop"
    echo
fi

if (( ! RUN )); then
    echo "演练完成，只生成了配置。确认无误后加 --run 生成视频:"
    echo "  bash scripts/compare_lora_a6000.sh --run --date $DATE"
    exit 0
fi

RUNCMD=("$PYTHON" "$ROOT/code/Python_3D_Scanner/training/validate_domain_lora.py"
        --config "$CONFIG" --run)
if (( OVERWRITE )); then
    RUNCMD+=(--overwrite)
fi

echo "开始生成验证视频 ..."
"${RUNCMD[@]}"

echo
echo "完成。视频在: $OUTDIR"
ls -l "$OUTDIR"/*.mp4 2>/dev/null || true
