"""对验证视频做客观量化分析：闪烁、时序稳定性、运动连续性、亮度漂移。

不依赖人工观看，直接对像素统计，输出每个权重的指标表。
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import imageio.v3 as iio
import numpy as np

VIDEO_DIR = Path("outputs/validation/validation_all_single_orbit")
OUT = VIDEO_DIR / "_review"
STEP_RE = re.compile(r"step-(\d+)\.mp4")


def load(path: Path) -> np.ndarray:
    frames = iio.imread(path, plugin="pyav")
    return frames.astype(np.float32) / 255.0


def analyze(frames: np.ndarray) -> dict:
    gray = frames.mean(axis=3)
    n = gray.shape[0]

    # 相邻帧差异：整体变化量（运动幅度）
    diffs = np.abs(np.diff(gray, axis=0))
    motion = diffs.reshape(n - 1, -1).mean(axis=1)

    # 闪烁：非刚体抖动。用每帧相对前一帧的残差，但是减去低秩的全局位移估计。
    # 简化：计算每个像素在时间轴上的二阶差分（加速度），再对前景区域取均值
    foreground = gray.max(axis=0) - gray.min(axis=0) > 0.05
    if foreground.sum() < 100:
        foreground = np.ones_like(gray[0], dtype=bool)
    accel = np.abs(np.diff(gray, n=2, axis=0)).reshape(n - 2, -1)
    flicker = accel.mean(axis=1)

    # 亮度漂移
    brightness = gray.reshape(n, -1).mean(axis=1)

    # 前景面积占用（建筑在画面里的比例）
    coverage = np.array([(f[foreground] < 0.9).mean() for f in gray])

    return {
        "frames": n,
        "motion_mean": float(motion.mean()),
        "motion_std": float(motion.std()),
        "motion_max": float(motion.max()),
        "flicker_mean": float(flicker.mean()),
        "flicker_std": float(flicker.std()),
        "brightness_mean": float(brightness.mean()),
        "brightness_drift": float(brightness.max() - brightness.min()),
        "fg_coverage_mean": float(coverage.mean()),
        "fg_coverage_std": float(coverage.std()),
    }


def main() -> None:
    videos = []
    for path in VIDEO_DIR.glob("*.mp4"):
        match = STEP_RE.fullmatch(path.name)
        if match:
            videos.append((int(match.group(1)), path))
    base = VIDEO_DIR / "base.mp4"
    videos.sort()
    if base.is_file():
        videos.insert(0, (-1, base))

    rows = []
    for step, path in videos:
        name = "base" if step < 0 else f"step-{step}"
        stats = analyze(load(path))
        stats["label"] = name
        stats["step"] = step
        rows.append(stats)
        print(
            f"{name:>12}  motion={stats['motion_mean']:.4f}  "
            f"flicker={stats['flicker_mean']:.5f}  "
            f"bright_drift={stats['brightness_drift']:.4f}  "
            f"coverage={stats['fg_coverage_mean']:.4f}±{stats['fg_coverage_std']:.4f}"
        )

    OUT.mkdir(exist_ok=True)
    with (OUT / "video_metrics.csv").open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n已写入 {OUT}/video_metrics.csv")


if __name__ == "__main__":
    main()
