"""抽取验证视频的静帧并拼接，供人工/模型审阅。

输出：
    frames/            每个视频的 3 帧（首/中/尾）单独图片
    montage_steps.png  全部 step 的中帧横向拼接
    montage_triplet.png  每个视频的首/中/尾三连图
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

VIDEO_DIR = Path("outputs/validation/validation_all_single_orbit")
OUT = VIDEO_DIR / "_review"
STEP_RE = re.compile(r"step-(\d+)\.mp4")


def load_frames(path: Path) -> np.ndarray:
    return iio.imread(path, plugin="pyav")


def label_tile(image: Image.Image, text: str, height: int = 22) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + height), "white")
    canvas.paste(image, (0, height))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15
        )
    except OSError:
        font = ImageFont.load_default()
    draw.text((4, 3), text, fill="black", font=font)
    return canvas


def grid(tiles: list[Image.Image], columns: int, gap: int = 6) -> Image.Image:
    rows = (len(tiles) + columns - 1) // columns
    w = max(t.width for t in tiles)
    h = max(t.height for t in tiles)
    canvas = Image.new(
        "RGB",
        (columns * w + (columns + 1) * gap, rows * h + (rows + 1) * gap),
        (235, 235, 235),
    )
    for index, tile in enumerate(tiles):
        r, c = divmod(index, columns)
        canvas.paste(tile, (gap + c * (w + gap), gap + r * (h + gap)))
    return canvas


def main() -> None:
    if not VIDEO_DIR.is_dir():
        raise SystemExit(f"找不到视频目录: {VIDEO_DIR}")
    OUT.mkdir(exist_ok=True)

    videos = []
    for path in VIDEO_DIR.glob("*.mp4"):
        match = STEP_RE.fullmatch(path.name)
        if match:
            videos.append((int(match.group(1)), path))
    base = VIDEO_DIR / "base.mp4"
    videos.sort()
    if base.is_file():
        videos.insert(0, (-1, base))
    if not videos:
        raise SystemExit("没有找到视频")

    print(f"读取 {len(videos)} 个视频")
    middles = []
    triplets = []
    report = []
    for step, path in videos:
        name = "base" if step < 0 else f"step-{step}"
        frames = load_frames(path)
        total = len(frames)
        picks = [0, total // 2, total - 1]
        triplet = []
        for index, frame_index in enumerate(picks):
            pil = Image.fromarray(frames[frame_index]).convert("RGB")
            tag = ["first", "mid", "last"][index]
            pil = pil.resize((256, 256), Image.LANCZOS)
            pil.save(OUT / f"{name}_{tag}.png")
            triplet.append(pil)
        middles.append(label_tile(triplet[1], name))
        triplets.append(grid([label_tile(t, name) for t in triplet], columns=3))
        report.append((name, total, frames.shape))
        print(f"  {name}: {total} 帧 {frames.shape}")

    grid(middles, columns=5).save(OUT / "montage_steps.png")
    stacked = []
    for tile in triplets:
        stacked.append(tile)
    grid(stacked, columns=1).save(OUT / "montage_triplet.png")
    print(f"\n已写入 {OUT}/montage_steps.png 和 {OUT}/montage_triplet.png")


if __name__ == "__main__":
    sys.exit(main())
