"""Measure simple visual diagnostics and build a checkpoint contact sheet."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def checkpoint_sort_key(path: Path) -> tuple[int, int | str]:
    if path.stem == "base":
        return (0, 0)
    match = re.fullmatch(r"step-(\d+)", path.stem)
    if match:
        return (1, int(match.group(1)))
    return (2, path.stem)


def load_frames(path: Path) -> np.ndarray:
    reader = imageio.get_reader(str(path))
    try:
        frames = np.asarray([frame[..., :3] for frame in reader], dtype=np.uint8)
    finally:
        reader.close()
    if frames.ndim != 4 or not len(frames):
        raise ValueError(f"No RGB video frames decoded from {path}")
    return frames


def video_metrics(frames: np.ndarray) -> dict[str, float | int]:
    values = frames.astype(np.float32) / 255.0
    gray = values.mean(axis=3)
    foreground = np.min(values, axis=3) < (235.0 / 255.0)
    height, width = foreground.shape[1:]
    border_y = max(1, round(height * 0.08))
    border_x = max(1, round(width * 0.08))
    border = np.zeros((height, width), dtype=bool)
    border[:border_y] = True
    border[-border_y:] = True
    border[:, :border_x] = True
    border[:, -border_x:] = True
    white_border = np.min(values[:, border, :], axis=2) > (240.0 / 255.0)

    yy, xx = np.indices((height, width), dtype=np.float32)
    offsets = []
    occupancies = []
    for mask in foreground:
        occupancy = float(mask.mean())
        occupancies.append(occupancy)
        if mask.any():
            cx = float(xx[mask].mean())
            cy = float(yy[mask].mean())
            offsets.append(
                ((cx - (width - 1) / 2) ** 2 + (cy - (height - 1) / 2) ** 2)
                ** 0.5
                / ((width**2 + height**2) ** 0.5 / 2)
            )

    frame_differences = np.abs(np.diff(values, axis=0))
    mean_luminance = gray.mean(axis=(1, 2))
    return {
        "frames": int(len(frames)),
        "width": int(width),
        "height": int(height),
        "foreground_occupancy_mean": round(float(np.mean(occupancies)), 6),
        "foreground_occupancy_std": round(float(np.std(occupancies)), 6),
        "foreground_centroid_offset_mean": round(float(np.mean(offsets)), 6),
        "foreground_centroid_offset_std": round(float(np.std(offsets)), 6),
        "white_border_ratio": round(float(white_border.mean()), 6),
        "mean_frame_difference": round(float(frame_differences.mean()), 6),
        "mean_luminance_change": round(
            float(np.abs(np.diff(mean_luminance)).mean()), 6
        ),
    }


def make_contact_sheet(
    videos: list[tuple[str, Path, np.ndarray]], output: Path, samples: int
) -> None:
    thumb_width = 256
    label_height = 30
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    font = ImageFont.load_default()
    sampled_rows = []
    for label, _, frames in videos:
        indices = np.linspace(0, len(frames) - 1, samples, dtype=int)
        thumbnails = []
        for index in indices:
            image = Image.fromarray(frames[index])
            thumb_height = round(image.height * thumb_width / image.width)
            thumbnails.append(image.resize((thumb_width, thumb_height), resampling))
        sampled_rows.append((label, indices.tolist(), thumbnails))

    thumb_height = sampled_rows[0][2][0].height
    canvas = Image.new(
        "RGB",
        (thumb_width * samples, (thumb_height + label_height) * len(sampled_rows)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for row, (label, indices, thumbnails) in enumerate(sampled_rows):
        top = row * (thumb_height + label_height)
        draw.rectangle((0, top, canvas.width, top + label_height), fill="#17324D")
        draw.text((10, top + 8), f"{label} | frames {indices}", fill="white", font=font)
        for column, image in enumerate(thumbnails):
            canvas.paste(image, (column * thumb_width, top + label_height))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validation_dir", type=Path)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    directory = args.validation_dir.resolve()
    paths = sorted(directory.glob("*.mp4"), key=checkpoint_sort_key)
    if not paths:
        raise SystemExit(f"No MP4 validation videos in {directory}")

    videos = []
    results = []
    for path in paths:
        frames = load_frames(path)
        videos.append((path.stem, path, frames))
        results.append(
            {"label": path.stem, "path": str(path), "metrics": video_metrics(frames)}
        )

    sheet_path = directory / "checkpoint_contact_sheet.png"
    make_contact_sheet(videos, sheet_path, args.samples)
    payload = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "validation_dir": str(directory),
        "contact_sheet": str(sheet_path),
        "threshold_notes": {
            "foreground": "at least one RGB channel below 235/255",
            "white_border": "all RGB channels above 240/255 in outer 8% border",
        },
        "results": results,
    }
    output = directory / "video_diagnostics.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)
    print(sheet_path)


if __name__ == "__main__":
    main()
