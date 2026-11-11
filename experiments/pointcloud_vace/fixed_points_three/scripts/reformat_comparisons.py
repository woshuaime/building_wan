#!/usr/bin/env python3
"""Re-layout completed comparisons on CPU; never regenerate model or point outputs."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import ROOT, CASES, D_RUN
from geometry import read_cloud

SAMPLES = (("01", "檐口", "eave_3230"), ("04", "长墙", "strip_5016"),
           ("07", "墙面", "wall_10416"))
W, H, FRAMES, FPS = 832, 480, 81, 16
TITLE, HEADER, GAP, FOOTER = 64, 80, 16, 48
SIZE = (3 * W, TITLE + 2 * (H + HEADER) + GAP + FOOTER)
PREVIOUS_SIZE = (3 * W, 1136)
PREVIOUS_FIXED_Y = 656
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def probe(path, expected_size):
    result = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_read_frames",
        "-of", "json", str(path)]))["streams"][0]
    if ((result["width"], result["height"]) != expected_size
            or int(result["nb_read_frames"]) != FRAMES
            or Fraction(result["r_frame_rate"]) != FPS
            or Fraction(result["avg_frame_rate"]) != FPS):
        raise ValueError(f"Unexpected video format: {path}: {result}")
    return result


def decode(path, crop=None):
    command = ["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-threads", "1",
               "-i", str(path), "-map", "0:v:0"]
    if crop:
        command += ["-vf", crop]
    command += ["-vsync", "0", "-threads", "1", "-f", "rawvideo", "-pix_fmt",
                "rgb24", "pipe:1"]
    raw = subprocess.check_output(command)
    return np.frombuffer(raw, np.uint8).reshape(FRAMES, H, W, 3)


def compose(frames, labels, display, name, frame):
    canvas = Image.new("RGB", SIZE, (225, 230, 237))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT, 28)
    draw.text((18, 9), f"{display}  {name}  ·  五栏对照", font=font, fill=(25, 36, 52))
    counter = f"帧 {frame:03d} / 080  ·  16 FPS"
    draw.text((SIZE[0] - draw.textlength(counter, font=font) - 18, 9),
              counter, font=font, fill=(25, 36, 52))
    positions = [(W // 2, TITLE), (3 * W // 2, TITLE)]
    positions += [(col * W, TITLE + HEADER + H + GAP) for col in range(3)]
    count_font = ImageFont.truetype(FONT, 24)
    for (x, y), (label, counts), source in zip(positions, labels, frames):
        draw.rectangle((x, y, x + W - 1, y + HEADER - 1), fill="white")
        draw.text((x + 16, y + 4), label, font=font, fill=(25, 36, 52))
        if draw.textlength(counts, font=count_font) > W - 32:
            raise ValueError(f"Point-count annotation exceeds its panel: {counts}")
        draw.text((x + 16, y + 43), counts, font=count_font, fill=(56, 70, 89))
        canvas.paste(Image.fromarray(source[frame]), (x, y + HEADER))
    draw.text((18, SIZE[1] - FOOTER + 6),
              "三维点数为整栋点集总量（含遮挡点）；D-3000仅输出视频，不能按点状纹理计数。",
              font=count_font, fill=(25, 36, 52))
    return np.asarray(canvas)


def encode(path, sources, labels, display, name):
    command = ["ffmpeg", "-nostdin", "-v", "error", "-n", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{SIZE[0]}x{SIZE[1]}", "-r", str(FPS),
               "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "16", "-threads", "2", "-pix_fmt", "yuv420p", "-movflags",
               "+faststart", str(path)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for index in range(FRAMES):
            process.stdin.write(compose(sources, labels, display, name, index).tobytes())
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError(f"Encoding failed: {path}")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if not process.stdin.closed:
            process.stdin.close()


def reported_counts(report, display, name, record, cloud):
    """Use retained experiment records, never count blobs in a compressed video."""
    pattern = (rf"\|\s*{display}/{re.escape(name)}\s*\|\s*(\d+)\s*\|\s*(\d+)"
               rf"\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|")
    rows = re.findall(pattern, report)
    if len(rows) != 1:
        raise ValueError(f"Need one original point-count row in the report: {name}")
    original, hidden_count, observed, merged = map(int, rows[0])
    points, _ = read_cloud(cloud)
    hidden = np.asarray(record["hidden_original_point_ids"], dtype=int)
    if (len(np.unique(hidden)) != len(hidden)
            or (len(hidden) and (hidden.min() < 0 or hidden.max() >= len(points)))
            or original != len(points) or original != record["cloud_size"]
            or hidden_count != len(hidden) or observed != original - hidden_count
            or merged < observed):
        raise ValueError(f"Source cloud/case/report point counts disagree: {name}")
    return dict(original=original, withheld=hidden_count, observed=observed,
                added=merged - observed, merged=merged, predicted_3d_points=None)


def reformat(delivery, annotate_counts=False):
    # Validate all sources before making any output. The measured reference is
    # the original cloud, NOT a mesh render. The extra distinct view is P-only.
    records = []
    report = (delivery / "实验报告.md").read_text()
    for display, title, name in SAMPLES:
        case = CASES / name
        record = json.loads((case / "case.json").read_text())
        clouds = list((ROOT / "data/pointcloud_vace/sample-data-organized/pointcloud")
                      .glob(record["id"] + ".*"))
        if len(clouds) != 1 or digest(clouds[0]) != record["point_cloud_sha256"]:
            raise ValueError(f"Original point-cloud identity mismatch: {name}")
        for relative in ("reference/observed_target.mp4", "reference/damaged.mp4",
                         "inputs/vace_video.mp4", "reference/camera_poses.json"):
            if digest(case / relative) != record["file_hashes"][relative]:
                raise ValueError(f"Original case identity mismatch: {name}/{relative}")
        final = delivery / f"{display}_{title}_五栏对比.mp4"
        old = final if annotate_counts else delivery / f"{display}_{title}_四栏对比.mp4"
        if not old.is_file() or (final.exists() and not annotate_counts):
            raise FileExistsError(f"Need the intended prior comparison: {old}")
        if annotate_counts:
            probe(old, PREVIOUS_SIZE)
            if digest(old) not in report:
                raise ValueError(f"Prior five-panel video differs from its recorded SHA256: {old}")
            crop = f"crop={W}:{H}:{2 * W}:{PREVIOUS_FIXED_Y}"
        else:
            probe(old, (4 * W, H + 32))
            crop = f"crop={W}:{H}:{2 * W}:32"
        counts = reported_counts(report, display, name, record, clouds[0])
        sources = [case / "reference/observed_target.mp4", case / "reference/damaged.mp4",
                   case / "inputs/vace_video.mp4", D_RUN / "cases" / name / "raw_vace.mp4"]
        for source in sources:
            probe(source, (W, H))
        records.append(dict(display=display, title=title, name=name, old=old,
                            final=final, sources=sources, counts=counts, crop=crop))
    summaries = []
    with tempfile.TemporaryDirectory(prefix=".five_panel_", dir=delivery) as temp:
        for item in records:
            counts = item["counts"]
            frames = [decode(path) for path in item["sources"]]
            # Preserve the existing fixed-point prediction, without refitting.
            frames.append(decode(item["old"], crop=item["crop"]))
            labels = [
                ("Origin · 原始实测点云", f"三维总点数：{counts['original']}"),
                ("残缺点云 · 未填灰", f"三维总点数：{counts['observed']}（人为遮去 {counts['withheld']} 点）"),
                ("VACE 输入 · 缺区填灰", f"来源点云 {counts['observed']} 点 · 灰块仅为编辑占位"),
                ("D-3000 · 视频预测", f"条件点数 {counts['observed']} · 输出三维点数：不适用"),
                ("固定三维补点 · 点云渲染", f"总 {counts['merged']} = 原观测 {counts['observed']} + 新增 {counts['added']} 点"),
            ]
            staged = Path(temp) / item["final"].name
            encode(staged, frames, labels, item["display"], item["name"])
            probe(staged, SIZE)
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-threads", "1",
                            "-i", str(staged), "-map", "0:v:0", "-f", "null", "-"], check=True)
            summaries.append(dict(case=item["name"], output=str(item["final"]),
                                  dimensions=list(SIZE), frames=FRAMES, fps=FPS,
                                  point_counts=counts,
                                  sha256=digest(staged), old_comparison_sha256=digest(item["old"]),
                                  source_paths=[str(p) for p in item["sources"]],
                                  fixed_source_crop=f"{item['old']} [{item['crop']}]"))
            del frames
            print(f"VALIDATED {item['name']}: {SIZE[0]}x{SIZE[1]}, 81 frames, 16 FPS", flush=True)
        # Remove old delivery videos only after all three new files pass full decode.
        for item in records:
            (Path(temp) / item["final"].name).replace(item["final"])
        for item in records:
            if item["old"] != item["final"]:
                item["old"].unlink()
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--annotate-counts", action="store_true",
                        help="Replace the existing 1136px five-panel videos with point-count labels")
    arguments = parser.parse_args()
    reformat(arguments.delivery.resolve(), arguments.annotate_counts)
