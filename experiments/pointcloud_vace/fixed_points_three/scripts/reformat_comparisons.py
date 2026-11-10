#!/usr/bin/env python3
"""Re-layout completed comparisons on CPU; never regenerate model or point outputs."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[4]
CASES = ROOT / "data/pointcloud_vace/微调任务/cases"
D_RUN = ROOT / "outputs/pointcloud_vace/adapter_d_eval/run_20261110_150749"
SAMPLES = (("01", "檐口", "eave_3230"), ("04", "长墙", "strip_5016"),
           ("07", "墙面", "wall_10416"))
W, H, FRAMES, FPS = 832, 480, 81, 16
TITLE, HEADER, GAP = 64, 48, 16
SIZE = (3 * W, TITLE + 2 * (H + HEADER) + GAP)
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
    for (x, y), label, source in zip(positions, labels, frames):
        draw.rectangle((x, y, x + W - 1, y + HEADER - 1), fill="white")
        draw.text((x + 16, y + 4), label, font=font, fill=(25, 36, 52))
        canvas.paste(Image.fromarray(source[frame]), (x, y + HEADER))
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


def reformat(delivery):
    # Validate all sources before making any output. The measured reference is
    # the original cloud, NOT a mesh render. The extra distinct view is P-only.
    records = []
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
        old = delivery / f"{display}_{title}_四栏对比.mp4"
        final = delivery / f"{display}_{title}_五栏对比.mp4"
        if not old.is_file() or final.exists():
            raise FileExistsError(f"Need old comparison and an unused new name: {old}, {final}")
        sources = [case / "reference/observed_target.mp4", case / "reference/damaged.mp4",
                   case / "inputs/vace_video.mp4", D_RUN / "cases" / name / "raw_vace.mp4"]
        probe(old, (4 * W, H + 32))
        for source in sources:
            probe(source, (W, H))
        records.append(dict(display=display, title=title, name=name, old=old,
                            final=final, sources=sources, record=record))
    summaries = []
    with tempfile.TemporaryDirectory(prefix=".five_panel_", dir=delivery) as temp:
        for item in records:
            record = item["record"]
            frames = [decode(path) for path in item["sources"]]
            # Fixed 3D frames survive in the old comparison's third column.
            # Cropping them preserves the existing prediction; no point refit.
            frames.append(decode(item["old"], crop=f"crop={W}:{H}:{2 * W}:32"))
            labels = [f"Origin 原始点云 · {record['cloud_size']} 点",
                      f"残缺点云（未填灰）· {record['cloud_size'] - len(record['hidden_original_point_ids'])} 点",
                      "VACE 输入（缺区填灰）", "D-3000 输出", "固定三维补点"]
            staged = Path(temp) / item["final"].name
            encode(staged, frames, labels, item["display"], item["name"])
            probe(staged, SIZE)
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-threads", "1",
                            "-i", str(staged), "-map", "0:v:0", "-f", "null", "-"], check=True)
            summaries.append(dict(case=item["name"], output=str(item["final"]),
                                  dimensions=list(SIZE), frames=FRAMES, fps=FPS,
                                  sha256=digest(staged), old_comparison_sha256=digest(item["old"]),
                                  source_paths=[str(p) for p in item["sources"]],
                                  fixed_source_crop=f"{item['old']} [x={2 * W}, y=32, {W}x{H}]"))
            del frames
            print(f"VALIDATED {item['name']}: {SIZE[0]}x{SIZE[1]}, 81 frames, 16 FPS", flush=True)
        # Remove old delivery videos only after all three new files pass full decode.
        for item in records:
            (Path(temp) / item["final"].name).replace(item["final"])
        for item in records:
            item["old"].unlink()
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", type=Path, required=True)
    reformat(parser.parse_args().delivery.resolve())
