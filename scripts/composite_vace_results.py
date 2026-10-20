"""CPU-only hard-mask compositing of completed Pilot8 predictions."""

import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

import buildingfiller_pilot8 as delivery
from reorganize_vace_results import comparison_profile, decode_video, player_compatible, require_cpu_only, validate_four
from vace_p01 import ROOT, adapt_images, require, sha256
from vace_p01_results import HEADER_HEIGHT, SIZE, comparison_panel, ffmpeg_binary, save_png_exclusive


INPUT_ROOT = ROOT / "outputs/vace/test"
OUTPUT_ROOT = ROOT / "outputs/vace/composite_check"
FINAL_NAMES = ("01_仅缺口补全.mp4", "02_四方视频对比.mp4", "03_关键帧对比.png")
LABELS = ("Before", "Raw Prediction", "Composite", "GT")


def read_sample(mapping):
    number, pilot, source_id, kind, tier = mapping
    prefix = delivery.PACKAGE_ID + "/"
    frames = {role: [] for role in ("damaged", "mask", "gt")}
    with zipfile.ZipFile(delivery.DEFAULT_ARCHIVE) as bundle:
        manifest = json.loads(bundle.read(prefix + "manifest.json"))
        matches = [sample for sample in manifest["samples"] if sample["pilot"] == pilot]
        require(len(matches) == 1, f"{pilot}: missing or duplicate sample identity")
        sample = matches[0]
        require((sample["source_id"], sample["kind"], sample["tier"]) == (source_id, kind, tier),
                f"{pilot}: source/type mapping mismatch")
        require(sample["relative_root"] == f"samples/{pilot}", f"{pilot}: unexpected source root")
        root = prefix + sample["relative_root"] + "/"
        metadata = json.loads(bundle.read(root + "sample_meta.json"))
        cameras = json.loads(bundle.read(root + "cameras.json"))
        require(metadata["sample_id"] == cameras["sample_id"] == sample["sample_id"]
                and metadata["source"]["sha256"] == sample["source_sha256"]
                and metadata["models"]["gt"]["sha256"] == sample["source_sha256"]
                and sample["stable_id"] == f"b{source_id}_" + sample["source_sha256"][:8],
                f"{pilot}: source metadata mismatch")
        poses = cameras["resolutions"]["512"]
        require(cameras["frames"] == len(poses) == 81 and cameras["fps_metadata"] == 16,
                f"{pilot}: invalid camera/frame count")
        names = bundle.namelist()
        for role in ("damaged", "exact", "gt"):
            expected = {root + f"512/{role}/{index:03}.png" for index in range(81)}
            actual = [name for name in names if name.startswith(root + f"512/{role}/")]
            require(len(actual) == 81 and set(actual) == expected, f"{pilot}: incomplete {role} frames")
        for index, pose in enumerate(poses):
            require(pose["frame"] == index and pose["frame_id"] == f"{index:03}"
                    and (pose["width"], pose["height"]) == SIZE, f"{pilot}: frame/camera alignment mismatch")
            images = {}
            for role in ("damaged", "exact", "gt"):
                with Image.open(io.BytesIO(bundle.read(root + f"512/{role}/{index:03}.png"))) as image:
                    image.load()
                    require(image.format == "PNG" and image.size == SIZE
                            and image.mode == ("L" if role == "exact" else "RGB"),
                            f"{pilot} B{index:03}: invalid {role} PNG")
                    images[role] = image.copy()
            model_mask, _ = adapt_images(images["damaged"], images["exact"])
            frames["damaged"].append(np.array(images["damaged"]))
            frames["mask"].append(np.array(model_mask))
            frames["gt"].append(np.array(images["gt"]))
    return frames


def hard_composite(prediction, damaged, mask):
    require(prediction.shape == damaged.shape == mask.shape == (512, 512, 3),
            "Composite inputs must all be 512x512 RGB")
    require(all(pixels.dtype == np.uint8 for pixels in (prediction, damaged, mask)),
            "Composite inputs must be uint8")
    require(np.isin(mask, [0, 255]).all()
            and np.array_equal(mask[:, :, 0], mask[:, :, 1])
            and np.array_equal(mask[:, :, 1], mask[:, :, 2]), "ModelMask must be binary RGB 0/255")
    return np.where(mask == 255, prediction, damaged)


def select_views(frames, prediction):
    areas = np.array([np.count_nonzero(mask[:, :, 0]) for mask in frames["mask"]])
    selected = {0, 80}
    if areas.max() > 0:
        largest = int(areas.argmax())
        candidates = np.flatnonzero(areas >= areas.max() * 0.25)
        errors = {}
        for index in candidates:
            white = frames["mask"][index][:, :, 0] == 255
            errors[int(index)] = np.abs(prediction[index][white].astype(np.int16)
                                        - frames["gt"][index][white].astype(np.int16)).mean()
        selected.update((largest, max(errors, key=errors.get)))
    black = np.flatnonzero(areas == 0)
    if len(black) and not any(areas[index] == 0 for index in selected):
        selected.add(int(black[len(black) // 2]))
    for index in (20, 40, 60, 10, 30, 50, 70):
        if len(selected) == 5:
            break
        selected.add(index)
    return sorted(selected)


def four_columns(groups, index):
    return comparison_panel(tuple((f"{label} B{index:03}", images[index])
                                  for label, images in zip(LABELS, groups)))


def overview(groups, views):
    canvas = Image.new("RGB", (5 * SIZE[0], 4 * (SIZE[1] + HEADER_HEIGHT)), "white")
    for row, (label, images) in enumerate(zip(LABELS, groups)):
        panel = comparison_panel(tuple((f"{label} B{index:03}", images[index]) for index in views))
        canvas.paste(panel, (0, row * (SIZE[1] + HEADER_HEIGHT)))
    return canvas


def encode_video(path, images, size, notify):
    command = [ffmpeg_binary(), "-nostdin", "-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{size[0]}x{size[1]}", "-r", "16", "-i", "pipe:0", "-an",
               "-vf", "scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p",
               "-c:v", "libx264", "-profile:v", "high", "-level:v", "4.0",
               "-crf", "12", "-preset", "fast", "-threads", "2", "-pix_fmt", "yuv420p",
               "-color_range", "tv", "-colorspace", "bt709", "-color_trc", "bt709", "-color_primaries", "bt709",
               "-movflags", "+faststart", str(path)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        count = 0
        for count, image in enumerate(images, 1):
            pixels = np.asarray(image)
            require(pixels.shape == (size[1], size[0], 3) and pixels.dtype == np.uint8,
                    "Encoder requires original-size RGB uint8 frames")
            process.stdin.write(pixels.tobytes())
            if count in (1, 21, 41, 61, 81):
                notify(f"Encoded input frames: {count}/81")
        require(count == 81, f"Expected 81 frames, got {count}")
        process.stdin.close()
        process.stdin = None
        _, errors = process.communicate(timeout=180)
        require(process.returncode == 0, f"CPU encoder failed: {errors.decode(errors='replace')}")
    except BrokenPipeError as error:
        process.stdin.close()
        process.stdin = None
        _, errors = process.communicate(timeout=30)
        raise RuntimeError(f"CPU encoder failed: {errors.decode(errors='replace')}") from error
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if process.stdin is not None:
            process.stdin.close()
        process.stderr.close()


def validate_encoded(path, size, expected_frames):
    require(player_compatible(comparison_profile(path)), "Expected player-compatible H.264/yuv420p/BT.709")
    decoded = decode_video(path, size)
    for index, expected in enumerate(expected_frames):
        error = np.abs(decoded[index].astype(np.int16) - np.asarray(expected).astype(np.int16))
        require(error.mean(axis=(0, 1)).max() <= 3, f"Color or frame alignment error at B{index:03}")


def process_sample(mapping, output_root):
    number, pilot, source_id, _, _ = mapping
    started = time.monotonic()

    def notify(message):
        print(f"[{number}/08] {pilot} source{source_id}: {message} ({time.monotonic() - started:.1f}s)",
              flush=True)

    notify("Reading existing prediction, Damaged, Exact and GT")
    raw_path = INPUT_ROOT / number / "01_补全结果.mp4"
    raw_hash = sha256(raw_path)
    prediction = decode_video(raw_path, SIZE)
    frames = read_sample(mapping)
    validate_four(INPUT_ROOT / number / "04_缺损_Mask_预测_GT_四栏验证.png", frames, prediction)
    composite = [hard_composite(prediction[index], frames["damaged"][index], frames["mask"][index])
                 for index in range(81)]
    for index, pixels in enumerate(composite):
        white = frames["mask"][index][:, :, 0] == 255
        require(np.array_equal(pixels[white], prediction[index][white])
                and np.array_equal(pixels[~white], frames["damaged"][index][~white]),
                f"{pilot}: hard composition changed source pixels")
    black = [index for index, mask in enumerate(frames["mask"]) if not mask.any()]
    views = select_views(frames, prediction)
    notify(f"PASS: 81 aligned frames; black masks={black}; selected views={views}")
    notify("Selection: B000/B080, largest visible mask, worst masked RGB error, black-mask view if available")
    groups = (frames["damaged"], prediction, composite, frames["gt"])
    with tempfile.TemporaryDirectory(prefix=".vace_composite_", dir=output_root.parent) as temporary:
        staged = Path(temporary) / number
        staged.mkdir()
        notify("Creating composite video")
        encode_video(staged / FINAL_NAMES[0], composite, SIZE, notify)
        notify("Creating Before | Raw Prediction | Composite | GT video")
        encode_video(staged / FINAL_NAMES[1], (four_columns(groups, index) for index in range(81)),
                     (2048, 544), notify)
        grid = overview(groups, views)
        save_png_exclusive(staged / FINAL_NAMES[2], grid)
        notify("Validating both 81-frame videos and the 4-row overview")
        validate_encoded(staged / FINAL_NAMES[0], SIZE, composite)
        validate_encoded(staged / FINAL_NAMES[1], (2048, 544),
                         (four_columns(groups, index) for index in range(81)))
        with Image.open(staged / FINAL_NAMES[2]) as saved:
            saved.load()
            require(saved.format == "PNG" and np.array_equal(np.asarray(saved), np.asarray(grid)),
                    "Overview PNG differs from the aligned RGB panels")
        require(sha256(raw_path) == raw_hash, f"{pilot}: original prediction changed")
        require({path.name for path in staged.iterdir()} == set(FINAL_NAMES), "Expected exactly three final files")
        require(not (output_root / number).exists(), "Refuse to overwrite existing sample results")
        staged.rename(output_root / number)
    notify(f"SUCCESS: 3 files; original prediction unchanged; completed output: {output_root / number}")


def new_output_root():
    output = OUTPUT_ROOT
    if output.exists():
        output = output.with_name(f"{output.name}_{datetime.now():%Y%m%d_%H%M%S_%f}_{os.getpid()}")
    output.mkdir()
    return output


def stop_on_signal(signum, frame):
    raise KeyboardInterrupt(f"Interrupted by signal {signum}")


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    require_cpu_only()
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, stop_on_signal)
    started = time.monotonic()
    output = new_output_root()
    print(f"CPU HARD-MASK COMPOSITE\nOUTPUT: {output}\nGPU INFERENCE RUNS: 0", flush=True)
    statuses = {}
    for mapping in delivery.SAMPLE_MAPPING:
        pilot = mapping[1]
        try:
            process_sample(mapping, output)
            statuses[pilot] = "SUCCESS"
        except Exception as error:
            statuses[pilot] = f"FAILED / SKIPPED: {type(error).__name__}: {error}"
            print(f"{pilot}: {statuses[pilot]}", flush=True)
    require_cpu_only()
    print("\nVACE COMPOSITE CHECK", flush=True)
    for pilot, status in statuses.items():
        print(f"{pilot}: {status}", flush=True)
    complete = all(status == "SUCCESS" for status in statuses.values())
    print(f"\nCompleted: {sum(status == 'SUCCESS' for status in statuses.values())}/8"
          f"\nGPU INFERENCE RUNS: 0\nElapsed: {time.monotonic() - started:.1f}s\nOUTPUT: {output}", flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt as error:
        print(f"\nFAILED: {error}; exit code 130", flush=True)
        sys.exit(130)
