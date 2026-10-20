"""CPU visualization of real, matched 17-frame VACE LoRA predictions."""

import json
import os
import shutil
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image

from vace_p01_results import HEADER_HEIGHT, SIZE, comparison_panel, ffmpeg_binary, require, save_png_exclusive


FRAME_COUNT = 17
FPS = 16
VIEWS = (0, 4, 8, 12, 16)
FINAL_NAMES = (
    "01_微调后补全.mp4",
    "02_缺损_原模型_微调后_GT_对比.mp4",
    "03_训练进展对比.png",
    "04_Mask验证.png",
)


def rgb_sequence(images, role):
    require(len(images) == FRAME_COUNT, f"{role}: expected 17 frames")
    arrays = []
    for index, image in enumerate(images):
        if isinstance(image, Image.Image):
            require(image.mode == "RGB", f"{role} local {index}: expected RGB, not {image.mode}")
        pixels = np.asarray(image)
        require(pixels.shape == (SIZE[1], SIZE[0], 3) and pixels.dtype == np.uint8,
                f"{role} local {index}: expected 512x512 RGB uint8")
        arrays.append(pixels)
    return arrays


def validate_sources(source_indices, groups):
    require(len(source_indices) == FRAME_COUNT
            and all(isinstance(index, int) and not isinstance(index, bool) for index in source_indices),
            "Expected 17 integer source frame indices")
    require(0 <= source_indices[0] <= 63
            and list(source_indices) == list(range(source_indices[0], source_indices[0] + FRAME_COUNT)),
            "Source frames must be a consecutive 17-frame window within B000-B079")
    arrays = {role: rgb_sequence(images, role) for role, images in groups.items()}
    for index, mask in enumerate(arrays["mask"]):
        require(np.isin(mask, [0, 255]).all()
                and np.array_equal(mask[:, :, 0], mask[:, :, 1])
                and np.array_equal(mask[:, :, 1], mask[:, :, 2]),
                f"ModelMask local {index}: expected identical binary RGB channels")
    return arrays


def make_panel(groups, source_indices, index, roles):
    return comparison_panel(tuple((f"{label} B{source_indices[index]:03}", groups[role][index])
                                  for label, role in roles))


def make_overview(groups, source_indices):
    roles = (("Before", "damaged"), ("Base", "base"), ("LoRA-100", "lora100"),
             ("LoRA-300", "lora300"), ("GT", "gt"))
    canvas = Image.new("RGB", (5 * SIZE[0], 5 * (SIZE[1] + HEADER_HEIGHT)), "white")
    for row, (label, role) in enumerate(roles):
        panel = comparison_panel(tuple((f"{label} B{source_indices[index]:03}", groups[role][index])
                                       for index in VIEWS))
        canvas.paste(panel, (0, row * (SIZE[1] + HEADER_HEIGHT)))
    return canvas


def encode_video(path, images, size):
    command = [
        ffmpeg_binary(), "-nostdin", "-v", "error", "-n",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}", "-r", str(FPS),
        "-i", "pipe:0", "-an",
        "-vf", "scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p",
        "-c:v", "libx264", "-profile:v", "high", "-level:v", "4.0",
        "-crf", "12", "-preset", "fast", "-threads", "2", "-pix_fmt", "yuv420p",
        "-color_range", "tv", "-colorspace", "bt709", "-color_trc", "bt709",
        "-color_primaries", "bt709", "-movflags", "+faststart", str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        count = 0
        for count, image in enumerate(images, 1):
            pixels = np.asarray(image)
            require(pixels.shape == (size[1], size[0], 3) and pixels.dtype == np.uint8,
                    "Video encoder expects original-size RGB uint8 frames")
            process.stdin.write(pixels.tobytes())
        require(count == FRAME_COUNT, f"Video encoder received {count}, not 17 frames")
        process.stdin.close()
        process.stdin = None
        _, errors = process.communicate(timeout=180)
        require(process.returncode == 0, f"CPU video encoding failed: {errors.decode(errors='replace')}")
    except BrokenPipeError as error:
        if process.stdin is not None:
            process.stdin.close()
            process.stdin = None
        _, errors = process.communicate(timeout=30)
        raise RuntimeError(f"CPU video encoding failed: {errors.decode(errors='replace')}") from error
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if process.stdin is not None:
            process.stdin.close()
        process.stderr.close()


def validate_encoded(path, size, expected_frames):
    probe = shutil.which("ffprobe")
    require(probe is not None, "Existing ffprobe is required; no dependency installation attempted")
    inspected = subprocess.run(
        [probe, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,avg_frame_rate,r_frame_rate,codec_name,profile,pix_fmt,color_range,"
         "color_space,color_transfer,color_primaries", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=120,
    )
    require(inspected.returncode == 0, f"Video inspection failed: {inspected.stderr.strip()}")
    streams = json.loads(inspected.stdout).get("streams", [])
    require(len(streams) == 1, "Expected one video stream")
    stream = streams[0]
    require((stream["width"], stream["height"]) == size, "Video resolution mismatch")
    require(Fraction(stream["avg_frame_rate"]) == Fraction(stream["r_frame_rate"]) == FPS,
            "Video FPS must be exactly 16")
    expected_metadata = dict(codec_name="h264", profile="High", pix_fmt="yuv420p", color_range="tv",
                             color_space="bt709", color_transfer="bt709", color_primaries="bt709")
    require(all(stream.get(key) == value for key, value in expected_metadata.items()),
            "Video must use player-compatible H.264 High/yuv420p/BT.709")
    decoded = subprocess.run(
        [ffmpeg_binary(), "-nostdin", "-v", "error", "-xerror", "-err_detect", "explode", "-threads", "1",
         "-i", str(path), "-map", "0:v:0", "-vsync", "0", "-threads", "1",
         "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"],
        capture_output=True, timeout=120,
    )
    require(decoded.returncode == 0, f"Video decoding failed: {decoded.stderr.decode(errors='replace')}")
    require(len(decoded.stdout) == FRAME_COUNT * size[0] * size[1] * 3,
            "Video must decode into exactly 17 complete RGB frames")
    frames = np.frombuffer(decoded.stdout, dtype=np.uint8).reshape(FRAME_COUNT, size[1], size[0], 3)
    expected_count = 0
    for expected_count, expected in enumerate(expected_frames, 1):
        require(expected_count <= FRAME_COUNT, "Too many reference frames")
        difference = np.abs(frames[expected_count - 1].astype(np.int16) - np.asarray(expected).astype(np.int16))
        require(float(difference.mean(axis=(0, 1)).max()) <= 4,
                f"Decoded frame {expected_count - 1}: unexpected RGB or frame alignment change")
    require(expected_count == FRAME_COUNT, "Expected 17 reference frames")


def pixel_summary(groups):
    areas = [int(np.count_nonzero(mask[:, :, 0])) for mask in groups["mask"]]
    masked = {}
    unmasked = {}
    for role in ("base", "lora100", "lora300"):
        inside_sum = outside_sum = 0
        for index, mask in enumerate(groups["mask"]):
            white = mask[:, :, 0] == 255
            prediction = groups[role][index].astype(np.int16)
            inside_sum += int(np.abs(prediction[white] - groups["gt"][index][white].astype(np.int16)).sum())
            outside_sum += int(np.abs(prediction[~white] - groups["damaged"][index][~white].astype(np.int16)).sum())
        inside_count = sum(areas) * 3
        outside_count = (FRAME_COUNT * SIZE[0] * SIZE[1] - sum(areas)) * 3
        masked[role] = inside_sum / inside_count if inside_count else None
        unmasked[role] = outside_sum / outside_count if outside_count else None
    return dict(masked_mae_0_255=masked, unmasked_change_mae_0_255=unmasked,
                assessment="requires_visual_review", note="RGB errors do not prove roof closure or 3D consistency")


def render_sample(destination, source_indices, damaged, masks, gt, base, lora100, lora300, notify=print):
    destination = Path(destination)
    require(not os.path.lexists(destination), f"Refuse to overwrite existing results: {destination}")
    require(destination.parent.is_dir(), f"Output parent must already exist: {destination.parent}")
    groups = validate_sources(source_indices, dict(damaged=damaged, mask=masks, gt=gt,
                                                   base=base, lora100=lora100, lora300=lora300))
    mask_index = int(np.argmax([np.count_nonzero(mask[:, :, 0]) for mask in groups["mask"]]))
    roles = (("Before", "damaged"), ("Base", "base"), ("LoRA-300", "lora300"), ("GT", "gt"))
    summary = pixel_summary(groups)
    with tempfile.TemporaryDirectory(prefix=".vace_roof_media_", dir=destination.parent) as temporary:
        staged = Path(temporary) / destination.name
        staged.mkdir()
        notify(f"CPU media: encoding LoRA-300 raw 17 frames for {destination.name}")
        encode_video(staged / FINAL_NAMES[0], groups["lora300"], SIZE)
        notify("CPU media: encoding Before | Base | LoRA-300 | GT, 17 aligned frames")
        encode_video(staged / FINAL_NAMES[1],
                     (make_panel(groups, source_indices, index, roles) for index in range(FRAME_COUNT)), (2048, 544))
        overview = make_overview(groups, source_indices)
        verification = make_panel(groups, source_indices, mask_index,
                                  (("Before", "damaged"), ("ModelMask", "mask"), ("Base", "base"),
                                   ("LoRA-300", "lora300"), ("GT", "gt")))
        save_png_exclusive(staged / FINAL_NAMES[2], overview)
        save_png_exclusive(staged / FINAL_NAMES[3], verification)
        notify("CPU media: checking complete decoding, FPS, RGB, frame alignment and four final files")
        validate_encoded(staged / FINAL_NAMES[0], SIZE, groups["lora300"])
        validate_encoded(staged / FINAL_NAMES[1], (2048, 544),
                         (make_panel(groups, source_indices, index, roles) for index in range(FRAME_COUNT)))
        for filename, expected in ((FINAL_NAMES[2], overview), (FINAL_NAMES[3], verification)):
            with Image.open(staged / filename) as saved:
                saved.load()
                require(saved.format == "PNG" and saved.mode == "RGB"
                        and np.array_equal(np.asarray(saved), np.asarray(expected)), f"PNG mismatch: {filename}")
        require({path.name for path in staged.iterdir()} == set(FINAL_NAMES), "Expected exactly four final files")
        require(not os.path.lexists(destination), f"Refuse to overwrite existing results: {destination}")
        destination.mkdir()
        try:
            for filename in FINAL_NAMES:
                (staged / filename).rename(destination / filename)
        except BaseException:
            for filename in FINAL_NAMES:
                published = destination / filename
                if published.exists():
                    published.rename(staged / filename)
            destination.rmdir()
            raise
    notify(f"CPU media: PASS; mask validation uses B{source_indices[mask_index]:03}; {destination.resolve()}")
    return dict(summary, source_indices=list(source_indices), mask_frame=source_indices[mask_index],
                output=str(destination.resolve()), frames=FRAME_COUNT)
