"""CPU-only, post-inference validation and evaluation of the frozen P01 VACE run."""

import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


FROZEN_SHA256 = "7e82161e270d8a24e231feb2ca95d08ae6e02e861ce5460a241730bfbcc07080"
SOURCE_FRAMES = tuple(range(5, 22))
SIZE = (512, 512)
FPS = 16
HEADER_HEIGHT = 32
COMPARISON_NAMES = (
    "comparison_before_after_B020.png",
    "comparison_before_after_grid.png",
    "comparison_4col_B020.png",
    "comparison_before_after.mp4",
    "metrics.json",
    "results_validation.json",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 ** 2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def ffmpeg_binary():
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    if importlib.util.find_spec("imageio_ffmpeg") is not None:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    raise RuntimeError("No existing ffmpeg decoder; no packages will be installed")


def validate_video(path, expected_size=SIZE, expected_frames=len(SOURCE_FRAMES)):
    path = Path(path)
    require(path.is_file() and path.stat().st_size > 0, f"Missing or empty prediction video: {path}")
    probe = shutil.which("ffprobe")
    require(probe is not None, "Existing ffprobe is required to verify exact video FPS; no installation attempted")
    inspected = subprocess.run(
        [probe, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,avg_frame_rate,r_frame_rate,codec_name", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=120,
    )
    require(inspected.returncode == 0, f"Video metadata inspection failed: {inspected.stderr.strip()}")
    streams = json.loads(inspected.stdout).get("streams", [])
    require(len(streams) == 1, "Expected exactly one selected video stream")
    stream = streams[0]
    require((stream["width"], stream["height"]) == tuple(expected_size), "Video resolution mismatch")
    try:
        average_fps = Fraction(stream["avg_frame_rate"])
        nominal_fps = Fraction(stream["r_frame_rate"])
    except (KeyError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("Video FPS cannot be verified") from exc
    require(average_fps == nominal_fps == FPS, "Video FPS must be exactly 16")
    decoded = subprocess.run(
        [ffmpeg_binary(), "-nostdin", "-v", "error", "-xerror", "-err_detect", "explode", "-threads", "1",
         "-i", str(path), "-map", "0:v:0", "-vsync", "0", "-threads", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"],
        capture_output=True, timeout=120,
    )
    require(decoded.returncode == 0,
            f"Video decoding failed: {decoded.stderr.decode(errors='replace').strip()}")
    bytes_per_frame = expected_size[0] * expected_size[1] * 3
    require(len(decoded.stdout) % bytes_per_frame == 0, "Video decoder produced an incomplete frame")
    frame_count = len(decoded.stdout) // bytes_per_frame
    require(frame_count == expected_frames, f"Decoded video must have {expected_frames} frames, got {frame_count}")
    return dict(path=str(path.resolve()), frames=frame_count, width=expected_size[0], height=expected_size[1],
                fps=float(average_fps), fps_fraction=str(average_fps), codec=stream.get("codec_name"),
                decode_complete=True, sha256=sha256(path))


def read_rgb(path):
    with Image.open(path) as image:
        image.load()
        require(image.format == "PNG" and image.mode == "RGB" and image.size == SIZE,
                f"Expected an original-size 512x512 RGB PNG: {path}")
        return np.array(image, copy=True)


def validate_inputs(output):
    config_path = output / "frozen_config.json"
    require(sha256(config_path) == FROZEN_SHA256, "Frozen run configuration changed")
    config = read_json(config_path)
    require(config["inference"]["fps"] == FPS and config["inference"]["num_frames"] == 17,
            "Unexpected frozen frame count or FPS")
    complete = read_json(output / "worker_complete.json")
    require(complete.get("status") == "complete" and complete.get("prediction_frames") == 17,
            "No complete real inference worker record")
    require(complete.get("lora_loaded") is False and complete.get("GT_loaded_by_worker") is False,
            "Inference worker must exclude LoRA and GT")
    frame_map = read_json(output / "frame_map.json")
    require(len(frame_map) == 17, "Frame map must contain 17 records")
    inventory = read_json(output / "artifacts_sha256.json")
    prediction_hashes = read_json(output / "prediction_raw_sha256.json")
    filenames = [f"{index:03}.png" for index in range(17)]
    require(sorted(path.name for path in (output / "prediction_raw").iterdir()) == filenames,
            "Raw prediction directory must contain exactly 17 indexed PNGs")
    require(sorted(prediction_hashes) == filenames, "Raw prediction hash inventory mismatch")
    roles = dict(damaged="inputs/damaged", mask="inputs/model_mask", condition="inputs/conditioning",
                 gt="evaluation/gt", prediction="prediction_raw")
    frames = []
    input_hashes = {}
    for output_index, source_index in enumerate(SOURCE_FRAMES):
        mapping = frame_map[output_index]
        require(mapping["output_index"] == output_index and mapping["source_index"] == source_index,
                "Frame map must independently align output 000-016 with source B005-B021")
        require(mapping["source_frame"]["frame_index"] == mapping["source_frame"]["camera_index"]
                == mapping["camera"]["frame"] == source_index, "Source camera/frame index mismatch")
        require(mapping["source_frame"]["frame_id"] == mapping["camera"]["frame_id"] == f"{source_index:03}",
                "Source camera/frame ID mismatch")
        pixels = {}
        for role, directory in roles.items():
            relative = f"{directory}/{output_index:03}.png"
            path = output / relative
            require(path.resolve().is_relative_to(output), f"Artifact escapes this run: {relative}")
            actual_hash = sha256(path)
            expected_hash = prediction_hashes[path.name] if role == "prediction" else inventory.get(relative)
            require(expected_hash is not None and actual_hash == expected_hash, f"Artifact SHA256 mismatch: {relative}")
            if role in ("damaged", "gt"):
                require(actual_hash == mapping["original_sha256"][role], f"Original source hash mismatch: {relative}")
            input_hashes[relative] = actual_hash
            pixels[role] = read_rgb(path)
        mask = pixels["mask"]
        require(np.isin(mask, [0, 255]).all() and np.array_equal(mask[:, :, 0], mask[:, :, 1])
                and np.array_equal(mask[:, :, 1], mask[:, :, 2]), "ModelMask must have identical binary RGB channels")
        white = mask[:, :, 0] == 255
        require(np.all(pixels["condition"][white] == 127), "Conditioning white-mask pixels must be RGB127")
        require(np.array_equal(pixels["condition"][~white], pixels["damaged"][~white]),
                "Conditioning changed unmasked source pixels")
        frames.append(dict(output_index=output_index, source_index=source_index, **pixels))
    return frames, input_hashes


def error_summary(prediction, reference, region):
    count = int(region.sum())
    if count == 0:
        return dict(pixels=0, mae_0_255=None, mae_normalized=None, psnr_db=None, perfect_match=None)
    difference = prediction.astype(np.float64)[region] - reference.astype(np.float64)[region]
    mae = float(np.abs(difference).mean())
    mse = float(np.square(difference).mean())
    return dict(pixels=count, mae_0_255=mae, mae_normalized=mae / 255,
                psnr_db=10 * math.log10(255 ** 2 / mse) if mse > 0 else None, perfect_match=mse == 0)


def silhouette_iou(prediction, reference):
    predicted_foreground = prediction.astype(np.float64).mean(axis=-1) < 229.5
    reference_foreground = reference.astype(np.float64).mean(axis=-1) < 229.5
    union = int(np.logical_or(predicted_foreground, reference_foreground).sum())
    return float(np.logical_and(predicted_foreground, reference_foreground).sum() / union) if union else 1.0


def temporal_diagnostics(frames, role):
    arrays = [frame[role].astype(np.float64) / 255 for frame in frames]
    adjacent = [dict(source_frames=[f"B{frames[index - 1]['source_index']:03}", f"B{frames[index]['source_index']:03}"],
                     mean_absolute_delta=float(np.abs(arrays[index] - arrays[index - 1]).mean()))
                for index in range(1, len(arrays))]
    second = [dict(source_frames=[f"B{frames[position]['source_index']:03}" for position in range(index - 1, index + 2)],
                   mean_absolute_second_difference=float(np.abs(arrays[index + 1] - 2 * arrays[index] + arrays[index - 1]).mean()))
              for index in range(1, len(arrays) - 1)]
    return dict(mean_absolute_consecutive_delta=float(np.mean([item["mean_absolute_delta"] for item in adjacent])),
                mean_absolute_second_difference=float(np.mean([item["mean_absolute_second_difference"] for item in second])),
                consecutive=adjacent, second_difference=second)


def calculate_metrics(frames):
    per_frame = []
    sums = {region: {role: dict(pixels=0, absolute_sum=0.0, squared_sum=0.0)
                     for role in ("raw_prediction", "damaged_baseline", "raw_vs_damaged_change")}
            for region in ("inside_mask", "outside_mask")}
    for frame in frames:
        white = frame["mask"][:, :, 0] == 255
        item = dict(source_frame=f"B{frame['source_index']:03}", output_index=frame["output_index"],
                    silhouette_iou_raw_vs_gt=silhouette_iou(frame["prediction"], frame["gt"]),
                    silhouette_iou_damaged_vs_gt=silhouette_iou(frame["damaged"], frame["gt"]))
        for region_name, region in (("inside_mask", white), ("outside_mask", ~white)):
            comparisons = dict(raw_prediction=(frame["prediction"], frame["gt"]),
                               damaged_baseline=(frame["damaged"], frame["gt"]),
                               raw_vs_damaged_change=(frame["prediction"], frame["damaged"]))
            item[region_name] = {}
            for role, (prediction, reference) in comparisons.items():
                item[region_name][role] = error_summary(prediction, reference, region)
                accumulator = sums[region_name][role]
                difference = prediction.astype(np.float64)[region] - reference.astype(np.float64)[region]
                accumulator["pixels"] += int(region.sum())
                accumulator["absolute_sum"] += float(np.abs(difference).sum())
                accumulator["squared_sum"] += float(np.square(difference).sum())
        per_frame.append(item)
    aggregate = {}
    for region_name, roles in sums.items():
        aggregate[region_name] = {}
        for role, accumulated in roles.items():
            count = accumulated["pixels"]
            samples = count * 3
            mae = accumulated["absolute_sum"] / samples if samples else None
            mse = accumulated["squared_sum"] / samples if samples else None
            aggregate[region_name][role] = dict(pixels=count, mae_0_255=mae,
                mae_normalized=mae / 255 if mae is not None else None,
                psnr_db=10 * math.log10(255 ** 2 / mse) if mse is not None and mse > 0 else None,
                perfect_match=mse == 0 if mse is not None else None)
        baseline = aggregate[region_name]["damaged_baseline"]["mae_0_255"]
        prediction = aggregate[region_name]["raw_prediction"]["mae_0_255"]
        aggregate[region_name]["damaged_minus_raw_mae_0_255"] = baseline - prediction if baseline is not None else None
    return dict(schema="building_wan.vace_p01.post_inference_metrics.v1", oracle_mask=True,
                GT_usage="post_inference_evaluation_only", quality_assessment="requires_human_review",
                automatic_quality_pass=None, postprocessed_prediction_used=False,
                metric_notes=dict(mae="RGB pixel-channel mean; normalized MAE is divided by 255; aggregate is pixel-weighted",
                    empty_region="MAE/PSNR are null when a region has no pixels",
                    psnr="PSNR is null for a perfect match rather than JSON Infinity; inspect perfect_match",
                    silhouette="Diagnostic only: mean RGB < 229.5 (white-background foreground); empty union IoU=1",
                    temporal="Uncompensated image differences include real camera motion; not isolated flicker, optical flow, or 3D quality",
                    acceptance="No metric threshold automatically accepts or rejects completion quality"),
                aggregate=aggregate, per_frame=per_frame,
                temporal={role: temporal_diagnostics(frames, source_role) for role, source_role in
                          (("raw_prediction", "prediction"), ("damaged_baseline", "damaged"), ("gt_reference", "gt"))})


def label_font():
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 22)
    except OSError:
        return ImageFont.load_default()


def comparison_panel(columns):
    canvas = Image.new("RGB", (SIZE[0] * len(columns), SIZE[1] + HEADER_HEIGHT), "white")
    draw = ImageDraw.Draw(canvas)
    font = label_font()
    for column, (label, pixels) in enumerate(columns):
        offset = column * SIZE[0]
        canvas.paste(Image.fromarray(pixels), (offset, HEADER_HEIGHT))
        draw.text((offset + 10, 3), label, fill="black", font=font)
    return canvas


def save_png_exclusive(path, image):
    with Path(path).open("xb") as stream:
        image.save(stream, format="PNG")


def save_comparison_video(output, frames):
    if importlib.util.find_spec("imageio_ffmpeg") is None:
        return dict(status="not_created", reason="Existing imageio_ffmpeg is unavailable; no installation attempted")
    import imageio_ffmpeg

    path = output / "comparison_before_after.mp4"
    with path.open("xb"):
        pass
    writer = imageio_ffmpeg.write_frames(str(path), (1024, 544), fps=FPS, quality=8,
        codec="libx264", macro_block_size=16, ffmpeg_log_level="error", output_params=["-threads", "1"])
    try:
        writer.send(None)
        for frame in frames:
            panel = comparison_panel(((f"Before B{frame['source_index']:03}", frame["damaged"]),
                                      (f"After B{frame['source_index']:03} (raw)", frame["prediction"])))
            writer.send(np.asarray(panel).tobytes())
    finally:
        writer.close()
    return dict(status="created", **validate_video(path, expected_size=(1024, 544)))


def generate_results(output):
    output = Path(output).resolve(strict=True)
    require(output.is_dir(), "Expected an existing independent run directory")
    for filename in COMPARISON_NAMES:
        path = output / filename
        require(not path.exists() and not path.is_symlink(), f"Refuse to overwrite an existing result: {path}")
    frames, input_hashes = validate_inputs(output)
    raw_video = validate_video(output / "prediction_raw.mp4")
    metrics = calculate_metrics(frames)
    selected = {frame["source_index"]: frame for frame in frames}
    chosen = selected[20]
    before_after = comparison_panel((("Before B020", chosen["damaged"]), ("After B020 (VACE raw)", chosen["prediction"])))
    save_png_exclusive(output / COMPARISON_NAMES[0], before_after)
    representative = (5, 10, 15, 20)
    upper = comparison_panel(tuple((f"Before B{source_index:03}", selected[source_index]["damaged"])
                                   for source_index in representative))
    lower = comparison_panel(tuple((f"After B{source_index:03} (raw)", selected[source_index]["prediction"])
                                   for source_index in representative))
    grid = Image.new("RGB", (upper.width, upper.height + lower.height), "white")
    grid.paste(upper, (0, 0))
    grid.paste(lower, (0, upper.height))
    save_png_exclusive(output / COMPARISON_NAMES[1], grid)
    four_columns = comparison_panel((("Before B020", chosen["damaged"]), ("ModelMask B020", chosen["mask"]),
                                     ("Prediction B020 (raw)", chosen["prediction"]), ("Ground Truth B020", chosen["gt"])))
    save_png_exclusive(output / COMPARISON_NAMES[2], four_columns)
    comparison_video = save_comparison_video(output, frames)
    write_json(output / "metrics.json", metrics)
    filenames = [name for name in COMPARISON_NAMES if name != "results_validation.json" and (output / name).is_file()]
    artifacts = {name: dict(path=str((output / name).resolve()), sha256=sha256(output / name)) for name in filenames}
    validation = dict(schema="building_wan.vace_p01.results_validation.v1", status="validated",
        completed_at_utc=datetime.now(timezone.utc).isoformat(), source_frames=[f"B{index:03}" for index in SOURCE_FRAMES],
        output_to_source=[dict(output_index=index, source_frame=f"B{source_index:03}")
                          for index, source_index in enumerate(SOURCE_FRAMES)],
        source_frozen_config_sha256=FROZEN_SHA256, actual_frame_count=17, width=512, height=512, fps=FPS,
        image_resizing=False, image_cropping=False, image_header_height=HEADER_HEIGHT,
        displayed_prediction="unmodified raw model prediction", GT_usage="four-column display and independent post-inference evaluation only",
        raw_video=raw_video, comparison_video=comparison_video, input_files_sha256=input_hashes,
        artifacts=artifacts, quality_assessment="requires_human_review", automatic_quality_pass=None)
    write_json(output / "results_validation.json", validation)
    paths = dict(raw_prediction=str((output / "prediction_raw").resolve()),
                 raw_prediction_video=str((output / "prediction_raw.mp4").resolve()),
                 before_after_image=str((output / COMPARISON_NAMES[0]).resolve()),
                 multi_frame_comparison=str((output / COMPARISON_NAMES[1]).resolve()),
                 four_column_comparison=str((output / COMPARISON_NAMES[2]).resolve()),
                 metrics=str((output / "metrics.json").resolve()), validation=str((output / "results_validation.json").resolve()))
    if comparison_video.get("path"):
        paths["comparison_video"] = comparison_video["path"]
    return dict(status="validated", paths=paths, video_validation=raw_video,
                **paths, quality_assessment="requires_human_review")
