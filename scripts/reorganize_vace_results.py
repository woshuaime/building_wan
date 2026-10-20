"""Rearrange completed Pilot8 predictions using CPU-only, player-compatible comparisons."""

import argparse
import contextlib
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

import buildingfiller_pilot8 as delivery
import vace_p01 as baseline
from vace_building_reuse import validate_completed_sample
from vace_p01_81f_results import FINAL_FILENAMES as OLD_NAMES, REPRESENTATIVE_FRAMES
from vace_p01_results import HEADER_HEIGHT, SIZE, comparison_panel, ffmpeg_binary, save_png_exclusive, validate_video
from vace_p01_reuse import check_no_symlink_components


OUTPUT_ROOT = baseline.ROOT / "outputs/vace/test"
FINAL_NAMES = ("01_补全结果.mp4", "02_补前_补后_GT_三栏对比.mp4",
               "03_补前_补后_GT_三行总览.png", "04_缺损_Mask_预测_GT_四栏验证.png")
VIDEO_SIZE = (1536, 544)
OVERVIEW_SIZE = (2560, 1632)
TEMP_PREFIX = ".vace_relayout_"


class RecoveryRequired(RuntimeError):
    pass


def require_cpu_only():
    baseline.require(not any(name == "torch" or name.startswith("torch.") or name == "diffsynth"
                             or name.startswith("diffsynth.") for name in sys.modules),
                     "Offline rearrangement must not import Torch or DiffSynth")


def inventory(directory, names):
    check_no_symlink_components(directory)
    baseline.require(directory.is_dir() and {path.name for path in directory.iterdir()} == set(names),
                     f"Unexpected or incomplete four-file layout: {directory}")
    baseline.require(all((directory / name).is_file() and not (directory / name).is_symlink() for name in names),
                     "Every result must be a regular file")
    return {name: baseline.sha256(directory / name) for name in names}


def decode_video(path, size):
    validate_video(path, size, expected_frames=81)
    decoded = subprocess.run(
        [ffmpeg_binary(), "-nostdin", "-v", "error", "-xerror", "-err_detect", "explode",
         "-hwaccel", "none", "-threads", "1", "-i", str(path), "-map", "0:v:0", "-vsync", "0",
         "-threads", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"],
        capture_output=True, timeout=180,
    )
    baseline.require(decoded.returncode == 0, decoded.stderr.decode(errors="replace"))
    baseline.require(len(decoded.stdout) == 81 * size[0] * size[1] * 3, "Unexpected decoded frame count")
    return np.frombuffer(decoded.stdout, dtype=np.uint8).reshape(81, size[1], size[0], 3)


def comparison_profile(path):
    executable = shutil.which("ffprobe")
    baseline.require(executable is not None, "Existing ffprobe is required")
    result = subprocess.run(
        [executable, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,profile,pix_fmt,color_range,color_space,color_transfer,color_primaries",
         "-of", "json", str(path)], capture_output=True, text=True, timeout=30,
    )
    baseline.require(result.returncode == 0, f"Comparison profile inspection failed: {result.stderr}")
    streams = json.loads(result.stdout).get("streams", [])
    baseline.require(len(streams) == 1, "Expected one comparison video stream")
    return streams[0]


def player_compatible(profile):
    expected = dict(codec_name="h264", profile="High", pix_fmt="yuv420p", color_range="tv",
                    color_space="bt709", color_transfer="bt709", color_primaries="bt709")
    return all(profile.get(key) == value for key, value in expected.items())


def validate_comparison(path, frames, prediction, allow_legacy=False):
    profile = comparison_profile(path)
    compatible = player_compatible(profile)
    baseline.require(compatible or (allow_legacy and profile.get("pix_fmt") == "gbrp"
                                   and profile.get("codec_name") == "h264"),
                     f"Comparison must use player-compatible H.264/yuv420p/BT.709: {profile}")
    video = decode_video(path, VIDEO_SIZE)
    for index in range(81):
        expected = np.asarray(three_columns(frames, prediction, index))
        if not compatible:
            baseline.require(np.array_equal(video[index], expected),
                             f"Legacy comparison pixels or frame alignment changed at B{index:03}")
            continue
        for column in range(3):
            left = column * SIZE[0]
            actual_panel = video[index, :, left:left + SIZE[0]].astype(np.int16)
            expected_panel = expected[:, left:left + SIZE[0]].astype(np.int16)
            error = actual_panel - expected_panel
            absolute = np.abs(error)
            mean_error = absolute.mean(axis=(0, 1)).max()
            bias = np.abs(error.mean(axis=(0, 1))).max()
            tail_error = np.percentile(absolute, 99)
            baseline.require(mean_error <= 3 and bias <= 3 and tail_error <= 12,
                             f"Comparison color or frame alignment mismatch at B{index:03} column {column}: "
                             f"MAE={mean_error:.4f}, bias={bias:.4f}, p99={tail_error:.4f}")
    return compatible


def source_frames(record):
    selected = delivery.read_sample(record, include_gt=True)
    frames = {role: [] for role in ("damaged", "gt", "mask")}
    for index, item in enumerate(selected):
        baseline.require(item["source_index"] == index, "Source frame order changed")
        for role in ("damaged", "gt"):
            with Image.open(io.BytesIO(item["original_bytes"][role])) as image:
                frames[role].append(np.array(image))
        frames["mask"].append(np.asarray(item["model_mask"]))
    return frames


def three_columns(frames, prediction, index):
    return comparison_panel(((f"Before B{index:03}", frames["damaged"][index]),
                             (f"Prediction B{index:03}", prediction[index]),
                             (f"Ground Truth B{index:03}", frames["gt"][index])))


def three_rows(frames, prediction):
    canvas = Image.new("RGB", OVERVIEW_SIZE, "white")
    for row, (label, images) in enumerate((("Before", frames["damaged"]),
                                           ("Prediction", prediction), ("Ground Truth", frames["gt"]))):
        panel = comparison_panel(tuple((f"{label} B{index:03}", images[index]) for index in REPRESENTATIVE_FRAMES))
        canvas.paste(panel, (0, row * (SIZE[1] + HEADER_HEIGHT)))
    return canvas


def crop_pixels(image, column, row=0):
    left = column * SIZE[0]
    top = row * (SIZE[1] + HEADER_HEIGHT) + HEADER_HEIGHT
    return np.asarray(image)[top:top + SIZE[1], left:left + SIZE[0]]


def validate_four(path, frames, prediction):
    with Image.open(path) as image:
        image.load()
        baseline.require(image.format == "PNG" and image.mode == "RGB" and image.size == (2048, 544),
                         "Invalid four-column validation image")
        for column, role in ((0, "damaged"), (1, "mask"), (3, "gt")):
            baseline.require(np.array_equal(crop_pixels(image, column), frames[role][20]),
                             f"Four-column B020 {role} source mismatch")
        error = np.abs(crop_pixels(image, 2).astype(np.int16) - prediction[20].astype(np.int16)).mean()
        baseline.require(error <= 3, f"Four-column prediction differs from existing MP4 B020: MAE={error:.6f}")
    return float(error)


def validate_old_prediction_panels(directory, prediction):
    with Image.open(directory / OLD_NAMES[2]) as overview, Image.open(directory / OLD_NAMES[3]) as four:
        baseline.require(np.array_equal(crop_pixels(overview, 1, 1), crop_pixels(four, 2)),
                         "Old overview and four-column B020 predictions disagree")
        for column, index in enumerate(REPRESENTATIVE_FRAMES):
            error = np.abs(crop_pixels(overview, column, 1).astype(np.int16) - prediction[index].astype(np.int16)).mean()
            baseline.require(error <= 3, f"Existing prediction frame B{index:03} does not match its original overview")


def encode_comparison(path, frames, prediction, notify=print):
    baseline.require(not path.exists(), "Refuse to overwrite a staged video")
    command = [ffmpeg_binary(), "-nostdin", "-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", "1536x544", "-r", "16", "-i", "pipe:0", "-an",
               "-vf", "scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p",
               "-c:v", "libx264", "-profile:v", "high", "-level:v", "4.0",
               "-crf", "12", "-preset", "fast", "-threads", "2", "-pix_fmt", "yuv420p",
               "-color_range", "tv", "-colorspace", "bt709", "-color_trc", "bt709", "-color_primaries", "bt709",
               "-movflags", "+faststart", str(path)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        for index in range(81):
            process.stdin.write(np.asarray(three_columns(frames, prediction, index)).tobytes())
            if index in (0, 20, 40, 60, 80):
                notify(f"Frames composed: {index + 1}/81")
        process.stdin.close()
        process.stdin = None
        _, errors = process.communicate(timeout=180)
        baseline.require(process.returncode == 0, f"CPU video encoder failed: {errors.decode(errors='replace')}")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if process.stdin is not None:
            process.stdin.close()
        process.stderr.close()


def validate_arrangement(directory, frames, prediction, raw_hash, four_hash, raw_path=None, allow_legacy=False):
    raw_path = directory / FINAL_NAMES[0] if raw_path is None else raw_path
    baseline.require(baseline.sha256(raw_path) == raw_hash, "Original prediction SHA256 changed")
    compatible = validate_comparison(directory / FINAL_NAMES[1], frames, prediction, allow_legacy=allow_legacy)
    with Image.open(directory / FINAL_NAMES[2]) as overview:
        overview.load()
        baseline.require(overview.format == "PNG" and overview.mode == "RGB" and overview.size == OVERVIEW_SIZE,
                         "Invalid three-row overview")
        baseline.require(np.array_equal(np.asarray(overview), np.asarray(three_rows(frames, prediction))),
                         "Three-row overview pixels or representative frame mapping changed")
    baseline.require(baseline.sha256(directory / FINAL_NAMES[3]) == four_hash,
                     "Preserved four-column image content changed")
    validate_four(directory / FINAL_NAMES[3], frames, prediction)
    return compatible


def replace_comparison(directory, staged, backup, before, validate):
    baseline.require(inventory(directory, FINAL_NAMES) == before, "Results changed before comparison replacement")
    video = directory / FINAL_NAMES[1]
    protected = {name: (directory / name).stat() for name in (FINAL_NAMES[0], *FINAL_NAMES[2:])}
    previous = backup / FINAL_NAMES[1]
    backup.mkdir()
    moved = False
    published = False
    try:
        video.rename(previous)
        moved = True
        baseline.require(not video.exists(), "Unexpected comparison destination appeared")
        staged.rename(video)
        published = True
        validate()
        after = inventory(directory, FINAL_NAMES)
        for name, original in protected.items():
            current = (directory / name).stat()
            baseline.require(after[name] == before[name]
                             and (original.st_ino, original.st_size, original.st_mtime_ns)
                             == (current.st_ino, current.st_size, current.st_mtime_ns),
                             f"Protected result changed: {name}")
    except BaseException as error:
        try:
            if published:
                video.rename(staged)
            if moved:
                baseline.require(not video.exists(), "Cannot restore occupied comparison destination")
                previous.rename(video)
            baseline.require(inventory(directory, FINAL_NAMES) == before, "Restored comparison differs from original")
        except BaseException as recovery_error:
            raise RecoveryRequired(f"Comparison replacement failed: {error}; rollback failed: {recovery_error}; "
                                   f"preserved recovery files: {backup.parent}") from recovery_error
        raise


def replace_results(directory, staged, backup, before, validate):
    baseline.require(inventory(directory, OLD_NAMES) == before, "Results changed before replacement")
    raw = directory / OLD_NAMES[0]
    original_stat = raw.stat()
    backup.mkdir()
    moved, published = [], []
    try:
        for name in OLD_NAMES[1:]:
            (directory / name).rename(backup / name)
            moved.append(name)
        for name in FINAL_NAMES[1:]:
            baseline.require(not (directory / name).exists(), f"Unexpected destination appeared: {name}")
            (staged / name).rename(directory / name)
            published.append(name)
        validate()
        inventory(directory, FINAL_NAMES)
        current = raw.stat()
        baseline.require((original_stat.st_ino, original_stat.st_size, original_stat.st_mtime_ns)
                         == (current.st_ino, current.st_size, current.st_mtime_ns)
                         and baseline.sha256(raw) == before[OLD_NAMES[0]], "Original prediction changed during replacement")
    except BaseException as error:
        try:
            for name in reversed(published):
                (directory / name).rename(staged / name)
            for name in reversed(moved):
                baseline.require(not (directory / name).exists(), f"Cannot restore occupied original name: {name}")
                (backup / name).rename(directory / name)
            baseline.require(inventory(directory, OLD_NAMES) == before, "Restored files differ from original results")
        except BaseException as recovery_error:
            raise RecoveryRequired(f"Replacement failed: {error}; rollback failed: {recovery_error}; "
                                   f"preserved recovery files: {backup.parent}") from recovery_error
        raise


@contextlib.contextmanager
def stop_on_signal():
    previous = {}

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Received {signal.Signals(signum).name}")

    try:
        for signum in (signal.SIGTERM, signal.SIGHUP):
            previous[signum] = signal.signal(signum, interrupt)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def run(apply=False):
    require_cpu_only()
    statuses = {entry[1]: "NOT PROCESSED" for entry in delivery.SAMPLE_MAPPING}
    current = None
    try:
        with stop_on_signal(), baseline.gpu_lock():
            root = check_no_symlink_components(OUTPUT_ROOT)
            baseline.require({path.name for path in root.iterdir()} == {entry[0] for entry in delivery.SAMPLE_MAPPING},
                             "Expected exactly sample directories 01 through 08")
            package = delivery.read_delivery()
            snapshots = {}
            compatible = {}
            for record in package["records"]:
                current = record["pilot"]
                directory = root / record["output_number"]
                names = FINAL_NAMES if (directory / FINAL_NAMES[1]).exists() else OLD_NAMES
                snapshots[current] = (names, inventory(directory, names))
                if names == OLD_NAMES:
                    validate_completed_sample(record, directory)
                prediction = decode_video(directory / names[0], SIZE)
                frames = source_frames(record)
                validate_four(directory / names[3], frames, prediction)
                if names == OLD_NAMES:
                    validate_old_prediction_panels(directory, prediction)
                else:
                    compatible[current] = validate_arrangement(
                        directory, frames, prediction, snapshots[current][1][names[0]],
                        snapshots[current][1][names[3]], allow_legacy=True)
                print(f"[{record['output_number']}/08] Checking prediction/source{record['source_id']}: PASS; "
                      f"SHA256={snapshots[current][1][names[0]]}", flush=True)
                del frames, prediction
            for record in package["records"]:
                current = record["pilot"]
                directory = root / record["output_number"]
                names, before = snapshots[current]
                baseline.require(inventory(directory, names) == before, f"{current} changed after preflight")
                if not apply:
                    baseline.require(names != FINAL_NAMES or compatible[current],
                                     f"{current} comparison needs CPU player-compatibility repair; use --apply")
                    statuses[current] = "CHECKED"
                    continue
                if names == FINAL_NAMES and compatible[current]:
                    statuses[current] = "SUCCESS (ALREADY REORGANIZED)"
                    continue
                prediction = decode_video(directory / names[0], SIZE)
                frames = source_frames(record)
                work = Path(tempfile.mkdtemp(prefix=TEMP_PREFIX, dir=root.parent))
                preserve = False
                try:
                    staged = work / "staged"
                    staged.mkdir()
                    print(f"[{record['output_number']}/08] Creating 3-column video...", flush=True)
                    encode_comparison(staged / FINAL_NAMES[1], frames, prediction,
                                      lambda message: print(f"[{record['output_number']}/08] {message}", flush=True))
                    if names == FINAL_NAMES:
                        validate_comparison(staged / FINAL_NAMES[1], frames, prediction)
                        replace_comparison(
                            directory, staged / FINAL_NAMES[1], work / "previous", before,
                            lambda: validate_arrangement(directory, frames, prediction,
                                                         before[FINAL_NAMES[0]], before[FINAL_NAMES[3]]))
                    else:
                        print(f"[{record['output_number']}/08] Creating 3-row overview...", flush=True)
                        save_png_exclusive(staged / FINAL_NAMES[2], three_rows(frames, prediction))
                        with (directory / OLD_NAMES[3]).open("rb") as reader, (staged / FINAL_NAMES[3]).open("xb") as writer:
                            shutil.copyfileobj(reader, writer)
                        validate_arrangement(staged, frames, prediction, before[OLD_NAMES[0]], before[OLD_NAMES[3]],
                                             raw_path=directory / OLD_NAMES[0])
                        replace_results(directory, staged, work / "previous", before,
                                        lambda: validate_arrangement(directory, frames, prediction, before[OLD_NAMES[0]],
                                                                     before[OLD_NAMES[3]]))
                except RecoveryRequired:
                    preserve = True
                    raise
                finally:
                    if not preserve:
                        shutil.rmtree(work)
                statuses[current] = "SUCCESS"
                print(f"[{record['output_number']}/08] Validation: PASS; original MP4 unchanged; exactly four files", flush=True)
                del frames, prediction
            for record in package["records"]:
                current = record["pilot"]
                names, before = snapshots[current]
                final_hashes = inventory(root / record["output_number"], FINAL_NAMES if apply else names)
                baseline.require(final_hashes[FINAL_NAMES[0]] == before[OLD_NAMES[0]], "Final raw prediction hash mismatch")
            require_cpu_only()
        code = 0
    except (Exception, KeyboardInterrupt) as error:
        if current:
            statuses[current] = "FAILED"
        print(f"FAILED: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        code = 130 if isinstance(error, KeyboardInterrupt) else 1
    print("\nVACE RESULTS REORGANIZATION", flush=True)
    for pilot, status in statuses.items():
        print(f"{pilot}: {status}", flush=True)
    print(f"\nGPU INFERENCE RUNS: 0\n\nOUTPUT:\n{OUTPUT_ROOT}\nExit code: {code}", flush=True)
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--apply", action="store_true", help="Replace verified comparisons using existing predictions only")
    sys.exit(run(apply=parser.parse_args().apply))
