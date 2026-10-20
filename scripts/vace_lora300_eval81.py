"""Evaluate the completed VACE LoRA-300, with a visual review between two batches."""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image

import buildingfiller_pilot8 as delivery
import vace_p01 as baseline
import vace_roof_data as roof_data
from composite_vace_results import encode_video, validate_encoded
from reorganize_vace_results import decode_video
from vace_p01_81f_console import ConsoleState, visible_worker
from vace_p01_results import HEADER_HEIGHT, SIZE, comparison_panel, save_png_exclusive
from vace_roof_smoke import inference_arguments, load_stage_lora


ROOT = baseline.ROOT
TRAINING = ROOT / "outputs/vace/lora_roof_smoke/run_20261020_165237_150325_2048720/training"
LORA = TRAINING / "lora_update0300.safetensors"
OUTPUT_ROOT = ROOT / "outputs/vace/lora300_eval81"
BASE_ROOT = ROOT / "outputs/vace/test"
TRAINED = ("04", "06")
TRANSFER = ("01", "02", "03", "05", "07", "08")
FINAL_NAMES = (
    "01_微调后补全.mp4",
    "02_Base_微调后_GT_对比.mp4",
    "03_关键帧总览.png",
    "04_Mask验证.png",
)
ROLES = (("Before", "damaged"), ("Base", "base"), ("LoRA-300", "lora300"), ("GT", "gt"))


def evaluation_config():
    experiment = json.loads((TRAINING / "experiment.json").read_text(encoding="utf-8"))
    runtime = experiment["runtime"]
    baseline.require(runtime["status"] == "complete" and runtime["actual_updates"] == 300
                     and runtime["stages"]["lora300"]["exit_code"] == 0,
                     "The selected LoRA-300 must come from the completed, evaluated run")
    config = experiment["config"]
    frozen = baseline.load_frozen_config()["inference"]
    # These are the unchanged WanVideoPipeline defaults used by the old Base entry.
    expected = dict(frozen, cfg_merge=False, denoising_strength=1.0,
                    tile_size=[30, 52], tile_stride=[15, 26])
    baseline.require(config["inference"] == expected, "Base/LoRA inference settings differ")
    baseline.require(config["evaluation"]["lora_multiplier"] == 1.0
                     and config["training"]["lora_base_model"] == "vace",
                     "Only the trained VACE branch at scale 1.0 is supported")
    baseline.require(LORA.is_file(), f"Missing approved LoRA-300: {LORA}")
    config["inference"]["num_frames"] = 81
    return config


def sample_root(bundle, number):
    mapping = next(item for item in delivery.SAMPLE_MAPPING if item[0] == number)
    _, pilot, source_id, kind, tier = mapping
    prefix = delivery.PACKAGE_ID + "/"
    manifest = json.loads(bundle.read(prefix + "manifest.json"))
    matches = [sample for sample in manifest["samples"] if sample["pilot"] == pilot]
    baseline.require(len(matches) == 1, f"{pilot}: missing or duplicate source")
    sample = matches[0]
    baseline.require((sample["source_id"], sample["kind"], sample["tier"]) == (source_id, kind, tier)
                     and sample["relative_root"] == f"samples/{pilot}"
                     and sample["stable_id"] == f"b{source_id}_" + sample["source_sha256"][:8],
                     f"{pilot}: source identity differs")
    root = prefix + sample["relative_root"] + "/"
    metadata = json.loads(bundle.read(root + "sample_meta.json"))
    cameras = json.loads(bundle.read(root + "cameras.json"))
    baseline.require(metadata["sample_id"] == cameras["sample_id"] == sample["sample_id"]
                     and metadata["source"]["sha256"] == sample["source_sha256"]
                     and cameras["frames"] == 81 and cameras["fps_metadata"] == 16,
                     f"{pilot}: metadata/camera mismatch")
    poses = cameras["resolutions"]["512"]
    baseline.require(len(poses) == 81 and all(
        pose["frame"] == index and pose["frame_id"] == f"{index:03}"
        and (pose["width"], pose["height"]) == SIZE for index, pose in enumerate(poses)),
        f"{pilot}: frame order or resolution mismatch")
    return root, mapping


def read_conditions(number):
    """Open Damaged and Exact only; GT pixels are not read by the inference loader."""
    result = dict(damaged=[], model_mask=[], conditioning=[])
    with zipfile.ZipFile(delivery.DEFAULT_ARCHIVE) as bundle:
        root, mapping = sample_root(bundle, number)
        for role in ("damaged", "exact"):
            names = [name for name in bundle.namelist() if name.startswith(root + f"512/{role}/")]
            baseline.require(len(names) == 81 and set(names) == {
                root + f"512/{role}/{index:03}.png" for index in range(81)},
                f"{mapping[1]}: expected B000-B080 {role}")
        for index in range(81):
            damaged = roof_data.read_png(bundle, root + f"512/damaged/{index:03}.png", "damaged")
            exact = roof_data.read_png(bundle, root + f"512/exact/{index:03}.png", "exact")
            mask, condition = baseline.adapt_images(damaged, exact)
            result["damaged"].append(damaged)
            result["model_mask"].append(mask)
            result["conditioning"].append(condition)
    result.update(number=number, pilot=mapping[1], source_id=mapping[2])
    return result


def read_gt(number):
    with zipfile.ZipFile(delivery.DEFAULT_ARCHIVE) as bundle:
        root, _ = sample_root(bundle, number)
        return [roof_data.read_png(bundle, root + f"512/gt/{index:03}.png", "gt") for index in range(81)]


def panel(groups, index, roles=ROLES):
    return comparison_panel(tuple((f"{label} B{index:03}", groups[role][index]) for label, role in roles))


def overview_views(masks):
    areas = [np.count_nonzero(np.asarray(mask)[:, :, 0]) for mask in masks]
    views = [0, 20, 40, 60, 80]
    # Keep the default orbit unless it misses the clearly visible part of the hole.
    if max(areas[index] for index in views) < 0.5 * max(areas):
        replacement = min((20, 40, 60), key=lambda index: areas[index])
        views[views.index(replacement)] = int(np.argmax(areas))
    return sorted(views)


def print_pixel_summary(groups):
    """A small diagnostic, never used to decide whether geometry is repaired."""
    for role in ("base", "lora300"):
        inside_sum = outside_sum = inside_count = outside_count = 0
        per_frame = []
        for index, mask in enumerate(groups["mask"]):
            white = np.asarray(mask)[:, :, 0] > 0
            pred = np.asarray(groups[role][index]).astype(np.int16)
            gt = np.asarray(groups["gt"][index]).astype(np.int16)
            damaged = np.asarray(groups["damaged"][index]).astype(np.int16)
            error = np.abs(pred - gt)
            inside_sum += int(error[white].sum())
            inside_count += int(white.sum()) * 3
            # Exclude the white background from the known-building diagnostic.
            known_building = ~white & (damaged.min(axis=2) < 245)
            outside_sum += int(np.abs(pred - damaged)[known_building].sum())
            outside_count += int(known_building.sum()) * 3
            if white.any():
                per_frame.append((float(error[white].mean()), index))
        worst = sorted(per_frame, reverse=True)[:5]
        print(f"{role}: masked GT RGB MAE={inside_sum / max(inside_count, 1):.4f}; "
              f"known-building change={outside_sum / max(outside_count, 1):.4f}; "
              f"worst visible frames={[(f'B{i:03}', round(e, 3)) for e, i in worst]}", flush=True)


def render_sample(destination, sample, base, prediction, raw_saved=False):
    raw_path = destination / FINAL_NAMES[0]
    if raw_saved:
        baseline.require({p.name for p in destination.iterdir()} == {FINAL_NAMES[0]},
                         "CPU completion requires exactly the existing raw prediction")
    else:
        baseline.require(not destination.exists(), f"Refuse to overwrite: {destination}")
        destination.mkdir()
        print(f"Saving raw LoRA-300 video for {sample['pilot']}", flush=True)
        encode_video(raw_path, prediction, SIZE, print)
        validate_encoded(raw_path, SIZE, prediction)
        print(f"RAW VIDEO SAVED: {raw_path}; retained even if later CPU rendering fails", flush=True)
    # The model has returned before this first GT pixel read.
    # comparison_panel expects NumPy pixels, whereas the model/loader return PIL images.
    groups = {role: [np.asarray(image) for image in images] for role, images in dict(
        damaged=sample["damaged"], mask=sample["model_mask"],
        base=base, lora300=prediction, gt=read_gt(sample["number"])).items()}
    views = overview_views(groups["mask"])
    mask_index = int(np.argmax([np.count_nonzero(np.asarray(mask)[:, :, 0]) for mask in groups["mask"]]))
    with tempfile.TemporaryDirectory(prefix=".media_", dir=destination.parent) as temporary:
        staged = Path(temporary)
        print(f"Creating comparison video for {sample['pilot']}: Before | Base | LoRA-300 | GT", flush=True)
        encode_video(staged / FINAL_NAMES[1], (panel(groups, i) for i in range(81)), (2048, 544), print)
        overview = Image.new("RGB", (2560, 4 * (512 + HEADER_HEIGHT)), "white")
        for row, (label, role) in enumerate(ROLES):
            strip = comparison_panel(tuple((f"{label} B{i:03}", groups[role][i]) for i in views))
            overview.paste(strip, (0, row * (512 + HEADER_HEIGHT)))
        verification = panel(groups, mask_index, (ROLES[0], ("ModelMask", "mask"), *ROLES[1:]))
        save_png_exclusive(staged / FINAL_NAMES[2], overview)
        save_png_exclusive(staged / FINAL_NAMES[3], verification)
        validate_encoded(staged / FINAL_NAMES[1], (2048, 544), (panel(groups, i) for i in range(81)))
        for name in FINAL_NAMES[2:]:
            with Image.open(staged / name) as image:
                image.load()
        baseline.require({p.name for p in staged.iterdir()} == set(FINAL_NAMES[1:]), "Expected three comparisons")
        for name in FINAL_NAMES[1:]:
            shutil.move(str(staged / name), str(destination / name))
    baseline.require({p.name for p in destination.iterdir()} == set(FINAL_NAMES), "Expected exactly four files")
    print_pixel_summary(groups)
    print(f"SAVED {sample['pilot']}: {destination}; overview={views}; Mask=B{mask_index:03}; "
          "81 frames decoded, four files verified", flush=True)


def worker(number, output_root):
    import torch
    from vace_p01_environment import check_environment

    lock_fd = int(os.environ["WAN_EVAL81_LOCK_FD"])
    baseline.require(os.fstat(lock_fd).st_ino == (ROOT / "logs/.wan_a6000_gpu.lock").stat().st_ino,
                     "Missing parent GPU lock")
    config = evaluation_config()
    environment = check_environment()
    baseline.require("A6000" in environment["cuda"]["device_name"]
                     and environment["cuda"]["free_bytes"] >= 24 * 1024 ** 3, "Need free A6000")
    sample = read_conditions(number)
    print(f"\nRunning {number} / {sample['pilot']} / source{sample['source_id']} / LoRA-300 / 81 frames", flush=True)
    print(f"Reusing Base for {sample['pilot']}: {BASE_ROOT / number / '01_补全结果.mp4'}", flush=True)
    base = decode_video(BASE_ROOT / number / "01_补全结果.mp4", SIZE)
    state = ConsoleState(total_steps=30)
    paths = roof_data.inspect_local_models(config)
    torch.cuda.reset_peak_memory_stats(0)
    state.stage("model_loading", "Fresh VACE-1.3B, BF16")
    pipe = baseline.load_local_vace(paths, memory_mode="full")
    pipe.eval()
    print(f"ACTUAL LORA ABSOLUTE PATH: {LORA.resolve()}", flush=True)
    load_stage_lora(pipe, LORA, config)
    arguments = inference_arguments(config, sample)
    arguments["progress_bar_cmd"] = state.progress
    with state.vae_stages(pipe.vae):
        prediction = pipe(**arguments)
    baseline.require(len(prediction) == 81 and all(
        image.mode == "RGB" and image.size == SIZE for image in prediction), "Invalid prediction dimensions")
    torch.cuda.synchronize()
    print(f"Inference done: elapsed={state.elapsed:.1f}s; peak allocated="
          f"{torch.cuda.max_memory_allocated() / 2**30:.3f} GiB; reserved="
          f"{torch.cuda.max_memory_reserved() / 2**30:.3f} GiB", flush=True)
    del pipe
    torch.cuda.empty_cache()
    render_sample(output_root / number, sample, base, prediction)
    print(f"SUCCESS {sample['pilot']}: elapsed={state.elapsed:.1f}s", flush=True)


def dry_run():
    from safetensors import safe_open

    config = evaluation_config()
    roof_data.inspect_local_models(config)
    with safe_open(LORA, framework="np") as weights:
        metadata = weights.metadata()
        baseline.require(metadata["base_model"] == "vace" and metadata["optimizer_update"] == "300"
                         and len(weights.keys()) == 300, "Unexpected LoRA checkpoint")
    print(f"Approved LoRA: {LORA.resolve()}", flush=True)
    print(f"Positive: {config['inference']['prompt']}\nNegative: {config['inference']['negative_prompt']}", flush=True)
    for number, pilot, source_id, _, _ in delivery.SAMPLE_MAPPING:
        sample = read_conditions(number)
        decode_video(BASE_ROOT / number / "01_补全结果.mp4", SIZE)
        black = sum(not np.any(np.asarray(mask)) for mask in sample["model_mask"])
        print(f"PASS {number}/{pilot}/source{source_id}: 81 readable conditions, "
              f"{black} unchanged black masks; Base 81F/512/16FPS decoded", flush=True)
    print("CPU PREPARATION PASSED; no model inference performed", flush=True)


def run_batch(transfer, output_root):
    started = time.monotonic()
    queue = TRANSFER if transfer else TRAINED
    with baseline.gpu_lock() as lock:
        if transfer:
            baseline.require(output_root is not None, "Transfer requires the reviewed first-stage output root")
            output_root = output_root.resolve()
            baseline.require(output_root == OUTPUT_ROOT or OUTPUT_ROOT in output_root.parents,
                             "All new results must remain under lora300_eval81")
            for number in TRAINED:
                baseline.require({p.name for p in (output_root / number).iterdir()} == set(FINAL_NAMES),
                                 "P04/P06 results must exist before transfer")
        else:
            baseline.require(output_root is None, "The first stage chooses a new output directory")
            output_root = OUTPUT_ROOT
            if output_root.exists():
                output_root /= "run_" + datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%d_%H%M%S_%f")
            output_root.mkdir(parents=True, exist_ok=False)
        print(f"OUTPUT ROOT: {output_root}", flush=True)
        for number in queue:
            baseline.require(not (output_root / number).exists(), f"Existing result: {number}; no rerun")
            environment = dict(os.environ, WAN_EVAL81_LOCK_FD=str(lock.fileno()))
            command = [sys.executable, "-u", "-B", str(Path(__file__).resolve()),
                       "--worker", number, "--output-root", str(output_root)]
            returncode, interrupted = visible_worker(command, environment, lock.fileno())
            print(f"P{number} worker exit code: {returncode}", flush=True)
            if returncode != 0 or interrupted:
                print("FAILED; stopping without retries or parameter changes", flush=True)
                return (returncode if returncode > 0 else 128 - returncode) if returncode else 130
        print(f"BATCH COMPLETE: {list(queue)}; elapsed={time.monotonic() - started:.1f}s", flush=True)
        if not transfer:
            print("P04/P06 ready for visual review. Other six NOT started.", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--run-trained", action="store_true")
    modes.add_argument("--run-transfer", action="store_true", help="Only after P04/P06 visual review")
    modes.add_argument("--render-only", choices=TRAINED + TRANSFER,
                       help="CPU only: complete comparisons from a retained raw prediction")
    modes.add_argument("--worker", choices=TRAINED + TRANSFER, help=argparse.SUPPRESS)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.render_only:
            baseline.require(args.output_root is not None, "CPU completion requires an output root")
            output_root = args.output_root.resolve()
            baseline.require(output_root == OUTPUT_ROOT or OUTPUT_ROOT in output_root.parents,
                             "CPU results must remain under lora300_eval81")
            number = args.render_only
            destination = output_root / number
            prediction = decode_video(destination / FINAL_NAMES[0], SIZE)
            base = decode_video(BASE_ROOT / number / "01_补全结果.mp4", SIZE)
            render_sample(destination, read_conditions(number), base, prediction, raw_saved=True)
            return 0
        if args.worker:
            worker(args.worker, args.output_root)
            return 0
        if args.run_trained or args.run_transfer:
            return run_batch(args.run_transfer, args.output_root)
        dry_run()
        return 0
    except Exception:
        traceback.print_exc()
        print("FAILED; no automatic retry, training, or other experiment", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
