"""Fixed paired PNG windows and CPU preparation for the P04/P06 overfit test."""

import io
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

import vace_p01 as baseline


ROOT = baseline.ROOT
CONFIG_PATH = ROOT / "configs/vace_roof_smoke_v1.json"
MAPPING = (("04", "P04", 1227, "A2"), ("06", "P06", 1848, "A3"))


def best_window(white_pixels):
    baseline.require(len(white_pixels) == 80, "Window selection requires B000-B079, excluding B080")
    scores = [sum(white_pixels[start:start + 17]) for start in range(64)]
    start = max(range(64), key=lambda index: scores[index])
    return list(range(start, start + 17)), scores[start]


def load_config():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    frozen = json.loads((ROOT / config["source"]["prompt_source"]).read_text(encoding="utf-8"))
    for key, value in frozen["inference"].items():
        baseline.require(config["inference"][key] == value, f"Frozen inference setting changed: {key}")
    baseline.require([(sample["number"], sample["pilot"], sample["source_id"], sample["kind"])
                      for sample in config["samples"]] == list(MAPPING), "Only P04/P06 may enter this run")
    training = config["training"]
    baseline.require(training["max_updates"] == 300 and training["save_updates"] == [100, 300]
                     and training["lora_base_model"] == "vace" and training["lora_rank"] == 16
                     and training["lora_alpha"] == 16 and training["sample_order"] == ["P04", "P06"]
                     and training["batch_size"] == training["gradient_accumulation_steps"] == 1,
                     "Unexpected training budget or LoRA target")
    for sample in config["samples"]:
        indices = sample["source_indices"]
        baseline.require(indices == list(range(indices[0], indices[0] + 17))
                         and 0 <= indices[0] <= 63, "Expected one fixed continuous 17-frame window")
    return config


def read_png(bundle, member, role):
    with Image.open(io.BytesIO(bundle.read(member))) as image:
        image.load()
        baseline.require(image.format == "PNG" and image.size == (512, 512)
                         and image.mode == ("L" if role == "exact" else "RGB"), f"Invalid source PNG: {member}")
        if role == "exact":
            baseline.require(np.isin(np.asarray(image), [0, 255]).all(), "Exact must be binary")
        return image.copy()


def load_samples(config, include_gt=True, verify_selection=False):
    archive = ROOT / config["source"]["archive"]
    prefix = config["source"]["package_id"] + "/"
    samples = []
    with zipfile.ZipFile(archive) as bundle:
        manifest = json.loads(bundle.read(prefix + "manifest.json"))
        for expected in config["samples"]:
            matches = [item for item in manifest["samples"] if item["pilot"] == expected["pilot"]]
            baseline.require(len(matches) == 1, "Missing or duplicate source identity")
            actual = matches[0]
            for key in ("pilot", "source_id", "kind", "tier", "stable_id", "sample_id", "relative_root"):
                baseline.require(actual[key] == expected[key], f"{expected['pilot']}: metadata {key} mismatch")
            root = prefix + expected["relative_root"] + "/"
            meta = json.loads(bundle.read(root + "sample_meta.json"))
            cameras = json.loads(bundle.read(root + "cameras.json"))
            baseline.require(meta["sample_id"] == cameras["sample_id"] == expected["sample_id"]
                             and meta["source"]["sha256"] == actual["source_sha256"]
                             and cameras["frames"] == 81, "Source metadata/camera identity mismatch")
            if verify_selection:
                white_pixels = []
                for index in range(80):
                    exact = read_png(bundle, root + f"512/exact/{index:03}.png", "exact")
                    white_pixels.append(int(np.count_nonzero(np.asarray(exact))))
                indices, score = best_window(white_pixels)
                baseline.require(indices == expected["source_indices"] and score == expected["window_white_pixels"],
                                 f"{expected['pilot']}: fixed maximum-mask window no longer matches source")
            sample = dict(expected, damaged=[], model_mask=[], conditioning=[])
            if include_gt:
                sample["gt"] = []
            white_sum = 0
            for index in expected["source_indices"]:
                pose = cameras["resolutions"]["512"][index]
                baseline.require(pose["frame"] == index and pose["frame_id"] == f"{index:03}"
                                 and (pose["width"], pose["height"]) == (512, 512),
                                 "Selected frame/camera mismatch")
                damaged = read_png(bundle, root + f"512/damaged/{index:03}.png", "damaged")
                exact = read_png(bundle, root + f"512/exact/{index:03}.png", "exact")
                mask, conditioning = baseline.adapt_images(damaged, exact)
                white_sum += int(np.count_nonzero(np.asarray(exact)))
                sample["damaged"].append(damaged)
                sample["model_mask"].append(mask)
                sample["conditioning"].append(conditioning)
                if include_gt:
                    sample["gt"].append(read_png(bundle, root + f"512/gt/{index:03}.png", "gt"))
            baseline.require(white_sum == expected["window_white_pixels"], "Selected Exact mask counts changed")
            samples.append(sample)
    return samples


def model_paths(config):
    return {name: str((ROOT / relative).resolve()) for name, relative in config["model"]["paths"].items()}


def inspect_local_models(config):
    paths = model_paths(config)
    sources = json.loads(baseline.MODEL_SOURCES.read_text(encoding="utf-8"))
    baseline.require(config["model"]["id"] == sources["model_id"]
                     and config["model"]["revision"] == sources["huggingface_revision"], "Wrong base model")
    for role in ("checkpoint", "text_encoder", "vae"):
        path = Path(paths[role])
        baseline.require(path.is_file(), f"Missing local model: {path}")
        expected = sources["checkpoint"]["accepted_files"][0]["bytes"] if role == "checkpoint" \
            else sources["reused_components"][role]["bytes"]
        baseline.require(path.stat().st_size == expected, f"Unexpected model file size: {role}")
    header = baseline.safetensors_header(Path(paths["checkpoint"]))
    baseline.require(header["structure_md5"] == sources["checkpoint"]["structure_md5"]
                     and header["vace_tensors"] == sources["checkpoint"]["vace_tensor_count"],
                     "Local checkpoint is not the VACE-1.3B branch structure")
    for filename in sources["reused_components"]["tokenizer"]["files_sha256"]:
        baseline.require((Path(paths["tokenizer"]) / filename).is_file(), f"Missing tokenizer {filename}")
    return paths


def prepare(config, check_interfaces=True):
    from vace_p01_environment import EXPECTED_PYTHON, _nvml_snapshot

    baseline.require(Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(), "Use the existing scj Python")
    samples = load_samples(config, include_gt=True, verify_selection=True)
    for sample in samples:
        indices = sample["source_indices"]
        print(f"PASS {sample['pilot']} source{sample['source_id']}: "
              f"B{indices[0]:03}-B{indices[-1]:03}, 17 aligned GT/Damaged/Exact PNGs, "
              f"Exact white pixels={sample['window_white_pixels']}", flush=True)
    paths = inspect_local_models(config)
    print("PASS local VACE/T5/VAE/tokenizer present; VACE header checked; no full-model hash scan", flush=True)
    for tool in ("ffmpeg", "ffprobe"):
        baseline.require(shutil.which(tool), f"Missing existing video tool: {tool}")
    for destination in (ROOT / "outputs/vace", Path(tempfile_root())):
        baseline.require(destination.is_dir() and os.access(destination, os.W_OK | os.X_OK)
                         and shutil.disk_usage(destination).free >= 2 * 1024 ** 3,
                         f"Need writable storage with at least 2 GiB free: {destination}")
    interface = {}
    if check_interfaces:
        from vace_roof_cpu_check import check_interfaces as inspect_interfaces

        interface = inspect_interfaces(config)
        print(f"PASS CPU interfaces: {json.dumps(interface, ensure_ascii=False)}", flush=True)
        for sample in samples:
            baseline.check_loader_contract(sample["conditioning"], sample["model_mask"], num_frames=17)
        print("PASS real CPU condition loader: masks 0/1, white pixels RGB127, 1x3x17x512x512", flush=True)
    try:
        snapshot = _nvml_snapshot()
        print(f"GPU read-only: {snapshot['device_name']}; compute tasks={snapshot['compute_processes']}; "
              f"graphics tasks={snapshot['graphics_processes']}", flush=True)
    except ValueError as error:
        print(f"GPU read-only status unavailable: {error}; CUDA check deferred to manual launch", flush=True)
    print(f"Positive prompt:\n{config['inference']['prompt']}\n"
          f"Negative prompt:\n{config['inference']['negative_prompt']}", flush=True)
    print("CPU PREPARATION: PASS\nGPU TRAINING/INFERENCE: NOT STARTED / NOT VALIDATED", flush=True)
    return samples, paths, interface


def tempfile_root():
    import tempfile

    return tempfile.gettempdir()
