"""Prepare or explicitly run the frozen P01 VACE baseline using local models only."""

import argparse
import ast
import contextlib
import fcntl
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FROZEN_CONFIG = ROOT / "configs/vace_p01_465_17f_v1.json"
FROZEN_SHA256 = "7e82161e270d8a24e231feb2ca95d08ae6e02e861ce5460a241730bfbcc07080"
MODEL_SOURCES = ROOT / "configs/vace_p01_model_sources_v1.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 ** 2), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def verify_file_sha256(path, approved, label):
    path = Path(path)
    approved = set(approved)
    require(bool(approved), f"Missing approved SHA256: {label}; path={path}; refusing to read or load an unapproved model")

    def identity(value):
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns

    before = path.stat()
    checksum = sha256(path)
    after = path.stat()
    require(identity(before) == identity(after),
            f"Model file changed during SHA256 verification: {label}; path={path}; actual={checksum}; "
            f"expected={','.join(sorted(approved))}; bytes={after.st_size}; "
            f"before={identity(before)}; after={identity(after)}; stopping without retry")
    require(checksum in approved,
            f"Model SHA256 mismatch: {label}; path={path}; bytes={after.st_size}; "
            f"expected={','.join(sorted(approved))}; actual={checksum}; mtime_ns={after.st_mtime_ns}; "
            "no file metadata change detected; cause is undetermined; stopping without retry or bypass")
    return checksum


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def load_frozen_config(path=FROZEN_CONFIG):
    require(sha256(path) == FROZEN_SHA256, "Frozen experiment config changed; create a new experiment instead")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def relative_path(value):
    path = PurePosixPath(value)
    require(value and not path.is_absolute() and ".." not in path.parts and "\\" not in value,
            f"Unsafe relative path: {value}")
    return path


def adapt_images(damaged, exact):
    require(damaged.mode == "RGB" and exact.mode == "L", "Expected RGB Damaged and L Exact")
    require(damaged.size == exact.size == (512, 512), "Images must remain 512x512")
    exact_pixels = np.asarray(exact)
    require(np.isin(exact_pixels, [0, 255]).all(), "Exact must be binary 0/255")
    mask = np.where(exact_pixels > 127, 255, 0).astype(np.uint8)
    condition = np.array(damaged, copy=True)
    condition[mask == 255] = 127
    require(np.array_equal(condition[mask == 0], np.asarray(damaged)[mask == 0]),
            "Conditioning changed pixels outside the mask")
    return Image.fromarray(np.repeat(mask[:, :, None], 3, axis=2)), Image.fromarray(condition)


def read_inputs(config, project_root=ROOT, include_gt=True):
    archive = project_root / config["source"]["archive"]
    require(sha256(archive) == config["source"]["archive_sha256"], "Source ZIP SHA256 mismatch")
    selected = []
    with zipfile.ZipFile(archive) as bundle:
        require(len(bundle.namelist()) == len(set(bundle.namelist())), "Duplicate ZIP members")
        prefix = config["source"]["package_id"] + "/"
        manifest = json.loads(bundle.read(prefix + "manifest.json"))
        require(manifest["source_identity"]["source_id"] == 465
                and manifest["source_identity"]["stable_id"] == config["source"]["stable_id"],
                "Unexpected BuildingFiller source identity")
        inventory = {item["relative_path"]: item for item in manifest["files"]}
        sample_root = config["source"]["sample_relative_root"]
        camera_bytes = bundle.read(prefix + sample_root + "/cameras.json")
        require(hashlib.sha256(camera_bytes).hexdigest()
                == manifest["original_source_context"]["cameras_sha256"], "Camera hash mismatch")
        cameras = json.loads(camera_bytes)
        require(manifest["frame_count"] == len(manifest["frames"]) == 81, "Expected 81 source frames")
        indices = range(config["source"]["frame_start_inclusive"],
                        config["source"]["frame_end_inclusive"] + 1, config["source"]["frame_stride"])
        require(len(indices) == config["inference"]["num_frames"], "Source frame selection does not match inference length")
        for output_index, source_index in enumerate(indices):
            frame = manifest["frames"][source_index]
            pose = cameras["resolutions"]["512"][source_index]
            require(frame["frame_index"] == frame["camera_index"] == pose["frame"] == source_index,
                    "Frame/camera index mismatch")
            require(frame["frame_id"] == pose["frame_id"] == f"{source_index:03}", "Frame ID mismatch")
            payload = {}
            images = {}
            for role in (("damaged", "exact", "gt") if include_gt else ("damaged", "exact")):
                expected = f"{sample_root}/512/{role}/{source_index:03}.png"
                require(frame[role] == expected, "Unexpected frame role mapping")
                content = bundle.read(prefix + str(relative_path(expected)))
                require(hashlib.sha256(content).hexdigest() == inventory[expected]["sha256"],
                        f"Source PNG hash mismatch: {expected}")
                payload[role] = content
                with Image.open(io.BytesIO(content)) as image:
                    image.load()
                    require(image.format == "PNG" and image.size == (512, 512)
                            and image.mode == ("L" if role == "exact" else "RGB"), "Invalid source PNG")
                    if role != "gt":
                        images[role] = image.copy()
            model_mask, conditioning = adapt_images(images["damaged"], images["exact"])
            selected.append(dict(source_index=source_index, output_index=output_index,
                                 frame=frame, camera=pose, original_bytes=payload,
                                 model_mask=model_mask, conditioning=conditioning))
    return selected


def safetensors_header(path):
    with Path(path).open("rb") as stream:
        prefix = stream.read(8)
        require(len(prefix) == 8, "Truncated safetensors header")
        length = struct.unpack("<Q", prefix)[0]
        require(0 < length < 4 * 1024 ** 2, "Invalid safetensors header size")
        header = json.loads(stream.read(length))
    keys = {key: item for key, item in header.items() if key != "__metadata__"}
    parts = []
    for key, item in keys.items():
        parts.extend((key + ":" + "_".join(map(str, item["shape"])), key))
    return dict(bytes=Path(path).stat().st_size, tensors=len(keys),
                vace_tensors=sum(key.startswith("vace") for key in keys),
                structure_md5=hashlib.md5(",".join(sorted(parts)).encode()).hexdigest())


def registered_models(structure_hash, project_root=ROOT):
    source = project_root / "code/DiffSynth-Studio/diffsynth/configs/model_configs.py"
    matches = []
    for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Dict):
            try:
                item = ast.literal_eval(node)
            except (ValueError, TypeError):
                continue
            if isinstance(item, dict) and item.get("model_hash") == structure_hash:
                matches.append(item["model_name"])
    return sorted(set(matches))


def inspect_models(config, checkpoint=None, verify=False, project_root=ROOT, sources=None, notify=None):
    sources = sources or json.loads(MODEL_SOURCES.read_text(encoding="utf-8"))
    require(sources["model_id"] == config["model"]["id"], "Model source ID mismatch")
    checkpoint = Path(checkpoint) if checkpoint else project_root / sources["checkpoint"]["default_local_path"]
    paths = dict(checkpoint=checkpoint.resolve())
    paths.update({name: (project_root / config["model"][name]).resolve()
                  for name in ("text_encoder", "vae", "tokenizer")})
    missing = [name for name, path in paths.items() if not path.exists()]
    report = dict(paths={name: str(path) for name, path in paths.items()}, missing=missing,
                  checksums_verified=False, model_id=sources["model_id"],
                  official_revision=sources["huggingface_revision"], headers={}, files_sha256={}, file_identities={})
    for name in ("checkpoint", "text_encoder", "vae"):
        if name in missing:
            continue
        header = safetensors_header(paths[name])
        report["headers"][name] = header
        if name == "checkpoint":
            expected = sources["checkpoint"]
            require(header["bytes"] in {item["bytes"] for item in expected["accepted_files"]},
                    "Checkpoint size is not an approved original VACE file")
            require(header["structure_md5"] == expected["structure_md5"]
                    and header["tensors"] == expected["tensor_count"]
                    and header["vace_tensors"] == expected["vace_tensor_count"],
                    "Checkpoint is not the frozen VACE-1.3B structure")
            require(registered_models(header["structure_md5"], project_root)
                    == ["wan_video_dit", "wan_video_vace"], "Local DiffSynth cannot load both VACE branches")
        else:
            require(header["bytes"] == sources["reused_components"][name]["bytes"],
                    f"Unexpected reused component size: {name}")
            expected_name = "wan_video_text_encoder" if name == "text_encoder" else "wan_video_vae"
            require(expected_name in registered_models(header["structure_md5"], project_root),
                    f"Local DiffSynth does not recognize {name}")
        if verify:
            if notify:
                notify(f"Reading full SHA256: {name}, {header['bytes']:,} bytes")
            if name == "checkpoint":
                approved = {item["sha256"] for item in sources["checkpoint"]["accepted_files"]
                            if item["bytes"] == header["bytes"]}
            else:
                expected = sources["reused_components"][name]["sha256"]
                require(expected is not None, f"Missing approved component SHA256: {name}")
                approved = {expected}
            checksum = verify_file_sha256(paths[name], approved, name)
            report["files_sha256"][name] = checksum
            if notify:
                notify(f"SHA256 verified: {name} = {checksum}")
        info = paths[name].stat()
        report["file_identities"][name] = dict(device=info.st_dev, inode=info.st_ino,
                                                bytes=info.st_size, mtime_ns=info.st_mtime_ns)
    if "tokenizer" not in missing:
        for filename, expected in sources["reused_components"]["tokenizer"]["files_sha256"].items():
            path = paths["tokenizer"] / filename
            require(path.is_file() and sha256(path) == expected, f"Tokenizer hash mismatch: {filename}")
            report["files_sha256"]["tokenizer/" + filename] = expected
    report["checksums_verified"] = verify and not missing
    report["ready_for_gpu"] = report["checksums_verified"]
    return report


def prepare_output(output, config, selected, model_report, config_path=FROZEN_CONFIG, reserved=False):
    output = Path(output)
    if reserved:
        require(output.is_dir() and (output / "run.log").is_file(), "Run directory was not reserved by the launcher")
    else:
        output.mkdir(parents=True, exist_ok=False)
    for relative in ("inputs/damaged", "inputs/exact", "inputs/model_mask",
                     "inputs/conditioning", "evaluation/gt", "provenance"):
        (output / relative).mkdir(parents=True)
    (output / "frozen_config.json").write_bytes(Path(config_path).read_bytes())
    frame_map, model_inputs, coverage = [], [], []
    for item in selected:
        output_id = f"{item['output_index']:03}"
        for role, content in item["original_bytes"].items():
            directory = "evaluation/gt" if role == "gt" else "inputs/" + role
            (output / directory / (output_id + ".png")).write_bytes(content)
        item["model_mask"].save(output / "inputs/model_mask" / (output_id + ".png"))
        item["conditioning"].save(output / "inputs/conditioning" / (output_id + ".png"))
        model_inputs.append(dict(output_index=item["output_index"], source_index=item["source_index"],
                                 conditioning=f"inputs/conditioning/{output_id}.png",
                                 model_mask=f"inputs/model_mask/{output_id}.png"))
        frame_map.append(dict(output_index=item["output_index"], source_index=item["source_index"],
                              source_frame=item["frame"], camera=item["camera"],
                              original_sha256={role: hashlib.sha256(content).hexdigest()
                                               for role, content in item["original_bytes"].items()}))
        coverage.append(dict(source_index=item["source_index"],
                             white_pixels=int((np.asarray(item["model_mask"])[:, :, 0] == 255).sum())))
    write_json(output / "frame_map.json", frame_map)
    write_json(output / "model_input_manifest.json", model_inputs)
    write_json(output / "model_preflight.json", model_report)
    write_json(output / "provenance/model_sources.json", json.loads(MODEL_SOURCES.read_text()))
    write_json(output / "provenance/preparation_environment.json", dict(
        python=sys.executable, python_version=platform.python_version(),
        packages={name: importlib.metadata.version(name) for name in ("numpy", "Pillow")},
        source_files_sha256={name: sha256(ROOT / name) for name in
                             ("scripts/vace_p01.py", "configs/vace_p01_model_sources_v1.json")},
    ))
    write_json(output / "adapter_report.json", dict(version=config["adapter"]["version"],
               input_shape=[1, 3, 17, 512, 512], mask_values=[0, 1], fill_RGB=[127, 127, 127],
               resized=False, mask_dilated=False, oracle_mask=True, GT_usage="evaluation_only", frames=coverage))
    write_json(output / "prepare_report.json", dict(status="cpu_prepared", created_at_utc=datetime.now(timezone.utc).isoformat(),
               frozen_config_sha256=FROZEN_SHA256, frame_count=len(selected), source_frames=list(range(5, 22)),
               model_weights_loaded=False, GPU_inference_started=False, prediction_created=False,
               model_preflight_ready=model_report["ready_for_gpu"]))
    static_roots = ("inputs", "evaluation", "provenance")
    static_names = ("frozen_config.json", "frame_map.json", "model_input_manifest.json", "model_preflight.json",
                    "adapter_report.json", "prepare_report.json")
    paths = [path for name in static_roots for path in (output / name).rglob("*") if path.is_file()]
    paths.extend(output / name for name in static_names)
    inventory = {path.relative_to(output).as_posix(): sha256(path) for path in sorted(paths)}
    write_json(output / "artifacts_sha256.json", inventory)
    for alias, target in {"input_damaged": "inputs/damaged", "input_modelmask": "inputs/model_mask",
                          "input_condition": "inputs/conditioning", "gt_reference": "evaluation/gt"}.items():
        (output / alias).symlink_to(target, target_is_directory=True)
    return output


def inference_arguments(config, conditions, masks):
    arguments = dict(config["inference"])
    arguments.pop("fps")
    arguments.pop("torch_dtype")
    require(len(conditions) == len(masks) == arguments["num_frames"], "Model input frame count mismatch")
    arguments.update(vace_video=conditions, vace_video_mask=masks, vace_reference_image=None)
    return arguments


def composite_frame(prediction, damaged, model_mask):
    return Image.composite(prediction, damaged, model_mask.convert("L"))


def check_loader_contract(conditions, masks, num_frames=17):
    import torch
    from diffsynth.pipelines.wan_video import WanVideoPipeline

    pipe = WanVideoPipeline(device="cpu", torch_dtype=torch.float32)
    require(len(conditions) == len(masks) == num_frames, "Loader input length mismatch")
    require(pipe.check_resize_height_width(512, 512, num_frames, verbose=0) == (512, 512, num_frames), "Pipeline would resize inputs")
    video = pipe.preprocess_video(conditions)
    mask = pipe.preprocess_video(masks, min_value=0, max_value=1)
    require(list(video.shape) == list(mask.shape) == [1, 3, num_frames, 512, 512], "Loader tensor shape mismatch")
    require(torch.isin(mask, torch.tensor([0.0, 1.0])).all().item(), "Loader mask is not binary")
    white = mask == 1
    require(torch.allclose(video[white], torch.full_like(video[white], -1 / 255), atol=1e-6),
            "Masked conditioning must be normalized RGB127")
    return dict(checked=True, device="cpu", shape=list(video.shape), mask_values=[0, 1],
                white_means="generate", white_condition_RGB=[127, 127, 127], model_loaded=False,
                GT_used=False, GPU_inference_started=False)


@contextlib.contextmanager
def gpu_lock(project_root=ROOT):
    directory = project_root / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".wan_a6000_gpu.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("An A6000 training or inference launcher holds the GPU lock") from exc
        yield stream


def check_old_training():
    entry = (ROOT / "code/Python_3D_Scanner/training/wan_train_entry.py").resolve()
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            command = [os.fsdecode(part) for part in (directory / "cmdline").read_bytes().split(b"\0") if part]
            working_dir = (directory / "cwd").resolve()
            require(not any((working_dir / value).resolve() == entry for value in command[1:] if value.endswith(".py")),
                    f"Old project training is running: PID {directory.name}")
        except (OSError, RuntimeError):
            continue


def load_local_vace(paths, memory_mode="full"):
    import torch
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

    require(memory_mode in ("full", "cpu-offload"), "Unknown model memory mode")
    options = {}
    if memory_mode == "cpu-offload":
        options = dict(offload_dtype=torch.bfloat16, offload_device="cpu",
                       onload_dtype=torch.bfloat16, onload_device="cpu",
                       preparing_dtype=torch.bfloat16, preparing_device="cuda",
                       computation_dtype=torch.bfloat16, computation_device="cuda")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device="cuda", redirect_common_files=False,
        model_configs=[ModelConfig(path=paths[name], skip_download=True, **options)
                       for name in ("checkpoint", "text_encoder", "vae")],
        tokenizer_config=ModelConfig(path=paths["tokenizer"], skip_download=True),
    )
    require(all(getattr(pipe, name) is not None for name in ("dit", "vace", "text_encoder", "vae")),
            "VACE/DiT/T5/VAE branch missing after loading")
    return pipe


def gpu_worker(output):
    from vace_p01_console import record_stage, inference_progress, vae_stages
    from vace_p01_environment import check_environment

    record_stage(output, "worker_preflight", "Checking verified files and runtime environment")
    descriptor = int(os.environ["WAN_VACE_LOCK_FD"])
    lock_path = ROOT / "logs/.wan_a6000_gpu.lock"
    require(os.fstat(descriptor).st_ino == lock_path.stat().st_ino, "Missing inherited GPU lock")
    config = load_frozen_config(output / "frozen_config.json")
    runtime = json.loads((output / "runtime.json").read_text())
    require(runtime["mode"] == "run" and runtime["models"]["ready_for_gpu"], "GPU worker was not authorized by the launcher")
    for name, identity in runtime["models"]["file_identities"].items():
        info = Path(runtime["models"]["paths"][name]).stat()
        require(identity == dict(device=info.st_dev, inode=info.st_ino,
                                 bytes=info.st_size, mtime_ns=info.st_mtime_ns),
                f"Verified model file changed before loading: {name}")
    records = json.loads((output / "model_input_manifest.json").read_text())
    inventory = json.loads((output / "artifacts_sha256.json").read_text())
    for record in records:
        for role in ("conditioning", "model_mask"):
            relative = str(relative_path(record[role]))
            require(sha256(output / relative) == inventory[relative], "Prepared model input was changed")
    conditions, masks = [], []
    for record in records:
        with Image.open(output / record["conditioning"]) as image:
            require(image.mode == "RGB" and image.size == (512, 512), "Invalid conditioning image")
            conditions.append(image.copy())
        with Image.open(output / record["model_mask"]) as image:
            require(image.mode == "RGB" and image.size == (512, 512), "Invalid model mask image")
            masks.append(image.copy())
    require([item["source_index"] for item in records] == list(range(5, 22)), "Model input frame order changed")
    record_stage(output, "condition_loader_check", "Checking actual saved conditioning and masks on CPU")
    write_json(output / "CPU_loader_contract.json", check_loader_contract(conditions, masks))
    record_stage(output, "condition_loader_check", "Actual saved input contract passed", completed=True)
    import torch
    import diffsynth
    from diffsynth.utils.data import save_video

    require(Path(diffsynth.__file__).resolve().is_relative_to(ROOT / "code/DiffSynth-Studio"),
            "Imported DiffSynth is not this project's vendor copy")
    require(importlib.metadata.version("diffsynth") == "2.1.5", "Unexpected DiffSynth version")
    environment = check_environment()
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    minimum_gib = 24 if runtime["memory_mode"] == "full" else 12
    require(free_bytes >= minimum_gib * 1024 ** 3,
            f"Insufficient free memory for {runtime['memory_mode']}; no automatic fallback or shape change")
    torch.cuda.reset_peak_memory_stats(0)
    environment.update(initial_free_bytes=free_bytes, total_bytes=total_bytes, memory_mode=runtime["memory_mode"])
    write_json(output / "environment.json", environment)
    record_stage(output, "worker_preflight", "CUDA, free memory, process occupancy and BF16 passed", completed=True)
    paths = runtime["models"]["paths"]
    record_stage(output, "model_loading", "Loading original VACE DiT + VACE branch, T5, VAE and tokenizer; no LoRA")
    pipe = load_local_vace(paths, runtime["memory_mode"])
    require(pipe.check_resize_height_width(512, 512, 17, verbose=0) == (512, 512, 17), "Pipeline changed shape")
    record_stage(output, "model_loading", "All four model branches initialized from verified local files", completed=True)
    arguments = inference_arguments(config, conditions, masks)
    arguments["progress_bar_cmd"] = inference_progress(output, total=config["inference"]["num_inference_steps"])
    record_stage(output, "conditioning_encoding", "Entering real pipeline: text and VACE conditioning encoding")
    with vae_stages(output, pipe.vae):
        frames = pipe(**arguments)
    require(len(frames) == 17 and all(image.size == (512, 512) for image in frames), "Unexpected prediction shape")
    record_stage(output, "saving_predictions", "Saving raw model PNGs and MP4; no GT read by worker")
    raw = output / "raw_prediction"
    raw.mkdir()
    for index, image in enumerate(frames):
        image.save(raw / f"{index:03}.png")
    write_json(output / "prediction_raw_sha256.json", {path.name: sha256(path) for path in sorted(raw.iterdir())})
    save_video(frames, str(output / "raw_prediction.mp4"), fps=config["inference"]["fps"], quality=9)
    (output / "prediction_raw").symlink_to("raw_prediction", target_is_directory=True)
    (output / "prediction_raw.mp4").symlink_to("raw_prediction.mp4")
    if runtime["composite"]:
        directory = output / "postprocessed_prediction"
        directory.mkdir()
        composited_frames = []
        for index, (frame, mask) in enumerate(zip(frames, masks)):
            with Image.open(output / "inputs/damaged" / f"{index:03}.png") as damaged:
                composited = composite_frame(frame, damaged, mask)
                composited.save(directory / f"{index:03}.png")
                composited_frames.append(composited)
        save_video(composited_frames, str(output / "postprocessed_prediction.mp4"), fps=config["inference"]["fps"], quality=9)
    torch.cuda.synchronize()
    write_json(output / "peak_memory.json", dict(allocated_bytes=torch.cuda.max_memory_allocated(0),
               reserved_bytes=torch.cuda.max_memory_reserved(0)))
    write_json(output / "worker_complete.json", dict(status="complete", prediction_frames=17,
               loaded_branches=["dit", "vace", "text_encoder", "vae"], lora_loaded=False,
               GT_loaded_by_worker=False, composite_saved=runtime["composite"]))
    record_stage(output, "saving_predictions", "Raw PNGs and MP4 saved; parent will validate decoding and evaluate", completed=True)


def run_gpu(output, config, selected, model_report, args, reserved=False):
    from vace_p01_console import record_stage, run_visible_process

    require(model_report["ready_for_gpu"], "Model weights are not verified; GPU run refused")
    with gpu_lock() as lock_stream:
        check_old_training()
        if reserved:
            record_stage(output, "preparing_inputs", "Saving aligned inputs, independent ModelMask and GT evaluation copies")
        prepare_output(output, config, selected, model_report, args.config, reserved=reserved)
        record_stage(output, "preparing_inputs", "17 aligned input sets saved; GT excluded from worker inputs", completed=True)
        runtime = dict(mode="run", memory_mode=args.memory_mode, composite=args.composite,
                       models=model_report, frozen_config_sha256=FROZEN_SHA256,
                       project_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                       source_files_sha256={name: sha256(ROOT / name) for name in (
                           "scripts/vace_p01.py", "configs/vace_p01_model_sources_v1.json",
                           "scripts/vace_p01_console.py", "scripts/vace_p01_results.py", "scripts/vace_p01_environment.py",
                           "code/DiffSynth-Studio/diffsynth/pipelines/wan_video.py",
                           "code/DiffSynth-Studio/diffsynth/models/model_loader.py")},
                       project_git_status=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
                       model_loading=dict(torch_dtype="bfloat16", device="cuda", skip_download=True,
                                          redirect_common_files=False, lora_checkpoint=None, memory_mode=args.memory_mode))
        write_json(output / "runtime.json", runtime)
        environment = os.environ.copy()
        environment.update(DIFFSYNTH_SKIP_DOWNLOAD="True", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                           WAN_VACE_LOCK_FD=str(lock_stream.fileno()), PYTHONDONTWRITEBYTECODE="1")
        command = [sys.executable, "-u", "-B", str(Path(__file__).resolve()), "--_worker", str(output)]
        started = time.monotonic()
        returncode, interrupted = run_visible_process(command, environment, lock_stream.fileno(), output)
        complete = returncode == 0 and not interrupted and (output / "worker_complete.json").is_file()
        write_json(output / "exit_status.json", dict(status="model_completed" if complete else "failed",
                   worker_returncode=returncode, worker_signal=-returncode if returncode < 0 else None,
                   interrupted=interrupted, seconds=time.monotonic() - started,
                   completion_marker_present=(output / "worker_complete.json").exists()))
        return 0 if complete else (128 - returncode if returncode < 0 else returncode or (130 if interrupted else 1))


def replace_run_json(path, value):
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def load_run_json(output, name):
    path = output / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def finish_run(output, status, exit_code, started_at, elapsed, results=None, error=None):
    runtime = load_run_json(output, "runtime.json")
    stage = load_run_json(output, "stage_state.json")
    worker = load_run_json(output, "exit_status.json")
    known_paths = dict(output_directory=output, run_log=output / "run.log", summary=output / "result_summary.md",
                       manifest=output / "run_manifest.json", raw_prediction=output / "raw_prediction",
                       raw_prediction_video=output / "raw_prediction.mp4",
                       before_after_image=output / "comparison_before_after_B020.png",
                       multi_frame_comparison=output / "comparison_before_after_grid.png",
                       four_column_comparison=output / "comparison_4col_B020.png",
                       comparison_video=output / "comparison_before_after.mp4", metrics=output / "metrics.json",
                       video_validation=output / "results_validation.json", worker_log=output / "worker.log",
                       peak_memory=output / "peak_memory.json", failure_peak_memory=output / "peak_memory_failure.json",
                       postprocessed_video=output / "postprocessed_prediction.mp4")
    paths = {name: str(path.resolve()) for name, path in known_paths.items() if path.exists()}
    ended_at = datetime.now(timezone.utc).isoformat()
    manifest = dict(status=status, exit_code=exit_code, started_at_utc=started_at, ended_at_utc=ended_at,
                    elapsed_seconds=elapsed, last_stage=stage, last_completed_stage=stage.get("last_completed_stage"),
                    models=runtime.get("models"), model_loading=runtime.get("model_loading"),
                    source_files_sha256=runtime.get("source_files_sha256"), project_head=runtime.get("project_head"),
                    software_environment=load_run_json(output, "environment.json"),
                    preparation=load_run_json(output, "prepare_report.json"),
                    frozen_config_sha256=FROZEN_SHA256, actual_worker_exit=worker, paths=paths,
                    results=results, error=error, oracle_mask=True, GT_model_input=False,
                    quality_accepted=False, quality_note="Execution success is not a completion-quality or 3D reconstruction acceptance.")
    summary = ["# VACE P01 17-frame experiment", "", f"Status: {status}",
               f"Exit code: {exit_code}", f"Start (UTC): {started_at}", f"End (UTC): {ended_at}",
               f"Elapsed seconds: {elapsed:.3f}", f"Last completed stage: {stage.get('last_completed_stage')}", "",
               "Frozen target: source 465, B005–B021, 17 frames, 512x512, 16 FPS, seed 20261018, 30 steps, CFG 5, no LoRA.",
               "Exact-derived oracle mask; GT RGB is evaluation-only. Raw predictions are not replaced by GT or compositing.",
               "Execution success does not certify completion quality or 3D reconstruction; inspect the comparison panels and metrics.", ""]
    checkpoint_sha = runtime.get("models", {}).get("files_sha256", {}).get("checkpoint")
    if checkpoint_sha:
        summary.extend([f"Actual verified VACE SHA256: `{checkpoint_sha}`", ""])
    if error:
        summary.extend(["## Failure", "", str(error), "", "No automatic retry, fallback or parameter change was performed.", ""])
    summary.extend(["## Existing artifacts", ""] + [f"- {name}: `{value}`" for name, value in paths.items()])
    summary.extend([f"- summary: `{(output / 'result_summary.md').resolve()}`",
                    f"- manifest: `{(output / 'run_manifest.json').resolve()}`", ""])
    (output / "result_summary.md").write_text("\n".join(summary), encoding="utf-8")
    paths.update(summary=str((output / "result_summary.md").resolve()), manifest=str((output / "run_manifest.json").resolve()))
    replace_run_json(output / "run_manifest.json", manifest)
    worker.update(status="complete" if status == "SUCCESS" else "failed", launcher_exit_code=exit_code,
                  final_stage=stage, all_required_results_validated=status == "SUCCESS")
    replace_run_json(output / "exit_status.json", worker)
    if status == "SUCCESS":
        from vace_p01_console import record_stage

        record_stage(output, "report_generation", "Summary, manifest and final exit status saved", completed=True)
        stage = load_run_json(output, "stage_state.json")
        manifest.update(last_stage=stage, last_completed_stage=stage.get("last_completed_stage"))
        worker["final_stage"] = stage
        summary = [f"Last completed stage: {stage.get('last_completed_stage')}"
                   if line.startswith("Last completed stage:") else line for line in summary]
        (output / "result_summary.md").write_text("\n".join(summary), encoding="utf-8")
        replace_run_json(output / "run_manifest.json", manifest)
        replace_run_json(output / "exit_status.json", worker)
    print("\n" + "=" * 64, flush=True)
    print(f"VACE P01 17-frame inference: {status}", flush=True)
    print(f"Exit code: {exit_code} | Elapsed: {elapsed:.1f}s", flush=True)
    if status == "SUCCESS":
        print("Frames: 17 | Resolution: 512x512 | FPS: 16 | Inference steps: 30", flush=True)
    if error:
        print(f"Failure reason: {error}", flush=True)
    for name, value in paths.items():
        print(f"{name.upper()}:\n{value}", flush=True)
    print("=" * 64, flush=True)
    return exit_code


def finalization_failure(output, error):
    message = f"Finalization failed: {type(error).__name__}: {error}"
    print(f"\nVACE P01 17-frame inference: FAILED\n{message}", file=sys.stderr, flush=True)
    if (output / "run.log").exists():
        print(f"RUN LOG:\n{(output / 'run.log').resolve()}", file=sys.stderr, flush=True)
    for name in ("run_manifest.json", "exit_status.json"):
        try:
            previous = load_run_json(output, name)
            previous.update(status="FAILED" if name == "run_manifest.json" else "failed", launcher_exit_code=1, exit_code=1,
                            finalization_error=message, all_required_results_validated=False,
                            final_stage=load_run_json(output, "stage_state.json"),
                            ended_at_utc=datetime.now(timezone.utc).isoformat(),
                            output_directory=str(output), run_log=str(output / "run.log"))
            replace_run_json(output / name, previous)
        except Exception as save_error:
            print(f"Could not persist {name}: {save_error}", file=sys.stderr, flush=True)
    try:
        (output / "result_summary.md").write_text(
            f"# VACE P01 experiment\n\nStatus: FAILED\n\n{message}\n\nRun log: `{output / 'run.log'}`\n", encoding="utf-8")
    except Exception as save_error:
        print(f"Could not persist failure summary: {save_error}", file=sys.stderr, flush=True)
    return 1


def launch_run(args):
    from vace_p01_console import record_stage, run_log
    from vace_p01_results import generate_results

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    args._run_directory_owned = True
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    with run_log(output):
        returncode, results, error = 1, None, None
        try:
            print(f"OUTPUT DIRECTORY:\n{output}\nOne real 17-frame run only; no automatic retry or download.", flush=True)
            record_stage(output, "environment_and_input_preflight", "Checking output space, frozen configuration and original ZIP")
            space = shutil.disk_usage(output)
            require(space.free >= 2 * 1024 ** 3, "Less than 2 GiB free on output volume")
            config = load_frozen_config(args.config)
            selected = read_inputs(config)
            write_json(output / "run_config.json", dict(frozen_experiment=config, frozen_config_sha256=FROZEN_SHA256,
                       checkpoint_override=str(args.checkpoint) if args.checkpoint else None, memory_mode=args.memory_mode,
                       composite=args.composite, model_sources=json.loads(MODEL_SOURCES.read_text()),
                       automatic_download=False, automatic_retry=False, output_directory=str(output),
                       input_source_indices=[item["source_index"] for item in selected],
                       output_volume_free_bytes=space.free))
            record_stage(output, "environment_and_input_preflight", "17 aligned frames, unchanged frozen JSON and source ZIP passed", completed=True)
            record_stage(output, "weight_verification", "Fully reading all local model hashes; this can take several minutes")
            models = inspect_models(config, args.checkpoint, verify=True, notify=lambda message: print(message, flush=True))
            require(models["ready_for_gpu"], "Required original model components are missing or not verified")
            write_json(output / "verified_models.json", models)
            record_stage(output, "weight_verification", "All official pinned model hashes and local loading structures passed", completed=True)
            returncode = run_gpu(output, config, selected, models, args, reserved=True)
            require(returncode == 0, f"GPU worker failed with exit code {returncode}; see live log and worker.log")
            record_stage(output, "evaluation_and_comparisons", "Validating video decoding and creating Before/After panels; GT used only now")
            results = generate_results(output)
            record_stage(output, "evaluation_and_comparisons", "Raw video and 17 frames validated; comparisons and diagnostic metrics saved", completed=True)
            record_stage(output, "report_generation", "Writing final result manifest, status and actual absolute artifact paths")
        except KeyboardInterrupt:
            returncode, error = 130, "Interrupted by user; no automatic retry"
            traceback.print_exc()
        except Exception as exc:
            returncode = returncode or 1
            error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        status = "SUCCESS" if error is None and returncode == 0 and results is not None else "FAILED"
        try:
            return finish_run(output, status, returncode, started_at, time.monotonic() - started, results, error)
        except Exception as exc:
            traceback.print_exc()
            return finalization_failure(output, exc)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Read-only inputs/model preflight (default)")
    mode.add_argument("--prepare", action="store_true", help="Save CPU-prepared inputs in a NEW output directory")
    mode.add_argument("--run", action="store_true", help="Explicitly authorize this command to load models and run GPU inference")
    parser.add_argument("--config", type=Path, default=FROZEN_CONFIG)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verify-models", action="store_true", help="Read full local model SHA256 (always done for --run)")
    parser.add_argument("--check-loader", action="store_true", help="Also check the real DiffSynth input loader on CPU; no model weights")
    parser.add_argument("--memory-mode", choices=("full", "cpu-offload"), default="full")
    parser.add_argument("--composite", action="store_true", help="Also save a separately named postprocessed result for --run")
    parser.add_argument("--_worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args._worker is not None:
            gpu_worker(args._worker.resolve())
            return 0
        require(not args.composite or args.run, "--composite requires --run")
        require(not (args.run or args.prepare) or args.output_dir is not None, "--prepare/--run requires --output-dir")
        if args.output_dir is not None:
            args.output_dir = args.output_dir.resolve()
            require(not args.output_dir.exists(), "Output directory already exists; refuse overwrite")
            allowed_root = (ROOT / "outputs/vace").resolve()
            require(args.output_dir.is_relative_to(allowed_root) and args.output_dir != allowed_root,
                    "Output directory must be a new child of project outputs/vace")
        if args.run:
            return launch_run(args)
        config = load_frozen_config(args.config)
        selected = read_inputs(config)
        models = inspect_models(config, args.checkpoint, verify=args.verify_models or args.run)
        summary = dict(status="preflight_complete", input_frames=len(selected), source_frames=list(range(5, 22)),
                       shape=[17, 512, 512], model_preflight=models, mode="prepare" if args.prepare else "run" if args.run else "dry-run",
                       frozen_config_sha256=FROZEN_SHA256, model_loaded=False, GPU_inference_started=False)
        if args.check_loader:
            summary["CPU_loader_contract"] = check_loader_contract(
                [item["conditioning"] for item in selected], [item["model_mask"] for item in selected])
        if args.prepare:
            prepare_output(args.output_dir, config, selected, models, args.config)
            summary.update(status="cpu_prepared", output=str(args.output_dir))
            if args.check_loader:
                write_json(args.output_dir / "CPU_loader_contract.json", summary["CPU_loader_contract"])
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if args.prepare or models["ready_for_gpu"] else 2
    except Exception as exc:
        if args._worker is not None:
            traceback.print_exc()
            torch_module = sys.modules.get("torch")
            if torch_module is not None and torch_module.cuda.is_initialized():
                memory_path = args._worker / "peak_memory_failure.json"
                if not memory_path.exists():
                    write_json(memory_path, dict(allocated_bytes=torch_module.cuda.max_memory_allocated(0),
                                                reserved_bytes=torch_module.cuda.max_memory_reserved(0)))
        else:
            print(json.dumps(dict(status="failed", error=str(exc)), ensure_ascii=False), file=sys.stderr)
            if args.run:
                if getattr(args, "_run_directory_owned", False):
                    return finalization_failure(args.output_dir, exc)
                print("VACE P01 17-frame inference: FAILED", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
