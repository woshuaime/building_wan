"""One foreground 81-frame VACE experiment, retaining only four deliverables."""

import argparse
import contextlib
import hashlib
import io
import json
import os
import shutil
import signal
import sys
import tempfile
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

import vace_p01 as baseline
from vace_p01_81f_console import ConsoleState, visible_worker
from vace_p01_81f_results import FINAL_FILENAMES, render_results, validate_final_results


ROOT = baseline.ROOT
FRAME_COUNT = 81
TEMP_PREFIX = "building_wan_vace81_"


class WorkerFailure(RuntimeError):
    def __init__(self, returncode, detail):
        super().__init__(detail)
        self.returncode = returncode


@contextlib.contextmanager
def cleanup_signals():
    previous = {}

    def interrupt(signum, frame):
        raise WorkerFailure(128 + signum, f"Parent received {signal.Signals(signum).name}")

    try:
        for signum in (signal.SIGTERM, signal.SIGHUP):
            previous[signum] = signal.signal(signum, interrupt)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def experiment_config():
    config = baseline.load_frozen_config()
    config["source"].update(frame_start_inclusive=0, frame_end_inclusive=80, frame_stride=1,
                            selection_policy="all_81_including_original_B080")
    config["inference"]["num_frames"] = FRAME_COUNT
    config["output_policy"] = dict(existing_directory="refuse", final_files=list(FINAL_FILENAMES),
                                   keep_intermediates=False, keep_logs=False)
    return config


def memory_inputs(config, project_root=ROOT):
    selected = baseline.read_inputs(config, project_root, include_gt=False)
    baseline.require([item["source_index"] for item in selected] == list(range(FRAME_COUNT)),
                     "Expected source B000 through B080 in original order")
    black_frames = [item["source_index"] for item in selected if not np.any(np.asarray(item["model_mask"]))]
    baseline.require(black_frames == list(range(46, 73)), "The original 27 all-black Exact frames changed")
    return selected


def read_gt_B020(config, project_root=ROOT):
    archive = project_root / config["source"]["archive"]
    baseline.require(baseline.sha256(archive) == config["source"]["archive_sha256"], "Source ZIP changed after inference")
    prefix = config["source"]["package_id"] + "/"
    relative = config["source"]["sample_relative_root"] + "/512/gt/020.png"
    with zipfile.ZipFile(archive) as bundle:
        manifest = json.loads(bundle.read(prefix + "manifest.json"))
        baseline.require(manifest["frames"][20]["gt"] == relative, "GT B020 frame mapping mismatch")
        expected = next(item["sha256"] for item in manifest["files"] if item["relative_path"] == relative)
        content = bundle.read(prefix + relative)
    baseline.require(hashlib.sha256(content).hexdigest() == expected, "GT B020 SHA256 mismatch")
    with Image.open(io.BytesIO(content)) as image:
        image.load()
        baseline.require(image.format == "PNG" and image.mode == "RGB" and image.size == (512, 512), "Invalid GT B020")
        return image.copy()


def check_output_volume():
    destination = ROOT / "outputs/vace"
    existing = destination
    while not existing.exists():
        existing = existing.parent
    baseline.require(existing.is_dir() and os.access(existing, os.W_OK | os.X_OK), "Output parent is not writable")
    free_bytes = shutil.disk_usage(existing).free
    baseline.require(free_bytes >= 2 * 1024 ** 3, "Output volume has less than 2 GiB available")
    baseline.require(shutil.disk_usage("/tmp").free >= 1024 ** 3, "System temporary volume has less than 1 GiB available")
    return free_bytes


def preflight(state):
    from vace_p01_environment import check_environment

    config = experiment_config()
    state.stage("preflight", "Checking source data and output volume; no files will be saved")
    free_bytes = check_output_volume()
    selected = memory_inputs(config)
    print("PASS: B000–B080 aligned; original 27 black masks retained; no GT RGB read", flush=True)
    state.stage("weight_verification", "Reading full local SHA256; no downloads")
    models = baseline.inspect_models(config, verify=True)
    baseline.require(models["ready_for_gpu"], "Required original VACE components are missing")
    print(f"PASS: original VACE-1.3B SHA256 {models['files_sha256']['checkpoint']}; T5/VAE/tokenizer verified", flush=True)
    state.stage("environment_check", "Checking A6000, other tasks, BF16 and video dependencies")
    environment = check_environment()
    baseline.require(environment["cuda"]["free_bytes"] >= 24 * 1024 ** 3,
                     "Less than 24 GiB free VRAM; refusing without changing the experiment")
    import torch

    baseline.require(torch.backends.cuda.flash_sdp_enabled() or torch.backends.cuda.mem_efficient_sdp_enabled(),
                     "81-frame attention would be math-only; enable the existing efficient backend explicitly before retrying")
    print(f"PASS: {environment['cuda']['device_name']}; free VRAM {environment['cuda']['free_bytes'] / 1024 ** 3:.2f} GiB; no other GPU tasks", flush=True)
    state.stage("input_loader_check", "Checking all 81 conditions and masks with the current CPU loader")
    baseline.check_loader_contract([item["conditioning"] for item in selected],
                                   [item["model_mask"] for item in selected], num_frames=FRAME_COUNT)
    baseline.check_old_training()
    print(f"PASS: 81x512x512/16FPS; output volume free {free_bytes / 1024 ** 3:.1f} GiB; video tools available", flush=True)
    print("PASS: full-sequence length supported (21 latent frames, 21504 tokens); peak VRAM is not yet measured", flush=True)
    state.stage("preflight", "All input, model, CPU interface and environment checks passed", completed=True)
    return config, selected, models


def execute_sample(config, selected, models, state, destination, gt_loader):
    import torch

    baseline.require(len(selected) == FRAME_COUNT
                     and [item["source_index"] for item in selected] == list(range(FRAME_COUNT)),
                     "One aligned full 81-frame sample is required")
    baseline.require(all("gt" not in item["original_bytes"] for item in selected), "GT RGB must not enter model inputs")
    state.stage("model_loading", "Loading the original VACE, DiT, T5 and VAE; no LoRA or offload fallback")
    torch.cuda.reset_peak_memory_stats(0)
    pipe = baseline.load_local_vace(models["paths"], memory_mode="full")
    baseline.require(pipe.check_resize_height_width(512, 512, FRAME_COUNT, verbose=0) == (512, 512, FRAME_COUNT),
                     "Pipeline attempted to change the frozen resolution or length")
    state.stage("model_loading", "Verified local model branches initialized", completed=True)
    arguments = baseline.inference_arguments(config, [item["conditioning"] for item in selected],
                                              [item["model_mask"] for item in selected])
    arguments["progress_bar_cmd"] = state.progress
    state.stage("conditioning", "Entering one full 81-frame VACE pipeline call, including B080")
    with state.vae_stages(pipe.vae):
        prediction = pipe(**arguments)
    baseline.require(len(prediction) == FRAME_COUNT, "The model did not return exactly 81 predictions")
    torch.cuda.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated(0)
    peak_reserved = torch.cuda.max_memory_reserved(0)
    state.stage("rendering", "Inference finished; now read GT B020 only for the four-column image")
    damaged = []
    for item in selected:
        with Image.open(io.BytesIO(item["original_bytes"]["damaged"])) as image:
            damaged.append(image.copy())
    render_results(destination, damaged, [item["model_mask"] for item in selected], prediction, gt_loader())
    state.stage("validation", "Two videos decode to 81 frames and both PNGs pass; exactly four files", completed=True)
    return dict(peak_allocated_bytes=peak_allocated, peak_reserved_bytes=peak_reserved)


def gpu_worker(temporary, state):
    lock_fd = int(os.environ["WAN_VACE81_LOCK_FD"])
    lock_path = ROOT / "logs/.wan_a6000_gpu.lock"
    baseline.require(os.fstat(lock_fd).st_ino == lock_path.stat().st_ino, "Missing inherited GPU lock")
    baseline.require(temporary.is_dir() and not temporary.is_symlink()
                     and temporary.parent == Path("/tmp") and temporary.name.startswith(TEMP_PREFIX),
                     "Worker requires the parent's private system temporary directory")
    status_fd = int(os.environ["WAN_VACE81_STATUS_FD"])

    def send_event(event):
        content = (json.dumps(event, ensure_ascii=True) + "\n").encode()
        baseline.require(len(content) < 4096, "Runtime event exceeds the anonymous pipe limit")
        baseline.require(os.write(status_fd, content) == len(content), "Runtime event pipe write failed")

    state.event_sink = send_event
    try:
        config, selected, models = preflight(state)
        peaks = execute_sample(config, selected, models, state, temporary / "deliverables", lambda: read_gt_B020(config))
        send_event(dict(worker_complete=True, **peaks,
                        stage=state.last_stage, last_completed_stage=state.last_completed_stage))
        return 0
    except BaseException as error:
        try:
            send_event(dict(worker_error=f"{type(error).__name__}: {str(error)[:400]}", stage=state.last_stage,
                            last_completed_stage=state.last_completed_stage))
        except (OSError, ValueError) as status_error:
            print(f"Runtime status pipe unavailable: {status_error}", file=sys.stderr, flush=True)
        print(f"VACE 81F worker FAILED; stopped stage: {state.last_stage}", file=sys.stderr, flush=True)
        raise


def publish_results(staged, destination):
    baseline.require(not destination.exists() and not destination.is_symlink(), "Final output directory already exists")
    baseline.require(sorted(path.name for path in staged.iterdir()) == sorted(FINAL_FILENAMES),
                     "Private staging directory must contain only the four deliverables")
    baseline.require(all((staged / name).is_file() and not (staged / name).is_symlink() for name in FINAL_FILENAMES),
                     "A staged deliverable is not a regular file")
    destination.mkdir(parents=True, exist_ok=False)
    for name in FINAL_FILENAMES:
        source = staged / name
        baseline.require(source.is_file() and not source.is_symlink(), "A staged deliverable is not a regular file")
        with source.open("rb") as reader, (destination / name).open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=1024 ** 2)
    validate_final_results(destination)
    return destination


def run_once(state):
    check_output_volume()
    experiment_config()
    final = ROOT / "outputs/vace" / f"P01_465_81f_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{os.getpid()}"
    events = {}

    def receive_event(event):
        events.update(event)
        state.last_stage = event.get("stage", state.last_stage)
        state.last_completed_stage = event.get("last_completed_stage", state.last_completed_stage)

    with cleanup_signals(), baseline.gpu_lock() as lock_stream:
        baseline.check_old_training()
        with tempfile.TemporaryDirectory(prefix=TEMP_PREFIX, dir="/tmp") as temporary_name:
            temporary = Path(temporary_name)
            state.stage("worker_start", "Foreground model process; terminal output is inherited, nothing is logged to disk")
            environment = os.environ.copy()
            environment.update(DIFFSYNTH_SKIP_DOWNLOAD="True", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                               PYTHONDONTWRITEBYTECODE="1", WAN_VACE81_LOCK_FD=str(lock_stream.fileno()))
            command = [sys.executable, "-u", "-B", str(Path(__file__).resolve()), "--_worker", str(temporary)]
            returncode, interrupted = visible_worker(command, environment, lock_stream.fileno(), receive_event)
            if returncode != 0 or interrupted or not events.get("worker_complete"):
                code = 128 - returncode if returncode < 0 else returncode or (130 if interrupted else 1)
                detail = f"worker returncode={returncode}, interrupted={interrupted}, completion={bool(events.get('worker_complete'))}"
                if events.get("worker_error"):
                    detail += f"; {events['worker_error']}"
                raise WorkerFailure(code, detail)
            baseline.require(isinstance(events.get("peak_allocated_bytes"), int) and events["peak_allocated_bytes"] > 0,
                             "Missing actual GPU peak memory measurement")
            baseline.require(isinstance(events.get("peak_reserved_bytes"), int)
                             and events["peak_reserved_bytes"] >= events["peak_allocated_bytes"], "Invalid GPU peak memory measurement")
            state.stage("publishing", f"Copying only four final files to {final}; no intermediate directory under outputs")
            publish_results(temporary / "deliverables", final)
            state.stage("publishing", "Final output contains exactly four valid files", completed=True)
    print(f"\nVACE 81F: SUCCESS\n\nFrames: 81\nResolution: 512x512\nElapsed: {state.elapsed:.1f}s\n"
          f"Peak VRAM: {events['peak_allocated_bytes'] / 1024 ** 3:.2f} GiB allocated / "
          f"{events['peak_reserved_bytes'] / 1024 ** 3:.2f} GiB reserved\n\n输出目录：{final.resolve()}\n", flush=True)
    print("\n".join(FINAL_FILENAMES), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Read-only preflight; never load model weights or write outputs")
    mode.add_argument("--run", action="store_true", help="Authorize one full 81-frame GPU experiment from this terminal")
    mode.add_argument("--_worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    state = ConsoleState(total_steps=30)
    try:
        if args._worker is not None:
            return gpu_worker(args._worker, state)
        if args.run:
            return run_once(state)
        preflight(state)
        print("PASS: 81F preparation only; no GPU model loaded, no inference, no output files created", flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        code = error.returncode if isinstance(error, WorkerFailure) else 130 if isinstance(error, KeyboardInterrupt) else 1
        print(f"\nVACE 81F: FAILED\nStopped stage: {state.last_stage}\nReason: {error}\nExit code: {code}", file=sys.stderr, flush=True)
        if args._worker is not None:
            traceback.print_exc()
        return code


if __name__ == "__main__":
    sys.exit(main())
