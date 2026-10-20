"""Visible, single-GPU P04/P06 LoRA smoke test; CPU preparation is the default."""

import argparse
import csv
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

import vace_p01 as baseline
import vace_roof_data as data
from vace_p01_81f_console import ConsoleState, visible_worker


STAGES = ("base", "train", "lora100", "lora300")


class InterruptedRun(KeyboardInterrupt):
    def __init__(self, signum):
        self.exit_code = 128 + signum
        super().__init__(f"Interrupted by {signal.Signals(signum).name}")


def stop_on_signal(signum, frame):
    raise InterruptedRun(signum)


def read_experiment(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_experiment(path, experiment):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".experiment_", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(experiment, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def patch_runtime(path, **updates):
    experiment = read_experiment(path)
    experiment["runtime"].update(updates)
    write_experiment(path, experiment)


def completed_updates(training_dir):
    path = training_dir / "loss.csv"
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    updates = [int(row["update"]) for row in rows]
    baseline.require(updates == list(range(1, len(updates) + 1)), "loss.csv update sequence is incomplete")
    return len(updates)


def inference_arguments(config, sample):
    baseline.require("gt" not in sample, "Inference loader must not read GT")
    arguments = baseline.inference_arguments(config, sample["conditioning"], sample["model_mask"])
    baseline.require(not any(arguments.get(key) is not None for key in
                             ("input_video", "input_image", "end_image", "reference_image", "vace_reference_image")),
                     "Inference must not receive a GT or image/video answer condition")
    return arguments


def load_stage_lora(pipe, path, config):
    import torch
    from safetensors.torch import load_file

    state = load_file(str(path), device="cpu")
    expected = {}
    targets = config["training"]["lora_target_modules"]
    rank = config["training"]["lora_rank"]
    for name, module in pipe.vace.named_modules():
        if isinstance(module, torch.nn.Linear) and any(name.endswith("." + target) for target in targets):
            expected[name + ".lora_A.default.weight"] = (rank, module.in_features)
            expected[name + ".lora_B.default.weight"] = (module.out_features, rank)
    baseline.require(expected and set(state) == set(expected), "LoRA keys must exactly match the VACE target layers")
    for name, tensor in state.items():
        baseline.require(tuple(tensor.shape) == expected[name] and torch.isfinite(tensor).all().item(),
                         f"Invalid VACE LoRA tensor: {name}")
    pipe.load_lora(pipe.vace, state_dict=state, alpha=config["evaluation"]["lora_multiplier"], hotload=False)
    print(f"Loaded {path.name} into vace only, multiplier=1.0, fresh original backbone", flush=True)


def worker(stage, experiment_path, temporary):
    import torch
    from vace_p01_environment import check_environment

    experiment = read_experiment(experiment_path)
    config = experiment["config"]
    descriptor = int(os.environ["WAN_VACE_ROOF_LOCK_FD"])
    baseline.require(os.fstat(descriptor).st_ino == (data.ROOT / "logs/.wan_a6000_gpu.lock").stat().st_ino,
                     "Missing parent GPU lock")
    environment = check_environment()
    baseline.require("A6000" in environment["cuda"]["device_name"]
                     and environment["cuda"]["free_bytes"] >= 24 * 1024 ** 3,
                     "A6000 with at least 24 GiB free required; no silent fallback")
    baseline.require(torch.backends.cuda.flash_sdp_enabled() or torch.backends.cuda.mem_efficient_sdp_enabled(),
                     "Existing efficient SDPA must be enabled")
    torch.cuda.reset_peak_memory_stats(0)
    paths = data.inspect_local_models(config)
    stage_started = time.monotonic()
    print(f"\nSTAGE: {stage}; free VRAM {environment['cuda']['free_bytes'] / 1024 ** 3:.2f} GiB", flush=True)
    if stage == "train":
        from vace_roof_training import run_training

        samples = data.load_samples(config, include_gt=True)

        def on_update(update, loss, sample):
            if update % 10 == 0 or update == 1:
                patch_runtime(experiment_path, actual_updates=update, last_training_sample=sample,
                              last_loss=float(loss))

        result = run_training(config, samples, experiment_path.parent, paths, on_update=on_update)
        baseline.require(result["actual_updates"] == 300, "Training did not complete 300 actual updates")
        patch_runtime(experiment_path, training_result=result, actual_updates=300)
    else:
        samples = data.load_samples(config, include_gt=False)
        pipe = baseline.load_local_vace(paths, memory_mode="full")
        pipe.eval()
        if stage != "base":
            update = 100 if stage == "lora100" else 300
            load_stage_lora(pipe, experiment_path.parent / f"lora_update{update:04}.safetensors", config)
        for sample in samples:
            state = ConsoleState(total_steps=30)
            state.stage(stage, f"{sample['pilot']} B{sample['source_indices'][0]:03}-"
                        f"B{sample['source_indices'][-1]:03}; GT excluded")
            arguments = inference_arguments(config, sample)
            arguments["progress_bar_cmd"] = state.progress
            runtime = read_experiment(experiment_path)["runtime"]
            calls = runtime.get("inference_calls_started", 0) + 1
            baseline.require(calls <= 6, "Six-call inference budget exhausted")
            patch_runtime(experiment_path, inference_calls_started=calls, current_sample=sample["pilot"])
            with state.vae_stages(pipe.vae):
                prediction = pipe(**arguments)
            baseline.require(len(prediction) == 17
                             and all(image.mode == "RGB" and image.size == (512, 512) for image in prediction),
                             "Unexpected model output dimensions")
            with (temporary / f"{stage}_{sample['number']}.npy").open("xb") as stream:
                np.save(stream, np.stack([np.asarray(image) for image in prediction]), allow_pickle=False)
            completed = read_experiment(experiment_path)["runtime"].get("inference_calls_completed", 0) + 1
            patch_runtime(experiment_path, inference_calls_completed=completed)
            print(f"{stage} {sample['pilot']}: 17 real predictions saved to this run's temporary area", flush=True)
    torch.cuda.synchronize()
    experiment = read_experiment(experiment_path)
    experiment["runtime"]["stages"][stage] = dict(
        status="worker_finished_pending_exit", elapsed_seconds=time.monotonic() - stage_started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
    )
    write_experiment(experiment_path, experiment)


def code_version():
    names = ("vace_roof_smoke.py", "vace_roof_training.py", "vace_roof_data.py",
             "vace_roof_results.py", "vace_roof_cpu_check.py", "vace_p01.py")
    vendor = ("pipelines/wan_video.py", "diffusion/loss.py", "models/wan_video_vace.py",
              "diffusion/training_module.py", "diffusion/base_pipeline.py")
    return dict(project_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=data.ROOT, text=True).strip(),
                diffsynth_source="vendored in project; no independent Git revision",
                source_sha256={name: baseline.sha256(data.ROOT / "scripts" / name) for name in names},
                vendor_source_sha256={name: baseline.sha256(data.ROOT / "code/DiffSynth-Studio/diffsynth" / name)
                                      for name in vendor},
                versions={name: importlib.metadata.version(name)
                          for name in ("torch", "diffsynth", "peft", "safetensors")})


def execute(config):
    samples, _, _ = data.prepare(config)
    started = time.monotonic()
    with baseline.gpu_lock() as lock:
        baseline.check_old_training()
        root = data.ROOT / config["evaluation"]["output_root"]
        root.mkdir(parents=True, exist_ok=True)
        output = root / f"run_{datetime.now():%Y%m%d_%H%M%S_%f}_{os.getpid()}"
        training = output / "training"
        training.mkdir(parents=True)
        experiment_path = training / "experiment.json"
        experiment = dict(config=config, code_version=code_version(),
                          runtime=dict(status="running", stage="base", actual_updates=0,
                                       inference_calls_started=0, inference_calls_completed=0,
                                       stages={}, output=str(output)))
        write_experiment(experiment_path, experiment)
        print(f"\nOUTPUT: {output}\nBase (2) -> 300 optimizer updates -> LoRA100 (2) -> LoRA300 (2)", flush=True)
        stage = "base"
        exit_code = 1
        try:
            with tempfile.TemporaryDirectory(prefix="vace_roof_smoke_") as temporary_name:
                temporary = Path(temporary_name)
                for stage in STAGES:
                    patch_runtime(experiment_path, stage=stage)
                    environment = os.environ.copy()
                    environment.update(WAN_VACE_ROOF_LOCK_FD=str(lock.fileno()),
                                       DIFFSYNTH_SKIP_DOWNLOAD="True", HF_HUB_OFFLINE="1",
                                       TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1",
                                       PYTHONDONTWRITEBYTECODE="1")
                    command = [sys.executable, "-u", "-B", str(Path(__file__).resolve()), "--_worker", stage,
                               "--experiment", str(experiment_path), "--temporary", str(temporary)]
                    exit_code, interrupted = visible_worker(command, environment, lock.fileno())
                    actual = completed_updates(training)
                    experiment = read_experiment(experiment_path)
                    experiment["runtime"]["actual_updates"] = actual
                    stage_result = experiment["runtime"]["stages"].setdefault(stage, {})
                    if exit_code == 0 and not interrupted:
                        baseline.require(stage_result.get("status") == "worker_finished_pending_exit",
                                         f"{stage}: process exited without its completion marker")
                    stage_result.update(exit_code=exit_code, status="complete" if exit_code == 0 and not interrupted else "failed")
                    write_experiment(experiment_path, experiment)
                    if interrupted or exit_code != 0:
                        raise RuntimeError(f"{stage} exited with code {exit_code}; no retry or fallback")
                    if stage == "train":
                        baseline.require(actual == 300 and all((training / f"lora_update{update:04}.safetensors").is_file()
                                                              for update in (100, 300)),
                                         "Training exited without the two checkpoints and 300 loss records")
                stage = "render"
                patch_runtime(experiment_path, stage=stage)
                from vace_roof_results import render_sample

                for sample in samples:
                    predictions = {name: np.load(temporary / f"{name}_{sample['number']}.npy", allow_pickle=False)
                                   for name in ("base", "lora100", "lora300")}
                    summary = render_sample(output / sample["number"], sample["source_indices"], sample["damaged"],
                                            sample["model_mask"], sample["gt"], predictions["base"],
                                            predictions["lora100"], predictions["lora300"])
                    print(f"\n{sample['pilot']}: training-set comparison produced; structural closure/flicker "
                          f"requires visual inspection.\nRGB-only reference: {summary['masked_mae_0_255']}"
                          f"\nKnown-region change: {summary['unmasked_change_mae_0_255']}"
                          f"\n{output / sample['number'] / '02_缺损_原模型_微调后_GT_对比.mp4'}", flush=True)
            baseline.require({path.name for path in output.iterdir()} == {"training", "04", "06"},
                             "Unexpected run output entries")
            baseline.require({path.name for path in training.iterdir()} == {
                "experiment.json", "loss.csv", "lora_update0100.safetensors", "lora_update0300.safetensors"},
                "Unexpected training output entries")
            patch_runtime(experiment_path, status="complete", stage="complete",
                          elapsed_seconds=time.monotonic() - started, visual_assessment="pending_user_review")
            peaks = [result.get("peak_allocated_bytes", 0)
                     for result in read_experiment(experiment_path)["runtime"]["stages"].values()]
            print(f"\nVACE ROOF SMOKE: COMPLETED\nActual optimizer updates: 300 (P04:150, P06:150)"
                  f"\n17-frame inference calls: 6; 81-frame reruns: 0"
                  f"\nPeak allocated VRAM: {max(peaks) / 1024 ** 3:.2f} GiB"
                  f"\nElapsed: {time.monotonic() - started:.1f}s"
                  f"\nWeights:\n{training / 'lora_update0100.safetensors'}"
                  f"\n{training / 'lora_update0300.safetensors'}"
                  f"\nOUTPUT: {output}\nTraining-set reconstruction only; not generalization or 81-frame consistency.",
                  flush=True)
            return 0
        except BaseException as error:
            try:
                actual = completed_updates(training)
            except (ValueError, KeyError):
                actual = None
            if isinstance(error, KeyboardInterrupt):
                exit_code = getattr(error, "exit_code", 130)
            elif exit_code == 0:
                exit_code = 1
            patch_runtime(experiment_path, status="failed", stage=stage, error=str(error),
                          actual_updates=actual, exit_code=exit_code, elapsed_seconds=time.monotonic() - started)
            print(f"\nVACE ROOF SMOKE: FAILED\nStage: {stage}\nReason: {error}"
                  f"\nActual updates: {actual}\nExit code: {exit_code}\nPreserved output: {output}", flush=True)
            traceback.print_exc()
            return 128 - exit_code if exit_code < 0 else exit_code


def parser():
    arguments = argparse.ArgumentParser(description=__doc__)
    mode = arguments.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="CPU checks only, default")
    mode.add_argument("--run", action="store_true", help="User-started six inferences and 300-update training")
    mode.add_argument("--_worker", choices=STAGES, help=argparse.SUPPRESS)
    arguments.add_argument("--experiment", type=Path, help=argparse.SUPPRESS)
    arguments.add_argument("--temporary", type=Path, help=argparse.SUPPRESS)
    return arguments


def main(argv=None):
    for key, value in dict(DIFFSYNTH_SKIP_DOWNLOAD="True", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                           TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1").items():
        os.environ[key] = value
    args = parser().parse_args(argv)
    if args._worker:
        baseline.require(args.experiment and args.temporary and args.temporary.is_dir(), "Worker needs private run paths")
        worker(args._worker, args.experiment, args.temporary)
        return 0
    config = data.load_config()
    if args.run:
        return execute(config)
    data.prepare(config)
    return 0


if __name__ == "__main__":
    for signal_number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signal_number, stop_on_signal)
    try:
        sys.exit(main())
    except InterruptedRun as error:
        print(f"VACE ROOF SMOKE: FAILED: {error}; exit code {error.exit_code}", flush=True)
        sys.exit(error.exit_code)
    except Exception as error:
        print(f"VACE ROOF SMOKE: FAILED: {error}; exit code 1", flush=True)
        traceback.print_exc()
        sys.exit(1)
