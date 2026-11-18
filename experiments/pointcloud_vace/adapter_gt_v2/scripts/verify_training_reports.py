"""Replay the exact same completed GPU step's reporting inputs to both writers, CPU-only."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import random
import statistics
import tempfile
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch

import common
import training_telemetry
import worker


def read_progress(trial):
    records = []
    for line in (trial / "baseline.log").read_text().splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and value.get("stage") == "training":
            records.append(value)
    return records


def training_state(parameter, optimizer):
    stream = io.BytesIO()
    torch.save({"weight": parameter.detach(), "gradient": parameter.grad,
                "optimizer": optimizer.state_dict()}, stream)
    return dict(cpu_model_and_optimizer=hashlib.sha256(stream.getvalue()).hexdigest(),
                torch_rng=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                python_rng=hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
                numpy_rng=hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest())


def replay(trial):
    source = json.loads((trial / "baseline.json").read_text())
    run = Path(source["source_run"])
    schedule = json.loads((run / "schedule.json").read_text())
    cfg = json.loads((run / "cache_fullframe/config.json").read_text())
    if not source["passed"] or common.sha(run / "schedule.json") != cfg["schedule_sha256"]:
        raise ValueError("Original benchmark and frozen schedule required")
    if any(common.sha(common.ROOT / name) != digest for name, digest in cfg["source_sha256"].items()):
        raise ValueError("The original training sources have changed")
    records = read_progress(trial)
    losses = [json.loads(line) for line in (trial / "baseline/loss.jsonl").read_text().splitlines()]
    if len(records) != source["warmup_steps"] + source["timed_steps"]:
        raise ValueError("Incomplete original trial")
    parameter = torch.nn.Parameter(torch.tensor([0.3, -0.2]))
    optimizer = torch.optim.AdamW([parameter])
    parameter.square().sum().backward()
    optimizer.step()
    state_before = training_state(parameter, optimizer)
    train = worker.train
    update = worker.update
    save = worker.atomic_torch_save
    baseline_write = worker.write_json
    times = {"baseline": [], "telemetry": []}
    same_device = False
    with tempfile.TemporaryDirectory(prefix="reports_", dir=trial) as tmp:
        original = Path(tmp) / "baseline"
        fast = Path(tmp) / "telemetry"
        original.mkdir()
        fast.mkdir()
        same_device = os.stat(run).st_dev == os.stat(fast).st_dev
        if not same_device:
            raise ValueError("Report measurements require the experiment filesystem")
        # Both paths receive the same already measured GPU memory byte count. No GPU access.
        with patch.object(training_telemetry, "_DeviceMemory", autospec=True):
            training_telemetry.install_telemetry(fast)
        if worker.train is not train or worker.update is not update or worker.atomic_torch_save is not save:
            raise ValueError("Reporting changed the training or checkpoint implementation")
        for i, (progress, loss, trajectory) in enumerate(zip(records, losses, source["trajectory"], strict=True)):
            index = source["cursor"] + i
            row = schedule[index]
            if (progress["samples_consumed"] != index+1 or progress["loss"] != loss["loss"] or
                    loss["loss"] != trajectory["loss"] or trajectory["id"] != row["id"]):
                raise ValueError("Recorded step, sample and loss do not match")
            batch = dict(samples_consumed=index, optimizer_updates=source["optimizer_updates"]+i,
                         rows=[row], candidate=source["selected_config"])
            paths = [("baseline", original, baseline_write), ("telemetry", fast, worker.write_json)]
            # Alternate the first writer to reduce ordering effects in paired disk measurements.
            for label, directory, write in paths[::(-1 if i%2 else 1)]:
                started = perf_counter()
                write(directory / "current_batch.json", batch)
                write(directory / "progress.json", progress)
                times[label].append(perf_counter()-started)
            for name, expected in (("current_batch.json", batch), ("progress.json", progress)):
                a = (original / name).read_bytes()
                b = (fast / name).read_bytes()
                if a != b or json.loads(a) != expected:
                    raise ValueError(f"Publication mismatch for {name} at update {index+1}")
    if state_before != training_state(parameter, optimizer):
        raise ValueError("Reporting changed CPU RNG or the CPU model, gradient or AdamW state")
    return dict(passed=True, time=common.now(), reference_checkpoint_sha256=source["checkpoint_sha256"],
                samples=len(records), byte_identical_publications=2*len(records),
                same_filesystem=same_device, frozen_training_files=len(cfg["source_sha256"]),
                train_update_checkpoint_functions_identical=True, cpu_state_and_rng_unchanged=True,
                gpu_steps=0, wall_seconds_for_two_publications={
                    name: {"mean": statistics.mean(samples), "median": statistics.median(samples)}
                    for name, samples in times.items()})


def formal_baseline_repeat(trial):
    """Compare the already-computed formal continuation with its baseline probe."""
    source = json.loads((trial / "baseline.json").read_text())
    run = Path(source["source_run"])
    selection = json.loads((run / "training_speed/selection.json").read_text())
    if (selection["profile"] != "baseline" or Path(selection["trial"]) != trial or
            selection["checkpoint_sha256"] != source["checkpoint_sha256"]):
        raise ValueError("The formal continuation must use the same baseline and checkpoint")
    probe_steps = [json.loads(line) for line in (trial / "baseline/loss.jsonl").read_text().splitlines()]
    records = {}
    with (run / "loss.jsonl").open() as stream:
        for line in stream:
            record = json.loads(line)
            if source["cursor"] < record["samples_consumed"] <= source["cursor"] + len(probe_steps):
                records[record["samples_consumed"]] = record
                if len(records) == len(probe_steps):
                    break
    if len(records) != len(probe_steps):
        raise ValueError("The formal run has not passed every probe window yet")
    relative = {"loss": [], "grad_norm": [], "adapter_gradient_norm": [], "lora_gradient_norm": []}
    for measured in probe_steps:
        formal = records[measured["samples_consumed"]]
        if formal["time"] < selection["time"]:
            raise ValueError("The recorded step predates baseline resumption")
        if any(formal[k] != measured[k] for k in ("optimizer_updates", "ids", "frame_ids", "timesteps",
                                                   "scheduler_weights", "lr", "region_areas")):
            raise ValueError("Source rows or independent diffusion timestep sequence differ")
        for key in relative:
            if key.endswith("_gradient_norm"):
                get = lambda row: row["gradient_norms"][key.split("_")[0]]
            else:
                get = lambda row: row[key]
            reference = get(measured)
            relative[key].append(abs(get(formal)-reference) / max(abs(reference), 1e-20))
    return dict(time=common.now(), samples=len(records), checkpoint_sha256=source["checkpoint_sha256"],
                selection_time=selection["time"], profile=selection["profile"], first_sample=source["cursor"]+1,
                observed_ids_frames_loss_areas_timesteps_weights_lr_match=True,
                first_loss_identical=relative["loss"][0] == 0,
                max_relative_difference={key: max(values) for key, values in relative.items()},
                new_gpu_steps=0, endpoint_tensors_compared=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("--formal-repeat", action="store_true")
    args = parser.parse_args()
    if args.formal_repeat:
        report = formal_baseline_repeat(args.trial.resolve())
        filename = "formal_baseline_repeat.json"
    else:
        report = replay(args.trial.resolve())
        filename = "report_io_equivalence.json"
    common.write_json(args.trial / filename, report)
    print(json.dumps(report, indent=2))
