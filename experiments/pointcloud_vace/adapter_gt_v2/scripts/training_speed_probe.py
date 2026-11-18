"""Run one isolated, checkpoint-matched A6000 speed trial under the supervisor's GPU lock."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import mean
import sys
import time
import traceback
from unittest.mock import patch

from common import ROOT, atomic_torch_save, now, runtime_env, sha, write_json

WARMUP = 5
TIMED = 20


def endpoint(groups, initial):
    """Keep the optimizer's changes (not the much larger starting parameter values)."""
    import torch
    result = {}
    for group, named in groups.items():
        result[group + "_change"] = torch.cat([
            (p.detach().cpu() - initial[name]).reshape(-1) for name, p in named])
        result[group + "_gradient"] = torch.cat([
            p.grad.detach().cpu().reshape(-1) for _, p in named])
    return result


def compare(reference, actual, baseline, trial):
    import torch
    errors = {}
    passed = baseline["checkpoint_sha256"] == trial["checkpoint_sha256"]
    for key, ref in reference.items():
        diff = (actual[key].double() - ref.double()).norm()
        relative = (diff / ref.double().norm().clamp_min(1e-20)).item()
        errors[key] = relative
        passed &= relative <= 0.001
    passed &= baseline["rng_hashes"] == trial["rng_hashes"]
    for bh, th in zip(baseline["trajectory"], trial["trajectory"], strict=True):
        for key in ("sample_index", "id", "timesteps", "lr"):
            passed &= bh[key] == th[key]
        for key in ("loss", "grad_norm", "adapter_grad", "lora_grad"):
            passed &= abs(bh[key] - th[key]) <= 1e-6 + 1e-3 * abs(bh[key])
    return dict(passed=bool(passed), endpoint_relative_l2=errors, relative_limit=0.001,
                trajectory_absolute_tolerance=1e-6, matching_rng=baseline["rng_hashes"] == trial["rng_hashes"])


def probe(run, mode, out, checkpoint, result):
    import torch
    import fullframe_cache as cache
    from integration import load_increments_state
    import worker

    scratch = out / mode
    scratch.mkdir(parents=True, exist_ok=False)
    cfg = cache.config(run)
    if not (run / "base_identity.json").is_file():
        raise ValueError("This benchmark requires an already verified training run")
    identity = sha(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected = {key: sha(run / filename) for key, filename in
                (("manifest_sha256", "train_manifest.json"), ("prompt_sha256", "prompt.txt"),
                 ("schedule_sha256", "schedule.json"))}
    if (state["config_id"] != cfg["config_id"] or
            any(state[key] != value for key, value in expected.items())):
        raise ValueError("Checkpoint/input identity mismatch")
    candidate = state["selected_config"]
    if candidate["batch_size"] != 1 or candidate["cache_mode"] != "frozen":
        raise ValueError("This probe compares the existing frozen, single-window configuration")
    cursor = state["samples_consumed"]
    updates = state["optimizer_updates"]
    rows = json.loads((run / "schedule.json").read_text())[cursor:cursor + WARMUP + TIMED]
    if len(rows) != WARMUP + TIMED:
        raise ValueError("Insufficient remaining schedule for a full feasibility benchmark")
    result.update(checkpoint_sha256=identity, cursor=cursor, optimizer_updates=updates, warmup_steps=WARMUP,
                  timed_steps=TIMED, selected_config=candidate, source_run=str(run), time=now())
    if mode != "baseline":
        baseline = json.loads((out / "baseline.json").read_text())
        if not baseline["passed"] or baseline["checkpoint_sha256"] != identity:
            raise ValueError("Run baseline with exactly the same checkpoint first")
    cache.configure_training_reader(run)
    # The original loader records its parameter inventory each invocation.
    # Keep that one startup write inside this trial's temporary directory.
    with patch.object(worker, "write_json", lambda path, value: write_json(scratch / path.name, value)):
        pipe = worker.load_pipe(run, candidate["checkpointing"])
    shared = cache.read_shared(run, pipe)
    pipe.vae.to("cpu")
    torch.cuda.empty_cache()
    optimizer, adapter, lora = worker.make_optimizer(pipe)
    load_increments_state(pipe, state["adapter"], state["lora"])
    optimizer.load_state_dict(state["optimizer"])
    if mode == "fused":
        from training_kernels import install_rotary, verify_rotary
        result["rotary"] = verify_rotary()
        install_rotary()
    if mode in ("fused", "telemetry"):
        from training_telemetry import install_telemetry
        install_telemetry(scratch)
    worker.restore_rng(state["rng"])
    groups = dict(adapter=adapter, lora=lora)
    initial = {name: p.detach().cpu().clone() for named in groups.values() for name, p in named}
    del state
    torch.cuda.reset_peak_memory_stats()
    budget = min(40 << 30, worker.nvml_snapshot()["total_bytes"] - (6 << 30))
    times, trajectory = [], []
    # Seed the throwaway progress file with the actual run's usual field volume.
    write_json(scratch / "progress.json", json.loads((run / "progress.json").read_text()))
    with worker.MemoryMonitor() as monitor, (scratch / "loss.jsonl").open("w", buffering=1) as loss_log:
        for i, row in enumerate(rows):
            if (out / "stop.request").exists() or (run / "stop.request").exists():
                raise InterruptedError("Benchmark stop requested")
            torch.cuda.synchronize()
            started = time.monotonic()
            worker.write_json(scratch / "current_batch.json", dict(samples_consumed=cursor+i,
                              optimizer_updates=updates+i, rows=[row], candidate=candidate))
            before_read = time.monotonic()
            value, geom = cache.read_sample(run, pipe, row, "frozen")
            before_assemble = time.monotonic()
            batch = worker.assemble([value], [geom], shared, pipe.device)
            before_update = time.monotonic()
            rec = worker.update(pipe, optimizer, adapter, lora, batch, cursor + i, updates + i + 1)
            torch.cuda.synchronize()
            trained = time.monotonic()
            recent = dict(sample_index=row["sample_index"], id=row["id"], loss=rec["loss"], lr=rec["lr"],
                          grad_norm=rec["grad_norm"], adapter_grad=rec["gradient_norms"]["adapter"],
                          lora_grad=rec["gradient_norms"]["lora"], timesteps=rec["timesteps"])
            trajectory.append(recent)
            rec.update(time=now(), optimizer_updates=updates+i+1, samples_consumed=cursor+i+1,
                       ids=[row["id"]], frame_ids=[row["frame_ids"]], step_seconds=trained-started)
            loss_log.write(json.dumps(rec) + "\n")
            worker.event(scratch, "training", status="running", samples_consumed=cursor+i+1,
                         optimizer_updates=updates+i+1, epoch=row["epoch"]+1, epoch_buildings=worker.PER_EPOCH,
                         epoch_completed=(cursor+i)%worker.PER_EPOCH+1, batch_size=1, total_samples=worker.TOTAL,
                         samples_per_second=1/max(trained-started, 1e-8), loss=rec["loss"],
                         gpu_memory_bytes=worker.nvml_snapshot()["used_bytes"], peak_nvml_bytes=monitor.peak,
                         stage_remaining_seconds=(worker.TOTAL-cursor-i-1)*(trained-started),
                         overall_remaining_seconds=(worker.TOTAL-cursor-i-1)*(trained-started),
                         latest_checkpoint=str(checkpoint), latest_checkpoint_update=updates)
            del batch, value, geom
            ended = time.monotonic()
            if monitor.peak > budget or monitor.errors:
                raise RuntimeError(f"GPU budget/monitor error: {monitor.peak}, {monitor.errors}")
            if i >= WARMUP:
                times.append(dict(step=ended-started, read=before_assemble-before_read,
                                  assemble=before_update-before_assemble, update=trained-before_update,
                                  bookkeeping=(ended-trained)+(before_read-started)))
    result.update(trajectory=trajectory, stable_step_seconds=[t["step"] for t in times],
                  mean_step_seconds=mean(t["step"] for t in times),
                  samples_per_second=1/mean(t["step"] for t in times),
                  peak_nvml_bytes=monitor.peak, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(), memory_budget_bytes=budget)
    for key in ("read", "assemble", "update", "bookkeeping"):
        result[key + "_seconds"] = mean(t[key] for t in times)
    rng = [torch.get_rng_state(), *torch.cuda.get_rng_state_all()]
    result["rng_hashes"] = [hashlib.sha256(s.numpy().tobytes()).hexdigest() for s in rng]
    final = endpoint(groups, initial)
    path = out / "baseline_endpoint.pt"
    if mode == "baseline":
        atomic_torch_save(path, final)
    else:
        result["comparison"] = compare(torch.load(path, map_location="cpu", weights_only=True), final, baseline, result)
        if not result["comparison"]["passed"]:
            raise RuntimeError("Numerical or RNG comparison failed")
    if monitor.errors or monitor.peak > budget:
        raise RuntimeError("Final memory check failed")
    result["passed"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("mode", choices=("baseline", "fused", "telemetry"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    out = (args.output_dir or run / "training_speed").resolve()
    if out == run:
        raise ValueError("An isolated output directory is required")
    out.mkdir(parents=True, exist_ok=True)
    if (out / args.mode).exists() or (out / (args.mode + ".json")).exists():
        raise FileExistsError("This trial already exists. Use a fresh output directory.")
    os.environ.update(runtime_env(run))
    sys.path.insert(0, str(ROOT / "scripts"))
    checkpoint = args.checkpoint or run / "checkpoints/resume_latest.pt"
    result = dict(passed=False, mode=args.mode)
    try:
        probe(run, args.mode, out, checkpoint, result)
    except Exception as error:
        result.update(error=str(error), traceback=traceback.format_exc())
    write_json(out / (args.mode + ".json"), result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("trajectory", "stable_step_seconds")}), flush=True)
    return int(not result["passed"])


if __name__ == "__main__":
    sys.exit(main())
