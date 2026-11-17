"""One-run cache execution tuning followed by the unchanged formal training.

All candidates use the original 17-frame windows, VAE weights, nine overlapping
tiles and blend weights. Existing cache files are read-only/reused. Failed
numerical or memory candidates are rejected, with the original encoder as the
fallback. This supervisor never generates evaluation videos.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback

from common import ROOT, PYTHON, CACHE, now, sha, write_json, atomic_torch_save, nvml_snapshot, runtime_env
import pipeline

TOTAL = 32241
CANDIDATES = [dict(name="original", original=True, tile_batch=1),
              dict(name="gpu_blend_1", tile_batch=1, gpu_blend=True),
              dict(name="tile_batch_2", tile_batch=2, gpu_blend=False),
              dict(name="tile_batch_3", tile_batch=3, gpu_blend=False),
              dict(name="tile_batch_9", tile_batch=9, gpu_blend=False)]


def stopped(run):
    return (run / "stop.request").exists()


def progress(run, stage, **values):
    record = dict(time=now(), stage=stage, status=stage, pid=os.getpid(),
                  total_samples=TOTAL, samples_consumed=0, optimizer_updates=0,
                  epoch=0, epoch_completed=0, epoch_buildings=10747,
                  validation_started=False, **values)
    write_json(run / "progress.json", record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def verify_sources(run):
    record = json.loads((run / "cache_optimization/source_identity.json").read_text())
    for relative, expected in record["source_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Cache acceleration source changed: {relative}")


def load_vae_pipe(run):
    import torch
    import worker
    verify_sources(run)
    pipe = worker.load_pipe(run)
    # Fixed text and the inactive latent already exist. Do not encode them again.
    pipe.text_encoder.to("cpu")
    pipe.dit.to("cpu")
    pipe.vace.to("cpu")
    pipe.structure_adapter.to("cpu")
    torch.cuda.empty_cache()
    return pipe


def difference(actual, expected):
    import torch
    delta = actual.float() - expected.float()
    maximum = float(delta.abs().max())
    relative = float(delta.square().sum().sqrt() / expected.float().square().sum().sqrt().clamp_min(1.e-12))
    return dict(passed=bool(torch.isfinite(actual).all()) and
                      bool(torch.allclose(actual.float(), expected.float(), rtol=.005, atol=.005)),
                max_absolute=maximum, relative_l2=relative)


def benchmark(run, candidate):
    import torch
    import worker
    from data import load_raw
    from fast_vae import install
    out = run / "cache_optimization" / (candidate["name"] + ".json")
    budget = min(40 << 30, nvml_snapshot()["total_bytes"] - (6 << 30))
    result = dict(candidate=candidate, passed=False, warmup_updates=5, timed_updates=20,
                  memory_budget_bytes=budget, unchanged_tile_size=[30, 52], unchanged_tile_stride=[15, 26])
    scratch = run / "cache_optimization" / (candidate["name"] + ".scratch.pt")
    with worker.MemoryMonitor() as monitor:
        try:
            pipe = load_vae_pipe(run)
            install(pipe.vae, candidate)
            rows = json.loads((run / "benchmark_samples.json").read_text())
            measured, encoding, validations = [], [], []
            torch.cuda.reset_peak_memory_stats()
            for step in range(25):
                if stopped(run):
                    raise InterruptedError("User requested stop")
                row = rows[step % len(rows)]
                torch.cuda.synchronize()
                began = time.monotonic()
                raw = load_raw(run, row)
                encode_began = time.monotonic()
                value = worker.encode_raw(pipe, raw)
                torch.cuda.synchronize()
                encode_seconds = time.monotonic() - encode_began
                atomic_torch_save(scratch, value)
                elapsed = time.monotonic() - began
                if step >= 5:
                    measured.append(elapsed)
                    encoding.append(encode_seconds)
                # Compare every real sample against its original frozen cache.
                # Regions and identity are required to stay exactly identical.
                expected = torch.load(worker.entry_path(run, row), weights_only=True, map_location="cpu")
                checks = {k: difference(value[k], expected[k]) for k in ("input_latents", "reactive")}
                exact = torch.equal(value["occupancy"], expected["occupancy"]) and value["cache_key"] == expected["cache_key"]
                validations.append(dict(id=row["id"], frame_ids=row["frame_ids"], **checks, exact_area_and_identity=exact))
                if not exact or not all(c["passed"] for c in checks.values()):
                    result["numerical_checks"] = validations
                    raise ValueError("Optimized encoding differs from the original beyond BF16 rtol=.005/atol=.005")
                if monitor.peak > budget:
                    raise MemoryError("Whole-GPU memory budget exceeded")
                progress(run, "autotuning_cache", cache_candidate=candidate["name"],
                         candidate_step=step + 1, candidate_total_steps=25,
                         seconds_per_window=elapsed, encode_seconds=encode_seconds,
                         gpu_memory_bytes=nvml_snapshot()["used_bytes"])
                del raw, value, expected
            result.update(passed=True, seconds_per_window=sum(measured) / len(measured),
                          samples_per_second=len(measured) / sum(measured),
                          encode_seconds_per_window=sum(encoding) / len(encoding),
                          stable_window_seconds=measured, numerical_checks=validations,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
        except BaseException as exc:
            result.update(error=str(exc), exception=type(exc).__name__, traceback=traceback.format_exc())
            print(result["traceback"], flush=True)
    result.update(time=now(), peak_nvml_bytes=monitor.peak, nvml_errors=monitor.errors)
    if monitor.errors or monitor.peak > budget:
        result.update(passed=False, error="Whole GPU memory budget or NVML monitor failure")
    scratch.unlink(missing_ok=True)
    write_json(out, result)
    return 0 if result["passed"] else 1


def cache_remaining(run, selected):
    import torch
    import worker
    from data import load_raw
    from fast_vae import install
    pipe = load_vae_pipe(run)
    install(pipe.vae, selected["candidate"])
    schedule = json.loads((run / "schedule.json").read_text())
    # Read existing paths once. Do not redo their encoding or alter their bytes.
    existing = {r["sample_index"] for r in schedule if worker.entry_path(run, r).exists()}
    original_selected = json.loads((run / "selected_config.json").read_text())
    training_seconds = original_selected["projected_train_seconds"]
    began = time.monotonic()
    fresh, fresh_seconds = 0, 0.
    count = 0
    with worker.MemoryMonitor() as monitor, (run / "cache_index.jsonl").open("w", buffering=1) as index:
        for row in schedule:
            if stopped(run):
                return
            path = worker.entry_path(run, row)
            if row["sample_index"] not in existing:
                if shutil.disk_usage(CACHE).free < 20 << 30:
                    raise OSError("Cache disk reserve reached")
                t = time.monotonic()
                raw = load_raw(run, row)
                value = worker.encode_raw(pipe, raw)
                atomic_torch_save(path, value)
                fresh_seconds += time.monotonic() - t
                fresh += 1
                del raw, value
            count += 1
            index.write(json.dumps(dict(sample_index=row["sample_index"], id=row["id"], frame_ids=row["frame_ids"], path=str(path))) + "\n")
            if count % 10 == 0 or count == TOTAL:
                secs = fresh_seconds / fresh if fresh else selected["seconds_per_window"]
                remaining = (TOTAL - count) * secs
                progress(run, "caching", cached_samples=count, reused_samples=len(existing),
                         newly_cached_samples=fresh, cache_candidate=selected["candidate"],
                         seconds_per_window=secs, stage_remaining_seconds=remaining,
                         overall_remaining_seconds=remaining + training_seconds,
                         gpu_memory_bytes=nvml_snapshot()["used_bytes"], peak_nvml_bytes=monitor.peak,
                         selected_config=original_selected)
            if monitor.peak > min(40 << 30, nvml_snapshot()["total_bytes"] - (6 << 30)):
                raise MemoryError("Optimized cache exceeded memory budget")
    write_json(run / "cache_index.json", dict(mode="frozen", samples=count, index=str(run / "cache_index.jsonl"),
               reused_samples=len(existing), newly_encoded_samples=fresh, elapsed_seconds=time.monotonic() - began,
               source_schedule_sha256=sha(run / "schedule.json"), cache_execution=selected["candidate"]))


def child(run, phase, candidate=None):
    command = [PYTHON, "-u", "-B", str(Path(__file__).resolve()), phase, str(run)]
    if candidate is not None:
        command += ["--candidate", json.dumps(candidate, separators=(",", ":"))]
    process = subprocess.Popen(command, cwd=ROOT, env=runtime_env(run), stdin=subprocess.DEVNULL)
    (run / "worker.pid").write_text(str(process.pid) + "\n")
    print(f"CACHE_WORKER phase={phase} pid={process.pid}", flush=True)
    code = process.wait()
    with (run / "worker_history.jsonl").open("a") as out:
        out.write(json.dumps(dict(time=now(), phase=phase, candidate=candidate, pid=process.pid, exit_code=code)) + "\n")
    if stopped(run):
        raise InterruptedError("User requested stop at a complete cache-file boundary")
    return code


def supervise(run):
    began = time.monotonic()
    code = 1
    with (run / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (run / "run.pid").write_text(str(os.getpid()) + "\n")
        try:
            verify_sources(run)
            progress(run, "autotuning_cache", supervisor_pid=os.getpid(),
                     detail="same nine tiles; numerical verification and serial timing")
            with pipeline.acquire_gpu(run):
                selection_path = run / "cache_optimization/selected.json"
                if selection_path.exists():
                    selected = json.loads(selection_path.read_text())
                else:
                    results = []
                    for candidate in CANDIDATES:
                        path = run / "cache_optimization" / (candidate["name"] + ".json")
                        if not path.exists():
                            child(run, "benchmark", candidate)
                        result = json.loads(path.read_text())
                        results.append(result)
                        write_json(run / "cache_optimization/results.json", dict(time=now(), completed=False, candidates=results))
                    passed = [x for x in results if x["passed"]]
                    if not passed:
                        raise RuntimeError("Original and optimized VAE encoding candidates all failed")
                    fastest = min(passed, key=lambda x: x["seconds_per_window"])
                    tied = [x for x in passed if x["seconds_per_window"] <= fastest["seconds_per_window"] * 1.05]
                    selected = min(tied, key=lambda x: (x["peak_nvml_bytes"], x["seconds_per_window"]))
                    baseline = next((x for x in results if x["candidate"]["name"] == "original" and x["passed"]), None)
                    selected = dict(selected, selected_time=now(),
                                    speedup_vs_original=baseline["seconds_per_window"] / selected["seconds_per_window"] if baseline else None)
                    write_json(selection_path, selected)
                    write_json(run / "cache_optimization/results.json", dict(time=now(), completed=True, candidates=results, selected=selected))
                print("CACHE_EXECUTION_SELECTED " + json.dumps(selected, ensure_ascii=False), flush=True)
                if not (run / "cache_complete.json").exists():
                    code = child(run, "cache", selected)
                    if code:
                        # A runtime cache OOM discards no published files; a
                        # fresh original encoder finishes the remaining files.
                        failed = json.loads((run / "cache_optimization/worker_failure.json").read_text())
                        if failed.get("oom") and not selected["candidate"].get("original"):
                            baseline = json.loads((run / "cache_optimization/original.json").read_text())
                            if not baseline["passed"]:
                                raise RuntimeError("Original cache fallback did not pass")
                            write_json(run / "cache_optimization/runtime_fallback.json", dict(time=now(), failed=selected["candidate"], fallback=baseline["candidate"]))
                            code = child(run, "cache", baseline)
                        if code:
                            raise RuntimeError(f"Cache worker failed with exit code {code}")
                    index = json.loads((run / "cache_index.json").read_text())
                    if index["samples"] != TOTAL:
                        raise RuntimeError("Incomplete cache")
                    write_json(run / "cache_complete.json", dict(time=now(), strategy="frozen", optimized=True))
                # No data import, no optimizer initialization replay, no new
                # samples: delegate the unchanged training/restoration code.
                autotune = json.loads((run / "autotune.json").read_text())
                original_selected = json.loads((run / "selected_config.json").read_text())
                code = pipeline.formal_training(run, autotune, original_selected, resume=bool(pipeline.latest_checkpoint(run)))
            pipeline.mark(run, "completed", training_complete=str(run / "training_complete.json"), validation_started=False)
        except InterruptedError as exc:
            code = 130
            pipeline.mark(run, "stopped", reason=str(exc))
        except BaseException as exc:
            code = 1
            failure = dict(time=now(), stage=pipeline.read_json(run / "progress.json", {}).get("stage"),
                           error=str(exc), traceback=traceback.format_exc(), validation_started=False,
                           resume_command=f"bash {run / 'resume.sh'}")
            write_json(run / "failure.json", failure)
            pipeline.mark(run, "failed", error=str(exc))
            print(failure["traceback"], flush=True)
        finally:
            write_json(run / "exit.json", dict(time=now(), supervisor_pid=os.getpid(), exit_code=code,
                       returncode=code, elapsed_seconds=time.monotonic() - began,
                       training_completed=(run / "training_complete.json").exists(), validation_started=False))
    return code


def launch(run):
    import run as launcher
    if launcher.running(run):
        raise RuntimeError("Stop the existing supervisor at its safe boundary before cache tuning")
    if (run / "training_complete.json").exists():
        print("Already completed; no restart")
        return
    folder = run / "cache_optimization"
    folder.mkdir(exist_ok=True)
    sources = [Path(__file__).resolve(), Path(__file__).with_name("fast_vae.py").resolve()]
    identity = dict(time=now(), reason="user requested faster caching; preserve all prior caches and formal training definition",
                    base_config_id=json.loads((run / "config.json").read_text())["config_id"],
                    source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sources})
    existing_identity = folder / "source_identity.json"
    if existing_identity.exists() and json.loads(existing_identity.read_text())["source_sha256"] != identity["source_sha256"]:
        raise ValueError("Cache execution sources changed after freezing")
    if not existing_identity.exists():
        write_json(existing_identity, identity)
        for source in sources:
            shutil.copyfile(source, folder / source.name)
    for name in ("exit.json", "stopped.json", "failure.json", "launch.json"):
        path = run / name
        if path.exists():
            (run / "invocation_history").mkdir(exist_ok=True)
            shutil.copyfile(path, run / "invocation_history" / f"{time.time_ns()}_{name}")
    (run / "stop.request").unlink(missing_ok=True)
    for name in ("exit.json", "stopped.json", "failure.json"):
        (run / name).unlink(missing_ok=True)
    command = [PYTHON, "-u", "-B", str(Path(__file__).resolve()), "supervise", str(run)]
    env = runtime_env(run)
    keys = [k for k in env if k in ("TMPDIR", "TMP", "TEMP", "HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME", "CUDA_CACHE_PATH",
            "TRITON_CACHE_DIR", "NUMBA_CACHE_DIR", "MPLCONFIGDIR", "CUDA_VISIBLE_DEVICES", "PYTHONUNBUFFERED", "PYTHONDONTWRITEBYTECODE",
            "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "DIFFSYNTH_SKIP_DOWNLOAD", "TOKENIZERS_PARALLELISM",
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")]
    shell = "set -o pipefail\n" + shlex.join(["env"] + [k + "=" + env[k] for k in keys] + command)
    shell += " 2>&1 | tee -a " + shlex.quote(str(run / "train.log"))
    shell += "\nexec bash --noprofile --norc"
    session = launcher.session_name(run)
    exists = subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0
    tmux = ["tmux", "new-window", "-t", session] if exists else ["tmux", "new-session", "-d", "-s", session]
    subprocess.run(tmux + ["-n", "cache-speed", "-c", str(ROOT), shlex.join(["bash", "--noprofile", "--norc", "-c", shell])], env=env, check=True)
    (run / "resume.sh").write_text("#!/usr/bin/env bash\nexec " + shlex.join([PYTHON, str(Path(__file__).resolve()), "launch", str(run)]) + "\n")
    time.sleep(.3)
    record = dict(time=now(), tmux_session=session, window="cache-speed", run=str(run),
                  supervisor_pid=int((run / "run.pid").read_text()), log=str(run / "train.log"),
                  attach_command=f"tmux attach -t {session}", stop_command=f"bash {run / 'stop.sh'}",
                  resume_command=f"bash {run / 'resume.sh'}", no_automatic_validation=True,
                  independent_of_codex_and_ssh=True)
    write_json(run / "launch.json", record)
    print(json.dumps(record, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("launch", "supervise", "benchmark", "cache"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--candidate", type=json.loads)
    args = parser.parse_args()
    run = args.run.resolve()
    os.environ.update(runtime_env(run))
    if args.mode == "launch":
        launch(run)
        return 0
    if args.mode == "supervise":
        return supervise(run)
    try:
        if args.mode == "benchmark":
            return benchmark(run, args.candidate)
        cache_remaining(run, args.candidate)
        return 0
    except BaseException as exc:
        import torch
        write_json(run / "cache_optimization/worker_failure.json", dict(time=now(), error=str(exc),
                   traceback=traceback.format_exc(), oom=isinstance(exc, (MemoryError, torch.cuda.OutOfMemoryError))))
        print(traceback.format_exc(), flush=True)
        return 42 if isinstance(exc, (MemoryError, torch.cuda.OutOfMemoryError)) else 1


if __name__ == "__main__":
    sys.exit(main())
