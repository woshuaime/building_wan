"""Independent, serial import/autotune/cache/train owner for one GT-video run.

This module deliberately imports no CUDA package.  Each GPU operation happens
in a fresh worker process while this owner retains the project-wide GPU lock.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from common import ROOT, PYTHON, now, write_json, nvml_snapshot, runtime_env

GIB = 1024 ** 3
TOTAL_SAMPLES = 32241
CHECKPOINT_ORDER = {"none": 0, "partial": 1, "full": 2}


def read_json(path, default=None):
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return default
    return json.loads(text)


def environment(run):
    try:
        return runtime_env(run)
    except TypeError:
        return runtime_env()


def mark(run, stage, **extra):
    previous = read_json(run / "progress.json", {})
    previous.update(time=now(), status=stage, stage=stage,
                    supervisor_pid=os.getpid(), **extra)
    write_json(run / "progress.json", previous)
    write_json(run / "status.json", previous)
    print(json.dumps(previous, ensure_ascii=False), flush=True)


def stopping(run):
    return (run / "stop.request").exists()


def latest_checkpoint(run):
    for path in (run / "checkpoints/resume_latest.pt", run / "resume_latest.pt"):
        if path.exists():
            return path
    return None


def verify_completion(run):
    record = read_json(run / "training_complete.json")
    if not record:
        raise RuntimeError("Formal worker exited without training_complete.json")
    if record.get("samples_consumed") != TOTAL_SAMPLES or record.get("epochs") != 3:
        raise RuntimeError("Completion marker does not cover all 32,241 samples and three epochs")
    for key in ("adapter_final", "lora_final"):
        path = Path(record.get(key, ""))
        if not path.is_absolute():
            path = run / path
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Completion marker has no saved final weight: {key}")
    return record


def total_gpu_bytes(snapshot):
    for key in ("memory_total_bytes", "total_bytes", "total"):
        if key in snapshot:
            return int(snapshot[key])
    for key in ("memory", "gpu_memory"):
        child = snapshot.get(key, {})
        if isinstance(child, dict):
            for field in ("total_bytes", "total"):
                if field in child:
                    return int(child[field])
    raise ValueError("NVML snapshot does not report the actual GPU total memory")


def external_processes(snapshot):
    result = []
    for kind in ("compute_processes", "graphics_processes"):
        for process in snapshot.get(kind, []):
            if int(process["pid"]) != os.getpid():
                result.append(dict(kind=kind, **process))
    return result


@contextlib.contextmanager
def acquire_gpu(run):
    """Wait without touching another task; reuse the existing launcher lock."""
    directory = ROOT / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ".wan_a6000_gpu.lock"
    with path.open("a") as stream:
        last_print = 0.0
        while True:
            if stopping(run):
                raise InterruptedError("User requested stop before GPU acquisition")
            acquired = False
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                pass
            snapshot = nvml_snapshot()
            occupied = external_processes(snapshot)
            if acquired and not occupied:
                write_json(run / "gpu_lock.json", dict(time=now(), owner_pid=os.getpid(),
                           path=str(path), nvml=snapshot,
                           memory_budget_bytes=min(40 * GIB, total_gpu_bytes(snapshot) - 6 * GIB)))
                break
            if acquired:
                fcntl.flock(stream, fcntl.LOCK_UN)
            if time.monotonic() - last_print >= 60:
                mark(run, "waiting_for_gpu", lock=str(path),
                     reason="external GPU processes" if occupied else "another launcher owns the GPU lock",
                     external_processes=occupied)
                last_print = time.monotonic()
            for _ in range(15):
                if stopping(run):
                    raise InterruptedError("User requested stop while waiting for GPU")
                time.sleep(1)
        try:
            yield snapshot
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def run_worker(run, mode, candidate=None, resume=False):
    if stopping(run):
        raise InterruptedError("User requested stop at a pipeline boundary")
    command = [str(PYTHON), "-u", "-B", str(Path(__file__).with_name("worker.py")),
               str(run), "--mode", mode]
    if candidate is not None:
        command += ["--candidate", json.dumps(candidate, separators=(",", ":"))]
    if resume:
        command.append("--resume")
    began = time.monotonic()
    process = subprocess.Popen(command, cwd=ROOT, env=environment(run), stdin=subprocess.DEVNULL)
    (run / "worker.pid").write_text(str(process.pid) + "\n", encoding="utf-8")
    launch = dict(time=now(), mode=mode, candidate=candidate, resume=resume,
                  pid=process.pid, command=command)
    print("WORKER " + json.dumps(launch, ensure_ascii=False), flush=True)
    exit_code = process.wait()
    launch.update(exit_time=now(), exit_code=exit_code, elapsed_seconds=time.monotonic() - began)
    with (run / "worker_history.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(launch, ensure_ascii=False) + "\n")
    print("WORKER_EXIT " + json.dumps(launch, ensure_ascii=False), flush=True)
    if stopping(run) and not (run / "training_complete.json").exists():
        raise InterruptedError("User requested stop; worker exited at its saved boundary")
    return exit_code, launch["elapsed_seconds"]


def candidate_peak(result):
    for key in ("peak_nvml_bytes", "nvml_peak_bytes", "peak_memory_nvml_bytes", "peak_whole_gpu_bytes"):
        if key in result:
            return int(result[key])
    value = result.get("memory", {})
    for key in ("peak_nvml_bytes", "nvml_peak_bytes", "peak_whole_gpu_bytes"):
        if key in value:
            return int(value[key])
    raise ValueError("Benchmark omitted the measured NVML whole-GPU peak")


def result_total(result, prep):
    if "projected_total_seconds" in result:
        return float(result["projected_total_seconds"])
    speed = float(result["stable_samples_per_second"])
    strategy = result["candidate"]["cache_mode"]
    cache_seconds = float(prep.get("projected_cache_seconds", 0)) if strategy == "frozen" else 0
    common_seconds = float(prep.get("projected_geometry_cache_seconds", 0))
    saving = float(result.get("estimated_checkpoint_seconds", 0))
    return cache_seconds + common_seconds + TOTAL_SAMPLES / speed + saving


def choose_result(results):
    """Compare complete projected duration; prefer lower memory within five percent."""
    qualified = [item for item in results if item.get("passed")]
    if not qualified:
        raise RuntimeError("No measured candidate passed memory, numerical and gradient checks")
    fastest = min(qualified, key=lambda item: item["projected_total_seconds"])
    equivalent = [item for item in qualified
                  if item["projected_total_seconds"] <= fastest["projected_total_seconds"] * 1.05]
    return min(equivalent, key=lambda item: (candidate_peak(item), item["projected_total_seconds"]))


def recovery_candidates(current, results):
    """Only measured successes, checkpointing first, then a smaller physical batch."""
    candidates = []
    for result in results:
        candidate = result["candidate"]
        if not result.get("passed") or candidate["cache_mode"] != current["cache_mode"]:
            continue
        if candidate["batch_size"] > current["batch_size"]:
            continue
        if candidate["batch_size"] == current["batch_size"] and (
            CHECKPOINT_ORDER[candidate["checkpointing"]] <= CHECKPOINT_ORDER[current["checkpointing"]]
        ):
            continue
        candidates.append(candidate)
    return sorted(candidates, key=lambda candidate: (
        current["batch_size"] - candidate["batch_size"],
        CHECKPOINT_ORDER[candidate["checkpointing"]],
    ))


def autotune(run, snapshot):
    (run / "autotune_candidates").mkdir(parents=True, exist_ok=True)
    saved = read_json(run / "autotune.json")
    selected = read_json(run / "selected_config.json")
    if saved and saved.get("completed") and selected:
        return saved, selected
    budget = min(40 * GIB, total_gpu_bytes(snapshot) - 6 * GIB)
    preparation = read_json(run / "benchmark_preparation.json")
    if preparation is None:
        mark(run, "preparing_benchmark", memory_budget_bytes=budget)
        code, duration = run_worker(run, "prepare_benchmark")
        if code:
            raise RuntimeError(f"Benchmark preparation failed with exit code {code}")
        preparation = read_json(run / "benchmark_preparation.json")
        if preparation is None:
            raise RuntimeError("Benchmark worker returned success without benchmark_preparation.json")
        preparation["actual_preparation_wall_seconds"] = duration
        write_json(run / "benchmark_preparation.json", preparation)
    cache_bytes = int(preparation.get("projected_cache_bytes", 0))
    config = read_json(run / "config.json", {})
    cache_root = Path(config.get("cache_root", ROOT / "cache/pointcloud_vace/adapter_gt_v2"))
    cache_root.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(cache_root).free
    cache_fits = cache_bytes > 0 and cache_bytes * 1.1 + 8 * GIB <= free_bytes
    results = []

    def measure(name, batch, checkpointing, cache_mode="frozen"):
        candidate = dict(name=name, batch_size=batch, checkpointing=checkpointing,
                         cache_mode=cache_mode, warmup_updates=5, timed_updates=20,
                         memory_budget_bytes=budget)
        output = run / "autotune_candidates" / f"{name}.json"
        result = read_json(output)
        if result is None:
            mark(run, "autotuning", candidate=candidate, tested_candidates=len(results),
                 memory_budget_bytes=budget)
            code, duration = run_worker(run, "benchmark", candidate)
            result = read_json(output, {})
            result.update(candidate=candidate, worker_exit_code=code,
                          benchmark_wall_seconds=duration)
            if code:
                result.update(passed=False, failure_reason=result.get("failure_reason", f"worker exited {code}"))
        result["candidate"] = candidate
        try:
            if result.get("passed"):
                speed = float(result["stable_samples_per_second"])
                if not math.isfinite(speed) or speed <= 0:
                    raise ValueError("Invalid samples/second")
                if candidate_peak(result) > budget:
                    raise ValueError("Measured whole-GPU peak exceeds the allowed budget")
                if int(result.get("warmup_updates", 5)) < 5 or int(result.get("timed_updates", 20)) < 20:
                    raise ValueError("Benchmark did not cover at least 5 warmup and 20 timed updates")
                result["projected_total_seconds"] = result_total(result, preparation)
                if not math.isfinite(result["projected_total_seconds"]) or result["projected_total_seconds"] <= 0:
                    raise ValueError("Invalid projected total duration")
                result["cache_disk_fit"] = cache_fits if cache_mode == "frozen" else True
                if cache_mode == "frozen" and not cache_fits:
                    result["selection_disqualified_reason"] = "Projected full cache does not fit with disk margin"
        except (ValueError, KeyError, TypeError) as exc:
            result.update(passed=False, failure_reason=str(exc))
        write_json(output, result)
        results.append(result)
        write_json(run / "autotune.json", dict(time=now(), completed=False,
                   memory_budget_bytes=budget, candidates=results, preparation=preparation,
                   cache_free_bytes=free_bytes, projected_cache_bytes=cache_bytes,
                   full_cache_fits=cache_fits))
        return result

    a = measure("A", 2, "full")
    b = measure("B", 1, "none")
    if not b.get("passed"):
        measure("B_partial", 1, "partial")
    measure("C", 1, "full")
    if a.get("passed") and budget - candidate_peak(a) >= 6 * GIB:
        d = measure("D_partial", 2, "partial")
        if d.get("passed") and budget - candidate_peak(d) >= 6 * GIB:
            measure("D_none", 2, "none")

    qualified = [item for item in results if item.get("passed")]
    if not qualified:
        # Online encoding can alter placement; try the conservative online path
        # even if no resident frozen candidate was viable.
        measure("C_online", 1, "full", "online")
    else:
        fastest = choose_result(qualified)["candidate"]
        measure(fastest["name"] + "_online", fastest["batch_size"], fastest["checkpointing"], "online")
        if fastest["name"] != "C":
            measure("C_online", 1, "full", "online")
        if fastest["batch_size"] == 2 and fastest["checkpointing"] != "full":
            measure("A_online", 2, "full", "online")

    selectable = [item for item in results if item.get("passed") and
                  not item.get("selection_disqualified_reason")]
    winner = choose_result(selectable)
    selected = dict(winner["candidate"], selected_time=now(),
                    projected_total_seconds=winner["projected_total_seconds"],
                    projected_cache_seconds=winner.get("projected_cache_seconds", preparation.get("projected_cache_seconds", 0)
                                                       if winner["candidate"]["cache_mode"] == "frozen" else 0),
                    projected_train_seconds=winner.get("projected_train_seconds", TOTAL_SAMPLES / winner["stable_samples_per_second"]),
                    projected_save_seconds=winner.get("projected_save_seconds", 0),
                    stable_samples_per_second=winner["stable_samples_per_second"],
                    peak_nvml_bytes=candidate_peak(winner),
                    measured_candidates_only=True,
                    selection_rule="minimum projected preparation+train+save duration; within 5% prefer lower whole-GPU peak")
    write_json(run / "selected_config.json", selected)
    record = dict(time=now(), completed=True, memory_budget_bytes=budget,
                  candidates=results, preparation=preparation, selected=selected,
                  cache_free_bytes=free_bytes, projected_cache_bytes=cache_bytes,
                  full_cache_fits=cache_fits,
                  benchmark_updates_count_toward_formal_epochs=False,
                  online_comparison="Measured same configuration with online encoding plus conservative recovery configurations")
    write_json(run / "autotune.json", record)
    return record, selected


def formal_training(run, autotune_record, selected, resume):
    if (run / "training_complete.json").exists():
        verify_completion(run)
        return 0
    state = read_json(run / "supervisor_recovery.json", dict(automatic_restarts=0, history=[]))
    current = state.get("current_candidate", selected)
    while True:
        mark(run, "training", selected_config=current,
             automatic_restarts=state["automatic_restarts"])
        code, duration = run_worker(run, "train", current,
                                   resume=bool(latest_checkpoint(run)))
        state["history"].append(dict(time=now(), candidate=current,
                                     exit_code=code, elapsed_seconds=duration))
        state["current_candidate"] = current
        write_json(run / "supervisor_recovery.json", state)
        if code == 0:
            verify_completion(run)
            return 0
        if code not in (42, 43):
            raise RuntimeError(f"Formal worker failed with non-OOM exit code {code}; not automatically retried")
        if state["automatic_restarts"] >= 2:
            raise RuntimeError("CUDA OOM: the maximum of two automatic formal worker restarts was exhausted")
        fallbacks = recovery_candidates(current, autotune_record["candidates"])
        if not fallbacks:
            raise RuntimeError("CUDA OOM: no measured passing configuration is more memory conservative")
        current = fallbacks[0]
        state["automatic_restarts"] += 1
        state["current_candidate"] = current
        write_json(run / "supervisor_recovery.json", state)
        mark(run, "recovering_oom", selected_config=current,
             automatic_restarts=state["automatic_restarts"],
             resume_checkpoint=str(latest_checkpoint(run)) if latest_checkpoint(run) else "clean initialization before first update")
        resume = bool(latest_checkpoint(run))


def supervise(run, resume=False):
    run = Path(run).resolve()
    if not (run / "config.json").exists():
        raise ValueError(f"Missing run config: {run / 'config.json'}")
    if (run / "training_complete.json").exists():
        verify_completion(run)
        print("Training is already complete; no work restarted.", flush=True)
        return 0
    began = time.monotonic()
    code = 1
    with (run / "run.lock").open("a") as own_lock:
        try:
            fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another supervisor already owns this run; no duplicate was launched.", flush=True)
            return 2
        (run / "run.pid").write_text(str(os.getpid()) + "\n", encoding="utf-8")
        try:
            if stopping(run):
                raise InterruptedError("Stop request is present")
            import data
            mark(run, "preparing", resumed=resume, total_samples=TOTAL_SAMPLES,
                 epoch_buildings_total=10747, epochs=3)
            phase = time.monotonic()
            data.prepare(run)
            durations = read_json(run / "stage_timings.json", {})
            durations["import_seconds"] = time.monotonic() - phase
            write_json(run / "stage_timings.json", durations)
            if stopping(run):
                raise InterruptedError("User requested stop after import")
            with acquire_gpu(run) as snapshot:
                phase = time.monotonic()
                record, selected = autotune(run, snapshot)
                durations["autotune_seconds_this_invocation"] = time.monotonic() - phase
                write_json(run / "stage_timings.json", durations)
                if not (run / "cache_complete.json").exists():
                    mark(run, "caching", selected_config=selected)
                    phase = time.monotonic()
                    cache_code, _ = run_worker(run, "cache", selected)
                    if cache_code:
                        raise RuntimeError(f"Required cache preparation failed with exit code {cache_code}")
                    index = read_json(run / "cache_index.json")
                    if not index or index.get("mode") != selected["cache_mode"]:
                        raise RuntimeError("Cache worker returned success without a matching cache index")
                    if selected["cache_mode"] == "frozen" and index.get("samples") != TOTAL_SAMPLES:
                        raise RuntimeError("Frozen cache index does not cover the complete training schedule")
                    durations["cache_seconds_this_invocation"] = time.monotonic() - phase
                    write_json(run / "stage_timings.json", durations)
                    write_json(run / "cache_complete.json", dict(time=now(), strategy=selected["cache_mode"]))
                phase = time.monotonic()
                code = formal_training(run, record, selected, resume)
                durations["formal_training_seconds_this_invocation"] = time.monotonic() - phase
                write_json(run / "stage_timings.json", durations)
            mark(run, "completed", total_samples=TOTAL_SAMPLES,
                 training_complete=str(run / "training_complete.json"), validation_started=False)
        except InterruptedError as exc:
            code = 130
            mark(run, "stopped", reason=str(exc), validation_started=False,
                 resume_command=f"{PYTHON} {Path(__file__).with_name('run.py')} resume {run}")
            write_json(run / "stopped.json", dict(time=now(), reason=str(exc),
                       complete_update_boundary=True, supervisor_exits=True))
        except BaseException as exc:
            code = 1
            print(traceback.format_exc(), flush=True)
            previous_failure = read_json(run / "worker_failure.json", read_json(run / "failure.json"))
            failure = dict(time=now(), stage=read_json(run / "progress.json", {}).get("stage"),
                           error=str(exc), traceback=traceback.format_exc(),
                           last_worker_failure=previous_failure,
                           current_batch=read_json(run / "current_batch.json"),
                           latest_checkpoint=str(latest_checkpoint(run)) if latest_checkpoint(run) else None,
                           validation_started=False,
                           resume_command=f"{PYTHON} {Path(__file__).with_name('run.py')} resume {run}")
            write_json(run / "failure.json", failure)
            mark(run, "failed", error=str(exc), validation_started=False)
        finally:
            write_json(run / "exit.json", dict(time=now(), supervisor_pid=os.getpid(),
                       exit_code=code, returncode=code,
                       elapsed_seconds=time.monotonic() - began,
                       training_completed=(run / "training_complete.json").exists(),
                       validation_started=False))
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    sys.exit(supervise(arguments.run, arguments.resume))
