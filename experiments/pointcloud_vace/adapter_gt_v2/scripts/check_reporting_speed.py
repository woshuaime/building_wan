"""Confirm the reporting-only change with deterministic comparison workers, then resume."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from common import PYTHON, ROOT, now, runtime_env, sha, write_json
import pipeline
from run_training_speed import verify


def deterministic_worker():
    # Flash attention uses split reductions in its normal backward. For these
    # two diagnostic processes ask Torch for repeatability on supported ops.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    from torch.utils._python_dispatch import TorchDispatchMode

    class DeterministicProjectionBackward(TorchDispatchMode):
        """In matched diagnostics, run the small unsupported grid backward on CPU."""
        def __torch_dispatch__(self, op, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if op == torch.ops.aten.grid_sampler_2d_backward.default and args[0].is_cuda:
                host = [a.cpu() if isinstance(a, torch.Tensor) else a for a in args]
                gradients = op(*host, **kwargs)
                return tuple(g.to(args[0].device) if g is not None else None for g in gradients)
            return op(*args, **kwargs)
    import training_speed_probe
    sys.argv.pop(1)
    with DeterministicProjectionBackward():
        return training_speed_probe.main()


def check(run):
    state = run / "training_speed"
    policy = state / "selection.json"
    result = json.loads(policy.read_text())
    verify(result, json.loads((run / "cache_complete.json").read_text())["encoding_id"])
    if result["profile"] != "baseline":
        raise ValueError("This check is for a previously rejected training candidate")
    first = Path(result["trial"])
    display = json.loads((first / "report_io_equivalence.json").read_text())
    if not display["passed"]:
        raise ValueError("Paired reporting replay did not pass")
    trial = state / ("strict_reporting_" + str(time.time_ns()))
    trial.mkdir()
    latest = pipeline.latest_checkpoint(run)
    record = dict(time=now(), checkpoint_sha256=sha(latest), trial=str(trial),
                  source_sha256=sha(Path(__file__)), original_selection=result, passed=False,
                  attention_deterministic=True, diagnostic_grid_backward="cpu")
    for profile in ("baseline", "telemetry"):
        if pipeline.stopping(run):
            raise InterruptedError("Stop requested")
        pipeline.mark(run, "testing_training_speed", speed_trial="repeatable_" + profile)
        command = [PYTHON, "-B", "-u", str(Path(__file__).resolve()), "worker", str(run),
                   profile, "--output-dir", str(trial), "--checkpoint", str(latest)]
        with (trial / (profile + ".log")).open("w") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=runtime_env(run),
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            (run / "worker.pid").write_text(f"{child.pid}\n")
            code = child.wait()
        if pipeline.stopping(run):
            raise InterruptedError("Stop requested")
        measured = pipeline.read_json(trial / (profile + ".json"), {})
        if code or not measured.get("passed"):
            record["failed_profile"] = profile
            break
        if measured["checkpoint_sha256"] != record["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed during check")
        record[profile + "_seconds"] = measured["mean_step_seconds"]
    else:
        record["passed"] = record["telemetry_seconds"] < record["baseline_seconds"]
    write_json(trial / "qualification.json", record)
    if record["passed"]:
        write_json(policy, dict(result, time=now(), profile="telemetry", qualification=str(trial / "qualification.json"),
                                reporting_replay=str(first / "report_io_equivalence.json"),
                                measured=dict(baseline=record["baseline_seconds"], telemetry=record["telemetry_seconds"])))


def handoff(run):
    marker = run / "stop.request"
    previous = json.loads((run / "exit.json").read_text())
    if previous["exit_code"] != 130 or json.loads((run / "progress.json").read_text())["stage"] != "stopped":
        raise ValueError("Requires a cleanly stopped training worker")
    with (run / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker.rename(run / "training_speed" / f"handover_stop_{time.time_ns()}.json")
        try:
            (run / "run.pid").write_text(f"{os.getpid()}\n")
            with pipeline.acquire_gpu(run):
                check(run)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            if not marker.exists():
                subprocess.run(["bash", str(run / "resume.sh")], cwd=ROOT, check=True)


if __name__ == "__main__":
    if sys.argv[1] == "worker":
        sys.exit(deterministic_worker())
    handoff(Path(sys.argv[1]).resolve())
