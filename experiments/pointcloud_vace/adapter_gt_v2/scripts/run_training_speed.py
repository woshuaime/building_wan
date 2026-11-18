"""Benchmark and resume the existing whole-frame run under its original GPU and run locks."""
import json
from pathlib import Path
import subprocess
import sys
import time

from common import PYTHON, ROOT, now, runtime_env, sha, write_json
import pipeline
import run_fullframe

SCRIPT = Path(__file__).resolve()
SOURCES = [SCRIPT, *(SCRIPT.with_name(n) for n in (
    "training_kernels.py", "training_telemetry.py", "training_speed_probe.py"))]


def read(path):
    return json.loads(path.read_text())


def verify(record, encoding_id):
    if record["encoding_id"] != encoding_id or record["profile"] not in ("baseline", "telemetry", "fused"):
        raise ValueError("Training execution choice does not match this whole-frame run")
    if record["source_sha256"] != {str(f.relative_to(ROOT)): sha(f) for f in SOURCES}:
        raise ValueError("Measured training implementation has changed")


def measure(run, parent, path):
    checkpoint = pipeline.latest_checkpoint(run)
    if not checkpoint or not (run / "cache_complete.json").is_file():
        raise ValueError("Training speed check requires completed cache and a full training checkpoint")
    trial = path.parent / ("trial_" + str(time.time_ns()))
    trial.mkdir(parents=True)
    sources = {str(f.relative_to(ROOT)): sha(f) for f in SOURCES}
    checkpoint_sha = sha(checkpoint)
    results = {}
    for profile in ("baseline", "telemetry", "fused"):
        if (run / "stop.request").exists():
            raise InterruptedError("Stop requested during speed comparison")
        pipeline.mark(run, "testing_training_speed", speed_trial=profile)
        args = [PYTHON, "-B", "-u", str(SCRIPT.with_name("training_speed_probe.py")),
                str(run), profile, "--output-dir", str(trial), "--checkpoint", str(checkpoint)]
        with (trial / f"{profile}.log").open("w") as log:
            child = subprocess.Popen(args, cwd=ROOT, env=runtime_env(run), stdout=log,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            (run / "worker.pid").write_text(f"{child.pid}\n")
            code = child.wait()
        if (run / "stop.request").exists():
            raise InterruptedError("Stop requested during speed comparison")
        result = pipeline.read_json(trial / f"{profile}.json", {})
        if code == 0 and result.get("passed") and result.get("checkpoint_sha256") == checkpoint_sha:
            results[profile] = result
        elif profile == "baseline":
            break
    if sha(checkpoint) != checkpoint_sha:
        raise ValueError("Formal checkpoint changed during the isolated comparison")
    chosen = "baseline"
    if results:
        lowest = min(r["mean_step_seconds"] for r in results.values())
        # The original selection policy prefers lower whole-card peak within 5%.
        eligible = [name for name, r in results.items() if r["mean_step_seconds"] <= 1.05 * lowest]
        chosen = min(eligible, key=lambda n: (results[n]["peak_nvml_bytes"], results[n]["mean_step_seconds"]))
        if results[chosen]["mean_step_seconds"] >= results["baseline"]["mean_step_seconds"]:
            chosen = "baseline"
    record = dict(time=now(), encoding_id=parent["encoding_id"], candidate=parent["name"],
                  profile=chosen, trial=str(trial), checkpoint_sha256=checkpoint_sha,
                  source_sha256=sources, measured={n: r["mean_step_seconds"] for n, r in results.items()})
    verify(record, parent["encoding_id"])
    write_json(path, record)
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return record


def main():
    mode, run = sys.argv[1], Path(sys.argv[2]).resolve()
    path = run / "training_speed" / "selection.json"
    # Reuse the cache identity, both locks, full-state recovery and OOM fallback.
    run_fullframe.SCRIPT = SCRIPT
    original = run_fullframe.activate_selection

    def choose(directory, config):
        selected = original(directory, config)
        record = read(path) if path.is_file() else measure(directory, selected, path)
        verify(record, config["config_id"])
        if selected["name"] != record["candidate"]:
            raise ValueError("Speed comparison belongs to another training configuration")
        if record["profile"] == "baseline":
            return selected
        return dict(selected, training_execution=record["profile"],
                    training_execution_sha256=sha(path))

    run_fullframe.activate_selection = choose
    if mode == "train":
        candidate = json.loads(sys.argv[sys.argv.index("--candidate") + 1])
        if candidate.get("training_execution"):
            record = read(path)
            verify(record, candidate["encoding_id"])
            if (candidate["name"] != record["candidate"] or candidate["training_execution"] != record["profile"]
                    or candidate["training_execution_sha256"] != sha(path)):
                raise ValueError("Training worker execution identity mismatch")
            from training_telemetry import install_telemetry
            install_telemetry(run)
            if record["profile"] == "fused":
                sys.path.insert(0, str(ROOT / "code" / "DiffSynth-Studio"))
                from training_kernels import install_rotary
                install_rotary()
    return run_fullframe.main()


if __name__ == "__main__":
    sys.exit(main())
