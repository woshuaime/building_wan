"""Measure a batch-one compute candidate and continue the same three-epoch training run."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

from common import ROOT, PYTHON, now, runtime_env, sha, write_json
import pipeline
import run_fullframe
import run_training_speed
from resume_training_readahead import attach

SCRIPT = Path(__file__).resolve()
TRIAL_WORKER = SCRIPT.with_name("compute_speed_trials.py")


def read(path):
    return json.loads(path.read_text())


def sources():
    files = [SCRIPT, TRIAL_WORKER, SCRIPT.with_name("training_compute_candidate.py"),
             SCRIPT.with_name("check_reporting_speed.py"), *run_training_speed.SOURCES]
    return {str(f.relative_to(ROOT)): sha(f) for f in files}


def timed_pair(run, trial, phase, checkpoint, checkpoint_sha):
    results = []
    for arm in ("baseline", "candidate"):
        if pipeline.stopping(run):
            raise InterruptedError("Stop requested at trial boundary")
        pipeline.mark(run, "testing_training_speed", speed_trial=f"{trial.name}/{phase}/{arm}")
        target = trial / phase
        target.mkdir(parents=True, exist_ok=True)
        cmd = [PYTHON, "-B", "-u", str(TRIAL_WORKER), str(run), str(trial), phase, arm,
               "--checkpoint", str(checkpoint)]
        with (target / (arm + ".log")).open("w") as log:
            process = subprocess.Popen(cmd, cwd=ROOT, env=runtime_env(run), stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT)
            (run / "worker.pid").write_text(f"{process.pid}\n")
            code = process.wait()
        if pipeline.stopping(run):
            raise InterruptedError("Stop requested during isolated trial")
        result = pipeline.read_json(target / (arm + ".json"), {})
        if (code or not result.get("passed") or result.get("checkpoint_sha256") != checkpoint_sha
                or result["peak_nvml_bytes"] > result["memory_budget_bytes"]):
            raise ValueError(f"{phase}/{arm} in {target} failed: {result.get('error', code)}")
        results.append(result)
    return results


def speed_choice(baseline, candidate):
    b, c = baseline["mean_step_seconds"], candidate["mean_step_seconds"]
    if c >= b:
        return False
    return b > c * 1.05 or candidate["peak_nvml_bytes"] <= baseline["peak_nvml_bytes"]


def decision(run, parent):
    root = run / "compute_speed"
    path = root / "selection.json"
    existing = run / "training_speed" / "selection.json"
    if path.exists():
        record = read(path)
        if (record["encoding_id"] != parent["encoding_id"] or record["previous_selection_sha256"] != sha(existing)
                or record["source_sha256"] != sources()):
            raise ValueError("Compute comparison source, cache or previous choice changed")
        return record
    if parent["batch_size"] != 1 or parent["checkpointing"] != "none" or read(existing)["profile"] != "telemetry":
        raise ValueError("Compute trial requires the selected whole-frame B configuration with telemetry")
    trial = root / ("trial_" + str(time.time_ns()))
    trial.mkdir(parents=True)
    checkpoint = pipeline.latest_checkpoint(run)
    if not checkpoint:
        raise RuntimeError("No complete formal training checkpoint")
    checkpoint_sha = sha(checkpoint)
    started = time.monotonic()
    record = dict(time=now(), encoding_id=parent["encoding_id"], previous_selection_sha256=sha(existing),
                  source_sha256=sources(), checkpoint_sha256=checkpoint_sha, trial=str(trial), selected="baseline",
                  selection_rule="faster complete formal steps; prefer lower whole-card peak within 5%; recover trial time")
    try:
        base, opt = timed_pair(run, trial, "normal", checkpoint, checkpoint_sha)
        strict_base, strict_opt = timed_pair(run, trial, "strict", checkpoint, checkpoint_sha)
        pairs = [(base, opt)]
        # Confirm small eligible gains with a second independent normal pair.
        if speed_choice(base, opt) and base["mean_step_seconds"] <= 1.05 * opt["mean_step_seconds"]:
            pairs.append(timed_pair(run, trial / "repeat", "normal", checkpoint, checkpoint_sha))
        b, c = [sum(pair[i]["mean_step_seconds"] for pair in pairs) / len(pairs) for i in (0, 1)]
        gross = (pipeline.TOTAL_SAMPLES - base["cursor"]) * (b-c)
        record.update(cursor=base["cursor"], baseline_seconds=b, candidate_seconds=c,
                      baseline_peak_bytes=max(x["peak_nvml_bytes"] for x,y in pairs),
                      candidate_peak_bytes=max(y["peak_nvml_bytes"] for x,y in pairs),
                      gross_seconds_saved=gross, strict_comparison=strict_opt["comparison"],
                      all_trials_passed=True, normal_pairs=len(pairs))
        if all(speed_choice(x,y) for x,y in pairs) and gross > time.monotonic()-started:
            record["selected"] = "candidate"
    except InterruptedError:
        raise
    except Exception as exc:
        record.update(rejection=str(exc), traceback=traceback.format_exc())
    if sha(checkpoint) != checkpoint_sha or sources() != record["source_sha256"]:
        raise RuntimeError("Frozen checkpoint or comparison sources changed")
    record.update(completed_time=now(), trial_elapsed_seconds=time.monotonic()-started)
    write_json(path, record)
    print(json.dumps(record), flush=True)
    return record


def main():
    mode, run = sys.argv[1], Path(sys.argv[2]).resolve()
    policy_path = run / "compute_speed" / "selection.json"
    original = run_fullframe.activate_selection
    def choose(directory, cfg):
        selected = original(directory, cfg)
        record = decision(directory, selected)
        if record["selected"] == "candidate":
            selected = dict(selected, compute_execution="direct_hints_and_batched_checks",
                            compute_selection_sha256=sha(policy_path))
        return selected
    run_fullframe.activate_selection = choose
    if mode == "train":
        candidate = json.loads(sys.argv[sys.argv.index("--candidate")+1])
        if candidate.get("compute_execution"):
            record = read(policy_path)
            if (record["selected"] != "candidate" or candidate["compute_selection_sha256"] != sha(policy_path)
                    or record["source_sha256"] != sources() or candidate["training_execution"] != "telemetry"
                    or record["encoding_id"] != candidate["encoding_id"]):
                raise ValueError("Compute worker identity mismatch")
            from training_compute_candidate import install_compute
            install_compute()
    # Keep the original telemetry, fullframe reader, complete-state resume, OOM
    # policy, GPU lock and checkpoint writer. Only the measured B worker opts in.
    run_training_speed.SCRIPT = SCRIPT
    code = run_training_speed.main()
    if mode == "launch" and not (run / "training_complete.json").exists():
        attach(run, resume=False)
        temp = run / f"resume.compute.{os.getpid()}.tmp"
        temp.write_text("#!/usr/bin/env bash\nexec " + shlex.join([PYTHON,"-B",str(SCRIPT),"launch",str(run)])+"\n")
        temp.chmod(0o755)
        os.replace(temp, run / "resume.sh")
    return code


if __name__ == "__main__":
    sys.exit(main())
