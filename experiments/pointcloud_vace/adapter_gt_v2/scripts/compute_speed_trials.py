"""Checkpoint-matched trials for the training-compute candidate, inside the run's GPU lock."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import sys
import traceback
from unittest.mock import patch

from common import ROOT, now, runtime_env, sha, write_json
import training_speed_probe as probe


def warm_trial_inputs(run, checkpoint):
    """Prime the same immutable files for all trials as the formal CPU reader does."""
    import torch
    import data
    from fullframe_cache import entry

    state = torch.load(checkpoint, weights_only=False, map_location="cpu")
    cursor = state["samples_consumed"]
    schedule = json.loads((run / "schedule.json").read_text())
    if state["schedule_sha256"] != sha(run / "schedule.json"):
        raise ValueError("Checkpoint schedule changed")
    rows = schedule[cursor:cursor + probe.WARMUP + probe.TIMED]
    cache = json.loads((run / "cache_complete.json").read_text())
    if cache["encoding_id"] != state["config_id"] or cache["schedule_sha256"] != state["schedule_sha256"]:
        raise ValueError("Frozen cache and checkpoint identities differ")
    if len(rows) != probe.WARMUP + probe.TIMED:
        raise ValueError("Insufficient remaining schedule for paired timing")
    directory = Path(cache["cache_dir"])
    geometry = data.directories(run)[1] / "geometry"
    buffer = bytearray(256 * 1024)
    total_bytes = 0
    for row in rows:
        if (run / "stop.request").exists():
            raise InterruptedError("Stop requested before timing")
        key = data.cache_key(run, dict(row, frame_ids=[]))
        for path in (entry(directory, row), geometry / key[:2] / (key + ".npz")):
            with path.open("rb", buffering=0) as stream:
                while count := stream.readinto(buffer):
                    total_bytes += count
    return {"cursor": cursor, "files": len(rows) * 2, "bytes": total_bytes}


def compare_pair(original_compare, phase, reference, actual, baseline, candidate):
    """Record stochastic variability separately from deterministic gradient equivalence."""
    import torch

    report = original_compare(reference, actual, baseline, candidate)
    report["numeric_tolerance_passed"] = report["passed"]
    report["trajectory_exact"] = baseline["trajectory"] == candidate["trajectory"]
    report["endpoint_exact"] = {name: torch.equal(value, actual[name]) for name, value in reference.items()}
    report["same_inputs_and_rng"] = (
        report["matching_rng"]
        and baseline["checkpoint_sha256"] == candidate["checkpoint_sha256"]
        and baseline["cursor"] == candidate["cursor"]
        and all(all(x[k] == y[k] for k in ("sample_index", "id", "timesteps", "lr"))
                for x, y in zip(baseline["trajectory"], candidate["trajectory"], strict=True))
    )
    # Normal mode scores throughput. The strict pair controls numerical acceptance.
    report["passed"] = bool(report["same_inputs_and_rng"]
                            and all(math.isfinite(v) for v in report["endpoint_relative_l2"].values())
                            and (phase == "normal" or report["numeric_tolerance_passed"]))
    return report


def trial(args, directory, result):
    import torch
    from training_telemetry import install_telemetry

    run, arm, phase, checkpoint = args.run, args.arm, args.phase, args.checkpoint
    result["warmed_inputs"] = warm_trial_inputs(run, checkpoint)
    install_telemetry(directory / arm)
    if arm == "candidate":
        import training_compute_candidate as compute
        compute.install_compute()

    original_compare = probe.compare

    def comparison(reference, actual, baseline, candidate):
        return compare_pair(original_compare, phase, reference, actual, baseline, candidate)

    with patch.object(probe, "compare", comparison):
        probe.probe(run, arm, directory, checkpoint, result)

    expected_steps = probe.WARMUP + probe.TIMED
    if len(result["trajectory"]) != expected_steps or len(result["stable_step_seconds"]) != probe.TIMED:
        raise RuntimeError("Incomplete measured training steps")
    if not all(math.isfinite(row[key]) for row in result["trajectory"]
               for key in ("loss", "grad_norm", "adapter_grad", "lora_grad", "lr")):
        raise ValueError("Non-finite training trajectory")
    if not all(math.isfinite(t) and t > 0 for t in result["stable_step_seconds"]):
        raise ValueError("Invalid step duration")
    if arm == "candidate":
        result["candidate_activity"] = compute.activity()
        counts = result["candidate_activity"]
        if (counts["vace_forwards"] != expected_steps or counts["diagnostic_updates"] != expected_steps
                or counts["fallback_updates"] != 0):
            raise ValueError(f"Candidate did not execute through every measured update: {counts}")
    import numpy as np
    result["host_rng_hashes"] = [hashlib.sha256(pickle.dumps(state, protocol=4)).hexdigest()
                                 for state in (random.getstate(), np.random.get_state())]
    if arm == "candidate":
        baseline = json.loads((directory / "baseline.json").read_text())
        if baseline["host_rng_hashes"] != result["host_rng_hashes"]:
            raise ValueError("Python or NumPy RNG diverged")
    result["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    if result["deterministic_algorithms"] != (phase == "strict"):
        raise RuntimeError("Trial did not use its designated numerical mode")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("trial_dir", type=Path)
    parser.add_argument("phase", choices=("normal", "strict"))
    parser.add_argument("arm", choices=("baseline", "candidate"))
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    args.run = args.run.resolve()
    args.checkpoint = (args.checkpoint or args.run / "checkpoints/resume_latest.pt").resolve()
    directory = args.trial_dir.resolve() / args.phase
    if directory.parent == args.run:
        raise ValueError("Use a dedicated trial directory")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (args.arm + ".json")
    if path.exists() or (directory / args.arm).exists():
        raise FileExistsError("Refusing to replace an existing isolated trial")
    os.environ.update(runtime_env(args.run))
    sys.path.insert(0, str(ROOT / "scripts"))
    result = dict(time=now(), passed=False, phase=args.phase, arm=args.arm, telemetry_enabled=True,
                  checkpoint=str(args.checkpoint))
    try:
        if args.phase == "strict":
            from check_reporting_speed import deterministic_worker
            # This established diagnostic calls probe.main() inside its deterministic
            # attention / CPU grid-sampling backward context. Run our matched arm there.
            with patch.object(sys, "argv", [__file__, "worker"]), \
                 patch.object(probe, "main", lambda: trial(args, directory, result)):
                deterministic_worker()
        else:
            trial(args, directory, result)
    except Exception as error:
        result.update(passed=False, error=str(error), traceback=traceback.format_exc())
    write_json(path, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("trajectory", "stable_step_seconds")}), flush=True)
    return int(not result["passed"])


if __name__ == "__main__":
    sys.exit(main())
