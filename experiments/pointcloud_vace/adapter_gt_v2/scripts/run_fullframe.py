"""Own the full-frame cache, then continue the authorized three-epoch training."""
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

from common import CACHE, PYTHON, ROOT, now, runtime_env, sha, write_json
import pipeline
import run as launcher

SCRIPT = Path(__file__).resolve()
NAME = "cache_fullframe"


def load(path):
    return json.loads(path.read_text())


def setup(run):
    from data import digest
    from fullframe_cache import config, CONTRACT
    state = run / NAME
    state.mkdir(exist_ok=True)
    path = state / "config.json"
    if not path.exists():
        if any((run / n).exists() for n in ("cache_complete.json", "training_complete.json", "supervisor_recovery.json",
                                             "loss.jsonl", "checkpoints/resume_latest.pt")):
            raise RuntimeError("Full-frame conversion is allowed only before formal training or cache completion")
        original = load(run / "config.json")
        cfg = dict(original)
        cfg["parent_config_id"] = cfg.pop("config_id")
        cfg["parent_config_sha256"] = sha(run / "config.json")
        cfg["parent_shared_sha256"] = sha(CACHE / cfg["parent_config_id"] / "shared.pt")
        cfg["encoding_contract"] = CONTRACT
        cfg["training"] = dict(cfg["training"], tiled=False, tile_size=None, tile_stride=None)
        cfg["schedule_sha256"] = sha(run / "schedule.json")
        cfg["manifest_file_sha256"] = sha(run / "train_manifest.json")
        cfg["prompt_file_sha256"] = sha(run / "prompt.txt")
        cfg["clean_initialization_sha256"] = sha(run / "checkpoints/clean_initialization.pt")
        cfg["source_sha256"] = dict(original["source_sha256"])
        for file in (SCRIPT, SCRIPT.with_name("fullframe_cache.py"), SCRIPT.with_name("cache_encoder.py"),
                     SCRIPT.with_name("cache_acceleration.py"),
                     ROOT / "code/DiffSynth-Studio/diffsynth/utils/state_dict_converters/wan_video_vae.py"):
            cfg["source_sha256"][str(file.relative_to(ROOT))] = sha(file)
            (state / "source").mkdir(exist_ok=True)
            shutil.copyfile(file, state / "source" / file.name)
        cfg["config_id"] = digest(cfg)
        write_json(path, cfg)
    cfg = config(run)
    for f, key in (("config.json", "parent_config_sha256"), ("schedule.json", "schedule_sha256"),
                   ("train_manifest.json", "manifest_file_sha256"), ("prompt.txt", "prompt_file_sha256"),
                   ("checkpoints/clean_initialization.pt", "clean_initialization_sha256")):
        if sha(run / f) != cfg[key]:
            raise ValueError(f"Original run input changed: {f}")
    for f, expected in cfg["source_sha256"].items():
        if sha(ROOT / f) != expected:
            raise ValueError(f"Full-frame source changed: {f}")
    return cfg


def child(run, mode, candidate=None, resume=False):
    command = [PYTHON, "-B", "-u", str(SCRIPT), mode, str(run)]
    if candidate is not None:
        command.extend(["--candidate", json.dumps(candidate, separators=(",", ":"))])
    if resume:
        command.append("--resume")
    start = time.monotonic()
    process = subprocess.Popen(command, cwd=ROOT, env=runtime_env(run), stdin=subprocess.DEVNULL)
    (run / "worker.pid").write_text(str(process.pid) + "\n")
    print(f"Full-frame {mode}, PID {process.pid}", flush=True)
    code = process.wait()
    duration = time.monotonic() - start
    with (run / "worker_history.jsonl").open("a") as stream:
        stream.write(json.dumps(dict(time=now(), phase=mode, encoding="whole_frame", pid=process.pid,
                                     exit_code=code, elapsed_seconds=duration, candidate=candidate)) + "\n")
    if (run / "stop.request").exists() and not (run / "training_complete.json").exists():
        raise InterruptedError("Stop requested for this run")
    return code, duration


def activate_selection(run, cfg):
    state = run / NAME
    selected_path = state / "selected_config.json"
    if selected_path.exists():
        result = load(selected_path)
        if result["encoding_id"] != cfg["config_id"]:
            raise ValueError("Selected cache and training configuration mismatch")
        return result
    original = load(run / "autotune.json")
    qualification = load(state / "qualified.json")
    if qualification["encoding_id"] != cfg["config_id"]:
        raise ValueError("Missing qualification for these training inputs")
    cache_seconds = qualification["seconds_per_window"] * pipeline.TOTAL_SAMPLES
    candidates = []
    for previous in original["candidates"]:
        base = previous["candidate"]
        if not previous["passed"] or base["cache_mode"] != "frozen":
            continue
        trial = dict(base, name=base["name"] + "_fullframe", encoding_id=cfg["config_id"])
        result_path = run / "autotune_candidates" / (trial["name"] + ".json")
        if not result_path.exists():
            code, _ = child(run, "benchmark", trial)
            if code and not result_path.exists():
                raise RuntimeError(f"Training-input benchmark {trial['name']} failed without a result ({code})")
        result = load(result_path)
        if result["candidate"].get("encoding_id") != cfg["config_id"]:
            raise ValueError("Training benchmark belongs to a different cache")
        if result["passed"]:
            result.update(projected_cache_seconds=cache_seconds,
                          projected_cache_bytes=int(qualification["bytes_per_window"] * pipeline.TOTAL_SAMPLES),
                          projected_total_seconds=cache_seconds + result["projected_train_seconds"] + result["projected_save_seconds"],
                          cache_cost_source=str(state / "qualified.json"))
            write_json(result_path, result)
        candidates.append(result)
    winner = pipeline.choose_result(candidates)
    selected = dict(winner["candidate"], encoding_id=cfg["config_id"],
                    selected_time=now(), peak_nvml_bytes=pipeline.candidate_peak(winner),
                    selection_rule="retimed passing frozen training candidates with full-frame inputs; within 5% prefer lower GPU peak")
    for key in ("stable_samples_per_second", "projected_cache_seconds", "projected_train_seconds",
                "projected_save_seconds", "projected_total_seconds"):
        selected[key] = winner[key]
    write_json(state / "autotune.json", dict(time=now(), candidates=candidates, selected=selected,
               parent_autotune_sha256=sha(run / "autotune.json"), benchmark_updates_count_toward_formal_epochs=False))
    write_json(selected_path, selected)
    return selected


def supervise(run):
    state = run / NAME
    code = 1
    started = time.monotonic()
    with (run / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 2
        (run / "run.pid").write_text(str(os.getpid()) + "\n")
        try:
            cfg = setup(run)
            with pipeline.acquire_gpu(run):
                timing = pipeline.read_json(run / "stage_timings.json", {})
                phase = time.monotonic()
                if not (state / "qualified.json").exists():
                    result, _ = child(run, "qualify")
                    if result:
                        raise RuntimeError(f"Full-frame cache qualification failed ({result})")
                    timing["fullframe_qualification_seconds"] = time.monotonic() - phase
                phase = time.monotonic()
                selected = activate_selection(run, cfg)
                if "fullframe_training_input_check_seconds" not in timing:
                    timing["fullframe_training_input_check_seconds"] = time.monotonic() - phase
                write_json(run / "stage_timings.json", timing)
                if not (run / "cache_complete.json").exists():
                    phase = time.monotonic()
                    result, _ = child(run, "cache")
                    if result:
                        raise RuntimeError(f"Full-frame production cache failed ({result})")
                    timing["fullframe_cache_seconds_this_invocation"] = time.monotonic() - phase
                    write_json(run / "stage_timings.json", timing)
                completed = load(run / "cache_complete.json")
                if completed["samples"] != pipeline.TOTAL_SAMPLES or completed["encoding_id"] != cfg["config_id"]:
                    raise ValueError("Full-frame cache coverage or identity mismatch before training")
                # Preserve the pinned V1 supervisor's entire optimizer/RNG
                # recovery policy. Its subprocess callback selects our strict
                # full-frame reader for training and every authorized retry.
                autotune = load(state / "autotune.json")
                pipeline.run_worker = child
                phase = time.monotonic()
                code = pipeline.formal_training(run, autotune, selected, resume=bool(pipeline.latest_checkpoint(run)))
                timing["formal_training_seconds_this_invocation"] = time.monotonic() - phase
                write_json(run / "stage_timings.json", timing)
            pipeline.mark(run, "completed", training_complete=str(run / "training_complete.json"), validation_started=False)
        except InterruptedError as error:
            code = 130
            pipeline.mark(run, "stopped", reason=str(error))
        except BaseException as error:
            code = 1
            failure = dict(time=now(), stage=pipeline.read_json(run / "progress.json", {}).get("stage"), error=str(error),
                           traceback=traceback.format_exc(), validation_started=False, resume_command=f"bash {run / 'resume.sh'}")
            write_json(run / "failure.json", failure)
            pipeline.mark(run, "failed", error=str(error))
            print(failure["traceback"], flush=True)
        finally:
            write_json(run / "exit.json", dict(time=now(), supervisor_pid=os.getpid(), exit_code=code,
                       elapsed_seconds=time.monotonic()-started, training_completed=(run/"training_complete.json").exists(),
                       validation_started=False))
    return code


def launch(run):
    if (run / "training_complete.json").exists():
        pipeline.verify_completion(run)
        return
    if launcher.running(run):
        print("A supervisor already owns this run", flush=True)
        return
    cfg = setup(run)
    state = run / NAME
    history = run / "invocation_history" / str(time.time_ns())
    history.mkdir(parents=True)
    for name in ("exit.json", "failure.json", "stopped.json", "stop.request"):
        path = run / name
        if path.exists():
            path.replace(history / name)
    if not (state / "previous_resume.sh").exists():
        shutil.copyfile(run / "resume.sh", state / "previous_resume.sh")
        for name in ("progress.json", "status.json", "launch.json", "cache_index.jsonl"):
            if (run / name).exists():
                shutil.copyfile(run / name, state / ("previous_" + name))
    # The detached supervisor sets its entire execution environment itself.
    session = launcher.session_name(run)
    cmd = [PYTHON, "-B", "-u", str(SCRIPT), "supervise", str(run)]
    shell = "set -o pipefail\n" + shlex.join(cmd) + " 2>&1 | tee -a " + shlex.quote(str(run/"train.log")) + "\nexec bash --noprofile --norc"
    exists = subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0
    tmux = ["tmux", "new-window", "-t", session] if exists else ["tmux", "new-session", "-d", "-s", session]
    window = subprocess.check_output(tmux + ["-P", "-F", "#{window_id}", "-n", "fullframe", "-c", str(ROOT),
                                      shlex.join(["bash", "--noprofile", "--norc", "-c", shell])], text=True).strip()
    resume = run / "resume.sh"
    temp = resume.with_name("resume.fullframe.tmp")
    temp.write_text("#!/usr/bin/env bash\nexec " + shlex.join([PYTHON, "-B", str(SCRIPT), "launch", str(run)]) + "\n")
    temp.chmod(0o755)
    os.replace(temp, resume)
    for _ in range(80):
        if launcher.running(run):
            break
        if (run / "failure.json").exists():
            raise RuntimeError(load(run / "failure.json")["error"])
        time.sleep(.1)
    if not launcher.running(run):
        raise RuntimeError("Full-frame supervisor did not acquire the run lock")
    receipt = dict(time=now(), run=str(run), tmux_session=session, window="fullframe", window_id=window,
                   supervisor_pid=int((run/"run.pid").read_text()), encoding_id=cfg["config_id"],
                   log=str(run/"train.log"), cache_dir=str(CACHE/cfg["config_id"]),
                   resume_command=f"bash {resume}", stop_command=f"bash {run/'stop.sh'}", independent_of_codex_and_ssh=True)
    write_json(run / "launch.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("launch", "supervise", "qualify", "cache", "benchmark", "train"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--candidate", type=json.loads)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    os.environ.update(runtime_env(run))
    sys.path.insert(0, str(ROOT / "scripts"))
    if args.mode == "launch":
        launch(run)
        return 0
    if args.mode == "supervise":
        return supervise(run)
    from fullframe_cache import cache_windows, config, configure_training_reader
    import worker
    try:
        if args.mode in ("qualify", "cache"):
            cache_windows(run, qualify=args.mode == "qualify")
        else:
            if args.candidate["encoding_id"] != config(run)["config_id"]:
                raise ValueError("Training candidate encoding mismatch")
            configure_training_reader(run)
            if args.mode == "benchmark":
                return worker.benchmark(run, args.candidate)
            worker.train(run, args.candidate, args.resume)
        return 0
    except Exception as error:
        import torch
        oom = isinstance(error, (torch.cuda.OutOfMemoryError, MemoryError))
        failure = dict(time=now(), stage=args.mode, error=str(error), exception=type(error).__name__,
                       traceback=traceback.format_exc(), oom=oom)
        write_json(run / "worker_failure.json", failure)
        print(failure["traceback"], flush=True)
        return 42 if oom else 1


if __name__ == "__main__":
    sys.exit(main())
