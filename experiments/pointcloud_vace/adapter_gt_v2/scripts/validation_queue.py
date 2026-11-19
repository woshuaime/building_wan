"""Generate and verify the final GT-video V2 validation MP4s, one building at a time."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import time

import av

from common import PYTHON, ROOT, runtime_env, sha
from pipeline import acquire_gpu, read_json, verify_completion
from validation_progress import queue_id


def video_identity(run, building_id, completion):
    return (f"GT-V2|{run.name}|{building_id}|{completion['adapter_sha256']}|"
            f"{completion['lora_sha256']}|81f-30s-cfg5-seed20261018-triptych-v1")


def verify_video(path, identity):
    """Decode all 81 frames and check the source, geometry, and presentation timestamps."""
    with av.open(str(path)) as container:
        if container.metadata.get("comment") != identity:
            raise ValueError(f"Video belongs to a different source or configuration: {path}")
        if len(container.streams) != 1 or len(container.streams.video) != 1:
            raise ValueError(f"Expected one video stream: {path}")
        stream = container.streams.video[0]
        if stream.codec_context.name != "h264" or stream.average_rate != 16:
            raise ValueError(f"Expected a 16 fps H.264 video: {path}")
        count = 0
        for frame in container.decode(stream):
            if (frame.width, frame.height) != (2496, 512) or frame.is_corrupt:
                raise ValueError(f"Invalid frame {count} in {path}")
            if frame.pts is None or frame.pts * frame.time_base * 16 != count:
                raise ValueError(f"Unexpected time at frame {count} in {path}")
            count += 1
        if count != 81:
            raise ValueError(f"Expected 81 decoded frames but got {count}: {path}")


def check_inputs(run, ids):
    completion = verify_completion(run)
    if read_json(run / "exit.json", {}).get("exit_code") != 0 or completion["status"] != "completed":
        raise ValueError(f"Training has not exited successfully: {run}")
    for kind in ("adapter", "lora"):
        if sha(run / completion[f"{kind}_final"]) != completion[f"{kind}_sha256"]:
            raise ValueError(f"Final {kind} weight does not match the completion record")
    data = read_json(run / "data_prepared.json")
    manifest = Path(data["data_directory"]) / "manifest.csv"
    if sha(manifest) != data["source_manifest_sha256"]:
        raise ValueError("Validation manifest differs from the training run")
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        validation_ids = {row["id"] for row in csv.DictReader(handle) if row["split_use"] == "validation"}
    if len(ids) != len(set(ids)) or not set(ids) <= validation_ids:
        raise ValueError(f"All building IDs must be distinct validation buildings: {ids}")
    return completion


def video_names(ids):
    return [f"{index:02d}_ID{building_id}.mp4" for index, building_id in enumerate(ids, 1)]


def run_cases(run, output, ids, staging):
    # This one child has the original GPU environment before importing the model.
    # Every building keeps its own inputs, geometry, sampled trajectory, and output.
    from validate_video import load_validation_model, render_building
    completion = verify_completion(run)
    model, speed_state = None, {}
    count = len(ids)
    for index, (building_id, name) in enumerate(zip(ids, video_names(ids)), 1):
        target = output / name
        identity = video_identity(run, building_id, completion)
        if target.exists():
            verify_video(target, identity)
            print(f"[{index}/{count}] 已核验完成的视频 · {name}", flush=True)
            continue
        started = time.monotonic()
        print(f"[{index}/{count}] 建筑 {building_id} 开始采样，已完成 {index - 1} 条", flush=True)
        if model is None:
            model = load_validation_model(run)
        candidate = staging / name
        render_building(run, building_id, candidate, model=model, speed_state=speed_state)
        verify_video(candidate, identity)
        os.link(candidate, target)  # Atomic publication without replacing any existing file.
        candidate.unlink()
        filled = 16 * index // count
        bar = "█" * filled + "░" * (16 - filled)
        print(f"[{index}/{count}] 已完成并核验 {name} · {time.monotonic() - started:.1f} 秒"
              f"  总进度 [{bar}] {index}/{count}", flush=True)


def generate(run, output, ids, gpu_worker=False):
    completion = check_inputs(run, ids)
    output.mkdir(parents=True, exist_ok=True)
    names = video_names(ids)
    if extra := {p.name for p in output.iterdir()} - set(names):
        raise ValueError(f"Output directory contains other files: {sorted(extra)}")
    if all((output / name).is_file() for name in names):
        for name, building_id in zip(names, ids):
            verify_video(output / name, video_identity(run, building_id, completion))
        print(f"已完成并核验 {len(ids)}/{len(ids)} 条视频 · {output}", flush=True)
        return

    if gpu_worker:
        os.environ.update(runtime_env(run))
        with tempfile.TemporaryDirectory(prefix=".gt-v2-validation-", dir=output.parent) as directory:
            staging = Path(directory)
            with acquire_gpu(staging):
                run_cases(run, output, ids, staging)
        return

    command = [PYTHON, "-u", "-B", str(Path(__file__).resolve()), "--run", str(run),
               "--output", str(output), "--ids", *ids, "--gpu-worker"]
    with subprocess.Popen(command, env=runtime_env(run), cwd=ROOT, stdin=subprocess.DEVNULL) as process:
        try:
            code = process.wait()
        except BaseException:
            # Let the lock owner clean up its unpublished video on an operator stop.
            process.terminate()
            process.wait()
            raise
        if code != 0:
            raise subprocess.CalledProcessError(code, command)
    print(f"全部 {len(ids)} 条验证视频生成完毕 · {output}", flush=True)


def show_dashboard(output, ids):
    source = os.environ.get("TMUX_PANE")
    if not source or not os.environ.get("TMUX") or not sys.stdout.isatty():
        return
    dashboard = Path(__file__).with_name("validation_progress.py")
    args = ["--pane", source, "--output", str(output), "--ids", *ids]
    command = "exec " + shlex.join([sys.executable, "-u", "-B", str(dashboard), *args])
    try:
        panes = subprocess.check_output(
            ["tmux", "list-panes", "-t", source, "-F", "#{pane_id}|#{pane_dead}|#{pane_start_command}"],
            text=True, stderr=subprocess.DEVNULL)
        for line in panes.splitlines():
            pane_id, dead, original = line.split("|", 2)
            words = shlex.split(original)
            if len(words) == 1:  # tmux quotes a command received as a single shell string
                words = shlex.split(words[0])
            if str(dashboard) not in words:
                continue
            options = words[words.index(str(dashboard)) + 1:]
            if options[:2] != ["--pane", source]:
                continue
            if dead == "0" and options == args:
                return
            subprocess.run(["tmux", "respawn-pane", "-k", "-t", pane_id, command], check=True)
            break
        else:
            pane_id = subprocess.check_output(
                ["tmux", "split-window", "-d", "-b", "-v", "-l", "7", "-P", "-F", "#{pane_id}",
                 "-t", source, command], text=True).strip()
        for pane in (source, pane_id):
            subprocess.run(["tmux", "set-option", "-p", "-t", pane, "remain-on-exit", "on"], check=True)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"进度面板未打开，采样进度将在本终端显示。{error}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="completed training run")
    parser.add_argument("--output", type=Path, required=True, help="directory for final MP4s")
    parser.add_argument("--ids", nargs="+", required=True, help="ordered validation building IDs")
    parser.add_argument("--gpu-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)

    for signum in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(signum, interrupted)
    run, output = args.run.resolve(), args.output.resolve()
    if args.gpu_worker:
        generate(run, output, args.ids, gpu_worker=True)
        return
    key = queue_id(output, args.ids)
    print(f"GT_V2_QUEUE_BEGIN {key}", flush=True)
    show_dashboard(output, args.ids)
    try:
        generate(run, output, args.ids)
    except BaseException:
        print(f"GT_V2_QUEUE_FAILED {key}", flush=True)
        raise
    print(f"GT_V2_QUEUE_DONE {key}", flush=True)


if __name__ == "__main__":
    main()
