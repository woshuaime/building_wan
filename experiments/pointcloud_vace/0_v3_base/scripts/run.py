"""One V3 three-wall VACE baseline, using the completed V2 configuration."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
import vace_p01 as baseline
from vace_p01_environment import check_environment
from vace_p01_results import validate_video


def now():
    return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def status(run, stage, **extra):
    value = dict(stage=stage, timestamp=now(), pid=os.getpid())
    value.update(extra)
    write(run / "status.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def decode(path):
    import numpy as np
    from PIL import Image
    raw = subprocess.check_output([
        "ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(path),
        "-map", "0:v:0", "-vsync", "0", "-threads", "1", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "pipe:1"])
    pixels = np.frombuffer(raw, dtype=np.uint8).reshape(81, 480, 832, 3)
    return [Image.fromarray(frame) for frame in pixels]


def inputs(run, config):
    import numpy as np
    videos = []
    for name in ("src_video.mp4", "src_mask.mp4"):
        item = config["inputs"][name]
        baseline.require(baseline.sha256(item["path"]) == item["sha256"], "Input changed: " + name)
        videos.append(decode(item["path"]))
    video, mask = videos
    a, m = np.asarray(video), np.asarray(mask)
    baseline.require(np.isin(m, [0, 255]).all() and np.all(m == m[..., :1]), "Mask must be binary RGB")
    white = m[..., 0] == 255
    baseline.require(white.any() and (~white).any(), "Mask must contain generate and preserve regions")
    baseline.require(np.all(a[white] == 127), "White mask region must already be gray 127")
    prompt = (run / "prompt.txt").read_text(encoding="utf-8")
    baseline.require(prompt == config["inference"]["prompt"], "Actual prompt copy changed")
    return video, mask


def prepare(run):
    import numpy as np
    from diffsynth.pipelines.wan_video import WanVideoPipeline
    import torch
    baseline.require(not (run / "config.json").exists(), "Run already prepared")
    source = ROOT / "data/pointcloud_vace/0_v3"
    reference = ROOT / "outputs/pointcloud_vace/0_v2_base/run_20261109_021750/config.json"
    config = json.loads(reference.read_text(encoding="utf-8"))
    config = {"model": config["model"], "inference": config["inference"]}
    config["inference"]["prompt"] = (source / "prompt.txt").read_text(encoding="utf-8")
    baseline.require(config["model"]["lora_checkpoint"] is None, "Reference must have no LoRA")
    config["reference_v2"] = dict(config_path=str(reference), config_sha256=baseline.sha256(reference),
        entry_path=str(ROOT / "experiments/pointcloud_vace/0_v2_base/scripts/run.py"),
        changes="V3 input paths and exact package prompt only; inference and save flow unchanged",
        c_gap_width_m_user_provided=5.72, c_gap_metric_measurement_performed=False)
    shutil.copyfile(source / "prompt.txt", run / "prompt.txt")
    config["inputs"] = {}
    timestamps = []
    for name in ("src_video.mp4", "src_mask.mp4"):
        path = source / name
        config["inputs"][name] = dict(path=str(path), **{k:v for k,v in
            validate_video(path, expected_size=(832, 480), expected_frames=81).items() if k != "path"})
        expected = {"src_video.mp4": "16eb946b679a056a", "src_mask.mp4": "db445e937f7f0e9a"}
        baseline.require(config["inputs"][name]["sha256"].startswith(expected[name]), "Wrong V3 input: " + name)
        probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(path)]))
        timestamps.append([float(f["best_effort_timestamp_time"]) for f in probe["frames"]])
    baseline.require(timestamps[0] == timestamps[1] == [i / 16 for i in range(81)], "Frame timing mismatch")
    video, masks = inputs(run, config)
    pipe = WanVideoPipeline(device="cpu", torch_dtype=torch.float32)
    baseline.require(pipe.check_resize_height_width(480, 832, 81, verbose=0) == (480, 832, 81), "Unexpected resizing")
    # Check the actual pipeline's preprocessing on every frame, with no model weights.
    for v, m in zip(video, masks):
        pv = pipe.preprocess_video([v])
        pm = pipe.preprocess_video([m], min_value=0, max_value=1)
        baseline.require(tuple(pm.shape) == (1, 3, 1, 480, 832), "Pipeline mask shape mismatch")
        baseline.require(torch.isin(pm, torch.tensor([0., 1.])).all().item(), "Pipeline mask values changed")
        baseline.require(torch.allclose(pv[pm == 1], torch.full_like(pv[pm == 1], -1/255), atol=1e-6), "Gray normalization mismatch")
    config["input_check"] = dict(frames=81, width=832, height=480, fps=16,
        aligned_timestamps=timestamps[0], mask_values=[0,255], white_means="generate",
        black_means="retain_context", white_region_rgb=[127,127,127],
        white_pixels_per_frame=[int((np.asarray(m)[...,0] == 255).sum()) for m in masks],
        modifications="none", original_unmasked_video_available=False, cache_used=False)
    config["models_verified"] = baseline.inspect_models(config, verify=True, notify=lambda msg: print(msg, flush=True))
    baseline.require(config["models_verified"]["ready_for_gpu"], "Original model verification failed")
    config["model"]["vace_checkpoint"] = config["models_verified"]["paths"]["checkpoint"]
    config["model"]["checkpoint_status"] = "verified_official_original"
    config["environment"] = check_environment()
    config["created_at"] = now()
    config["git_head_at_prepare"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    config["entry_sha256"] = baseline.sha256(__file__)
    config["outputs"] = dict(raw_video=str(run / "raw_vace.mp4"),
        raw_pngs=str(run / "raw_frames"), postprocessing=False,
        optional_comparison="input_mask_raw.mp4", final_exit="exit.json")
    write(run / "config.json", config)
    status(run, "prepared", inference_calls=0)


def worker(run):
    import torch
    from diffsynth.utils.data import save_video
    from PIL import Image, ImageDraw
    config = json.loads((run / "config.json").read_text())
    baseline.require(config["entry_sha256"] == baseline.sha256(__file__), "Entry changed after preparation")
    baseline.require(config["model"]["lora_checkpoint"] is None, "LoRA is forbidden")
    for name, identity in config["models_verified"]["file_identities"].items():
        st = Path(config["models_verified"]["paths"][name]).stat()
        baseline.require(identity == dict(device=st.st_dev, inode=st.st_ino, bytes=st.st_size, mtime_ns=st.st_mtime_ns), "Model changed: " + name)
    write(run / "launch_environment.json", check_environment())
    video, masks = inputs(run, config)
    status(run, "model_loading", lora=None)
    torch.cuda.reset_peak_memory_stats()
    pipe = baseline.load_local_vace(config["models_verified"]["paths"], memory_mode="full")
    baseline.require(pipe.check_resize_height_width(480, 832, 81, verbose=0) == (480, 832, 81), "Pipeline would resize")
    baseline.require(not any("lora" in n.lower() for model in (pipe.dit, pipe.vace) for n, _ in model.named_parameters()), "Unexpected LoRA parameters")

    def progress(timesteps):
        status(run, "diffusion_sampling", completed_steps=0, total_steps=len(timesteps))
        for i, step in enumerate(timesteps, 1):
            yield step
            # A synchronized completed iteration is stronger evidence than model loading.
            torch.cuda.synchronize()
            status(run, "diffusion_sampling", completed_steps=i, total_steps=len(timesteps))
            if i == 1:
                write(run / "sampling_started.json", dict(timestamp=now(), worker_pid=os.getpid(),
                    completed_steps=1, device=torch.cuda.get_device_name(),
                    allocated_bytes=torch.cuda.memory_allocated(), model=config["model"]["id"], lora=None))

    args = baseline.inference_arguments(config, video, masks)
    args["progress_bar_cmd"] = progress
    print("ACTUAL INPUTS " + json.dumps(config["inputs"]), flush=True)
    print("VACE masked video editing; 81 frames; original prompt; LoRA=None; one pipeline call", flush=True)
    status(run, "conditioning", inference_calls=1)
    prediction = pipe(**args)
    baseline.require(len(prediction) == 81 and all(f.size == (832,480) for f in prediction), "Unexpected prediction dimensions")
    status(run, "saving_raw")
    # Preserve exact quantized model frames before any optional visualization.
    (run / "raw_frames").mkdir()
    for i, frame in enumerate(prediction):
        frame.save(run / "raw_frames" / f"{i:03d}.png")
    save_video(prediction, str(run / "raw_vace.mp4"), fps=16, quality=9, ffmpeg_params=["-threads", "1"])
    report = validate_video(run / "raw_vace.mp4", expected_size=(832,480), expected_frames=81)
    report.update(raw_uncomposited=True, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved())
    write(run / "raw_result.json", report)
    try:
        panels = []
        for i, (v,m,p) in enumerate(zip(video,masks,prediction)):
            panel = Image.new("RGB", (2496,512), "white")
            draw = ImageDraw.Draw(panel)
            for j,(label,frame) in enumerate(zip(("Gray-filled input", "Mask: white=generate", "Raw VACE"), (v,m,p))):
                panel.paste(frame,(832*j,32))
                draw.text((832*j+8,8),f"{label} / {i:03d}",fill="black")
            panels.append(panel)
            if i in (0,20,40,60,80):
                panel.save(run / f"comparison_{i:03d}.png")
        save_video(panels, str(run / "input_mask_raw.mp4"), fps=16, quality=9, ffmpeg_params=["-threads", "1"])
        write(run / "comparison_result.json", validate_video(run / "input_mask_raw.mp4", expected_size=(2496,512), expected_frames=81))
    except Exception:
        error = traceback.format_exc()
        (run / "optional_visualization_error.txt").write_text(error)
        print(error, flush=True)
    status(run, "raw_saved", raw_result=report, visual_review="pending")


def supervise(run):
    with (run / "run.pid").open("x") as stream:
        stream.write(str(os.getpid()) + "\n")
    started = time.monotonic()
    code = 1
    error = None
    try:
        with baseline.gpu_lock():
            baseline.check_old_training()
            # Exclusive marker prevents replay even after a failed attempt.
            with (run / "attempt_started.json").open("x") as stream:
                json.dump(dict(timestamp=now(), supervisor_pid=os.getpid()), stream)
            process = subprocess.Popen([sys.executable, "-u", "-B", __file__, "worker", str(run)], cwd=ROOT)
            (run / "worker.pid").write_text(str(process.pid) + "\n")
            print(f"SUPERVISOR PID={os.getpid()} WORKER PID={process.pid} SESSION={os.getsid(0)}", flush=True)
            code = process.wait()
    except BaseException:
        error = traceback.format_exc()
        print(error, flush=True)
    result = dict(timestamp=now(), worker_returncode=code, elapsed_seconds=time.monotonic()-started,
        raw_saved=(run / "raw_result.json").is_file(), error=error)
    write(run / "exit.json", result)
    status(run, "complete_awaiting_visual_review" if code == 0 and result["raw_saved"] else "failed", **result)
    return 0 if code == 0 and result["raw_saved"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "supervise", "worker"])
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ.update(DIFFSYNTH_SKIP_DOWNLOAD="True", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false", CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1")
    if args.mode == "prepare":
        prepare(args.run.resolve())
    elif args.mode == "worker":
        worker(args.run.resolve())
    else:
        sys.exit(supervise(args.run.resolve()))
