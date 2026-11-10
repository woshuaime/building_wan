"""Independent, exclusive, eight-case D-3000 masked VACE inference queue."""
import argparse
import fcntl
import shlex
import time
import traceback
from common import *


def worker(run, name):
    import numpy as np
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from diffsynth.utils.data import save_video
    from vace_p01 import load_local_vace, inference_arguments
    from vace_p01_results import validate_video
    from integration import load_increments
    from geometry import grouped_occupancy
    from binding import FixedGeometryModelFn
    c = verify_frozen(run)
    case = next(r for r in c["cases"] if r["name"] == name)
    out = Path(case["output_dir"])
    torch.set_num_threads(4)
    if (out / "raw_result.json").exists():
        raise ValueError("This case already has raw predictions; refusing a GPU replay")
    video, mask = decode(case["inputs"]["vace_video.mp4"]), decode(case["inputs"]["vace_video_mask.mp4"])
    white = validate_inputs(video, mask)
    for key, identity in c["models_verified"]["file_identities"].items():
        st = Path(c["models_verified"]["paths"][key]).stat()
        if identity != dict(device=st.st_dev, inode=st.st_ino, bytes=st.st_size, mtime_ns=st.st_mtime_ns):
            raise ValueError("Verified original base model changed: " + key)
    torch.cuda.reset_peak_memory_stats()
    event(out, "loading_D3000", case=name, adapter=c["weights"]["adapter"], lora=c["weights"]["lora"])
    pipe = load_local_vace(c["models_verified"]["paths"], memory_mode="full")
    load_increments(pipe, c["weights"]["adapter"], c["weights"]["lora"])
    if any(p.requires_grad for p in pipe.parameters()):
        raise ValueError("Inference must not optimize any parameter")
    # Prove the actual live modules match both saved increments, not just a filename.
    expected = load_file(c["weights"]["lora"])
    live = {n.removeprefix("vace."): p for n, p in pipe.named_parameters() if n.startswith("vace.") and ".lora_" in n}
    if set(expected) != set(live) or not all(torch.equal(p.detach().cpu(), expected[n]) for n, p in live.items()):
        raise ValueError("D-3000 LoRA was not loaded into the actual VACE attention layers")
    expected_adapter = load_file(c["weights"]["adapter"])
    if not all(torch.equal(v.detach().cpu(), expected_adapter[n]) for n, v in pipe.structure_adapter.state_dict().items()):
        raise ValueError("D-3000 Adapter state differs from the saved increment")
    del expected, expected_adapter, live
    with np.load(case["geometry_cache"]) as archive:
        g = {k: torch.from_numpy(archive[k].copy())[None].to(pipe.device) for k in archive.files}
    gate = torch.from_numpy(grouped_occupancy(white[:, None].astype(np.float32), 16).reshape(1, -1, 1)).to(pipe.device)
    with torch.no_grad():
        tokens = pipe.structure_adapter(g).float()
    if tuple(tokens.shape) != (1, 32760, 64) or tuple(gate.shape) != (1, 32760, 1):
        raise ValueError("Full-orbit structure tokens are misaligned")
    if not bool(torch.isfinite(tokens).all()):
        raise ValueError("Non-finite structure features")
    del g
    bound = FixedGeometryModelFn(tokens, gate, pipe.structure_adapter.structure_proj)
    pipe.model_fn = bound
    core = pipe.vace.base_model.model if hasattr(pipe.vace, "base_model") else pipe.vace
    write(out / "loaded_increments.json", dict(time=now(), weights=c["weights"], sha256=c["weights_sha256"],
        adapter_tensors=24, lora_tensors=240, live_parameters_equal_saved_weights=True,
        observed_geometry=case["geometry"], structure_tokens=list(tokens.shape), structure_mask=list(gate.shape),
        all_parameters_frozen=True, structure_enabled=True, cfg_geometry_shared=True,
        input_video=case["inputs"]["vace_video.mp4"], input_mask=case["inputs"]["vace_video_mask.mp4"],
        original_prompt_used=True, gt_decoded_before_inference=False))

    def progress(timesteps):
        event(out, "diffusion_sampling", case=name, completed_steps=0, total_steps=len(timesteps))
        for i, step in enumerate(timesteps, 1):
            yield step
            torch.cuda.synchronize()
            if bound.calls != 2 * i or not core.structure_last_call["injected_before_first_vace_block"]:
                raise ValueError("Geometry did not enter both actual CFG denoising branches")
            event(out, "diffusion_sampling", case=name, completed_steps=i, total_steps=len(timesteps))
            if i == 1:
                record = dict(time=now(), case=name, worker_pid=os.getpid(), completed_steps=1,
                    total_steps=len(timesteps), gpu=torch.cuda.get_device_name(),
                    actual_denoising_calls=bound.calls, structure_call=core.structure_last_call,
                    adapter_checkpoint=c["weights"]["adapter"], lora_checkpoint=c["weights"]["lora"],
                    allocated_bytes=torch.cuda.memory_allocated(), cfg_scale=5, geometry_shared_across_cfg=True)
                write(out / "sampling_started.json", record)
                if not (run / "sampling_started.json").exists():
                    write(run / "sampling_started.json", record)
    args = inference_arguments(c, [Image.fromarray(f) for f in video], [Image.fromarray(f) for f in mask])
    args.update(prompt=case["prompt"], progress_bar_cmd=progress)
    if pipe.check_resize_height_width(480, 832, 81, verbose=0) != (480, 832, 81):
        raise ValueError("Pipeline would resize the supplied clip")
    print("ACTUAL_MASKED_INPUTS " + json.dumps(case["inputs"], ensure_ascii=False), flush=True)
    event(out, "conditioning", case=name, inference_calls=1, input_video_used_for_initial_latents=False)
    began = time.monotonic()
    prediction = pipe(**args)
    if len(prediction) != 81 or not all(f.size == (832, 480) for f in prediction):
        raise ValueError("Unexpected output frame count or dimensions")
    event(out, "saving_raw", case=name, completed_steps=30)
    (out / "raw_frames").mkdir()
    for i, frame in enumerate(prediction):
        frame.save(out / "raw_frames" / f"{i:03d}.png")
    save_video(prediction, str(out / "raw_vace.mp4"), fps=16, quality=9, ffmpeg_params=["-threads", "1"])
    result = validate_video(out / "raw_vace.mp4", expected_size=(832, 480), expected_frames=81)
    result.update(time=now(), case=name, method="D3000", raw_uncomposited=True, prediction_frames=81,
        denoising_calls=bound.calls, elapsed_sampling_and_saving_seconds=time.monotonic() - began,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        frame080_is_actual_prediction=True, gt_decoded_before_inference=False)
    write(out / "raw_result.json", result)
    event(out, "raw_saved", case=name, raw_result=result, visual_review="pending")


def summarize(run, c):
    import csv
    records = []
    for case in c["cases"]:
        metrics = json.loads((Path(case["output_dir"]) / "metrics.json").read_text())
        records.append(dict(case=case["name"], building_id=case["building_id"], withheld_points=case["withheld_observed_points"], **metrics))
    write(run / "metrics_summary.json", dict(time=now(), cases=records, method="D3000",
        development_only=True, blind_generalization_test=False, visual_review="pending",
        metric_source=c["evaluator"], codec="metrics from decoded raw_vace.mp4, original uploaded evaluator"))
    keys = ["case", "building_id", "withheld_points", "background_falsely_filled_pct", "new_foreground_missed_pct",
            "adjacent_visible_track_hits_pct", "reference_foreground_pct", "prediction_foreground_pct",
            "directional_neighbor_mean_error_pp", "same_pose_mae_0_255"]
    with (run / "metrics_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def supervise(run):
    from vace_p01 import gpu_lock, check_old_training
    from vace_p01_environment import _nvml_snapshot
    began = time.monotonic()
    completed, code, error = [], 1, None
    try:
        with (run / "run.lock").open("a") as own, gpu_lock():
            fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if _nvml_snapshot()["compute_processes"]:
                raise ValueError("A6000 has another compute process")
            check_old_training()
            c = verify_frozen(run)
            with (run / "attempt_started.json").open("x") as f:
                json.dump(dict(time=now(), supervisor_pid=os.getpid()), f)
            (run / "run.pid").write_text(str(os.getpid()) + "\n")
            for case in c["cases"]:
                name, out = case["name"], Path(case["output_dir"])
                if (out / "attempt_started.json").exists():
                    raise ValueError("A case was already launched; GPU replays are forbidden")
                write(out / "attempt_started.json", dict(time=now(), method="D3000"))
                event(run, "case_running", current_case=name, completed_cases=completed, total_cases=8)
                command = [PYTHON, "-u", "-B", str(Path(__file__).resolve()), "worker", str(run), "--case", name]
                with (out / "run.log").open("a") as log:
                    process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                               stderr=subprocess.STDOUT, text=True, bufsize=1)
                    (run / "worker.pid").write_text(str(process.pid) + "\n")
                    (out / "worker.pid").write_text(str(process.pid) + "\n")
                    print(f"GPU_CASE_START case={name} pid={process.pid} log={out / 'run.log'}", flush=True)
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
                    code = process.wait()
                write(out / "gpu_exit.json", dict(time=now(), worker_pid=process.pid, returncode=code,
                    raw_saved=(out / "raw_result.json").exists()))
                if code != 0 or not (out / "raw_result.json").exists():
                    raise RuntimeError(f"{name}: GPU worker exit {code}; see {out / 'run.log'}")
                cpu_env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
                command = [PYTHON, "-u", "-B", str(Path(__file__).with_name("postprocess.py")), str(run), name]
                with (out / "postprocess.log").open("a") as log:
                    result = subprocess.run(command, cwd=ROOT, env=cpu_env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode != 0:
                    raise RuntimeError(f"{name}: raw output retained; CPU media/metrics failed; see {out / 'postprocess.log'}")
                completed.append(name)
                print(f"CASE_COMPLETE {len(completed)}/8 {name}; raw, comparison and metrics saved", flush=True)
                write(run / "progress.json", dict(time=now(), completed_cases=completed, total_cases=8, stage="running"))
            summarize(run, c)
            code = 0
            event(run, "complete_awaiting_visual_review", completed_cases=completed, total_cases=8)
    except BaseException:
        error = traceback.format_exc()
        code = code if code != 0 else 1
        print(error, flush=True)
        event(run, "failed", completed_cases=completed, error=error)
    write(run / "exit.json", dict(time=now(), returncode=code, completed_cases=completed, total_cases=8,
        elapsed_seconds=time.monotonic() - began, error=error, visual_review="pending"))
    return code


def launch(run, hold):
    session = "apc2mesh_D_eval_" + run.name.removeprefix("run_")
    if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
        print(json.dumps(dict(existing_session=session, run=str(run))))
        return
    if (run / "attempt_started.json").exists():
        raise ValueError("This run has already been launched; inspect its state instead of restarting")
    command = [PYTHON, "-u", "-B", str(Path(__file__).resolve()), "supervise", str(run)]
    env = ["CUDA_VISIBLE_DEVICES=0", "PYTHONUNBUFFERED=1", "DIFFSYNTH_SKIP_DOWNLOAD=True", "HF_HUB_OFFLINE=1",
           "TRANSFORMERS_OFFLINE=1", "TOKENIZERS_PARALLELISM=false", "OMP_NUM_THREADS=4", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=4"]
    shell = "exec env " + shlex.join(env + command) + " >> " + shlex.quote(str(run / "run.log")) + " 2>&1"
    if hold:
        shell = "while [ ! -f " + shlex.quote(str(run / "start.ready")) + " ]; do sleep 1; done\n" + shell
    subprocess.run(["tmux", "new-session", "-d", "-s", session, "-c", str(ROOT), shell], check=True)
    pid = int(subprocess.check_output(["tmux", "list-panes", "-t", session, "-F", "#{pane_pid}"], text=True).strip())
    record = dict(time=now(), run=str(run), tmux_session=session, supervisor_pid=pid, command=command,
                  held_before_release=hold, independent_of_ssh_and_codex=True, log=str(run / "run.log"))
    write(run / "launch.json", record)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("launch", "supervise", "worker"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--case")
    parser.add_argument("--hold", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    if args.mode == "launch":
        launch(run, args.hold)
    elif args.mode == "supervise":
        sys.exit(supervise(run))
    else:
        try:
            worker(run, args.case)
        except BaseException:
            trace = traceback.format_exc()
            print(trace, flush=True)
            write(run / "cases" / args.case / "worker_error.json", dict(time=now(), traceback=trace))
            sys.exit(1)
