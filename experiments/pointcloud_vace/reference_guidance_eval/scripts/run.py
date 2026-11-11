"""Start or resume the six formal A/B/C calls in an independent serial tmux queue."""
import argparse
import fcntl
import shlex
import time
import traceback
from runtime import *


def worker(run, group, seed):
    import numpy as np
    import torch
    from PIL import Image
    from diffsynth.utils.data import save_video
    from vace_p01 import load_local_vace, inference_arguments
    from vace_p01_results import validate_video
    from integration import load_increments
    from geometry import grouped_occupancy
    from reference_binding import ReferenceGeometryModelFn
    c = read(run / "config.json")
    output = output_path(run, group, seed)
    if output.exists():
        raise ValueError("A saved prediction exists; inspect it instead of overwriting")
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    with np.load(run / "cache/clip.npz") as archive:
        video, mask = archive["video"], archive["mask"]
    white = mask[..., 0] == 255
    event(run, "loading_model", group=group, seed=seed)
    pipe = load_local_vace(c["models"], memory_mode=c["memory_mode"])
    load_increments(pipe, c["weights"]["adapter"], c["weights"]["lora"])
    with np.load(run / "cache/geometry.npz") as archive:
        g = {key: torch.from_numpy(archive[key].copy())[None].to(pipe.device) for key in archive.files}
    gate = torch.from_numpy(grouped_occupancy(white[:, None].astype(np.float32), 16).reshape(1, -1, 1)).to(pipe.device)
    with torch.no_grad():
        tokens = pipe.structure_adapter(g).float()
    if list(tokens.shape) != c["target_geometry_shape"] or not torch.isfinite(tokens).all():
        raise ValueError("Unexpected 17-frame structure features")
    del g
    reference_count = int(group != "A")
    bound = ReferenceGeometryModelFn(tokens, gate, pipe.structure_adapter.structure_proj, reference_count)
    pipe.model_fn = bound
    core = pipe.vace.base_model.model if hasattr(pipe.vace, "base_model") else pipe.vace
    write(run / "logs" / f"{group}_{seed}.loaded.json", dict(time=now(), weights=c["weights"],
        all_parameters_frozen=not any(p.requires_grad for p in pipe.parameters()),
        adapter_state_tensors=len(pipe.structure_adapter.state_dict()),
        lora_parameter_tensors=sum(".lora_" in name for name, _ in pipe.vace.named_parameters()),
        reference_interface="vace_reference_image", reference_latents=reference_count,
        geometry_frames=c["frame_indices"], tokens=list(bound.tokens.shape), mask=list(bound.mask.shape),
        native_video_noise_shape=[1, 16, 5 + reference_count, 60, 104], cfg_geometry_shared=True))

    def progress(timesteps):
        event(run, "sampling", group=group, seed=seed, completed_steps=0, total_steps=len(timesteps))
        for i, timestep in enumerate(timesteps, 1):
            yield timestep
            torch.cuda.synchronize()
            if bound.calls != 2 * i or not core.structure_last_call["injected_before_first_vace_block"]:
                raise ValueError("Adapter was not used in both actual CFG branches")
            record = dict(time=now(), group=group, seed=seed, worker_pid=os.getpid(), completed_steps=i,
                          total_steps=len(timesteps), denoising_calls=bound.calls,
                          layout=bound.last_layout, structure_call=core.structure_last_call,
                          gpu=torch.cuda.get_device_name(), reference_condition_used=bool(reference_count),
                          reference_file=c["reference"].get(group), frame_indices=c["frame_indices"],
                          allocated_bytes=torch.cuda.memory_allocated())
            if i == 1:
                write(run / "logs" / f"{group}_{seed}.sampling_started.json", record)
                if not (run / "sampling_started.json").exists():
                    write(run / "sampling_started.json", record)
            event(run, "sampling", group=group, seed=seed, completed_steps=i, total_steps=len(timesteps))

    args = inference_arguments(c, [Image.fromarray(frame) for frame in video], [Image.fromarray(frame) for frame in mask])
    args.update(seed=seed, progress_bar_cmd=progress)
    if reference_count:
        with Image.open(run / c["reference"][group]) as image:
            args["vace_reference_image"] = image.convert("RGB").copy()
    if pipe.check_resize_height_width(480, 832, 17, verbose=0) != (480, 832, 17):
        raise ValueError("Pipeline changed the requested clip shape")
    started = time.monotonic()
    with torch.no_grad():
        prediction = pipe(**args)
    if len(prediction) != 17 or any(frame.size != (832, 480) for frame in prediction):
        raise ValueError("Expected exactly 17 real output frames after native reference removal")
    pixels = np.stack([np.asarray(frame) for frame in prediction])
    known = ~white
    known_mae = float(np.abs(pixels.astype(np.float32) - video.astype(np.float32))[known].mean())
    save_video(prediction, str(output), fps=16, quality=9,
               ffmpeg_params=["-threads", "1", "-movflags", "+faststart"])
    result = validate_video(output, expected_size=(832, 480), expected_frames=17)
    result.update(time=now(), group=group, seed=seed, reference=c["reference"].get(group),
                  raw_uncomposited=True, known_region_mae_0_255=known_mae,
                  metric_source="raw model RGB before encoding versus input, mask black only; no GT",
                  denoising_calls=bound.calls, final_layout=bound.last_layout,
                  elapsed_sampling_and_saving_seconds=time.monotonic() - started,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  gt_mesh_hidden_conditioning=False, same_pose_endpoint_metric=False)
    write(result_path(run, group, seed), result)
    event(run, "video_saved", group=group, seed=seed, result=result)


def supervise(run):
    from vace_p01 import gpu_lock
    from vace_p01_environment import _nvml_snapshot
    started = time.monotonic()
    completed, code = [], 1
    try:
        with (run / "run.lock").open("a") as own, gpu_lock():
            fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if _nvml_snapshot()["compute_processes"]:
                raise RuntimeError("The A6000 has another compute process")
            c = read(run / "config.json")
            (run / "run.pid").write_text(str(os.getpid()) + "\n")
            for task in c["tasks"]:
                group, seed = task["group"], task["seed"]
                if result_path(run, group, seed).exists() and output_path(run, group, seed).exists():
                    completed.append(task)
                    continue
                event(run, "task_running", **task, completed_tasks=completed, total_tasks=6)
                command = [PYTHON, "-u", "-B", str(SOURCE / "run.py"), "worker", str(run), "--group", group, "--seed", str(seed)]
                with (run / "logs" / f"{group}_{seed}.log").open("a") as log:
                    process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                               stderr=subprocess.STDOUT, text=True, bufsize=1)
                    (run / "worker.pid").write_text(str(process.pid) + "\n")
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
                    result = process.wait()
                write(run / "logs" / f"{group}_{seed}.exit.json", dict(time=now(), returncode=result))
                if result != 0 or not result_path(run, group, seed).exists():
                    raise RuntimeError(f"{group}/{seed} worker exited {result}; completed videos retained")
                completed.append(task)
                write(run / "progress.json", dict(time=now(), completed_tasks=completed, total_tasks=6))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
            subprocess.run([PYTHON, "-u", "-B", str(SOURCE / "postprocess.py"), str(run)],
                           cwd=ROOT, env=env, check=True)
            code = 0
            event(run, "complete_awaiting_visual_review", completed_tasks=completed, total_tasks=6,
                  report="deferred until user follow-up", new_training_updates=0)
    except BaseException:
        error = traceback.format_exc()
        print(error, flush=True)
        event(run, "failed", completed_tasks=completed, total_tasks=6, error=error)
    write(run / "exit.json", dict(time=now(), returncode=code, completed_tasks=completed, total_tasks=6,
                                  elapsed_seconds=time.monotonic() - started, visual_review="pending", report="pending user follow-up"))
    return code


def launch(run):
    session = "vace_ref_v2_" + run.name.removeprefix("run_")
    if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
        print(json.dumps(dict(existing_session=session, run=str(run))))
        return
    command = [PYTHON, "-u", "-B", str(SOURCE / "run.py"), "supervise", str(run)]
    env = ["CUDA_VISIBLE_DEVICES=0", "PYTHONUNBUFFERED=1", "DIFFSYNTH_SKIP_DOWNLOAD=True", "HF_HUB_OFFLINE=1",
           "TRANSFORMERS_OFFLINE=1", "TOKENIZERS_PARALLELISM=false", "OMP_NUM_THREADS=4", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=4"]
    shell = "exec env " + shlex.join(env + command) + " >> " + shlex.quote(str(run / "run.log")) + " 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, "-c", str(ROOT), shell], check=True)
    pid = int(subprocess.check_output(["tmux", "list-panes", "-t", session, "-F", "#{pane_pid}"], text=True).strip())
    record = dict(time=now(), run=str(run), tmux_session=session, supervisor_pid=pid,
                  command=command, independent_of_codex_and_ssh=True, planned_formal_calls=6)
    write(run / "launch.json", record)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("launch", "supervise", "worker", "status"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--group", choices=("A", "B", "C"))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    run = args.run.resolve()
    if args.mode == "launch":
        launch(run)
    elif args.mode == "supervise":
        sys.exit(supervise(run))
    elif args.mode == "status":
        for name in ("status.json", "progress.json", "exit.json"):
            if (run / name).exists():
                print(name, json.dumps(read(run / name), ensure_ascii=False))
    else:
        worker(run, args.group, args.seed)
