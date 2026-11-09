"""One formal D run: original BF16 VACE + new structure adapter + attention LoRA."""
import argparse
import gc
import random
import shutil
import sys
import time
import traceback

from common import *


def region_loss(prediction, target, occupancy):
    import torch
    if occupancy.shape[1] != 3 or occupancy.shape[2:] != prediction.shape[2:]:
        raise ValueError("Loss-region layout mismatch")
    if bool((occupancy < 0).any()) or bool((occupancy.sum(1) > 1.00001).any()):
        raise ValueError("Supervision regions must be mutually exclusive fractional areas")
    error = (prediction.float() - target.float()).square().mean(1)
    denominator = occupancy.sum(dim=(2, 3, 4))
    means = (error[:, None] * occupancy).sum(dim=(2, 3, 4)) / denominator.clamp_min(1.e-8)
    total = (means * means.new_tensor([1., 1., .3])).sum(1).mean()
    return total, means.mean(0)


def increments(pipe):
    from integration import parameter_groups
    adapter, lora = parameter_groups(pipe)
    a = {n.removeprefix("structure_adapter."): p.detach().cpu().contiguous().clone() for n, p in adapter}
    l = {n.removeprefix("vace."): p.detach().cpu().contiguous().clone() for n, p in lora}
    return a, l


def save_weights(path, values, metadata):
    from safetensors.torch import save_file
    temporary = path.with_suffix(".writing.safetensors")
    save_file(values, str(temporary), metadata={k: str(v) for k, v in metadata.items()})
    os.replace(temporary, path)


def train(run, attempt, resume=False):
    import numpy as np
    import torch
    from geometry import make_sample, grouped_occupancy
    from integration import attach_trainable_adapter, parameter_groups
    sys.path.insert(0, str(ROOT / "scripts"))
    from vace_p01 import load_local_vace
    from vace_roof_training import encode_sample
    from diffsynth.pipelines.wan_video import WanVideoUnit_PromptEmbedder
    config = json.loads((run / "config.json").read_text())
    for relative, key in (("train_manifest.json", "manifest_sha256"), ("schedule.json", "schedule_sha256"),
                          ("cache/first_batch.npz", "first_batch_sha256")):
        if sha(run / relative) != config["data"][key]:
            raise ValueError(f"Frozen data/schedule changed: {relative}")
    if (run / "prompt.txt").read_text() != config["inference"]["prompt"]:
        raise ValueError("Frozen prompt changed")
    source = Path(__file__).parent
    for name, digest in config["source_sha256"].items():
        if sha(source / name) != digest:
            raise ValueError(f"Frozen training code changed: {name}")
    for name, identity in config["models_verified"]["file_identities"].items():
        st = Path(config["models_verified"]["paths"][name]).stat()
        if identity != dict(device=st.st_dev, inode=st.st_ino, bytes=st.st_size, mtime_ns=st.st_mtime_ns):
            raise ValueError(f"Verified V3 original model changed: {name}")
    settings = dict(config["training"])
    settings["gradient_checkpointing_offload"] = attempt >= 1
    settings["num_frames"] = 13 if attempt >= 2 else 17
    write_json(run / "runtime_config.json", dict(attempt=attempt, training=settings, config_id=config["config_id"]))
    if (run / "first_update.json").exists() and not resume:
        raise ValueError("A successful formal update already exists; refusing a fresh replay")
    torch.set_num_threads(4)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("A6000 CUDA/BF16 required")
    torch.cuda.reset_peak_memory_stats()
    write_json(run / "environment.json", dict(time=now(), python=sys.executable, python_version=sys.version,
               torch_version=torch.__version__, torch_cuda=torch.version.cuda,
               gpu=torch.cuda.get_device_name(0), bf16_supported=torch.cuda.is_bf16_supported(),
               cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), working_directory=str(ROOT)))
    began = time.monotonic()
    event(run, "loading_original_vace", actual_updates=0, attempt=attempt, frames=settings["num_frames"])
    pipe = load_local_vace(config["models_verified"]["paths"], memory_mode="full")
    core = attach_trainable_adapter(pipe, config["seed"])
    pipe.scheduler.set_timesteps(1000, training=True, shift=settings["scheduler_shift"])
    adapter, lora = parameter_groups(pipe)
    named = adapter + lora
    parameters = [p for _, p in named]
    optimizer = torch.optim.AdamW([dict(params=[p for _, p in adapter], name="Adapter"),
                                   dict(params=[p for _, p in lora], name="VACE_attention_LoRA")],
                                  lr=settings["learning_rate"], betas=tuple(settings["adam_betas"]),
                                  weight_decay=settings["weight_decay"], eps=settings["adam_eps"])
    if {id(p) for g in optimizer.param_groups for p in g["params"]} != {id(p) for _, p in named}:
        raise ValueError("Optimizer parameter ownership mismatch")
    ownership = dict(adapter_parameters=sum(p.numel() for _, p in adapter), lora_parameters=sum(p.numel() for _, p in lora),
                     total_trainable=sum(p.numel() for _, p in named),
                     frozen_parameters=sum(p.numel() for p in pipe.parameters() if not p.requires_grad),
                     trainable_names=[n for n, _ in named], optimizer_groups=[g["name"] for g in optimizer.param_groups],
                     original_weights_frozen=True, checkpoint_base=config["models_verified"]["paths"]["checkpoint"])
    write_json(run / "optimizer_ownership.json", ownership)
    print("TRAINABLE_PARAMETER_COUNTS", json.dumps({k: v for k, v in ownership.items() if k != "trainable_names"}), flush=True)
    if not resume:
        initial_adapter, initial_lora = increments(pipe)
        save_weights(run / "checkpoints/adapter_initial.safetensors", initial_adapter, {"group": "D", "update": 0})
        save_weights(run / "checkpoints/lora_initial.safetensors", initial_lora, {"group": "D", "update": 0})
        del initial_adapter, initial_lora
    event(run, "encoding_fixed_prompt", actual_updates=0)
    with torch.no_grad():
        context = WanVideoUnit_PromptEmbedder().process(pipe, prompt=config["inference"]["prompt"], positive=True)["context"]
    pipe.text_encoder.to("cpu")
    pipe.vae.to("cpu")
    torch.cuda.empty_cache()
    atomic_torch_save(run / "cache/text_context.pt", dict(config_id=config["config_id"], context=context.cpu()))
    context = context.to(pipe.device)
    manifest = {r["id"]: r for r in json.loads((run / "train_manifest.json").read_text())}
    schedule = json.loads((run / "schedule.json").read_text())
    completed, history = 0, []
    if resume:
        state = torch.load(run / "checkpoints/resume_latest.pt", map_location="cpu", weights_only=False)
        if state["config_id"] != config["config_id"] or state["runtime_training"] != settings:
            raise ValueError("Resume configuration differs")
        if state["schedule_sha256"] != sha(run / "schedule.json") or state["prompt_sha256"] != sha(run / "prompt.txt"):
            raise ValueError("Resume schedule/prompt identity differs")
        pipe.structure_adapter.load_state_dict(state["adapter"], strict=True)
        with torch.no_grad():
            for name, p in lora:
                p.copy_(state["lora"][name.removeprefix("vace.")])
        optimizer.load_state_dict(state["optimizer"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        completed, history = state["completed_updates"], state["history"]
        if len(history) != completed:
            raise ValueError("Resume history counter mismatch")

    def checkpoint(final=False):
        a, l = increments(pipe)
        state = dict(config_id=config["config_id"], completed_updates=completed, runtime_training=settings,
                     adapter=a, lora=l, optimizer=optimizer.state_dict(), history=history,
                     python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all(), schedule_sha256=sha(run / "schedule.json"),
                     prompt_sha256=sha(run / "prompt.txt"), checkpoint_type="complete_training_state")
        if completed % 100 == 0 or completed == 1:
            atomic_torch_save(run / "checkpoints" / f"resume_step{completed:06d}.pt", state)
        atomic_torch_save(run / "checkpoints/resume_latest.pt", state)
        suffix = "final" if final else f"step{completed:06d}"
        metadata = dict(config_id=config["config_id"], group="D", optimizer_updates=completed,
                        base_checkpoint=config["models_verified"]["paths"]["checkpoint"])
        save_weights(run / "checkpoints" / f"adapter_{suffix}.safetensors", a, metadata)
        save_weights(run / "checkpoints" / f"lora_{suffix}.safetensors", l, metadata)
        print(f"CHECKPOINT update={completed} resume={run / 'checkpoints/resume_latest.pt'}", flush=True)

    log_path = run / "loss.jsonl"
    if resume and log_path.exists():
        shutil.copyfile(log_path, run / "attempts" / f"loss_before_resume_{time.time_ns()}.jsonl")
        log_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in history))
    if not resume and log_path.exists():
        if log_path.stat().st_size:
            raise ValueError("Existing formal loss log cannot be replayed")
    with log_path.open("a", buffering=1) as log:
        for row in schedule[completed:settings["max_updates"]]:
            step_start = time.monotonic()
            update = row["update"]
            frames = list(range(row["start"], row["start"] + settings["num_frames"]))
            event(run, "preparing_training_batch", actual_updates=completed, next_update=update,
                  building_id=row["building_id"], frame_ids=frames)
            if update == 1:
                sample, labels, geometry, audit = load_sample(run / "cache/first_batch.npz")
                sample = {k: v[:settings["num_frames"]] for k, v in sample.items()}
                labels = labels[:settings["num_frames"]]
                for k in ("target_C", "target_R", "target_depths"):
                    geometry[k] = geometry[k][:settings["num_frames"]]
                audit["frame_ids"] = frames
            else:
                sample, labels, geometry, audit = make_sample(manifest[row["building_id"]], frames, row["gap_seed"])
            write_json(run / "samples" / f"update_{update:06d}.json", audit)
            if not labels.any():
                raise ValueError("Training batch has no trusted supervision")
            pipe.vae.to(pipe.device)
            with torch.no_grad():
                inputs = encode_sample(pipe, sample, settings, context, 1.)
            pipe.vae.to("cpu")
            torch.cuda.empty_cache()
            occupancy = torch.from_numpy(grouped_occupancy(labels.astype(np.float32), 8))[None].to(pipe.device)
            masks = np.stack([np.asarray(m)[..., :1].transpose(2, 0, 1) / 255 for m in sample["model_mask"]])
            gate = grouped_occupancy(masks, 16).reshape(1, -1, 1)
            gate = torch.from_numpy(gate).to(pipe.device)
            g = {k: torch.from_numpy(np.asarray(v).copy())[None].to(pipe.device) for k, v in geometry.items()}
            expected = (1, 16, 1 + (settings["num_frames"] - 1) // 4, 60, 104)
            if tuple(inputs["input_latents"].shape) != expected:
                raise ValueError(f"Unexpected latent shape: {inputs['input_latents'].shape}")
            del sample, labels, geometry, masks
            optimizer.zero_grad(set_to_none=True)
            generator = torch.Generator(device=pipe.device).manual_seed(row["noise_seed"])
            timestep = pipe.scheduler.timesteps[[row["timestep_index"]]].to(device=pipe.device, dtype=pipe.torch_dtype)
            noise = torch.randn(inputs["input_latents"].shape, dtype=pipe.torch_dtype, device=pipe.device, generator=generator)
            inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
            target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
            event(run, "forward_backward", actual_updates=completed, next_update=update, building_id=row["building_id"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tokens = pipe.structure_adapter(g)
                models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
                prediction = pipe.model_fn(**models, **inputs, timestep=timestep, structure_tokens=tokens,
                                           structure_mask=gate, structure_projection=pipe.structure_adapter.structure_proj)
                base_loss, terms = region_loss(prediction, target, occupancy)
                time_weight = pipe.scheduler.training_weight(timestep)
                loss = base_loss * time_weight
            if not bool(torch.isfinite(loss)) or not loss.requires_grad:
                raise ValueError("Non-finite or disconnected loss")
            loss.backward()
            if any(p.grad is not None for p in pipe.parameters() if not p.requires_grad):
                raise ValueError("Frozen original parameter has a gradient")
            norm = torch.nn.utils.clip_grad_norm_(parameters, settings["max_grad_norm"], error_if_nonfinite=True)
            watched = {}
            for name, group in (("Adapter", adapter), ("LoRA", lora)):
                selected = next(((n, p) for n, p in group if p.grad is not None and bool(p.grad.abs().max() > 0)), None)
                if update == 1 and selected is None:
                    raise ValueError(f"First formal batch has no {name} gradient")
                if selected:
                    watched[name] = (selected[0], selected[1], selected[1].detach().clone())
            upstream_norm = sum(float(p.grad.detach().float().square().sum()) for n, p in adapter
                                if ".structure_proj." not in n and p.grad is not None) ** .5
            optimizer.step()
            torch.cuda.synchronize()
            changes = {name: not torch.equal(p.detach(), before) for name, (_, p, before) in watched.items()}
            if update == 1 and changes != {"Adapter": True, "LoRA": True}:
                raise ValueError(f"First update did not change both increments: {changes}")
            if not all(bool(torch.isfinite(p).all()) for p in parameters):
                raise ValueError("Non-finite parameter after AdamW")
            completed = update
            step_range = [int(s["step"]) for s in optimizer.state.values()]
            if min(step_range) != update or max(step_range) != update:
                raise ValueError("AdamW step counters diverged")
            record = dict(time=now(), update=update, planned_updates=settings["max_updates"], building_id=row["building_id"],
                          frame_ids=frames, loss=float(loss.detach()), region_losses=terms.detach().float().tolist(),
                          loss_region_order=["known", "real_complement", "weak_mesh"], timestep=float(timestep),
                          time_weight=float(time_weight), grad_norm=float(norm), parameters_changed=changes,
                          adapter_upstream_grad_norm=upstream_norm, weak_target=audit["weak_reliable"],
                          region_area=occupancy.sum((0, 2, 3, 4)).tolist(),
                          optimizer_parameter_step_range=[min(step_range), max(step_range)],
                          adapter_parameters=ownership["adapter_parameters"], lora_parameters=ownership["lora_parameters"],
                          structure_call=core.structure_last_call, step_seconds=time.monotonic() - step_start,
                          elapsed_seconds=time.monotonic() - began, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(), worker_pid=os.getpid(), attempt=attempt)
            history.append(record)
            log.write(json.dumps(record, ensure_ascii=False) + "\n")
            print("FORMAL_UPDATE " + json.dumps(record, ensure_ascii=False), flush=True)
            if completed == 1 or completed % settings["checkpoint_interval"] == 0:
                checkpoint()
            event(run, "training", actual_updates=completed, planned_updates=settings["max_updates"],
                  loss=record["loss"], checkpoint=str(run / "checkpoints/resume_latest.pt"))
            if completed == 1:
                record["resume_checkpoint"] = str(run / "checkpoints/resume_latest.pt")
                record["checkpoint_saved"] = True
                write_json(run / "first_update.json", record)
            del inputs, g, occupancy, gate, noise, target, tokens, prediction, loss, base_loss, terms, watched
            gc.collect()
    checkpoint(final=True)
    event(run, "training_complete_pending_user_evaluation", actual_updates=completed,
          elapsed_seconds=time.monotonic() - began, evaluation_started=False)
    write_json(run / "training_complete.json", dict(time=now(), actual_updates=completed, evaluation_started=False,
               final_adapter=str(run / "checkpoints/adapter_final.safetensors"), final_lora=str(run / "checkpoints/lora_final.safetensors")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--attempt", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    os.environ.update(DIFFSYNTH_SKIP_DOWNLOAD="True", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    try:
        train(run, args.attempt, args.resume)
    except BaseException as exc:
        import torch
        trace = traceback.format_exc()
        print(trace, flush=True)
        oom = isinstance(exc, torch.cuda.OutOfMemoryError)
        write_json(run / "attempts" / f"attempt_{args.attempt}_error.json",
                   dict(time=now(), error=str(exc), traceback=trace, cuda_oom=oom, pid=os.getpid()))
        event(run, "failed", error=str(exc), cuda_oom=oom, attempt=args.attempt)
        sys.exit(42 if oom else 1)
