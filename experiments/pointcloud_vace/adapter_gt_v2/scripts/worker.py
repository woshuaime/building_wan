"""Isolated CUDA stages. Frozen input caching never includes Adapter features."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import threading
import time
import traceback

from common import ROOT, CACHE, PROMPT, now, write_json, sha, atomic_torch_save, event, nvml_snapshot, runtime_env

SEED = 20261115
TOTAL = 32241
PER_EPOCH = 10747
LOSS_VERSION = "gt-foreground-lt239-dilate3-area4n1-spatial8-balanced-v1"


def learning_rate(consumed):
    warmup = TOTAL * .02
    if consumed <= warmup:
        return 1.e-4 * max(consumed, 1) / warmup
    progress = min(1., (consumed - warmup) / (TOTAL - warmup))
    return 1.e-5 + .5 * (1.e-4 - 1.e-5) * (1 + math.cos(math.pi * progress))


def region_loss(prediction, target, occupancy, time_weights):
    import torch
    if occupancy.shape != (prediction.shape[0], 2, *prediction.shape[2:]):
        raise ValueError(f"Loss-grid mismatch: {occupancy.shape} vs {prediction.shape}")
    error = (prediction.float() - target.float()).square().mean(1)
    areas = occupancy.float().sum((2, 3, 4))
    means = (error[:, None] * occupancy.float()).sum((2, 3, 4)) / areas.clamp_min(1.e-8)
    active = areas > 0
    per_sample = (means * active).sum(1) / active.sum(1).clamp_min(1)
    return (per_sample * time_weights.float().reshape(-1)).mean(), means.detach(), areas.detach()


def rng_state():
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


class MemoryMonitor:
    def __init__(self):
        self.peak = nvml_snapshot()["used_bytes"]
        self.errors = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self):
        while not self.stop_event.wait(.15):
            try:
                self.peak = max(self.peak, nvml_snapshot()["used_bytes"])
            except Exception as exc:
                self.errors.append(str(exc))

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join(timeout=2)
        self.peak = max(self.peak, nvml_snapshot()["used_bytes"])


def read_config(run):
    return json.loads((run / "config.json").read_text())


def cache_root(run):
    path = CACHE / read_config(run)["config_id"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_pipe(run, checkpointing="full"):
    import torch
    import numpy as np
    from integration import attach, parameter_groups, attention_identity
    sys.path.insert(0, str(ROOT / "scripts"))
    from vace_p01 import load_local_vace
    cfg = read_config(run)
    for key, path in cfg["models"]["paths"].items():
        if key == "tokenizer":
            continue
        st = Path(path).stat()
        expected = cfg["models"]["identities"][key]
        if st.st_size != expected["bytes"] or st.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError(f"Model identity changed: {path}")
    for relative, digest in cfg["source_sha256"].items():
        if sha(ROOT / relative) != digest:
            raise ValueError(f"Source identity changed: {relative}")
    verification_path = run / "base_identity.json"
    if not verification_path.exists():
        verified = {}
        for key, expected in cfg["models"]["expected_sha256"].items():
            event(run, "preparing_benchmark", detail="verifying existing local model SHA256", component=key)
            actual = sha(cfg["models"]["paths"][key])
            if actual != expected:
                raise ValueError(f"Official local model SHA256 mismatch: {key} {actual}")
            verified[key] = dict(sha256=actual, **cfg["models"]["identities"][key])
        for name, expected in cfg["models"]["tokenizer_sha256"].items():
            if sha(Path(cfg["models"]["paths"]["tokenizer"]) / name) != expected:
                raise ValueError(f"Tokenizer identity mismatch: {name}")
        write_json(verification_path, dict(time=now(), repository=cfg["models"]["repository"],
                                          revision=cfg["models"]["revision"], files=verified, downloaded=False))
    torch.set_num_threads(4)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is required")
    pipe = load_local_vace(cfg["models"]["paths"], memory_mode="full")
    attach(pipe, SEED, checkpointing)
    pipe.scheduler.set_timesteps(1000, training=True, shift=5)
    adapter, lora = parameter_groups(pipe)
    write_json(run / "optimizer_ownership.json", dict(adapter_parameters=sum(p.numel() for _, p in adapter),
               lora_parameters=sum(p.numel() for _, p in lora),
               trainable_parameters=sum(p.numel() for _, p in adapter + lora),
               frozen_parameters=sum(p.numel() for p in pipe.parameters() if not p.requires_grad),
               trainable_names=[n for n, _ in adapter + lora], attention=attention_identity(),
               original_parameters_frozen=all(not p.requires_grad for n, p in pipe.named_parameters()
                                               if n not in {k for k, _ in adapter + lora})))
    return pipe


def save_increment_files(run, pipe, suffix):
    from safetensors.torch import save_file
    from integration import increments
    a, l = increments(pipe)
    for role, state in (("adapter", a), ("lora", l)):
        path = run / "checkpoints" / f"{role}_{suffix}.safetensors"
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + ".tmp")
        save_file(state, str(temp), metadata=dict(config_id=read_config(run)["config_id"], kind="increment", suffix=suffix))
        os.replace(temp, path)


def text_and_inactive(run, pipe):
    import torch
    from diffsynth.pipelines.wan_video import WanVideoUnit_PromptEmbedder
    directory = cache_root(run)
    path = directory / "shared.pt"
    if path.exists():
        shared = torch.load(path, map_location="cpu", weights_only=True)
    else:
        event(run, "autotuning", detail="encoding the fixed prompt once and shared VACE inactive input")
        with torch.no_grad():
            context = WanVideoUnit_PromptEmbedder().process(pipe, prompt=PROMPT, positive=True)["context"]
            # All-one output mask -> inactive *input* is normalized zero; its VAE encoding is not assumed zero.
            inactive = pipe.vae.encode(torch.zeros((1, 3, 17, 480, 832), device=pipe.device, dtype=torch.bfloat16),
                                       device=pipe.device, tiled=True, tile_size=(30, 52), tile_stride=(15, 26))
        shared = dict(context=context.cpu().contiguous(), inactive=inactive.cpu().to(torch.bfloat16).contiguous())
        atomic_torch_save(path, shared)
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()
    return shared


def encode_raw(pipe, raw):
    import numpy as np
    import torch
    def encode(frames):
        # Copy the full-range grayscale channel, then replicate it. No colorspace/range conversion.
        video = torch.from_numpy(np.ascontiguousarray(frames)).to(device=pipe.device, dtype=torch.bfloat16)
        video = (video * (2 / 255) - 1)[None, None].expand(-1, 3, -1, -1, -1).contiguous()
        with torch.no_grad():
            return pipe.vae.encode(video, device=pipe.device, tiled=True,
                                   tile_size=(30, 52), tile_stride=(15, 26)).to(torch.bfloat16)
    return dict(input_latents=encode(raw["target"]).cpu().contiguous(),
                reactive=encode(raw["partial"]).cpu().contiguous(),
                occupancy=torch.from_numpy(raw["occupancy"]).unsqueeze(0).float(),
                id=raw["id"], frame_ids=raw["frame_ids"], cache_key=raw["cache_key"])


def verify_official_encoding(pipe, raw, value, shared):
    import numpy as np
    import torch
    from PIL import Image
    sys.path.insert(0, str(ROOT / "scripts"))
    from vace_roof_training import encode_sample
    sample = {name: [Image.fromarray(np.repeat(frame[..., None], 3, axis=2)) for frame in raw[key]]
              for name, key in (("gt", "target"), ("conditioning", "partial"))}
    sample["model_mask"] = [Image.new("RGB", (832, 480), (255, 255, 255)) for _ in range(17)]
    settings = dict(tiled=True, tile_size=[30, 52], tile_stride=[15, 26], gradient_checkpointing=True,
                    gradient_checkpointing_offload=False, min_timestep_boundary=0., max_timestep_boundary=1.)
    official = encode_sample(pipe, sample, settings, shared["context"].to(pipe.device), 1.)
    _, constructed, _, _, _ = assemble([value], [raw["geometry"]], shared, pipe.device)
    target_error = float((official["input_latents"].cpu().float() - value["input_latents"].float()).abs().max())
    condition_error = float((official["vace_context"].float() - constructed.float()).abs().max())
    if not torch.allclose(official["input_latents"].cpu().float(), value["input_latents"].float(), rtol=.005, atol=.005):
        raise RuntimeError("Target encoding differs from the official input embedder")
    if not torch.allclose(official["vace_context"].float(), constructed.float(), rtol=.005, atol=.005):
        raise RuntimeError("Shared inactive/reactive/mask cache differs from official VACE condition")
    return dict(passed=True, max_target_error=target_error, max_condition_error=condition_error,
                vace_shape=list(constructed.shape), all_one_mask_preserves_reactive=True)


def entry_path(run, row):
    key = f"{row['sample_index']:05d}_{row['id']}_{row['start']:02d}"
    path = cache_root(run) / "latents" / key[:2] / (key + ".pt")
    return path


def read_manifest(run):
    return {str(r["id"]): r for r in json.loads((run / "train_manifest.json").read_text())}


def geometry_torch(geometry, device="cpu"):
    import numpy as np
    import torch
    return {k: torch.from_numpy(np.ascontiguousarray(v)).float().unsqueeze(0).to(device) for k, v in geometry.items()}


def sample_encoded(run, pipe, row, mode, persist=False):
    import torch
    from data import load_raw, geometry_for, cache_key
    path = entry_path(run, row)
    if mode == "frozen" and path.exists():
        value = torch.load(path, map_location="cpu", weights_only=True)
        if value["id"] != row["id"] or value["frame_ids"] != row["frame_ids"]:
            raise ValueError("Cached building/window mismatch")
        if value["cache_key"] != cache_key(run, row):
            raise ValueError("Cached source/model/prompt/condition identity mismatch")
        geometry = geometry_for(run, row)
    else:
        raw = load_raw(run, row)
        value = encode_raw(pipe, raw)
        geometry = raw["geometry"]
        if persist:
            atomic_torch_save(path, value)
    return value, geometry


def assemble(values, geometries, shared, device):
    import torch
    batch = len(values)
    latent = torch.cat([v["input_latents"] for v in values]).to(device)
    reactive = torch.cat([v["reactive"] for v in values]).to(device)
    inactive = shared["inactive"].to(device).expand(batch, -1, -1, -1, -1)
    mask = torch.ones((batch, 64, 5, 60, 104), dtype=latent.dtype, device=device)
    vace = torch.cat([inactive, reactive, mask], dim=1)
    context = shared["context"].to(device).expand(batch, -1, -1).contiguous()
    occupancy = torch.cat([v["occupancy"] for v in values]).to(device)
    converted = [geometry_torch(v) for v in geometries]
    g = {key: torch.cat([v[key] for v in converted]).to(device) for key in geometries[0]}
    if latent.shape != (batch, 16, 5, 60, 104) or vace.shape != (batch, 96, 5, 60, 104):
        raise ValueError("Encoded VACE shape mismatch")
    return latent, vace, context, occupancy, g


def make_optimizer(pipe):
    import torch
    from integration import parameter_groups
    adapter, lora = parameter_groups(pipe)
    optimizer = torch.optim.AdamW([dict(params=[p for _, p in adapter], name="Adapter"),
                                  dict(params=[p for _, p in lora], name="LoRA")],
                                 lr=1.e-4, betas=(.9, .999), eps=1.e-8, weight_decay=0.)
    return optimizer, adapter, lora


def update(pipe, optimizer, adapter, lora, batch, consumed, update_number):
    import torch
    clean, vace, context, occupancy, geometry = batch
    optimizer.zero_grad(set_to_none=True)
    noisy, targets, times, weights = [], [], [], []
    # Scalar scheduler API is invoked separately; RNG calls retain sample order even when batch changes.
    for i in range(len(clean)):
        index = torch.randint(0, 1000, (1,), device="cpu")
        timestep = pipe.scheduler.timesteps[index].to(device=pipe.device, dtype=torch.bfloat16)
        noise = torch.randn(clean[i:i + 1].shape, device=pipe.device, dtype=torch.bfloat16)
        noisy.append(pipe.scheduler.add_noise(clean[i:i + 1], noise, timestep))
        targets.append(pipe.scheduler.training_target(clean[i:i + 1], noise, timestep))
        times.append(timestep)
        weights.append(pipe.scheduler.training_weight(timestep).to(pipe.device).reshape(1))
    timestep = torch.cat(times)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        tokens = pipe.structure_adapter(geometry)
        prediction = pipe.model_fn(**models, latents=torch.cat(noisy), timestep=timestep, context=context,
                                   vace_context=vace, vace_scale=1., use_gradient_checkpointing=True,
                                   use_gradient_checkpointing_offload=False,
                                   structure_tokens=tokens,
                                   structure_mask=tokens.new_ones((*tokens.shape[:2], 1)),
                                   structure_projection=pipe.structure_adapter.structure_proj)
        loss, terms, areas = region_loss(prediction, torch.cat(targets), occupancy, torch.cat(weights))
    if not bool(torch.isfinite(loss)) or not loss.requires_grad:
        raise RuntimeError("Nonfinite/disconnected loss")
    loss.backward()
    parameters = [p for _, p in adapter + lora]
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    group_norms = {name: sum(float(p.grad.detach().float().square().sum()) for _, p in group if p.grad is not None) ** .5
                   for name, group in (("adapter", adapter), ("lora", lora))}
    upstream = sum(float(p.grad.detach().float().square().sum()) for n, p in adapter
                   if "structure_proj" not in n and p.grad is not None) ** .5
    nonzero_time_weight = bool(torch.cat(weights).sum() > 0)
    if nonzero_time_weight and (group_norms["adapter"] <= 0 or group_norms["lora"] <= 0):
        raise RuntimeError(f"Trainable increment has no gradient: {group_norms}")
    if nonzero_time_weight and update_number >= 5 and upstream <= 0:
        raise RuntimeError("Adapter upstream gradient is absent after nonzero projection updates")
    watched = {}
    if update_number <= 5 and nonzero_time_weight:
        for name, group in (("adapter", adapter), ("lora", lora)):
            parameter = next(p for _, p in group if p.grad is not None and bool(p.grad.abs().max() > 0))
            watched[name] = (parameter, parameter.detach().clone())
    lr = learning_rate(consumed + len(clean))
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.step()
    changed = {name: not torch.equal(parameter.detach(), before) for name, (parameter, before) in watched.items()}
    if changed and not all(changed.values()):
        raise RuntimeError(f"Optimizer did not change a trainable increment: {changed}")
    if any(p.dtype != torch.float32 or not bool(torch.isfinite(p).all()) for p in parameters):
        raise RuntimeError("Invalid increment precision or values")
    if any(p.grad is not None for p in pipe.parameters() if not p.requires_grad):
        raise RuntimeError("Frozen model acquired gradients")
    return dict(loss=float(loss.detach()), region_losses=terms.cpu().tolist(), region_areas=areas.cpu().tolist(),
                grad_norm=float(norm), gradient_norms=group_norms, adapter_upstream_grad_norm=upstream,
                timesteps=timestep.float().cpu().tolist(), scheduler_weights=torch.cat(weights).float().cpu().tolist(),
                scheduler_zero_weight_batch=not nonzero_time_weight, parameters_changed=changed, lr=lr)


def load_clean(run, pipe):
    import torch
    from integration import load_increments_state
    state = torch.load(run / "checkpoints/clean_initialization.pt", map_location="cpu", weights_only=False)
    load_increments_state(pipe, state["adapter"], state["lora"])
    restore_rng(state["rng"])


def prepare_benchmark(run):
    import torch
    from integration import increments, verify_batch
    from data import load_raw
    started = time.monotonic()
    pipe = load_pipe(run)
    shared = text_and_inactive(run, pipe)
    init_path = run / "checkpoints/clean_initialization.pt"
    if not init_path.exists():
        a, l = increments(pipe)
        # Architecture construction uses the historical separate initialization
        # stream; formal diffusion draws start from the requested seed itself.
        import numpy as np
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        atomic_torch_save(init_path, dict(adapter=a, lora=l, rng=rng_state()))
        save_increment_files(run, pipe, "initial")
    schedule = json.loads((run / "schedule.json").read_text())
    # Include several large point sets, determined by records, plus shuffled schedule windows.
    manifest = read_manifest(run)
    train_rows = list(manifest.values())
    def point_count(row):
        if "partial_display_points" in row:
            return row["partial_display_points"]
        rec = json.loads(Path(row.get("record", row.get("record_path"))).read_text(encoding="utf-8-sig"))
        return rec["partial_display_points"]
    largest = {r["id"] for r in sorted(train_rows, key=point_count, reverse=True)[:4]}
    selected = schedule[:12] + [r for r in schedule[:PER_EPOCH] if r["id"] in largest][:4]
    selected = list({r["sample_index"]: r for r in selected}.values())
    write_json(run / "benchmark_samples.json", selected)
    durations, sizes, geometries, values = [], [], [], []
    for index, row in enumerate(selected):
        if (run / "stop.request").exists():
            return
        event(run, "autotuning", detail="preparing real train-window measurement inputs", prepared=index, total=len(selected), building_id=row["id"])
        t = time.monotonic()
        value, geometry = sample_encoded(run, pipe, row, "frozen", persist=True)
        torch.cuda.synchronize()
        durations.append(time.monotonic() - t)
        sizes.append(entry_path(run, row).stat().st_size)
        if index < 2:
            # Read back low-precision tensors and compare with the freshly encoded path.
            loaded = torch.load(entry_path(run, row), weights_only=True, map_location="cpu")
            for key in ("input_latents", "reactive", "occupancy"):
                if not torch.equal(value[key], loaded[key]):
                    raise RuntimeError(f"Frozen cache numerical mismatch: {key}")
            raw = load_raw(run, row)
            repeated = encode_raw(pipe, raw)
            for key in ("input_latents", "reactive"):
                if not torch.allclose(value[key].float(), repeated[key].float(), rtol=.005, atol=.005):
                    raise RuntimeError(f"Online/cache re-encoding mismatch: {key}")
            if index == 0:
                write_json(run / "checks/official_encoding.json", verify_official_encoding(pipe, raw, value, shared))
            values.append(value)
            geometries.append(geometry)
    pipe.vae.to("cpu")
    torch.cuda.empty_cache()
    examples = []
    for value, geometry in zip(values, geometries):
        clean, vace, context, occ, geom = assemble([value], [geometry], shared, pipe.device)
        examples.append(dict(input_latents=clean, vace_context=vace, geometry=geom, occupancy=occ))
    try:
        check = verify_batch(pipe, examples, shared["context"].to(pipe.device))
    except torch.cuda.OutOfMemoryError as exc:
        check = dict(passed=False, reason="batch2 equivalence exhausted memory", error=str(exc))
    write_json(run / "checks/batch_equivalence.json", check)
    load_clean(run, pipe)
    mean_cost = sum(durations) / len(durations)
    mean_bytes = sum(sizes) / len(sizes)
    projected_bytes = int(mean_bytes * TOTAL)
    stat = shutil.disk_usage(CACHE)
    result = dict(time=now(), measured_samples=len(durations), preparation_seconds=time.monotonic() - started,
                  per_sample_encode_seconds=durations, online_encoding_seconds_per_sample=mean_cost,
                  projected_cache_seconds=mean_cost * TOTAL, projected_cache_bytes=projected_bytes,
                  free_bytes=stat.free, cache_fits_disk=projected_bytes + (25 << 30) < stat.free,
                  clean_initialization_ready=True, batch2_check=check,
                  cache_numeric_check="exact disk BF16 roundtrip and repeated online encoding allclose rtol=.005 atol=.005",
                  shared_inactive_latent_nonzero=bool(torch.count_nonzero(shared["inactive"]) > 0),
                  vae_tiling=dict(latent_tile_size=[30, 52], latent_stride=[15, 26], pixel_tile_size=[240, 416], pixel_stride=[120, 208]),
                  loss_definition=LOSS_VERSION)
    write_json(run / "benchmark_preparation.json", result)


def benchmark(run, candidate):
    import torch
    candidate_name = candidate["name"]
    result_path = run / "autotune_candidates" / (candidate_name + ".json")
    result = dict(candidate=candidate, passed=False, warmup_steps=5, measured_steps=20)
    began = time.monotonic()
    budget = min(40 << 30, nvml_snapshot()["total_bytes"] - (6 << 30))
    with MemoryMonitor() as monitor:
        try:
            if candidate["batch_size"] == 2:
                check = json.loads((run / "checks/batch_equivalence.json").read_text())
                if not check["passed"]:
                    raise RuntimeError(f"Batch2 input/output/gradient verification failed: {check}")
            pipe = load_pipe(run, candidate["checkpointing"])
            shared = text_and_inactive(run, pipe)
            load_clean(run, pipe)
            optimizer, adapter, lora = make_optimizer(pipe)
            rows = json.loads((run / "benchmark_samples.json").read_text())
            mode = candidate.get("cache_mode", "frozen")
            if mode == "frozen":
                pipe.vae.to("cpu")
                torch.cuda.empty_cache()
            timings, completed = [], 0
            for step in range(25):
                if (run / "stop.request").exists():
                    raise InterruptedError("User requested stop")
                torch.cuda.synchronize()
                t = time.monotonic()
                if mode == "online":
                    pipe.vae.to(pipe.device)
                selected = [rows[(step * candidate["batch_size"] + i) % len(rows)] for i in range(candidate["batch_size"])]
                pairs = [sample_encoded(run, pipe, r, mode) for r in selected]
                if mode == "online":
                    # Transfer behavior is part of the timing; no cached VAE outputs are substituted.
                    pipe.vae.to("cpu")
                    torch.cuda.empty_cache()
                batch = assemble([v for v, g in pairs], [g for v, g in pairs], shared, pipe.device)
                record = update(pipe, optimizer, adapter, lora, batch, completed, step + 1)
                completed += len(selected)
                torch.cuda.synchronize()
                elapsed = time.monotonic() - t
                if step >= 5:
                    timings.append(elapsed)
                event(run, "autotuning", candidate=candidate_name, step=step + 1, warmup=step < 5,
                      samples_per_second=len(selected) / elapsed, loss=record["loss"],
                      gpu_memory_bytes=nvml_snapshot()["used_bytes"])
                if monitor.peak > budget:
                    raise MemoryError("Whole GPU memory budget exceeded")
                del batch, pairs
            preparation = json.loads((run / "benchmark_preparation.json").read_text())
            throughput = 20 * candidate["batch_size"] / sum(timings)
            cache_cost = preparation["projected_cache_seconds"] if mode == "frozen" else 0.
            # Checkpoint cost is measured with a complete optimizer/RNG state, excluding frozen base.
            from integration import increments
            a, l = increments(pipe)
            t = time.monotonic()
            atomic_torch_save(run / "autotune_candidates" / f"{candidate_name}.save_probe.pt",
                              dict(adapter=a, lora=l, optimizer=optimizer.state_dict(), rng=rng_state()))
            save_seconds = time.monotonic() - t
            (run / "autotune_candidates" / f"{candidate_name}.save_probe.pt").unlink()
            updates = 3 * math.ceil(PER_EPOCH / candidate["batch_size"])
            projected_save = max(updates / 200, (TOTAL / throughput) / 300) * save_seconds
            result.update(passed=(mode == "online" or preparation["cache_fits_disk"]),
                          failure_reason=None if mode == "online" or preparation["cache_fits_disk"] else "projected cache exceeds disk reserve",
                          stable_step_seconds=timings, stable_samples_per_second=throughput,
                          projected_cache_seconds=cache_cost, projected_cache_bytes=preparation["projected_cache_bytes"] if mode == "frozen" else 0,
                          projected_train_seconds=TOTAL / throughput, measured_save_seconds=save_seconds,
                          projected_save_seconds=projected_save,
                          projected_total_seconds=cache_cost + TOTAL / throughput + projected_save,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        except BaseException as exc:
            result.update(error=str(exc), failure_reason=type(exc).__name__, traceback=traceback.format_exc())
            print(result["traceback"], flush=True)
    result.update(peak_nvml_bytes=monitor.peak, memory_budget_bytes=budget,
                  nvml_errors=monitor.errors, elapsed_seconds=time.monotonic() - began)
    if monitor.errors or monitor.peak > budget:
        result.update(passed=False, failure_reason="NVML sampling failed or whole-card memory budget exceeded")
    write_json(result_path, result)
    return 0 if result["passed"] else 1


def cache_all(run, candidate):
    import torch
    if candidate.get("cache_mode", "frozen") != "frozen":
        write_json(run / "cache_index.json", dict(mode="online", frozen_text_and_geometry=True, latent_cache="benchmark subset only"))
        return
    from data import load_raw
    pipe = load_pipe(run)
    text_and_inactive(run, pipe)
    # Encoding does not require resident transformer blocks; release them to CPU.
    pipe.dit.to("cpu")
    pipe.vace.to("cpu")
    pipe.structure_adapter.to("cpu")
    torch.cuda.empty_cache()
    schedule = json.loads((run / "schedule.json").read_text())
    began = time.monotonic()
    index_path = run / "cache_index.jsonl"
    count = 0
    with index_path.open("w", buffering=1) as out:
        for row in schedule:
            if (run / "stop.request").exists():
                return
            path = entry_path(run, row)
            if not path.exists():
                if shutil.disk_usage(CACHE).free < 20 << 30:
                    raise OSError("Disk reserve reached during frozen caching; no inputs removed")
                raw = load_raw(run, row)
                value = encode_raw(pipe, raw)
                atomic_torch_save(path, value)
            count += 1
            out.write(json.dumps(dict(sample_index=row["sample_index"], id=row["id"], frame_ids=row["frame_ids"], path=str(path))) + "\n")
            elapsed = time.monotonic() - began
            if count == 1 or count % 10 == 0:
                event(run, "caching", cached_samples=count, total_samples=TOTAL,
                      stage_remaining_seconds=(TOTAL - count) * elapsed / count,
                      overall_remaining_seconds=(TOTAL - count) * elapsed / count + candidate.get("projected_train_seconds", 0),
                      gpu_memory_bytes=nvml_snapshot()["used_bytes"])
    write_json(run / "cache_index.json", dict(mode="frozen", samples=count, index=str(index_path),
                                             elapsed_seconds=time.monotonic() - began, source_schedule_sha256=sha(run / "schedule.json")))


def train(run, candidate, resume):
    import torch
    from integration import increments, load_increments_state
    cfg = read_config(run)
    pipe = load_pipe(run, candidate["checkpointing"])
    shared = text_and_inactive(run, pipe)
    pipe.vae.to("cpu")
    torch.cuda.empty_cache()
    optimizer, adapter, lora = make_optimizer(pipe)
    schedule_path = run / "schedule.json"
    schedule = json.loads(schedule_path.read_text())
    schedule_digest = sha(schedule_path)
    if len(schedule) != TOTAL:
        raise ValueError("Formal sample budget differs")
    checkpoint_path = run / "checkpoints/resume_latest.pt"
    consumed, updates, elapsed_previous, batch_history = 0, 0, 0., []
    if resume and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if state["config_id"] != cfg["config_id"] or state["schedule_sha256"] != schedule_digest:
            raise ValueError("Resume identity mismatch")
        if state["manifest_sha256"] != sha(run / "train_manifest.json") or state["prompt_sha256"] != sha(run / "prompt.txt"):
            raise ValueError("Resume source manifest or prompt identity changed")
        load_increments_state(pipe, state["adapter"], state["lora"])
        optimizer.load_state_dict(state["optimizer"])
        consumed, updates = state["samples_consumed"], state["optimizer_updates"]
        elapsed_previous = state["elapsed_training_seconds"]
        batch_history = state["batch_history"]
        restore_rng(state["rng"])
        # Remove only uncommitted log rows from a failed worker; checkpoint cursor owns progress.
        path = run / "loss.jsonl"
        if path.exists():
            valid = [line for line in path.read_text().splitlines() if json.loads(line)["optimizer_updates"] <= updates]
            tmp = path.with_suffix(".jsonl.tmp")
            tmp.write_text("".join(line + "\n" for line in valid))
            os.replace(tmp, path)
    else:
        if checkpoint_path.exists() or ((run / "loss.jsonl").exists() and (run / "loss.jsonl").stat().st_size > 0):
            raise ValueError("Existing formal state requires --resume")
        load_clean(run, pipe)
    batch_history.append(dict(time=now(), at_sample=consumed, optimizer_updates=updates, configuration=candidate))
    began = time.monotonic()
    last_save_time, last_saved_updates = began, updates
    mode = candidate.get("cache_mode", "frozen")
    budget = min(40 << 30, nvml_snapshot()["total_bytes"] - (6 << 30))
    torch.cuda.reset_peak_memory_stats()

    def save_checkpoint(epoch_end=None, final=False):
        nonlocal last_save_time, last_saved_updates
        a, l = increments(pipe)
        state = dict(checkpoint_type="complete_increment_optimizer_rng_state", config_id=cfg["config_id"],
                     schedule_sha256=schedule_digest, manifest_sha256=sha(run / "train_manifest.json"),
                     prompt_sha256=sha(run / "prompt.txt"), adapter=a, lora=l,
                     optimizer=optimizer.state_dict(), rng=rng_state(), samples_consumed=consumed,
                     optimizer_updates=updates, epoch=consumed // PER_EPOCH,
                     epoch_sample_cursor=consumed % PER_EPOCH, learning_rate_progress=consumed / TOTAL,
                     selected_config=candidate, batch_history=batch_history,
                     elapsed_training_seconds=elapsed_previous + time.monotonic() - began)
        temp = checkpoint_path.with_name("resume_next.pt")
        atomic_torch_save(temp, state)
        if checkpoint_path.exists():
            previous = run / "checkpoints/resume_previous.pt"
            previous_temp = previous.with_suffix(".copying.pt")
            shutil.copyfile(checkpoint_path, previous_temp)
            with previous_temp.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(previous_temp, previous)
        os.replace(temp, checkpoint_path)
        if epoch_end is not None:
            atomic_torch_save(run / "checkpoints" / f"resume_epoch{epoch_end}.pt", state)
            save_increment_files(run, pipe, f"epoch{epoch_end}")
        if final:
            save_increment_files(run, pipe, "final")
        last_save_time, last_saved_updates = time.monotonic(), updates
        print(f"SAVED samples={consumed}/{TOTAL} optimizer_updates={updates} checkpoint={checkpoint_path}", flush=True)

    with MemoryMonitor() as monitor, (run / "loss.jsonl").open("a", buffering=1) as loss_log:
        # An interrupted artifact export can leave a committed boundary cursor;
        # finish that epoch's exports before advancing its next sample.
        if consumed > 0 and consumed % PER_EPOCH == 0:
            epoch = consumed // PER_EPOCH
            if not all((run / "checkpoints" / name).exists() for name in
                       (f"resume_epoch{epoch}.pt", f"adapter_epoch{epoch}.safetensors", f"lora_epoch{epoch}.safetensors")):
                save_checkpoint(epoch_end=epoch)
        while consumed < TOTAL:
            if (run / "stop.request").exists():
                if updates > 0:
                    save_checkpoint()
                event(run, "stopped", status="stopped", samples_consumed=consumed, optimizer_updates=updates)
                return
            step_began = time.monotonic()
            end = min(consumed + candidate["batch_size"], ((consumed // PER_EPOCH) + 1) * PER_EPOCH, TOTAL)
            rows = schedule[consumed:end]
            write_json(run / "current_batch.json", dict(samples_consumed=consumed, optimizer_updates=updates,
                                                        rows=rows, candidate=candidate))
            if mode == "online":
                pipe.vae.to(pipe.device)
            pairs = [sample_encoded(run, pipe, row, mode) for row in rows]
            if mode == "online":
                pipe.vae.to("cpu")
                torch.cuda.empty_cache()
            batch = assemble([v for v, g in pairs], [g for v, g in pairs], shared, pipe.device)
            record = update(pipe, optimizer, adapter, lora, batch, consumed, updates + 1)
            torch.cuda.synchronize()
            consumed, updates = end, updates + 1
            duration = time.monotonic() - step_began
            elapsed = elapsed_previous + time.monotonic() - began
            speed = (consumed - batch_history[-1]["at_sample"]) / max(time.monotonic() - began, 1.e-6)
            remaining = (TOTAL - consumed) / max(speed, 1.e-8)
            record.update(time=now(), optimizer_updates=updates, samples_consumed=consumed, total_samples=TOTAL,
                          epoch=rows[0]["epoch"] + 1, epoch_completed=consumed - rows[0]["epoch"] * PER_EPOCH,
                          epoch_buildings=PER_EPOCH, ids=[r["id"] for r in rows], frame_ids=[r["frame_ids"] for r in rows],
                          batch_size=len(rows), configured_batch_size=candidate["batch_size"],
                          step_seconds=duration, elapsed_training_seconds=elapsed,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                          peak_nvml_bytes=monitor.peak)
            loss_log.write(json.dumps(record, ensure_ascii=False) + "\n")
            epoch_end = consumed // PER_EPOCH if consumed % PER_EPOCH == 0 else None
            if updates == 1 or updates - last_saved_updates >= 200 or time.monotonic() - last_save_time >= 300 or epoch_end:
                save_checkpoint(epoch_end=epoch_end)
            event(run, "training", status="running", epoch=record["epoch"], epoch_completed=record["epoch_completed"],
                  epoch_buildings=PER_EPOCH, samples_consumed=consumed, total_samples=TOTAL, optimizer_updates=updates,
                  batch_size=candidate["batch_size"], samples_per_second=speed, loss=record["loss"],
                  gpu_memory_bytes=nvml_snapshot()["used_bytes"], peak_nvml_bytes=monitor.peak,
                  stage_remaining_seconds=remaining, overall_remaining_seconds=remaining,
                  latest_checkpoint=str(checkpoint_path) if checkpoint_path.exists() else None,
                  latest_checkpoint_update=last_saved_updates)
            if monitor.errors:
                raise RuntimeError(f"NVML monitoring failed: {monitor.errors}")
            if monitor.peak > budget:
                raise MemoryError("Whole GPU memory budget exceeded")
            del pairs, batch
        if consumed != TOTAL:
            raise RuntimeError("Incomplete formal sample cursor")
        save_checkpoint(final=True)
        final_adapter = run / "checkpoints/adapter_final.safetensors"
        final_lora = run / "checkpoints/lora_final.safetensors"
        from safetensors.torch import load_file
        load_file(str(final_adapter))
        load_file(str(final_lora))
        complete = dict(time=now(), status="completed", epochs=3, samples_consumed=consumed,
                        optimizer_updates=updates, batch_history=batch_history,
                        elapsed_training_seconds=elapsed_previous + time.monotonic() - began,
                        adapter_final=str(final_adapter), lora_final=str(final_lora),
                        adapter_sha256=sha(final_adapter), lora_sha256=sha(final_lora),
                        peak_nvml_bytes=monitor.peak, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(), validation_started=False,
                        visual_effects_verified=False)
        write_json(run / "training_complete.json", complete)
        event(run, "completed", **{k: v for k, v in complete.items() if k not in ("time",)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--mode", required=True, choices=("prepare_benchmark", "benchmark", "cache", "train"))
    parser.add_argument("--candidate", type=json.loads)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    os.environ.update(runtime_env(run))
    (run / "checkpoints").mkdir(exist_ok=True)
    try:
        if args.mode == "prepare_benchmark":
            prepare_benchmark(run)
        elif args.mode == "benchmark":
            return benchmark(run, args.candidate)
        elif args.mode == "cache":
            cache_all(run, args.candidate)
        elif args.mode == "train":
            train(run, args.candidate, args.resume)
        return 0
    except BaseException as exc:
        import torch
        oom = isinstance(exc, (torch.cuda.OutOfMemoryError, MemoryError))
        failure = dict(time=now(), stage=args.mode, error=str(exc), exception=type(exc).__name__,
                       traceback=traceback.format_exc(), oom=oom,
                       current_batch=json.loads((run / "current_batch.json").read_text()) if (run / "current_batch.json").exists() else None)
        write_json(run / "worker_failure.json", failure)
        print(failure["traceback"], flush=True)
        return 42 if oom else 1


if __name__ == "__main__":
    sys.exit(main())
