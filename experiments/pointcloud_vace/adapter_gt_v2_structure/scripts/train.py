"""Joint training with bounded lookahead and complete checkpoint recovery."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from types import MethodType

import numpy as np
import torch
from safetensors.torch import save_file

from structure_data import (BASE, EPOCHS, SEED, TRAIN_COUNT, VALIDATION_IDS, PreparedData,
                            data, now, sha, write_json)
import integration
import validate_video as video
from cache_encoder import make_encoder
from common import atomic_torch_save
from frame_model import BuildingAdapter
from objectives import (CROP, RGB_EVERY, camera_tensor, choose_rgb_window, decode_pixels,
                        framework_loss, observed_tensor, point_loss, rgb_losses, target_tensor)
from training_compute_candidate import _direct_forward
from worker import region_loss

CHECKPOINT_EVERY = 64
PREFETCH = 4
GROUP_RATES = dict(geometry=1.e-4, adapter=3.e-5, lora=1.e-5)


def learning_rate(step, initial):
    # Extending after epoch 3 keeps its optimizer and final LR. Phase 2 decays
    # continuously from that LR and never restarts the original warmup.
    boundary = EPOCHS * TRAIN_COUNT
    if step <= 256:
        return initial * step / 256
    if step <= boundary:
        fraction = (step - 256) / (boundary - 256)
        return initial * (.2 + .4 * (1 + math.cos(math.pi * fraction)))
    fraction = min(1., (step - boundary) / (2 * TRAIN_COUNT))
    return initial * (.1 + .05 * (1 + math.cos(math.pi * fraction)))


def rng_state():
    state = list(np.random.get_state())
    state[1] = torch.from_numpy(state[1].astype(np.int64))
    return dict(python=random.getstate(), numpy=state, cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state["python"])
    numpy = list(state["numpy"])
    numpy[1] = numpy[1].numpy().astype(np.uint32)
    np.random.set_state(tuple(numpy))
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state_all(state["cuda"])


def tensor_state(module):
    return {name: value.detach().cpu().contiguous() for name, value in module.state_dict().items()}


def load_state(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def make_model(training=False):
    pipe, core, cfg = video.load_validation_model(BASE)
    if training:
        pipe.structure_adapter.requires_grad_(True).train()
        for name, parameter in pipe.vace.named_parameters():
            if ".lora_" in name:
                parameter.requires_grad_(True)
    torch.manual_seed(SEED)
    model = BuildingAdapter().to(pipe.device)
    return (pipe, core, cfg), model


class RollingWindows:
    """Overlaps CPU reads with GPU work. Retain at most one checkpoint interval of latents."""
    def __init__(self, run, prepared, encode):
        identifier = hashlib.sha256(str(run).encode()).hexdigest()[:20]
        self.root = Path("/tmp") / f"building-wan-structure-{os.getuid()}-{identifier}"
        self.root.mkdir(mode=0o700, exist_ok=True)
        self.prepared, self.encode = prepared, encode

    def build(self, index, building, frames, needs_rgb):
        path = self.root / f"{index:06d}.pt"
        encoded = load_state(path) if path.is_file() else None
        if encoded is not None and (encoded["id"] != building or encoded["frame_ids"] != frames
                                    or encoded["dataset"] != self.prepared.summary["schedule_sha256"]):
            raise ValueError(f"Rolling cache identity mismatch for {building}/{frames}")
        raw = self.prepared.window(building, frames, pixels=encoded is None or needs_rgb)
        raw["geometry"] = geometry_for_window(raw["observed"], frames)
        if encoded is None:
            raw.update(occupancy=data.occupancy_for(raw["target"]), cache_key=str(index))
        raw["sample_index"] = index
        return raw, encoded

    def extract(self, index, raw, encoded):
        if encoded is None:
            with torch.no_grad():
                encoded = self.encode(raw)
            encoded["dataset"] = self.prepared.summary["schedule_sha256"]
            atomic_torch_save(self.root / f"{index:06d}.pt", encoded)
        return encoded

    def discard_committed(self, count):
        for path in self.root.glob("*.pt"):
            if path.stem.isdecimal() and int(path.stem) < count:
                path.unlink()


def geometry_for_window(observed, frames):
    # Reuse the V2 adapter's observed-point feature layout and projection.
    centers, rotations = observed["centers"], observed["rotations"]
    intrinsics = observed["intrinsics"]
    points = observed["xyz"]
    scale = float(observed["scale"])
    anchors = points[data.fps(points)]
    if len(anchors) < 128:
        anchors = anchors[np.arange(128) % len(anchors)]
    from scipy.spatial import cKDTree

    density = cKDTree(points).query_ball_point(anchors, .5, return_length=True)
    references = list(range(0, 80, 10))
    base = dict(_points_f64=points, _centers_f64=centers, _rotations_f64=rotations,
                _K_f64=intrinsics, _scale=scale, anchors=(anchors / scale).astype(np.float32),
                anchor_density=np.log1p(density).astype(np.float32),
                ref_C=(centers[references] / scale).astype(np.float32),
                ref_R=rotations[references].astype(np.float32),
                depth_tolerance=np.asarray(.45 / scale, dtype=np.float32), K=intrinsics.astype(np.float32))
    base["reference_views"] = np.stack([data.depth_view(points, centers[f], rotations[f], intrinsics, scale)
                                        for f in references])
    return data._window_geometry(base, frames)


class Trainer:
    def __init__(self, run, prepared):
        from diffsynth.pipelines.wan_video import WanVideoUnit_PromptEmbedder

        self.run, self.prepared = run, prepared
        (self.pipe, self.core, cfg), self.model = make_model(training=True)
        self.pipe.scheduler.set_timesteps(1000, training=True, shift=5)
        torch.manual_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED)
        self.device = self.pipe.device
        self.model_args = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        shared_path = run / ".work/shared.pt"
        if not shared_path.exists():
            with torch.no_grad():
                context = WanVideoUnit_PromptEmbedder().process(
                    self.pipe, prompt=cfg["prompt"], positive=True)["context"]
                zeros = torch.zeros((1, 3, 17, 480, 832), device=self.device, dtype=torch.bfloat16)
                inactive = self.pipe.vae.encode(zeros, device=self.device, tiled=False)
            atomic_torch_save(shared_path, dict(context=context.cpu(), inactive=inactive.cpu()))
        shared = load_state(shared_path)
        self.context, self.inactive = (shared[key].to(self.device) for key in ("context", "inactive"))
        self.mask = torch.ones((1, 64, 5, 60, 104), device=self.device, dtype=torch.bfloat16)
        self.pipe.text_encoder.to("cpu")
        self.pipe.vae.model.clear_cache()
        torch.cuda.empty_cache()
        self.encoder = make_encoder(self.pipe.vae, "whole_frame")
        adapter, lora = integration.parameter_groups(self.pipe)
        self.groups = dict(geometry=list(self.model.parameters()), adapter=[p for _, p in adapter],
                           lora=[p for _, p in lora])
        self.trainable = [p for parameters in self.groups.values() for p in parameters]
        self.optimizer = torch.optim.AdamW(
            [dict(name=key, params=values, lr=GROUP_RATES[key]) for key, values in self.groups.items()],
            betas=(.9, .999), eps=1.e-8, weight_decay=0., fused=True)
        self.available_sigma = (self.pipe.scheduler.sigmas > 0) & (self.pipe.scheduler.linear_timesteps_weights > 0)
        self.rgb_sigma = torch.where(self.available_sigma & (self.pipe.scheduler.sigmas <= .45)
                                     & (self.pipe.scheduler.sigmas >= .15))[0]

    def denoise(self, raw, encoded, index, rgb=False, review=False):
        pipe = self.pipe
        device = pipe.device
        clean, reactive, occupancy = (encoded[key].to(device) for key in ("input_latents", "reactive", "occupancy"))
        condition = torch.cat((self.inactive, reactive, self.mask), dim=1)
        sampler = torch.Generator().manual_seed(SEED + index)
        noise_rng = torch.Generator(device=device).manual_seed(SEED + index)
        choice = self.rgb_sigma[torch.randint(len(self.rgb_sigma), (), generator=sampler)] if rgb else (
            torch.randint(0, len(self.available_sigma), (), generator=sampler))
        sigma = pipe.scheduler.sigmas[choice]
        # Use a single matched coefficient for x_t, the flow target and x_0.
        timestep = pipe.scheduler.timesteps[choice].reshape(1).to(device=device, dtype=torch.bfloat16)
        weight = pipe.scheduler.linear_timesteps_weights[choice].reshape(1).to(device)
        noise = torch.randn(clean.shape, dtype=clean.dtype, device=device, generator=noise_rng)
        noisy = (1. - sigma) * clean + sigma * noise
        target = noise - clean
        geometry = {key: torch.as_tensor(value, device=device)[None]
                    for key, value in raw["geometry"].items()}
        xyz = observed_tensor(raw["observed"], device)
        cam = camera_tensor(raw["camera"], device)
        # Only the infrequent RGB branch retains VAE backward activations. Use
        # the original gradient-checkpointed VACE path for those updates.
        memory_mode = "full" if rgb and not review else "none"
        integration.configure_checkpointing(pipe, memory_mode)
        self.core.forward = MethodType(integration.structured_vace_forward if memory_mode == "full"
                                       else _direct_forward, self.core)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            building = self.model(xyz)
        targets = target_tensor(raw, device)
        # Finish the small CPU corner assignment before dispatching the long
        # VACE forward. No synchronization on the denoiser output is needed.
        terms = dict(framework=framework_loss(building, targets), distribution=point_loss(building, targets))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hints = self.model.project(building, cam)
            tokens = pipe.structure_adapter(geometry)
            projected = pipe.structure_adapter.structure_proj(tokens) + hints
            prediction = integration.model_fn_wan_video(
                **self.model_args, latents=noisy, timestep=timestep, context=self.context,
                vace_context=condition, vace_scale=1., use_gradient_checkpointing=memory_mode == "full",
                structure_tokens=tokens, structure_mask=tokens.new_ones((*tokens.shape[:2], 1)),
                structure_projection=lambda _: projected)
        flow, _, _ = region_loss(prediction, target, occupancy, weight)
        terms["flow"] = flow
        total = flow + .2 * terms["framework"] + .3 * terms["distribution"]
        if rgb:
            clip_index = index if review else index // RGB_EVERY
            top, left, tracks = choose_rgb_window(raw, clip_index)
            window = (slice(top // 8, (top + CROP) // 8), slice(left // 8, (left + CROP) // 8))
            region = (slice(None), slice(None), slice(None), *window)
            predicted_clean = noisy[region].float() - sigma * prediction[region].float()
            rgb_pred = decode_pixels(pipe.vae, predicted_clean.to(torch.bfloat16), use_checkpoint=not review)
            expected = torch.from_numpy(raw["target"][:, top:top + CROP, left:left + CROP]).to(
                device=device, dtype=torch.float32) / 255
            extra = rgb_losses(rgb_pred, expected, torch.as_tensor(tracks, device=device, dtype=torch.float32))
            terms.update(extra)
            total = (total + .25 * extra["pixel"] + .15 * extra["chroma"]
                     + .35 * extra["density"] + .2 * extra["track"])
        terms["loss"] = total
        return terms

    def update(self, raw, encoded, index):
        self.optimizer.zero_grad(set_to_none=True)
        rgb = index % RGB_EVERY == 0
        terms = self.denoise(raw, encoded, index, rgb=rgb)
        terms["loss"].backward()
        grad = torch.nn.utils.clip_grad_norm_(self.trainable, 1., error_if_nonfinite=True, foreach=True)
        scalars = torch.stack([*(value.detach().float() for value in terms.values()), grad.detach()]).tolist()
        if not all(math.isfinite(value) for value in scalars):
            raise ValueError("Nonfinite training loss")
        before = {}
        if index == 0:
            for name, parameters in self.groups.items():
                parameter = next((p for p in parameters if p.grad is not None and bool(p.grad.abs().max())), None)
                if parameter is None:
                    raise ValueError(f"First update misses {name} gradients")
                before[name] = (parameter, parameter.detach().clone())
        rates = {}
        for group in self.optimizer.param_groups:
            name = group["name"]
            group["lr"] = rates[name] = learning_rate(index + 1, GROUP_RATES[name])
        self.optimizer.step()
        changed = {name: not torch.equal(p.detach(), value) for name, (p, value) in before.items()}
        if changed and not all(changed.values()):
            raise ValueError(f"First update did not change every group: {changed}")
        return dict(zip(terms, scalars[:-1]), grad_norm=scalars[-1], rates=rates,
                    groups_changed=changed, rgb_supervision=rgb)

    def save(self, path, count, epochs, log_bytes, elapsed, peak):
        adapter, lora = integration.increments(self.pipe)
        value = dict(dataset=self.prepared.summary["schedule_sha256"], updates=count, goal_epochs=epochs,
                     adapter=adapter, lora=lora, structure=tensor_state(self.model),
                     optimizer=self.optimizer.state_dict(), rng=rng_state(), log_bytes=log_bytes,
                     elapsed_seconds=elapsed, peak_vram_bytes=peak)
        atomic_torch_save(path, value)
        return value

    def restore(self, value):
        if value["dataset"] != self.prepared.summary["schedule_sha256"]:
            raise ValueError("Checkpoint data identity differs")
        integration.load_increments_state(self.pipe, value["adapter"], value["lora"])
        self.model.load_state_dict(value["structure"], strict=True)
        self.optimizer.load_state_dict(value["optimizer"])
        restore_rng(value["rng"])

    @torch.no_grad()
    def epoch_review(self, epoch, progress):
        progress.update(announce=True, stage="epoch_review", epoch=epoch)
        summary = {}
        self.model.eval()
        self.pipe.structure_adapter.eval()
        try:
            for n, building in enumerate(VALIDATION_IDS):
                progress.stop()
                first = (10 * n) % 64
                frames = list(range(first, first + 17))
                raw = self.prepared.window(building, frames)
                raw.update(geometry=geometry_for_window(raw["observed"], frames),
                           occupancy=data.occupancy_for(raw["target"]), cache_key=f"review:{building}")
                encoded = self.encoder(raw)
                terms = self.denoise(raw, encoded, 100000 + n, rgb=True, review=True)
                summary[building] = {key: float(value) for key, value in terms.items()}
        finally:
            self.model.train()
            self.pipe.structure_adapter.train()
        write_json(self.run / f"epoch_{epoch}_review.json",
                   dict(time=now(), epoch=epoch, teacher_forced=True, windows=summary))


def _checkpoint_file(run, epoch):
    return run / "checkpoints" / f"epoch_{epoch}.pt"


def publish_epoch(run, latest, epoch):
    path = _checkpoint_file(run, epoch)
    if not path.exists():
        os.link(latest, path)


def finalize_training(run, state, epochs):
    output = run / f"weights_{epochs}"
    output.mkdir(exist_ok=True)
    identities = {}
    for key in ("adapter", "lora", "structure"):
        destination = output / f"{key}.safetensors"
        temp = destination.with_suffix(".tmp")
        save_file(state[key], str(temp), metadata=dict(epochs=str(epochs), updates=str(state["updates"]),
                                                       dataset=state["dataset"]))
        os.replace(temp, destination)
        identities[key] = dict(file=str(destination.relative_to(run)), sha256=sha(destination))
    write_json(run / f"training_{epochs}.json",
               dict(time=now(), epochs=epochs, updates=state["updates"], weights=identities,
                    dataset=state["dataset"], elapsed_seconds=state["elapsed_seconds"],
                    peak_vram_bytes=state["peak_vram_bytes"], cpu_prepared_only=False))


def train(run, epochs, progress, snapshot):
    finish_path = run / f"training_{epochs}.json"
    if finish_path.exists():
        progress.update(announce=True, stage="training_complete", updates=epochs * TRAIN_COUNT,
                        total=epochs * TRAIN_COUNT)
        return
    prepared = PreparedData(run)
    checkpoint = run / "checkpoints/latest.pt"
    state = load_state(checkpoint) if checkpoint.exists() else None
    if state and int(state["goal_epochs"]) > epochs:
        raise ValueError("该 run 已续到第 5 轮，请使用 resume --epochs 5")
    progress.stop()
    trainer = Trainer(run, prepared)
    windows = RollingWindows(run, prepared, trainer.encoder)
    count, start_time, old_seconds, peak = 0, time.monotonic(), 0., 0
    if state:
        trainer.restore(state)
        count, old_seconds, peak = (state[key] for key in ("updates", "elapsed_seconds", "peak_vram_bytes"))
        windows.discard_committed(count)
    progress.update(announce=True, stage="training", updates=count, total=epochs * TRAIN_COUNT,
                    durable_updates=count, epoch=min(epochs, count // TRAIN_COUNT + 1),
                    device=snapshot.get("device_name"), gpu_used_before_bytes=snapshot["used_bytes"])
    steps = epochs * TRAIN_COUNT
    durations = deque(maxlen=32)
    started = time.monotonic()

    with (run / "loss.jsonl").open("a+b") as log, ThreadPoolExecutor(max_workers=PREFETCH) as cpu:
        length = state["log_bytes"] if state else 0
        if log.tell() < length:
            raise ValueError("Training log shorter than its saved checkpoint")
        log.truncate(length)
        log.seek(0, os.SEEK_END)

        def store():
            log.flush()
            os.fsync(log.fileno())
            saved = trainer.save(checkpoint, count, epochs, log.tell(), old_seconds + time.monotonic() - start_time, peak)
            windows.discard_committed(count)
            progress.update(durable_updates=count)
            return saved

        if count and count % TRAIN_COUNT == 0:
            publish_epoch(run, checkpoint, count // TRAIN_COUNT)
            review = run / f"epoch_{count // TRAIN_COUNT}_review.json"
            if not review.exists():
                trainer.epoch_review(count // TRAIN_COUNT, progress)

        def submit(index):
            building, frames = prepared.schedule[index]
            return cpu.submit(windows.build, index, building, frames, index % RGB_EVERY == 0)

        requests = deque(submit(i) for i in range(count, min(steps, count + PREFETCH)))
        next_index = count + len(requests)
        try:
            for index in range(count, steps):
                progress.stop()
                tick = time.monotonic()
                raw, encoded = requests.popleft().result()
                if next_index < steps:
                    requests.append(submit(next_index))
                    next_index += 1
                encoded = windows.extract(index, raw, encoded)
                if index % RGB_EVERY:
                    raw.pop("partial", None)
                    raw.pop("target", None)
                result = trainer.update(raw, encoded, index)
                count = index + 1
                peak = max(peak, torch.cuda.max_memory_reserved())
                del raw, encoded
                result.update(update=count, building=prepared.schedule[index][0], epoch=index // TRAIN_COUNT + 1,
                              frames=prepared.schedule[index][1], vram_bytes=peak)
                log.write((json.dumps(result, allow_nan=False) + "\n").encode())
                log.flush()
                if count == 1 or count % CHECKPOINT_EVERY == 0 or count % TRAIN_COUNT == 0:
                    state = store()
                if count == 1:
                    write_json(run / "first_update.json", dict(time=now(), **result, checkpoint=str(checkpoint)))
                durations.append(time.monotonic() - tick)
                average = sum(durations) / len(durations)
                progress.update(announce=count == 1 or count % 16 == 0, stage="training", updates=count,
                                epoch=index // TRAIN_COUNT + 1, building=result["building"], loss=result["loss"],
                                seconds_per_update=average, eta_seconds=average * (steps - count),
                                peak_vram_bytes=peak)
                if count % TRAIN_COUNT == 0:
                    epoch = count // TRAIN_COUNT
                    publish_epoch(run, checkpoint, epoch)
                    trainer.epoch_review(epoch, progress)
            if count == steps:
                state = store()
                finalize_training(run, state, epochs)
                progress.update(announce=True, stage="training_complete", updates=count, total=steps,
                                elapsed_seconds=time.monotonic() - started)
        except InterruptedError:
            if count:
                store()
            raise
        finally:
            for request in requests:
                request.cancel()
