"""Train and validate one isolated roof-adapter pilot from the delivered V2 weights."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from contextlib import contextmanager
import csv
import gc
import hashlib
import os
from pathlib import Path
import shutil
import signal
import sys
import tempfile
import time
from types import MethodType

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "adapter_gt_v2/scripts"))

import av
import torch
from safetensors.torch import save_file

import data
import integration
import roof
import validation_delivery as delivery
import validate_video as video
from cache_encoder import make_encoder
from common import ROOT, atomic_torch_save, now, sha, write_json
from pipeline import acquire_gpu
from progress import Progress, TOTALS
from training_compute_candidate import _direct_forward
from validation_queue import verify_video
from validation_speed import ValidationSpeed
from worker import region_loss

SEED = 20261120
BASE = ROOT / "outputs/pointcloud_vace/adapter_gt_v2" / delivery.RUN_NAME
OUTPUTS = ROOT / "outputs/pointcloud_vace/adapter_gt_v2_roof"
IDS = ("358", "2825", "5954", "4850", "3538", "2016", "6485", "5466")
PROBES = ("2825", "5954")
WARM, PAIR, WINDOWS = 512, 128, 1024
LR, LORA_LR = 1.e-4, 1.e-5
STAGES = ("roof", "joint")


def schedule():
    manifest = delivery.DATASET / "manifest.csv"
    if sha(manifest) != delivery.MANIFEST_SHA256:
        raise ValueError("Training/validation manifest differs from GT V2")
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        train = [row for row in csv.DictReader(stream) if row["split_use"] == "train"]
    if len(train) != 10747 or len({row["id"] for row in train}) != len(train):
        raise ValueError("Expected the original 10,747 distinct training buildings")
    train.sort(key=lambda r: hashlib.sha256(f"{SEED}:roof:{r['id']}".encode()).digest())
    rows = []
    for row in train[:WINDOWS]:
        building = row["id"]
        paths = {key: str(data._relative(delivery.DATASET, row[key], key, building))
                 for key in ("camera", "record", "partial_video", "GT_video")}
        cloud = next((delivery.POINTS / f"{building}{ext}" for ext in (".ply", ".xyz")
                      if (delivery.POINTS / f"{building}{ext}").is_file()), None)
        if cloud is None:
            raise FileNotFoundError(f"Missing observed cloud for {building}")
        offset = int(data.digest(f"{SEED}:{building}".encode()), 16) % 64
        rows.append(dict(id=building, split_use="train", **paths, pointcloud=str(cloud),
                         frame_ids=list(range(offset, offset + 17)), sample_index=len(rows)))
    return rows


def observed_inputs(row, frame_ids):
    camera, record = (data.read_json(row[key]) for key in ("camera", "record"))
    data.validate_camera(camera, row["id"], row["split_use"])
    if str(record["id"]) != row["id"] or record["split_use"] != row["split_use"]:
        raise ValueError(f"Wrong observation records for {row['id']}")
    points = data.selected_partial(row, camera, record)
    return camera, record, video.observed_geometry(points, camera, frame_ids), roof.tracks(points, camera, frame_ids)


def prepare_window(row):
    ids = row["frame_ids"]
    camera, _, geometry, hints = observed_inputs(row, ids)
    partial = data.decode_gray_window(row["partial_video"], ids)
    target = data.decode_gray_window(row["GT_video"], ids)
    fractions, weights = roof.density_targets(row["id"], camera, ids, target)
    return dict(id=row["id"], frame_ids=ids, cache_key=str(row["sample_index"]),
                target=target, partial=partial, occupancy=data.occupancy_for(target),
                geometry=geometry, hints=hints, density=fractions, density_weights=weights)


def read_state(path):
    return torch.load(path, weights_only=True, map_location="cpu")


def trainable_lora(pipe):
    params = {name: param for name, param in pipe.vace.named_parameters()
              if ".self_attn." in name and (".q.lora_" in name or ".o.lora_" in name)}
    if len(params) != 60 or any(p.dtype != torch.float32 for p in params.values()):
        raise ValueError("Expected 60 FP32 self-attention q/o LoRA tensors")
    return params


def assign_lora(params, saved):
    if set(params) != set(saved):
        raise ValueError("LoRA state keys do not match V2")
    with torch.no_grad():
        for name, parameter in params.items():
            parameter.copy_(saved[name])


def tensor_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def cached_inputs(encoded, raw, tokens):
    return dict(input_latents=encoded["input_latents"], reactive=encoded["reactive"],
                occupancy=encoded["occupancy"], old_tokens=tokens,
                hints=torch.from_numpy(raw["hints"])[None].to(torch.bfloat16),
                density=torch.from_numpy(raw["density"])[None],
                density_weights=torch.from_numpy(raw["density_weights"])[None],
                id=raw["id"], frame_ids=raw["frame_ids"])


class Run:
    def __init__(self, output, scratch, rows, progress):
        self.output, self.scratch, self.rows, self.progress = output, scratch, rows, progress
        self.cache = scratch / "windows"
        self.cache.mkdir(exist_ok=True)
        self.cache_done = 0
        self.started = time.monotonic()
        self.validation_scores = {}
        self.preview_scores = {}
        self.vae_scores = {}
        self.speed_policies = {}
        self.step_time = deque(maxlen=10)
        self.physical_updates = 0

    def stop(self):
        if (self.scratch / "stop.request").exists():
            raise InterruptedError("Stop requested")

    def start_model(self):
        self.progress.update(stage="starting", detail="载入 A6000 模型和已训练的 V2 增量")
        self.model = video.load_validation_model(BASE)
        self.pipe, self.core, self.cfg = self.model
        self.device = self.pipe.device
        torch.manual_seed(SEED)
        self.roof = roof.RoofAdapter().to(self.device)
        self.lora = trainable_lora(self.pipe)
        self.model_args = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        self.pipe.scheduler.set_timesteps(1000, training=True, shift=5)
        from diffsynth.pipelines.wan_video import WanVideoUnit_PromptEmbedder
        with torch.no_grad():
            self.context = WanVideoUnit_PromptEmbedder().process(
                self.pipe, prompt=self.cfg["prompt"], positive=True)["context"]
            self.inactive = self.pipe.vae.encode(
                torch.zeros((1, 3, 17, 480, 832), device=self.device, dtype=torch.bfloat16),
                device=self.device, tiled=False)
        self.mask = torch.ones((1, 64, 5, 60, 104), device=self.device, dtype=torch.bfloat16)
        self.pipe.text_encoder.to("cpu")
        torch.cuda.empty_cache()
        print(f"Loaded verified V2 weights on {torch.cuda.get_device_name()} · frozen Adapter and backbone", flush=True)

    def cached_path(self, index):
        return self.cache / f"{index:04d}.pt"

    def cache_windows(self, end):
        self.stop()
        self.pipe.vae.to(self.device)
        encode = make_encoder(self.pipe.vae, "whole_frame")
        pending = [i for i in range(end) if not self.cached_path(i).exists()]
        self.cache_done = sum(1 for i in range(WINDOWS) if self.cached_path(i).exists())
        self.progress.update(stage="cache", cache=self.cache_done, eta=None,
                             detail=f"整帧 VAE · 本批目标累计 {end} 窗")
        rates = deque(maxlen=15)
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="roof-prep") as pool:
            prefetch = deque((i, pool.submit(prepare_window, self.rows[i])) for i in pending[:4])
            next_at = 4
            while prefetch:
                self.stop()
                started = time.monotonic()
                i, future = prefetch.popleft()
                raw = future.result()
                if next_at < len(pending):
                    next_index = pending[next_at]
                    prefetch.append((next_index, pool.submit(prepare_window, self.rows[next_index])))
                    next_at += 1
                with torch.no_grad():
                    encoded = encode(raw)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    geometry = {key: torch.from_numpy(value)[None].to(self.device)
                                for key, value in raw["geometry"].items()}
                    tokens = self.pipe.structure_adapter(geometry).cpu()
                sample = cached_inputs(encoded, raw, tokens)
                if (sample["input_latents"].shape != (1, 16, 5, 60, 104)
                        or sample["reactive"].shape != (1, 16, 5, 60, 104)
                        or tokens.shape != (1, 7800, 64) or sample["hints"].shape != (1, 7800, 256)
                        or not all(torch.isfinite(v).all() for v in sample.values() if isinstance(v, torch.Tensor))):
                    raise ValueError(f"Invalid cached window {raw['id']}")
                atomic_torch_save(self.cached_path(i), sample)
                self.cache_done += 1
                rates.append(time.monotonic() - started)
                rate = sum(rates) / len(rates)
                self.progress.update(cache=self.cache_done, rate=rate, eta=rate * (len(prefetch) + max(0, len(pending) - next_at)),
                                     detail=f"整帧 VAE · 建筑 {raw['id']} · 本批目标累计 {end} 窗")
                if self.cache_done == 1 or self.cache_done % 16 == 0 or not prefetch:
                    print(f"缓存 {self.cache_done}/{WINDOWS} · ID {raw['id']} · {rate:.2f} 秒/窗", flush=True)
        self.pipe.vae.to("cpu")
        torch.cuda.empty_cache()

    def sample(self, index):
        saved = read_state(self.cached_path(index))
        row = self.rows[index]
        if saved["id"] != row["id"] or saved["frame_ids"] != row["frame_ids"]:
            raise ValueError(f"Cache mismatch at window {index}")
        return {key: value.to(self.device) for key, value in saved.items() if isinstance(value, torch.Tensor)}

    def backward(self, index, sample):
        clean = sample["input_latents"]
        scheduler = self.pipe.scheduler
        host_rng = torch.Generator().manual_seed(SEED + index)
        cuda_rng = torch.Generator(device=self.device).manual_seed(SEED + index)
        chosen = torch.randint(0, 1000, (1,), generator=host_rng)
        timestep = scheduler.timesteps[chosen].to(dtype=torch.bfloat16)
        noise = torch.randn(clean.shape, device=self.device, dtype=torch.bfloat16, generator=cuda_rng)
        noisy = scheduler.add_noise(clean, noise, timestep)
        weight = scheduler.training_weight(timestep).reshape(1).to(self.device)
        target = scheduler.training_target(clean, noise, timestep)
        tokens = sample["old_tokens"]
        vace = torch.cat((self.inactive, sample["reactive"], self.mask), dim=1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            residual, density, occupied = self.roof(sample["hints"])
            frozen = self.pipe.structure_adapter.structure_proj(tokens)
            prediction = integration.model_fn_wan_video(
                **self.model_args, latents=noisy, timestep=timestep.to(self.device),
                context=self.context, vace_context=vace, vace_scale=1.,
                use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False,
                structure_tokens=tokens, structure_mask=tokens.new_ones((*tokens.shape[:2], 1)),
                structure_projection=lambda _: frozen + residual.to(frozen.dtype))
            flow, _, _ = region_loss(prediction, target, sample["occupancy"], weight)
            local = (density - sample["density"]).square().reshape(*occupied.shape, 2).mean(-1)
            mask = sample["density_weights"] * occupied
            density_loss = (local * mask).sum() / mask.sum().clamp_min(1)
            loss = flow + .1 * density_loss
        loss.backward()
        return torch.stack((loss.detach(), flow.detach(), density_loss.detach(), weight[0]))

    @contextmanager
    def train_mode(self, joint):
        self.roof.train()
        self.pipe.scheduler.set_timesteps(1000, training=True, shift=5)
        for parameter in self.lora.values():
            parameter.requires_grad_(joint)
        reference = self.core.forward
        self.core.forward = MethodType(_direct_forward, self.core)
        try:
            yield
        finally:
            self.core.forward = reference
            for parameter in self.lora.values():
                parameter.grad = None
                parameter.requires_grad_(False)
            self.roof.eval()

    def check_update(self):
        self.progress.update(stage="checking", eta=None, detail="核对首窗 GPU 反传与屋顶分支梯度")
        with self.train_mode(False):
            self.roof.zero_grad(set_to_none=True)
            loss = self.backward(0, self.sample(0))
            grad = self.roof.project.weight.grad
            if (not torch.isfinite(loss).all() or grad is None
                    or not torch.isfinite(grad).all() or not torch.count_nonzero(grad)):
                raise RuntimeError("First observed roof window did not produce a finite projection gradient")
            self.roof.zero_grad(set_to_none=True)
        print("GPU 首窗反传通过，检查阶段未计入训练更新", flush=True)

    def train(self, name, end, physical_base, start=0, parent=None, joint=False):
        path = self.scratch / f"{name}.pt"
        initial = read_state(parent) if parent is not None else None
        saved = read_state(path) if path.exists() else initial
        if saved is not None:
            self.roof.load_state_dict(saved["roof"])
            assign_lora(self.lora, saved["lora"])
        with self.train_mode(joint):
            optim = torch.optim.AdamW(self.roof.parameters(), lr=LR, weight_decay=0.)
            if saved is not None and saved.get("joint") == joint:
                if joint:
                    optim.add_param_group({"params": list(self.lora.values()), "lr": LORA_LR})
                optim.load_state_dict(saved["optimizer"])
            elif saved is not None:
                optim.load_state_dict(saved["optimizer"])
                if joint:
                    optim.add_param_group({"params": list(self.lora.values()), "lr": LORA_LR})
            elif joint:
                optim.add_param_group({"params": list(self.lora.values()), "lr": LORA_LR})
            cursor = saved["cursor"] if path.exists() else start
            self.physical_updates = physical_base + cursor - start
            loss_sum = saved.get("loss_sum", [0., 0., 0.]) if path.exists() else [0., 0., 0.]
            max_vram = saved.get("max_vram_bytes", 0) if path.exists() else 0
            self.progress.update(stage="train", train=self.physical_updates, eta=None,
                                 detail=f"{name} · 当前分支 {cursor}/{end}")
            parameters = [param for group in optim.param_groups for param in group["params"]]
            rates = deque(maxlen=15)
            torch.cuda.reset_peak_memory_stats()
            for i in range(cursor, end):
                self.stop()
                tick = time.monotonic()
                optim.zero_grad(set_to_none=True)
                losses = self.backward(i, self.sample(i))
                g = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                result = torch.cat((losses, g.detach().view(1))).tolist()
                if not all(abs(x) < float("inf") for x in result):
                    raise RuntimeError(f"Nonfinite {name} update {i + 1}: {result}")
                if i == cursor and not result[-1] > 0:
                    raise RuntimeError(f"No useful gradient at {name} window {i}")
                optim.step()
                self.physical_updates = physical_base + i + 1 - start
                loss_sum = [a + b for a, b in zip(loss_sum, result[:3])]
                max_vram = max(max_vram, torch.cuda.max_memory_reserved())
                if (i + 1) % 32 == 0 or i + 1 == end:
                    atomic_torch_save(path, dict(cursor=i + 1, roof=tensor_state(self.roof),
                                      lora={key: value.detach().cpu().clone() for key, value in self.lora.items()},
                                      optimizer=optim.state_dict(), joint=joint, loss_sum=loss_sum,
                                      max_vram_bytes=max_vram))
                rates.append(time.monotonic() - tick)
                rate = sum(rates) / len(rates)
                self.progress.update(train=self.physical_updates, rate=rate, eta=rate * (end - i - 1),
                                     detail=f"{name} · 分支 {i + 1}/{end} · 建筑 {self.rows[i]['id']}")
                if i == cursor or (i + 1) % 16 == 0 or i + 1 == end:
                    print(f"训练 {self.physical_updates}/{TOTALS['train']} · {name} {i + 1}/{end} · "
                          f"loss {result[0]:.5f} · {rate:.2f} 秒/步", flush=True)
        return path

    def apply(self, path):
        state = read_state(path)
        self.roof.load_state_dict(state["roof"])
        assign_lora(self.lora, state["lora"])
        self.roof.eval()
        return state["joint"]

    def start_video(self):
        self.pipe.eval()
        self.pipe.text_encoder.to(self.device)
        self.pipe.vae.to(self.device)

    def release_video(self):
        self.pipe.text_encoder.to("cpu")
        self.pipe.vae.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    def score(self, building, camera, selection, row, gt, **clips):
        reference = BASE / "验证视频_八栋_20261119" / f"{IDS.index(building) + 1:02d}_ID{building}.mp4"
        if sha(row["GT_video"]) != row["gt_video_sha256"]:
            raise ValueError(f"Held-out GT differs from its checksum: {building}")
        return roof.scores(building, camera, selection, gt,
                           {"v2": roof.video_luma(reference, comparison=True), **clips})

    def vae_check(self):
        self.pipe.vae.to(self.device)
        for index, building in enumerate(PROBES):
            self.stop()
            self.progress.update(stage="vae", vae=index, detail=f"建筑 {building} · 81 帧 GT 编解码", eta=None)
            cached = self.scratch / f"vae_{building}.json"
            if cached.exists():
                self.vae_scores[building] = data.read_json(cached)
                continue
            row = video.validation_row(BASE, building)
            camera, selection = (data.read_json(row[key]) for key in ("camera", "record"))
            gt = video.source_frames(row["GT_video"])
            with torch.inference_mode():
                latent = self.pipe.vae.encode(self.pipe.preprocess_video(gt), device=self.device, tiled=False)
                decoded = self.pipe.vae.decode(latent, device=self.device, tiled=False)
                frames = self.pipe.vae_output_to_video(decoded)
            del latent, decoded
            metrics = self.score(building, camera, selection, row, roof.rgb_luma(gt), vae=roof.rgb_luma(frames))
            write_json(cached, metrics)
            self.vae_scores[building] = metrics
            self.progress.update(vae=index + 1)
            print(f"VAE 诊断 {index + 1}/2 · 建筑 {building}", flush=True)

    def render(self, label, building, index, done, final=False):
        self.step_time.clear()
        detail = f"{label} · 建筑 {building}"
        self.progress.update(stage="validation", step=0, eta=None, validation=done, detail=detail + " · 准备输入")
        saved = self.scratch / f"metrics_{label}_{building}.json"
        filename = f"{index:02d}_ID{building}.mp4"
        comparison = self.output / "验证对照" / filename
        prediction = self.output / "Prediction" / filename
        if saved.is_file() and (not final or (comparison.is_file() and prediction.is_file())):
            self.progress.update(validation=done + 1, step=30, detail=detail + " · 已核验")
            return data.read_json(saved)
        row = video.validation_row(BASE, building)
        camera, selection, geometry, tracks = observed_inputs(row, range(81))
        partial = video.source_frames(row["partial_video"])
        last = time.monotonic()

        def step(n):
            nonlocal last
            self.stop()
            tick = time.monotonic()
            if n > 1:
                self.step_time.append(tick - last)
            last = tick
            rate = sum(self.step_time) / max(1, len(self.step_time))
            self.progress.update(step=n, detail=detail + " · 采样",
                                 rate=rate, eta=rate * (30 - n) if self.step_time else None)

        def projection(tokens):
            residual = self.roof(torch.from_numpy(tracks)[None].to(self.device, dtype=torch.bfloat16))[0]
            frozen = self.pipe.structure_adapter.structure_proj(tokens)
            combined = frozen + residual.to(frozen.dtype)
            return lambda _: combined

        policy = self.speed_policies.setdefault(label, {})
        with ValidationSpeed(self.pipe, self.core, policy, lambda _: None) as speed:
            frames = video.predict_video(self.model, row, partial, speed, geometry=geometry,
                                         make_projection=projection, on_step=step)
        self.progress.update(detail=detail + " · 编码与核验", eta=None)
        gt = video.source_frames(row["GT_video"])
        identity = f"GT-V2-ROOF|{self.output.name}|{label}|{building}|81f-30s-cfg5-seed20261018"
        staging = self.output if final else self.scratch
        with tempfile.TemporaryDirectory(prefix=".video-", dir=staging) as temp:
            temp = Path(temp)
            side = temp / "comparison.mp4"
            video.save_comparison(side, building, partial, frames, gt, identity)
            verify_video(side, identity)
            metrics = self.score(building, camera, selection, row, roof.rgb_luma(gt),
                                 new=roof.video_luma(side, comparison=True))
            if final:
                one = temp / "prediction.mp4"
                video.encode_rgb_video(one, frames, (832, 480), identity)
                check_prediction(one, identity)
                for src, destination in ((side, comparison), (one, prediction)):
                    destination.parent.mkdir(exist_ok=True)
                    os.replace(src, destination)
            write_json(saved, metrics)
        self.progress.update(validation=done + 1, step=30, detail=detail + " · 成片已核验")
        print(f"验证 {done + 1}/{TOTALS['validation']} · {label} · 建筑 {building} 已核验", flush=True)
        return metrics

    def execute(self):
        self.start_model()
        self.cache_windows(1)
        self.check_update()
        self.cache_windows(WARM + PAIR)
        warm = self.train("warm", WARM, 0)
        # Both alternatives start from the same roof weights, moments, samples and noise.
        branches = {
            "roof": self.train("roof", WARM + PAIR, WARM, WARM, warm),
            "joint": self.train("joint", WARM + PAIR, WARM + PAIR, WARM, warm, True),
        }
        self.vae_check()
        self.start_video()
        done = 0
        for label, state in branches.items():
            self.apply(state)
            self.preview_scores[label] = {}
            for i, building in enumerate(PROBES, 1):
                self.preview_scores[label][building] = self.render(label, building, i, done)
                done += 1
        selected, costs = select(self.preview_scores)
        write_json(self.scratch / "selection.json", dict(selected=selected, costs=costs))
        print(f"短对照选中 {selected} · 评测代价 {costs}", flush=True)
        self.release_video()
        self.cache_windows(WINDOWS)
        finish = self.train("finish", WINDOWS, WARM + 2 * PAIR, WARM + PAIR,
                            branches[selected], selected == "joint")
        joint = self.apply(finish)
        metadata = dict(base=delivery.RUN_NAME, selected=selected, updates=str(WINDOWS))
        save_weights(self.output / "roof_adapter.safetensors", tensor_state(self.roof), metadata)
        if joint:
            complete = {name: param.detach().cpu().contiguous() for name, param in self.pipe.vace.named_parameters()
                        if ".lora_" in name}
            save_weights(self.output / "vace_lora_joint.safetensors", complete, metadata)
        self.start_video()
        for index, building in enumerate(IDS, 1):
            self.validation_scores[building] = self.render("final", building, index, done, final=True)
            done += 1
        self.report(selected, costs)
        self.progress.update(stage="complete", detail="已保存最终权重、八栋成片及实验报告", eta=None)
        print(f"完成 · {self.output}", flush=True)
        shutil.rmtree(self.cache)
        for path in [*(self.scratch.glob("*.pt")), *(self.scratch.glob("metrics_*.json"))]:
            path.unlink()

    def report(self, selected, costs):
        average = {role: {metric: sum(v[role][metric] for v in self.validation_scores.values()) / len(IDS)
                          for metric in ("cv", "motion", "wall_iou", "loop", "ink", "deep_ink")}
                   for role in ("gt", "v2", "new")}
        cv, motion, wall = ("cv", "motion", "wall_iou")
        qualify = (all(abs(average["new"][key] - average["gt"][key])
                       <= .75 * abs(average["v2"][key] - average["gt"][key]) for key in (cv, motion))
                   and average["new"][wall] >= average["v2"][wall] - .02)
        summaries = []
        for name in ("warm", "roof", "joint", "finish"):
            state = read_state(self.scratch / f"{name}.pt")
            steps = {"warm": WARM, "roof": PAIR, "joint": PAIR, "finish": WINDOWS - WARM - PAIR}[name]
            loss = [value / steps for value in state["loss_sum"]]
            gib = state["max_vram_bytes"] / (1 << 30)
            summaries.append(f"| {name} | {steps} | {loss[1]:.6f} | {loss[2]:.6f} | {gib:.2f} |")
        wins = []
        for building in IDS:
            a, b = (self.validation_scores[building][name] for name in ("v2", "new"))
            wins.append(f"| {building} | {a[cv]:.3f} → {b[cv]:.3f} | {a[motion]:.3f} → {b[motion]:.3f} | "
                        f"{a[wall]:.3f} → {b[wall]:.3f} | {a['loop']:.2f} → {b['loop']:.2f} |")
        targets = [f"| {role} | {val[cv]:.3f} | {val[motion]:.3f} | {val[wall]:.3f} | "
                   f"{val['loop']:.2f} | {val['ink']:.3f} | {val['deep_ink']:.3f} |"
                   for role, val in average.items()]
        probes = []
        for building, result in self.vae_scores.items():
            for role, val in result.items():
                probes.append(f"| {building} | {role} | {val[cv]:.3f} | {val[motion]:.3f} | {val['ink']:.3f} |")
        weights = "\n".join(f"- {p.name} · SHA-256 {sha(p)}" for p in sorted(self.output.glob("*.safetensors")))
        elapsed = (time.monotonic() - self.started) / 60
        report = f"""# GT V2 屋顶增量与 LoRA 短对照

运行 {self.output.name} 于 {now()} 完成，单张 RTX A6000。本次进程耗时 {elapsed:.1f} 分钟。
从正式 GT V2 的两份最终增量加载 A6000 本地的 Wan2.1-VACE-1.3B，采用原相机和原始残缺视频。
新分支用实测点的楼内 XYZ 和同一相机投影学习屋面点纹，完整 GT 和 OBJ 参与训练监督与验证评分。

## 数据与训练

原 train 中固定种子 {SEED} 选择 1,024 栋，逐栋一个 17 帧窗口、832×480、整帧 BF16 VAE、batch 1。
输入由 12 近邻平面法向、平面性和高度筛选屋面。每个 16×16 像素 token 保存 4×4 个子格的点数和 XYZ 均值。
灰度 220 和 150 阈值的屋面覆盖率辅助损失乘 0.1，加至原区域均衡 Flow Matching 损失。
屋顶分支 AdamW 学习率 {LR:g}，联合阶段原 VACE 15 个控制块的自注意力 q/o LoRA 学习率 {LORA_LR:g}。
当前实际执行 {self.physical_updates} 次更新，最终路径累计 1,024 步。
短对照共同从第 512 步出发，两个分支采用相同的 128 窗和噪声。

| 阶段 | 实际更新 | 平均去噪损失 | 平均密度损失 | PyTorch 峰值保留 GiB |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(summaries)}

两栋短对照选中 **{selected}**。代价越低越接近 GT，屋顶变化与疏密各占 0.35，深色与浅色覆盖误差各占 0.15。
分支屋顶 {costs['roof']:.4f}，联合 LoRA {costs['joint']:.4f}。联合阶段另受墙面占据 IoU 门槛约束。

## VAE 诊断

用原 VAE 整帧编码和解码两栋完整 GT，保留原像素尺寸和 81 帧。

| 建筑 | 来源 | 屋面疏密 CV | 相邻帧变化 | 灰度低于 220 的屋面比例 |
| --- | --- | ---: | ---: | ---: |
{chr(10).join(probes)}

## 正式八栋验证

每栋 81 帧、16 fps、832×480、30 步、CFG 5、shift 5、种子 20261018。
输出是模型直接解码的预测。每栋分别发布独立 MP4 与三栏 MP4，均已逐帧解码核验。
屋顶 CV 在可见屋面 16×16 小格测量，相邻变化沿两帧均可见的 GT 三维点投影计算。
墙面 IoU 在可见墙面以 8×8 占据格计算。首尾为同一相机的屋面灰度差。指标按栋等权。

| 来源 | 屋面 CV | 相邻帧变化 | 墙面 IoU | 首尾灰度差 | 屋面点纹 | 屋面深色点纹 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(targets)}

| 建筑 | 屋面 CV | 相邻帧变化 | 墙面 IoU | 首尾灰度差 |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(wins)}

自动量化门槛结果为 **{'通过' if qualify else '未通过'}**。
门槛要求屋面 CV 和相邻帧变化分别向 GT 缩小至少 25% 的差距，墙面 IoU 均值下降至多 0.02。
逐栋人工视觉复核待进行，优先复看 2825 的黑檐与 5954 的转角暗片。

## 权重

{weights}

原 V2 Adapter 继续从 {BASE.relative_to(ROOT) / '交付_训练完成'} 读取。
基础 LoRA 使用本组联合增量文件或原 V2 正式 LoRA，取决于上方的短对照选型。
"""
        local = self.output / "实验报告.md"
        local.write_text(report, encoding="utf-8")
        upload = ROOT / "upload" / f"{self.output.name[4:12]}_GT视频V2_屋顶增量_{self.output.name[13:]}_实验报告.md"
        upload.write_text(report, encoding="utf-8")


def check_prediction(path, comment):
    with av.open(str(path)) as clip:
        stream = clip.streams.video[0]
        codec = stream.codec_context
        if (len(clip.streams) != 1 or clip.metadata.get("comment") != comment
                or stream.average_rate != 16 or codec.name != "h264" or codec.profile != "High"
                or codec.format.name != "yuv420p" or codec.colorspace != 1 or codec.color_range != 1):
            raise ValueError(f"Incompatible prediction MP4: {path}")
        count = 0
        for frame in clip.decode(video=0):
            if (frame.width, frame.height) != (832, 480) or frame.is_corrupt or frame.pts is None or frame.pts * frame.time_base * 16 != count:
                raise ValueError(f"Invalid standalone prediction frame {count}")
            count += 1
        if count != 81:
            raise ValueError(f"Expected 81 decoded prediction frames at {path}")


def save_weights(path, state, metadata):
    temp = path.with_suffix(".tmp")
    save_file(state, str(temp), metadata=metadata)
    os.replace(temp, path)


def select(results):
    costs = {}
    wall = {role: 0. for role in STAGES}
    baseline_wall = sum(results["roof"][key]["v2"]["wall_iou"] for key in PROBES) / len(PROBES)
    for role in STAGES:
        cost = 0.
        for sample in results[role].values():
            v, gt, new = (sample[k] for k in ("v2", "gt", "new"))
            for metric, weight, floor in (("cv", .35, .03), ("motion", .35, .3),
                                          ("ink", .15, .02), ("deep_ink", .15, .02)):
                cost += weight * abs(new[metric] - gt[metric]) / max(floor, abs(v[metric] - gt[metric]))
            wall[role] += new["wall_iou"] / len(PROBES)
        costs[role] = cost / len(PROBES)
    chosen = ("joint" if costs["joint"] < costs["roof"] * .98
              and wall["joint"] >= max(wall["roof"], baseline_wall) - .02 else "roof")
    return chosen, costs


def check(rows):
    delivery.final_weights(BASE)
    delivery.model_config()
    valid = delivery.validation_rows()
    if not set(IDS) <= valid.keys():
        raise ValueError("Invalid held-out building selection")
    for row in (rows[0], rows[WARM], rows[-1]):
        sample = prepare_window(row)
        if (sample["hints"].shape != (7800, 256) or sample["density"].shape != (7800, 8)
                or not sample["density_weights"].any() or not sample["hints"][:, ::4].any()):
            raise ValueError(f"Invalid CPU input or no roof supervision for {row['id']}")
        print(f"CPU 窗口 {row['id']} · {row['frame_ids'][0]}–{row['frame_ids'][-1]} · 条件与监督通过", flush=True)
    adapter = roof.RoofAdapter()
    hints = torch.randn(1, 32, 256).abs()
    delta, density, _ = adapter(hints)
    if delta.count_nonzero() or delta.shape != (1, 32, 1536) or density.shape != (1, 32, 8):
        raise ValueError("Failed zero-initialization or token layout")
    (delta.mean() + density.mean()).backward()
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.parameters()):
        raise ValueError("Auxiliary training has a disconnected or nonfinite gradient")
    print(f"CPU 检查通过 · {len(rows)} 栋训练 · {len(IDS)} 栋留出验证", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scratch", type=Path)
    args = parser.parse_args()
    rows = schedule()
    if args.check:
        check(rows)
        return
    if not args.output or not args.scratch:
        parser.error("Use --check or provide --output and --scratch to run the pilot")
    output, scratch = args.output.resolve(), args.scratch.resolve()
    if output.parent != OUTPUTS or not output.name.startswith("run_") or not scratch.is_dir():
        parser.error("Require a dedicated GT V2 roof run directory and an existing scratch directory")
    if (output / "实验报告.md").exists():
        raise FileExistsError("This run already delivered its final report")
    import fcntl
    with (scratch / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        progress = Progress(scratch)
        output.mkdir(parents=True, exist_ok=True)
        identity = dict(output=str(output), seed=SEED, warm=WARM, pair=PAIR, cases=rows,
                        base_lora=delivery.FINAL_SHA256["lora"], base_adapter=delivery.FINAL_SHA256["adapter"])
        config = scratch / "pilot.json"
        if config.exists() and data.read_json(config) != identity:
            raise ValueError("Scratch directory belongs to a different pilot")
        if not config.exists():
            write_json(config, identity)

        def interrupt(signum, _frame):
            raise InterruptedError(f"Received signal {signum}")

        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGHUP, interrupt)
        try:
            progress.update(stage="waiting_for_gpu", detail="核对 GPU 独占锁")
            with acquire_gpu(scratch):
                Run(output, scratch, rows, progress).execute()
        except (KeyboardInterrupt, InterruptedError) as exc:
            progress.update(stage="stopped", detail=str(exc) or "操作员停止")
            raise
        except BaseException as exc:
            progress.update(stage="failed", detail=f"{type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    main()
