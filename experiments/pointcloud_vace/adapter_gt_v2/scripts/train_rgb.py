"""Single-card continuation of the V2 increments with sigma-gated five-frame RGB supervision."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
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

import data
import integration
import rgb_supervision as rgb
import rgb_window as windows
from cache_encoder import normalized
from common import atomic_torch_save, now, sha, write_json
from training_compute_candidate import _direct_forward
from validate_video import load_validation_model
from rgb_recipe import PARALLAX_TASK
from worker import PER_EPOCH, TOTAL, learning_rate, make_optimizer, region_loss, restore_rng, rng_state

PREFETCH = 4
SUMMARY = 50
SAVE_UPDATES = 200
SAVE_SECONDS = 300


def progress(run, state, **values):
    """The live board does not need a mechanical-disk fsync per frame/window."""
    state.update(values, time=now())
    path = run / 'train_progress.json'
    temporary = path.with_suffix('.writing')
    temporary.write_text(json.dumps(state, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def lookup(schedule, bounds):
    # The original V2 worker converts its sampled timestep to BF16 *before*
    # the scheduler's nearest-index lookup. Multiple initial indices may give
    # the same effective sigma. Keep exactly this distribution on ordinary steps.
    times = schedule.timesteps.cpu()
    rounded = times.to(torch.bfloat16)
    nearest = (rounded[:, None].float() - times[None]).abs().argmin(1)
    sigmas = schedule.sigmas.cpu()[nearest]
    eligible = torch.where((sigmas >= bounds[0]) & (sigmas <= bounds[1]))[0]
    return rounded, nearest, eligible


class Trainer:
    def __init__(self, run, config):
        from diffsynth.pipelines.wan_video import WanVideoUnit_PromptEmbedder

        self.run = run
        base = Path(config['source_weights']['adapter_final']).parents[1]
        self.pipe, core, pretrained = load_validation_model(base)
        pipe = self.pipe
        self.native = config.get('task') == PARALLAX_TASK
        self.features = config['rgb_features'] if self.native else None
        self.size = (config['training']['height'], config['training']['width'])
        self.core = core
        if self.native:
            pipe.structure_adapter.enable_parallax()
        pipe.structure_adapter.train().requires_grad_(True)
        pipe.vace.train()
        for name, parameter in pipe.vace.named_parameters():
            if '.lora_' in name:
                parameter.requires_grad_(True)
        self.optimizer, self.adapter, self.lora = make_optimizer(pipe)
        self.parameters = [p for _, p in self.adapter + self.lora]
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda index: learning_rate(index + 1) / 1.e-4)
        pipe.scheduler.set_timesteps(1000, training=True, shift=5)
        core.forward = MethodType(_direct_forward, core)
        self.models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        self.bounds = (config['auxiliary']['sigma_min'], config['auxiliary']['sigma_max'])
        self.times, self.nearest, self.first_indices = lookup(pipe.scheduler, self.bounds)
        shared_path = run / 'checkpoints/shared.pt'
        if shared_path.exists():
            shared = torch.load(shared_path, map_location='cpu', weights_only=True)
        else:
            with torch.no_grad():
                context = WanVideoUnit_PromptEmbedder().process(pipe, prompt=pretrained['prompt'], positive=True)['context']
                if self.native:
                    pipe.text_encoder.to('cpu')
                zero = torch.zeros((1, 3, 17, *self.size), device=pipe.device, dtype=torch.bfloat16)
                inactive = pipe.vae.model.encode(zero, pipe.vae.scale)
            shared = dict(context=context.cpu(), inactive=inactive.cpu())
            atomic_torch_save(shared_path, shared)
            del zero, context, inactive
        self.context, self.inactive = (shared[key].to(pipe.device) for key in ('context', 'inactive'))
        self.mask = self.inactive.new_ones((1, 64, *self.inactive.shape[2:]))
        self.valid = torch.ones((5, *self.size), device=pipe.device)
        if not self.native:
            self.valid[:, :windows.VALID_START] = self.valid[:, windows.VALID_STOP:] = 0
        pipe.text_encoder.to('cpu')
        pipe.vae.model.clear_cache()
        torch.cuda.empty_cache()
        seed = config['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @torch.no_grad()
    def encode(self, raw):
        vae = self.pipe.vae
        try:
            # The V2 whole-frame encoder. The resulting pair stays on device
            # for this immediate batch and is released with that batch.
            clean = vae.model.encode(normalized(raw['target']), vae.scale)
            reactive = vae.model.encode(normalized(raw['partial']), vae.scale)
            return clean, torch.cat((self.inactive, reactive, self.mask), 1)
        finally:
            vae.model.clear_cache()

    def choose_noise(self, first):
        source = int(self.first_indices[torch.randint(len(self.first_indices), ())]) if first else int(
            torch.randint(len(self.times), ()))
        i = int(self.nearest[source])
        t = self.times[source].reshape(1).to(self.pipe.device)
        return t, self.pipe.scheduler.sigmas[i], self.pipe.scheduler.linear_timesteps_weights[i]

    def update(self, raw, encoded, step):
        pipe = self.pipe
        self.optimizer.zero_grad(set_to_none=True)
        clean, condition = encoded
        t, sigma, weight = self.choose_noise(step == 1)
        noise = torch.randn_like(clean)
        noisy = (1 - sigma) * clean + sigma * noise
        target = noise - clean
        geometry = {key: torch.as_tensor(v, device=pipe.device)[None] for key, v in raw['geometry'].items()}
        occupancy = torch.as_tensor(raw['occupancy'], device=pipe.device)[None]
        active = self.bounds[0] <= float(sigma) <= self.bounds[1]
        if self.native:
            mode = 'full' if active else 'none'
            if pipe.adapter_gt_v2_checkpointing['mode'] != mode:
                integration.configure_checkpointing(pipe, mode)
                forward = integration.structured_vace_forward if active else _direct_forward
                self.core.forward = MethodType(forward, self.core)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            tokens = pipe.structure_adapter(geometry)
            prediction = pipe.model_fn(
                **self.models, latents=noisy, timestep=t, context=self.context,
                vace_context=condition, vace_scale=1., use_gradient_checkpointing=self.native and active,
                structure_tokens=tokens, structure_mask=tokens.new_ones((*tokens.shape[:2], 1)),
                structure_projection=pipe.structure_adapter.structure_proj)
        fm, _, _ = region_loss(prediction, target, occupancy, weight.to(pipe.device))
        losses = dict(fm=fm)
        if active:
            # Only the native causal prefix is decoded. Spatial calibration
            # and frame order are identical across RGB, GT, partial and bands.
            prefix = noisy[:, :, :2].float() - sigma * prediction[:, :, :2].float()
            estimate = rgb.decode_prefix(pipe.vae, prefix)
            affine = (1., 1., 0., 0.) if self.native else windows.AFFINE
            band = rgb.structure_weights(raw['mesh'], raw['camera'], raw['frame_ids'][:5],
                                         size=self.size, image_affine=affine)
            gt, partial = (torch.as_tensor(raw[k][:5], device=pipe.device).float() / 255. for k in ('target', 'partial'))
            losses.update(rgb.rgb_losses(estimate, gt, partial, self.valid, band, step, self.features))

        loss = fm + losses['aux'] if active else fm
        confirmation = {}
        if step == 1:
            # A single last-block LoRA tensor witnesses the RGB-only VJP.
            # No other tensor is probed and no VACE forward is repeated.
            probe_name, probe = next((n, p) for n, p in reversed(self.lora)
                                     if '.cross_attn.o.lora_B.' in n)
            probe_grad, = torch.autograd.grad(losses['aux'], probe, retain_graph=True)
            proof = float(probe_grad.detach().float().norm())
            if not math.isfinite(proof) or proof == 0:
                raise RuntimeError('The combined RGB term did not reach the trainable VACE LoRA.')
            confirmation.update(aux_probe=probe_name, aux_parameter_gradient_norm=proof)
            watched = dict(adapter=self.pipe.structure_adapter.structure_proj.weight, lora=probe)
            if self.native:
                watched['parallax'] = self.pipe.structure_adapter.parallax_proj.weight
            before = {k: p.detach().clone() for k, p in watched.items()}

        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(self.parameters, 1., error_if_nonfinite=True)
        scalars = torch.stack([loss.detach(), grad.detach(), *(v.detach() for v in losses.values())]).float().tolist()
        if not all(math.isfinite(v) for v in scalars):
            raise RuntimeError('Nonfinite V2 training values')
        lr = self.optimizer.param_groups[0]['lr']
        self.optimizer.step()
        self.lr_scheduler.step()
        if step == 1:
            change = {k: float((p.detach() - before[k]).abs().max()) for k, p in watched.items()}
            if not all(math.isfinite(v) and v > 0 for v in change.values()):
                raise RuntimeError(f'First AdamW update did not change both increments: {change}')
            confirmation.update(parameter_delta_max=change)
        return dict(zip(losses, scalars[2:]), loss=scalars[0], grad_norm=scalars[1], lr=lr,
                    sigma=float(sigma), timestep=float(t.item()), rgb_active=active), confirmation

    def store(self, path, config_identity, state):
        adapter, lora = integration.increments(self.pipe)
        value = dict(adapter=adapter, lora=lora, optimizer=self.optimizer.state_dict(),
                     lr_scheduler=self.lr_scheduler.state_dict(), flow_scheduler=dict(steps=1000, shift=5.),
                     rng=rng_state(), config_sha256=config_identity, **state)
        next_path = path.with_name('next.pt')
        atomic_torch_save(next_path, value)
        if path.exists():
            # A hard link preserves the previous good state even if interrupted
            # between rotating the old checkpoint and publishing the new one.
            previous_tmp = path.with_name('previous.link')
            previous_tmp.unlink(missing_ok=True)
            os.link(path, previous_tmp)
            os.replace(previous_tmp, path.with_name('previous.pt'))
        os.replace(next_path, path)
        return value

    def restore(self, state, config_identity):
        if state['config_sha256'] != config_identity:
            raise ValueError('The run configuration differs from the saved checkpoint.')
        integration.load_increments_state(self.pipe, state['adapter'], state['lora'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.lr_scheduler.load_state_dict(state['lr_scheduler'])
        restore_rng(state['rng'])


def finalize(run, trainer, state):
    out = run / ('交付' if trainer.native else '交付_训练完成')
    out.mkdir(exist_ok=True)
    done = dict(time=now(), status='completed', optimizer_updates=TOTAL, samples_consumed=TOTAL, epochs=3,
                aux_updates=state['aux_updates'], elapsed_seconds=state['seconds'], peak_vram_bytes=state['peak'])
    for name, tensors in zip(('adapter', 'lora'), integration.increments(trainer.pipe)):
        path = out / f'{name}_final.safetensors'
        temporary = path.with_suffix('.writing')
        save_file(tensors, str(temporary))
        os.replace(temporary, path)
        done[name + '_final'] = str(path)
        done[name + '_sha256'] = sha(path)
    write_json(run / 'training_complete.json', done)


def summarize(block, seconds):
    active = [v for v in block if v['rgb_active']]
    rows = {}
    keys = ['fm', 'loss', 'appearance', 'structure', 'temp', 'aux']
    if any('pair' in r for r in block):
        keys.extend(('pair', 'keep'))
    for key in keys:
        values = [r[key] for r in block if key in r]
        rows[key] = sum(values) / len(values) if values else None
    return dict(**rows, steps=len(block), aux_count=len(active), mean_seconds=seconds / len(block))


def train(run):
    from rgb_recipe import load_schedule
    if (run / 'training_complete.json').is_file():
        return 0
    config = data.read_json(run / 'config.json')
    rows, order = load_schedule(run)
    if len(order) != TOTAL:
        raise ValueError(f'Expected {TOTAL} fixed windows, got {len(order)}')
    identity = sha(run / 'config.json')
    checkpoint = run / 'checkpoints/latest.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False) if checkpoint.exists() else {}
    count = int(state.get('updates', 0))
    board = dict(total=TOTAL, updates=count, durable_updates=count, cache=count, cached=count,
                 aux_updates=state.get('aux_updates', 0), peak_vram_bytes=state.get('peak', 0))
    progress(run, board, stage='loading_model')
    block = state.get('summary_rows', [])
    partial_seconds = state.get('summary_seconds', 0.)
    completed_seconds = state.get('seconds', 0.)
    samples = deque(maxlen=32)
    peak = board['peak_vram_bytes']
    aux_count = board['aux_updates']
    with ThreadPoolExecutor(max_workers=PREFETCH) as pool:
        pending = {}
        def submit(i):
            if i < TOTAL:
                building, frames = order[i]
                pending[i] = pool.submit(windows.load_window, rows[building], frames, config['task'] == PARALLAX_TASK)
        for i in range(count, min(count + PREFETCH, TOTAL)):
            submit(i)
        trainer = Trainer(run, config)
        if state:
            trainer.restore(state, identity)
            if count == 1 and not (run / 'first_update.json').exists():
                write_json(run / 'first_update.json', state['first_update'])
        torch.cuda.reset_peak_memory_stats()
        measure_fm = trainer.native and not (run / 'first_fm.json').is_file()
        start = last_save = block_start = time.monotonic()
        last_count = count
        with (run / 'loss.jsonl').open('a+b') as log:
            log.truncate(state.get('log_bytes', 0))
            log.seek(0, os.SEEK_END)
            proof = state.get('first_update')
            def store():
                nonlocal last_save, last_count
                log.flush()
                os.fsync(log.fileno())
                saved = dict(updates=count, seconds=completed_seconds + time.monotonic() - start, peak=peak,
                             aux_updates=aux_count, summary_rows=block,
                             summary_seconds=partial_seconds + time.monotonic() - block_start,
                             log_bytes=log.tell(), first_update=proof)
                trainer.store(checkpoint, identity, saved)
                last_save, last_count = time.monotonic(), count
                board['durable_updates'] = count
                return saved

            while count < TOTAL:
                if (run / 'stop.request').exists():
                    if count > last_count:
                        store()
                    progress(run, board, stage='paused')
                    return 75
                began = time.monotonic()
                building, frames = order[count]
                progress(run, board, stage='caching', building=building, epoch=count // PER_EPOCH + 1)
                raw = pending.pop(count).result()
                submit(count + PREFETCH)
                if measure_fm:
                    torch.cuda.reset_peak_memory_stats()
                encoded = trainer.encode(raw)
                progress(run, board, stage='training', cache=count+1, cached=count+1)
                record, confirmation = trainer.update(raw, encoded, count + 1)
                del raw, encoded
                count += 1
                aux_count += int(record['rgb_active'])
                peak = max(peak, torch.cuda.max_memory_allocated())
                block.append(record)
                if measure_fm and not record['rgb_active']:
                    write_json(run / 'first_fm.json', dict(updates=count, building=building,
                               allocated_bytes=torch.cuda.max_memory_allocated(),
                               reserved_bytes=torch.cuda.max_memory_reserved(), **record))
                    measure_fm = False
                samples.append(time.monotonic() - began)
                if count == 1:
                    proof = dict(time=now(), updates=1, building=building, frame_ids=frames,
                                 auxiliary_frames=frames[:5], actual_vace_forwards=1,
                                 total_backwards=1, optimizer_updates=1, peak_allocated_bytes=peak,
                                 peak_reserved_bytes=torch.cuda.max_memory_reserved(), **record, **confirmation)
                if count % SUMMARY == 0 or count == TOTAL:
                    report = dict(time=now(), updates=count, aux_updates=aux_count,
                                  peak_vram_bytes=peak, lr=record['lr'],
                                  **summarize(block, partial_seconds + time.monotonic() - block_start))
                    line = json.dumps(report, ensure_ascii=False, allow_nan=False)
                    log.write((line + '\n').encode())
                    log.flush()
                    print(line, flush=True)
                    block = []
                    partial_seconds, block_start = 0., time.monotonic()
                if count == 1 or count - last_count >= SAVE_UPDATES or time.monotonic() - last_save >= SAVE_SECONDS or count == TOTAL:
                    store()
                    if count == 1:
                        write_json(run / 'first_update.json', proof)
                progress(run, board, stage='training', updates=count, aux_updates=aux_count,
                         seconds_per_update=sum(samples) / len(samples), peak_vram_bytes=peak,
                         last_fm=record['fm'], last_sigma=record['sigma'])
            if count == TOTAL:
                final = dict(aux_updates=aux_count, seconds=completed_seconds + time.monotonic() - start, peak=peak)
                finalize(run, trainer, final)
                progress(run, board, stage='training_complete')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    raise SystemExit(train(parser.parse_args().run.resolve()))
