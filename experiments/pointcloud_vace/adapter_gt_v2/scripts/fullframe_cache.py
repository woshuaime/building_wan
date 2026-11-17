"""Untiled cache and its strict training reader for an existing frozen GT-video run."""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json
from pathlib import Path
import os
import shutil
import time

from common import CACHE, ROOT, atomic_torch_save, now, sha, write_json
import data
import worker

CONTRACT = "wan21-untiled-bf16-independent-17-v1"
LATENT_SHAPE = (1, 16, 5, 60, 104)
AREA_SHAPE = (1, 2, 5, 60, 104)


def folder(run):
    return run / "cache_fullframe"


@lru_cache(maxsize=2)
def config(run):
    cfg = json.loads((folder(run) / "config.json").read_text())
    check = cfg.copy()
    if check.pop("config_id") != data.digest(check) or cfg["encoding_contract"] != CONTRACT:
        raise ValueError("Full-frame configuration digest mismatch")
    return cfg


def cache_dir(run):
    return CACHE / config(run)["config_id"]


def entry(directory, row):
    key = f"{row['sample_index']:05d}_{row['id']}_{row['start']:02d}"
    return directory / "latents" / key[:2] / (key + ".pt")


def cache_key(run, input_key):
    # The frozen data reader fingerprints the original inputs, camera and loss.
    # The active encoding and its namespace are bound independently.
    return data.digest([input_key, CONTRACT, config(run)["config_id"]])


def validate(value, row, expected_key, encoding_id):
    import torch
    if (value.get("encoding_id") != encoding_id or value.get("cache_key") != expected_key
            or value["id"] != row["id"] or value["frame_ids"] != row["frame_ids"]):
        raise ValueError(f"Wrong cache identity for window {row['sample_index']}")
    for key, shape, dtype in (("input_latents", LATENT_SHAPE, torch.bfloat16),
                              ("reactive", LATENT_SHAPE, torch.bfloat16),
                              ("occupancy", AREA_SHAPE, torch.float32)):
        tensor = value[key]
        if tuple(tensor.shape) != shape or tensor.dtype != dtype or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid {key} for window {row['sample_index']}")


def read_sample(run, pipe, row, mode, persist=False):
    import torch
    if mode != "frozen" or persist:
        raise ValueError("Full-frame training requires the completed frozen cache")
    value = torch.load(entry(cache_dir(run), row), weights_only=True, map_location="cpu")
    validate(value, row, cache_key(run, data.cache_key(run, row)), config(run)["config_id"])
    return value, data.geometry_for(run, row)


def read_shared(run, pipe):
    import torch
    shared = torch.load(cache_dir(run) / "shared.pt", weights_only=True, map_location="cpu")
    if (shared.get("encoding_id") != config(run)["config_id"] or tuple(shared["inactive"].shape) != LATENT_SHAPE
            or shared["inactive"].dtype != torch.bfloat16 or not torch.isfinite(shared["inactive"]).all()
            or not torch.count_nonzero(shared["inactive"]) or not torch.isfinite(shared["context"]).all()):
        raise ValueError("Missing or invalid full-frame common conditioning")
    if getattr(pipe, "text_encoder", None) is not None:
        pipe.text_encoder.to("cpu")
        torch.cuda.empty_cache()
    return shared


def make_shared(run, vae):
    import torch
    original = json.loads((run / "config.json").read_text())
    source = CACHE / original["config_id"] / "shared.pt"
    if sha(source) != config(run)["parent_shared_sha256"]:
        raise ValueError("Original fixed text cache changed")
    context = torch.load(source, weights_only=True, map_location="cpu")["context"]
    zero_input = torch.zeros((1, 3, 17, 480, 832), device="cuda", dtype=torch.bfloat16)
    inactive = vae.encode(zero_input, device="cuda", tiled=False).cpu().contiguous()
    if not torch.count_nonzero(inactive):
        raise ValueError("Full-frame inactive VAE latent is unexpectedly zero")
    atomic_torch_save(cache_dir(run) / "shared.pt",
                      dict(context=context, inactive=inactive, encoding_id=config(run)["config_id"]))


def configure_training_reader(run):
    # V1's training and checkpoint routines are frozen by the live run. Bind
    # only their input/configuration functions inside this dedicated child.
    worker.read_config = config
    worker.sample_encoded = read_sample
    worker.text_and_inactive = read_shared


def prefetch(rows, load):
    """One CPU reader prepares the next window while the caller uses the current one."""
    items = iter(rows)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="cache-cpu") as pool:
        row = next(items, None)
        if row is None:
            return
        future = pool.submit(load, row)
        while row is not None:
            raw = future.result()
            following = next(items, None)
            if following is not None:
                future = pool.submit(load, following)
            yield row, raw
            row = following


def raw_input(run, row):
    raw = data.load_raw(run, row)
    # The CPU reader has populated the fixed building geometry on disk. Its
    # per-window projection is constructed again by the training reader.
    del raw["geometry"]
    return raw


def official_check(run, vae, encode, rows):
    import numpy as np
    import torch
    from PIL import Image
    from diffsynth.pipelines.wan_video import WanVideoPipeline
    from cache_acceleration import difference
    from vace_roof_training import encode_sample

    pipe = WanVideoPipeline(device="cuda", torch_dtype=torch.bfloat16)
    pipe.vae = vae
    pipe.scheduler.set_timesteps(1000, training=True, shift=5)
    shared = read_shared(run, pipe)
    settings = dict(tiled=False, tile_size=[30, 52], tile_stride=[15, 26], gradient_checkpointing=True,
                    gradient_checkpointing_offload=False, min_timestep_boundary=0., max_timestep_boundary=1.)
    records = []
    encoding_id = config(run)["config_id"]
    for row in rows:
        raw = data.load_raw(run, row)
        optimized = encoded_value(run, row, raw, encode, encoding_id)
        sample = {name: [Image.fromarray(np.repeat(frame[..., None], 3, axis=-1)) for frame in raw[key]]
                  for name, key in (("gt", "target"), ("conditioning", "partial"))}
        sample["model_mask"] = [Image.new("RGB", (832, 480), "white") for _ in row["frame_ids"]]
        reference = encode_sample(pipe, sample, settings, shared["context"].to("cuda"), 1.)
        clean, conditioned, _, _, _ = worker.assemble([optimized], [raw["geometry"]], shared, "cuda")
        target = difference(clean, reference["input_latents"])
        condition = difference(conditioned, reference["vace_context"])
        if not target["passed"] or not condition["passed"]:
            raise ValueError(f"Full-frame official VACE comparison failed: {row['id']} {target} {condition}")
        records.append(dict(id=row["id"], frame_ids=row["frame_ids"], target=target, vace_96_channels=condition))
    return records


def encoded_value(run, row, raw, encode, encoding_id):
    value = encode(raw)
    value.update(cache_key=cache_key(run, raw["cache_key"]), encoding_id=encoding_id)
    validate(value, row, cache_key(run, raw["cache_key"]), encoding_id)
    return value


def report(run, stage, **fields):
    state = dict(time=now(), stage=stage, status=stage, pid=os.getpid(),
                 supervisor_pid=int((run / "run.pid").read_text()), cache_candidate="whole_frame",
                 total_samples=worker.TOTAL, samples_consumed=0, optimizer_updates=0, epoch=0,
                 epoch_completed=0, epoch_buildings=worker.PER_EPOCH, validation_started=False, **fields)
    write_json(run / "progress.json", state)
    write_json(run / "status.json", state)
    print(json.dumps(state, ensure_ascii=False), flush=True)


def check_stop(run):
    if (run / "stop.request").exists():
        raise InterruptedError("Stopped at a complete full-frame cache file")


def check_memory(monitor, budget):
    if monitor.errors or monitor.peak > budget:
        raise MemoryError(f"NVML budget/monitor failure: peak={monitor.peak}, errors={monitor.errors}")


def cache_windows(run, qualify=False):
    import torch
    from cache_encoder import load_vae, make_encoder
    from common import nvml_snapshot

    schedule = json.loads((run / "schedule.json").read_text())
    cfg = config(run)
    if len(schedule) != worker.TOTAL or sha(run / "schedule.json") != cfg["schedule_sha256"]:
        raise ValueError("Original schedule identity or size changed")
    directory = cache_dir(run)
    directory.mkdir(parents=True, exist_ok=True)
    budget = min(40 << 30, nvml_snapshot()["total_bytes"] - (6 << 30))
    with worker.MemoryMonitor() as monitor, torch.no_grad():
        vae = load_vae(run)
        encode = make_encoder(vae, "whole_frame")
        if qualify:
            report(run, "checking_fullframe", cached_samples=0, detail="official target and full VACE condition")
            make_shared(run, vae)
            examples = json.loads((run / "benchmark_samples.json").read_text())
            checks = official_check(run, vae, encode, [examples[0], examples[len(examples)//2], examples[-1]])
            check_stop(run)
            write_json(folder(run) / "official_encoding.json", dict(time=now(), encoding_id=cfg["config_id"], checks=checks))
            examples = {row["sample_index"]: row for row in examples + schedule[16:41]}
            rows = list(examples.values())[:25]
            if len(rows) != 25:
                raise ValueError("Need 25 distinct real windows for end-to-end timing")
            existing = set()
        else:
            qualification = json.loads((folder(run) / "qualified.json").read_text())
            if qualification["encoding_id"] != cfg["config_id"]:
                raise ValueError("Missing qualification for this encoding")
            read_shared(run, None)
            # Match complete atomic publications against this exact schedule.
            files = set(directory.glob("latents/*/*.pt"))
            existing = {r["sample_index"] for r in schedule if entry(directory, r) in files}
            rows = [row for row in schedule if row["sample_index"] not in existing]
            required = qualification["bytes_per_window"] * len(rows)
            if shutil.disk_usage(directory).free < required + (20 << 30):
                raise OSError("Insufficient space for all remaining windows and the disk reserve")
            report(run, "caching", cached_samples=len(existing), reused_fullframe_samples=len(existing),
                   seconds_per_window=qualification["seconds_per_window"], detail="full-frame production")
        done = len(existing)
        choice_path = folder(run) / "selected_config.json"
        choice = json.loads((choice_path if choice_path.exists() else run / "selected_config.json").read_text())
        training_seconds = choice["projected_train_seconds"] + choice.get("projected_save_seconds", 0)
        started = time.monotonic()
        measured_start = started
        sample_bytes = 0
        torch.cuda.reset_peak_memory_stats()
        for index, (row, raw) in enumerate(prefetch(rows, lambda r: raw_input(run, r)), 1):
            check_stop(run)
            value = encoded_value(run, row, raw, encode, cfg["config_id"])
            path = entry(directory, row)
            atomic_torch_save(path, value)
            if qualify:
                loaded = torch.load(path, weights_only=True, map_location="cpu")
                validate(loaded, row, value["cache_key"], cfg["config_id"])
                if not all(torch.equal(value[k], loaded[k]) for k in ("input_latents", "reactive", "occupancy")):
                    raise ValueError("Full-frame cache disk round trip changed values")
                sample_bytes += path.stat().st_size
            done += 1
            check_memory(monitor, budget)
            elapsed = time.monotonic() - started
            if qualify and index == 5:
                measured_start = time.monotonic()
            if index == 1 or index % 10 == 0 or index == len(rows):
                if shutil.disk_usage(directory).free < 20 << 30:
                    raise OSError("Disk reserve reached")
                seconds = elapsed / index
                report(run, "timing_fullframe" if qualify else "caching", cached_samples=done,
                       new_windows=index, reused_fullframe_samples=len(existing), seconds_per_window=seconds,
                       samples_per_second=1/seconds, stage_remaining_seconds=(worker.TOTAL-done)*seconds,
                       overall_remaining_seconds=(worker.TOTAL-done)*seconds + training_seconds,
                       gpu_memory_bytes=nvml_snapshot()["used_bytes"], peak_nvml_bytes=monitor.peak)
        check_stop(run)
        if qualify:
            seconds = (time.monotonic() - measured_start) / 20
            write_json(folder(run) / "qualified.json", dict(time=now(), encoding_id=cfg["config_id"],
                       warmup=5, timed_windows=20, unique_train_windows=len(rows), seconds_per_window=seconds,
                       bytes_per_window=sample_bytes/len(rows), peak_nvml_bytes=monitor.peak,
                       torch_peak_allocated=torch.cuda.max_memory_allocated(), torch_peak_reserved=torch.cuda.max_memory_reserved(),
                       nvml_errors=monitor.errors, memory_budget_bytes=budget, row_indices=[r["sample_index"] for r in rows]))
            return
    if done != worker.TOTAL:
        raise RuntimeError("Incomplete full-frame cache")
    index_path = run / "cache_index.jsonl"
    temp = folder(run) / "cache_index.building"
    with temp.open("w", encoding="utf-8") as out:
        for row in schedule:
            path = entry(directory, row)
            if not path.is_file():
                raise FileNotFoundError(path)
            out.write(json.dumps(dict(sample_index=row["sample_index"], id=row["id"], frame_ids=row["frame_ids"],
                                      path=str(path), encoding_id=cfg["config_id"])) + "\n")
        out.flush()
        os.fsync(out.fileno())
    os.replace(temp, index_path)
    summary = dict(time=now(), mode="frozen", encoding=CONTRACT, encoding_id=cfg["config_id"], samples=done,
                   index=str(index_path), schedule_sha256=cfg["schedule_sha256"], cache_dir=str(directory),
                   reused_fullframe_samples=len(existing), produced_this_invocation=len(rows),
                   elapsed_seconds_this_invocation=time.monotonic() - started, peak_nvml_bytes=monitor.peak)
    write_json(run / "cache_index.json", summary)
    write_json(run / "cache_complete.json", summary)
