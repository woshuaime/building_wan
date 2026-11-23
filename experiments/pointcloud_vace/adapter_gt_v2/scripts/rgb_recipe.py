"""Freeze the three-epoch V2 continuation using only the existing split manifest."""
import csv
import json
import os
import random
import shlex
import shutil
from pathlib import Path

import data
import validation_delivery as delivery
from common import ROOT, PYTHON, now, sha, write_json

OUT = ROOT / "outputs/pointcloud_vace/adapter_gt_v2_rgb"
PARALLAX_OUT = OUT.with_name("adapter_gt_v2_parallax")
PARALLAX_TASK = "v2_native_parallax"
IDS = ("358", "2825", "5954", "4850", "3538", "2016", "6485", "5466")
TOTAL = data.TRAIN_COUNT * 3
AUXILIARY = dict(appearance=.10, structure=.10, temporal=.02, warmup_updates=1000,
                 warmup_initial=.10, sigma_min=.10, sigma_max=.60, prefix_frames=5)
MANIFEST = delivery.DATASET / "manifest.csv"


def train_rows():
    if sha(MANIFEST) != delivery.MANIFEST_SHA256:
        raise ValueError("V2 split manifest identity changed")
    with MANIFEST.open(encoding="utf-8-sig", newline="") as stream:
        records = [r for r in csv.DictReader(stream) if r["split_use"] == "train"]
    # List names only. Individual files, selections and cameras are read on demand.
    with os.scandir(delivery.POINTS) as entries:
        xyz = {Path(e.name).stem for e in entries if e.name.endswith(".xyz")}
    rows = {}
    for record in records:
        building = record["id"]
        paths = {key: str(delivery.DATASET / record[key]) for key in
                 ("camera", "record", "partial_video", "GT_video")}
        suffix = ".xyz" if building in xyz else ".ply"
        rows[building] = dict(id=building, split_use="train", **paths,
                              pointcloud=str(delivery.POINTS / (building + suffix)),
                              mesh=str(delivery.POINTS.parent / "mesh" / (building + ".obj")))
    if len(rows) != len(records) or len(rows) != data.TRAIN_COUNT or set(rows).intersection(IDS):
        raise ValueError("Unexpected training split")
    return rows


def orbit_schedule(ids, offset=0):
    ids = list(ids)
    starts = {b: int(data.digest(f"{data.SEED}:{b}".encode()), 16) % 80 for b in ids}
    schedule = []
    for epoch in range(3):
        order = ids.copy()
        random.Random(data.SEED + epoch).shuffle(order)
        schedule.extend((b, (starts[b] + offset + 27 * epoch) % 80) for b in order)
    return schedule


def load_schedule(run):
    rows = train_rows()
    path = run / "schedule.json"
    config = data.read_json(run / "config.json")
    if sha(path) != config["schedule_sha256"]:
        raise ValueError("Run schedule changed")
    order = [(b, [(s + i) % 80 for i in range(17)]) for b, s in data.read_json(path)]
    return rows, order


def prepare(run, native=False):
    parent = PARALLAX_OUT if native else OUT
    if run.parent != parent or not run.name.startswith("run_"):
        raise ValueError(f"Create run_* directly under {parent}")
    if run.exists() and any(run.iterdir()):
        raise FileExistsError(run)
    weights = delivery.final_weights(ROOT / "outputs/pointcloud_vace/adapter_gt_v2" / delivery.RUN_NAME)
    model = delivery.model_config()
    rows = train_rows()
    schedule = orbit_schedule(rows, offset=13 if native else 0)
    building = schedule[0][0]
    data.validate_camera(data.read_json(rows[building]["camera"]), building, "train")
    run.mkdir(parents=True, exist_ok=True)
    (run / "checkpoints").mkdir(exist_ok=True)
    path = run / "schedule.json"
    path.write_text(json.dumps(schedule, separators=(",", ":")) + "\n")
    cfg = dict(model, created=now(), task="v2_rgb_continuation", source_weights=weights,
               paths={"data": str(delivery.DATASET)}, manifest_sha256=delivery.MANIFEST_SHA256,
               schedule_sha256=sha(path), schedule_rule="unique orbit 0..79, start=(u+27*epoch)%80",
               training=dict(train_buildings=data.TRAIN_COUNT, epochs=3, total_samples=TOTAL,
                             width=512, height=512, frames=17, fps=16, batch=1, gradient_accumulation_steps=1,
                             learning_rate=1.e-4, final_learning_rate=1.e-5, warmup_sample_fraction=.02,
                             adam_betas=[.9, .999], adam_eps=1.e-8, weight_decay=0., gradient_clip_norm=1.,
                             scheduler_steps=1000, scheduler_shift=5., vace_scale=1., lora_rank=8, lora_alpha=8,
                             checkpointing="none", backbone_dtype="bfloat16", trainable_dtype="float32",
                             trainable="original V2 eight-view Adapter and 15 VACE blocks attention q/k/v/o LoRA",
                             input_mask="original V2 all-one edit mask, partial in reactive channels",
                             fm_loss=data.LOSS_VERSION, encoding="full frame", cpu_prefetch_windows=4),
               resize=dict(source=[832,480], canvas=[832,832], pad_top=176, pad_bottom=176, side=512,
                           background=240, interpolation="area"),
               auxiliary=AUXILIARY, appearance_target="original GT sparse full-range video",
               geometry_cache_buildings=256, validation=dict(ids=IDS, width=832,height=480,frames=81,
                   steps=30,seed=20261018,cfg=5,automatic=True),
               recovery=dict(updates=200,seconds=300,keep=["latest","previous"],first_update=True))
    import rgb_supervision
    cfg["rgb_features"] = rgb_supervision.RGB_CONFIG
    if native:
        cfg.update(task=PARALLAX_TASK, schedule_rule="unique orbit 0..79, start=(u+13+27*epoch)%80",
                   resize=dict(source=[832, 480], interpolation="identity"),
                   rgb_features=rgb_supervision.PARALLAX_FEATURES,
                   auxiliary=dict(AUXILIARY, appearance=.05, pair=.03, keep=.02),
                   geometry=dict(cell_pixels=4, layer_transition_radii=[-.1, .1],
                                 frame_slots=4, residual_channels=64),
                   rgb_baseline=str(OUT / "run_20261120_041625"))
        cfg["training"].update(width=832, height=480, checkpointing="full on RGB, none on FM",
                               trainable="V2 eight-view Adapter + ordered depth residual, VACE q/k/v/o LoRA")
    cfg["config_id"] = data.digest(cfg)
    if not native:
        (run / "references").mkdir(exist_ok=True)
        for building in ("2825", "5954"):
            matches = sorted(Path("/tmp").glob(f"scj-v2-roof-20261120_014636-*/vae_{building}.json"))
            if matches:
                shutil.copyfile(matches[-1], run / "references" / f"vae_{building}.json")
    for name, action in (("start","launch"), ("stop","stop"), ("resume","resume"), ("status","status")):
        script = run / f"{name}.sh"
        entry = Path(__file__).resolve().with_name("run.py")
        script.write_text("#!/usr/bin/env bash\nexec " + shlex.join([PYTHON, "-Bu", str(entry), action, str(run)]) + "\n")
        script.chmod(0o755)
    write_json(run / "config.json", cfg)
    return cfg
