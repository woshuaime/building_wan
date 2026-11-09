"""CPU-only source identities, fixed budget/sampling, and first reliable training batch."""
import argparse
import csv
import random
import shutil
import sys
from collections import Counter

from common import *
from geometry import read_cloud, observed_gap, weak_target, make_sample


def prepare(run):
    import numpy as np
    if run.exists():
        raise ValueError("A new run directory is required")
    for directory in ("checks", "checkpoints", "cache", "samples", "attempts"):
        (run / directory).mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "scripts"))
    from vace_p01_environment import _nvml_snapshot
    gpu = _nvml_snapshot()
    if gpu["compute_processes"]:
        raise ValueError(f"GPU busy: {gpu}")
    split_path = DATA / "微调任务/候选空间划分.csv"
    rows = list(csv.DictReader(split_path.open(encoding="utf-8-sig")))
    clouds = {p.stem: p.resolve() for p in (DATA / "sample-data-organized/pointcloud").iterdir() if p.is_file()}
    if len(clouds) != 15881:
        raise ValueError("Wrong point-cloud count")
    train = []
    for r in rows:
        if r["use"] != "train":
            continue
        mesh = DATA / f"sample-data-organized/mesh/{r['id']}.obj"
        if r["id"] not in clouds or not mesh.is_file():
            raise ValueError(f"Missing training pair {r['id']}")
        train.append({**r, "pointcloud": str(clouds[r["id"]]), "mesh": str(mesh.resolve())})
    if len(train) != 10747:
        raise ValueError("Candidate split changed")
    seed = 20261110
    random.Random(seed).shuffle(train)
    probes, first = [], None
    for row in train[:256]:
        points, rgb = read_cloud(row["pointcloud"])
        origin = (points.min(0) + points.max(0)) / 2
        gap_seed = seed + int(row["id"])
        gap = observed_gap(points - origin, gap_seed)
        _, _, _, audit = weak_target(points - origin, rgb, gap, row["mesh"], origin)
        probes.append({"id": row["id"], **audit})
        print("FIRST_BATCH_CANDIDATE", probes[-1], flush=True)
        if audit["weak_reliable"]:
            first = row
            break
    if first is None:
        write_json(run / "checks/first_batch_probes.json", probes)
        raise ValueError("No reliable natural-gap target among first 256 train candidates")
    train = [first] + [r for r in train if r["id"] != first["id"]]
    rng = random.Random(seed + 2)
    schedule = []
    for i, row in enumerate(train[:3000], 1):
        schedule.append(dict(update=i, building_id=row["id"], start=rng.randrange(64),
                             gap_seed=seed + int(row["id"]), noise_seed=rng.randrange(2**31),
                             timestep_index=rng.randrange(1, 999)))
    # The acceptance batch must carry a nonzero scheduler weight as well as a real gradient.
    schedule[0]["timestep_index"] = 500
    first_row = schedule[0]
    sample, labels, geom, audit = make_sample(first, list(range(first_row["start"], first_row["start"] + 17)), first_row["gap_seed"])
    if labels[:, 2].sum() == 0:
        raise ValueError("First fixed window has no weak supervision")
    save_sample(run / "cache/first_batch.npz", sample, labels, geom, audit)
    # Change only the supervision source: no model condition is allowed to change.
    unknown_row = {**first, "mesh": str(run / "missing_mesh_for_leakage_check.obj")}
    # weak_target handles an absent mesh; make_sample's identity record requires a file.
    (run / "checks/unreliable_mesh.obj").write_text("# no reliable reference surface\n")
    unknown_row["mesh"] = str(run / "checks/unreliable_mesh.obj")
    unknown_sample, unknown_labels, unknown_geom, _ = make_sample(
        unknown_row, list(range(first_row["start"], first_row["start"] + 17)), first_row["gap_seed"])
    for key in ("conditioning", "model_mask"):
        if not np.array_equal(np.asarray(sample[key]), np.asarray(unknown_sample[key])):
            raise ValueError("Mesh leaked into input RGB or mask")
    if not all(np.array_equal(geom[k], unknown_geom[k]) for k in geom) or unknown_labels[:, 2].any():
        raise ValueError("Mesh leaked into structural inputs or unknown weak loss")
    write_json(run / "checks/supervision_isolation.json", dict(mesh_swap_preserves_condition=True,
               mesh_swap_preserves_mask=True, mesh_swap_preserves_all_geometry=True, unknown_weak_area=0))
    for k in ("conditioning", "model_mask", "gt"):
        sample[k][0].save(run / "checks" / f"first_training_frame_{k}.png")
    np.savez_compressed(run / "checks/first_training_labels.npz", known=labels[:, 0], real=labels[:, 1], weak=labels[:, 2])
    reference_path = ROOT / "outputs/pointcloud_vace/0_v3_base/run_20261109_175027/config.json"
    reference = json.loads(reference_path.read_text())
    prompt = (DATA / "0_v3/prompt.txt").read_text()
    (run / "prompt.txt").write_text(prompt)
    source = Path(__file__).parent
    write_json(run / "train_manifest.json", train)
    write_json(run / "schedule.json", schedule)
    config = dict(created_at=now(), group="D", python=PYTHON, seed=seed,
                  model=reference["model"], models_verified=reference["models_verified"],
                  baseline_config=str(reference_path), baseline_config_sha256=sha(reference_path),
                  data=dict(root=str(DATA), split=str(split_path), split_sha256=sha(split_path),
                            split_counts=dict(Counter(r["use"] for r in rows)), training_buildings=10747,
                            actual_budget_unique_buildings=3000, first_batch_selection="first reliable gap in seed-shuffled training candidates",
                            first_batch_probes=probes, input_mesh_allowed=False, complementary_scan_metadata_available=False,
                            manifest_sha256=sha(run / "train_manifest.json"), schedule_sha256=sha(run / "schedule.json"),
                            first_batch_sha256=sha(run / "cache/first_batch.npz"),
                            weak_target_weight=.3, unknown_gap_weight=0),
                  training=dict(max_updates=3000, checkpoint_interval=100, first_update_checkpoint=True,
                                batch_size=1, gradient_accumulation=1, width=832, height=480, num_frames=17,
                                learning_rate=1.e-4, weight_decay=0., adam_betas=[.9, .999], adam_eps=1.e-8,
                                max_grad_norm=1., backbone_dtype="bfloat16", increment_dtype="float32",
                                gradient_checkpointing=True, gradient_checkpointing_offload=False,
                                tiled=True, tile_size=[30, 52], tile_stride=[15, 26],
                                min_timestep_boundary=0., max_timestep_boundary=1., scheduler_training_steps=1000,
                                scheduler_shift=5., loss_region_weights=[1., 1., .3],
                                lora_base_model="vace", lora_target_modules=["q", "k", "v", "o"], lora_rank=8, lora_alpha=8,
                                trainable_models=None, adapter_parameters=142929, lora_parameters=2949120,
                                oom_fallbacks=["17_frames_checkpoint_cpu_offload", "13_frames_checkpoint_cpu_offload"]),
                  inference=dict(prompt=prompt, negative_prompt="", prompt_extension=False, vace_scale=1., fps=16),
                  renderer=dict(background_rgb=[237, 240, 244], point_radius_px=2, azimuth_views=80,
                                elevation_degrees=20., vertical_fov_degrees=50., sphere_fit_margin=1.12,
                                camera_basis="right,-up,forward", source="package sprite renderer and stored orbit conventions; original external generator absent",
                                first_last_frame="frame080 excluded from training"),
                  source_sha256={p.name: sha(p) for p in sorted(source.glob("*.py"))}, gpu_at_prepare=gpu,
                  outputs=dict(final_adapter="checkpoints/adapter_final.safetensors", final_lora="checkpoints/lora_final.safetensors",
                               resume="checkpoints/resume_latest.pt", log="run.log", first_update="first_update.json",
                               status="status.json", exit="exit.json", automatic_evaluation=False))
    config["config_id"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    write_json(run / "config.json", config)
    write_json(run / "checks/first_batch_probes.json", probes)
    shutil.copyfile(DATA / "apc2mesh_full_v1_import.json", run / "import.json")
    event(run, "prepared", actual_updates=0, planned_updates=3000, first_building=first["id"], first_label_pixels=audit["label_pixel_counts"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    prepare(parser.parse_args().run.resolve())
