"""Validate the eight supplied clips and derive observed-only camera features on CPU."""
import argparse
import csv
import shutil
from common import *


def prepare(run):
    import numpy as np
    from geometry import geometry_bundle, render_ids, BG
    from vace_p01_results import validate_video
    from vace_p01_environment import _nvml_snapshot
    if run.exists():
        raise ValueError("Prepare requires an independent new run")
    complete = json.loads((TRAIN_RUN / "training_complete.json").read_text())
    exit_status = json.loads((TRAIN_RUN / "exit.json").read_text())
    if complete["actual_updates"] != 3000 or exit_status["returncode"] != 0:
        raise ValueError("D training has not completed successfully")
    gpu = _nvml_snapshot()
    if gpu["compute_processes"]:
        raise ValueError("GPU is occupied")
    for d in ("cache", "checks", "cases", "logs"):
        (run / d).mkdir(parents=True, exist_ok=True)
    training = json.loads((TRAIN_RUN / "config.json").read_text())
    baseline = json.loads((ROOT / "outputs/pointcloud_vace/0_v3_base/run_20261109_175027/config.json").read_text())
    split_path = ROOT / "data/pointcloud_vace/微调任务/候选空间划分.csv"
    split = {r["id"]: r for r in csv.DictReader(split_path.open(encoding="utf-8-sig"))}
    training_ids = {r["building_id"] for r in json.loads((TRAIN_RUN / "schedule.json").read_text())}
    weights = {k: str(TRAIN_RUN / f"checkpoints/{k}_final.safetensors") for k in EXPECTED_WEIGHTS}
    frozen = {p: sha(p) for p in weights.values()}
    if any(frozen[p] != EXPECTED_WEIGHTS[k] for k, p in weights.items()):
        raise ValueError("Wrong D-3000 increments")
    evaluator = CASES.parent / "evaluate.py"
    frozen[str(evaluator)] = sha(evaluator)
    for source in (TRAIN_SOURCE, Path(__file__).parent):
        for p in source.glob("*.py"):
            frozen[str(p.resolve())] = sha(p)
    records = []
    for case in sorted(CASES.iterdir()):
        if not (case / "case.json").exists():
            continue
        out = run / "cases" / case.name
        out.mkdir()
        print("PREPARING", case.name, flush=True)
        points, colors, cameras, scale, identity = observed_points(case)
        record = json.loads((case / "case.json").read_text())
        if record["id"] in training_ids or split[record["id"]]["use"] != "diagnostic_only":
            raise ValueError("Development evaluation building leaked into training")
        paths = {k: case / "inputs" / k for k in ("vace_video.mp4", "vace_video_mask.mp4", "prompt.txt")}
        for p in [*paths.values(), case / "case.json", case / "reference/camera_poses.json", Path(identity["original_cloud"])]:
            frozen[str(p)] = sha(p)
        video, mask = decode(paths["vace_video.mp4"]), decode(paths["vace_video_mask.mp4"])
        white = validate_inputs(video, mask)
        reports, timestamps = {}, []
        for key in ("vace_video.mp4", "vace_video_mask.mp4"):
            reports[key] = validate_video(paths[key], expected_size=(832, 480), expected_frames=81)
            probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                "frame=best_effort_timestamp_time", "-of", "json", str(paths[key])]))
            timestamps.append([float(x["best_effort_timestamp_time"]) for x in probe["frames"]])
        if timestamps[0] != timestamps[1] or timestamps[0] != [i / 16 for i in range(81)]:
            raise ValueError("Video/mask frame timestamps mismatch")
        # Reconstruct only P, and prove the provided RGB outside its edit mask
        # agrees exactly with the same cameras/projection/point sprites in all 81 frames.
        matches = []
        for i, camera in enumerate(cameras):
            ids, _ = render_ids(points, camera)
            rgb = np.vstack([colors, BG])[ids]
            match = bool(np.array_equal(rgb[~white[i]], video[i][~white[i]]))
            matches.append(match)
        if not all(matches):
            raise ValueError(f"Observed geometry/RGB is misregistered: {case.name}, frames {[i for i,x in enumerate(matches) if not x]}")
        bundle = geometry_bundle(points, cameras, scale, list(range(81)))
        cache = run / "cache" / f"{case.name}_geometry.npz"
        np.savez_compressed(cache, **bundle)
        frozen[str(cache)] = sha(cache)
        prompt = paths["prompt.txt"].read_text(encoding="utf-8")
        if prompt.strip() != training["inference"]["prompt"].strip():
            raise ValueError("Development prompt differs from fixed training prompt")
        shutil.copyfile(paths["prompt.txt"], out / "prompt.txt")
        geometry_record = dict(**identity, all_81_observed_rgb_outside_mask_exact=all(matches),
            reference_frames=list(range(0, 80, 10)), target_frames=list(range(81)),
            expected_structure_shape=[1, 32760, 64], expected_gate_shape=[1, 32760, 1],
            camera_path=str(case / "reference/camera_poses.json"))
        write(out / "input_audit.json", geometry_record)
        records.append(dict(name=case.name, case_dir=str(case), output_dir=str(out), building_id=record["id"],
            split=split[record["id"]], inputs={k: str(v) for k,v in paths.items()}, input_validation=reports,
            prompt=prompt, geometry_cache=str(cache), geometry=geometry_record,
            withheld_observed_points=record["count"], role="development_not_blind_test"))
        del video, mask, bundle
    if len(records) != 8:
        raise ValueError("Exactly eight uploaded development cases are required")
    config = dict(created_at=now(), method="D", checkpoint_updates=3000, training_run=str(TRAIN_RUN),
        training_config_id=training["config_id"], weights=weights, weights_sha256=EXPECTED_WEIGHTS,
        model=baseline["model"], models_verified=baseline["models_verified"], inference=baseline["inference"],
        cases=records, evaluator=str(evaluator), frozen_files=frozen, gpu_at_prepare=gpu,
        inference_budget=8, single_gpu=True, geometry_source="P minus withheld original IDs; original 81 cameras",
        postprocessing=False, replace_last_frame=False, read_gt_before_prediction=False,
        optional_group_training=False, output_contract=dict(raw="cases/<case>/raw_vace.mp4",
            frames="cases/<case>/raw_frames/000.png..080.png", comparison="cases/<case>/input_mask_D_target.mp4",
            metrics="cases/<case>/metrics.json", summary="metrics_summary.json", final_state="exit.json"))
    config["inference"].update(negative_prompt="")
    config["prompt_extension"] = False
    from vace_p01 import inference_arguments
    from diffsynth.pipelines.wan_video import WanVideoPipeline
    import inspect
    args = inference_arguments(config, [None] * 81, [None] * 81)
    if set(args) - set(inspect.signature(WanVideoPipeline.__call__).parameters):
        raise ValueError("Inference configuration contains an unsupported pipeline argument")
    config["config_id"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    write(run / "config.json", config)
    write(run / "progress.json", dict(time=now(), completed_cases=[], total_cases=8, stage="prepared"))
    event(run, "prepared", inference_calls=0, total_cases=8, observed_geometry_alignment="648/648 frames exact outside mask")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    prepare(parser.parse_args().run.resolve())
