"""Validate the retained final weights and read-only inputs of the released GT V2 run."""
import csv
import hashlib
from pathlib import Path

import data
from common import PROMPT, ROOT, sha

RUN_NAME = "run_20261117_225803"
DATASET = ROOT / "data/pointcloud_vace/pointcloud_video_pairs_v2"
POINTS = ROOT / "data/pointcloud_vace/sample-data-organized/pointcloud"
SOURCES = Path(__file__).resolve().parents[1] / "validation_source_sha256.csv"
MANIFEST_SHA256 = "1832b979f640833c7007b65d68479a1fe46fd445a35fbbc66df8bb6d83b73405"
FINAL_SHA256 = {
    "adapter": "d627cac0e2b8aed908c4efe5a069415a1e1c8cf2f3d9d90e27ae62868ff750f2",
    "lora": "d34c9f5e38c98461015abe3b873a00437fd6ae561f96090525586db8c8ca75f0",
}


def final_weights(run):
    """These paired checksums were recorded after all three epochs and exit 0."""
    if run.name != RUN_NAME:
        raise ValueError(f"Unknown completed delivery: {run.name}")
    result = {}
    for kind, expected in FINAL_SHA256.items():
        path = run / "交付_训练完成" / f"{kind}_final.safetensors"
        if sha(path) != expected:
            raise ValueError(f"Final {kind} differs from the completed training run")
        result[f"{kind}_final"] = str(path.resolve())
        result[f"{kind}_sha256"] = expected
    return result


def model_config():
    """Retain the validation-relevant parts of the frozen full-frame config."""
    config_path = ROOT / "configs/vace_p01_model_sources_v1.json"
    if sha(config_path) != "69b67825427eaa03c1eb18d8207be068cb52d1de4503624f04d1beb09859a753":
        raise ValueError("The approved base model specification changed")
    if data.digest(PROMPT.encode()) != "47e0fb3fb6ee989e0301d496f2d1a7f8acc787c297c5a72777de6c7ce6e6b469":
        raise ValueError("The training prompt changed")
    source = data.read_json(config_path)
    parts = source["reused_components"]
    paths = {key: str(ROOT / spec["path"]) for key, spec in parts.items()}
    paths["checkpoint"] = str(ROOT / source["checkpoint"]["default_local_path"])
    expected_mtime = {"checkpoint": 1792401763160656800, "text_encoder": 1790005496022008300,
                      "vae": 1790002874432994200}
    for key, mtime in expected_mtime.items():
        spec = source["checkpoint"]["accepted_files"][0] if key == "checkpoint" else parts[key]
        stat = Path(paths[key]).stat()
        if stat.st_size != spec["bytes"] or (stat.st_mtime_ns != mtime and sha(paths[key]) != spec["sha256"]):
            raise ValueError(f"Frozen base model changed: {key}")
    for name, expected in parts["tokenizer"]["files_sha256"].items():
        if sha(Path(paths["tokenizer"]) / name) != expected:
            raise ValueError(f"Frozen tokenizer changed: {name}")
    return dict(prompt=PROMPT, negative_prompt="", seed=data.SEED, models={"paths": paths})


def validation_rows():
    manifest = DATASET / "manifest.csv"
    if sha(manifest) != MANIFEST_SHA256:
        raise ValueError("Uploaded split manifest changed after training")
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        return {row["id"]: row for row in csv.DictReader(handle) if row["split_use"] == "validation"}


def source_row(building_id):
    row = validation_rows().get(str(building_id))
    if row is None:
        raise ValueError(f"Not a held-out validation building: {building_id}")
    with SOURCES.open(newline="") as handle:
        frozen = next((item for item in csv.DictReader(handle) if item["id"] == row["id"]), None)
    if frozen is None:
        raise ValueError(f"No validation source identity for {building_id}")
    paths = {key: str(data._relative(DATASET, row[key], key, row["id"]))
             for key in ("camera", "record", "partial_video", "GT_video")}
    pointcloud = POINTS / f"{row['id']}.ply"
    camera_hash, record_hash = sha(paths["camera"]), sha(paths["record"])
    # Hash canonical binary SHA-256 digests in the same order for every held-out ID.
    # The GT video is checked separately, after the final model prediction.
    digests = (camera_hash, record_hash, sha(pointcloud), sha(paths["partial_video"]))
    combined = hashlib.sha256(b"".join(bytes.fromhex(value) for value in digests)).hexdigest()
    if combined != frozen["conditioning_sha256"]:
        raise ValueError(f"Frozen conditioning inputs changed for {building_id}")
    return dict(id=row["id"], split_use="validation", **paths, camera_sha256=camera_hash,
                record_sha256=record_hash, pointcloud=str(pointcloud), source_identity=data.identity(pointcloud),
                gt_video_sha256=frozen["gt_video_sha256"])
