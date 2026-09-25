"""Create deterministic, source-OBJ-grouped train/val/test video splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

try:
    from .experiment_config import ConfigError, PROJECT_ROOT
except ImportError:  # Direct script execution.
    from experiment_config import ConfigError, PROJECT_ROOT


MODEL_NAME_PATTERN = re.compile(r"model(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class Sample:
    video: str
    prompt: str
    source_obj: str
    model_folder: str


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_source_id(value):
    return str(value).replace("\\", "/").casefold()


def video_sort_key(sample):
    stem = Path(sample.video).stem
    match = MODEL_NAME_PATTERN.fullmatch(stem)
    return (0, int(match.group(1))) if match else (1, sample.video.casefold())


def load_metadata(path):
    rows = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"video", "prompt"}.issubset(reader.fieldnames):
            raise ConfigError("metadata.csv必须包含video和prompt两列")
        for line_number, row in enumerate(reader, start=2):
            video = (row.get("video") or "").strip()
            prompt = (row.get("prompt") or "").strip()
            if not video or not prompt:
                raise ConfigError(f"metadata.csv第{line_number}行不完整")
            key = video.casefold()
            if key in rows:
                raise ConfigError(f"metadata.csv存在重复视频: {video}")
            rows[key] = {"video": video, "prompt": prompt}
    if not rows:
        raise ConfigError("metadata.csv为空")
    return rows


def load_samples(manifest_path, metadata_path, dataset_root):
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(f"无法读取视频manifest: {error}") from error
    metadata = load_metadata(metadata_path)
    manifest_items = manifest.get("videos")
    if not isinstance(manifest_items, list) or not manifest_items:
        raise ConfigError("video_manifest.json缺少videos列表")

    samples = []
    manifest_video_names = set()
    for index, item in enumerate(manifest_items):
        if not isinstance(item, dict):
            raise ConfigError(f"manifest第{index}项不是对象")
        video_name = Path(str(item.get("output_mp4", ""))).name
        source_obj = str(item.get("source_obj", "")).strip()
        model_folder = str(item.get("model_folder", "")).strip()
        if not video_name or not source_obj or not model_folder:
            raise ConfigError(f"manifest第{index}项缺少输出、source_obj或model_folder")
        key = video_name.casefold()
        if key in manifest_video_names:
            raise ConfigError(f"manifest存在重复视频: {video_name}")
        manifest_video_names.add(key)
        if key not in metadata:
            raise ConfigError(f"manifest视频不在metadata.csv中: {video_name}")
        video_path = dataset_root / metadata[key]["video"]
        if not video_path.is_file():
            raise ConfigError(f"视频文件不存在: {video_path}")
        samples.append(
            Sample(
                video=metadata[key]["video"],
                prompt=metadata[key]["prompt"],
                source_obj=source_obj,
                model_folder=model_folder,
            )
        )

    metadata_only = set(metadata) - manifest_video_names
    if metadata_only:
        preview = ", ".join(sorted(metadata_only)[:5])
        raise ConfigError(f"metadata.csv包含manifest之外的视频: {preview}")
    declared_count = manifest.get("video_count")
    if declared_count is not None and int(declared_count) != len(samples):
        raise ConfigError(
            f"manifest video_count={declared_count}，实际条目={len(samples)}"
        )
    return samples


def group_samples(samples):
    groups = {}
    for sample in samples:
        group_id = normalized_source_id(sample.source_obj)
        groups.setdefault(group_id, []).append(sample)
    return groups


def choose_pilot_groups(train_group_ids, groups, pilot_count):
    selected = []
    sample_count = 0
    for group_id in train_group_ids:
        group_size = len(groups[group_id])
        if sample_count + group_size > pilot_count:
            continue
        selected.append(group_id)
        sample_count += group_size
        if sample_count == pilot_count:
            return selected
    raise ConfigError(
        f"无法在保持源OBJ分组的前提下精确组成{pilot_count}条pilot样本"
    )


def build_splits(samples, seed, train_ratio, val_ratio, pilot_count):
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1:
        raise ConfigError("train_ratio和val_ratio必须在0到1之间")
    if train_ratio + val_ratio >= 1:
        raise ConfigError("train_ratio + val_ratio必须小于1")
    groups = group_samples(samples)
    group_ids = sorted(groups)
    random.Random(seed).shuffle(group_ids)
    train_group_count = int(len(group_ids) * train_ratio)
    val_group_count = int(len(group_ids) * val_ratio)
    train_ids = group_ids[:train_group_count]
    val_ids = group_ids[train_group_count : train_group_count + val_group_count]
    test_ids = group_ids[train_group_count + val_group_count :]
    pilot_ids = choose_pilot_groups(train_ids, groups, pilot_count)

    def flatten(ids):
        return sorted(
            (sample for group_id in ids for sample in groups[group_id]),
            key=video_sort_key,
        )

    return {
        "train": flatten(train_ids),
        "val": flatten(val_ids),
        "test": flatten(test_ids),
        f"pilot_{pilot_count}": flatten(pilot_ids),
    }


def audit_split_leakage(splits):
    group_sets = {
        name: {normalized_source_id(sample.source_obj) for sample in samples}
        for name, samples in splits.items()
        if not name.startswith("pilot_")
    }
    intersections = {}
    names = sorted(group_sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            key = f"{left}_vs_{right}"
            intersections[key] = len(group_sets[left] & group_sets[right])
    return intersections


def write_metadata(path, samples):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video", "prompt"])
        writer.writeheader()
        for sample in samples:
            writer.writerow({"video": sample.video, "prompt": sample.prompt})


def write_split_index(path, splits):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "video", "source_obj", "model_folder"],
        )
        writer.writeheader()
        for split_name in ("train", "val", "test"):
            for sample in splits[split_name]:
                writer.writerow(
                    {
                        "split": split_name,
                        "video": sample.video,
                        "source_obj": sample.source_obj,
                        "model_folder": sample.model_folder,
                    }
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=str(PROJECT_ROOT / "src" / "video_wan81" / "video_manifest.json"),
    )
    parser.add_argument(
        "--metadata",
        default=str(PROJECT_ROOT / "src" / "video_wan81" / "metadata.csv"),
    )
    parser.add_argument(
        "--dataset-root",
        default=str(PROJECT_ROOT / "src" / "video_wan81"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "training" / "data_splits" / "building_v1"),
    )
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--pilot-count", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    metadata_path = Path(args.metadata).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    for path, label in (
        (manifest_path, "manifest"),
        (metadata_path, "metadata"),
    ):
        if not path.is_file():
            raise ConfigError(f"{label}不存在: {path}")
    if not dataset_root.is_dir():
        raise ConfigError(f"dataset-root不存在: {dataset_root}")

    target_names = [
        "metadata_train.csv",
        "metadata_val.csv",
        "metadata_test.csv",
        f"metadata_pilot_{args.pilot_count}.csv",
        "split_index.csv",
        "split_manifest.json",
    ]
    existing = [output_dir / name for name in target_names if (output_dir / name).exists()]
    if existing and not args.overwrite:
        raise ConfigError("划分文件已存在；确认重建时添加--overwrite")

    samples = load_samples(manifest_path, metadata_path, dataset_root)
    splits = build_splits(
        samples,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        pilot_count=args.pilot_count,
    )
    leakage = audit_split_leakage(splits)
    if any(leakage.values()):
        raise ConfigError(f"检测到源OBJ跨集合泄漏: {leakage}")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_metadata(output_dir / "metadata_train.csv", splits["train"])
    write_metadata(output_dir / "metadata_val.csv", splits["val"])
    write_metadata(output_dir / "metadata_test.csv", splits["test"])
    write_metadata(
        output_dir / f"metadata_pilot_{args.pilot_count}.csv",
        splits[f"pilot_{args.pilot_count}"],
    )
    write_split_index(output_dir / "split_index.csv", splits)

    split_group_counts = {
        name: len({normalized_source_id(sample.source_obj) for sample in values})
        for name, values in splits.items()
    }
    payload = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "version": "building_v1",
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": 1.0 - args.train_ratio - args.val_ratio,
        },
        "pilot_count": args.pilot_count,
        "inputs": {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "metadata": str(metadata_path),
            "metadata_sha256": sha256_file(metadata_path),
            "dataset_root": str(dataset_root),
        },
        "sample_counts": {name: len(values) for name, values in splits.items()},
        "source_group_counts": split_group_counts,
        "leakage_intersections": leakage,
        "files": {name: str(output_dir / name) for name in target_names},
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload["sample_counts"], ensure_ascii=False))
    print(f"源OBJ交叉审计: {leakage}")
    print(f"划分输出: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except ConfigError as error:
        raise SystemExit(f"错误: {error}") from error
