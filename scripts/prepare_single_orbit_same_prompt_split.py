"""Create a deterministic same-prompt split for the single-orbit videos."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path


DEFAULT_PROMPT = (
    "A matte gray untextured architectural massing model stands on a plain "
    "white background as the camera completes a smooth 360-degree orbit at a "
    "fixed elevation."
)


def write_metadata(path: Path, videos: list[str], prompt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video", "prompt"])
        writer.writeheader()
        writer.writerows({"video": video, "prompt": prompt} for video in videos)


def sha256_file_list(dataset_root: Path, videos: list[str]) -> str:
    digest = hashlib.sha256()
    for video in videos:
        path = dataset_root / video
        digest.update(video.encode("utf-8"))
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=project_root / "data" / "single_orbit_videos",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "configs" / "single_orbit_same_prompt_v1",
    )
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()

    if not 0 < args.train_ratio < 1 or not 0 < args.val_ratio < 1:
        raise SystemExit("train and validation ratios must be between 0 and 1")
    if args.train_ratio + args.val_ratio >= 1:
        raise SystemExit("train_ratio + val_ratio must be less than 1")

    dataset_root = args.dataset_root.resolve()
    videos = sorted(
        path.name
        for path in dataset_root.glob("model[0-9][0-9][0-9][0-9][0-9].mp4")
    )
    if not videos:
        raise SystemExit(f"No modelXXXXX.mp4 files found under {dataset_root}")
    if len(set(videos)) != len(videos):
        raise SystemExit("Duplicate video names found")

    shuffled = videos[:]
    random.Random(args.seed).shuffle(shuffled)
    train_count = round(len(shuffled) * args.train_ratio)
    val_count = round(len(shuffled) * args.val_ratio)
    splits = {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }
    if any(not values for values in splits.values()):
        raise SystemExit(f"Split produced an empty subset: { {k: len(v) for k, v in splits.items()} }")

    all_split_videos = sum(splits.values(), [])
    if set(all_split_videos) != set(videos) or len(all_split_videos) != len(set(all_split_videos)):
        raise SystemExit("Split coverage or overlap check failed")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name, values in splits.items():
        write_metadata(output / f"metadata_{name}.csv", values, args.prompt)

    pilot = splits["train"][: min(32, len(splits["train"]))]
    write_metadata(output / "metadata_pilot_32.csv", pilot, args.prompt)
    manifest = {
        "version": "single_orbit_same_prompt_v1",
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": 1 - args.train_ratio - args.val_ratio,
        },
        "prompt": args.prompt,
        "dataset_root": str(dataset_root),
        "dataset_count": len(videos),
        "counts": {name: len(values) for name, values in splits.items()},
        "pilot_count": len(pilot),
        "ordered_video_sha256": hashlib.sha256("\n".join(videos).encode("utf-8")).hexdigest(),
        "split_video_sha256": {
            name: hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()
            for name, values in splits.items()
        },
        "content_sha256": {
            name: sha256_file_list(dataset_root, values) for name, values in splits.items()
        },
    }
    (output / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"dataset={len(videos)} train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
