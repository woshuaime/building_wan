"""Split long training videos into overlapping Wan-compatible temporal clips."""

from __future__ import annotations

import argparse
import csv
import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ClipWindow:
    """A zero-based, end-exclusive temporal window."""

    start: int
    end: int


def build_clip_windows(
    total_frames: int,
    clip_frames: int = 17,
    stride: int = 16,
) -> list[ClipWindow]:
    if total_frames < clip_frames:
        raise ValueError(
            f"Video has {total_frames} frames, fewer than clip_frames={clip_frames}."
        )
    if clip_frames <= 0 or (clip_frames - 1) % 4:
        raise ValueError("clip_frames must satisfy 4n+1, for example 17, 33, or 81.")
    if stride <= 0:
        raise ValueError("stride must be positive.")

    last_start = total_frames - clip_frames
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    return [ClipWindow(start, start + clip_frames) for start in starts]


def select_clip_windows(
    windows: list[ClipWindow],
    selection: str,
    source_index: int,
) -> list[ClipWindow]:
    if selection == "all":
        return windows
    if selection == "balanced":
        return [windows[source_index % len(windows)]]
    raise ValueError(f"Unknown clip selection mode: {selection}")


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_rows(metadata_path: Path) -> list[dict[str, str]]:
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"video", "prompt"}.issubset(
            reader.fieldnames
        ):
            raise ValueError("Metadata must contain video and prompt columns.")
        rows = []
        for line_number, row in enumerate(reader, start=2):
            video = (row.get("video") or "").strip()
            prompt = (row.get("prompt") or "").strip()
            if not video or not prompt:
                raise ValueError(f"Metadata line {line_number} is incomplete.")
            rows.append({"video": video, "prompt": prompt})
    if not rows:
        raise ValueError("Metadata contains no samples.")
    return rows


def _decode_video(video_path: Path):
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(video_path))
    metadata = reader.get_meta_data()
    fps = float(metadata.get("fps", 0))
    try:
        frames = [frame for frame in reader]
    finally:
        reader.close()
    height, width = frames[0].shape[:2] if frames else (0, 0)
    if not frames or fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video stream: {video_path}")
    return frames, fps, (width, height)


def _write_clip(output_path: Path, frames, fps: float, size: tuple[int, int]):
    import imageio.v2 as imageio

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.stem}.{uuid.uuid4().hex}.tmp.mp4"
    )
    writer = imageio.get_writer(
        str(temporary),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )
    try:
        for frame in frames:
            if (frame.shape[1], frame.shape[0]) != size:
                raise RuntimeError(f"Unexpected frame size while writing {output_path}")
            writer.append_data(frame)
    finally:
        writer.close()

    reader = imageio.get_reader(str(temporary))
    try:
        decoded = sum(1 for _ in reader)
    finally:
        reader.close()
    if decoded != len(frames):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Clip verification failed: decoded={decoded}, expected={len(frames)}"
        )
    os.replace(temporary, output_path)


def prepare_temporal_clips(
    dataset_root: Path,
    metadata_path: Path,
    output_root: Path,
    *,
    clip_frames: int,
    stride: int,
    selection: str = "all",
    limit: int | None = None,
    overwrite: bool = False,
) -> dict:
    dataset_root = _resolve(dataset_root)
    metadata_path = _resolve(metadata_path)
    output_root = _resolve(output_root)
    rows = _read_rows(metadata_path)
    if limit is not None:
        rows = rows[: max(0, limit)]
    if not rows:
        raise ValueError("No samples selected.")

    metadata_output = output_root / "metadata.csv"
    manifest_output = output_root / "clip_manifest.json"
    if (metadata_output.exists() or manifest_output.exists()) and not overwrite:
        raise FileExistsError(
            f"Output metadata already exists in {output_root}; use --overwrite."
        )

    generated_rows: list[dict[str, str]] = []
    source_records = []
    for source_index, row in enumerate(rows, start=1):
        source_path = (dataset_root / row["video"]).resolve()
        try:
            source_path.relative_to(dataset_root)
        except ValueError as error:
            raise ValueError(f"Video path escapes dataset root: {row['video']}") from error
        if not source_path.is_file():
            raise FileNotFoundError(source_path)

        frames, fps, size = _decode_video(source_path)
        available_windows = build_clip_windows(len(frames), clip_frames, stride)
        windows = select_clip_windows(
            available_windows,
            selection,
            source_index - 1,
        )
        clips = []
        for clip_index, window in enumerate(windows, start=1):
            relative_output = (
                Path("clips")
                / source_path.stem
                / (
                    f"{source_path.stem}__"
                    f"{window.start + 1:04d}-{window.end:04d}.mp4"
                )
            )
            clip_path = output_root / relative_output
            if overwrite or not clip_path.is_file():
                _write_clip(
                    clip_path,
                    frames[window.start : window.end],
                    fps,
                    size,
                )
            generated_rows.append(
                {"video": relative_output.as_posix(), "prompt": row["prompt"]}
            )
            clips.append(
                {
                    "clip_index": clip_index,
                    "video": relative_output.as_posix(),
                    **asdict(window),
                }
            )
        source_records.append(
            {
                "source_index": source_index,
                "source_video": row["video"],
                "total_frames": len(frames),
                "fps": fps,
                "width": size[0],
                "height": size[1],
                "clips": clips,
            }
        )
        print(
            f"[{source_index}/{len(rows)}] {row['video']}: "
            f"{len(frames)} frames -> {len(windows)} clips"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    temporary_metadata = metadata_output.with_suffix(".csv.tmp")
    with temporary_metadata.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("video", "prompt"))
        writer.writeheader()
        writer.writerows(generated_rows)
    os.replace(temporary_metadata, metadata_output)

    manifest = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "dataset_root": str(dataset_root),
        "source_metadata": str(metadata_path),
        "output_root": str(output_root),
        "clip_frames": clip_frames,
        "stride": stride,
        "selection": selection,
        "source_count": len(rows),
        "clip_count": len(generated_rows),
        "sources": source_records,
    }
    temporary_manifest = manifest_output.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_output)
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("src/video_wan81"))
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("training/data_splits/building_v1/metadata_train.csv"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("src/video_wan17_train")
    )
    parser.add_argument("--clip-frames", type=int, default=17)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument(
        "--selection", choices=("all", "balanced"), default="all"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = prepare_temporal_clips(
        args.dataset_root,
        args.metadata,
        args.output_root,
        clip_frames=args.clip_frames,
        stride=args.stride,
        selection=args.selection,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(
        f"Prepared {manifest['clip_count']} clips from "
        f"{manifest['source_count']} source videos."
    )
    print(Path(manifest["output_root"]) / "metadata.csv")


if __name__ == "__main__":
    main()
