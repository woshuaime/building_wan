"""Build validated MP4 videos from test2 video sequences."""

import argparse
import csv
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "src" / "test2"
DEFAULT_OUTPUT = PROJECT_ROOT / "src" / "video_more"
MODEL_PATTERN = re.compile(r"model(\d{5})", flags=re.IGNORECASE)
DEFAULT_FPS = 18.0
DEFAULT_CODEC = "mp4v"
DEFAULT_TRAINING_PROMPT = (
    "a clean 3D architectural building model rotating on a white background"
)


def _model_index(path):
    match = MODEL_PATTERN.fullmatch(path.name)
    return int(match.group(1)) if match else None


def discover_model_dirs(input_root):
    return sorted(
        (
            path
            for path in input_root.iterdir()
            if path.is_dir() and MODEL_PATTERN.fullmatch(path.name)
        ),
        key=_model_index,
    )


def load_ordered_frames(model_dir):
    sequence_path = model_dir / "video_sequence.json"
    if not sequence_path.is_file():
        raise RuntimeError(f"缺少视频顺序文件: {sequence_path}")
    payload = json.loads(sequence_path.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"视频顺序为空或格式错误: {sequence_path}")

    ordered = sorted(frames, key=lambda item: int(item["video_frame_index"]))
    actual_indices = [int(item["video_frame_index"]) for item in ordered]
    expected_indices = list(range(1, len(ordered) + 1))
    if actual_indices != expected_indices:
        raise RuntimeError(
            f"video_frame_index不连续: {model_dir.name}"
        )

    paths = []
    for frame in ordered:
        file_name = frame.get("file_path")
        if not isinstance(file_name, str) or not file_name:
            raise RuntimeError(f"帧文件名无效: {model_dir.name}")
        frame_path = (model_dir / file_name).resolve()
        try:
            frame_path.relative_to(model_dir.resolve())
        except ValueError as error:
            raise RuntimeError(f"帧路径越界: {frame_path}") from error
        if not frame_path.is_file():
            raise RuntimeError(f"帧图片不存在: {frame_path}")
        paths.append(frame_path)
    return paths, payload


def verify_video(video_path, expected_frames, expected_fps, expected_size):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"生成的MP4无法打开: {video_path}")
    reported_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    decoded_frames = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame is None or frame.shape[1] != width or frame.shape[0] != height:
            capture.release()
            raise RuntimeError(f"MP4中存在尺寸异常帧: {video_path}")
        decoded_frames += 1
    capture.release()

    if reported_frames != expected_frames or decoded_frames != expected_frames:
        raise RuntimeError(
            f"MP4帧数异常: reported={reported_frames}, "
            f"decoded={decoded_frames}, expected={expected_frames}"
        )
    if abs(reported_fps - expected_fps) > 0.05:
        raise RuntimeError(
            f"MP4帧率异常: actual={reported_fps}, expected={expected_fps}"
        )
    if (width, height) != expected_size:
        raise RuntimeError(
            f"MP4尺寸异常: actual={width}x{height}, "
            f"expected={expected_size[0]}x{expected_size[1]}"
        )
    return {
        "frame_count": decoded_frames,
        "fps": reported_fps,
        "width": width,
        "height": height,
        "duration_seconds": decoded_frames / reported_fps,
    }


def build_video(
    model_dir,
    output_path,
    fps=DEFAULT_FPS,
    codec=DEFAULT_CODEC,
    frame_count=None,
    output_size=None,
):
    frame_paths, sequence_payload = load_ordered_frames(model_dir)
    source_frame_count = len(frame_paths)
    if frame_count is not None:
        frame_count = int(frame_count)
        if frame_count <= 0:
            raise RuntimeError("目标帧数必须大于0")
        if source_frame_count < frame_count:
            raise RuntimeError(
                f"源序列帧数不足: actual={source_frame_count}, "
                f"expected_at_least={frame_count} ({model_dir.name})"
            )
        selected_paths = frame_paths[:frame_count]
    else:
        selected_paths = frame_paths

    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"首帧无法读取: {frame_paths[0]}")
    source_height, source_width = first.shape[:2]
    if output_size is None:
        width, height = source_width, source_height
    else:
        width, height = (int(output_size[0]), int(output_size[1]))
        if width <= 0 or height <= 0:
            raise RuntimeError("输出宽高必须大于0")
    if width % 2 or height % 2:
        raise RuntimeError(
            f"视频尺寸必须为偶数: {width}x{height} ({model_dir.name})"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.parent / (
        f".{output_path.stem}.{uuid.uuid4().hex}.tmp.mp4"
    )
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*codec),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"无法创建MP4编码器: codec={codec}, output={temporary}"
        )

    try:
        for frame_index, frame_path in enumerate(selected_paths, 1):
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"第{frame_index}帧无法读取: {frame_path}")
            if frame.shape[:2] != (source_height, source_width):
                raise RuntimeError(
                    f"第{frame_index}帧尺寸不一致: "
                    f"{frame.shape[1]}x{frame.shape[0]}"
                )
            if (source_width, source_height) != (width, height):
                interpolation = (
                    cv2.INTER_AREA
                    if width <= source_width and height <= source_height
                    else cv2.INTER_LANCZOS4
                )
                frame = cv2.resize(
                    frame,
                    (width, height),
                    interpolation=interpolation,
                )
            writer.write(frame)
    finally:
        writer.release()

    try:
        verification = verify_video(
            temporary,
            len(selected_paths),
            float(fps),
            (width, height),
        )
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    return {
        "model_folder": model_dir.name,
        "source_obj": sequence_payload.get("source_obj"),
        "sequence_json": str(
            (model_dir / "video_sequence.json").resolve()
        ),
        "output_mp4": str(output_path.resolve()),
        "codec": codec,
        "source_frame_count": source_frame_count,
        "selected_frame_count": len(selected_paths),
        "dropped_frame_files": [
            path.name for path in frame_paths[len(selected_paths):]
        ],
        "source_width": source_width,
        "source_height": source_height,
        **verification,
    }


def write_training_metadata(output_root, results, prompt):
    metadata_path = output_root / "metadata.csv"
    temporary = metadata_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("video", "prompt"))
        writer.writeheader()
        for result in results:
            writer.writerow({
                "video": Path(result["output_mp4"]).name,
                "prompt": prompt,
            })
    os.replace(temporary, metadata_path)
    return metadata_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="把test2中的视频序列合成为MP4，默认18 FPS。",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--codec", default=DEFAULT_CODEC)
    parser.add_argument(
        "--frame-count",
        type=int,
        default=None,
        help="只保留JSON顺序中的前N帧；Wan训练建议设为81。",
    )
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument(
        "--write-training-metadata",
        action="store_true",
        help="在输出目录生成DiffSynth训练用metadata.csv。",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_TRAINING_PROMPT,
        help="metadata.csv中每个视频使用的训练提示词。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅处理自然排序后的前N个model文件夹。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_root = args.input.resolve()
    output_root = args.output.resolve()
    if not input_root.is_dir():
        raise SystemExit(f"输入目录不存在: {input_root}")
    if args.fps <= 0:
        raise SystemExit("fps必须大于0")
    if len(args.codec) != 4:
        raise SystemExit("codec必须是4个字符，例如mp4v")
    if (args.width is None) != (args.height is None):
        raise SystemExit("--width和--height必须同时提供")
    if args.frame_count is not None:
        if args.frame_count <= 0:
            raise SystemExit("--frame-count必须大于0")
        if (args.frame_count - 1) % 4 != 0:
            raise SystemExit(
                "Wan训练帧数必须满足4n+1，例如17、49或81"
            )
    if args.width is not None:
        if args.width <= 0 or args.height <= 0:
            raise SystemExit("输出宽高必须大于0")
        if args.width % 16 or args.height % 16:
            raise SystemExit("Wan训练视频宽高必须是16的倍数")

    model_dirs = discover_model_dirs(input_root)
    if args.limit is not None:
        model_dirs = model_dirs[: max(0, int(args.limit))]
    if not model_dirs:
        raise SystemExit(f"没有找到modelXXXXX文件夹: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for index, model_dir in enumerate(model_dirs, 1):
        output_path = output_root / f"{model_dir.name}.mp4"
        print(
            f"[{index}/{len(model_dirs)}] {model_dir.name} -> {output_path.name}"
        )
        result = build_video(
            model_dir,
            output_path,
            fps=args.fps,
            codec=args.codec,
            frame_count=args.frame_count,
            output_size=(args.width, args.height)
            if args.width is not None
            else None,
        )
        results.append(result)
        print(
            f"  已验收: {result['frame_count']}帧, "
            f"{result['fps']:.2f} FPS, "
            f"{result['duration_seconds']:.3f}秒, "
            f"{result['width']}x{result['height']}"
        )

    manifest = {
        "generated_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "requested_frame_count": args.frame_count,
        "requested_output_size": (
            [args.width, args.height] if args.width is not None else None
        ),
        "video_count": len(results),
        "videos": results,
    }
    manifest_path = output_root / "video_manifest.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    if args.write_training_metadata:
        metadata_path = write_training_metadata(
            output_root,
            results,
            args.prompt,
        )
        print(f"训练元数据: {metadata_path}")
    print(f"完成: {len(results)}个MP4，清单: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
