"""Build validated MP4 videos from the legacy 64-frame test captures."""

import argparse
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "src" / "test"
DEFAULT_OUTPUT = PROJECT_ROOT / "src" / "video"
MODEL_PATTERN = re.compile(r"model(\d{5})", flags=re.IGNORECASE)
EXPECTED_FRAME_COUNT = 64
DEFAULT_FPS = 18.0
DEFAULT_CODEC = "mp4v"


def model_index(path):
    match = MODEL_PATTERN.fullmatch(path.name)
    return int(match.group(1)) if match else None


def discover_model_dirs(input_root):
    return sorted(
        (
            path
            for path in input_root.iterdir()
            if path.is_dir() and MODEL_PATTERN.fullmatch(path.name)
        ),
        key=model_index,
    )


def load_ordered_frames(model_dir):
    poses_path = model_dir / "camera_poses.json"
    if not poses_path.is_file():
        raise RuntimeError(f"缺少相机姿态文件: {poses_path}")
    payload = json.loads(poses_path.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list) or len(frames) != EXPECTED_FRAME_COUNT:
        actual = len(frames) if isinstance(frames, list) else "invalid"
        raise RuntimeError(
            f"相机姿态帧数异常: {model_dir.name}, "
            f"actual={actual}, expected={EXPECTED_FRAME_COUNT}"
        )

    paths = []
    seen = set()
    for frame in frames:
        if not isinstance(frame, dict):
            raise RuntimeError(f"相机姿态帧格式异常: {model_dir.name}")
        file_name = frame.get("file_path")
        if not isinstance(file_name, str) or not file_name:
            raise RuntimeError(f"图片文件名无效: {model_dir.name}")
        frame_path = (model_dir / file_name).resolve()
        try:
            frame_path.relative_to(model_dir.resolve())
        except ValueError as error:
            raise RuntimeError(f"图片路径越界: {frame_path}") from error
        if file_name in seen:
            raise RuntimeError(f"图片文件重复引用: {model_dir.name}/{file_name}")
        seen.add(file_name)
        if not frame_path.is_file():
            raise RuntimeError(f"图片不存在: {frame_path}")
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


def build_video(model_dir, output_path, fps=DEFAULT_FPS, codec=DEFAULT_CODEC):
    frame_paths, poses_payload = load_ordered_frames(model_dir)
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"首帧无法读取: {frame_paths[0]}")
    height, width = first.shape[:2]
    if width % 2 or height % 2:
        raise RuntimeError(f"视频尺寸必须为偶数: {width}x{height}")

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
        for frame_index, frame_path in enumerate(frame_paths, 1):
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"第{frame_index}帧无法读取: {frame_path}")
            if frame.shape[:2] != (height, width):
                raise RuntimeError(
                    f"第{frame_index}帧尺寸不一致: "
                    f"{frame.shape[1]}x{frame.shape[0]}"
                )
            writer.write(frame)
    finally:
        writer.release()

    try:
        verification = verify_video(
            temporary,
            len(frame_paths),
            float(fps),
            (width, height),
        )
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    return {
        "model_folder": model_dir.name,
        "source_obj": poses_payload.get("source_obj"),
        "camera_poses": str((model_dir / "camera_poses.json").resolve()),
        "output_mp4": str(output_path.resolve()),
        "codec": codec,
        **verification,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="把test中的64帧建筑截图合成为MP4，默认18 FPS。",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--codec", default=DEFAULT_CODEC)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="跳过已有MP4，不重新编码。",
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

    model_dirs = discover_model_dirs(input_root)
    start = max(0, int(args.start))
    if args.limit is None:
        model_dirs = model_dirs[start:]
    else:
        model_dirs = model_dirs[start : start + max(0, int(args.limit))]
    if not model_dirs:
        raise SystemExit(f"没有找到待处理的modelXXXXX文件夹: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    failures = []
    for index, model_dir in enumerate(model_dirs, 1):
        output_path = output_root / f"{model_dir.name}.mp4"
        if args.skip_existing and output_path.is_file() and output_path.stat().st_size:
            print(f"[{index}/{len(model_dirs)}] 跳过 {model_dir.name}")
            continue
        print(f"[{index}/{len(model_dirs)}] {model_dir.name} -> {output_path.name}")
        try:
            result = build_video(
                model_dir,
                output_path,
                fps=args.fps,
                codec=args.codec,
            )
        except Exception as error:
            failures.append({"model_folder": model_dir.name, "error": str(error)})
            print(f"  失败: {error}")
            continue
        results.append(result)
        print(
            f"  已验收: {result['frame_count']}帧, "
            f"{result['fps']:.2f} FPS, "
            f"{result['duration_seconds']:.3f}秒, "
            f"{result['width']}x{result['height']}"
        )

    manifest = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "video_count": len(results),
        "failure_count": len(failures),
        "failures": failures,
        "videos": results,
    }
    manifest_path = output_root / "video_manifest.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    print(
        f"完成: 成功{len(results)}个MP4，失败{len(failures)}个，"
        f"清单: {manifest_path}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
