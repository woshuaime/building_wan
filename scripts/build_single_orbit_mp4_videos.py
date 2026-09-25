"""Convert single-orbit image folders into verified MP4 videos."""

import argparse
import json
import os
import time
import uuid
from pathlib import Path

import cv2


DEFAULT_INPUT = Path(r"D:\code\single_orbit_image")
DEFAULT_OUTPUT = Path(r"D:\code\single_orbit_videos")
DEFAULT_FPS = 18.0
DEFAULT_CODEC = "mp4v"
EXPECTED_FRAME_COUNT = 81


def discover_model_dirs(input_root):
    return sorted(
        (path for path in input_root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )


def load_ordered_frames(model_dir):
    params_path = model_dir / "camera_params.json"
    if not params_path.is_file():
        raise RuntimeError(f"缺少相机参数文件: {params_path}")

    payload = json.loads(params_path.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list) or len(frames) != EXPECTED_FRAME_COUNT:
        actual = len(frames) if isinstance(frames, list) else "invalid"
        raise RuntimeError(
            f"帧数异常: model={model_dir.name}, actual={actual}, "
            f"expected={EXPECTED_FRAME_COUNT}"
        )

    model_root = model_dir.resolve()
    frame_paths = []
    for index, frame in enumerate(frames):
        expected_name = f"view_{index:03d}.png"
        expected_relative = f"images/{expected_name}"
        if not isinstance(frame, dict) or frame.get("file_path") != expected_relative:
            raise RuntimeError(
                f"第{index + 1}帧路径或顺序异常: model={model_dir.name}"
            )
        frame_path = (model_dir / expected_relative).resolve()
        try:
            frame_path.relative_to(model_root)
        except ValueError as error:
            raise RuntimeError(f"图片路径越界: {frame_path}") from error
        if not frame_path.is_file():
            raise RuntimeError(f"图片不存在: {frame_path}")
        frame_paths.append(frame_path)
    return frame_paths


def verify_video(video_path, expected_frames, expected_fps, expected_size):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"生成的视频无法打开: {video_path}")

    reported_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    decoded_frames = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame is None or frame.shape[:2] != (height, width):
            capture.release()
            raise RuntimeError(f"视频中存在无法解码或尺寸异常的帧: {video_path}")
        decoded_frames += 1
    capture.release()

    if reported_frames != expected_frames or decoded_frames != expected_frames:
        raise RuntimeError(
            f"视频帧数异常: reported={reported_frames}, "
            f"decoded={decoded_frames}, expected={expected_frames}"
        )
    if abs(reported_fps - expected_fps) > 0.05:
        raise RuntimeError(
            f"视频帧率异常: actual={reported_fps}, expected={expected_fps}"
        )
    if (width, height) != expected_size:
        raise RuntimeError(
            f"视频尺寸异常: actual={width}x{height}, "
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
    started = time.perf_counter()
    frame_paths = load_ordered_frames(model_dir)
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"首帧无法读取: {frame_paths[0]}")
    height, width = first.shape[:2]
    if width % 2 or height % 2:
        raise RuntimeError(f"视频尺寸必须为偶数: {width}x{height}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.parent / f".{output_path.stem}.{uuid.uuid4().hex}.tmp.mp4"
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*codec),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"无法创建MP4编码器: codec={codec}")

    try:
        for index, frame_path in enumerate(frame_paths):
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"第{index + 1}帧无法读取: {frame_path}")
            if frame.shape[:2] != (height, width):
                raise RuntimeError(
                    f"第{index + 1}帧尺寸不一致: "
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

    verification["conversion_seconds"] = time.perf_counter() - started
    verification["output"] = str(output_path.resolve())
    verification["bytes"] = output_path.stat().st_size
    return verification


def parse_args():
    parser = argparse.ArgumentParser(
        description="把单水平轨道的81张图片合成为经过解码验收的MP4。"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--codec", default=DEFAULT_CODEC)
    parser.add_argument("--model", type=int, default=None, help="只转换指定模型编号")
    parser.add_argument("--start", type=int, default=0, help="按编号排序后的起始位置")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
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
    if args.model is not None:
        model_dirs = [path for path in model_dirs if int(path.name) == args.model]
    else:
        start = max(0, args.start)
        model_dirs = model_dirs[start:]
        if args.limit is not None:
            model_dirs = model_dirs[: max(0, args.limit)]
    if not model_dirs:
        raise SystemExit("没有找到待转换的模型目录")

    failures = []
    for index, model_dir in enumerate(model_dirs, 1):
        output_path = output_root / f"model{int(model_dir.name):05d}.mp4"
        if args.skip_existing and output_path.is_file() and output_path.stat().st_size:
            print(f"[{index}/{len(model_dirs)}] 跳过 {output_path.name}")
            continue
        print(f"[{index}/{len(model_dirs)}] {model_dir.name} -> {output_path.name}")
        try:
            result = build_video(model_dir, output_path, args.fps, args.codec)
        except Exception as error:
            failures.append((model_dir.name, str(error)))
            print(f"  失败: {error}")
            continue
        print(json.dumps(result, ensure_ascii=False, indent=2))

    if failures:
        raise SystemExit(f"转换失败: {failures}")


if __name__ == "__main__":
    main()
