"""Render a few building models as one upper-hemisphere video sequence."""

import argparse
import json
import math
import os
import shutil
import sys
import time
import uuid
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from core.obj_loader import OBJLoader
from core.track import TrackSystem

# Keep geometry planning importable on headless machines. VTK/PyVista is only
# needed when a real capture is started, and importing it at module load time
# made even pure sequence tests depend on a GUI stack.
BATCH_ORBIT_RADIUS_FACTOR = 3.5
DEFAULT_SOURCE = Path(r"D:\scj\down_load\三七\建筑物模型")
PHOTOS_PER_TRACK = 16


DEFAULT_OUTPUT = PROJECT_ROOT / "src" / "test2"
DEFAULT_VIDEO_SAVE_WORKERS = 2
DEFAULT_PNG_COMPRESS_LEVEL = 1
BASE_TRACK_STEP_DEG = 180.0 / PHOTOS_PER_TRACK
MAX_ANGULAR_STEP_DEG = BASE_TRACK_STEP_DEG / 2.0
PREVIEW_SIZE = 512
PREVIEW_FRAME_MS = 170
VIDEO_TRACK_PLAN = (
    (0, 1),
    (2, 1),
    (1, -1),
    (3, 1),
)


def _unit(vector):
    array = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a zero-length vector")
    return array / norm


def _slerp_unit(start, end, fraction):
    start = _unit(start)
    end = _unit(end)
    dot = float(np.clip(np.dot(start, end), -1.0, 1.0))
    angle = math.acos(dot)
    if angle <= 1e-10:
        return start.copy()
    sin_angle = math.sin(angle)
    return (
        math.sin((1.0 - fraction) * angle) / sin_angle * start
        + math.sin(fraction * angle) / sin_angle * end
    )


def _camera_up(position, center, candidate, fallback):
    look = _unit(np.asarray(center, dtype=np.float64) - position)
    up = np.asarray(candidate, dtype=np.float64)
    up = up - np.dot(up, look) * look
    if float(np.linalg.norm(up)) <= 1e-8:
        up = np.asarray(fallback, dtype=np.float64)
        up = up - np.dot(up, look) * look
    if float(np.linalg.norm(up)) <= 1e-8:
        axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(axis, look))) > 0.9:
            axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        up = axis - np.dot(axis, look) * look
    return _unit(up)


def _rotate_around_axis(vector, axis, angle):
    vector = np.asarray(vector, dtype=np.float64)
    axis = _unit(axis)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return (
        vector * cosine
        + np.cross(axis, vector) * sine
        + axis * np.dot(axis, vector) * (1.0 - cosine)
    )


def _smooth_transition_view_ups(positions, center, start_up, end_up):
    center = np.asarray(center, dtype=np.float64)
    positions = [np.asarray(position, dtype=np.float64) for position in positions]
    looks = [_unit(center - position) for position in positions]
    transported = [
        _camera_up(positions[0], center, start_up, end_up)
    ]

    for index in range(1, len(positions)):
        previous_up = transported[-1]
        look = looks[index]
        candidate = previous_up - np.dot(previous_up, look) * look
        up = _camera_up(
            positions[index],
            center,
            candidate,
            previous_up,
        )
        if float(np.dot(up, previous_up)) < 0.0:
            up = -up
        transported.append(up)

    target_end_up = _camera_up(
        positions[-1],
        center,
        end_up,
        transported[-1],
    )
    roll_correction = math.atan2(
        float(
            np.dot(
                looks[-1],
                np.cross(transported[-1], target_end_up),
            )
        ),
        float(np.clip(np.dot(transported[-1], target_end_up), -1.0, 1.0)),
    )

    smoothed = []
    denominator = max(1, len(positions) - 1)
    for index, (up, look) in enumerate(zip(transported, looks)):
        fraction = index / float(denominator)
        corrected = _rotate_around_axis(
            up,
            look,
            roll_correction * fraction,
        )
        smoothed.append(_unit(corrected))
    return smoothed, math.degrees(roll_correction)


def _rotation_minimizing_view_ups(positions, center, initial_up):
    center = np.asarray(center, dtype=np.float64)
    positions = [np.asarray(position, dtype=np.float64) for position in positions]
    looks = [_unit(center - position) for position in positions]
    view_ups = [
        _camera_up(positions[0], center, initial_up, [0.0, 0.0, 1.0])
    ]

    for index in range(1, len(positions)):
        previous_look = looks[index - 1]
        look = looks[index]
        cross = np.cross(previous_look, look)
        cross_norm = float(np.linalg.norm(cross))
        dot = float(np.clip(np.dot(previous_look, look), -1.0, 1.0))
        if cross_norm <= 1e-10:
            transported = view_ups[-1]
        else:
            axis = cross / cross_norm
            angle = math.atan2(cross_norm, dot)
            transported = _rotate_around_axis(view_ups[-1], axis, angle)
        up = _camera_up(
            positions[index],
            center,
            transported,
            view_ups[-1],
        )
        if float(np.dot(up, view_ups[-1])) < 0.0:
            up = -up
        view_ups.append(up)
    return view_ups


def upper_hemisphere_transition(
    start,
    end,
    center,
    start_up,
    end_up,
    max_step_deg=MAX_ANGULAR_STEP_DEG,
):
    center = np.asarray(center, dtype=np.float64)
    start_rel = np.asarray(start, dtype=np.float64) - center
    end_rel = np.asarray(end, dtype=np.float64) - center
    start_radius = float(np.linalg.norm(start_rel))
    end_radius = float(np.linalg.norm(end_rel))
    radius = 0.5 * (start_radius + end_radius)
    if not np.isclose(start_radius, end_radius, rtol=1e-8, atol=1e-8):
        raise ValueError("Transition endpoints must share one orbit radius")

    start_unit = _unit(start_rel)
    end_unit = _unit(end_rel)
    angle_deg = math.degrees(
        math.acos(float(np.clip(np.dot(start_unit, end_unit), -1.0, 1.0)))
    )
    transition_count = max(0, int(math.ceil(angle_deg / max_step_deg)) - 1)
    positions = []
    for index in range(1, transition_count + 1):
        fraction = index / float(transition_count + 1)
        direction = _slerp_unit(start_unit, end_unit, fraction)
        position = center + radius * direction
        if position[0] < center[0] - 1e-9:
            raise RuntimeError(
                "Upper-hemisphere transition crossed below the virtual ground"
            )
        positions.append(position)

    orientation_positions = [
        np.asarray(start, dtype=np.float64),
        *positions,
        np.asarray(end, dtype=np.float64),
    ]
    orientation_ups, roll_correction_deg = _smooth_transition_view_ups(
        orientation_positions,
        center,
        start_up,
        end_up,
    )
    view_ups = orientation_ups[1:-1]

    return {
        "angle_deg": angle_deg,
        "roll_correction_deg": roll_correction_deg,
        "positions": positions,
        "view_ups": view_ups,
    }


def build_video_sequence(track_system, photos_per_track=PHOTOS_PER_TRACK):
    blocks = track_system.get_building_camera_positions(photos_per_track)
    center = np.asarray(track_system.center, dtype=np.float64)
    sequence = []
    transitions = []

    for plan_index, (block_index, direction) in enumerate(VIDEO_TRACK_PLAN):
        block = blocks[block_index]
        track_index = block_index + 1
        view_up = track_system.get_building_camera_view_up(block_index)
        indexed_positions = list(enumerate(block["positions"], 1))
        if direction < 0:
            indexed_positions.reverse()
        for photo_index, position in indexed_positions:
            sequence.append(
                {
                    "kind": "track",
                    "track_index": track_index,
                    "photo_index": photo_index,
                    "file_path": f"track{track_index}_{photo_index:02d}.png",
                    "position": np.asarray(position, dtype=np.float64),
                    "view_up": np.asarray(view_up, dtype=np.float64),
                }
            )

        if plan_index == len(VIDEO_TRACK_PLAN) - 1:
            continue
        next_block_index, next_direction = VIDEO_TRACK_PLAN[plan_index + 1]
        next_block = blocks[next_block_index]
        next_positions = list(next_block["positions"])
        if next_direction < 0:
            next_positions.reverse()
        next_view_up = track_system.get_building_camera_view_up(next_block_index)
        transition = upper_hemisphere_transition(
            indexed_positions[-1][1],
            next_positions[0],
            center,
            view_up,
            next_view_up,
        )
        boundary_index = plan_index + 1
        transitions.append(
            {
                "boundary": (
                    f"track{track_index}_to_track{next_block_index + 1}"
                ),
                "angle_deg": transition["angle_deg"],
                "frame_count": len(transition["positions"]),
            }
        )
        for transition_index, (position, transition_up) in enumerate(
            zip(transition["positions"], transition["view_ups"]),
            1,
        ):
            sequence.append(
                {
                    "kind": "transition",
                    "boundary_index": boundary_index,
                    "transition_index": transition_index,
                    "file_path": (
                        f"transition{boundary_index}_{transition_index:02d}.png"
                    ),
                    "position": position,
                    "view_up": transition_up,
                }
            )

    continuous_view_ups = _rotation_minimizing_view_ups(
        [frame["position"] for frame in sequence],
        center,
        track_system.get_building_camera_view_up(0),
    )
    for frame, view_up in zip(sequence, continuous_view_ups):
        frame["view_up"] = view_up
    for video_index, frame in enumerate(sequence, 1):
        frame["video_frame_index"] = video_index
    return sequence, transitions


def _save_preview(output_dir, ordered_names):
    frames = []
    for name in ordered_names:
        with Image.open(output_dir / name) as image:
            resized = image.convert("RGB").resize(
                (PREVIEW_SIZE, PREVIEW_SIZE),
                Image.Resampling.LANCZOS,
            )
            frames.append(
                resized.quantize(
                    colors=128,
                    method=Image.Quantize.MEDIANCUT,
                )
            )
    preview_path = output_dir / "video_preview.gif"
    frames[0].save(
        preview_path,
        save_all=True,
        append_images=frames[1:],
        duration=PREVIEW_FRAME_MS,
        loop=0,
        optimize=True,
        disposal=2,
    )
    return preview_path


def capture_video_model(
    source_obj,
    output_dir,
    save_workers=DEFAULT_VIDEO_SAVE_WORKERS,
    gpu_index=0,
    create_preview=True,
    png_compress_level=DEFAULT_PNG_COMPRESS_LEVEL,
):
    from scripts.batch_capture_buildings import (
        _capture_image,
        _make_plotter,
        _render_backend_info,
    )
    from ui.tk_widget import (
        MAX_TRIANGLES,
        _mesh_max_extent,
        _obj_to_pv_coords,
        build_pyvista_mesh,
    )
    from utils.screenshot import ScreenshotManager

    started = time.perf_counter()
    source_obj = Path(source_obj)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    loader = OBJLoader()
    vertices, _, faces = loader.load(str(source_obj))
    if not vertices:
        raise RuntimeError("OBJ contains no usable vertices")
    display_vertices = _obj_to_pv_coords(vertices)
    mesh = build_pyvista_mesh(display_vertices, faces, MAX_TRIANGLES)
    if mesh is None:
        raise RuntimeError("OBJ contains no usable faces")
    center_before = np.asarray(mesh.center, dtype=np.float64)
    mesh.translate(-center_before, inplace=True)
    extent = _mesh_max_extent(mesh)
    radius = max(float(extent) * BATCH_ORBIT_RADIUS_FACTOR, 1.0)

    track_system = TrackSystem()
    track_system.set_center([0.0, 0.0, 0.0])
    track_system.set_camera_distance(radius)
    sequence, transitions = build_video_sequence(track_system)
    underside_violations = [
        frame for frame in sequence if frame["position"][0] < -1e-9
    ]
    if underside_violations:
        raise RuntimeError("Video sequence contains cameras below the ground plane")

    manager = ScreenshotManager(str(output_dir))
    plotter = None
    rendered_frames = []
    pending_saves = set()
    save_workers = max(1, int(save_workers))
    max_pending_saves = max(4, save_workers * 2)
    backend = {}
    actual_device_index = None

    with ThreadPoolExecutor(max_workers=save_workers) as executor:
        try:
            plotter = _make_plotter(mesh)
            render_window = plotter.render_window
            vtk_device_count = int(render_window.GetNumberOfDevices())
            render_window.SetDeviceIndex(int(gpu_index))
            if int(render_window.GetDeviceIndex()) != int(gpu_index):
                raise RuntimeError(
                    f"VTK未接受GPU索引: requested={gpu_index}, "
                    f"actual={render_window.GetDeviceIndex()}"
                )
            actual_device_index = int(render_window.GetDeviceIndex())
            for frame in sequence:
                image, real_position = _capture_image(
                    plotter,
                    frame["position"],
                    frame["view_up"],
                )
                if not backend:
                    backend = _render_backend_info(plotter)
                pending_saves.add(
                    executor.submit(
                        manager.save_array_to_path,
                        image,
                        str(output_dir / frame["file_path"]),
                        False,
                        png_compress_level,
                    )
                )
                if len(pending_saves) >= max_pending_saves:
                    completed, pending_saves = wait(
                        pending_saves,
                        return_when=FIRST_COMPLETED,
                    )
                    for future in completed:
                        future.result()
                rendered_frame = {
                    key: value
                    for key, value in frame.items()
                    if key not in {"position", "view_up"}
                }
                rendered_frame.update(
                    {
                        "target_pos": [
                            float(value) for value in frame["position"]
                        ],
                        "real_pos": [
                            float(value) for value in real_position
                        ],
                        "view_up": [
                            float(value) for value in frame["view_up"]
                        ],
                    }
                )
                rendered_frames.append(rendered_frame)
        finally:
            if plotter is not None:
                plotter.close()
        if pending_saves:
            completed, _ = wait(pending_saves)
            for future in completed:
                future.result()

    missing = [
        frame["file_path"]
        for frame in rendered_frames
        if not (output_dir / frame["file_path"]).is_file()
    ]
    if missing:
        raise RuntimeError(f"Missing rendered frames: {missing}")

    preview_path = None
    if create_preview:
        preview_path = _save_preview(
            output_dir,
            [frame["file_path"] for frame in rendered_frames],
        )
    backend["vtk_requested_device_index"] = int(gpu_index)
    backend["vtk_reported_device_index"] = actual_device_index
    backend["vtk_reported_device_count"] = vtk_device_count
    backend["physical_gpu_binding_verified"] = bool(
        vtk_device_count > 1 and actual_device_index == int(gpu_index)
    )
    metadata = {
        "source_obj": str(source_obj.resolve()),
        "sequence_profile": {
            "profile_version": "building-video-v1-upper-hemisphere-transitions",
            "logical_video_count": 1,
            "base_track_frame_count": PHOTOS_PER_TRACK * 4,
            "transition_frame_count": sum(
                item["frame_count"] for item in transitions
            ),
            "total_video_frame_count": len(rendered_frames),
            "max_angular_step_deg": MAX_ANGULAR_STEP_DEG,
            "base_track_angular_step_deg": BASE_TRACK_STEP_DEG,
            "track_order": [
                block_index + 1
                for block_index, _ in VIDEO_TRACK_PLAN
            ],
            "track_directions": [
                "forward" if direction > 0 else "reverse"
                for _, direction in VIDEO_TRACK_PLAN
            ],
            "camera_orientation_rule": (
                "global_rotation_minimizing_parallel_transport"
            ),
            "upper_hemisphere_rule": "camera_x >= track_center_x",
            "transitions": transitions,
        },
        "validation": {
            "camera_below_ground_count": len(underside_violations),
            "minimum_camera_x": min(
                frame["target_pos"][0] for frame in rendered_frames
            ),
            "all_output_files_present": not missing,
        },
        "normalization_report": loader.get_normalization_report(),
        "render": {
            "model_extent": float(extent),
            "orbit_radius": float(radius),
            "center_before_track_alignment": [
                float(value) for value in center_before
            ],
            "backend": backend,
        },
        "preview": preview_path.name if preview_path is not None else None,
        "elapsed_seconds": time.perf_counter() - started,
        "frames": rendered_frames,
    }
    metadata_path = output_dir / "video_sequence.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def _capture_staged(
    source_obj,
    final_dir,
    save_workers,
    overwrite,
    gpu_index,
    create_preview,
    png_compress_level,
):
    final_dir = Path(final_dir)
    staging = final_dir.parent / f".{final_dir.name}.video-{uuid.uuid4().hex}"
    backup = final_dir.parent / f".{final_dir.name}.backup-{uuid.uuid4().hex}"
    try:
        metadata = capture_video_model(
            source_obj,
            staging,
            save_workers,
            gpu_index=gpu_index,
            create_preview=create_preview,
            png_compress_level=png_compress_level,
        )
        if final_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"{final_dir} already exists; pass --overwrite to replace it"
                )
            os.replace(final_dir, backup)
        try:
            os.replace(staging, final_dir)
        except Exception:
            if backup.exists() and not final_dir.exists():
                os.replace(backup, final_dir)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        return metadata
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _capture_job(
    source_obj,
    final_dir,
    save_workers,
    overwrite,
    gpu_index,
    create_preview,
    png_compress_level,
):
    return _capture_staged(
        Path(source_obj),
        Path(final_dir),
        save_workers,
        overwrite,
        gpu_index,
        create_preview,
        png_compress_level,
    )


def _is_complete_video_output(output_dir):
    sequence_path = output_dir / "video_sequence.json"
    if not sequence_path.is_file():
        return False
    try:
        payload = json.loads(sequence_path.read_text(encoding="utf-8"))
        frames = payload.get("frames")
        if not isinstance(frames, list) or not frames:
            return False
        return all(
            isinstance(frame, dict)
            and isinstance(frame.get("file_path"), str)
            and (output_dir / frame["file_path"]).is_file()
            for frame in frames
        )
    except Exception:
        return False


def _parse_gpu_indices(value):
    indices = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        index = int(part)
        if index < 0:
            raise ValueError("GPU索引不能为负数")
        if index not in indices:
            indices.append(index)
    if not indices:
        raise ValueError("至少指定一个GPU索引")
    return indices


def parse_args():
    parser = argparse.ArgumentParser(
        description="试拍少量建筑模型，并生成带上半球过渡帧的单视频序列。",
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", type=int, default=1, help="源OBJ起始序号，从1开始。")
    parser.add_argument("--limit", type=int, default=3, help="试拍模型数量，默认3。")
    parser.add_argument(
        "--save-workers",
        type=int,
        default=DEFAULT_VIDEO_SAVE_WORKERS,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖同名试验输出；覆盖前会先完整生成并验收临时目录。",
    )
    parser.add_argument(
        "--gpus",
        default="0",
        help="VTK逻辑GPU索引，英文逗号分隔，Windows默认0。",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=2,
        help="每个逻辑GPU池的并行模型进程数，默认2。",
    )
    parser.add_argument(
        "--skip-preview",
        action="store_true",
        help="不生成GIF预览，批量生产时可显著减少CPU开销。",
    )
    parser.add_argument(
        "--png-compress-level",
        type=int,
        default=DEFAULT_PNG_COMPRESS_LEVEL,
        help="PNG压缩等级0-9，默认1；等级越低CPU越快、文件越大。",
    )
    return parser.parse_args()


def main():
    from scripts.batch_capture_buildings import discover_obj_files

    args = parse_args()
    source_root = args.source.resolve()
    output_root = args.output.resolve()
    if not source_root.is_dir():
        raise SystemExit(f"输入目录不存在: {source_root}")
    try:
        gpu_indices = _parse_gpu_indices(args.gpus)
    except ValueError as error:
        raise SystemExit(f"GPU参数错误: {error}") from error
    workers_per_gpu = max(1, int(args.workers_per_gpu))
    save_workers = max(1, int(args.save_workers))
    png_compress_level = int(args.png_compress_level)
    if not 0 <= png_compress_level <= 9:
        raise SystemExit("png-compress-level必须在0到9之间")
    create_preview = not args.skip_preview
    source_files = discover_obj_files(source_root)
    start = max(1, int(args.start))
    limit = max(0, int(args.limit))
    selected = source_files[start - 1 : start - 1 + limit]
    if not selected:
        raise SystemExit("没有选中任何OBJ模型")
    output_root.mkdir(parents=True, exist_ok=True)

    tasks = []
    skipped = 0
    for offset, source_obj in enumerate(selected):
        source_sequence = start + offset
        output_dir = output_root / f"model{source_sequence - 1:05d}"
        if output_dir.exists() and not args.overwrite:
            if _is_complete_video_output(output_dir):
                skipped += 1
                continue
            raise SystemExit(
                f"发现不完整输出目录: {output_dir}\n"
                "请检查后使用 --overwrite 原位安全覆盖。"
            )
        tasks.append((source_sequence, source_obj, output_dir))

    print(
        f"选中 {len(selected)} 个模型，源序号 "
        f"{start}..{start + len(selected) - 1}"
    )
    print(f"已完成并跳过: {skipped}")
    print(f"本次待处理: {len(tasks)}")
    print(f"GPU: {gpu_indices}")
    print(f"每张GPU模型进程: {workers_per_gpu}")
    print(f"每模型PNG保存线程: {save_workers}")
    print(f"PNG压缩等级: {png_compress_level}")
    print(f"GIF预览: {'生成' if create_preview else '跳过'}")
    if not tasks:
        print(f"全部已完成: {output_root}")
        return 0

    executors = {
        gpu_index: ProcessPoolExecutor(max_workers=workers_per_gpu)
        for gpu_index in gpu_indices
    }
    task_iterator = iter(enumerate(tasks, 1))
    active = {}
    completed_count = 0
    failed_count = 0
    started_at = time.perf_counter()

    def submit_next(gpu_index):
        try:
            task_number, task = next(task_iterator)
        except StopIteration:
            return False
        source_sequence, source_obj, output_dir = task
        print(
            f"[提交 {task_number}/{len(tasks)}] GPU {gpu_index}: "
            f"{source_obj.name} -> {output_dir.name}"
        )
        future = executors[gpu_index].submit(
            _capture_job,
            str(source_obj),
            str(output_dir),
            save_workers,
            bool(args.overwrite),
            gpu_index,
            create_preview,
            png_compress_level,
        )
        active[future] = (task_number, task, gpu_index)
        return True

    for gpu_index in gpu_indices:
        for _ in range(workers_per_gpu):
            submit_next(gpu_index)

    try:
        while active:
            finished, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in finished:
                task_number, task, gpu_index = active.pop(future)
                _, source_obj, output_dir = task
                try:
                    metadata = future.result()
                    completed_count += 1
                    profile = metadata["sequence_profile"]
                    backend = metadata["render"]["backend"]
                    print(
                        f"[完成 {task_number}/{len(tasks)}] VTK设备 "
                        f"{backend.get('vtk_reported_device_index')}: "
                        f"{source_obj.name} -> {output_dir.name} | "
                        f"{profile['total_video_frame_count']}帧，"
                        f"底面违规 "
                        f"{metadata['validation']['camera_below_ground_count']}"
                    )
                except Exception as error:
                    failed_count += 1
                    error_dir = output_root / "_video_errors"
                    error_dir.mkdir(parents=True, exist_ok=True)
                    error_path = error_dir / f"{output_dir.name}.txt"
                    error_path.write_text(
                        f"source_obj: {source_obj}\n"
                        f"gpu_index: {gpu_index}\n"
                        f"error: {error}\n",
                        encoding="utf-8",
                    )
                    print(
                        f"[失败 {task_number}/{len(tasks)}] GPU {gpu_index}: "
                        f"{source_obj.name} | {error}"
                    )
                processed = completed_count + failed_count
                elapsed = max(time.perf_counter() - started_at, 1e-6)
                remaining = len(tasks) - processed
                eta_minutes = (
                    elapsed / max(processed, 1) * remaining / 60.0
                )
                print(
                    f"进度: 完成 {completed_count}，失败 {failed_count}，"
                    f"剩余 {remaining}，预计 {eta_minutes:.1f} 分钟"
                )
                submit_next(gpu_index)
    except KeyboardInterrupt:
        print("\n收到停止信号，正在停止提交新任务...")
        for future in active:
            future.cancel()
        return 130
    finally:
        for executor in executors.values():
            executor.shutdown(wait=True)

    print(f"试验输出: {output_root}")
    print(f"本次完成: {completed_count}，失败: {failed_count}")
    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
