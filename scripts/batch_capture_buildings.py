"""Batch-capture OBJ files with the application's 64-view building profile."""

import argparse
import gc
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from datetime import datetime
from pathlib import Path

import numpy as np
import pyvista as pv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from core.obj_loader import OBJLoader
from core.track import TrackSystem
from ui.tk_widget import (
    CAMERA_FAR,
    CAMERA_FOV,
    CAMERA_NEAR,
    CAPTURE_SIZE_PX,
    MAX_TRIANGLES,
    _mesh_max_extent,
    _obj_to_pv_coords,
    build_pyvista_mesh,
)
from utils.model_folder import allocate_next_model_folder
from utils.screenshot import ScreenshotManager


DEFAULT_SOURCE = Path(r"D:\scj\down_load\三七\建筑物模型")
DEFAULT_OUTPUT = Path(r"D:\code\Python_3D_Scanner\src\test")
PHOTOS_PER_TRACK = 16
TOTAL_PHOTOS = PHOTOS_PER_TRACK * 4
BATCH_ORBIT_RADIUS_FACTOR = 3.5
BATCH_CAPTURE_PROFILE_VERSION = "building-v9-horizontal-parallel-ground"
MANIFEST_NAME = "batch_capture_manifest.jsonl"
DEFAULT_MODEL_WORKERS = 2
DEFAULT_SAVE_WORKERS = 4


def _now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _natural_obj_key(path):
    parts = re.split(r"(\d+)", path.stem)
    return [
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in parts
    ]


def discover_obj_files(source_dir):
    return sorted(
        (path for path in source_dir.rglob("*.obj") if path.is_file()),
        key=lambda path: (_natural_obj_key(path), str(path).casefold()),
    )


def _append_manifest(manifest_path, event):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(event)
    payload["timestamp"] = _now()
    with manifest_path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        stream.flush()


def _load_latest_manifest_events(manifest_path):
    latest = {}
    if not manifest_path.exists():
        return latest
    with manifest_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] 清单第 {line_number} 行损坏，已忽略")
                continue
            source = event.get("source_obj")
            if source:
                latest[os.path.normcase(os.path.abspath(source))] = event
    return latest


def _expected_frame_names():
    return [
        f"track{track_index}_{photo_index:02d}.png"
        for track_index in range(1, 5)
        for photo_index in range(1, PHOTOS_PER_TRACK + 1)
    ]


def is_complete_output(output_dir):
    output_dir = Path(output_dir)
    if not (output_dir / "camera_poses.json").is_file():
        return False
    return all((output_dir / name).is_file() for name in _expected_frame_names())


def _safe_manifest_output(output_root, event):
    value = event.get("output_dir")
    if not value:
        return None
    candidate = Path(value).resolve()
    root = output_root.resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _building_profile():
    return {
        "profile_version": BATCH_CAPTURE_PROFILE_VERSION,
        "capture_subject": "building",
        "photo_count": TOTAL_PHOTOS,
        "photos_per_track": PHOTOS_PER_TRACK,
        "model_top_axis": [1.0, 0.0, 0.0],
        "model_pose_rule": "roof_plus_x_points_to_legacy_track_anchor",
        "model_pose_anchor_source": "legacy_four_track_common_anchor_plus_x",
        "ground_rule": "model_bottom_on_virtual_yz_plane",
        "underside_rule": "camera_x_greater_than_or_equal_to_track_center_x",
        "camera_roll_rule": "track1_fixed_model_up_x_tracks234_fixed_world_up_z",
        "tracks": {
            "track1": "horizontal_yz_parallel_ground_360",
            "track2": "original_vertical_upper_half_180",
            "track3": "original_left_diagonal_upper_half_180",
            "track4": "original_right_diagonal_upper_half_180",
        },
    }


def _make_plotter(mesh):
    plotter = pv.Plotter(
        off_screen=True,
        window_size=(CAPTURE_SIZE_PX, CAPTURE_SIZE_PX),
    )
    plotter.set_background("white")
    plotter.add_mesh(
        mesh.copy(deep=True),
        color="#c8cdd2",
        style="surface",
        smooth_shading=False,
        ambient=0.10,
        diffuse=0.90,
        specular=0.10,
        specular_power=50.0,
        reset_camera=False,
    )
    plotter.disable_anti_aliasing()
    plotter.enable_lightkit()
    return plotter


def _capture_image(plotter, cam_pos, view_up):
    pos = np.asarray(cam_pos, dtype=np.float64).reshape(3)
    up = np.asarray(view_up, dtype=np.float64).reshape(3)
    camera = plotter.camera

    camera.position = tuple(float(value) for value in pos)
    camera.focal_point = (0.0, 0.0, 0.0)
    camera.up = tuple(float(value) for value in up)
    camera.view_angle = CAMERA_FOV
    camera.clipping_range = (CAMERA_NEAR, CAMERA_FAR)
    camera.ParallelProjectionOn()
    plotter.render()

    image = plotter.screenshot(
        transparent_background=False,
        return_img=True,
    )
    if image is None:
        raise RuntimeError("PyVista returned no screenshot")
    return image[:, :, :3].astype(np.uint8), tuple(plotter.camera.position)


def _render_backend_info(plotter):
    render_window = plotter.render_window
    capabilities = render_window.ReportCapabilities()

    def capability_value(prefix):
        for line in capabilities.splitlines():
            if line.startswith(prefix):
                return line.split(":", 1)[1].strip()
        return ""

    return {
        "render_window": render_window.GetClassName(),
        "supports_opengl": bool(render_window.SupportsOpenGL()),
        "opengl_vendor": capability_value("OpenGL vendor string"),
        "opengl_renderer": capability_value("OpenGL renderer string"),
        "opengl_version": capability_value("OpenGL version string"),
        "hardware_acceleration": "hardware acceleration:  True" in capabilities,
    }


def capture_one_model(
    source_obj,
    output_dir,
    save_workers=DEFAULT_SAVE_WORKERS,
):
    total_started = time.perf_counter()
    source_obj = Path(source_obj)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    phase_started = time.perf_counter()
    loader = OBJLoader()
    vertices, _, faces = loader.load(str(source_obj))
    if not vertices:
        raise RuntimeError("OBJ contains no usable vertices")
    load_seconds = time.perf_counter() - phase_started

    phase_started = time.perf_counter()
    display_vertices = _obj_to_pv_coords(vertices)
    mesh = build_pyvista_mesh(display_vertices, faces, MAX_TRIANGLES)
    if mesh is None:
        raise RuntimeError("OBJ contains no usable faces")

    # Match normal-mode import: place the model AABB center at the track center.
    center_before_track_alignment = np.asarray(mesh.center, dtype=np.float64)
    track_alignment_translation = -center_before_track_alignment
    mesh.translate(track_alignment_translation, inplace=True)

    extent = _mesh_max_extent(mesh)
    orbit_radius = max(float(extent) * BATCH_ORBIT_RADIUS_FACTOR, 1.0)
    track_system = TrackSystem()
    track_system.set_center([0.0, 0.0, 0.0])
    track_system.set_camera_distance(orbit_radius)
    blocks = track_system.get_building_camera_positions(
        num_positions=PHOTOS_PER_TRACK,
    )
    setup_seconds = time.perf_counter() - phase_started

    manager = ScreenshotManager(str(output_dir))
    plotter = None
    frames = []
    render_seconds = 0.0
    save_wait_seconds = 0.0
    render_backend = {}
    save_workers = max(1, int(save_workers))
    max_pending_saves = max(save_workers * 2, 4)
    pending_saves = set()

    with ThreadPoolExecutor(max_workers=save_workers) as save_executor:
        try:
            phase_started = time.perf_counter()
            plotter = _make_plotter(mesh)
            plotter_setup_seconds = time.perf_counter() - phase_started
            for track_index, block in enumerate(blocks, 1):
                positions = block["positions"]
                view_up = track_system.get_building_camera_view_up(track_index - 1)
                for photo_index, position in enumerate(positions, 1):
                    phase_started = time.perf_counter()
                    image, real_position = _capture_image(
                        plotter,
                        position,
                        view_up,
                    )
                    render_seconds += time.perf_counter() - phase_started
                    if not render_backend:
                        render_backend = _render_backend_info(plotter)

                    image_name = f"track{track_index}_{photo_index:02d}.png"
                    pending_saves.add(
                        save_executor.submit(
                            manager.save_array_to_path,
                            image,
                            str(output_dir / image_name),
                            False,
                        )
                    )
                    if len(pending_saves) >= max_pending_saves:
                        phase_started = time.perf_counter()
                        completed_saves, pending_saves = wait(
                            pending_saves,
                            return_when=FIRST_COMPLETED,
                        )
                        save_wait_seconds += time.perf_counter() - phase_started
                        for future in completed_saves:
                            future.result()

                    frames.append(
                        {
                            "file_path": image_name,
                            "track_index": track_index,
                            "photo_index": photo_index,
                            "target_pos": [float(value) for value in position],
                            "real_pos": [float(value) for value in real_position],
                            "view_up": [float(value) for value in view_up],
                        }
                    )
        finally:
            if plotter is not None:
                plotter.close()

        if pending_saves:
            phase_started = time.perf_counter()
            completed_saves, _ = wait(pending_saves)
            save_wait_seconds += time.perf_counter() - phase_started
            for future in completed_saves:
                future.result()

    total_seconds = time.perf_counter() - total_started

    metadata = {
        "source_obj": str(source_obj.resolve()),
        "capture_profile": _building_profile(),
        "normalization_report": loader.get_normalization_report(),
        "render": {
            "image_size": [CAPTURE_SIZE_PX, CAPTURE_SIZE_PX],
            "fov": CAMERA_FOV,
            "near": CAMERA_NEAR,
            "far": CAMERA_FAR,
            "parallel_projection": True,
            "model_extent": float(extent),
            "orbit_radius": float(orbit_radius),
            "center_before_track_alignment": [
                float(value) for value in center_before_track_alignment
            ],
            "track_alignment_translation": [
                float(value) for value in track_alignment_translation
            ],
            "center_after_track_alignment": [
                float(value) for value in mesh.center
            ],
            "backend": render_backend,
        },
        "performance": {
            "total_seconds": float(total_seconds),
            "load_normalize_seconds": float(load_seconds),
            "mesh_track_setup_seconds": float(setup_seconds),
            "plotter_setup_seconds": float(plotter_setup_seconds),
            "render_seconds": float(render_seconds),
            "save_wait_seconds": float(save_wait_seconds),
            "save_workers": int(save_workers),
        },
        "frames": frames,
    }
    poses_path = output_dir / "camera_poses.json"
    with poses_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)

    if len(frames) != TOTAL_PHOTOS or not is_complete_output(output_dir):
        raise RuntimeError(
            f"Output validation failed: expected {TOTAL_PHOTOS} complete frames"
        )
    renderer_name = render_backend.get("opengl_renderer") or "unknown renderer"
    print(
        f"[Perf] {source_obj.name}: {total_seconds:.2f}s | "
        f"render {render_seconds:.2f}s | save-wait {save_wait_seconds:.2f}s | "
        f"{renderer_name}"
    )
    return metadata


def _capture_worker(source_obj, output_dir, save_workers):
    return capture_one_model(
        Path(source_obj),
        Path(output_dir),
        save_workers=save_workers,
    )


def _write_error(output_dir, source_obj, error):
    output_dir.mkdir(parents=True, exist_ok=True)
    error_path = output_dir / "batch_error.txt"
    with error_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(f"source_obj: {source_obj}\n")
        stream.write(f"time: {_now()}\n")
        stream.write(f"error: {error}\n\n")
        stream.write(traceback.format_exc())


def parse_args():
    parser = argparse.ArgumentParser(
        description="批量按建筑物模式为 OBJ 拍摄64张图片。",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"OBJ根目录，默认：{DEFAULT_SOURCE}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"输出根目录，默认：{DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只处理前N个尚未完成的模型，用于试跑。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_MODEL_WORKERS,
        help=(
            f"同时处理的模型进程数，默认 {DEFAULT_MODEL_WORKERS}；"
            "本机建议使用 2，设为 1 可恢复串行模式。"
        ),
    )
    parser.add_argument(
        "--save-workers",
        type=int,
        default=DEFAULT_SAVE_WORKERS,
        help=(
            f"每个模型用于PNG压缩和写盘的线程数，默认 {DEFAULT_SAVE_WORKERS}。"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只列出任务，不加载模型、不拍摄。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source_root = args.source.resolve()
    output_root = args.output.resolve()
    if not source_root.is_dir():
        raise SystemExit(f"输入目录不存在：{source_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    obj_files = discover_obj_files(source_root)
    manifest_path = output_root / MANIFEST_NAME
    latest_events = _load_latest_manifest_events(manifest_path)

    pending = []
    skipped = 0
    for source_obj in obj_files:
        source_key = os.path.normcase(os.path.abspath(source_obj))
        event = latest_events.get(source_key)
        previous_output = (
            _safe_manifest_output(output_root, event)
            if event is not None
            else None
        )
        if (
            event is not None
            and event.get("status") == "complete"
            and event.get("profile_version") == BATCH_CAPTURE_PROFILE_VERSION
            and previous_output is not None
            and is_complete_output(previous_output)
        ):
            skipped += 1
            continue
        pending.append((source_obj, previous_output))

    if args.limit is not None:
        pending = pending[:max(0, args.limit)]

    print(f"发现 OBJ：{len(obj_files)}")
    print(f"已完成并跳过：{skipped}")
    print(f"本次待处理：{len(pending)}")
    print(f"输出目录：{output_root}")
    if args.dry_run:
        for source_obj, previous_output in pending[:20]:
            destination = previous_output or "<自动分配 modelXXXXX>"
            print(f"  {source_obj.name} -> {destination}")
        if len(pending) > 20:
            print(f"  ... 其余 {len(pending) - 20} 个")
        return 0

    model_workers = max(1, int(args.workers))
    save_workers = max(1, int(args.save_workers))
    print(f"并行模型进程：{model_workers}")
    print(f"每模型PNG保存线程：{save_workers}")

    completed = 0
    failed = 0
    processed = 0
    started_at = datetime.now()

    def start_task(task_index, source_obj, previous_output):
        output_dir = previous_output
        if output_dir is None:
            output_dir = Path(allocate_next_model_folder(str(output_root)))

        print(
            f"\n[{task_index}/{len(pending)}] {source_obj.name}"
            f" -> {output_dir.name}"
        )
        _append_manifest(
            manifest_path,
            {
                "status": "started",
                "profile_version": BATCH_CAPTURE_PROFILE_VERSION,
                "source_obj": str(source_obj.resolve()),
                "output_dir": str(output_dir.resolve()),
                "model_workers": model_workers,
                "save_workers": save_workers,
            },
        )
        return task_index, source_obj, output_dir

    def record_success(task, metadata):
        nonlocal completed, processed
        _, source_obj, output_dir = task
        completed += 1
        processed += 1
        performance = dict(metadata.get("performance", {}))
        backend = dict(metadata.get("render", {}).get("backend", {}))
        _append_manifest(
            manifest_path,
            {
                "status": "complete",
                "profile_version": BATCH_CAPTURE_PROFILE_VERSION,
                "source_obj": str(source_obj.resolve()),
                "output_dir": str(output_dir.resolve()),
                "photo_count": TOTAL_PHOTOS,
                "elapsed_seconds": performance.get("total_seconds"),
                "opengl_renderer": backend.get("opengl_renderer"),
            },
        )
        print(f"[OK] {source_obj.name}：{TOTAL_PHOTOS} 张")

    def record_failure(task, error):
        nonlocal failed, processed
        _, source_obj, output_dir = task
        failed += 1
        processed += 1
        _write_error(output_dir, source_obj, error)
        _append_manifest(
            manifest_path,
            {
                "status": "failed",
                "profile_version": BATCH_CAPTURE_PROFILE_VERSION,
                "source_obj": str(source_obj.resolve()),
                "output_dir": str(output_dir.resolve()),
                "error": str(error),
            },
        )
        print(f"[FAIL] {source_obj.name}：{error}")

    def print_progress():
        elapsed = max((datetime.now() - started_at).total_seconds(), 1e-6)
        average = elapsed / max(processed, 1)
        remaining = average * (len(pending) - processed)
        print(
            f"进度：完成 {completed}，失败 {failed}，"
            f"剩余 {len(pending) - processed}，"
            f"预计剩余 {remaining / 60.0:.1f} 分钟"
        )

    if model_workers == 1:
        for task_index, (source_obj, previous_output) in enumerate(pending, 1):
            task = start_task(task_index, source_obj, previous_output)
            try:
                metadata = capture_one_model(
                    source_obj,
                    task[2],
                    save_workers=save_workers,
                )
                record_success(task, metadata)
            except KeyboardInterrupt:
                _append_manifest(
                    manifest_path,
                    {
                        "status": "interrupted",
                        "profile_version": BATCH_CAPTURE_PROFILE_VERSION,
                        "source_obj": str(source_obj.resolve()),
                        "output_dir": str(task[2].resolve()),
                    },
                )
                print("\n已收到停止信号；当前进度已写入清单，下次运行会续拍。")
                return 130
            except Exception as error:
                record_failure(task, error)
            finally:
                gc.collect()
            print_progress()
    else:
        task_iterator = iter(enumerate(pending, 1))
        active = {}
        executor = ProcessPoolExecutor(max_workers=model_workers)

        def submit_next():
            try:
                task_index, (source_obj, previous_output) = next(task_iterator)
            except StopIteration:
                return False
            task = start_task(task_index, source_obj, previous_output)
            future = executor.submit(
                _capture_worker,
                str(source_obj),
                str(task[2]),
                save_workers,
            )
            active[future] = task
            return True

        for _ in range(min(model_workers, len(pending))):
            submit_next()

        interrupted = False
        try:
            while active:
                completed_futures, _ = wait(
                    active,
                    return_when=FIRST_COMPLETED,
                )
                for future in completed_futures:
                    task = active.pop(future)
                    try:
                        metadata = future.result()
                        record_success(task, metadata)
                    except Exception as error:
                        record_failure(task, error)
                    print_progress()
                    submit_next()
        except KeyboardInterrupt:
            interrupted = True
            for future, task in active.items():
                future.cancel()
                _, source_obj, output_dir = task
                _append_manifest(
                    manifest_path,
                    {
                        "status": "interrupted",
                        "profile_version": BATCH_CAPTURE_PROFILE_VERSION,
                        "source_obj": str(source_obj.resolve()),
                        "output_dir": str(output_dir.resolve()),
                    },
                )
            print(
                "\n已收到停止信号；正在等待已启动的模型进程退出。"
                "下次运行会自动续拍。"
            )
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

        if interrupted:
            return 130

    print(
        f"\n批处理结束：成功 {completed}，失败 {failed}，"
        f"清单：{manifest_path}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
