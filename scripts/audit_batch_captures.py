"""Audit every batch-capture output and produce JSON/CSV problem reports."""

import argparse
import csv
import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from PIL import Image


DEFAULT_SOURCE = Path(r"D:\scj\down_load\三七\建筑物模型")
DEFAULT_OUTPUT = Path(r"D:\code\Python_3D_Scanner\src\test")
MANIFEST_NAME = "batch_capture_manifest.jsonl"
EXPECTED_PROFILE_VERSION = "building-v9-horizontal-parallel-ground"
EXPECTED_SIZE = (1024, 1024)
EXPECTED_FRAME_NAMES = [
    f"track{track_index}_{photo_index:02d}.png"
    for track_index in range(1, 5)
    for photo_index in range(1, 17)
]
MODEL_FOLDER_PATTERN = re.compile(r"model\d{5}", flags=re.IGNORECASE)


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


def load_latest_manifest_events(manifest_path):
    latest = {}
    malformed_lines = []
    if not manifest_path.is_file():
        return latest, malformed_lines

    with manifest_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                malformed_lines.append(
                    {"line": line_number, "error": str(error)}
                )
                continue
            if not isinstance(event, dict):
                malformed_lines.append(
                    {
                        "line": line_number,
                        "error": "清单行不是JSON对象",
                    }
                )
                continue
            source_obj = event.get("source_obj")
            if source_obj:
                key = os.path.normcase(os.path.abspath(source_obj))
                latest[key] = event
    return latest, malformed_lines


def _safe_output_dir(output_root, event):
    value = event.get("output_dir") if event else None
    if not value:
        return None
    candidate = Path(value).resolve()
    try:
        candidate.relative_to(output_root.resolve())
    except ValueError:
        return None
    return candidate


def _model_folder_index(output_dir):
    if output_dir is None:
        return None
    match = re.fullmatch(r"model(\d{5})", output_dir.name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _issue(code, detail, file_name=None):
    item = {"code": code, "detail": detail}
    if file_name:
        item["file"] = file_name
    return item


def inspect_png(path):
    if not path.is_file():
        return [_issue("missing_png", "图片文件不存在", path.name)]
    if path.stat().st_size <= 0:
        return [_issue("zero_byte_png", "图片文件为0字节", path.name)]

    try:
        with Image.open(path) as image:
            image.load()
            size = tuple(image.size)
            mode = image.mode
            gray = image.convert("L").resize((64, 64))
            histogram = gray.histogram()
    except Exception as error:
        return [
            _issue(
                "corrupt_png",
                f"无法完整解码PNG: {error}",
                path.name,
            )
        ]

    issues = []
    if size != EXPECTED_SIZE:
        issues.append(
            _issue(
                "wrong_image_size",
                f"图片尺寸为 {size[0]}x{size[1]}，预期1024x1024",
                path.name,
            )
        )
    if mode not in {"RGB", "RGBA"}:
        issues.append(
            _issue(
                "wrong_image_mode",
                f"图片模式为 {mode}，预期RGB/RGBA",
                path.name,
            )
        )

    pixel_count = float(sum(histogram))
    mean_luma = sum(value * count for value, count in enumerate(histogram))
    mean_luma /= max(pixel_count, 1.0)
    dark_ratio = sum(histogram[:6]) / max(pixel_count, 1.0)
    white_ratio = sum(histogram[250:]) / max(pixel_count, 1.0)
    variance = sum(
        ((value - mean_luma) ** 2) * count
        for value, count in enumerate(histogram)
    ) / max(pixel_count, 1.0)
    std_luma = variance ** 0.5

    if mean_luma < 8.0 or dark_ratio > 0.985:
        issues.append(
            _issue(
                "near_black_png",
                (
                    f"图片近乎全黑: mean={mean_luma:.2f}, "
                    f"dark_ratio={dark_ratio:.4f}"
                ),
                path.name,
            )
        )
    # Edge-on views of very thin models can legitimately occupy well below 1%
    # of the frame. Only classify an image as blank when it is effectively
    # pure white, leaving those valid line/point silhouettes untouched.
    if white_ratio > 0.9999 or (mean_luma > 254.98 and std_luma < 0.25):
        issues.append(
            _issue(
                "near_blank_white_png",
                (
                    f"图片近乎全白/主体疑似缺失: mean={mean_luma:.2f}, "
                    f"white_ratio={white_ratio:.4f}"
                ),
                path.name,
            )
        )
    return issues


def audit_model(source_sequence, source_obj, event, output_root):
    output_dir = _safe_output_dir(output_root, event)
    entry = {
        "source_sequence": int(source_sequence),
        "source_name": source_obj.name,
        "source_obj": str(source_obj.resolve()),
        "manifest_status": event.get("status") if event else "missing",
        "output_dir": str(output_dir) if output_dir else None,
        "model_folder": output_dir.name if output_dir else None,
        "model_folder_index": _model_folder_index(output_dir),
        "issues": [],
    }
    issues = entry["issues"]

    if event is None:
        issues.append(_issue("missing_manifest_event", "清单中没有该OBJ记录"))
        return entry
    if event.get("status") != "complete":
        issues.append(
            _issue(
                "manifest_not_complete",
                f"清单最终状态为 {event.get('status')!r}",
            )
        )
    if output_dir is None:
        issues.append(
            _issue(
                "invalid_output_mapping",
                "清单输出目录缺失或位于指定输出根目录之外",
            )
        )
        return entry
    if not output_dir.is_dir():
        issues.append(_issue("missing_output_dir", "模型输出文件夹不存在"))
        return entry

    poses_path = output_dir / "camera_poses.json"
    metadata = None
    if not poses_path.is_file():
        issues.append(
            _issue("missing_camera_poses", "缺少camera_poses.json")
        )
    else:
        try:
            metadata = json.loads(poses_path.read_text(encoding="utf-8"))
        except Exception as error:
            issues.append(
                _issue(
                    "invalid_camera_poses",
                    f"camera_poses.json无法解析: {error}",
                )
            )

    if metadata is not None and not isinstance(metadata, dict):
        issues.append(
            _issue(
                "invalid_camera_poses",
                "camera_poses.json根节点不是JSON对象",
            )
        )
        metadata = None

    if metadata is not None:
        frames = metadata.get("frames")
        if not isinstance(frames, list) or len(frames) != 64:
            actual = len(frames) if isinstance(frames, list) else "invalid"
            issues.append(
                _issue(
                    "wrong_metadata_frame_count",
                    f"元数据帧数为 {actual}，预期64",
                )
            )
        else:
            valid_frame_objects = all(
                isinstance(frame, dict) for frame in frames
            )
            frame_names = (
                [frame.get("file_path") for frame in frames]
                if valid_frame_objects
                else []
            )
            if (
                not valid_frame_objects
                or not all(isinstance(name, str) for name in frame_names)
                or sorted(frame_names) != sorted(EXPECTED_FRAME_NAMES)
            ):
                issues.append(
                    _issue(
                        "wrong_metadata_frame_names",
                        "元数据中的64个图片文件名不完整或顺序映射异常",
                    )
                )
        profile = metadata.get("capture_profile")
        profile_version = (
            profile.get("profile_version")
            if isinstance(profile, dict)
            else None
        )
        if profile_version != EXPECTED_PROFILE_VERSION:
            issues.append(
                _issue(
                    "wrong_profile_version",
                    (
                        f"拍摄版本为 {profile_version!r}，"
                        f"预期 {EXPECTED_PROFILE_VERSION!r}"
                    ),
                )
            )

    for frame_name in EXPECTED_FRAME_NAMES:
        issues.extend(inspect_png(output_dir / frame_name))
    return entry


def audit_orphan_folder(output_dir):
    entry = {
        "source_sequence": None,
        "source_name": None,
        "source_obj": None,
        "manifest_status": "unmapped",
        "output_dir": str(output_dir),
        "model_folder": output_dir.name,
        "model_folder_index": _model_folder_index(output_dir),
        "issues": [
            _issue(
                "orphan_output_folder",
                "该modelXXXXX文件夹没有被任何源OBJ的最新清单记录引用",
            )
        ],
    }
    poses_path = output_dir / "camera_poses.json"
    if not poses_path.is_file():
        entry["issues"].append(
            _issue("missing_camera_poses", "缺少camera_poses.json")
        )
    else:
        try:
            metadata = json.loads(poses_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError("根节点不是JSON对象")
        except Exception as error:
            entry["issues"].append(
                _issue(
                    "invalid_camera_poses",
                    f"camera_poses.json无法解析: {error}",
                )
            )
    for frame_name in EXPECTED_FRAME_NAMES:
        entry["issues"].extend(inspect_png(output_dir / frame_name))
    return entry


def _atomic_write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_write_csv(path, problem_models):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        fieldnames = [
            "source_sequence",
            "source_name",
            "source_obj",
            "model_folder",
            "model_folder_index",
            "output_dir",
            "manifest_status",
            "issue_count",
            "issue_codes",
            "issue_files",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for model in problem_models:
            issues = model["issues"]
            writer.writerow(
                {
                    "source_sequence": model["source_sequence"],
                    "source_name": model["source_name"],
                    "source_obj": model["source_obj"],
                    "model_folder": model["model_folder"],
                    "model_folder_index": model["model_folder_index"],
                    "output_dir": model["output_dir"],
                    "manifest_status": model["manifest_status"],
                    "issue_count": len(issues),
                    "issue_codes": ";".join(
                        sorted({item["code"] for item in issues})
                    ),
                    "issue_files": ";".join(
                        sorted(
                            {
                                item["file"]
                                for item in issues
                                if item.get("file")
                            }
                        )
                    ),
                }
            )
    os.replace(temporary, path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="检查建筑物批量拍摄结果并生成问题模型序号报告。",
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="JSON报告路径，默认保存在输出根目录。",
    )
    parser.add_argument(
        "--report-csv",
        type=Path,
        default=None,
        help="CSV报告路径，默认保存在输出根目录。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅检查自然排序后的前N个OBJ，用于试跑。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="并行检查文件夹数量，默认8。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source_root = args.source.resolve()
    output_root = args.output.resolve()
    if not source_root.is_dir():
        raise SystemExit(f"输入目录不存在: {source_root}")
    if not output_root.is_dir():
        raise SystemExit(f"输出目录不存在: {output_root}")

    report_json = (
        args.report_json.resolve()
        if args.report_json
        else output_root / "capture_audit_report.json"
    )
    report_csv = (
        args.report_csv.resolve()
        if args.report_csv
        else output_root / "capture_audit_report.csv"
    )
    obj_files = discover_obj_files(source_root)
    if args.limit is not None:
        obj_files = obj_files[:max(0, args.limit)]

    manifest_path = output_root / MANIFEST_NAME
    latest_events, malformed_lines = load_latest_manifest_events(manifest_path)
    audited_models = []
    total = len(obj_files)
    futures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for source_sequence, source_obj in enumerate(obj_files, 1):
            key = os.path.normcase(os.path.abspath(source_obj))
            futures.append(
                executor.submit(
                    audit_model,
                    source_sequence,
                    source_obj,
                    latest_events.get(key),
                    output_root,
                )
            )
        for completed, future in enumerate(as_completed(futures), 1):
            audited_models.append(future.result())
            if completed % 25 == 0 or completed == total:
                print(f"检查进度: {completed}/{total}")
    audited_models.sort(key=lambda model: model["source_sequence"])

    mapped_output_dirs = {
        Path(model["output_dir"]).resolve()
        for model in audited_models
        if model["output_dir"]
    }
    orphan_models = []
    if args.limit is None:
        all_model_dirs = {
            path.resolve()
            for path in output_root.iterdir()
            if path.is_dir() and MODEL_FOLDER_PATTERN.fullmatch(path.name)
        }
        orphan_dirs = sorted(
            all_model_dirs - mapped_output_dirs,
            key=lambda path: _model_folder_index(path),
        )
        if orphan_dirs:
            print(f"发现未登记的model文件夹: {len(orphan_dirs)}，正在检查")
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            orphan_models = list(
                executor.map(audit_orphan_folder, orphan_dirs)
            )

    problem_models = [
        model for model in audited_models if model["issues"]
    ] + orphan_models
    source_problem_count = sum(
        bool(model["issues"]) for model in audited_models
    )
    issue_counts = Counter(
        issue["code"]
        for model in problem_models
        for issue in model["issues"]
    )
    final_status_count = sum(
        1
        for source_obj in obj_files
        if (
            latest_events.get(
                os.path.normcase(os.path.abspath(source_obj)), {}
            ).get("status")
            in {"complete", "failed"}
        )
    )
    payload = {
        "generated_at": _now(),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "manifest_path": str(manifest_path),
        "expected_profile_version": EXPECTED_PROFILE_VERSION,
        "summary": {
            "source_model_count": total,
            "final_manifest_status_count": final_status_count,
            "qualified_model_count": total - source_problem_count,
            "problem_model_count": len(problem_models),
            "orphan_output_folder_count": len(orphan_models),
            "issue_counts": dict(sorted(issue_counts.items())),
            "malformed_manifest_line_count": len(malformed_lines),
        },
        "malformed_manifest_lines": malformed_lines,
        "problem_models": problem_models,
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(report_json, payload)
    _atomic_write_csv(report_csv, problem_models)

    print(
        f"检查完成: 源模型 {total}，合格 {total - source_problem_count}，"
        f"问题记录 {len(problem_models)}，孤立文件夹 {len(orphan_models)}"
    )
    for model in problem_models:
        codes = sorted({item["code"] for item in model["issues"]})
        print(
            f"  序号 {model['source_sequence']}: {model['source_name']} -> "
            f"{model['model_folder'] or '无输出文件夹'} | {', '.join(codes)}"
        )
    print(f"JSON报告: {report_json}")
    print(f"CSV报告: {report_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
