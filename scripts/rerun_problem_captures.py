"""Safely recapture problem models into their original modelXXXXX folders."""

import argparse
import json
import os
import re
import shutil
import sys
import uuid
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.batch_capture_buildings import (
    BATCH_CAPTURE_PROFILE_VERSION,
    DEFAULT_MODEL_WORKERS,
    DEFAULT_OUTPUT,
    DEFAULT_SAVE_WORKERS,
    TOTAL_PHOTOS,
    _append_manifest,
    capture_one_model,
    is_complete_output,
)


DEFAULT_REPORT = DEFAULT_OUTPUT / "capture_audit_report.json"
MODEL_FOLDER_PATTERN = re.compile(r"model\d{5}", flags=re.IGNORECASE)


def _safe_report_items(report_path, output_root):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    items = []
    skipped = []
    root = output_root.resolve()

    for model in report.get("problem_models", []):
        source_value = model.get("source_obj")
        if not source_value:
            skipped.append((model, "未登记遗留目录没有对应的源OBJ，不能直接补拍"))
            continue
        source_obj = Path(source_value).resolve()
        output_value = model.get("output_dir")
        if not output_value:
            skipped.append((model, "报告中没有原输出目录"))
            continue
        output_dir = Path(output_value).resolve()
        try:
            output_dir.relative_to(root)
        except ValueError:
            skipped.append((model, "输出目录位于指定根目录之外"))
            continue
        if not MODEL_FOLDER_PATTERN.fullmatch(output_dir.name):
            skipped.append((model, "输出目录名称不是modelXXXXX"))
            continue
        if not source_obj.is_file():
            skipped.append((model, "源OBJ不存在"))
            continue
        items.append(
            {
                "source_sequence": int(model["source_sequence"]),
                "source_name": model.get("source_name") or source_obj.name,
                "source_obj": source_obj,
                "model_folder": output_dir.name,
                "model_folder_index": model.get("model_folder_index"),
                "output_dir": output_dir,
                "issue_codes": sorted(
                    {issue["code"] for issue in model.get("issues", [])}
                ),
            }
        )

    items.sort(
        key=lambda item: (
            item["model_folder_index"]
            if item["model_folder_index"] is not None
            else 10**9,
            item["source_sequence"],
        )
    )
    return report, items, skipped


def _parse_only_filter(value):
    if not value:
        return None
    return {part.strip().casefold() for part in value.split(",") if part.strip()}


def _matches_only(item, only_filter):
    if only_filter is None:
        return True
    candidates = {
        str(item["source_sequence"]).casefold(),
        item["source_name"].casefold(),
        item["source_obj"].stem.casefold(),
        item["model_folder"].casefold(),
    }
    if item["model_folder_index"] is not None:
        candidates.add(str(item["model_folder_index"]))
    return bool(candidates & only_filter)


def _swap_staged_output(staging_dir, output_dir):
    backup_dir = output_dir.parent / (
        f".{output_dir.name}.backup-{uuid.uuid4().hex}"
    )
    had_existing_output = output_dir.exists()
    if had_existing_output:
        os.replace(output_dir, backup_dir)
    try:
        os.replace(staging_dir, output_dir)
    except Exception:
        if had_existing_output and backup_dir.exists() and not output_dir.exists():
            os.replace(backup_dir, output_dir)
        raise
    if backup_dir.exists():
        shutil.rmtree(backup_dir)


def _rerun_one(item, save_workers):
    source_obj = Path(item["source_obj"])
    output_dir = Path(item["output_dir"])
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir.parent / (
        f".{output_dir.name}.rerun-{uuid.uuid4().hex}"
    )

    try:
        metadata = capture_one_model(
            source_obj,
            staging_dir,
            save_workers=save_workers,
        )
        if not is_complete_output(staging_dir):
            raise RuntimeError("临时补拍目录未通过64张完整性检查")
        _swap_staged_output(staging_dir, output_dir)
        return metadata
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description="根据质量检查报告，在原modelXXXXX文件夹中安全覆盖补拍。",
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_MODEL_WORKERS,
        help=f"并行补拍模型数，默认 {DEFAULT_MODEL_WORKERS}。",
    )
    parser.add_argument(
        "--save-workers",
        type=int,
        default=DEFAULT_SAVE_WORKERS,
        help=f"每模型PNG保存线程数，默认 {DEFAULT_SAVE_WORKERS}。",
    )
    parser.add_argument(
        "--only",
        default=None,
        help=(
            "只补拍指定序号/OBJ/model文件夹，多个值用英文逗号分隔；"
            "例如 132,205.obj,model00420。"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只显示将要原位覆盖的模型，不执行补拍。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    report_path = args.report.resolve()
    output_root = args.output.resolve()
    if not report_path.is_file():
        raise SystemExit(f"检查报告不存在: {report_path}")
    if not output_root.is_dir():
        raise SystemExit(f"输出根目录不存在: {output_root}")

    report, items, skipped = _safe_report_items(report_path, output_root)
    only_filter = _parse_only_filter(args.only)
    items = [item for item in items if _matches_only(item, only_filter)]

    print(
        f"报告生成时间: {report.get('generated_at', 'unknown')}\n"
        f"报告问题模型: {len(report.get('problem_models', []))}\n"
        f"本次可原位补拍: {len(items)}"
    )
    for item in items:
        print(
            f"  源序号 {item['source_sequence']}: {item['source_name']} -> "
            f"{item['model_folder']} | {', '.join(item['issue_codes'])}"
        )
    for model, reason in skipped:
        print(
            f"[跳过] 源序号 {model.get('source_sequence')}: "
            f"{model.get('source_name')} | {reason}"
        )

    if args.dry_run or not items:
        return 0

    manifest_path = output_root / "batch_capture_manifest.jsonl"
    workers = max(1, int(args.workers))
    save_workers = max(1, int(args.save_workers))
    succeeded = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        item_iterator = iter(items)
        active = {}

        def submit_next():
            try:
                item = next(item_iterator)
            except StopIteration:
                return False
            future = executor.submit(_rerun_one, item, save_workers)
            active[future] = item
            return True

        for _ in range(min(workers, len(items))):
            submit_next()

        while active:
            completed_futures, _ = wait(
                active,
                return_when=FIRST_COMPLETED,
            )
            for future in completed_futures:
                item = active.pop(future)
                try:
                    metadata = future.result()
                    performance = metadata.get("performance", {})
                    backend = metadata.get("render", {}).get("backend", {})
                    _append_manifest(
                        manifest_path,
                        {
                            "status": "complete",
                            "profile_version": BATCH_CAPTURE_PROFILE_VERSION,
                            "source_obj": str(item["source_obj"]),
                            "output_dir": str(item["output_dir"]),
                            "photo_count": TOTAL_PHOTOS,
                            "rerun_from_audit": str(report_path),
                            "elapsed_seconds": performance.get("total_seconds"),
                            "opengl_renderer": backend.get("opengl_renderer"),
                        },
                    )
                    succeeded += 1
                    print(
                        f"[补拍成功] 源序号 {item['source_sequence']} -> "
                        f"{item['model_folder']}"
                    )
                except Exception as error:
                    failed += 1
                    print(
                        f"[补拍失败] 源序号 {item['source_sequence']} -> "
                        f"{item['model_folder']}: {error}"
                    )
                submit_next()

    print(f"补拍结束: 成功 {succeeded}，失败 {failed}")
    print("原modelXXXXX编号未重新分配。建议重新运行质量检查确认。")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
