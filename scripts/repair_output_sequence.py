"""Realign valid batch outputs to canonical model00000... folder numbers."""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.batch_capture_buildings import (
    BATCH_CAPTURE_PROFILE_VERSION,
    DEFAULT_OUTPUT,
    DEFAULT_SOURCE,
    MANIFEST_NAME,
    TOTAL_PHOTOS,
    _load_latest_manifest_events,
    discover_obj_files,
    is_complete_output,
)


MODEL_PATTERN = re.compile(r"model(\d{5})", flags=re.IGNORECASE)


def _model_index(path):
    match = MODEL_PATTERN.fullmatch(path.name)
    return int(match.group(1)) if match else None


def build_repair_plan(source_root, output_root):
    source_files = discover_obj_files(source_root)
    manifest_path = output_root / MANIFEST_NAME
    latest = _load_latest_manifest_events(manifest_path)
    mappings = []
    current_outputs = set()

    for source_sequence, source_obj in enumerate(source_files, 1):
        key = os.path.normcase(os.path.abspath(source_obj))
        event = latest.get(key)
        if event is None or event.get("status") != "complete":
            raise RuntimeError(
                f"源序号 {source_sequence} 没有complete清单状态: {source_obj}"
            )
        current = Path(event.get("output_dir", "")).resolve()
        try:
            current.relative_to(output_root)
        except ValueError as error:
            raise RuntimeError(f"输出目录越界: {current}") from error
        if not MODEL_PATTERN.fullmatch(current.name):
            raise RuntimeError(f"输出目录名称不是modelXXXXX: {current}")
        if current in current_outputs:
            raise RuntimeError(f"多个源OBJ映射到同一目录: {current}")
        if not is_complete_output(current):
            raise RuntimeError(f"映射输出不完整，拒绝整理: {current}")
        current_outputs.add(current)

        canonical = output_root / f"model{source_sequence - 1:05d}"
        mappings.append(
            {
                "source_sequence": source_sequence,
                "source_obj": source_obj.resolve(),
                "current": current,
                "canonical": canonical.resolve(),
                "current_index": _model_index(current),
                "canonical_index": source_sequence - 1,
            }
        )

    all_model_dirs = {
        path.resolve()
        for path in output_root.iterdir()
        if path.is_dir() and MODEL_PATTERN.fullmatch(path.name)
    }
    orphan_dirs = sorted(
        all_model_dirs - current_outputs,
        key=lambda path: _model_index(path),
    )
    moves = [
        item for item in mappings if item["current"] != item["canonical"]
    ]
    deltas = {
        item["current_index"] - item["canonical_index"]
        for item in moves
    }
    if len(deltas) > 1 or (deltas and next(iter(deltas)) <= 0):
        raise RuntimeError(
            f"检测到复杂或反向编号偏移 {sorted(deltas)}，拒绝自动整理"
        )

    occupied_orphan_targets = sorted(
        {
            item["canonical"]
            for item in moves
            if item["canonical"] in orphan_dirs
        },
        key=lambda path: _model_index(path),
    )
    unexpected_orphans = sorted(
        set(orphan_dirs) - set(occupied_orphan_targets),
        key=lambda path: _model_index(path),
    )
    if unexpected_orphans:
        raise RuntimeError(
            "发现不在编号冲突位置的孤立目录，拒绝自动整理: "
            + ", ".join(path.name for path in unexpected_orphans)
        )

    return {
        "source_files": source_files,
        "manifest_path": manifest_path,
        "mappings": mappings,
        "moves": sorted(moves, key=lambda item: item["canonical_index"]),
        "orphan_dirs": orphan_dirs,
        "occupied_orphan_targets": occupied_orphan_targets,
        "deltas": sorted(deltas),
    }


def _write_journal(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def execute_plan(plan, output_root):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    quarantine = output_root / f"_sequence_repair_orphans_{timestamp}"
    journal_path = output_root / "sequence_repair_journal.json"
    quarantine.mkdir(parents=False, exist_ok=False)

    journal = {
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "running",
        "quarantine": str(quarantine),
        "orphan_moves": [],
        "output_moves": [],
    }
    _write_journal(journal_path, journal)
    moved_outputs = []
    moved_orphans = []

    try:
        for orphan in plan["orphan_dirs"]:
            destination = quarantine / orphan.name
            os.replace(orphan, destination)
            moved_orphans.append((orphan, destination))
            journal["orphan_moves"].append(
                {"from": str(orphan), "to": str(destination)}
            )
            _write_journal(journal_path, journal)

        for item in plan["moves"]:
            current = item["current"]
            canonical = item["canonical"]
            if not current.is_dir():
                raise RuntimeError(f"待移动源目录不存在: {current}")
            if canonical.exists():
                raise RuntimeError(f"目标目录仍被占用: {canonical}")
            os.replace(current, canonical)
            moved_outputs.append((current, canonical))
            journal["output_moves"].append(
                {
                    "source_sequence": item["source_sequence"],
                    "source_obj": str(item["source_obj"]),
                    "from": str(current),
                    "to": str(canonical),
                }
            )
            _write_journal(journal_path, journal)

        for item in plan["mappings"]:
            if not is_complete_output(item["canonical"]):
                raise RuntimeError(
                    f"整理后输出不完整: {item['canonical']}"
                )

        repair_time = datetime.now().astimezone().isoformat(timespec="seconds")
        with plan["manifest_path"].open(
            "a", encoding="utf-8", newline="\n"
        ) as stream:
            for item in plan["moves"]:
                event = {
                    "status": "complete",
                    "profile_version": BATCH_CAPTURE_PROFILE_VERSION,
                    "source_obj": str(item["source_obj"]),
                    "output_dir": str(item["canonical"]),
                    "photo_count": TOTAL_PHOTOS,
                    "sequence_repaired": True,
                    "previous_output_dir": str(item["current"]),
                    "timestamp": repair_time,
                }
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            stream.flush()

        journal["status"] = "complete"
        journal["completed_at"] = repair_time
        _write_journal(journal_path, journal)
        return quarantine, journal_path
    except Exception:
        for current, canonical in reversed(moved_outputs):
            if canonical.exists() and not current.exists():
                os.replace(canonical, current)
        for orphan, destination in reversed(moved_orphans):
            if destination.exists() and not orphan.exists():
                os.replace(destination, orphan)
        journal["status"] = "rolled_back"
        journal["rolled_back_at"] = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )
        _write_journal(journal_path, journal)
        raise


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "把清单中的有效输出重新对齐为：第1个OBJ→model00000，"
            "第N个OBJ→model(N-1)。孤立目录会移入隔离文件夹，不删除。"
        ),
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="真正执行整理；不提供时只显示计划。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source_root = args.source.resolve()
    output_root = args.output.resolve()
    plan = build_repair_plan(source_root, output_root)

    print(f"源OBJ总数: {len(plan['source_files'])}")
    print(f"有效清单映射: {len(plan['mappings'])}")
    print(f"需要移动的有效目录: {len(plan['moves'])}")
    print(f"孤立目录: {len(plan['orphan_dirs'])}")
    print(f"编号偏移: {plan['deltas'] or [0]}")
    if plan["moves"]:
        first = plan["moves"][0]
        last = plan["moves"][-1]
        print(
            f"移动范围: {first['current'].name} -> {first['canonical'].name}"
            f" ... {last['current'].name} -> {last['canonical'].name}"
        )
    if plan["orphan_dirs"]:
        print(
            "隔离范围: "
            f"{plan['orphan_dirs'][0].name} ... {plan['orphan_dirs'][-1].name}"
        )

    if not args.execute:
        print("当前为预览模式，未移动任何文件。确认后添加 --execute。")
        return 0

    quarantine, journal_path = execute_plan(plan, output_root)
    print("编号整理完成。")
    print(f"孤立目录已保留在: {quarantine}")
    print(f"操作日志: {journal_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
