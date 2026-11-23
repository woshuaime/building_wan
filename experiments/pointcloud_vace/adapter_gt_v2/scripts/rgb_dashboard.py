"""Responsive terminal display for a V2 RGB run. Reads the live JSON snapshots."""
import datetime
import json
from pathlib import Path
import re
import shutil
import sys
import threading
import unicodedata

from pipeline import read_json
from rgb_recipe import TOTAL

SGR = re.compile(r"\x1b\[[0-9;]*m")
CYAN, GREEN, PURPLE, GOLD = 81, 84, 147, 221
WHITE, SECONDARY, TRACK = 255, 110, 241
RAIL_COLORS = {CYAN: 24, GREEN: 29, PURPLE: 60, GOLD: 94}
LABELS = {"prepared": "配置就绪", "waiting_for_gpu": "等待 GPU",
          "training": "正式训练", "validation": "八栋验证", "reporting": "生成报告",
          "paused": "已暂停", "failed": "运行异常", "complete": "已完成"}
ACTIONS = {"loading_model": "加载模型", "caching": "编码窗口", "training": "更新参数",
           "sampling": "采样", "encoding": "编码视频", "verifying": "核验成片",
           "loading": "准备模型", "verified": "成片已核验", "scoring": "逐栋检查"}


def cell_width(text):
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1
               for c in SGR.sub("", text) if not unicodedata.combining(c))


def fit(text, width):
    """Fit an unstyled label in terminal cells, including wide Chinese glyphs."""
    if cell_width(text) <= width:
        return text
    chars, used = [], 0
    for c in text:
        size = cell_width(c)
        if used + size > width - 1:
            break
        chars.append(c)
        used += size
    return "".join(chars) + ("…" if width else "")


def read_progress(run, previous=None):
    """Carry forward the last readable snapshot per file while its writer publishes."""
    previous = previous or {}
    current = {}
    unavailable = set()
    for name, filename in (("queue", "progress.json"), ("train", "train_progress.json"),
                           ("validation", "validation_progress.json")):
        try:
            value = read_json(run / filename)
        except (json.JSONDecodeError, UnicodeDecodeError):
            value = None
        if value is None:
            unavailable.add(name)
            value = previous.get(name, {})
        current[name] = value
    stage = current["queue"].get("stage")
    current["waiting"] = (
        "queue" in unavailable or (stage == "training" and "train" in unavailable)
        or (stage in ("validation", "reporting") and "validation" in unavailable)
    )
    return current


def panel(run, columns=None, color=None, *, snapshot=None):
    width = max(1, columns if columns is not None else shutil.get_terminal_size((80, 24)).columns)
    colored = sys.stdout.isatty() if color is None else color
    margin = "  " if width >= 60 else ""
    usable = max(1, width - 2 * len(margin) - 1)  # Leave one cell for the terminal's wrap flag.
    wide = usable >= 65
    if snapshot is None:
        snapshot = read_progress(run)
    queue, train, val = snapshot["queue"], snapshot["train"], snapshot["validation"]
    stage = queue.get("stage", "prepared")
    label = LABELS.get(stage, stage)
    status_color = {"failed": 203, "paused": GOLD, "waiting_for_gpu": GOLD,
                    "validation": PURPLE, "reporting": PURPLE}.get(stage, GREEN)

    def ink(text, tint, bold=False):
        if colored:
            return f"\033[{('1;' if bold else '')}38;5;{tint}m{text}\033[0m"
        return text

    def pair(left, right, left_color=WHITE, right_color=SECONDARY):
        right = fit(right, usable)
        space = usable - cell_width(right)
        left = fit(left, max(0, space - 2))
        return ink(left, left_color, True) + " " * (space - cell_width(left)) + ink(right, right_color, True)

    def meter(label, value, total, tint):
        total = max(1, int(total))
        value = min(total, max(0, int(value)))
        percent = f"{100 * value / total:.2f}%"
        ratio = f"{value:,}/{total:,}"
        suffix = ink(percent.rjust(7), tint, True) + " " + ink(ratio.rjust(14 if wide else len(ratio)), WHITE)
        prefix = ink(label, tint, True)
        length = usable - cell_width(prefix) - cell_width(suffix) - 3
        if length < 1:
            return pair(label, ratio, tint)
        units = value * length * 8 // total
        full, part = divmod(units, 8)
        if value and full == 0 and part == 0:
            part = 1
        head = "▏▎▍▌▋▊▉"[part-1] if part else ""
        rail = ink("█" * full + head, tint) + ink("━" * (length - full - bool(part)), RAIL_COLORS[tint])
        return prefix + "  " + rail + " " + suffix

    epoch = train.get("epoch", 1)
    phase = f"第 {epoch} / 3 轮   ● {label}" if stage == "training" and wide else f"● {label}"
    if snapshot["waiting"]:
        phase = "等待进度刷新 · " + phase
    lines = [pair("V2  建筑点云训练", phase, WHITE, status_color),
             ink("─" * usable, TRACK),
             meter("窗口编码", train.get("cached", train.get("cache", 0)), TOTAL, CYAN),
             meter("训练更新", train.get("updates", 0), TOTAL, GREEN),
             meter("验证成片", val.get("completed", 0), 8, PURPLE)]

    if stage == "training":
        action = ACTIONS.get(train.get("stage"), "准备训练")
        building = train.get("building")
        current = f"建筑 {building} · {action}" if building else action
        detail = f"RGB 辅助 {train.get('aux_updates',0):,} 次 · 已保存 {train.get('durable_updates',0):,} 步"
        lines.append(pair(current, detail, SECONDARY, GOLD))
        if train.get("last_fm") is not None:
            metrics = f"第 {train['updates']:,} 步 · FM 损失 {train['last_fm']:.5f}"
            if train.get("last_sigma") is not None:
                metrics += f" · σ {train['last_sigma']:.3f}"
            peak = train.get("peak_vram_bytes", 0) / (1024 ** 3)
            lines.append(pair(metrics, f"峰值显存 {peak:.2f} GiB"))
        rate = train.get("seconds_per_update")
        if rate:
            eta = str(datetime.timedelta(seconds=int(rate * max(0, TOTAL - train.get("updates",0)))))
            lines.append(pair(f"近期 {rate:.2f} 秒/步", f"训练预计剩余 {eta}"))
    elif stage in ("validation", "reporting"):
        if stage == "reporting":
            lines.append(meter("逐栋检查", val.get("scored",0), 8, GOLD))
        else:
            lines.append(meter("当前采样", val.get("step",0), val.get("steps",30), GOLD))
        description = ACTIONS.get(val.get("stage"), "准备中")
        eta = val.get("eta_seconds")
        rate = val.get("seconds_per_step")
        timing = " · ".join(filter(None, [f"{rate:.2f} 秒/步" if rate else "",
                            f"预计剩余 {datetime.timedelta(seconds=int(eta))}" if eta is not None else ""]))
        lines.append(pair(f"建筑 {val.get('building') or '…'} · {description}", timing))
    elif stage == "waiting_for_gpu" and queue.get("reason"):
        lines.append(ink(fit(queue["reason"], usable), GOLD))
    elif stage == "failed":
        lines.append(ink(fit(str(queue.get("reason","")), usable), 203))
    return "\n".join(margin + row for row in lines)


def dashboard(run, finished=None):
    standalone = finished is None
    finished = finished or threading.Event()
    tty = sys.stdout.isatty()
    last = None
    snapshot = {}
    if tty:
        print("\033[?25l", end="", flush=True)
    try:
        while True:
            snapshot = read_progress(run, snapshot)
            frame = panel(run, snapshot=snapshot)
            if frame != last:
                if tty:
                    # Update in place. A trailing newline would scroll a short pane.
                    print("\033[H" + "\033[K\r\n".join(frame.splitlines()) + "\033[J", end="", flush=True)
                else:
                    print(frame, flush=True)
                last = frame
            if finished.is_set():
                return
            if standalone and snapshot["queue"].get("stage") in ("complete","paused","failed"):
                return
            finished.wait(.8)
    finally:
        if tty:
            print("\033[0m\033[?25h", end="", flush=True)


if __name__ == "__main__":
    dashboard(Path(sys.argv[1]).resolve())
