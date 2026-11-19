"""Live view of an existing GT V2 validation queue; all state stays in memory."""
import argparse
import hashlib
from collections import deque
from datetime import datetime
import os
from pathlib import Path
import re
import shutil
from statistics import median
import subprocess
import sys
import time
from zoneinfo import ZoneInfo


TOKYO = ZoneInfo("Asia/Tokyo")
CASE = re.compile(r"\[(\d+)/(\d+)\]\s*建筑\s+(\d+)\s+开始采样")
STEP = re.compile(r"building\s+(\d+)\s+sampling\s+(\d+)/(\d+)")
DURATION = re.compile(r"已完成并核验\s+\d+_ID(\d+)\.mp4\s*·\s*([\d.]+)\s*秒")
FINISH = re.compile(r"全部\s+(\d+)\s+条验证视频生成完毕|已完成并核验\s+(\d+)/\d+\s+条视频")


def tmux(*args):
    return subprocess.check_output(["tmux", *args], text=True, stderr=subprocess.DEVNULL)


def queue_id(output, ids):
    return hashlib.sha256("\0".join([str(output.resolve()), *ids]).encode()).hexdigest()[:20]


def snapshot(pane, ids, output):
    # -J rejoins lines that wrap as the user resizes the terminal.
    history = tmux("capture-pane", "-p", "-J", "-S", "-450", "-t", pane)
    info = tmux("display-message", "-p", "-t", pane,
                "#{pane_id}|#{pane_dead}|#{pane_dead_status}|#{pane_dead_signal}").strip()
    if not info.startswith(pane + "|"):
        raise ValueError("原队列窗口已关闭")
    _, dead, code, sig = info.split("|")
    published = {}
    for index, building in enumerate(ids, 1):
        path = output / f"{index:02d}_ID{building}.mp4"
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        if stat.st_size:
            published[building] = stat.st_mtime

    key = queue_id(output, ids)
    start = history.rfind("GT_V2_QUEUE_BEGIN ")
    if start >= 0:
        history = history[start:]
        if not history.startswith("GT_V2_QUEUE_BEGIN " + key):
            history = ""
    queue_done = f"GT_V2_QUEUE_DONE {key}" in history
    queue_failed = f"GT_V2_QUEUE_FAILED {key}" in history
    current, step, steps, finished = None, 0, 30, False
    durations = {}
    for line in history.splitlines():
        start = CASE.search(line)
        progress = STEP.search(line)
        measured = DURATION.search(line)
        end = FINISH.search(line)
        if start and int(start[2]) == len(ids) and start[3] in ids:
            current, step, steps, finished = start[3], 0, 30, False
        if progress and progress[1] in ids:
            current, step, steps = progress[1], int(progress[2]), int(progress[3])
        if measured and measured[1] in published:
            durations[measured[1]] = float(measured[2])
        if end and int(end[1] or end[2]) == len(ids):
            finished = True
    done = len(published) == len(ids)
    result = None
    if dead == "1":
        if code == "0" and done and not queue_failed:
            phase, result = "全部完成", 0
        else:
            detail = f"退出码 {code}" if code else f"信号 {sig or '未知'}"
            phase, result = f"队列已停止，{detail}", 1
    elif queue_failed:
        phase, result = "验证队列已停止，请查看下方输出", 1
    elif queue_done:
        phase, result = ("全部完成", 0) if done else ("队列结束，成片数量异常", 1)
    elif finished and done:
        phase = "全部成片已核验，等待队列退出"
    elif done:
        phase = "正在核验成片"
    elif current is None or current in published:
        phase = "等待下一栋"
    elif step == 0:
        phase = "正在准备模型和输入"
    elif step >= steps:
        phase = "正在解码、编码及核验"
    else:
        phase = "正在采样"
    return current, step, steps, published, durations, phase, result


class StepTiming:
    def __init__(self):
        self.building = None
        self.observed = deque(maxlen=8)
        self.last_step = None

    def update(self, building, step, now):
        if self.building != building or (self.last_step is not None and step < self.last_step):
            self.observed.clear()
            self.last_step = None
            self.building = building
        if building is not None and step > 0 and step != self.last_step:
            self.observed.append((now, step))
            self.last_step = step
        if len(self.observed) >= 3:
            # The first observation may be halfway through an existing step.
            start, first = self.observed[1]
            end, last = self.observed[-1]
            return (end - start) / (last - first)
        return None


def clock(epoch):
    return datetime.fromtimestamp(epoch, TOKYO).strftime("%H:%M:%S")


def bar(value, total, width):
    filled = int(width * value / total)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def duration(seconds):
    seconds = max(1, round(seconds))
    if seconds >= 3600:
        return f"{seconds // 3600}时{seconds % 3600 // 60:02d}分"
    return f"{seconds // 60}分{seconds % 60:02d}秒"


def render(ids, state, timing, now, width):
    current, step, steps, published, durations, phase, result = state
    speed = timing.update(current, step, now)
    done, total = len(published), len(ids)
    active = current is not None and current not in published
    lines = [f"验证视频  ·  东京 {clock(now)}  ·  {phase}",
             f"已出片  {bar(done, total, width)}  {done}/{total} 栋  {done / total:.0%}"]
    if result == 0 and step != steps:
        lines.append("本栋采样  已结束")
    elif current:
        lines.append(f"本栋采样 {bar(step, steps, width)}  {step}/{steps} 步  ·  建筑 {current}")
    elif done == total:
        lines.append(f"本栋采样 {bar(steps, steps, width)}  已有全部成片")
    else:
        lines.append(f"本栋采样 {bar(0, steps, width)}  等待队列")
    timing_text = f"近期采样 {speed:.1f} 秒/步" if speed else ""
    remaining = total - done
    if remaining and result is None:
        recent = list(durations.values())[-3:]
        if recent:
            per_case = median(recent)
            left = remaining * per_case
            if active:
                # Recent full-case times include input loading and final encoding.
                # With an earlier publication in this invocation, the next case
                # starts immediately. This excludes files from old queues.
                if current != ids[0] and ids[ids.index(current) - 1] in durations:
                    since = max(0, now - published[ids[ids.index(current) - 1]])
                    left -= min(per_case - 1, since)
                else:
                    left -= min(per_case - 1, step * (speed or per_case / steps))
            forecast = f"预计剩余 {duration(left)}（东京 {clock(now + left)}）"
        elif active and speed:
            left = (remaining * steps - step) * speed
            forecast = f"剩余采样约 {duration(left)}，另计准备和出片"
        else:
            forecast = "预计剩余时间估算中"
        lines.append("  ·  ".join(s for s in (forecast, timing_text) if s))
    if published:
        latest = max(published, key=published.get)
        lines.append(f"最近成片  建筑 {latest}  ·  东京 {clock(published[latest])}")
    if result is not None:
        lines.append(f"结果  ·  {phase}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pane", required=True, help="original queue tmux pane, e.g. %13")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if len(set(args.ids)) != len(args.ids) or any(not item.isdecimal() for item in args.ids):
        parser.error("请按队列顺序提供不同的数字建筑编号")
    timing = StepTiming()
    try:
        source = tmux("display-message", "-p", "-t", args.pane, "#{pane_id}").strip()
        if not source or source == os.environ.get("TMUX_PANE") and not args.once:
            parser.error("请指定显示采样输出的原队列窗格")
        while True:
            state = snapshot(source, args.ids, args.output)
            width = min(32, max(6, shutil.get_terminal_size().columns - 42))
            body = render(args.ids, state, timing, time.time(), width)
            streaming = sys.stdout.isatty() and not args.once
            print(("\033[H\033[J" if streaming else "") + body, flush=True)
            if args.once or state[-1] is not None:
                return state[-1] or 0
            time.sleep(1)
    except (subprocess.SubprocessError, FileNotFoundError, ValueError) as error:
        print(f"无法读取验证进度，请检查原队列窗口。{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
