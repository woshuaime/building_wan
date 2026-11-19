"""Live tmux view of all three stages of the experiment."""
import argparse
import json
import os
from pathlib import Path
import time


TERMINAL = {"complete", "failed", "stopped"}
TOTALS = {"cache": 1024, "train": 1152, "validation": 12}


class Progress:
    def __init__(self, directory):
        self.path = directory / "progress.json"
        self.values = dict(cache=0, train=0, validation=0, step=0, vae=0,
                           pid=os.getpid(), stage="starting", detail="读取原始数据和 V2 权重")
        self.update()

    def update(self, **fields):
        self.values.update(fields, updated_at=time.time())
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.values, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        os.replace(temp, self.path)


def line(label, value, total, width):
    filled = min(width, width * int(value) // total)
    return f"{label}  [{'█' * filled}{'░' * (width - filled)}]   {value:>4}/{total:<4}"


def display(state, width=26):
    stage = state.get("stage", "starting")
    lines = ["V2 屋顶增量实验  |  缓存 · 训练 · 验证",
             line("缓存窗口", state.get("cache", 0), TOTALS["cache"], width),
             line("训练更新", state.get("train", 0), TOTALS["train"], width),
             line("验证成片", state.get("validation", 0), TOTALS["validation"], width)]
    status = state.get("detail") or state.get("reason") or "载入任务"
    lines.append(status)
    if stage == "validation":
        lines.append(line("当前采样", state.get("step", 0), 30, width))
    if stage == "vae":
        lines.append(line("VAE 诊断", state.get("vae", 0), 2, width))
    if stage in ("cache", "train", "validation") and state.get("eta") is not None:
        minutes = max(0, round(state["eta"] / 60, 1))
        lines.append(f"当前阶段剩余约 {minutes:.1f} 分钟 · 最近实测 {state.get('rate', 0):.2f} 秒/项")
    elif stage == "complete":
        lines.append("全部完成")
    elif stage in ("failed", "stopped"):
        lines.append("任务停止 · 查看下方终端")
    return "\n".join(lines)


def watch(directory):
    state = {}
    try:
        print("\033[?25l", end="", flush=True)
        while True:
            try:
                state = json.loads((directory / "progress.json").read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            print("\033[H\033[2J" + display(state, max(8, min(26, os.get_terminal_size().columns - 26))),
                  flush=True)
            if state.get("stage") in TERMINAL:
                return
            time.sleep(1)
    finally:
        print("\033[?25h", end="", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    watch(parser.parse_args().directory)
