"""Small Tk dashboard for the two-worker single-orbit MP4 batch."""

import argparse
import json
import os
import re
import subprocess
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText


def process_running(pid):
    if not pid:
        return False
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def read_tail(path, limit=10000):
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - limit))
            return stream.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


class ProgressWindow:
    def __init__(self, args):
        self.input_root = args.input.resolve()
        self.output_root = args.output.resolve()
        self.log_root = args.log_dir.resolve()
        self.state_path = self.log_root / "batch_state.json"
        self.total = args.total
        self.started_at = time.time()

        self.root = tk.Tk()
        self.root.title("单轨照片转视频 — 批处理进度")
        self.root.geometry("850x560")
        self.root.minsize(760, 490)
        self.root.configure(padx=18, pady=16)

        ttk.Label(self.root, text="单轨照片转 MP4", font=("Microsoft YaHei UI", 17, "bold")).pack(anchor="w")
        ttk.Label(
            self.root,
            text=f"输入：{self.input_root}\n输出：{self.output_root}",
            justify="left",
        ).pack(anchor="w", pady=(5, 10))

        self.summary = ttk.Label(self.root, text="读取批次状态…", font=("Microsoft YaHei UI", 11, "bold"))
        self.summary.pack(anchor="w", pady=(0, 6))
        self.progress = ttk.Progressbar(self.root, maximum=max(1, self.total), mode="determinate")
        self.progress.pack(fill="x", pady=(0, 9))

        self.details = ttk.Label(self.root, text="", justify="left")
        self.details.pack(anchor="w", pady=(0, 9))

        ttk.Label(self.root, text="双工作进程（分段并行）；mp4v 编码由 CPU 执行，显卡不参与此步骤。", foreground="#555").pack(anchor="w", pady=(0, 9))
        ttk.Label(self.root, text="工作进程状态", font=("Microsoft YaHei UI", 10, "bold")).pack(anchor="w")
        self.workers = ttk.Label(self.root, text="", justify="left")
        self.workers.pack(anchor="w", pady=(3, 9))

        ttk.Label(self.root, text="最近日志", font=("Microsoft YaHei UI", 10, "bold")).pack(anchor="w")
        self.log_view = ScrolledText(self.root, height=13, wrap="none", font=("Consolas", 9), state="disabled")
        self.log_view.pack(fill="both", expand=True, pady=(4, 5))
        ttk.Label(self.root, text="关闭此窗口不会停止转换；视频会逐个校验后写入目标目录。", foreground="#555").pack(anchor="w")

        self.root.after(400, self.refresh)

    def refresh(self):
        try:
            self._update()
        except Exception as error:  # keep the progress window usable even if a log is temporarily locked
            self.summary.configure(text=f"监控刷新中（{error}）")
        self.root.after(2000, self.refresh)

    def _update(self):
        state = {}
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        self.started_at = float(state.get("started_at", self.started_at))

        successful = sum(1 for path in self.output_root.glob("model*.mp4") if path.is_file() and path.stat().st_size > 0)
        logs = []
        failed = set()
        latest = {}
        for index, worker in enumerate(state.get("workers", [])):
            name = worker.get("name", f"worker{index}")
            log_path = Path(worker.get("stdout", self.log_root / f"worker{index}.stdout.log"))
            content = read_tail(log_path)
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            for line in lines:
                output_match = re.search(r"->\s*model(\d+)\.mp4", line)
                if output_match:
                    latest[name] = output_match.group(1)
                if "失败:" in line and latest.get(name):
                    failed.add(latest[name])
            if lines:
                logs.append((name, lines[-8:]))

        failed -= {f"{index:05d}" for index in range(1, self.total + 1) if (self.output_root / f"model{index:05d}.mp4").is_file()}
        processed = min(self.total, successful + len(failed))
        elapsed = max(0.0, time.time() - self.started_at)
        rate = successful / elapsed if elapsed > 0 else 0.0
        eta = (self.total - successful) / rate if rate > 0 else None
        remaining = max(0, self.total - successful - len(failed))
        self.progress.configure(value=processed)
        self.summary.configure(text=f"已完成 {successful:,} / {self.total:,} 个视频    （{processed / self.total * 100:.1f}% 已处理）" if self.total else "没有待处理项目")
        eta_text = self._duration(eta) if eta is not None else "计算中…"
        self.details.configure(text=f"成功：{successful:,}    失败：{len(failed):,}    待处理：{remaining:,}\n已运行：{self._duration(elapsed)}    平均速度：{rate * 60:.1f} 个/分钟    预计剩余：{eta_text}")

        worker_lines = []
        for index, worker in enumerate(state.get("workers", [])):
            name = worker.get("name", f"worker{index}")
            pid = worker.get("pid")
            status = "运行中" if process_running(pid) else "已结束/未启动"
            current = latest.get(name)
            worker_lines.append(f"{name}  PID {pid or '—'}  {status}" + (f"  最近处理：model{current}" if current else ""))
        self.workers.configure(text="\n".join(worker_lines) if worker_lines else "等待批次启动…")

        display_lines = []
        for name, lines in logs:
            display_lines.extend(f"[{name}] {line}" for line in lines)
        self.log_view.configure(state="normal")
        self.log_view.delete("1.0", "end")
        self.log_view.insert("end", "\n".join(display_lines[-16:]))
        self.log_view.see("end")
        self.log_view.configure(state="disabled")

    @staticmethod
    def _duration(seconds):
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours}小时{minutes:02d}分{secs:02d}秒" if hours else f"{minutes}分{secs:02d}秒"

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="显示单轨照片转视频的实时进度窗口。")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--total", type=int, required=True)
    args = parser.parse_args()
    ProgressWindow(args).run()


if __name__ == "__main__":
    main()
