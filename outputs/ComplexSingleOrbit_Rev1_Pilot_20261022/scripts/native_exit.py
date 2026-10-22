"""GDB command file: capture inferior exit status and dump only stopped faults.

Loaded by GDB, not by the training interpreter. Run using -x this_file and
-ex wan-run, with WAN_GDB_EXIT_RECORD pointing to a new JSON output file.
"""
import json
import os
from pathlib import Path
import signal

import gdb


class WanRun(gdb.Command):
    def __init__(self):
        super().__init__("wan-run", gdb.COMMAND_RUNNING)
        self.exited = False
        self.exit_code = None
        self.signals = []
        gdb.events.stop.connect(self.on_stop)
        gdb.events.exited.connect(self.on_exit)

    def on_exit(self, event):
        self.exited = True
        self.exit_code = getattr(event, "exit_code", None)

    def on_stop(self, event):
        if not isinstance(event, gdb.SignalEvent):
            return
        name = event.stop_signal
        self.signals.append(name)
        if name not in ("SIGSEGV", "SIGABRT", "SIGBUS", "SIGILL", "SIGFPE"):
            return
        for command in ("thread apply all bt 20",
                        "info registers rax rbx rcx rdx rsi rdi rbp rsp rip",
                        "x/12i $pc"):
            try:
                gdb.execute(command)
            except gdb.error as error:
                gdb.write(f"Diagnostic command unavailable: {error}\n")
        # Conda's Python often lacks the optional python-gdb auto-load helper.
        try:
            gdb.execute("help py-bt", to_string=True)
        except gdb.error:
            gdb.write("Optional Python backtrace helper is unavailable.\n")
        else:
            try:
                gdb.execute("py-bt")
            except gdb.error as error:
                gdb.write(f"Python backtrace unavailable: {error}\n")

    def invoke(self, argument, from_tty):
        gdb.execute("set pagination off")
        gdb.execute("set confirm off")
        gdb.execute("set print thread-events off")
        gdb.execute("set debuginfod enabled off")
        gdb.execute("run")
        while not self.exited:
            # faulthandler can re-raise the same fatal signal once. Permit
            # delivery so GDB records the actual termination rather than
            # leaving a stopped inferior behind when batch commands end.
            if len(self.signals) > 8:
                gdb.execute("kill")
                break
            gdb.execute("continue")
        code = self.exit_code
        if code is None:
            number = getattr(signal, self.signals[-1], None) if self.signals else None
            code = 128 + int(number) if number is not None else 1
        result = {"inferior_exit_code": self.exit_code,
                  "signals": self.signals, "launcher_exit_code": code,
                  "status": "exited" if self.exit_code is not None else "signaled"}
        path = os.environ.get("WAN_GDB_EXIT_RECORD")
        if path:
            with Path(path).open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(result, indent=2) + "\n")
        gdb.execute(f"quit {code}")


WanRun()
