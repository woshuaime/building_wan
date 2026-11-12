"""Start the all-mesh generator in an independent tmux session."""
import json
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from build_fps import DATA, RUNTIME

PYTHON = "/home/shi/miniconda3/envs/scj/bin/python"


def launch():
    stamp = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%d_%H%M%S")
    prefix = RUNTIME / "logs" / ("generate_" + stamp)
    session = "mesh_gt_" + stamp
    command = [PYTHON, "-u", "-B", str(Path(__file__).resolve().with_name("sample_mesh_gt.py")),
               "generate", "--log-prefix", str(prefix)]
    environment = ["CUDA_VISIBLE_DEVICES=0", "OMP_NUM_THREADS=2", "OPENBLAS_NUM_THREADS=1",
                   "MKL_NUM_THREADS=2", "PYTHONUNBUFFERED=1",
                   "TMPDIR=" + str(RUNTIME / "tmp"), "PYTHONPYCACHEPREFIX=" + str(RUNTIME / "pycache")]
    shell = "exec env " + shlex.join(environment + command) + " >> " + shlex.quote(str(prefix) + ".log") + " 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, "-c", str(DATA), shell], check=True)
    pid = int(subprocess.check_output(["tmux", "list-panes", "-t", session, "-F", "#{pane_pid}"], text=True).strip())
    record = dict(time=datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(), session=session, pid=pid,
                  log=str(prefix) + ".log", summary=str(prefix) + ".summary.json",
                  input=str(DATA / "sample-data-organized/mesh"), output=str(DATA / "mesh_gt_pointcloud"),
                  command=command, independent_of_codex_and_ssh=True)
    Path(str(prefix) + ".launch.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    launch()
