import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "data/pointcloud_vace"
PYTHON = "/home/shi/miniconda3/envs/scj/bin/python"


def now():
    return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def event(run, stage, **extra):
    value = dict(time=now(), stage=stage, pid=os.getpid(), **extra)
    write_json(run / "status.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def atomic_torch_save(path, value):
    import torch
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def save_sample(path, sample, labels, geometry, audit):
    import numpy as np
    values = {k: np.stack([np.array(i) for i in v]) for k, v in sample.items()}
    values.update({"geom_" + k: v for k, v in geometry.items()})
    values["labels"] = labels
    np.savez_compressed(path, **values)
    write_json(Path(path).with_suffix(".json"), audit)


def load_sample(path):
    import numpy as np
    from PIL import Image
    with np.load(path) as z:
        sample = {k: [Image.fromarray(a) for a in z[k]] for k in ("gt", "conditioning", "model_mask")}
        geometry = {k[5:]: z[k] for k in z.files if k.startswith("geom_")}
        labels = z["labels"]
    return sample, labels, geometry, json.loads(Path(path).with_suffix(".json").read_text())
