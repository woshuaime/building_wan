"""Frozen Complex12 Rev1 data, label and bookkeeping helpers."""
import csv
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

RUN = Path(__file__).resolve().parents[1]
PROJECT = RUN.parents[1]
DATA = PROJECT / 'data/buildingfiller_import/Complex12/BuildingFiller_Complex12_512_Rev1_20261022'
PYTHON = '/home/shi/miniconda3/envs/scj/bin/python'
sys.path.insert(0, str(PROJECT / 'scripts'))
sys.path.insert(0, str(PROJECT / 'code/DiffSynth-Studio'))
TRAIN = ['CS01', 'CS03', 'CS05', 'CS06', 'CS07', 'CS09', 'CS10', 'CS11']
VAL = ['CS02', 'CS04', 'CS08', 'CS12']
IDS = [f'CS{i:02}' for i in range(1, 13)]
TEMPORAL_GROUPS = [[0], list(range(1, 5)), list(range(5, 9)), list(range(9, 13)), list(range(13, 17))]

def require(ok, message):
    if not ok:
        raise ValueError(message)

def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def config():
    c = read_json(RUN / 'resolved_config.json')
    fingerprint = c.pop('config_id')
    require(hashlib.sha256(json.dumps(c, sort_keys=True, ensure_ascii=False).encode()).hexdigest() == fingerprint,
            'Frozen configuration changed')
    c['config_id'] = fingerprint
    return c

def index():
    return {s['display_id']: s for s in read_json(DATA / 'dataset_index.json')['samples']}

def source_path(cs, role, frame):
    require(cs in IDS and role in ('gt', 'damaged', 'exact', 'structure_raw') and 0 <= frame <= 80,
            'Unexpected source identity')
    return DATA / index()[cs][role] / f'{frame:03}.png'

def read_image(path, mode='RGB'):
    with Image.open(path) as im:
        require(im.size == (512, 512), f'Unexpected image dimensions: {path}')
        return im.convert(mode).copy()

def source(cs, role, frame):
    return read_image(source_path(cs, role, frame), 'L' if role in ('exact', 'structure_raw') else 'RGB')

def conditions(cs, start=0, count=81):
    # This inference loader never opens GT RGB or GT structure.
    from vace_p01 import adapt_images
    result = {'conditioning': [], 'model_mask': [], 'cs': cs, 'start': start}
    for frame in range(start, start + count):
        damaged, exact = source(cs, 'damaged', frame), source(cs, 'exact', frame)
        mask, condition = adapt_images(damaged, exact)
        result['conditioning'].append(condition)
        result['model_mask'].append(mask)
    return result

def labels(cs, start, count):
    ms, es = [], []
    for frame in range(start, start + count):
        m = np.asarray(source(cs, 'exact', frame)) > 0
        # Dilation precedes intersection; damaged_region is intentionally unused.
        raw = source(cs, 'structure_raw', frame)
        e = (np.asarray(raw.filter(ImageFilter.MaxFilter(7))) > 0) & m
        ms.append(m)
        es.append(e)
    return np.stack(ms), np.stack(es)

def loss_grid(pixel):
    require(pixel.shape == (17, 512, 512), 'Map labels after selecting exactly 17 original frames')
    spatial = pixel.reshape(17, 64, 8, 64, 8).max(axis=(2, 4))
    return np.stack([spatial[g].max(axis=0) for g in TEMPORAL_GROUPS]).astype(np.float32)

def training_sample(cs, start):
    require(cs in TRAIN and 0 <= start <= 63, 'Training leakage or illegal window')
    sample = conditions(cs, start, 17)
    sample['gt'] = [source(cs, 'gt', frame) for frame in range(start, start + 17)]
    m, e = labels(cs, start, 17)
    require(m.any(), 'Empty Exact window is not eligible')
    return sample, loss_grid(m), loss_grid(e)

def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def completion_path(job):
    return RUN / 'logs' / (job + '.done.json')

def mark_complete(job, **extra):
    write_json(completion_path(job), {'job': job, 'config_id': config()['config_id'], 'completed_at': now(), **extra})

def valid_prediction(method, cs, decode=True):
    marker = completion_path(f'infer_{method}_{cs}')
    folder = RUN / 'predictions' / method / cs
    if not marker.exists():
        return False
    value = read_json(marker)
    require(value['config_id'] == config()['config_id'], f'Configuration mismatch: {marker}')
    require(sorted(p.name for p in folder.glob('*.png')) == [f'{i:03}.png' for i in range(81)],
            f'Incomplete marked prediction: {folder}')
    if decode:
        for i in range(81):
            read_image(folder / f'{i:03}.png')
    return True

def lora_path(method):
    if method == 'L':
        return Path(config()['legacy_lora']['path'])
    require(method in ('C', 'U', 'S'), 'Unknown adapter')
    return RUN / 'checkpoints' / method / 'step300.safetensors'
