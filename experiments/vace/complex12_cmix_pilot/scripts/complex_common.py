"""C_mix run isolation and version-aware data; reuse the Rev1 engine unchanged."""
import csv
import hashlib
import importlib.util
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

SCRIPTS = Path(__file__).resolve().parent
PROJECT = SCRIPTS.parents[3]
RUN = Path(os.environ['CMIX_RUN_ROOT']).resolve()
if not RUN.is_relative_to(PROJECT / 'outputs/vace/complex12_cmix_pilot'):
    raise ValueError('C_mix outputs must remain in their own run')
REV1_RUN = PROJECT / 'outputs/ComplexSingleOrbit_Rev1_Pilot_20261022'
HARD_RUN = PROJECT / 'outputs/vace/complex12_originalhard_eval/run_20261026_012033'
PYTHON = '/home/shi/miniconda3/envs/scj/bin/python'
sys.path.insert(0, str(PROJECT / 'scripts'))
sys.path.insert(0, str(PROJECT / 'code/DiffSynth-Studio'))
TRAIN = ['CS01', 'CS03', 'CS05', 'CS06', 'CS07', 'CS09', 'CS10', 'CS11']
VAL = ['CS02', 'CS04', 'CS08', 'CS12']
IDS = [f'CS{i:02}' for i in range(1, 13)]
VERSIONS = ('Rev1', 'OriginalHard')
HARD_IDS = ['CS09', 'CS10', 'CS11', 'CS12']
TOTALS = dict(zip(TRAIN, (39, 34, 44, 46, 36, 40, 29, 32)))
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

def config_id(c):
    return hashlib.sha256(json.dumps(c, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

@lru_cache(maxsize=1)
def config():
    c = read_json(RUN / 'resolved_config.json')
    fingerprint = c.pop('config_id')
    require(config_id(c) == fingerprint, 'Frozen configuration changed')
    c['config_id'] = fingerprint
    return c

def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def mixed_sampling(original):
    require(len(original) == 300, 'Expected original 300-row sampling table')
    require([int(r['update']) for r in original] == list(range(1, 301)), 'Update order changed')
    require(Counter(r['cs'] for r in original) == TOTALS, 'Training building counts changed')
    require(all(r['cs'] in TRAIN and 0 <= int(r['start']) <= 63 for r in original), 'Training leakage or illegal window')
    rows = [dict(r, data_version='Rev1') for r in original]
    rng = random.Random(20261026)
    for cs, count in [('CS09', 20), ('CS10', 14), ('CS11', 16)]:
        choices = [i for i, row in enumerate(rows) if row['cs'] == cs]
        for i in rng.sample(choices, count):
            rows[i]['data_version'] = 'OriginalHard'
    return rows

def sampling():
    rows = read_csv(RUN / 'shared_sampling_table.csv')
    expected = mixed_sampling(read_csv(REV1_RUN / 'shared_sampling_table.csv'))
    require(rows == expected, 'C_mix table differs from the authorized fixed allocation')
    return rows

def data_root(version):
    require(version in VERSIONS, 'Unknown data version')
    return Path(config()['data_roots'][version])

@lru_cache(maxsize=2)
def index(version='Rev1'):
    return {s['display_id']: s for s in read_json(data_root(version) / 'dataset_index.json')['samples']}

def source_path(cs, role, frame, version='Rev1'):
    require(cs in IDS and role in ('gt', 'damaged', 'exact', 'structure_raw') and 0 <= frame <= 80, 'Invalid source identity')
    return data_root(version) / index(version)[cs][role] / f'{frame:03}.png'

def read_image(path, mode='RGB'):
    with Image.open(path) as im:
        require(im.size == (512, 512), f'Unexpected dimensions: {path}')
        return im.convert(mode).copy()

def source(cs, role, frame, version='Rev1'):
    return read_image(source_path(cs, role, frame, version), 'L' if role in ('exact', 'structure_raw') else 'RGB')

def conditions(cs, start=0, count=81, version='Rev1'):
    from vace_p01 import adapt_images
    result = {'conditioning': [], 'model_mask': [], 'cs': cs, 'start': start, 'data_version': version}
    for frame in range(start, start + count):
        mask, condition = adapt_images(source(cs, 'damaged', frame, version), source(cs, 'exact', frame, version))
        result['conditioning'].append(condition)
        result['model_mask'].append(mask)
    return result

def labels(cs, start, count, version='Rev1'):
    ms, es = [], []
    for frame in range(start, start + count):
        m = np.asarray(source(cs, 'exact', frame, version)) > 0
        raw = source(cs, 'structure_raw', frame, version)
        e = (np.asarray(raw.filter(ImageFilter.MaxFilter(7))) > 0) & m
        ms.append(m)
        es.append(e)
    return np.stack(ms), np.stack(es)

def loss_grid(pixel):
    require(pixel.shape == (17, 512, 512), 'Select 17 original frames before mapping labels')
    spatial = pixel.reshape(17, 64, 8, 64, 8).max(axis=(2, 4))
    return np.stack([spatial[g].max(axis=0) for g in TEMPORAL_GROUPS]).astype(np.float32)

def training_sample(cs, start, version):
    require(cs in TRAIN and 0 <= start <= 63, 'Training leakage or illegal window')
    require(version == 'Rev1' or cs in ('CS09', 'CS10', 'CS11'), 'Unapproved Hard training identity')
    sample = conditions(cs, start, 17, version)
    sample['gt'] = [source(cs, 'gt', frame, version) for frame in range(start, start + 17)]
    m, e = labels(cs, start, 17, version)
    require(m.any(), 'Unexpected fully empty training window')
    return sample, loss_grid(m), loss_grid(e)

def completion_path(job):
    return RUN / 'logs' / (job + '.done.json')

def mark_complete(job, **extra):
    write_json(completion_path(job), {'job': job, 'config_id': config()['config_id'], 'completed_at': now(), **extra})

def trajectory_order():
    return [('OriginalHard', cs) for cs in HARD_IDS] + [('Rev1', cs) for cs in IDS]

def prediction_dir(version, cs, old=False):
    if old:
        return (REV1_RUN if version == 'Rev1' else HARD_RUN) / 'predictions/C' / cs
    if version == 'OriginalHard' and cs not in HARD_IDS:
        version = 'Rev1'
    return RUN / 'predictions' / version / cs

def valid_prediction(version, cs):
    marker = completion_path(f'infer_{version}_{cs}')
    if not marker.exists():
        return False
    require(read_json(marker)['config_id'] == config()['config_id'], 'Prediction configuration mismatch')
    folder = prediction_dir(version, cs)
    require(sorted(p.name for p in folder.glob('*.png')) == [f'{i:03}.png' for i in range(81)], 'Incomplete prediction')
    for i in range(81):
        read_image(folder / f'{i:03}.png')
    return True

@lru_cache(maxsize=None)
def legacy(name):
    path = PROJECT / 'experiments/vace/complex12_rev1/scripts' / f'{name}.py'
    spec = importlib.util.spec_from_file_location('cmix_legacy_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
