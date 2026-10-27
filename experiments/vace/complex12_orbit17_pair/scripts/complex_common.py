"""Orbit17 run adapter; shared Rev1 engine functions bind to this run context."""
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
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image, ImageFilter

SCRIPTS = Path(__file__).resolve().parent
PROJECT = SCRIPTS.parents[3]
RUN = Path(os.environ['ORBIT17_RUN_ROOT']).resolve()
if not RUN.is_relative_to(PROJECT / 'outputs/vace/complex12_orbit17_pair'):
    raise ValueError('Orbit17 outputs must stay in their own run')
REV1_RUN = PROJECT / 'outputs/ComplexSingleOrbit_Rev1_Pilot_20261022'
HARD_RUN = PROJECT / 'outputs/vace/complex12_originalhard_eval/run_20261026_012033'
PYTHON = '/home/shi/miniconda3/envs/scj/bin/python'
sys.path.insert(0, str(PROJECT / 'scripts'))
sys.path.insert(0, str(PROJECT / 'code/DiffSynth-Studio'))
TRAIN = ['CS01', 'CS03', 'CS05', 'CS06', 'CS07', 'CS09', 'CS10', 'CS11']
VAL = ['CS02', 'CS04', 'CS08', 'CS12']
IDS = [f'CS{i:02}' for i in range(1, 13)]
METHODS = ('C_lin17', 'C_ring17')
VERSIONS = ('Rev1', 'OriginalHard')
PROBES = [('Rev1', 'CS10'), ('Rev1', 'CS12'), ('OriginalHard', 'CS10'), ('OriginalHard', 'CS12')]
TEMPORAL_GROUPS = [[0], list(range(1, 5)), list(range(5, 9)), list(range(9, 13)), list(range(13, 17))]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def now():
    return datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(timespec='seconds')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def identity(value):
    # Fingerprint small configuration/identity records, never model/data files.
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


@lru_cache(maxsize=1)
def config():
    value = read_json(RUN / 'resolved_config.json')
    fingerprint = value.pop('config_id')
    require(identity(value) == fingerprint, 'Frozen Orbit17 configuration changed')
    value['config_id'] = fingerprint
    return value


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def window_ids(method, start):
    require(method in METHODS, 'Unknown window method')
    require(0 <= start <= (63 if method == 'C_lin17' else 79), 'Illegal window start')
    return [(start + i) % 80 for i in range(17)]


def eligible_starts(visible, method):
    require(len(visible) == 80, 'Expected visibility for frames 000-079')
    return [s for s in range(64 if method == 'C_lin17' else 80)
            if any(visible[f] for f in window_ids(method, s))]


def paired_tables(original, eligible):
    require(len(original) == 300 and [int(r['update']) for r in original] == list(range(1, 301)),
            'Original 300-row table is incomplete')
    shared = [dict(row) for row in original]
    rng = random.Random(20261027)
    for update in range(301, 601):
        cs = rng.choice(TRAIN)
        shared.append(dict(update=str(update), cs=cs,
                           start=str(rng.choice(eligible['C_lin17'][cs])),
                           timestep_index=str(rng.randrange(1000)),
                           noise_seed=str(rng.randrange(2**31)),
                           stochastic_seed=str(rng.randrange(2**31))))
    ring_rng = random.Random(20261028)
    result = {method: [] for method in METHODS}
    for row in shared:
        cs = row['cs']
        require(cs in TRAIN and int(row['start']) in eligible['C_lin17'][cs], 'Ineligible linear row')
        ring_start = ring_rng.choice(eligible['C_ring17'][cs])
        for method, start in [('C_lin17', int(row['start'])), ('C_ring17', ring_start)]:
            result[method].append(dict(row, start=str(start), method=method, data_version='Rev1',
                                       frame_ids=json.dumps(window_ids(method, start))))
    return result


def sampling(method):
    require(method in METHODS, 'Unknown training method')
    rows = read_csv(RUN / f'sampling_{method}.csv')
    require(identity(rows) == config()['sampling_ids'][method], 'Frozen sampling table changed')
    require(len(rows) == 600 and [int(r['update']) for r in rows] == list(range(1, 601)), 'Invalid update budget')
    for row in rows:
        require(row['method'] == method and row['data_version'] == 'Rev1' and row['cs'] in TRAIN,
                'Wrong training identity')
        require(json.loads(row['frame_ids']) == window_ids(method, int(row['start'])), 'Frame order changed')
    return rows


@lru_cache(maxsize=2)
def index(version='Rev1'):
    require(version in VERSIONS, 'Unknown data version')
    return {s['display_id']: s for s in read_json(Path(config()['data_roots'][version]) / 'dataset_index.json')['samples']}


def source_path(cs, role, frame, version='Rev1'):
    require(cs in IDS and role in ('gt', 'damaged', 'exact', 'structure_raw') and 0 <= frame <= 80,
            'Invalid input identity')
    return Path(config()['data_roots'][version]) / index(version)[cs][role] / f'{frame:03}.png'


def read_image(path, mode='RGB'):
    with Image.open(path) as image:
        require(image.size == (512, 512), f'Unexpected dimensions: {path}')
        return image.convert(mode).copy()


def source(cs, role, frame, version='Rev1'):
    return read_image(source_path(cs, role, frame, version), 'L' if role in ('exact', 'structure_raw') else 'RGB')


def conditions(cs, frame_ids, version='Rev1'):
    from vace_p01 import adapt_images
    frames = list(frame_ids)
    result = {'conditioning': [], 'model_mask': [], 'cs': cs, 'frame_ids': frames, 'data_version': version}
    for frame in frames:
        mask, image = adapt_images(source(cs, 'damaged', frame, version), source(cs, 'exact', frame, version))
        result['conditioning'].append(image)
        result['model_mask'].append(mask)
    return result


def labels(cs, frame_ids, version='Rev1'):
    masks, edges = [], []
    for frame in frame_ids:
        mask = np.asarray(source(cs, 'exact', frame, version)) > 0
        raw = source(cs, 'structure_raw', frame, version)
        edge = (np.asarray(raw.filter(ImageFilter.MaxFilter(7))) > 0) & mask
        masks.append(mask)
        edges.append(edge)
    return np.stack(masks), np.stack(edges)


def loss_grid(pixel):
    require(pixel.shape == (17, 512, 512), 'Assemble the ordered 17 frames before mapping')
    spatial = pixel.reshape(17, 64, 8, 64, 8).max(axis=(2, 4))
    return np.stack([spatial[group].max(axis=0) for group in TEMPORAL_GROUPS]).astype(np.float32)


def training_sample(row):
    frames = json.loads(row['frame_ids'])
    require(row['cs'] in TRAIN and row['data_version'] == 'Rev1', 'Training split/version violation')
    require(frames == window_ids(row['method'], int(row['start'])), 'Training frames are not the specified window')
    sample = conditions(row['cs'], frames)
    sample['gt'] = [source(row['cs'], 'gt', frame) for frame in frames]
    masks, edges = labels(row['cs'], frames)
    require(masks.any(), 'Fully empty training window')
    return sample, loss_grid(masks), loss_grid(edges)


def cache_identity(row):
    return dict(data_version=row['data_version'], cs=row['cs'], frame_ids=json.loads(row['frame_ids']),
                encoding_config=config()['encoding_config_id'])


def cache_path(row):
    key = cache_identity(row)
    return RUN / 'cache' / key['data_version'] / key['cs'] / (identity(key) + '.pt')


def full_inputs():
    return [('OriginalHard', cs) for cs in ('CS10', 'CS12', 'CS09', 'CS11')] + [('Rev1', cs) for cs in IDS]


def trajectory_order():
    return [(method, step, version, cs) for step in (300, 600)
            for method in METHODS for version, cs in (PROBES if step == 300 else full_inputs())]


def prediction_dir(method, step, version, cs):
    if method == 'C_old':
        return (REV1_RUN if version == 'Rev1' else HARD_RUN) / 'predictions/C' / cs
    require((method, int(step), version, cs) in trajectory_order(), 'Unapproved prediction identity')
    return RUN / 'predictions' / method / f'step{step}' / version / cs


def checkpoint_path(method, step):
    require(method in METHODS and step in (300, 600), 'Unapproved checkpoint identity')
    return RUN / 'checkpoints' / method / f'step{step}.safetensors'


def infer_job(method, step, version, cs):
    return f'infer_{method}_{step}_{version}_{cs}'


def completion_path(job):
    return RUN / 'logs' / f'{job}.done.json'


def mark_complete(job, **extra):
    write_json(completion_path(job), dict(job=job, config_id=config()['config_id'], completed_at=now(), **extra))


def valid_prediction(method, step, version, cs):
    marker = completion_path(infer_job(method, step, version, cs))
    if not marker.exists():
        return False
    record = read_json(marker)
    require(record['config_id'] == config()['config_id'], 'Prediction config differs')
    require((record['method'], record['checkpoint_step'], record['data_version'], record['cs']) ==
            (method, step, version, cs), 'Prediction identity differs')
    require(record.get('worker_exit_code', 0) == 0, 'Previous worker did not exit normally')
    folder = prediction_dir(method, step, version, cs)
    require(sorted(p.name for p in folder.glob('*.png')) == [f'{i:03}.png' for i in range(81)], 'Incomplete trajectory')
    for frame in range(81):
        read_image(folder / f'{frame:03}.png')
    return True


@lru_cache(maxsize=None)
def legacy(name):
    # Rev1's helper imports complex_common; this adapter supplies the explicit
    # Orbit17 run/config. Never call its fixed-budget train/evaluate entrypoints.
    path = PROJECT / 'experiments/vace/complex12_rev1/scripts' / f'{name}.py'
    spec = importlib.util.spec_from_file_location('orbit17_legacy_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
