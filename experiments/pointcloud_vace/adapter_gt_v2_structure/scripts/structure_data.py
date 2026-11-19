"""Fixed data split, exact source geometry, and CPU preparation for this run."""

import csv
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
import random
import sqlite3
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(1, str(HERE.parents[1] / "adapter_gt_v2/scripts"))

import data
import validation_delivery as delivery
from common import ROOT, identity, now, sha, write_json
from mesh_labels import read_obj, structural_edges

OUTPUTS = ROOT / "outputs/pointcloud_vace/adapter_gt_v2_structure"
BASE = ROOT / "outputs/pointcloud_vace/adapter_gt_v2" / delivery.RUN_NAME
CLOUDS = delivery.DATASET.parent / "mesh_gt_pointcloud_stable"
MESHES = delivery.DATASET.parent / "sample-data-organized/mesh"
VALIDATION_IDS = ("358", "2825", "5954", "4850", "3538", "2016", "6485", "5466")
SEED = 20261120
TRAIN_COUNT = 10747
EPOCHS = 3
MAX_EPOCHS = 5
CACHE_VERSION = 1
VERTEX_QUERIES = 96
SURFACE_QUERIES = 64
MAX_POINTS = 4096


def frame_window(building, epoch):
    start = (int(data.digest(f"{SEED}:{building}".encode()), 16) + 21 * epoch) % 64
    return list(range(start, start + 17))


def split_and_schedule():
    manifest = delivery.DATASET / "manifest.csv"
    manifest_sha = sha(manifest)
    if manifest_sha != delivery.MANIFEST_SHA256:
        raise ValueError("The V2 manifest changed")
    rows = {}
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            building, split = row["id"], row["split_use"]
            if split != "train" and building not in VALIDATION_IDS:
                continue
            expected = "validation" if building in VALIDATION_IDS else "train"
            if not building.isdecimal() or split != expected or building in rows:
                raise ValueError(f"Unexpected ID/split {building}/{split}")
            paths = {key: str(data._relative(delivery.DATASET, row[key], key, building))
                     for key in ("camera", "record", "partial_video", "GT_video")}
            observed = delivery.POINTS / f"{building}.ply"
            if not observed.exists():
                observed = observed.with_suffix(".xyz")
            paths.update(pointcloud=str(observed), mesh=str(MESHES / f"{building}.obj"),
                         gt_cloud=str(CLOUDS / f"{building}.ply"))
            for path in paths.values():
                if not Path(path).is_file():
                    raise FileNotFoundError(path)
            rows[building] = dict(id=building, split_use=split, **paths)
    train = sorted((i for i, row in rows.items() if row["split_use"] == "train"), key=int)
    if len(train) != TRAIN_COUNT or not set(VALIDATION_IDS) <= rows.keys():
        raise ValueError("Missing original training or eight validation IDs")
    order = []
    for epoch in range(MAX_EPOCHS):
        names = train.copy()
        random.Random(SEED + epoch).shuffle(names)
        order.extend((building, frame_window(building, epoch)) for building in names)
    return rows, order, manifest_sha


def _npz(arrays):
    stream = BytesIO()
    np.savez(stream, **arrays)
    return stream.getvalue()


def _unpack(value):
    with np.load(BytesIO(value), allow_pickle=False) as arrays:
        return {name: arrays[name] for name in arrays.files}


def load_observation(row, camera, record):
    if str(record["id"]) != row["id"] or record["split_use"] != row["split_use"]:
        raise ValueError(f"Record belongs to a different building: {row['id']}")
    xyz = data.selected_partial(row, camera, record)
    centers, rotations, intrinsics = data.camera_arrays(camera)
    midpoint = (xyz.min(0) + xyz.max(0)) / 2
    scale = max(float(np.linalg.norm(xyz - midpoint, axis=1).max()), 1.)
    return dict(xyz=xyz, centers=centers, rotations=rotations, intrinsics=intrinsics,
                midpoint=midpoint, scale=np.asarray(scale))


def _gt_cloud(path):
    fields = (b"property double x", b"property double y", b"property double z",
              b"property float nx", b"property float ny", b"property float nz")
    dtype = np.dtype([("xyz", "<f8", (3,)), ("normals", "<f4", (3,))])
    with Path(path).open("rb") as stream:
        header = []
        for line in stream:
            header.append(line.strip())
            if header[-1] == b"end_header":
                break
            if len(header) > 50:
                raise ValueError(f"Invalid GT header: {path}")
        if (header[:2] != [b"ply", b"format binary_little_endian 1.0"]
                or b"element vertex 16384" not in header or tuple(
                    line for line in header if line.startswith(b"property ")) != fields):
            raise ValueError(f"Unexpected fixed GT PLY fields: {path}")
        values = np.fromfile(stream, dtype=dtype, count=16384)
    if len(values) != 16384 or not all(np.isfinite(values[name]).all() for name in dtype.names):
        raise ValueError(f"Unusable GT point cloud: {path}")
    return values


def load_targets(row, camera, record):
    origin = np.asarray(camera["world_to_local_origin"], dtype=np.float64)
    vertices, faces, normals = read_obj(Path(row["mesh"]))
    labels = structural_edges(vertices - origin, faces, normals)
    geometry = _gt_cloud(row["gt_cloud"])
    xyz = geometry["xyz"] + (np.asarray(camera["GT_PLY_local_to_world_offset"]) - origin)
    selected = np.asarray(record["selected_source_point_indices"]["gt"], dtype=np.int64)
    if (len(selected) != int(record["gt_display_points"])
            or not 64 <= len(selected) <= MAX_POINTS or len(np.unique(selected)) != len(selected)
            or selected.min() < 0 or selected.max() >= len(xyz)):
        raise ValueError(f"Invalid display subset for {row['id']}")
    labels.update(full_xyz=xyz.astype(np.float32), display_xyz=xyz[selected].astype(np.float32),
                  display_normals=geometry["normals"][selected])
    for name in ("vertices", "supports", "triangles", "normals"):
        labels[name] = labels[name].astype(np.float32)
    return labels


def _prepare(row):
    camera, record = (data.read_json(row[key]) for key in ("camera", "record"))
    data.validate_camera(camera, row["id"], row["split_use"])
    observed = load_observation(row, camera, record)
    targets = load_targets(row, camera, record)
    files = {k: identity(row[k]) for k in ("partial_video", "GT_video")}
    if any(files[key]["bytes"] != record["video_bytes"][role]
           for key, role in (("partial_video", "partial"), ("GT_video", "gt"))):
        raise ValueError(f"Video sizes disagree with V2 record for {row['id']}")
    return (row["id"], _npz(observed), _npz(targets), json.dumps(files),
            len(targets["vertices"]), len(targets["edges"]))


def prepare(run, workers):
    rows, order, manifest_sha = split_and_schedule()
    config = run / "data.json"
    source = dict(version=CACHE_VERSION, manifest_sha256=manifest_sha, seed=SEED,
                  rows=list(rows.values()), schedule=order, validation=list(VALIDATION_IDS))
    if config.exists():
        if data.digest(data.read_json(config)) != data.digest(source):
            raise ValueError("Prepared schedule differs from the source")
    else:
        write_json(config, source)
    target_path = run / "geometry.sqlite"
    with sqlite3.connect(target_path) as database:
        database.execute("PRAGMA synchronous=NORMAL")
        database.execute("""CREATE TABLE IF NOT EXISTS buildings (
            id TEXT PRIMARY KEY, observed BLOB, targets BLOB, files TEXT, vertices INT, edges INT
        ) WITHOUT ROWID""")
        done = {key[0] for key in database.execute("SELECT id FROM buildings")}
        pending = [row for key, row in rows.items() if key not in done]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(0, len(pending), 64):
                group = pending[start:start + 64]
                payloads = list(pool.map(_prepare, group))
                database.executemany("INSERT INTO buildings VALUES (?,?,?,?,?,?)", payloads)
                database.commit()
                count = len(done) + start + len(group)
                print(f"CPU 几何准备 {count}/{len(rows)}", flush=True)
        count, max_vertices, max_edges, over, empty = database.execute(
            "SELECT count(*), max(vertices), max(edges), sum(vertices > ?), sum(edges = 0) FROM buildings",
            (VERTEX_QUERIES,)).fetchone()
    if count != len(rows):
        raise RuntimeError(f"Geometry dataset has {count} rows, expected {len(rows)}")
    ready = dict(time=now(), version=CACHE_VERSION, train=len(rows) - len(VALIDATION_IDS),
                 validation=len(VALIDATION_IDS), windows=len(order), max_vertices=max_vertices,
                 max_edges=max_edges, over_vertex_queries=over, no_structural_edges=empty,
                 schedule_sha256=sha(config), geometry_bytes=target_path.stat().st_size,
                 base_weights=delivery.final_weights(BASE))
    write_json(run / "prepared.json", ready)
    print(f"CPU 准备完成 · {ready['train']} 栋训练 / {ready['validation']} 栋验证 · {run}", flush=True)
    return ready


class PreparedData:
    def __init__(self, run):
        self.database = run / "geometry.sqlite"
        self.summary = data.read_json(run / "prepared.json")
        if self.summary["version"] != CACHE_VERSION or sha(run / "data.json") != self.summary["schedule_sha256"]:
            raise ValueError("Prepared data identity changed")
        if sha(delivery.DATASET / "manifest.csv") != delivery.MANIFEST_SHA256:
            raise ValueError("V2 split identity changed")
        config = data.read_json(run / "data.json")
        self.rows = {row["id"]: row for row in config["rows"]}
        self.schedule = config["schedule"]

    def read(self, building, *columns):
        # Each thread opens its own read-only connection and fetches only the requested role.
        uri = self.database.as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True) as database:
            query = "SELECT " + ", ".join(columns) + " FROM buildings WHERE id=?"
            values = database.execute(query, (building,)).fetchone()
        return tuple(json.loads(value) if name == "files" else _unpack(value)
                     for name, value in zip(columns, values))

    def window(self, building, frames, pixels=True):
        row = self.rows[building]
        observed, targets, files = self.read(building, "observed", "targets", "files")
        camera = dict(centers=observed["centers"][frames], rotations=observed["rotations"][frames],
                      intrinsics=observed["intrinsics"], midpoint=observed["midpoint"], scale=observed["scale"])
        result = dict(id=building, frame_ids=frames, observed=observed, targets=targets, camera=camera)
        if pixels:
            for key in files:
                if identity(row[key]) != files[key]:
                    raise ValueError(f"Frozen video changed for {building}/{key}")
            result.update(partial=data.decode_gray_window(row["partial_video"], frames),
                          target=data.decode_gray_window(row["GT_video"], frames))
        return result
