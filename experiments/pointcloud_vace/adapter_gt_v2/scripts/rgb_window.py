"""Lossless V2 luma, matching square camera calibration and observed-only Adapter geometry."""
import av
import numpy as np
from scipy.spatial import cKDTree

import data

SIDE = 512
PAD = (data.WIDTH - data.HEIGHT) // 2
SCALE = SIDE / data.WIDTH
AFFINE = (SCALE, SCALE, (SCALE - 1) / 2, (PAD + .5) * SCALE - .5)
# Only pixels fully inside the original camera frame contribute to RGB supervision.
VALID_START = int(np.ceil(PAD * SCALE))
VALID_STOP = SIDE - VALID_START


def square_frames(gray):
    canvas = np.full((data.WIDTH, data.WIDTH), 240, np.uint8)
    if gray.ndim != 3 or tuple(gray.shape[1:]) != (data.HEIGHT, data.WIDTH):
        raise ValueError(f"Expected original V2 [T,480,832] planes, received {gray.shape}")
    result = np.empty((len(gray), SIDE, SIDE), np.uint8)
    for i, frame in enumerate(gray):
        canvas[PAD:PAD + data.HEIGHT] = frame
        image = av.VideoFrame.from_ndarray(canvas, format="gray")
        result[i] = image.reformat(width=SIDE, height=SIDE, interpolation="AREA").to_ndarray()
    return result


def read_window(path, frame_ids):
    """Read both sides of a wrapped clip in one sequential pass; preserve requested order."""
    wanted = {number: index for index, number in enumerate(frame_ids)}
    original = np.empty((len(frame_ids), data.HEIGHT, data.WIDTH), dtype=np.uint8)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        for number, frame in enumerate(container.decode(stream)):
            index = wanted.pop(number, None)
            if index is None:
                continue
            if (frame.width, frame.height, int(frame.color_range)) != (data.WIDTH, data.HEIGHT, 2):
                raise ValueError(f"V2 luma shape/range changed: {path}")
            if frame.format.name not in ("gray", "gray8", "yuvj420p", "yuvj444p", "yuv420p", "yuv444p"):
                raise ValueError(f"Unexpected V2 format {frame.format.name}: {path}")
            plane = frame.planes[0]
            original[index] = np.frombuffer(plane, np.uint8).reshape(frame.height, plane.line_size)[:, :frame.width]
            if not wanted:
                break
    if wanted:
        raise ValueError(f"Missing frames {sorted(wanted)} in {path}")
    return square_frames(original)


def observed_geometry(row, camera, record, frame_ids):
    points = data.selected_partial(row, camera, record)
    centers, rotations, K = data.camera_arrays(camera)
    K[:2, :] *= SCALE
    K[:2, 2] += AFFINE[2:]
    scale = max(float(np.linalg.norm(points - (points.min(0) + points.max(0)) / 2, axis=1).max()), 1.)
    anchors = points[data.fps(points)]
    if len(anchors) < 128:
        anchors = anchors[np.arange(128) % len(anchors)]
    density = cKDTree(points).query_ball_point(anchors, .5, return_length=True)
    refs = list(range(0, 80, 10))
    frames = list(frame_ids)
    depths = {i: data.depth_view(points, centers[i], rotations[i], K, scale, (SIDE, SIDE))
              for i in dict.fromkeys([*refs, *frames])}
    return dict(reference_views=np.stack([depths[i] for i in refs]),
                anchors=(anchors / scale).astype(np.float32),
                anchor_density=np.log1p(density).astype(np.float32),
                ref_C=(centers[refs] / scale).astype(np.float32), ref_R=rotations[refs].astype(np.float32),
                K=K.astype(np.float32), depth_tolerance=np.asarray(.45 / scale, np.float32),
                target_depths=np.stack([depths[i] for i in frames]),
                target_C=(centers[frames] / scale).astype(np.float32), target_R=rotations[frames].astype(np.float32))


def load_window(row, frame_ids):
    camera, record = (data.read_json(row[key]) for key in ("camera", "record"))
    data.validate_camera(camera, row["id"], "train")
    if str(record["id"]) != row["id"] or record['split_use'] != 'train':
        raise ValueError(f"Wrong training record: {row['id']}")
    geometry = observed_geometry(row, camera, record, frame_ids)
    target = read_window(row['GT_video'], frame_ids)
    return dict(id=row['id'], frame_ids=frame_ids, mesh=row["mesh"], camera=camera, geometry=geometry,
                target=target, partial=read_window(row['partial_video'], frame_ids),
                occupancy=data.occupancy_for(target, (SIDE, SIDE)))
