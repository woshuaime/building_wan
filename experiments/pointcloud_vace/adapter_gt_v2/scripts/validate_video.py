"""Generate a held-out GT-video V2 comparison from the completed training run."""
import argparse
import csv
import json
import sys
from pathlib import Path

import av
import numpy as np
from av.video.reformatter import ColorPrimaries, ColorRange, ColorTrc, Colorspace, VideoReformatter
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

import data
import validation_delivery
from common import ROOT, sha

WIDTH, HEIGHT, FRAMES, FPS = data.WIDTH, data.HEIGHT, data.FRAMES, data.FPS
STEPS, SEED, HEADER = 30, 20261018, 32


def validation_row(run, building_id):
    """Use the original uploaded split and the sources resolved at import."""
    if run.name == validation_delivery.RUN_NAME:
        return validation_delivery.source_row(building_id)
    building_id = str(building_id)
    cfg = data.read_json(run / 'config.json')
    dataset = Path(cfg['paths']['data']).resolve()
    manifest = dataset / 'manifest.csv'
    if sha(manifest) != cfg['manifest_sha256']:
        raise ValueError('Uploaded split manifest changed after training')
    with manifest.open(encoding='utf-8-sig', newline='') as handle:
        rows = [r for r in csv.DictReader(handle) if r['id'] == building_id]
    if len(rows) != 1 or rows[0]['split_use'] != 'validation':
        raise ValueError(f'Not a held-out validation building: {building_id}')
    record = next(r for r in data.read_json(run / 'import_inventory.json') if r['id'] == building_id)
    if record['split_use'] != 'validation':
        raise ValueError('Stored import split disagrees with uploaded split')
    for key in ('camera', 'record', 'partial_video', 'GT_video'):
        if Path(record[key]) != (dataset / rows[0][key]).resolve():
            raise ValueError(f'Imported path disagrees with uploaded {key}')
    return record


def validation_geometry(row):
    """Build all reference and target views in RAM from the observed subset."""
    for key in ('camera', 'record'):
        if sha(row[key]) != row[key + '_sha256']:
            raise ValueError(f'Frozen {key} changed for {row["id"]}')
    camera, record = (data.read_json(row[key]) for key in ('camera', 'record'))
    data.validate_camera(camera, row['id'], 'validation')
    if str(record['id']) != row['id'] or record['split_use'] != 'validation':
        raise ValueError('Wrong partial selection or split')
    points = data.selected_partial(row, camera, record)
    return observed_geometry(points, camera, range(FRAMES))


def observed_geometry(points, camera, frame_ids):
    """Project one observed building into the frozen reference and requested views."""
    centers, rotations, K = data.camera_arrays(camera)
    scale = max(float(np.linalg.norm(points - (points.min(0) + points.max(0)) / 2, axis=1).max()), 1.)
    anchors = points[data.fps(points)]
    if len(anchors) < 128:
        anchors = anchors[np.arange(128) % len(anchors)]
    density = cKDTree(points).query_ball_point(anchors, .5, return_length=True)
    refs = list(range(0, 80, 10))
    base = dict(_points_f64=points, _centers_f64=centers, _rotations_f64=rotations,
                _K_f64=K, _scale=scale, anchors=(anchors / scale).astype(np.float32),
                anchor_density=np.log1p(density).astype(np.float32),
                ref_C=(centers[refs] / scale).astype(np.float32), ref_R=rotations[refs].astype(np.float32),
                depth_tolerance=np.asarray(.45 / scale, np.float32), K=K.astype(np.float32))
    geometry = data._window_geometry(base, list(frame_ids))
    geometry['reference_views'] = np.stack([data.depth_view(points, centers[i], rotations[i], K, scale)
                                            for i in refs])
    if not all(np.isfinite(v).all() for v in geometry.values()):
        raise ValueError(f'Nonfinite validation geometry for {camera["id"]}')
    return geometry


def source_frames(path):
    """Decode the original full-range Y plane and replicate it to RGB."""
    data.video_metadata(path)
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        for frame in container.decode(stream):
            if ((frame.width, frame.height) != (WIDTH, HEIGHT) or int(frame.color_range) != 2
                    or frame.format.name not in ('gray', 'gray8', 'yuvj420p', 'yuvj444p', 'yuv420p', 'yuv444p')):
                raise ValueError(f'Source format changed in {path}')
            plane = frame.planes[0]
            pixels = np.frombuffer(plane, np.uint8).reshape(HEIGHT, plane.line_size)[:, :WIDTH]
            frames.append(Image.fromarray(pixels).convert('RGB'))
    if len(frames) != FRAMES:
        raise ValueError(f'Expected {FRAMES} frames from {path}, decoded {len(frames)}')
    return frames


def completed_config(run):
    if run.name == validation_delivery.RUN_NAME:
        return validation_delivery.model_config(), validation_delivery.final_weights(run)
    cfg = data.read_json(run / 'cache_fullframe/config.json')
    done = data.read_json(run / 'training_complete.json')
    exit_info = data.read_json(run / 'exit.json')
    if (done['status'] != 'completed' or done['epochs'] != 3 or done['samples_consumed'] != data.TOTAL_SAMPLES
            or done['optimizer_updates'] != data.TOTAL_SAMPLES or exit_info['exit_code'] != 0
            or not exit_info['training_completed']):
        raise ValueError('Validation requires the completed three-epoch training run')
    for key in ('adapter', 'lora'):
        path = run / 'checkpoints' / (key + '_final.safetensors')
        if path.resolve() != Path(done[key + '_final']).resolve() or sha(path) != done[key + '_sha256']:
            raise ValueError(f'Final {key} checksum or location differs from the completion record')
    if (sha(run / 'config.json') != cfg['parent_config_sha256']
            or sha(run / 'prompt.txt') != cfg['prompt_file_sha256']
            or (run / 'prompt.txt').read_text() != cfg['prompt']):
        raise ValueError('Training config or prompt identity changed')
    for key, identity in cfg['models']['identities'].items():
        st = Path(cfg['models']['paths'][key]).stat()
        if st.st_size != identity['bytes'] or st.st_mtime_ns != identity['mtime_ns']:
            raise ValueError(f'Frozen base model changed: {key}')
    return cfg, done


def load_validation_model(run, *, release=None):
    import torch
    from safetensors.torch import load_file
    from integration import attach, attention_identity, increments, load_increments_state
    sys.path.append(str(ROOT / 'scripts'))
    from vace_p01 import load_local_vace

    cfg, done = completed_config(run) if release is None else release
    torch.set_num_threads(4)
    pipe = load_local_vace(cfg['models']['paths'], memory_mode='full')
    core = attach(pipe, cfg['seed'], checkpointing='none')
    attention_identity()
    saved = [load_file(done[role + '_final']) for role in ('adapter', 'lora')]
    load_increments_state(pipe, *saved)
    for expected, actual in zip(saved, increments(pipe)):
        if expected.keys() != actual.keys() or any(not torch.equal(v, actual[k]) for k, v in expected.items()):
            raise ValueError('Loaded VACE/Adapter increments disagree with saved tensors')
    pipe.eval().requires_grad_(False)
    return pipe, core, cfg


def predict_video(model, row, partial=None, speed=None, *, geometry=None, make_projection=None, on_step=None):
    """Return original model RGB frames. The only video read here is partial."""
    import torch
    from integration import model_fn_wan_video

    pipe, core, cfg = model
    if row['split_use'] != 'validation':
        raise ValueError('Only validation inputs are accepted')
    if partial is None:
        partial = source_frames(row['partial_video'])
    if geometry is None:
        geometry = validation_geometry(row)
    geometry = {k: torch.from_numpy(v)[None].to(pipe.device) for k, v in geometry.items()}
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        tokens = pipe.structure_adapter(geometry)
        projection = (pipe.structure_adapter.structure_proj if make_projection is None
                      else make_projection(tokens))
    if tuple(tokens.shape) != (1, 32760, 64) or not torch.isfinite(tokens).all().item():
        raise ValueError('Unexpected full-orbit geometry tokens')
    del geometry
    mask = tokens.new_ones((*tokens.shape[:2], 1))
    calls = 0

    def forward(**kwargs):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            result = model_fn_wan_video(**kwargs, structure_tokens=tokens, structure_mask=mask,
                                        structure_projection=projection)
        if not core.structure_last_call['injected_before_first_vace_block']:
            raise RuntimeError('The denoising model did not receive observed geometry')
        return result

    def conditioned(**kwargs):
        nonlocal calls
        result = forward(**kwargs) if speed is None else speed.forward(forward, kwargs)
        calls += 1
        return result

    def progress(timesteps):
        for n, timestep in enumerate(timesteps, 1):
            yield timestep
            if calls != 2 * n:
                raise RuntimeError('Expected identical geometry on both CFG denoising branches')
            print(f'building {row["id"]} sampling {n}/{STEPS}  [{"█" * n}{"░" * (STEPS - n)}]', flush=True)
            if on_step is not None:
                on_step(n)

    old_model_fn = pipe.model_fn
    pipe.model_fn = conditioned
    try:
        with torch.inference_mode():
            frames = pipe(prompt=cfg['prompt'], negative_prompt=cfg['negative_prompt'],
                          height=HEIGHT, width=WIDTH, num_frames=FRAMES, vace_video=partial,
                          vace_video_mask=[Image.new('RGB', (WIDTH, HEIGHT), 'white')] * FRAMES,
                          vace_scale=1., seed=SEED, rand_device='cpu', cfg_scale=5., cfg_merge=False,
                          num_inference_steps=STEPS, sigma_shift=5., tiled=False, progress_bar_cmd=progress)
    finally:
        pipe.model_fn = old_model_fn
    if calls != 2 * STEPS or len(frames) != FRAMES or any(f.size != (WIDTH, HEIGHT) or f.mode != 'RGB' for f in frames):
        raise ValueError('Incorrect denoising or output video shape')
    return frames


def save_comparison(output, building_id, partial, prediction, gt, comment):
    """Write a native-size comparison in 8-bit BT.709 H.264 for normal MP4 players."""
    canvas = Image.new('RGB', (WIDTH * 3, HEIGHT + HEADER), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('DejaVuSans.ttf', 18)
    for index, label in enumerate(('Input', 'Prediction', 'GT')):
        draw.text((index * WIDTH + 10, 4), f'{label}   |   Building {building_id}', fill='white', font=font)
    if not all(len(seq) == FRAMES for seq in (partial, prediction, gt)):
        raise ValueError('Panel frame counts differ from 81')

    def composite():
        for panels in zip(partial, prediction, gt):
            for index, frame in enumerate(panels):
                if frame.size != (WIDTH, HEIGHT) or frame.mode != 'RGB':
                    raise ValueError('Panel must be native-resolution RGB')
                canvas.paste(frame, (index * WIDTH, HEADER))
            yield canvas

    encode_rgb_video(output, composite(), canvas.size, comment)


def encode_rgb_video(output, frames, size, comment):
    """Use the same compatible MP4 format for previews and standalone prediction."""
    # faststart reopens its output by filename when moving the MP4 index to the front.
    with output.open('xb'):
        pass
    try:
        with av.open(str(output), mode='w', format='mp4', options={'movflags': '+faststart'}) as container:
            container.metadata['comment'] = comment
            stream = container.add_stream('libx264', rate=FPS)
            stream.width, stream.height = size
            stream.pix_fmt = 'yuv420p'
            stream.options = {'profile': 'high', 'crf': '10', 'preset': 'fast', 'threads': '3'}
            codec = stream.codec_context
            codec.colorspace, codec.color_range = Colorspace.ITU709, ColorRange.MPEG
            codec.color_primaries, codec.color_trc = ColorPrimaries.BT709, ColorTrc.BT709
            converter = VideoReformatter()
            count = 0
            for frame in frames:
                if frame.size != size or frame.mode != 'RGB':
                    raise ValueError('Unexpected output frame size or color mode')
                encoded_frame = converter.reformat(
                    av.VideoFrame.from_image(frame), format='yuv420p', dst_colorspace=Colorspace.ITU709,
                    src_color_range=ColorRange.JPEG, dst_color_range=ColorRange.MPEG)
                for packet in stream.encode(encoded_frame):
                    container.mux(packet)
                count += 1
            if count != FRAMES:
                raise ValueError(f'Expected {FRAMES} output frames, received {count}')
            for packet in stream.encode():
                container.mux(packet)
    except BaseException:
        output.unlink(missing_ok=True)
        raise


def render_building(run, building_id, output, model=None, speed_state=None):
    from validation_speed import ValidationSpeed
    from common import write_json
    if output.exists():
        raise FileExistsError(output)
    row = validation_row(run, building_id)
    partial = source_frames(row['partial_video'])
    if model is None:
        model = load_validation_model(run)
    # Earlier, already-running queues start a new worker per building. Share
    # their measured policy only inside that queue's temporary staging folder.
    scratch = output.parent / 'sampling-speed.json'
    queued_worker = speed_state is None and output.parent.name.startswith('.gt-v2-validation-')
    if speed_state is None:
        speed_state = data.read_json(scratch) if queued_worker and scratch.exists() else {}

    def save(policy):
        if queued_worker:
            write_json(scratch, policy)

    with ValidationSpeed(model[0], model[1], speed_state, save) as speed:
        if speed_state:
            print(f'采样配置 {speed_state["selected"]} · {speed_state["reason"]}', flush=True)
        predicted = predict_video(model, row, partial, speed)
    # GT is loaded only after the complete denoising trajectory has returned.
    if 'gt_video_sha256' in row and sha(row['GT_video']) != row['gt_video_sha256']:
        raise ValueError(f'Frozen GT video changed for {building_id}')
    gt = source_frames(row['GT_video'])
    done = (validation_delivery.final_weights(run) if run.name == validation_delivery.RUN_NAME
            else data.read_json(run / 'training_complete.json'))
    comment = '|'.join(('GT-V2', run.name, row['id'], done['adapter_sha256'], done['lora_sha256'],
                        '81f-30s-cfg5-seed20261018-triptych-v1'))
    save_comparison(output, row['id'], partial, predicted, gt, comment)
    print(f'Complete: building {row["id"]} ({FRAMES} frames)', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('building_id')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    render_building(args.run.resolve(), args.building_id, args.output)


if __name__ == '__main__':
    main()
