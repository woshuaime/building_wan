"""CPU preparation of the fixed 600/600 paired experiment."""
import copy
import shutil
import subprocess
from complex_common import *


def write_coverage(tables, visible):
    from PIL import ImageDraw, ImageFont
    records, arrays = [], {}
    for budget in (300, 600):
        for method in METHODS:
            counts = {cs: np.zeros((80, 17), dtype=np.int64) for cs in TRAIN}
            for row in tables[method][:budget]:
                for local, frame in enumerate(json.loads(row['frame_ids'])):
                    counts[row['cs']][frame, local] += 1
            for cs in TRAIN:
                arrays[budget, method, cs] = counts[cs]
                for frame, values in enumerate(counts[cs]):
                    total = int(values.sum())
                    records.append(dict(budget=budget, method=method, cs=cs, frame=f'{frame:03}',
                                        occurrences=total, visible_occurrences=total if visible[cs][frame] else 0,
                                        **{f'local_{j:02}': int(v) for j, v in enumerate(values)}))
    write_csv(RUN / 'metrics/sampling_coverage.csv', records)
    maximum = max(int(x.max()) for x in arrays.values())
    canvas = Image.new('RGB', (2192, 1760), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
    for row, cs in enumerate(TRAIN):
        for col, (budget, method) in enumerate([(300, m) for m in METHODS] + [(600, m) for m in METHODS]):
            x, y = col*548+40, row*210+32
            values = arrays[budget, method, cs].T.astype(np.float64) / maximum
            white, blue = np.array([255, 255, 255]), np.array([25, 72, 145])
            colors = (white + values[..., None]*(blue-white)).round().astype(np.uint8)
            canvas.paste(Image.fromarray(colors).resize((480, 136), Image.Resampling.NEAREST), (x, y))
            draw.rectangle((x-1, y-1, x+480, y+136), outline='gray')
            draw.text((x, y-26), f'{cs} / {method} / {budget}', fill='black', font=font)
            for frame in (0, 20, 40, 60, 79):
                draw.text((x+frame*6-8, y+140), str(frame), fill='black', font=font)
            for local in (0, 8, 16):
                draw.text((x-32, y+local*8-3), str(local), fill='black', font=font)
    draw.text((32, 1694), f'X: original frame 000-079 | Y: local position 0-16 | White=0, blue={maximum} occurrences', fill='black', font=font)
    draw.text((32, 1724), 'Common color scale. Counts are frame exposures, not nonzero gradient contributions. Visibility counts: sampling_coverage.csv', fill='black', font=font)
    canvas.save(RUN / '00_看结果/采样覆盖总览.png')


def main():
    if (RUN / 'progress.json').exists():
        require(config()['schema'] == 'complex12.orbit17.paired.v1', 'Existing run has another scheme')
        for method in METHODS:
            sampling(method)
        print(f'REUSE prepared run {RUN}', flush=True)
        return
    old = read_json(REV1_RUN / 'resolved_config.json')
    roots = {'Rev1': str(PROJECT / 'data/buildingfiller_import/Complex12/BuildingFiller_Complex12_512_Rev1_20261022'),
             'OriginalHard': str(HARD_RUN / 'input_data/BuildingFiller_Complex12_OriginalHard_512_20261025')}
    initial = REV1_RUN / 'checkpoints/shared_initialization.safetensors'
    required = [*old['model']['paths'].values(), initial, *roots.values(), REV1_RUN / 'shared_sampling_table.csv']
    require(all(Path(p).exists() and os.access(p, os.R_OK) for p in required), 'Required source unavailable')
    require(old['split'] == {'train': TRAIN, 'val': VAL}, 'Original split differs')
    require(shutil.disk_usage(PROJECT).free > 20 * 1024**3, 'Insufficient disk space')
    visible, eligible = {}, {m: {} for m in METHODS}
    idx = {s['display_id']: s for s in read_json(Path(roots['Rev1']) / 'dataset_index.json')['samples']}
    for cs in TRAIN:
        require(idx[cs]['split'] == 'train', 'Validation building in training')
        visible[cs] = [bool(np.asarray(read_image(Path(roots['Rev1']) / idx[cs]['exact'] / f'{f:03}.png', 'L')).any())
                       for f in range(80)]
        for method in METHODS:
            eligible[method][cs] = eligible_starts(visible[cs], method)
            require(eligible[method][cs], f'No eligible windows for {method}/{cs}')
        print(f'{cs}: linear={len(eligible["C_lin17"][cs])}, circular={len(eligible["C_ring17"][cs])}', flush=True)
    tables = paired_tables(read_csv(REV1_RUN / 'shared_sampling_table.csv'), eligible)
    for directory in ('logs', 'cache', 'metrics', 'predictions', '00_看结果', 'tmp'):
        (RUN / directory).mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        (RUN / 'checkpoints' / method).mkdir(parents=True, exist_ok=True)
        write_csv(RUN / f'sampling_{method}.csv', tables[method])
    from vace_p01_environment import _nvml_snapshot
    c = {k: copy.deepcopy(old[k]) for k in ('model', 'training', 'inference', 'evaluation', 'split', 'samples')}
    c.update(schema='complex12.orbit17.paired.v1', run_root=str(RUN), created_at=now(),
             settings_source=str(REV1_RUN / 'resolved_config.json'), data_roots=roots,
             source_runs={'Rev1': str(REV1_RUN), 'OriginalHard': str(HARD_RUN)},
             methods=list(METHODS), expected_trajectories=40, expected_raw_pngs=3240, expected_videos=48,
             sampling_ids={m: identity(tables[m]) for m in METHODS}, eligible_starts=eligible,
             sampling_seeds={'extension': 20261027, 'circular': 20261028},
             environment=dict(python=PYTHON, project_git_commit_at_preparation=subprocess.check_output(
                 ['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip(), gpu=_nvml_snapshot()),
             report_policy='Automatic metrics/media; human visual review and upload report after user follow-up')
    c['training'].update(max_updates=600, save_updates=[300, 600], initialization=str(initial),
                         sample_order='sampling_<method>.csv',
                         loss='ordinary C: original FlowMatch all-latent MSE and scheduler time weight',
                         sampling='paired non-window fields; eligible linear/circular windows',
                         frozen_encoding_cache='new run: version + cs + ordered frame_ids + encoding config')
    c['evaluation'].update(methods=['C_old', *METHODS], video_layout='2x3 Before/GT/C_old/lin600/ring600/info')
    c['encoding_config_id'] = identity(dict(model=c['model'], prompt=c['inference']['prompt'],
        training={k: c['training'][k] for k in ('tiled', 'tile_size', 'tile_stride', 'backbone_dtype', 'window_frames')},
        preprocessing='Exact binary -> RGB mask; RGB127; original official temporal mapping'))
    c['config_id'] = identity(c)
    write_json(RUN / 'resolved_config.json', c)
    for method in METHODS:
        sampling(method)
    for version, cs in full_inputs():
        require(index(version)[cs]['split'] == ('train' if cs in TRAIN else 'val'), 'Dataset split mismatch')
        for role in ('gt', 'damaged', 'exact', 'structure_raw'):
            for frame in range(81):
                require(source_path(cs, role, frame, version).is_file(), f'Missing {version}/{cs}/{role}/{frame}')
            source(cs, role, 0, version)
        require(all((prediction_dir('C_old', 300, version, cs) / f'{f:03}.png').is_file() for f in range(81)),
                'Old C raw predictions missing')
    # One actual crossing is read synchronously, without model loading.
    crossing = dict(tables['C_ring17'][0], cs='CS10', start='72', frame_ids=json.dumps(window_ids('C_ring17', 72)))
    sample, masks, _ = training_sample(crossing)
    require(sample['frame_ids'] == list(range(72, 80)) + list(range(9)) and masks.shape == (5, 64, 64),
            'Cross-boundary data assembly failed')
    write_coverage(tables, visible)
    script = f'''#!/usr/bin/env bash
set -euo pipefail
export ORBIT17_RUN_ROOT='{RUN}'
export WAN_PYTHON='{PYTHON}'
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
if [[ $# -gt 0 ]]; then
  case "$1" in
    --run|--resume) exec '{PYTHON}' -u '{SCRIPTS}/run_pilot.py' ;;
    --status) ;;
    *) echo 'Use no arguments for status, --run or --resume for the authorized queue'; exit 2 ;;
  esac
fi
exec '{PYTHON}' '{SCRIPTS}/status.py'
'''
    (RUN / 'resume.sh').write_text(script, encoding='utf-8')
    (RUN / 'resume.sh').chmod(0o755)
    for directory, target in [('runs', RUN), ('configs', RUN / 'resolved_config.json')]:
        folder = SCRIPTS.parent / directory
        folder.mkdir(exist_ok=True)
        link = folder / (RUN.name + ('.json' if directory == 'configs' else ''))
        if not link.exists():
            link.symlink_to(target, target_is_directory=directory == 'runs')
    from complex_evaluate import write_entry
    write_entry(False)
    write_json(RUN / 'progress.json', dict(config_id=c['config_id'], status='prepared', stage='配对表与CPU准备完成',
        actual_updates={m: 0 for m in METHODS}, completed_trajectories=0, expected_trajectories=40, updated_at=now()))
    print(f'PREPARED {RUN}: two groups x600, 40 trajectories, 3240 PNGs, 48 videos', flush=True)


if __name__ == '__main__':
    main()
