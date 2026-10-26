"""Prepare the single authorized run; no model loading or preflight training."""
import shutil
from complex_common import *

def main():
    if (RUN / 'resolved_config.json').exists():
        require(config()['schema'] == 'complex12.cmix.fixed.v1', 'Existing run has another scheme')
        sampling()
        print(f'REUSE existing C_mix run: {RUN}', flush=True)
        return
    c = read_json(REV1_RUN / 'resolved_config.json')
    c.pop('config_id')
    for key in ('legacy_lora', 'paired_sampling_sha256'):
        c.pop(key, None)
    c.update(schema='complex12.cmix.fixed.v1', run_root=str(RUN), created_at=now(),
             source_runs={'Rev1': str(REV1_RUN), 'OriginalHard': str(HARD_RUN)},
             data_roots={'Rev1': str(PROJECT / 'data/buildingfiller_import/Complex12/BuildingFiller_Complex12_512_Rev1_20261022'),
                         'OriginalHard': str(HARD_RUN / 'input_data/BuildingFiller_Complex12_OriginalHard_512_20261025')},
             mix={'selection_seed': 20261026, 'rev1_updates': 250, 'hard_updates': 50,
                  'hard_counts': {'CS09': 20, 'CS10': 14, 'CS11': 16}},
             expected_trajectories=16, expected_raw_pngs=1296)
    c.pop('data_root', None)
    c['training']['initialization'] = str(REV1_RUN / 'checkpoints/shared_initialization.safetensors')
    c['training']['frozen_encoding_cache'] = 'new-run cache keyed by data_version + cs + start; same 17-frame encoding'
    c['evaluation']['methods'] = ['C_old', 'C_mix']
    c['evaluation']['video_layout'] = '2x2 Before/GT/old C/C_mix; 512x512 content plus title bars'
    c['config_id'] = config_id(c)
    require(c['split']['train'] == TRAIN and c['split']['val'] == VAL, 'Source split changed')
    paths = [*c['model']['paths'].values(), c['training']['initialization'], *c['data_roots'].values()]
    require(all(Path(p).exists() and os.access(p, os.R_OK) for p in paths), 'Required source is unreadable')
    require(shutil.disk_usage(PROJECT).free > 12 * 1024**3, 'Insufficient free disk for this run')
    rows = mixed_sampling(read_csv(REV1_RUN / 'shared_sampling_table.csv'))
    for folder in ('logs', 'cache', 'checkpoints/C_mix', 'training', 'metrics', 'predictions', '00_看结果', 'tmp'):
        (RUN / folder).mkdir(parents=True, exist_ok=True)
    write_json(RUN / 'resolved_config.json', c)
    write_csv(RUN / 'shared_sampling_table.csv', rows)
    sampling()
    # Check the actual evaluation inputs and baseline filenames; no dataset re-audit.
    for version, cs in trajectory_order():
        require(index(version)[cs]['split'] == ('train' if cs in TRAIN else 'val'), 'Dataset split mismatch')
        for role in ('gt', 'damaged', 'exact', 'structure_raw'):
            for frame in range(81):
                require(source_path(cs, role, frame, version).is_file(), f'Missing {version}/{cs}/{role}/{frame}')
            source(cs, role, 0, version)
        for frame in range(81):
            require((prediction_dir(version, cs, old=True) / f'{frame:03}.png').is_file(), 'Old C prediction missing')
    write_json(RUN / 'progress.json', {'config_id': c['config_id'], 'status': 'prepared',
        'stage': '准备完成', 'actual_updates': {'C_mix': 0}, 'completed_trajectories': 0,
        'expected_trajectories': 16, 'updated_at': now()})
    script = f'''#!/usr/bin/env bash
set -euo pipefail
export CMIX_RUN_ROOT='{RUN}'
export WAN_PYTHON='{PYTHON}'
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
if [[ "${{1:-}}" == "--run" || "${{1:-}}" == "--resume" ]]; then
  exec '{PYTHON}' -u '{SCRIPTS}/run_pilot.py'
fi
exec '{PYTHON}' '{SCRIPTS}/status.py'
'''
    (RUN / 'resume.sh').write_text(script, encoding='utf-8')
    (RUN / 'resume.sh').chmod(0o755)
    for folder, target in [('runs', RUN), ('configs', RUN / 'resolved_config.json')]:
        base = SCRIPTS.parent / folder
        base.mkdir(exist_ok=True)
        (base / (RUN.name + ('.json' if folder == 'configs' else ''))).symlink_to(target, target_is_directory=folder == 'runs')
    print(f'Prepared {RUN}; 300 updates: Rev1=250, OriginalHard=50; 16 trajectories', flush=True)

if __name__ == '__main__':
    main()
