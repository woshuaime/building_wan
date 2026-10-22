"""One-time CPU preparation; never starts GPU work."""
import collections
import copy
import importlib.metadata
import platform
import random
import subprocess
from complex_common import *

def main():
    require(not (RUN / 'resolved_config.json').exists(), 'Already prepared; use resume.sh')
    print('数据适配：Complex12 Rev1; existing package verification reused', flush=True)
    for name in ('00_看结果', 'checkpoints', 'checks', 'metrics', 'logs', 'training', 'cache', 'tmp'):
        (RUN / name).mkdir(exist_ok=True)
    old_log = RUN / 'logs/verify_package.log'
    if old_log.exists():
        old_log.rename(RUN / 'logs/package_verify.log')
    idx = index()
    require([x for x in IDS if idx[x]['split'] == 'train'] == TRAIN, 'Training split differs')
    require([x for x in IDS if idx[x]['split'] == 'val'] == VAL, 'Validation split differs')
    acceptance = read_json(DATA / 'human_acceptance.json')
    require(acceptance['human_QA'] == 'accepted' and acceptance['revision'] == 'Rev1', 'Not accepted Rev1')
    eligible, coverage = {}, []
    for cs in IDS:
        m, e = labels(cs, 0, 80)
        starts = [s for s in range(64) if m[s:s+17].any()]
        eligible[cs] = starts
        lm = [loss_grid(m[s:s+17]) for s in starts]
        le = [loss_grid(e[s:s+17]) for s in starts]
        coverage.append(dict(cs=cs, split=idx[cs]['split'], source_id=idx[cs]['source_id'],
            pixel_M_coverage=float(m.mean()), pixel_E_coverage=float(e.mean()),
            loss_M_coverage=float(np.mean(lm)), loss_E_coverage=float(np.mean(le)),
            eligible_windows=len(starts), E_empty_window_fraction=float(np.mean([not x.any() for x in le])),
            black_exact_frames=' '.join(f'{i:03}' for i in range(80) if not m[i].any())))
        print(f'{cs}: {len(starts)} eligible windows; Exact={m.mean():.5f}, E={e.mean():.5f}', flush=True)
    write_csv(RUN / 'metrics/mask_coverage.csv', coverage)
    write_json(RUN / 'checks/eligible_windows.json', eligible)
    rng = random.Random(20261022)
    sampling = []
    for update in range(1, 301):
        cs = rng.choice(TRAIN)
        sampling.append(dict(update=update, cs=cs, start=rng.choice(eligible[cs]),
            timestep_index=rng.randrange(1000), noise_seed=rng.randrange(2**31), stochastic_seed=rng.randrange(2**31)))
    write_csv(RUN / 'shared_sampling_table.csv', sampling)
    from vace_roof_data import inspect_local_models
    from vace_p01_environment import _nvml_snapshot
    import torch
    original = read_json(PROJECT / 'configs/vace_roof_smoke_v1.json')
    paths = inspect_local_models(original)
    c = {k: copy.deepcopy(original[k]) for k in ('model', 'training', 'inference')}
    c['schema'] = 'complex12.rev1.paired.v1'
    c['run_root'], c['data_root'] = str(RUN), str(DATA)
    c['model']['paths'] = paths
    c['settings_source'] = str(PROJECT / 'configs/vace_roof_smoke_v1.json')
    c['training'].update(seed=20261022, sample_order='shared_sampling_table.csv', save_updates=[100, 200, 300],
        window_frames=17, initialization=str(RUN / 'checkpoints/shared_initialization.safetensors'),
        sampling='uniform building then uniform nonempty-Exact window within frames 000-079',
        frozen_encoding_cache='per-window GT VAE latent and Damaged/ModelMask VAE context, shared text; disk CPU cache in RUN_ROOT/cache',
        loss='original FlowMatch noise-minus-latent target and scheduler time weight; C/U/S spatial weights only')
    c['inference'].update(num_frames=81, prompt='A complete untextured gray building with continuous roof surfaces and walls, viewed during a smooth 360-degree orbit against a plain light-gray background.')
    c['evaluation'] = {'lora_multiplier': 1.0, 'methods': ['B','L','C','U','S'], 'unique_frames': list(range(80)),
        'closure_frames': [0,80], 'rgb_range': [0,1], 'rgb_source': 'raw quantized pipeline PNG, not video decoding',
        'gray': [0.299,0.587,0.114], 'sobel_kernel': [[-1,0,1],[-2,0,2],[-1,0,1]],
        'sobel_divisor': 8, 'sobel_padding': 'numpy reflect', 'sobel_error': '(abs(Gx_pred-Gx_gt)+abs(Gy_pred-Gy_gt))/2 in E_pixel',
        'within_building': 'sum errors / valid pixels per metric; PSNR from aggregated MSE',
        'across_buildings': 'equal building means, separate train and val', 'empty_mask': 'N/A plus valid counts',
        'zero_mse_psnr': 'Infinity', 'composite': False, 'force_loop': False,
        'video_layout': '2 rows x 4 columns; Before, GT, B, L / C, U, S, blank; each content 512x512',
        'keyframes': [0,20,40,60,80], 'inference_GT_inputs': False}
    c['loss_mapping'] = {'dilation': '7x7 max on raw_visible then intersect Exact', 'temporal_groups': TEMPORAL_GROUPS,
        'spatial_pool': '8x8 max', 'shape': [1,5,64,64], 'S': '1+2E', 'U': '1+(2*sum(E)/sum(M))*M',
        'empty': 'original unweighted loss if M or E empty', 'model_mask': 'official unmodified VACE mapping, distinct from max-pooled loss labels'}
    c['split'] = {'train': TRAIN, 'val': VAL}
    c['samples'] = [idx[cs] for cs in IDS]
    old_weight = PROJECT / 'outputs/vace/lora_roof_smoke/run_20261020_165237_150325_2048720/training/lora_update0300.safetensors'
    require(old_weight.is_file(), 'Expected reference adapter is absent')
    c['legacy_lora'] = {'path': str(old_weight), 'sha256': sha(old_weight), 'bytes': old_weight.stat().st_size,
        'branch': 'vace', 'scale': 1.0, 'source': 'Pilot8 P04/source1227 and P06/source1848, B000-B016, 300 updates, seed20261018',
        'use': 'reference only; never initializes C/U/S', 'original_positive_prompt': original['inference']['prompt']}
    c['base_identity'] = {'model_sources_record': str(PROJECT / 'configs/vace_p01_model_sources_v1.json'),
        'current_check': 'file sizes and VACE header verified; prior approved full-model hashes not recalculated',
        'files': {k: {'path':v, 'bytes':Path(v).stat().st_size} for k,v in paths.items() if k != 'tokenizer'}}
    c['package'] = {'sha256_verified_at_extraction': '8fd043cd5826488593c29eae3f19ee271daa998dab1bcdd30b7fc8d4d291096e',
        'verification_log': str(RUN / 'logs/package_verify.log'), 'human_acceptance': acceptance}
    c['environment'] = {'python': PYTHON, 'python_version': platform.python_version(), 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'diffsynth': importlib.metadata.version('diffsynth'),
        'peft': importlib.metadata.version('peft'), 'gpu': _nvml_snapshot(), 'cpu_threads': 4,
        'project_git_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
        'engine_files_sha256': {f:sha(PROJECT/f) for f in ['code/DiffSynth-Studio/diffsynth/diffusion/loss.py',
            'code/DiffSynth-Studio/diffsynth/pipelines/wan_video.py','code/DiffSynth-Studio/diffsynth/models/wan_video_vae.py']}}
    c['paired_sampling_sha256'] = sha(RUN / 'shared_sampling_table.csv')
    c['created_at'] = now()
    c['config_id'] = hashlib.sha256(json.dumps(c, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    write_json(RUN / 'resolved_config.json', c)
    write_json(RUN / 'progress.json', {'config_id':c['config_id'],'status':'prepared','stage':'CPU checks',
        'completed_trajectories':0,'expected_trajectories':60,'actual_updates':{'C':0,'U':0,'S':0},'updated_at':now()})
    print('Frozen configuration:', RUN / 'resolved_config.json', flush=True)
    print('Shared 300-update sampling:', dict(collections.Counter(r['cs'] for r in sampling)), flush=True)

if __name__ == '__main__':
    main()
