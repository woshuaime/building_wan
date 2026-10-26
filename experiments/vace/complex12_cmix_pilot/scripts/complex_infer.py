"""One fresh Base plus C_mix per 81-frame trajectory; no GT conditioning."""
import time
from complex_common import *

def infer(version, cs):
    import torch
    from safetensors.torch import load_file
    from vace_p01 import load_local_vace
    from vace_roof_smoke import load_stage_lora, inference_arguments
    from vace_p01_81f_console import ConsoleState
    require((version, cs) in trajectory_order(), 'Unapproved inference trajectory')
    if valid_prediction(version, cs):
        print(f'REUSE C_mix / {version} / {cs}', flush=True)
        return
    c = config()
    folder = prediction_dir(version, cs)
    folder.mkdir(parents=True, exist_ok=True)
    ready_path = folder / 'raw_ready.json'
    job = f'infer_{version}_{cs}'
    if ready_path.exists():
        record = read_json(ready_path)
        require(record['config_id'] == c['config_id'], 'Raw output configuration changed')
        for i in range(81):
            read_image(folder / f'{i:03}.png')
        mark_complete(job, **{k: v for k, v in record.items() if k != 'config_id'})
        return
    require(not list(folder.glob('*.png')), 'Partial raw output retained; inspect before repeating inference')
    begin = time.monotonic()
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    path = RUN / 'checkpoints/C_mix/step300.safetensors'
    require(read_json(completion_path('train_C_mix'))['actual_updates'] == 300, 'Training has not finished')
    live = read_json(RUN / 'progress.json')
    live.update(data_version=version, current_building=cs, inference_steps=0, updated_at=now())
    write_json(RUN / 'progress.json', live)
    print(f'Running C_mix / {version} / {cs} / 81 frames', flush=True)
    print(f'ACTUAL Base absolute path: {c["model"]["paths"]["checkpoint"]}', flush=True)
    print(f'ACTUAL LoRA absolute path: {path}; branch=vace; scale=1.0', flush=True)
    sample = conditions(cs, version=version)
    pipe = load_local_vace(c['model']['paths'], memory_mode='full')
    state = load_file(str(path), device='cpu')
    key = next(n for n, t in state.items() if '.lora_B.' in n and bool(t.any()))
    layer = dict(pipe.vace.named_modules())[key.split('.lora_B.')[0]]
    before = layer.weight.detach().cpu().clone()
    load_stage_lora(pipe, path, c)
    require(not torch.equal(before, layer.weight.detach().cpu()), 'C_mix did not change VACE weights')
    del state, before
    def progress_event(event):
        live = read_json(RUN / 'progress.json')
        live.update(updated_at=now(), inference_stage=event['stage'], inference_steps=event['completed_steps'],
                    inference_elapsed_seconds=round(event['elapsed'], 2))
        write_json(RUN / 'progress.json', live)
    console = ConsoleState(total_steps=30, event_sink=progress_event)
    with torch.inference_mode(), console.vae_stages(pipe.vae):
        output = pipe(**inference_arguments(c, sample), progress_bar_cmd=console.progress)
    require(len(output) == 81 and console.completed_steps == 30, 'Incomplete inference')
    for i, frame in enumerate(output):
        require(frame.mode == 'RGB' and frame.size == (512, 512), 'Unexpected decoded output')
        frame.save(folder / f'{i:03}.png')
    record = {'config_id': c['config_id'], 'data_version': version, 'cs': cs, 'method': 'C_mix',
              'frame_count': 81, 'size': [512, 512], 'seed': 20261018, 'steps': 30,
              'lora': {'path': str(path), 'branch': 'vace', 'scale': 1.0, 'weight_changed': True},
              'raw_output': 'quantized decoded RGB; no compositing; original frame080 retained',
              'elapsed_seconds': time.monotonic()-begin, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
    write_json(ready_path, record)
    for i in range(81):
        read_image(folder / f'{i:03}.png')
    mark_complete(job, **{k: v for k, v in record.items() if k != 'config_id'})
    print(f'SAVED {version}/{cs}: 81 raw PNGs; elapsed={time.monotonic()-begin:.1f}s; {folder}', flush=True)

if __name__ == '__main__':
    infer(sys.argv[1], sys.argv[2])
