"""A fresh VACE Base plus one paired checkpoint, with original 81-frame conditions."""
import time
from complex_common import *


def infer(method, step, version, cs):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    from vace_p01 import load_local_vace
    from vace_roof_smoke import load_stage_lora, inference_arguments
    from vace_p01_81f_console import ConsoleState
    require((method, step, version, cs) in trajectory_order(), 'Unapproved trajectory')
    if valid_prediction(method, step, version, cs):
        print(f'REUSE {method} / {step} / {version} / {cs}', flush=True)
        return
    c, folder = config(), prediction_dir(method, step, version, cs)
    folder.mkdir(parents=True, exist_ok=True)
    ready_path, job = folder / 'raw_ready.json', infer_job(method, step, version, cs)
    if ready_path.exists():
        record = read_json(ready_path)
        require(record['config_id'] == c['config_id'] and record['method'] == method and
                record['checkpoint_step'] == step and record['data_version'] == version and record['cs'] == cs,
                'Raw output identity differs')
        for i in range(81):
            read_image(folder / f'{i:03}.png')
        mark_complete(job, **{k: v for k, v in record.items() if k != 'config_id'})
        return
    require(not list(folder.glob('*.png')), 'Partial trajectory retained; inspect before repeating GPU work')
    begin = time.monotonic()
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    path = checkpoint_path(method, step)
    with safe_open(path, framework='pt', device='cpu') as handle:
        meta = handle.metadata()
    require(meta['method'] == method and meta['optimizer_updates'] == str(step) and
            meta['config_id'] == c['config_id'], 'Wrong checkpoint')
    live = read_json(RUN / 'progress.json')
    live.update(current_method=method, checkpoint_step=step, data_version=version,
                current_building=cs, inference_steps=0, updated_at=now())
    write_json(RUN / 'progress.json', live)
    print(f'Running {method} / step{step} / {version} / {cs} / 81 frames', flush=True)
    print(f'ACTUAL Base absolute path: {c["model"]["paths"]["checkpoint"]}', flush=True)
    print(f'ACTUAL LoRA absolute path: {path}; branch=vace; scale=1.0', flush=True)
    sample = conditions(cs, range(81), version)
    pipe = load_local_vace(c['model']['paths'], memory_mode='full')
    state = load_file(str(path), device='cpu')
    key = next(n for n, t in state.items() if '.lora_B.' in n and bool(t.any()))
    layer = dict(pipe.vace.named_modules())[key.split('.lora_B.')[0]]
    before = layer.weight.detach().cpu().clone()
    load_stage_lora(pipe, path, c)
    require(not torch.equal(before, layer.weight.detach().cpu()), 'LoRA did not change VACE weights')
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
        require(frame.mode == 'RGB' and frame.size == (512, 512), 'Unexpected decoded frame')
        frame.save(folder / f'{i:03}.png')
    record = dict(config_id=c['config_id'], method=method, checkpoint_step=step, data_version=version, cs=cs,
                  frame_count=81, size=[512, 512], seed=20261018, steps=30,
                  lora=dict(path=str(path), branch='vace', scale=1.0, weight_changed=True),
                  raw_output='quantized decoded RGB; no composite; original frame080 retained',
                  elapsed_seconds=time.monotonic()-begin, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved())
    write_json(ready_path, record)
    for i in range(81):
        read_image(folder / f'{i:03}.png')
    mark_complete(job, **{k: v for k, v in record.items() if k != 'config_id'})
    print(f'SAVED {method}/{step}/{version}/{cs}: 81 raw PNGs; elapsed={time.monotonic()-begin:.1f}s; {folder}', flush=True)


if __name__ == '__main__':
    infer(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4])
