"""Ordinary C from the shared initialization; only data-version allocation changes."""
import time
from complex_common import *

def cached_window(pipe, context, row):
    import torch
    from vace_roof_training import encode_sample
    base = legacy('complex_train')
    version, cs, start = row['data_version'], row['cs'], int(row['start'])
    path = RUN / 'cache' / version / f'{cs}_{start:03}_17.pt'
    path.parent.mkdir(exist_ok=True)
    sample, m, e = training_sample(cs, start, version)
    identity = {'config_id': config()['config_id'], 'data_version': version, 'cs': cs, 'start': start}
    if path.exists():
        value = torch.load(path, map_location='cpu', weights_only=True)
        require(all(value[k] == v for k, v in identity.items()), 'Condition cache identity mismatch')
        encoded = value['inputs']
    else:
        print(f'Encoding {version} / {cs} / B{start:03}-B{start+16:03}', flush=True)
        pipe.vae.to(pipe.device)
        encoded = encode_sample(pipe, sample, config()['training'], context, 1.0)
        encoded = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in encoded.items() if k != 'context'}
        pipe.vae.to('cpu')
        torch.cuda.empty_cache()
        base.atomic_torch_save(dict(inputs=encoded, **identity), path)
    require(tuple(encoded['input_latents'].shape) == (1, 16, 5, 64, 64), 'Unexpected target latent')
    require(tuple(encoded['vace_context'].shape) == (1, 96, 5, 64, 64), 'Unexpected condition latent')
    encoded = {k: v.to(pipe.device) if isinstance(v, torch.Tensor) else v for k, v in encoded.items()}
    encoded['context'] = context
    return encoded, torch.from_numpy(m[None]).to(pipe.device), torch.from_numpy(e[None]).to(pipe.device)

def train():
    import torch
    from safetensors.torch import load_file
    from vace_roof_training import make_optimizer
    from tqdm import tqdm
    base = legacy('complex_train')
    begin = time.monotonic()
    c = config()
    out = RUN / 'checkpoints/C_mix'
    pipe = base.new_pipe()
    initial = Path(c['training']['initialization'])
    print(f'ACTUAL shared initialization: {initial}; fresh AdamW; C_mix only', flush=True)
    base.restore_parameters(pipe, load_file(str(initial)))
    optimizer = make_optimizer(pipe, c['training'])
    require(not optimizer.state, 'Expected fresh optimizer before optional full-state restore')
    context = base.text_context(pipe)
    history = []
    completed = 0
    resume = out / 'resume_latest.pt'
    rows = sampling()
    if resume.exists():
        state = torch.load(resume, map_location='cpu', weights_only=False)
        require(state['config_id'] == c['config_id'] and state['initialization_path'] == str(initial), 'Resume identity mismatch')
        base.restore_parameters(pipe, state['adapter'])
        optimizer.load_state_dict(state['optimizer'])
        completed, history = state['actual_updates'], state['history']
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        require(len(history) == completed, 'Resume history mismatch')
        for expected, actual in zip(rows, history):
            require(all(str(actual[k]) == str(v) for k, v in expected.items()), 'Resume sampling changed')
        print(f'Resuming C_mix after optimizer update {completed}/300', flush=True)
    def save_resume(milestone=False):
        state = {'config_id': c['config_id'], 'actual_updates': completed, 'initialization_path': str(initial),
                 'adapter': base.parameter_state(pipe), 'optimizer': optimizer.state_dict(), 'history': history,
                 'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}
        base.atomic_torch_save(state, resume)
        if milestone:
            import shutil
            shutil.copyfile(resume, out / f'resume_step{completed}.pt')
    fields = list(rows[0]) + ['loss', 'grad_norm', 'time_weight', 'timestep', 'parameters_changed', 'lr', 'elapsed_seconds']
    logpath = RUN / 'metrics/train_C_mix_loss.csv'
    write_csv(logpath, history, fields)
    with logpath.open('a', newline='', encoding='utf-8') as log:
        writer = csv.DictWriter(log, fieldnames=fields)
        bar = tqdm(total=300, initial=completed, desc='C_mix actual AdamW updates', file=sys.stdout, dynamic_ncols=True)
        try:
            for row in rows[completed:]:
                inputs, m, e = cached_window(pipe, context, row)
                result = base.optimize(pipe, optimizer, inputs, m, e, 'C', row)
                completed = int(row['update'])
                entry = {**row, **result, 'lr': optimizer.param_groups[0]['lr'], 'elapsed_seconds': round(time.monotonic()-begin, 3)}
                history.append(entry); writer.writerow(entry); log.flush()
                live = read_json(RUN / 'progress.json')
                live.update(actual_updates={'C_mix': completed}, current_building=row['cs'], data_version=row['data_version'],
                            current_window_start=int(row['start']), updated_at=now())
                write_json(RUN / 'progress.json', live)
                bar.set_postfix(version=row['data_version'], cs=row['cs'], loss=f'{result["loss"]:.6f}'); bar.update(1)
                if completed % 10 == 0:
                    save_resume(milestone=completed in (100, 200))
                    print(f'C_mix {completed}/300; elapsed={time.monotonic()-begin:.1f}s; full resume saved', flush=True)
                del inputs, m, e
        except BaseException:
            save_resume()
            raise
        finally:
            bar.close()
    require(completed == len(history) == 300, 'Incomplete training budget')
    counts = Counter(r['data_version'] for r in history)
    require(counts == {'Rev1': 250, 'OriginalHard': 50}, 'Actual version allocation changed')
    base.save_adapter(pipe, out / 'step300.safetensors', 300)
    zero = {v: sum(r['data_version'] == v and float(r['time_weight']) == 0 for r in history) for v in VERSIONS}
    steps = [int(s['step']) for s in optimizer.state.values()]
    require(min(steps) == max(steps) == 300, 'Optimizer steps do not match budget')
    mark_complete('train_C_mix', actual_updates=300, version_updates=dict(counts), zero_weight_updates=zero,
                  optimizer_parameter_step_range=[min(steps), max(steps)], initialization_path=str(initial),
                  weight_path=str(out / 'step300.safetensors'), elapsed_seconds=time.monotonic()-begin,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    print(f'TRAIN COMPLETE: 300/300; versions={dict(counts)}; zero scheduler weights={zero}; {out / "step300.safetensors"}', flush=True)

if __name__ == '__main__':
    train()
