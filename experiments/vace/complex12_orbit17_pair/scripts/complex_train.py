"""One ordinary-C trainer for paired linear/circular windows and full-state resume."""
import argparse
import shutil
import time
from complex_common import *


def remaining_rows(rows, completed, stop_update):
    require(0 <= completed <= stop_update <= len(rows), 'Invalid absolute training target')
    require([int(r['update']) for r in rows] == list(range(1, len(rows) + 1)), 'Nonsequential updates')
    return rows[completed:stop_update]


def validate_history(rows, history, completed):
    require(len(history) == completed, 'Resume history length differs')
    for expected, actual in zip(rows, history):
        require(all(str(actual[k]) == str(v) for k, v in expected.items()), 'Resume sampling changed')


def cached_window(pipe, context, row):
    import torch
    from vace_roof_training import encode_sample
    base = legacy('complex_train')
    path, key = cache_path(row), cache_identity(row)
    if path.exists():
        value = torch.load(path, map_location='cpu', weights_only=True)
        require(value['identity'] == key, 'Condition cache identity mismatch')
    else:
        sample, masks, edges = training_sample(row)
        print(f'Encoding Rev1 / {row["cs"]} / ordered frames {sample["frame_ids"]}', flush=True)
        pipe.vae.to(pipe.device)
        encoded = encode_sample(pipe, sample, config()['training'], context, 1.0)
        encoded = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                   for k, v in encoded.items() if k != 'context'}
        pipe.vae.to('cpu')
        torch.cuda.empty_cache()
        value = dict(identity=key, inputs=encoded, masks=torch.from_numpy(masks[None]),
                     edges=torch.from_numpy(edges[None]))
        path.parent.mkdir(parents=True, exist_ok=True)
        base.atomic_torch_save(value, path)
    encoded = value['inputs']
    require(tuple(encoded['input_latents'].shape) == (1, 16, 5, 64, 64), 'Unexpected target latent')
    require(tuple(encoded['vace_context'].shape) == (1, 96, 5, 64, 64), 'Unexpected condition latent')
    require(tuple(value['masks'].shape) == tuple(value['edges'].shape) == (1, 5, 64, 64), 'Unexpected loss grid')
    encoded = {k: v.to(pipe.device) if isinstance(v, torch.Tensor) else v for k, v in encoded.items()}
    encoded['context'] = context
    return encoded, value['masks'].to(pipe.device), value['edges'].to(pipe.device)


def train(method, stop_update):
    import torch
    from safetensors.torch import load_file, save_file
    from vace_roof_training import make_optimizer
    from tqdm import tqdm
    require(method in METHODS and stop_update in (300, 600), 'Unapproved training task')
    base, c, rows = legacy('complex_train'), config(), sampling(method)
    out = RUN / 'checkpoints' / method
    begin = time.monotonic()
    pipe = base.new_pipe()
    initial = Path(c['training']['initialization'])
    print(f'ACTUAL shared initialization: {initial}; method={method}; fresh AdamW before optional full resume', flush=True)
    base.restore_parameters(pipe, load_file(str(initial)))
    optimizer = make_optimizer(pipe, c['training'])
    require(not optimizer.state, 'Optimizer must start fresh')
    context = base.text_context(pipe)
    completed, history, previous_elapsed = 0, [], 0.0
    resume = out / 'resume_latest.pt'
    if resume.exists():
        state = torch.load(resume, map_location='cpu', weights_only=False)
        require(state['config_id'] == c['config_id'] and state['method'] == method and
                state['initialization_path'] == str(initial), 'Resume identity differs')
        completed, history = state['actual_updates'], state['history']
        validate_history(rows, history, completed)
        base.restore_parameters(pipe, state['adapter'])
        optimizer.load_state_dict(state['optimizer'])
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        previous_elapsed = float(state['training_elapsed_seconds'])
        print(f'Resuming {method} after {completed}/600 actual updates; stage target={stop_update}', flush=True)

    def save_resume(milestone=False):
        state = dict(config_id=c['config_id'], method=method, actual_updates=completed,
                     initialization_path=str(initial), adapter=base.parameter_state(pipe),
                     optimizer=optimizer.state_dict(), history=history,
                     python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                     torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                     training_elapsed_seconds=previous_elapsed + time.monotonic() - begin)
        base.atomic_torch_save(state, resume)
        if milestone:
            shutil.copyfile(resume, out / 'resume_step300.pt')

    def save_adapter(step):
        steps = [int(s['step']) for s in optimizer.state.values()]
        require(steps and min(steps) == max(steps) == step, 'AdamW parameter steps differ from recorded updates')
        path = checkpoint_path(method, step)
        temporary = path.with_suffix('.writing.safetensors')
        save_file(base.parameter_state(pipe), str(temporary), metadata={
            'method': method, 'optimizer_updates': str(step), 'config_id': c['config_id'],
            'base_model': 'vace', 'initialization_path': str(initial)})
        os.replace(temporary, path)
        print(f'SAVED {method} checkpoint update {step}: {path}', flush=True)

    fields = list(rows[0]) + ['loss', 'grad_norm', 'time_weight', 'timestep', 'parameters_changed', 'lr', 'elapsed_seconds']
    loss_path = RUN / 'metrics' / f'train_{method}_loss.csv'
    write_csv(loss_path, history, fields)
    pending = remaining_rows(rows, completed, stop_update)
    with loss_path.open('a', newline='', encoding='utf-8') as log:
        writer = csv.DictWriter(log, fieldnames=fields)
        bar = tqdm(total=600, initial=completed, desc=f'{method} actual AdamW updates', file=sys.stdout, dynamic_ncols=True)
        try:
            for row in pending:
                inputs, masks, edges = cached_window(pipe, context, row)
                result = base.optimize(pipe, optimizer, inputs, masks, edges, 'C', row)
                completed = int(row['update'])
                entry = {**row, **result, 'lr': optimizer.param_groups[0]['lr'],
                         'elapsed_seconds': round(previous_elapsed + time.monotonic() - begin, 3)}
                history.append(entry); writer.writerow(entry); log.flush()
                live = read_json(RUN / 'progress.json')
                live['actual_updates'][method] = completed
                live.update(current_method=method, current_building=row['cs'], data_version='Rev1',
                            current_frame_ids=json.loads(row['frame_ids']), updated_at=now())
                write_json(RUN / 'progress.json', live)
                bar.set_postfix(cs=row['cs'], start=row['start'], loss=f'{result["loss"]:.6f}')
                bar.update(1)
                if completed % 10 == 0:
                    save_resume(milestone=completed == 300)
                    print(f'{method} {completed}/600; elapsed={entry["elapsed_seconds"]:.1f}s; full resume saved', flush=True)
                if completed in (300, 600):
                    save_adapter(completed)
                del inputs, masks, edges
        finally:
            # A failed/partial optimizer step must not replace the last valid full state.
            bar.close()
    require(completed == len(history) == stop_update, 'Incomplete training stage')
    if not checkpoint_path(method, completed).exists():
        save_adapter(completed)
    zero = sum(float(r['time_weight']) == 0 for r in history)
    steps = [int(s['step']) for s in optimizer.state.values()]
    require(min(steps) == max(steps) == stop_update, 'Optimizer counter differs at stage completion')
    mark_complete(f'train_{method}_{stop_update}', method=method, actual_updates=completed,
                  zero_weight_updates=zero, building_updates=dict(Counter(r['cs'] for r in history)),
                  optimizer_parameter_step_range=[min(steps), max(steps)], initialization_path=str(initial),
                  weight_path=str(checkpoint_path(method, completed)), elapsed_seconds=time.monotonic()-begin,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    print(f'TRAIN STAGE COMPLETE: {method} {completed}/600; zero scheduler weights={zero}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('method', choices=METHODS)
    parser.add_argument('--stop-update', type=int, choices=(300, 600), required=True)
    args = parser.parse_args()
    train(args.method, args.stop_update)
