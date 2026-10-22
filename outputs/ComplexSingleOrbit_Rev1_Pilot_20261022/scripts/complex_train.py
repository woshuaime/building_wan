"""Paired VACE LoRA training; only the error-grid weights differ."""
import csv
import random
import time
from complex_common import *

def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def weighted_error(pred, target, m, e, method, strength=2.0):
    import torch
    import torch.nn.functional as F
    require(method in ('C', 'U', 'S'), 'Unknown loss')
    require(m.shape == e.shape == pred.shape[:1] + pred.shape[2:], 'Loss-label grid mismatch')
    require(bool(((e >= 0) & (e <= m) & (m <= 1)).all()), 'Invalid E/M containment')
    original = F.mse_loss(pred.float(), target.float())
    if method == 'C' or strength == 0:
        return original
    error = (pred.float() - target.float()).square().mean(dim=1)
    sample_losses = []
    for i in range(pred.shape[0]):
        if not bool(m[i].any()) or not bool(e[i].any()):
            sample_losses.append(F.mse_loss(pred[i].float(), target[i].float()))
        else:
            weight = 1 + (strength * e[i] if method == 'S' else strength * e[i].sum() / m[i].sum() * m[i])
            sample_losses.append((weight * error[i]).sum() / weight.sum())
    return torch.stack(sample_losses).mean()

def flow_loss(pipe, inputs, m, e, method, row):
    import torch
    # Same original FlowMatch target, scheduler and BF16 timestep conversion.
    inputs = dict(inputs)
    index = int(row['timestep_index'])
    timestep = pipe.scheduler.timesteps[[index]].to(dtype=pipe.torch_dtype, device=pipe.device)
    generator = torch.Generator(device=inputs['input_latents'].device).manual_seed(int(row['noise_seed']))
    noise = torch.randn(inputs['input_latents'].shape, dtype=inputs['input_latents'].dtype,
                        device=inputs['input_latents'].device, generator=generator)
    inputs['latents'] = pipe.scheduler.add_noise(inputs['input_latents'], noise, timestep)
    target = pipe.scheduler.training_target(inputs['input_latents'], noise, timestep)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    pred = pipe.model_fn(**models, **inputs, timestep=timestep)
    error = weighted_error(pred, target, m, e, method)
    time_weight = pipe.scheduler.training_weight(timestep)
    return error * time_weight, float(time_weight), float(timestep)

def parameter_state(pipe):
    from vace_roof_training import trainable_lora_parameters
    return {n.removeprefix('vace.'): p.detach().cpu().contiguous().clone()
            for n, p in trainable_lora_parameters(pipe)}

def restore_parameters(pipe, state):
    import torch
    from vace_roof_training import trainable_lora_parameters
    params = {n.removeprefix('vace.'): p for n, p in trainable_lora_parameters(pipe)}
    require(set(params) == set(state), 'Adapter initialization keys do not match')
    with torch.no_grad():
        for name, p in params.items():
            p.copy_(state[name])
            require(torch.equal(p.detach().cpu(), state[name]), 'Initial adapter differs')

def save_adapter(pipe, path, update):
    from safetensors.torch import save_file
    temp = str(path) + '.tmp'
    save_file(parameter_state(pipe), temp, metadata={'base_model':'vace', 'optimizer_update':str(update),
        'checkpoint_type':'lora_delta_not_resume_state', 'config_id':config()['config_id']})
    os.replace(temp, path)

def atomic_torch_save(value, path):
    import torch
    temp = str(path) + '.tmp'
    torch.save(value, temp)
    os.replace(temp, path)

def new_pipe():
    import torch
    from vace_p01 import load_local_vace
    from vace_roof_training import configure_lora
    c = config()
    torch.set_num_threads(4)
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'CUDA BF16 required')
    torch.cuda.reset_peak_memory_stats()
    seed_all(c['training']['seed'])
    pipe = load_local_vace(c['model']['paths'], memory_mode='full')
    seed_all(c['training']['seed'])
    configure_lora(pipe, c['training'])
    return pipe

def text_context(pipe):
    import torch
    from diffsynth.pipelines.wan_video import WanVideoUnit_PromptEmbedder
    path = RUN / 'cache/text_context.pt'
    if path.exists():
        value = torch.load(path, map_location='cpu', weights_only=True)
        require(value['config_id'] == config()['config_id'], 'Text-cache config mismatch')
        context = value['context']
    else:
        with torch.no_grad():
            context = WanVideoUnit_PromptEmbedder().process(pipe, prompt=config()['inference']['prompt'], positive=True)['context'].cpu()
        atomic_torch_save({'context':context, 'config_id':config()['config_id']}, path)
    pipe.text_encoder.to('cpu')
    pipe.vae.to('cpu')
    torch.cuda.empty_cache()
    return context.to(pipe.device)

def cached_window(pipe, context, cs, start, inspect=False):
    import torch
    import torch.nn.functional as F
    from vace_roof_training import encode_sample
    from einops import rearrange
    path = RUN / 'cache' / f'{cs}_{start:03}_17.pt'
    sample, m, e = training_sample(cs, start)
    if path.exists():
        value = torch.load(path, map_location='cpu', weights_only=True)
        require(value['config_id'] == config()['config_id'], 'Window-cache config mismatch')
        encoded = value['inputs']
    else:
        print(f'Encoding contiguous {cs} B{start:03}-B{start+16:03}; GT target and Damaged/Mask separately', flush=True)
        pipe.vae.to(pipe.device)
        encoded = encode_sample(pipe, sample, config()['training'], context, 1.0)
        encoded = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k,v in encoded.items() if k != 'context'}
        pipe.vae.to('cpu')
        torch.cuda.empty_cache()
        atomic_torch_save({'inputs':encoded, 'config_id':config()['config_id'], 'cs':cs,'start':start}, path)
    require(tuple(encoded['input_latents'].shape) == (1,16,5,64,64), 'Actual VAE loss grid differs')
    require(tuple(encoded['vace_context'].shape) == (1,96,5,64,64), 'Actual VACE condition grid differs')
    if inspect:
        from vace_p01 import check_loader_contract
        check_loader_contract(sample['conditioning'], sample['model_mask'], num_frames=17)
        mask = pipe.preprocess_video(sample['model_mask'], min_value=0, max_value=1).cpu()
        expected = rearrange(mask[0,0], 'T (H P) (W Q) -> 1 (P Q) T H W', P=8,Q=8)
        expected = F.interpolate(expected, size=(5,64,64), mode='nearest-exact')
        require(torch.equal(expected, encoded['vace_context'][:,32:].float()), 'Official ModelMask tensor mismatch')
    encoded = {k:v.to(pipe.device) if isinstance(v,torch.Tensor) else v for k,v in encoded.items()}
    encoded['context'] = context
    return encoded, torch.from_numpy(m[None]).to(pipe.device), torch.from_numpy(e[None]).to(pipe.device)

def optimize(pipe, optimizer, inputs, m, e, method, row):
    import torch
    from vace_roof_training import trainable_lora_parameters
    seed_all(int(row['stochastic_seed']))
    optimizer.zero_grad(set_to_none=True)
    params = [p for _,p in trainable_lora_parameters(pipe)]
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        loss, tw, timestep = flow_loss(pipe, inputs, m, e, method, row)
    require(loss.ndim == 0 and bool(torch.isfinite(loss)) and loss.requires_grad, 'Invalid or disconnected loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(params, config()['training']['max_grad_norm'], error_if_nonfinite=True)
    # Zero scheduler weight can yield zero current gradients while Adam moments
    # still change weights; observe an actual parameter in that case as well.
    watched = next((p for p in params if p.grad is not None and bool(p.grad.abs().max() > 0)), params[0])
    before = watched.detach().clone()
    optimizer.step()
    changed = not torch.equal(watched.detach(), before)
    require(bool(torch.stack([torch.isfinite(p).all() for p in params]).all()), 'Non-finite LoRA weight')
    return {'loss':float(loss.detach()), 'grad_norm':float(norm), 'time_weight':tw,'timestep':timestep,'parameters_changed':changed}

def mapping_image(cs, start):
    from PIL import ImageDraw
    m, e = labels(cs,start,17)
    gm, ge = loss_grid(m), loss_grid(e)
    canvas = Image.new('RGB',(5*512, 3*544),'white')
    draw = ImageDraw.Draw(canvas)
    for col, group in enumerate(TEMPORAL_GROUPS):
        representative = max(group, key=lambda i: int(m[i].sum()))
        rgb = np.asarray(source(cs,'damaged',start+representative)).copy()
        rgb[m[representative]] = (rgb[m[representative]].astype(float)*0.6 + np.array([255,0,0])*0.4).astype('uint8')
        rgb[e[representative]] = [0,255,0]
        for row, image in enumerate((Image.fromarray(rgb), Image.fromarray((gm[col]*255).astype('uint8')).convert('RGB').resize((512,512),Image.Resampling.NEAREST),
                                      Image.fromarray((ge[col]*255).astype('uint8')).convert('RGB').resize((512,512),Image.Resampling.NEAREST))):
            canvas.paste(image,(col*512,row*544+32))
            title = f'{cs} pixel B{start+representative:03}' if row==0 else f'{"M" if row==1 else "E"} grid group ' + ','.join(f'{start+i:03}' for i in group)
            draw.text((col*512+8,row*544+8),title,fill='black')
    canvas.save(RUN/'checks'/f'{cs}_window_mapping.png')

def preflight():
    import torch
    from safetensors.torch import load_file
    from vace_roof_training import make_optimizer
    begin = time.monotonic()
    pipe = new_pipe()
    init_path = RUN/'checkpoints/shared_initialization.safetensors'
    if not init_path.exists():
        save_adapter(pipe, init_path, 0)
    initial = load_file(str(init_path))
    restore_parameters(pipe,initial)
    require(all(not bool(v.any()) for n,v in initial.items() if '.lora_B.' in n), 'New initialization must have zero B deltas')
    context = text_context(pipe)
    results = []
    for cs in ('CS06','CS09'):
        m,_ = labels(cs,0,80)
        start = max(range(64),key=lambda s:int(m[s:s+17].sum()))
        inputs, m, e = cached_window(pipe,context,cs,start,inspect=True)
        mapping_image(cs,start)
        for method in ('C','U','S'):
            restore_parameters(pipe,initial)
            optimizer = make_optimizer(pipe,config()['training'])
            result = optimize(pipe,optimizer,inputs,m,e,method,{'timestep_index':500,'noise_seed':20261022,'stochastic_seed':20261022})
            require(result['grad_norm']>0 and result['parameters_changed'], 'Temporary backward/update failed')
            result.update(cs=cs,start=start,method=method,M_sum=float(m.sum()),E_sum=float(e.sum()),
                          U_extra=float((2*e.sum()/m.sum()*m).sum()),S_extra=float((2*e).sum()))
            results.append(result)
            print('TEMPORARY CHECK:',result,flush=True)
            del optimizer
        del inputs,m,e
    mark_complete('preflight', temporary_updates=6, formal_updates=0, results=results,
        initialization_sha256=sha(init_path),elapsed_seconds=time.monotonic()-begin,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    print('CS06/CS09 actual tensor and C/U/S backward checks PASSED; formal training starts again from saved initial adapter.',flush=True)

def train(method):
    import torch
    from safetensors.torch import load_file
    from vace_roof_training import make_optimizer
    from tqdm import tqdm
    require(method in ('C','U','S'),'Invalid training group')
    begin = time.monotonic()
    c = config()
    out = RUN/'checkpoints'/method
    out.mkdir(exist_ok=True)
    pipe = new_pipe()
    initial_path=RUN/'checkpoints/shared_initialization.safetensors'
    initial_hash=sha(initial_path)
    restore_parameters(pipe,load_file(str(initial_path)))
    optimizer=make_optimizer(pipe,c['training'])
    require(not optimizer.state, 'New group did not get fresh optimizer')
    context=text_context(pipe)
    completed=0
    resume_path=out/'resume_latest.pt'
    history=[]
    if resume_path.exists():
        state=torch.load(resume_path,map_location='cpu',weights_only=False)
        require(state['config_id']==c['config_id'] and state['method']==method and state['initialization_sha256']==initial_hash,'Invalid training resume identity')
        restore_parameters(pipe,state['adapter'])
        optimizer.load_state_dict(state['optimizer'])
        completed=state['actual_updates']
        history=state['history']
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        print(f'Resuming {method} after actual optimizer update {completed}',flush=True)
    logpath=RUN/'metrics'/f'train_{method}_loss.csv'
    rows=read_csv(RUN/'shared_sampling_table.csv')
    require(sha(RUN/'shared_sampling_table.csv')==c['paired_sampling_sha256'],'Paired sampling table changed')
    require(len(rows)==300,'Expected exactly 300 updates')
    def save_resume(milestone=False):
        state={'config_id':c['config_id'],'method':method,'actual_updates':completed,'initialization_sha256':initial_hash,
            'adapter':parameter_state(pipe),'optimizer':optimizer.state_dict(),'history':history,
            'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all()}
        atomic_torch_save(state,resume_path)
        if milestone:
            import shutil
            shutil.copyfile(resume_path,out/f'resume_step{completed}.pt')
    fields=['update','cs','start','timestep_index','noise_seed','stochastic_seed','loss','grad_norm','time_weight','timestep','parameters_changed','lr','elapsed_seconds']
    write_csv(logpath,history,fields)
    effective=False if not history else any(r['parameters_changed'] for r in history)
    with logpath.open('a',newline='',encoding='utf-8') as log:
        writer=csv.DictWriter(log,fieldnames=fields)
        try:
            bar=tqdm(total=300,initial=completed,desc=f'{method} actual AdamW updates',file=sys.stdout,dynamic_ncols=True,unit='update')
            for row in rows[completed:]:
                inputs,m,e=cached_window(pipe,context,row['cs'],int(row['start']))
                result=optimize(pipe,optimizer,inputs,m,e,method,row)
                completed=int(row['update'])
                effective |= result['parameters_changed']
                if completed==5:
                    require(effective,'No effective update in first five optimizer steps')
                entry={**row,**result,'lr':optimizer.param_groups[0]['lr'],'elapsed_seconds':round(time.monotonic()-begin,3)}
                history.append(entry)
                writer.writerow(entry);log.flush()
                write_json(RUN/'training'/f'{method}_progress.json',{'method':method,'actual_updates':completed,'updated_at':now(),**result})
                bar.set_postfix(cs=row['cs'],start=row['start'],loss=f'{result["loss"]:.6f}');bar.update(1)
                if completed%10==0:
                    save_resume(milestone=completed in (100,200))
                    live=read_json(RUN/'progress.json')
                    live['config_id']=c['config_id']
                    live['actual_updates'][method]=completed
                    live.update(updated_at=now(),current_building=row['cs'],current_window_start=int(row['start']))
                    write_json(RUN/'progress.json',live)
                    print(f'{method}: {completed}/300 actual updates; elapsed={time.monotonic()-begin:.1f}s; resume saved',flush=True)
                if completed==300:
                    save_adapter(pipe,out/f'step{completed}.safetensors',completed)
                del inputs,m,e
            bar.close()
        except BaseException:
            save_resume()
            raise
    require(completed==len(history)==300 and effective,'Incomplete optimizer budget')
    # If interruption happened after the final resume state was saved but before
    # its adapter export, finish that export without another optimizer update.
    if not (out/'step300.safetensors').exists():
        save_adapter(pipe,out/'step300.safetensors',300)
    mark_complete(f'train_{method}',actual_updates=completed,elapsed_seconds=time.monotonic()-begin,
        initialization_sha256=initial_hash,sampling_sha256=c['paired_sampling_sha256'],
        weight_path=str(out/'step300.safetensors'),weight_sha256=sha(out/'step300.safetensors'),
        optimizer_parameter_step_range=[min(int(s['step']) for s in optimizer.state.values()),
                                        max(int(s['step']) for s in optimizer.state.values())],
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    print(f'TRAINING {method} COMPLETE: 300/300 optimizer updates; {out / "step300.safetensors"}',flush=True)

if __name__=='__main__':
    if sys.argv[1]=='preflight': preflight()
    else: train(sys.argv[1])
