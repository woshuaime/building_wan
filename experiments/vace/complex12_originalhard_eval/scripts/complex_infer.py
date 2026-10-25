"""Fresh Base per trajectory; preserve all raw decoded output PNGs."""
import time
from complex_common import *

def infer(method,cs):
    import torch
    from vace_p01 import load_local_vace
    from vace_roof_smoke import load_stage_lora,inference_arguments
    from vace_p01_81f_console import ConsoleState
    c=config()
    require(method in c['evaluation']['methods'] and cs in IDS,'Invalid trajectory')
    if valid_prediction(method,cs):
        print(f'Reusing completed {method} / {cs} / 81 frames',flush=True)
        return
    begin=time.monotonic()
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    folder=RUN/'predictions'/method/cs
    folder.mkdir(parents=True,exist_ok=True)
    raw_ready=folder/'raw_ready.json'
    # A CPU-only failure after successful inference must not trigger GPU recomputation.
    if raw_ready.exists():
        ready=read_json(raw_ready)
        require(ready['config_id']==c['config_id'],'Raw output config mismatch')
        for i in range(81): read_image(folder/f'{i:03}.png')
        mark_complete(f'infer_{method}_{cs}',**{k:v for k,v in ready.items() if k!='config_id'})
        return
    require(not list(folder.glob('*.png')),'Partial PNGs retained; inspect before replacing a failed trajectory')
    print(f'OriginalHard Running {method} / {cs} / source{index()[cs]["source_id"]} / 81 frames; fresh original VACE-1.3B',flush=True)
    print(f'ACTUAL Base absolute path: {c["model"]["paths"]["checkpoint"]}',flush=True)
    sample=conditions(cs)
    pipe=load_local_vace(c['model']['paths'],memory_mode='full')
    loaded=None
    changed=False
    if method!='B':
        path=lora_path(method)
        print(f'ACTUAL LoRA absolute path: {path}; branch=vace; scale=1.0',flush=True)
        from safetensors.torch import load_file
        state=load_file(str(path),device='cpu')
        bkey=next(n for n,t in state.items() if '.lora_B.' in n and bool(t.any()))
        module_name=bkey.split('.lora_B.')[0]
        layer=dict(pipe.vace.named_modules())[module_name]
        before=layer.weight.detach().cpu().clone()
        load_stage_lora(pipe,path,c)
        changed=not torch.equal(before,layer.weight.detach().cpu())
        require(changed,'Adapter did not change actual VACE backbone weights')
        weight_id=c['weights'][method]['sha256']
        loaded={'path':str(path),'sha256':weight_id,'scale':1.0,'branch':'vace','weight_changed':True}
        del state,before
    require(pipe.audio_encoder is None,'Unexpected postprocessing audio branch')
    def progress_event(event):
        live=read_json(RUN/'progress.json')
        live.update(config_id=c['config_id'],updated_at=now(),inference_stage=event['stage'],
            inference_steps=event['completed_steps'],inference_elapsed_seconds=round(event['elapsed'],2))
        write_json(RUN/'progress.json',live)
    console=ConsoleState(total_steps=30,event_sink=progress_event)
    with torch.inference_mode(),console.vae_stages(pipe.vae):
        output=pipe(**inference_arguments(c,sample),progress_bar_cmd=console.progress)
    require(len(output)==81 and console.completed_steps==30,'Incomplete model result')
    # No compositing, color editing, resize or closure replacement.
    for i,frame in enumerate(output):
        require(isinstance(frame,Image.Image) and frame.size==(512,512) and frame.mode=='RGB','Unexpected raw output type')
        frame.save(folder/f'{i:03}.png')
    ready={'config_id':c['config_id'],'method':method,'cs':cs,'frame_count':81,'size':[512,512],
        'seed':20261018,'steps':30,'lora':loaded,'elapsed_seconds':time.monotonic()-begin,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'raw_output':'WanVideoPipeline quantized decoded RGB; no post-composite; frame080 preserved'}
    write_json(raw_ready,ready)
    for i in range(81): read_image(folder/f'{i:03}.png')
    mark_complete(f'infer_{method}_{cs}',**{k:v for k,v in ready.items() if k!='config_id'})
    print(f'SAVED {method}/{cs}: 81 raw PNGs; {time.monotonic()-begin:.1f}s; peak {ready["peak_allocated_bytes"]/1024**3:.3f} GiB; {folder}',flush=True)

if __name__=='__main__': infer(sys.argv[1],sys.argv[2])
