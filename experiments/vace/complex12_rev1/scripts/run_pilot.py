"""Visible serial runner; completed matching trajectories are never regenerated."""
import fcntl
import subprocess
import time
from complex_common import *

def main():
    c=config()
    progress=read_json(RUN/'progress.json')
    if progress.get('status')=='complete':
        require(progress.get('config_id')==c['config_id'],'Completed experiment config mismatch')
        print('实验已完成；所有任务保持原结果。',flush=True)
        print(RUN/'00_看结果/README_从这里开始.md',flush=True)
        return
    with (RUN/'logs/run.lock').open('a') as lock, (PROJECT/'logs/.wan_a6000_gpu.lock').open('r') as gpu_lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from vace_p01_environment import _nvml_snapshot
        gpu=_nvml_snapshot()
        require(not gpu['compute_processes'],'Another GPU process is already running')
        jobs=[('preflight',['complex_train.py','preflight'],'数据适配/实际反向联调')]
        order=['CS09']+[s for s in IDS if s!='CS09']
        for method in ('B','L'):
            jobs += [(f'infer_{method}_{cs}',['complex_infer.py',method,cs],'B/L基线推理') for cs in order]
        jobs += [(f'train_{m}',['complex_train.py',m],'C/U/S训练') for m in ('C','U','S')]
        for method in ('C','U','S'):
            jobs += [(f'infer_{method}_{cs}',['complex_infer.py',method,cs],'正式推理') for cs in IDS]
        jobs += [('evaluate',['complex_evaluate.py'],'评测与可视化')]
        progress=read_json(RUN/'progress.json')
        progress.update(config_id=c['config_id'],status='running',pid=os.getpid(),tmux=os.environ.get('TMUX_PANE'),updated_at=now())
        complete_count=0
        finished=set()
        for job,args,stage in jobs:
            marker=completion_path(job)
            if marker.exists():
                record=read_json(marker)
                require(record['config_id']==c['config_id'],f'Completed job config mismatch: {job}')
                if record.get('worker_exit_code',0)!=0:
                    require(job.startswith('infer_') and record.get('independent_output_validation'),
                            f'Abnormal worker exit requires independent output review: {job}')
                if job.startswith('infer_'):
                    _,method,cs=job.split('_')
                    require(valid_prediction(method,cs),'Marked output invalid')
                    complete_count+=1
                elif job.startswith('train_'):
                    require(record['actual_updates']==300 and Path(record['weight_path']).is_file(),'Training result incomplete')
                    progress['actual_updates'][job[-1]]=300
                finished.add(job)
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PYTHONFAULTHANDLER='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                 TMPDIR=str(RUN/'tmp'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
        progress['completed_trajectories']=complete_count
        for job,args,stage in jobs:
            if job in finished:
                print(f'REUSE {job}; completed trajectories {complete_count}/60',flush=True)
                continue
            progress.update(stage=stage,current_job=job,updated_at=now())
            write_json(RUN/'progress.json',progress)
            print(f'\nSTAGE {stage}: {job}; completed trajectories {complete_count}/60; {now()}',flush=True)
            command=[PYTHON,'-u',str(SCRIPTS/args[0]),*args[1:]]
            if args[0] in ('complex_train.py','complex_infer.py'):
                command=['gdb','--batch','--quiet','-x',str(SCRIPTS/'native_exit.py'),'-ex','wan-run','--args',*command]
            with (RUN/'logs'/f'{job}.log').open('a',encoding='utf-8') as log:
                log.write(f'\nSTART {now()}\n');log.flush()
                process=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                         text=True,bufsize=1)
                for line in process.stdout:
                    print(line,end='',flush=True);log.write(line);log.flush()
                returncode=process.wait()
                log.write(f'\nEXIT {returncode} {now()}\n')
            if completion_path(job).exists():
                record=read_json(completion_path(job))
                record['worker_exit_code']=returncode
                write_json(completion_path(job),record)
            if returncode!=0:
                progress.update(status='failed',failed_job=job,exit_code=returncode,updated_at=now())
                for method in ('C','U','S'):
                    p=RUN/'training'/f'{method}_progress.json'
                    if p.exists(): progress['actual_updates'][method]=read_json(p)['actual_updates']
                write_json(RUN/'progress.json',progress)
                raise RuntimeError(f'{job} exited {returncode}; existing outputs preserved; no automatic retries')
            require(completion_path(job).exists(),f'Worker exited without completion evidence: {job}')
            if job.startswith('infer_'): complete_count+=1
            if job.startswith('train_'): progress['actual_updates'][job[-1]]=300
            progress.update(completed_trajectories=complete_count,updated_at=now(),last_completed_job=job)
            write_json(RUN/'progress.json',progress)
        progress.update(status='awaiting_visual_review',stage='中文结论与最终验收',updated_at=now())
        write_json(RUN/'progress.json',progress)
        print('All GPU work and metric generation finished. Visual review and Chinese conclusions must be completed before final success.',flush=True)

if __name__=='__main__': main()
