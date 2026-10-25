"""Visible serial inference-only queue; fresh workers, no automatic retries."""
import fcntl
import subprocess
import time
from complex_common import *


def jobs():
    order=['CS09','CS10','CS11','CS12']+[cs for cs in IDS if cs not in ('CS09','CS10','CS11','CS12')]
    tasks=[(f'infer_{m}_{cs}',['complex_infer.py',m,cs],'已有权重81帧推理') for cs in order for m in ('B','L','C','U','S')]
    tasks += [('evaluate',['complex_evaluate.py'],'评测与可视化'),
              ('review_sheets',['review_frames.py','CS09','CS10','CS11','CS12'],'完整轨迹复核图')]
    return tasks


def main():
    c=config();state=read_json(RUN/'progress.json')
    if state['status'] in ('complete','awaiting_visual_review'):
        print(f'No GPU work pending: {state["status"]}; {RUN}',flush=True);return
    require(state['status']!='failed','Prior failed job is retained; inspect before authorizing a retry')
    with (RUN/'logs/run.lock').open('a') as lock,(PROJECT/'logs/.wan_a6000_gpu.lock').open('r') as gpu_lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from vace_p01_environment import _nvml_snapshot
        require(not _nvml_snapshot()['compute_processes'],'GPU currently has another compute process')
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PYTHONFAULTHANDLER='1',
                 OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TMPDIR=str(RUN/'tmp'),HF_HUB_OFFLINE='1',
                 TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
        state.update(status='running',pid=os.getpid(),tmux=os.environ.get('TMUX_PANE'),updated_at=now())
        for job,args,stage in jobs():
            marker=completion_path(job)
            if marker.exists():
                record=read_json(marker);require(record['config_id']==c['config_id'],'Completed task configuration changed')
                if job.startswith('infer_'):
                    _,method,cs=job.split('_');require(valid_prediction(method,cs),'Invalid completed trajectory')
                    require(record.get('origin')=='reused' or record.get('worker_exit_code')==0,'Unreviewed abnormal exit')
                print(f'REUSE {job}; no GPU rerun',flush=True);continue
            state.update(stage=stage,current_job=job,updated_at=now());write_json(RUN/'progress.json',state)
            print(f'\nSTAGE {stage}: {job}; valid trajectories {state["completed_trajectories"]}/60; {now()}',flush=True)
            command=[PYTHON,'-u',str(SCRIPTS/args[0]),*args[1:]]
            if args[0]=='complex_infer.py':
                command=['gdb','--batch','--quiet','-x',str(PROJECT/'experiments/vace/complex12_rev1/scripts/native_exit.py'),'-ex','wan-run','--args',*command]
            with (RUN/'logs'/f'{job}.log').open('a',encoding='utf-8') as log:
                log.write(f'START {now()}\n');log.flush()
                process=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                state['worker_pid']=process.pid;write_json(RUN/'progress.json',state)
                for line in process.stdout:print(line,end='',flush=True);log.write(line);log.flush()
                code=process.wait();log.write(f'\nEXIT {code} {now()}\n')
            # Read back live step updates rather than overwrite them with stale parent state.
            state=read_json(RUN/'progress.json');state.pop('worker_pid',None)
            if job=='review_sheets' and code==0:mark_complete(job,buildings=['CS09','CS10','CS11','CS12'])
            if marker.exists():
                record=read_json(marker);record.update(worker_exit_code=code,origin='inferred' if job.startswith('infer_') else 'cpu')
                write_json(marker,record)
            if code!=0 or not marker.exists():
                state.update(status='failed',failed_job=job,exit_code=code,updated_at=now())
                write_json(RUN/'progress.json',state)
                raise RuntimeError(f'{job} failed (exit {code}); retained outputs, no automatic retry')
            if job.startswith('infer_'):
                state['completed_trajectories']+=1;state['inferred_trajectories']+=1
            state.update(last_completed_job=job,updated_at=now());write_json(RUN/'progress.json',state)
        state.update(status='awaiting_visual_review',stage='连续帧复核与中文报告',updated_at=now())
        state.pop('pid',None);write_json(RUN/'progress.json',state)
        print(f'60/60 trajectories, metrics and media ready. Visual review/report pending: {RUN}',flush=True)


if __name__=='__main__':main()
