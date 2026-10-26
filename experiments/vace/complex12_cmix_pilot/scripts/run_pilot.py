"""Visible serial train/infer/evaluate queue with full-state resume and result reuse."""
import fcntl
import subprocess
from complex_common import *

def jobs():
    return [('train_C_mix', ['complex_train.py'], 'C_mix训练')] + [
        (f'infer_{version}_{cs}', ['complex_infer.py', version, cs], 'C_mix正式推理')
        for version, cs in trajectory_order()] + [('evaluate', ['complex_evaluate.py'], '配对评测与视频交付')]

def main():
    c = config()
    state = read_json(RUN / 'progress.json')
    if state['status'] in ('complete', 'awaiting_visual_review'):
        print(f'No GPU tasks pending: {state["status"]}; {RUN}', flush=True)
        return
    with (RUN / 'logs/run.lock').open('a') as lock, (PROJECT / 'logs/.wan_a6000_gpu.lock').open('r') as gpu_lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from vace_p01_environment import _nvml_snapshot
        require(not _nvml_snapshot()['compute_processes'], 'GPU is occupied by another compute process')
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', PYTHONFAULTHANDLER='1',
                   OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TMPDIR=str(RUN / 'tmp'),
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
        finished = set()
        for job, args, _ in jobs():
            marker = completion_path(job)
            if not marker.exists():
                continue
            record = read_json(marker)
            require(record['config_id'] == c['config_id'], 'Completed task configuration changed')
            if job.startswith('infer_'):
                valid_prediction(args[1], args[2])
            elif job == 'train_C_mix':
                require(record['actual_updates'] == 300 and Path(record['weight_path']).is_file(), 'Training result incomplete')
            require(record.get('worker_exit_code', 0) == 0 or record.get('independent_output_validation'), 'Unreviewed abnormal worker exit')
            finished.add(job)
        count = sum(job.startswith('infer_') for job in finished)
        state.update(status='running', pid=os.getpid(), tmux=os.environ.get('TMUX_PANE'),
                     completed_trajectories=count, updated_at=now())
        if 'train_C_mix' in finished:
            state['actual_updates'] = {'C_mix': 300}
        write_json(RUN / 'progress.json', state)
        for job, args, stage in jobs():
            if job in finished:
                print(f'REUSE {job}; inference completed {count}/16', flush=True)
                continue
            state.update(stage=stage, current_job=job, updated_at=now())
            write_json(RUN / 'progress.json', state)
            print(f'\nSTAGE {stage}: {job}; inference completed {count}/16; {now()}', flush=True)
            command = [PYTHON, '-u', str(SCRIPTS / args[0]), *args[1:]]
            if args[0] in ('complex_train.py', 'complex_infer.py'):
                command = ['gdb', '--batch', '--quiet', '-x', str(PROJECT / 'experiments/vace/complex12_rev1/scripts/native_exit.py'),
                           '-ex', 'wan-run', '--args', *command]
            job_env = dict(env)
            if job == 'evaluate':
                job_env['CUDA_VISIBLE_DEVICES'] = ''
            with (RUN / 'logs' / f'{job}.log').open('a', encoding='utf-8') as log:
                log.write(f'\nSTART {now()}\n'); log.flush()
                process = subprocess.Popen(command, cwd=PROJECT, env=job_env, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, bufsize=1)
                state['worker_pid'] = process.pid
                write_json(RUN / 'progress.json', state)
                for line in process.stdout:
                    print(line, end='', flush=True); log.write(line); log.flush()
                code = process.wait()
                log.write(f'\nEXIT {code} {now()}\n')
            state = read_json(RUN / 'progress.json')
            state.pop('worker_pid', None)
            marker = completion_path(job)
            if marker.exists():
                record = read_json(marker); record['worker_exit_code'] = code; write_json(marker, record)
            if code != 0 or not marker.exists():
                state.update(status='failed', failed_job=job, exit_code=code, updated_at=now())
                state.pop('pid', None)
                write_json(RUN / 'progress.json', state)
                raise RuntimeError(f'{job} failed (exit {code}); outputs and full resume retained')
            if job.startswith('infer_'):
                count += 1
            state.update(completed_trajectories=count, updated_at=now(), last_completed_job=job)
            write_json(RUN / 'progress.json', state)
        state.update(status='awaiting_visual_review', stage='计算与媒体完成，等待用户通知视觉复核及报告', updated_at=now())
        state.pop('pid', None); state.pop('current_job', None)
        write_json(RUN / 'progress.json', state)
        print(f'300/300 updates; 16/16 trajectories. Visual review/report deferred by user: {RUN / "00_看结果"}', flush=True)

if __name__ == '__main__':
    main()
