"""Visible single-GPU queue: both 600-update groups, 40 trajectories, CPU delivery."""
import fcntl
import subprocess
from complex_common import *


def infer_tasks(method, step, inputs):
    return [(infer_job(method, step, version, cs), ['complex_infer.py', method, str(step), version, cs],
             f'{method} step{step} 完整81帧推理') for version, cs in inputs]


def jobs():
    tasks = [('train_C_lin17_300', ['complex_train.py', 'C_lin17', '--stop-update', '300'], '线性组训练至300')]
    tasks += infer_tasks('C_lin17', 300, PROBES)
    tasks += [('early_probe', ['complex_evaluate.py', '--early-probe'], '线性300步复现核对'),
              ('train_C_lin17_600', ['complex_train.py', 'C_lin17', '--stop-update', '600'], '线性组完整状态恢复至600'),
              ('train_C_ring17_600', ['complex_train.py', 'C_ring17', '--stop-update', '600'], '循环组从共享初始化训练至600')]
    tasks += infer_tasks('C_ring17', 300, PROBES)
    for method in METHODS:
        tasks += infer_tasks(method, 600, full_inputs())
    return tasks + [('evaluate', ['complex_evaluate.py'], '配对评测、分段端点指标与视频交付')]


def validate_completed(job, args):
    marker = completion_path(job)
    if not marker.exists():
        return False
    record = read_json(marker)
    require(record['config_id'] == config()['config_id'], 'Completed task config differs')
    require(record.get('worker_exit_code', 0) == 0, 'Previous task exited abnormally; preserve its evidence')
    if job.startswith('infer_'):
        require(valid_prediction(args[1], int(args[2]), args[3], args[4]), 'Incomplete inference marker')
    elif job.startswith('train_'):
        require(record['method'] == args[1] and record['actual_updates'] == int(args[3]) and
                Path(record['weight_path']).is_file(), 'Training result incomplete')
    return True


def main():
    config()
    state = read_json(RUN / 'progress.json')
    if state['status'] in ('complete', 'awaiting_visual_review'):
        print(f'No GPU work pending: {state["status"]}; {RUN}', flush=True)
        return
    with (RUN / 'logs/run.lock').open('a') as lock, (PROJECT / 'logs/.wan_a6000_gpu.lock').open('a') as gpu_lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from vace_p01_environment import _nvml_snapshot
        require(not _nvml_snapshot()['compute_processes'], 'GPU has another compute process')
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', PYTHONFAULTHANDLER='1',
                   OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TMPDIR=str(RUN / 'tmp'),
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
        finished = {job for job, args, _ in jobs() if validate_completed(job, args)}
        count = sum(job.startswith('infer_') for job in finished)
        state.update(status='running', pid=os.getpid(), tmux=os.environ.get('TMUX_PANE'),
                     source_git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip(),
                     completed_trajectories=count, updated_at=now())
        write_json(RUN / 'progress.json', state)
        print(f'PAIRED QUEUE: C_lin17 600 + C_ring17 600; 40 trajectories; RUN={RUN}', flush=True)
        for job, args, stage in jobs():
            if job in finished:
                print(f'REUSE {job}; inference completed {count}/40', flush=True)
                continue
            state = read_json(RUN / 'progress.json')
            state.update(stage=stage, current_job=job, updated_at=now())
            write_json(RUN / 'progress.json', state)
            print(f'\nSTAGE {stage}: {job}; inference completed {count}/40; {now()}', flush=True)
            command = [PYTHON, '-u', str(SCRIPTS / args[0]), *args[1:]]
            gpu_job = args[0] in ('complex_train.py', 'complex_infer.py')
            if gpu_job:
                command = ['gdb', '--batch', '--quiet', '-x', str(PROJECT / 'experiments/vace/complex12_rev1/scripts/native_exit.py'),
                           '-ex', 'wan-run', '--args', *command]
            job_env = dict(env)
            if not gpu_job:
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
                raise RuntimeError(f'{job} failed (exit {code}); prior valid state and outputs retained')
            if job.startswith('infer_'):
                count += 1
            state.update(completed_trajectories=count, last_completed_job=job, updated_at=now())
            write_json(RUN / 'progress.json', state)
        state.update(status='awaiting_visual_review', stage='两组计算与媒体完成，等待视觉复核及最终报告', updated_at=now())
        state.pop('pid', None); state.pop('current_job', None)
        write_json(RUN / 'progress.json', state)
        print(f'BOTH GROUPS 600/600; inference 40/40. Results: {RUN / "00_看结果"}', flush=True)


if __name__ == '__main__':
    main()
