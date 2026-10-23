"""Run after the actual visual review and Chinese report are finished."""
import math
from safetensors import safe_open
from complex_evaluate import *


def main():
    c = config()
    methods = c['evaluation']['methods']
    report = RUN / '00_看结果/最终实验结论.md'
    require(report.is_file() and report.stat().st_size > 0, 'Chinese report is missing')
    sampled = read_csv(RUN / 'shared_sampling_table.csv')
    init = read_json(completion_path('preflight'))['initialization_sha256']
    for method in ('C', 'U', 'S'):
        record = read_json(completion_path(f'train_{method}'))
        require(record['config_id'] == c['config_id'], 'Training config mismatch')
        require(record['actual_updates'] == 300 and record['optimizer_parameter_step_range'] == [300, 300],
                'Actual optimizer updates are incomplete')
        require(record['initialization_sha256'] == init and
                record['sampling_sha256'] == c['paired_sampling_sha256'], 'Paired training mismatch')
        history = read_csv(RUN / f'metrics/train_{method}_loss.csv')
        require(len(history) == 300, 'Training history is incomplete')
        for row, expected in zip(history, sampled):
            require(all(str(row[key]) == str(value) for key, value in expected.items()), 'Sampling history differs')
            require(math.isfinite(float(row['loss'])) and math.isfinite(float(row['grad_norm'])),
                    'Non-finite training record')
        with safe_open(str(lora_path(method)), framework='pt', device='cpu') as weights:
            require(weights.metadata()['optimizer_update'] == '300', 'Wrong adapter export')
            require(weights.metadata()['config_id'] == c['config_id'], 'Adapter config mismatch')
            require(len(weights.keys()) > 0, 'Empty adapter')
            for key in weights.keys():
                require(bool(weights.get_tensor(key).isfinite().all()), 'Unreadable/non-finite adapter')

    for method in methods:
        for cs in IDS:
            require(valid_prediction(method, cs), 'Incomplete raw trajectory')
    records = read_json(RUN / 'checks/delivery_media.json')
    require(len(records) == 12 and {r['cs'] for r in records} == set(IDS), 'Media checks are incomplete')
    for r in records:
        require(r['frames'] == 81 and r['fps'] == 16, 'Video check failed')
    for cs in IDS:
        folder = RUN / '00_看结果' / cs
        require({p.name for p in folder.iterdir()} == {'01_五方法同步对比.mp4', '02_关键帧总览.png'},
                f'Unexpected delivery contents: {cs}')
        video = folder / '01_五方法同步对比.mp4'
        require(video.stat().st_size > 0, 'Video is empty')
        with Image.open(folder / '02_关键帧总览.png') as im:
            im.load()
            require(im.size == (2560, 544 * (len(methods) + 2)), 'Overview dimensions differ')
    require(len(read_csv(RUN / 'metrics/frame_metrics.csv')) == 80 * 12 * (len(methods) + 1),
            'Frame metrics are incomplete')
    require(len(read_csv(RUN / 'metrics/building_metrics.csv')) == 12 * (len(methods) + 1),
            'Building metrics are incomplete')
    require(len(read_csv(RUN / 'metrics/damaged_baseline.csv')) == 12, 'Damaged baseline is incomplete')
    require(len(read_csv(RUN / 'metrics/paired_differences.csv')) == 36, 'Paired differences are incomplete')
    for split, n in [('train', 8), ('val', 4)]:
        rows = read_csv(RUN / f'metrics/{split}_summary.csv')
        require({r['method'] for r in rows} == set(methods + ['Damaged']), 'Summary methods differ')
        require(all(int(r['buildings']) == n for r in rows), 'Split counts differ')

    n = len(methods) * len(IDS)
    state = read_json(RUN / 'progress.json')
    status = f'已完成：C/U/S各300次优化器更新，{n}条81帧轨迹、{n * 81}张原始PNG、12条同步视频、12张总览、指标及中文结论齐全'
    write_entry(methods, status)
    entry = RUN / '00_看结果/README_从这里开始.md'
    entry.write_text(entry.read_text(encoding='utf-8').replace(
        f'未完成/失败说明：{status}。实时事实见',
        '未完成任务：无。历史异常及恢复情况见下文，全部正式交付已完成。实际记录见'), encoding='utf-8')
    with entry.open('a', encoding='utf-8') as f:
        f.write('\n执行记录：B/CS04曾在保存完整预测后异常退出，81帧已独立复核并复用，未重跑；'
                '服务器重启后恢复了原磁盘，并从C的第40次更新继续。历史异常保留在日志中，未解决的原生退出原因不作已修复声明。\n')
        f.write('\n续跑命令（仅恢复未完成部分；当前已全部完成，执行后直接报告完成）：\n\n'
                f'```bash\nbash {RUN / "resume.sh"}\n```\n')
    state.update(status='complete', stage='实验完成', current_job='complete', updated_at=now(),
                 completed_at=now(), actual_updates={'C': 300, 'U': 300, 'S': 300},
                 completed_trajectories=n, expected_trajectories=n, png_frames=n * 81,
                 comparison_videos=12, overviews=12, final_report=str(report),
                 last_completed_job='visual_review_and_report')
    state.pop('pid', None)
    write_json(RUN / 'progress.json', state)
    print(status, flush=True)
    for path in [RUN / '00_看结果', report, RUN / 'metrics', *(lora_path(m) for m in ('C', 'U', 'S')),
                 RUN / 'progress.json', RUN / 'resume.sh']:
        require(path.exists(), f'Missing final path: {path}')
        print(path, flush=True)


if __name__ == '__main__':
    main()
