"""CPU metrics: same-budget window comparisons, reproduction probes and endpoints."""
import math
import time
from complex_common import *

metrics = legacy('complex_evaluate')
ERRORS = metrics.ERRORS
ENDPOINT_KEYS = ['first_gt_full_mae', 'first_gt_masked_mae', 'last_gt_full_mae', 'last_gt_masked_mae']
SUM_KEYS = ['masked_pixels', 'structure_pixels', 'known_pixels', 'masked_absolute_sum', 'masked_squared_sum',
            'structure_absolute_sum', 'known_absolute_sum']
SEGMENTS = [('000-016', range(17)), ('017-062', range(17, 63)), ('063-079', range(63, 80))]


def paired_change(old, new, metric):
    a, b = metrics.number(old), metrics.number(new)
    finite = a is not None and b is not None and math.isfinite(a) and math.isfinite(b)
    delta = b - a if finite else None
    relative = 100 * delta / abs(a) if finite and a != 0 else None
    worse = (delta < 0 if metric == 'masked_psnr' else delta > 0) if finite else 'N/A'
    return dict(old=old, new=new, absolute_change=metrics.safe_value(delta),
                relative_change_percent=metrics.safe_value(relative), regressed=worse)


def summed(rows):
    return {key: sum(float(r[key]) for r in rows) for key in SUM_KEYS}


def endpoints(method, step, version, cs):
    result = {}
    predictions = []
    first_mask = None
    for frame, prefix in [(0, 'first'), (80, 'last')]:
        prediction = np.asarray(read_image(prediction_dir(method, step, version, cs) / f'{frame:03}.png'), dtype=np.float64)/255
        gt = np.asarray(source(cs, 'gt', frame, version), dtype=np.float64)/255
        mask = np.asarray(source(cs, 'exact', frame, version)) > 0
        if frame == 0:
            first_mask = mask
        error = np.abs(prediction - gt).mean(axis=-1)
        result[prefix + '_gt_full_mae'] = float(error.mean())
        result[prefix + '_gt_masked_mae'] = metrics.safe_value(float(error[mask].mean()) if mask.any() else None)
        predictions.append(prediction)
    loop = np.abs(predictions[0] - predictions[1]).mean(axis=-1)
    result.update(loop_full_mae=float(loop.mean()),
                  loop_masked_mae=metrics.safe_value(float(loop[first_mask].mean()) if first_mask.any() else None))
    return result


def pair_definitions(version, cs, early):
    result = []
    if (version, cs) in PROBES:
        result.append(('lin300_vs_oldC300', ('C_old', 300), ('C_lin17', 300)))
        if not early:
            result += [('ring300_vs_lin300', ('C_lin17', 300), ('C_ring17', 300)),
                       ('lin600_vs_lin300', ('C_lin17', 300), ('C_lin17', 600)),
                       ('ring600_vs_ring300', ('C_ring17', 300), ('C_ring17', 600))]
    if not early:
        result += [('ring600_vs_lin600', ('C_lin17', 600), ('C_ring17', 600)),
                   ('lin600_vs_oldC300', ('C_old', 300), ('C_lin17', 600)),
                   ('ring600_vs_oldC300', ('C_old', 300), ('C_ring17', 600))]
    return result


def make_pairs(rows, early=False, segment=False):
    result = []
    for version, cs in (PROBES if early else full_inputs()):
        for segment_id in ([s[0] for s in SEGMENTS] if segment else [None]):
            selected = {(r['method'], int(r['checkpoint_step'])): r for r in rows
                        if r['data_version'] == version and r['cs'] == cs and
                        (not segment or r['segment'] == segment_id)}
            for name, old_key, new_key in pair_definitions(version, cs, early):
                old, new = selected[old_key], selected[new_key]
                for key in (ERRORS[:5] if segment else ERRORS + ENDPOINT_KEYS):
                    result.append(dict(data_version=version, cs=cs, split=old['split'],
                        comparison=name, segment=segment_id or '000-079 plus independent endpoints', metric=key,
                        **paired_change(old[key], new[key], key)))
    return result


def group_metrics(buildings):
    rows = []
    for version, group, ids in [('Rev1', 'train8', TRAIN), ('Rev1', 'val4', VAL),
                               ('OriginalHard', 'train3', ['CS09', 'CS10', 'CS11']),
                               ('OriginalHard', 'val1', ['CS12'])]:
        for method, step in [('C_old', 300), ('C_lin17', 600), ('C_ring17', 600)]:
            selected = [r for r in buildings if r['data_version'] == version and r['cs'] in ids
                        and r['method'] == method and int(r['checkpoint_step']) == step]
            require(len(selected) == len(ids), 'Incomplete group')
            row = dict(data_version=version, group=group, method=method, checkpoint_step=step, buildings=len(selected))
            for key in ERRORS + ENDPOINT_KEYS:
                values = [metrics.number(r[key]) for r in selected if metrics.number(r[key]) is not None]
                row[key] = metrics.safe_value(sum(values)/len(values) if values else None)
                row[key + '_valid_buildings'] = len(values)
            rows.append(row)
    return rows


def compute_metrics(early=False):
    frames, buildings, segments = [], [], []
    inputs = PROBES if early else full_inputs()
    old_frames = {v: read_csv((REV1_RUN if v == 'Rev1' else HARD_RUN) / 'metrics/frame_metrics.csv') for v in VERSIONS}
    old_buildings = {v: read_csv((REV1_RUN if v == 'Rev1' else HARD_RUN) / 'metrics/building_metrics.csv') for v in VERSIONS}
    prior_frames = read_csv(RUN / 'metrics/probe300_frame_metrics.csv') if not early else []
    prior_buildings = read_csv(RUN / 'metrics/probe300_building_metrics.csv') if not early else []
    for version, cs in inputs:
        old = next(r for r in old_buildings[version] if r['cs'] == cs and r['method'] == 'C')
        previous = [r for r in old_frames[version] if r['cs'] == cs and r['method'] == 'C']
        require([int(r['frame']) for r in previous] == list(range(80)), 'Old frame metrics incomplete')
        order = [('C_old', 300)] + ([('C_lin17', 300)] if early else
            [(m, s) for m, s, v, b in trajectory_order() if (v, b) == (version, cs)])
        masks, edges = labels(cs, range(80), version)
        for method, step in order:
            common = dict(data_version=version, method=method, checkpoint_step=step)
            reuse = [r for r in prior_frames if r['method'] == method and int(r['checkpoint_step']) == step
                     and r['data_version'] == version and r['cs'] == cs]
            if method == 'C_old':
                current_frames = [dict(r, **common) for r in previous]
                building = dict(old, **common, **endpoints(method, step, version, cs))
            elif reuse:
                require(len(reuse) == 80, 'Incomplete early probe metrics')
                current_frames = reuse
                building = next(r for r in prior_buildings if r['method'] == method and int(r['checkpoint_step']) == step
                                and r['data_version'] == version and r['cs'] == cs)
            else:
                require(valid_prediction(method, step, version, cs), 'Prediction not complete')
                current_frames = []
                for frame in range(80):
                    pred = np.asarray(read_image(prediction_dir(method, step, version, cs) / f'{frame:03}.png'), dtype=np.float64)/255
                    gt = np.asarray(source(cs, 'gt', frame, version), dtype=np.float64)/255
                    damaged = np.asarray(source(cs, 'damaged', frame, version), dtype=np.float64)/255
                    values = metrics.pixel_sums(pred, gt, damaged, masks[frame], edges[frame])
                    current_frames.append(dict(previous[frame], **common, **metrics.reduced(values), **values))
                totals = summed(current_frames)
                valid = {role + '_valid_frames': sum(float(r[role + '_pixels']) > 0 for r in current_frames)
                         for role in ('masked', 'structure', 'known')}
                building = dict(old, **common, **metrics.reduced(totals), **totals, **valid,
                                **endpoints(method, step, version, cs))
            frames.extend(current_frames); buildings.append(building)
            for name, frame_ids in SEGMENTS:
                totals = summed([r for r in current_frames if int(r['frame']) in frame_ids])
                segments.append(dict(cs=cs, split=old['split'], **common, segment=name, frames=len(frame_ids),
                                     **metrics.reduced(totals), **totals))
            print(f'Metrics {method}/{step}/{version}/{cs}: Exact MAE={building["masked_mae"]}; loop={building["loop_full_mae"]}', flush=True)
    prefix = 'probe300_' if early else ''
    for name, rows in [('frame_metrics', frames), ('building_metrics', buildings), ('segment_metrics', segments),
                       ('paired_differences', make_pairs(buildings, early)),
                       ('segment_paired_differences', make_pairs(segments, early, segment=True))]:
        write_csv(RUN / 'metrics' / f'{prefix}{name}.csv', rows)
    endpoint_fields = ['cs', 'split', 'data_version', 'method', 'checkpoint_step', *ENDPOINT_KEYS,
                       'loop_full_mae', 'loop_masked_mae']
    write_csv(RUN / 'metrics' / f'{prefix}endpoints.csv', [{k: r[k] for k in endpoint_fields} for r in buildings])
    if not early:
        groups = group_metrics(buildings)
        write_csv(RUN / 'metrics/group_summary.csv', groups)
        differences = make_pairs(buildings)
        fields = list(differences[0])
        write_csv(RUN / 'metrics/rev1_regressions.csv',
                  [r for r in differences if r['data_version'] == 'Rev1' and r['regressed'] is True], fields)
        write_csv(RUN / 'metrics/unchanged_input_controls.csv', [r for r in differences if r['cs'] in IDS[:8]], fields)
        group_pairs = []
        for start in range(0, len(groups), 3):
            old, linear, ring = groups[start:start+3]
            for name, a, b in [('ring600_vs_lin600', linear, ring), ('lin600_vs_oldC300', old, linear),
                               ('ring600_vs_oldC300', old, ring)]:
                for key in ERRORS + ENDPOINT_KEYS:
                    group_pairs.append(dict(data_version=a['data_version'], group=a['group'], comparison=name,
                                            metric=key, **paired_change(a[key], b[key], key)))
        write_csv(RUN / 'metrics/group_paired_differences.csv', group_pairs)
        for version in VERSIONS:
            write_csv(RUN / '00_看结果' / version / '逐栋指标对比.csv', [r for r in differences if r['data_version'] == version])
    return buildings


def early_probe():
    from media import overview
    buildings = compute_metrics(early=True)
    lines = ['# 线性17帧300步复现检查', '', f'生成时间：{now()}；只使用已完成的四条探针。', '',
             '原初始化、前300行采样及旧C训练/推理设置均固定；完整参数见resolved_config.json。', '',
             '| 输入 | 旧C缺口MAE | lin300缺口MAE | 相对变化 |', '| --- | --- | --- | --- |']
    for version, cs in PROBES:
        pair = {r['method']: r for r in buildings if r['data_version'] == version and r['cs'] == cs}
        delta = paired_change(pair['C_old']['masked_mae'], pair['C_lin17']['masked_mae'], 'masked_mae')
        lines.append(f'| {version}/{cs} | {delta["old"]} | {delta["new"]} | {delta["relative_change_percent"]}% |')
        overview(version, cs, early=True)
    lines += ['', '固定问题帧与整条预测均已保留。数值差异不直接判为结构修复或失败；图像复核待用户后续通知。', '',
              '本检查记录复现差异，不按主观效果筛掉实验组。队列继续线性组至600及循环组600，保持配对预算。',
              '若之后发现复现不一致，报告须同时说明训练采样、缓存/编码路径、恢复过程和数值差异，不能直接归因于窗口。', '']
    (RUN / '00_看结果/300步复现检查/README.md').write_text('\n'.join(lines), encoding='utf-8')
    mark_complete('early_probe', trajectories=4, visual_review='pending_user_request', queue_continues=True)
    print('EARLY PROBE SAVED; continuing both authorized groups through update600', flush=True)


def write_entry(ready=False):
    folder = RUN / '00_看结果'
    folder.mkdir(parents=True, exist_ok=True)
    status = ('两组各600更新、40条81帧预测、3240张原始PNG、指标和48视频已完成；视觉复核与最终报告待用户通知。'
              if ready else '已建立两组串行任务；以下为交付入口，具体进度见progress.json，不表示成果已完成。')
    lines = ['# Complex12 Orbit17：线性窗与循环窗配对实验', '', status, '', f'实际run：`{RUN}`。', '',
        '训练只使用Rev1八栋，OriginalHard只评测；普通C、共同新初始化和独立AdamW，每组600次更新。',
        '线性窗起点0–63；循环窗起点0–79，跨界按模80读完整17帧。训练不读取080，也不强制推理首尾相同。', '',
        '优先打开：OriginalHard CS10（连接面）、CS12（形状与首尾），再看Rev1 CS10是否产生新条纹。', '',
        '| 版本/建筑 | 身份 | lin600原始视频 | ring600原始视频 | 同步对比 |', '| --- | --- | --- | --- | --- |']
    order = [('OriginalHard', 'CS10'), ('OriginalHard', 'CS12'), ('Rev1', 'CS10')]
    order += [p for p in full_inputs() if p not in order]
    for version, cs in order:
        base = f'{version}/{cs}'
        lines.append(f'| {version}/{cs} | {"训练身份" if cs in TRAIN else "验证身份"} | '
                     f'[lin600]({base}/01_C_lin17_step600_补全.mp4) | [ring600]({base}/02_C_ring17_step600_补全.mp4) | '
                     f'[Before/GT/旧C/lin/ring]({base}/03_配对同步对比.mp4) |')
    lines += ['', '[300步提前核对](300步复现检查/README.md)；[采样覆盖图](采样覆盖总览.png)；[覆盖次数明细](../metrics/sampling_coverage.csv)。', '',
        '300/600七行探针总览：']
    for version, cs in PROBES:
        lines.append(f'- [{version}/{cs}](300与600探针对比/{version}_{cs}.png)')
    lines += ['', '指标：[逐栋与端点](../metrics/building_metrics.csv)、[配对变化](../metrics/paired_differences.csv)、'
        '[四组汇总](../metrics/group_summary.csv)、[四组变化](../metrics/group_paired_differences.csv)、'
        '[前中后分段](../metrics/segment_metrics.csv)、[Rev1退步](../metrics/rev1_regressions.csv)。', '',
        '原指标口径不变：常规000–079、080只用于首尾与端点GT检查；楼内像素加权、跨楼等权。闭环低误差不代表结构正确。',
        'Hard固定反例：CS09/006；CS10/024、058；CS11/073；CS12/000、080；另保留Rev1 CS10/020。',
        'Hard CS01–08与Rev1输入相同，复用对应Rev1结果，不额外推理。300步只生成四输入探针，不另做视频。', '',
        '自动编码/解码检查不等于人工播放、闪烁量化或三维验证。两组的视觉结论、最终中文报告及upload便携报告另行收尾；不预设循环组胜出。', '']
    (folder / 'README_从这里开始.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    if sys.argv[1:] == ['--early-probe']:
        early_probe()
        return
    begin = time.monotonic()
    for task in trajectory_order():
        require(valid_prediction(*task), 'Missing final trajectory')
    compute_metrics()
    from media import deliver
    deliver()
    write_entry(ready=True)
    mark_complete('evaluate', trajectories=40, raw_pngs=3240, videos=48, fixed_problem_images=7,
                  probe_overviews=4, elapsed_seconds=time.monotonic()-begin,
                  visual_review='pending_user_request', real_time_playback=False)
    print(f'DELIVERY READY: {RUN / "00_看结果"}; visual review/report pending user follow-up', flush=True)


if __name__ == '__main__':
    main()
