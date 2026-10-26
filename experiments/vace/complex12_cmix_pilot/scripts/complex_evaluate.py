"""Paired old-C/C_mix metrics and 2x2 media, using the original pixel metric code."""
import math
import subprocess
import time
from PIL import ImageDraw
from complex_common import *

metrics = legacy('complex_evaluate')
ERRORS = metrics.ERRORS
FIXED = {'CS09': [6], 'CS10': [24, 58], 'CS11': [73], 'CS12': [0, 80]}

def paired_change(old, new, metric):
    a, b = metrics.number(old), metrics.number(new)
    finite = a is not None and b is not None and math.isfinite(a) and math.isfinite(b)
    delta = b-a if finite else None
    relative = 100*delta/abs(a) if finite and a != 0 else None
    worse = (delta < 0 if metric == 'masked_psnr' else delta > 0) if finite else 'N/A'
    return {'old': old, 'new': new, 'absolute_change': metrics.safe_value(delta),
            'relative_change_percent': metrics.safe_value(relative), 'regressed': worse}

def summary(rows, version, group, method):
    result = {'data_version': version, 'group': group, 'method': method, 'buildings': len(rows)}
    for key in ERRORS:
        values = [metrics.number(r[key]) for r in rows if metrics.number(r[key]) is not None]
        result[key] = metrics.safe_value(sum(values)/len(values) if values else None)
        result[key + '_valid_buildings'] = len(values)
    return result

def compute_metrics():
    frames, buildings = [], []
    old_frames = {v: read_csv((REV1_RUN if v == 'Rev1' else HARD_RUN) / 'metrics/frame_metrics.csv') for v in VERSIONS}
    old_buildings = {v: read_csv((REV1_RUN if v == 'Rev1' else HARD_RUN) / 'metrics/building_metrics.csv') for v in VERSIONS}
    for version, cs in trajectory_order():
        old = next(r for r in old_buildings[version] if r['cs'] == cs and r['method'] == 'C')
        building = dict(old, data_version=version, method='C_old', unchanged_input=cs not in HARD_IDS)
        buildings.append(building)
        previous = [r for r in old_frames[version] if r['cs'] == cs and r['method'] == 'C']
        require([r['frame'] for r in previous] == [f'{i:03}' for i in range(80)], 'Old C frame metrics incomplete')
        frames += [dict(r, data_version=version, method='C_old') for r in previous]
        m, e = labels(cs, 0, 80, version)
        total = None
        valid = {'masked_valid_frames': 0, 'structure_valid_frames': 0, 'known_valid_frames': 0}
        for frame in range(80):
            pred = np.asarray(read_image(prediction_dir(version, cs) / f'{frame:03}.png'), dtype=np.float64)/255
            gt = np.asarray(source(cs, 'gt', frame, version), dtype=np.float64)/255
            damaged = np.asarray(source(cs, 'damaged', frame, version), dtype=np.float64)/255
            sums = metrics.pixel_sums(pred, gt, damaged, m[frame], e[frame])
            if total is None:
                total = {k: 0 for k in sums}
            for key, value in sums.items():
                total[key] += value
            for role in ('masked', 'structure', 'known'):
                valid[role + '_valid_frames'] += int(sums[role + '_pixels'] > 0)
            frames.append(dict(previous[frame], data_version=version, method='C_mix', **metrics.reduced(sums), **sums))
        first, last = [np.asarray(read_image(prediction_dir(version, cs) / f'{i:03}.png'), dtype=np.float64)/255 for i in (0, 80)]
        loop = np.abs(first-last).mean(axis=-1)
        current = dict(building, method='C_mix', **metrics.reduced(total), **total, **valid)
        current.update(loop_full_mae=float(loop.mean()), loop_masked_mae=metrics.safe_value(float(loop[m[0]].mean()) if m[0].any() else None))
        buildings.append(current)
        print(f'Paired metrics {version}/{cs}: C={old["masked_mae"]}; C_mix={current["masked_mae"]}', flush=True)
    write_csv(RUN / 'metrics/frame_metrics.csv', frames)
    write_csv(RUN / 'metrics/building_metrics.csv', buildings)
    differences = []
    for version, cs in trajectory_order():
        pair = {r['method']: r for r in buildings if r['data_version'] == version and r['cs'] == cs}
        for key in ERRORS:
            differences.append({'data_version': version, 'cs': cs, 'split': pair['C_old']['split'],
                'unchanged_input': cs not in HARD_IDS, 'metric': key,
                **paired_change(pair['C_old'][key], pair['C_mix'][key], key)})
    write_csv(RUN / 'metrics/paired_differences.csv', differences)
    regressions = [r for r in differences if r['data_version'] == 'Rev1' and r['regressed'] is True]
    write_csv(RUN / 'metrics/rev1_regressions.csv', regressions, list(differences[0]))
    write_csv(RUN / 'metrics/unchanged_controls.csv', [r for r in differences if r['unchanged_input']], list(differences[0]))
    summaries = []
    for version, group, ids in [('Rev1', 'train8', TRAIN), ('Rev1', 'val4', VAL),
                               ('OriginalHard', 'train3', HARD_IDS[:3]), ('OriginalHard', 'val1', ['CS12']),
                               ('Rev1', 'unchanged_train5', [cs for cs in TRAIN if cs not in HARD_IDS]),
                               ('Rev1', 'unchanged_val3', [cs for cs in VAL if cs not in HARD_IDS])]:
        for method in ('C_old', 'C_mix'):
            selected = [r for r in buildings if r['data_version'] == version and r['cs'] in ids and r['method'] == method]
            summaries.append(summary(selected, version, group, method))
    write_csv(RUN / 'metrics/group_summary.csv', summaries)
    summary_pairs = []
    for old, new in zip(summaries[::2], summaries[1::2]):
        for key in ERRORS:
            summary_pairs.append({'data_version': old['data_version'], 'group': old['group'], 'buildings': old['buildings'],
                                  'metric': key, **paired_change(old[key], new[key], key)})
    write_csv(RUN / 'metrics/group_paired_differences.csv', summary_pairs)
    for version in VERSIONS:
        folder = RUN / '00_看结果' / version
        folder.mkdir(parents=True, exist_ok=True)
        write_csv(folder / '逐栋指标对比.csv', [r for r in differences if r['data_version'] == version])
    write_csv(RUN / '00_看结果/Rev1/退步指标明细.csv', regressions, list(differences[0]))
    return buildings

def pixels(version, cs, method, frame):
    if method in ('Before', 'GT'):
        return np.asarray(source(cs, 'gt' if method == 'GT' else 'damaged', frame, version))
    return np.asarray(read_image(prediction_dir(version, cs, old=method == 'C_old') / f'{frame:03}.png'))

def panel(version, cs, frame):
    canvas = Image.new('RGB', (1024, 1088), 'white')
    draw = ImageDraw.Draw(canvas)
    for position, (method, label) in enumerate([('Before', 'Before (Damaged)'), ('GT', 'GT (Reference)'),
                                              ('C_old', 'Old C | Rev1 trained'), ('C_mix', 'C_mix | 250 Rev1 + 50 Hard')]):
        x, y = (position % 2)*512, (position // 2)*544
        canvas.paste(Image.fromarray(pixels(version, cs, method, frame)), (x, y+32))
        draw.text((x+6, y+5), f'{label}  {cs} B{frame:03}', fill='black', font=metrics.font(17))
    return canvas

def encode_and_check(path, version, cs, comparison):
    size = (1024, 1088) if comparison else (512, 512)
    def image_at(frame):
        return np.asarray(panel(version, cs, frame)) if comparison else pixels(version, cs, 'C_mix', frame)
    if not path.exists():
        temp = path.with_name(path.stem + '.writing.mp4')
        # Only this run's incomplete video is replaced, never a raw prediction.
        if temp.exists():
            temp.unlink()
        command = ['ffmpeg', '-nostdin', '-v', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                   '-s', f'{size[0]}x{size[1]}', '-r', '16', '-i', 'pipe:0', '-an',
                   '-vf', 'scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p',
                   '-c:v', 'libx264', '-profile:v', 'high', '-level:v', '4.2', '-crf', '12', '-preset', 'fast',
                   '-threads', '2', '-pix_fmt', 'yuv420p', '-color_range', 'tv', '-colorspace', 'bt709',
                   '-color_trc', 'bt709', '-color_primaries', 'bt709', '-movflags', '+faststart', str(temp)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for frame in range(81):
                process.stdin.write(image_at(frame).tobytes())
            process.stdin.close(); process.stdin = None
            _, err = process.communicate(timeout=180)
            require(process.returncode == 0, f'Video encoding failed: {err.decode(errors="replace")}')
        finally:
            if process.poll() is None:
                process.kill(); process.wait()
        os.replace(temp, path)
    info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=width,height,nb_frames,r_frame_rate', '-of', 'json', str(path)], text=True))['streams'][0]
    require((info['width'], info['height']) == size and int(info['nb_frames']) == 81 and info['r_frame_rate'] == '16/1', 'Video metadata mismatch')
    process = subprocess.Popen(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(path), '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    worst = 0.0
    try:
        for frame in range(81):
            data = process.stdout.read(size[0]*size[1]*3)
            require(len(data) == size[0]*size[1]*3, 'Incomplete video decode')
            decoded = np.frombuffer(data, dtype=np.uint8).reshape(size[1], size[0], 3)
            error = float(np.abs(decoded.astype(np.int16)-image_at(frame).astype(np.int16)).mean(axis=(0, 1)).max())
            worst = max(worst, error)
            require(error <= 3, f'Video frame/color mismatch: {version}/{cs}/{frame}: {error}')
        require(not process.stdout.read(1), 'Extra video frames')
        err = process.stderr.read()
        require(process.wait() == 0, f'Decode failed: {err.decode(errors="replace")}')
    finally:
        if process.poll() is None:
            process.kill(); process.wait()
    return {'data_version': version, 'cs': cs, 'video': str(path), 'frames': 81, 'size': list(size), 'fps': 16,
            'max_channel_encoding_mae_255': worst}

def write_entry(ready=False):
    entry = RUN / '00_看结果'
    state = '300更新、16条原始预测、指标与32个视频已生成；视觉结论及最终报告等待用户通知复核。' if ready else '队列准备就绪；以下成果入口在对应阶段完成后可用，当前不代表实验完成。'
    lines = ['# Complex12 C_mix 固定预算混合训练', '', state, '', f'实际run：`{RUN}`。', '',
             '训练8栋/验证4栋划分不变；250次Rev1、50次OriginalHard。两版改变了缺损位置及连接线索，不是同一缺口单纯扩大。', '',
             '优先查看 OriginalHard CS10（连接面）和 CS12（折面/闭环）；不要只用平均MAE判断结构。', '',
             '| 版本/建筑 | 身份 | C_mix原始补后视频 | Before/GT/旧C/C_mix同步对比 |', '| --- | --- | --- | --- |']
    order = [('OriginalHard', cs) for cs in ('CS10', 'CS12', 'CS09', 'CS11')] + [('Rev1', cs) for cs in IDS]
    for version, cs in order:
        folder = f'{version}/{cs}'
        lines.append(f'| {version}/{cs} | {"训练身份" if cs in TRAIN else "验证身份"} | [原始补后]({folder}/01_C_mix_补全.mp4) | [四画面对比]({folder}/02_旧C_C_mix_GT_对比.mp4) |')
    lines += ['', 'Hard固定问题帧：CS09/B006；CS10/B024、B058；CS11/B073；CS12/B000、B080，保存在对应文件夹。', '',
              'OriginalHard CS01–08与Rev1输入相同，展示目录链接到对应Rev1结果，不额外推理。', '',
              '指标入口：[Rev1逐栋](Rev1/逐栋指标对比.csv)、[Rev1退步明细](Rev1/退步指标明细.csv)、[Hard逐栋](OriginalHard/逐栋指标对比.csv)、[分组汇总](../metrics/group_summary.csv)、[分组新旧变化](../metrics/group_paired_differences.csv)、[输入未变对照](../metrics/unchanged_controls.csv)。', '',
              '所有误差沿用旧口径：000–079常规统计、000/080独立闭环；楼内像素加权、跨楼等权；零基准不计算相对百分比。Hard只有CS12一个未训练建筑，属于开发验证。', '',
              '按用户后续要求，本轮启动后结束对话；后续视觉复核、最终中文报告、upload便携报告另行通知后完成。本入口不提前写效果结论。', '']
    (entry / 'README_从这里开始.md').write_text('\n'.join(lines), encoding='utf-8')

def main():
    begin = time.monotonic()
    for version, cs in trajectory_order():
        require(valid_prediction(version, cs), 'Missing prediction')
    compute_metrics()
    checks = []
    for version, cs in trajectory_order():
        folder = RUN / '00_看结果' / version / cs
        folder.mkdir(parents=True, exist_ok=True)
        print(f'Creating videos: {version} / {cs}', flush=True)
        checks.append(encode_and_check(folder / '01_C_mix_补全.mp4', version, cs, False))
        checks.append(encode_and_check(folder / '02_旧C_C_mix_GT_对比.mp4', version, cs, True))
        if version == 'OriginalHard':
            for frame in FIXED[cs]:
                path = folder / f'03_问题帧_B{frame:03}.png'
                panel(version, cs, frame).save(path)
                with Image.open(path) as im:
                    im.load(); require(im.size == (1024, 1088), 'Problem frame unreadable')
    for cs in IDS[:8]:
        for base in (RUN / '00_看结果', RUN / 'predictions'):
            link = base / 'OriginalHard' / cs
            if not link.exists():
                link.symlink_to(Path('../Rev1') / cs, target_is_directory=True)
    write_json(RUN / 'metrics/delivery_media.json', checks)
    write_entry(ready=True)
    mark_complete('evaluate', trajectories=16, raw_pngs=1296, videos=32, fixed_problem_images=6,
                  elapsed_seconds=time.monotonic()-begin, visual_review='pending_user_request', real_time_playback=False)
    print(f'SAVED: {RUN / "00_看结果"}; metrics and media readable; visual report pending user request', flush=True)

if __name__ == '__main__':
    main()
