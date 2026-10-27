"""CPU presentation of full-size raw predictions; no prediction postprocessing."""
import subprocess
from PIL import ImageDraw
from complex_common import *

FIXED = {'CS09': [6], 'CS10': [24, 58], 'CS11': [73], 'CS12': [0, 80]}


def pixels(version, cs, method, step, frame):
    if method in ('Before', 'GT'):
        return np.asarray(source(cs, 'gt' if method == 'GT' else 'damaged', frame, version))
    return np.asarray(read_image(prediction_dir(method, step, version, cs) / f'{frame:03}.png'))


def tile(version, cs, method, step, frame):
    image = Image.new('RGB', (512, 544), 'white')
    image.paste(Image.fromarray(pixels(version, cs, method, step, frame)), (0, 32))
    label = method if method in ('Before', 'GT') else f'{method} step{step}'
    ImageDraw.Draw(image).text((6, 5), f'{label} | {cs} B{frame:03}', fill='black', font=legacy('complex_evaluate').font(18))
    return image


def panel(version, cs, frame, early=False):
    order = [('Before', 0), ('GT', 0), ('C_old', 300), ('C_lin17', 300)] if early else [
        ('Before', 0), ('GT', 0), ('C_old', 300), ('C_lin17', 600), ('C_ring17', 600)]
    cols = 2 if early else 3
    canvas = Image.new('RGB', (512 * cols, 1088), 'white')
    for i, (method, step) in enumerate(order):
        canvas.paste(tile(version, cs, method, step, frame), ((i % cols)*512, (i // cols)*544))
    if not early:
        draw = ImageDraw.Draw(canvas)
        draw.text((1044, 584), f'{version} / {cs}', fill='black', font=legacy('complex_evaluate').font(24))
        draw.text((1044, 628), f'B{frame:03} / B080 | 16 FPS', fill='black', font=legacy('complex_evaluate').font(24))
        draw.text((1044, 674), 'Raw predictions / no composite', fill='black', font=legacy('complex_evaluate').font(21))
    return canvas


def save_image(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    with Image.open(path) as check:
        check.load()
        require(check.size == image.size, 'Unreadable comparison image')


def probe_views(version, cs):
    return [0, 24, 40, 58, 80] if (version, cs) == ('OriginalHard', 'CS10') else [0, 20, 40, 60, 80]


def overview(version, cs, early=False):
    order = [('Before', 0), ('GT', 0), ('C_old', 300), ('C_lin17', 300)]
    if not early:
        order += [('C_lin17', 600), ('C_ring17', 300), ('C_ring17', 600)]
    canvas = Image.new('RGB', (2560, 544*len(order)), 'white')
    for row, (method, step) in enumerate(order):
        for col, frame in enumerate(probe_views(version, cs)):
            canvas.paste(tile(version, cs, method, step, frame), (col*512, row*544))
    path = RUN / '00_看结果' / ('300步复现检查' if early else '300与600探针对比') / f'{version}_{cs}.png'
    save_image(canvas, path)
    return path


def encode_and_check(path, image_at, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_name(path.stem + '.writing.mp4')
        if temporary.exists():
            temporary.unlink()  # Only this run's incomplete CPU encoding.
        command = ['ffmpeg', '-nostdin', '-v', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                   '-s', f'{size[0]}x{size[1]}', '-r', '16', '-i', 'pipe:0', '-an',
                   '-vf', 'scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p',
                   '-c:v', 'libx264', '-profile:v', 'high', '-level:v', '4.2', '-crf', '12', '-preset', 'fast',
                   '-threads', '2', '-pix_fmt', 'yuv420p', '-color_range', 'tv', '-colorspace', 'bt709',
                   '-color_trc', 'bt709', '-color_primaries', 'bt709', '-movflags', '+faststart', str(temporary)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for frame in range(81):
                data = np.asarray(image_at(frame), dtype=np.uint8)
                require(data.shape == (size[1], size[0], 3), 'Video source dimensions differ')
                process.stdin.write(data.tobytes())
            process.stdin.close(); process.stdin = None
            _, error = process.communicate(timeout=180)
            require(process.returncode == 0, f'Video encoding failed: {error.decode(errors="replace")}')
        finally:
            if process.poll() is None:
                process.kill(); process.wait()
            if process.stdin is not None:
                process.stdin.close()
            process.stderr.close()
        os.replace(temporary, path)
    info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=width,height,nb_frames,r_frame_rate', '-of', 'json', str(path)], text=True))['streams'][0]
    require((info['width'], info['height']) == size and int(info['nb_frames']) == 81 and info['r_frame_rate'] == '16/1',
            'Video metadata differs')
    process = subprocess.Popen(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(path), '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    worst = 0.0
    try:
        for frame in range(81):
            data = process.stdout.read(size[0]*size[1]*3)
            require(len(data) == size[0]*size[1]*3, 'Incomplete video decode')
            decoded = np.frombuffer(data, dtype=np.uint8).reshape(size[1], size[0], 3)
            expected = np.asarray(image_at(frame), dtype=np.uint8)
            error = float(np.abs(decoded.astype(np.int16)-expected.astype(np.int16)).mean(axis=(0, 1)).max())
            worst = max(worst, error)
            require(error <= 3, f'Video frame/color mismatch: {path.name}/{frame}: {error}')
        require(not process.stdout.read(1), 'Extra video frames')
        error = process.stderr.read()
        require(process.wait() == 0, f'Video decode failed: {error.decode(errors="replace")}')
    finally:
        if process.poll() is None:
            process.kill(); process.wait()
        process.stdout.close()
        process.stderr.close()
    return dict(video=str(path), frames=81, size=list(size), fps=16, max_channel_encoding_mae_255=worst)


def deliver():
    checks = []
    for version, cs in full_inputs():
        folder = RUN / '00_看结果' / version / cs
        print(f'Creating videos: {version}/{cs}; completed {len(checks)}/48', flush=True)
        for i, method in enumerate(METHODS, 1):
            record = encode_and_check(folder / f'0{i}_{method}_step600_补全.mp4',
                lambda f, m=method: pixels(version, cs, m, 600, f), (512, 512))
            checks.append(dict(record, data_version=version, cs=cs, method=method))
        record = encode_and_check(folder / '03_配对同步对比.mp4', lambda f: panel(version, cs, f), (1536, 1088))
        checks.append(dict(record, data_version=version, cs=cs, method='comparison'))
        fixed = FIXED[cs] if version == 'OriginalHard' else ([20] if cs == 'CS10' else [])
        for frame in fixed:
            save_image(panel(version, cs, frame), folder / f'04_固定问题帧_B{frame:03}.png')
    for version, cs in PROBES:
        overview(version, cs)
    require(len(checks) == 48, 'Incomplete video delivery')
    write_json(RUN / 'metrics/delivery_media.json', checks)
    return checks
