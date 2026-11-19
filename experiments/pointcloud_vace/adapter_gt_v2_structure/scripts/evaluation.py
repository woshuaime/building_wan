"""Manual original-eight validation with resumable, verified video publication."""

import os
from pathlib import Path
import shutil
import sys
import time

import av
import numpy as np
import torch
from safetensors.torch import load_file

from structure_data import BASE, OUTPUTS, ROOT, SEED, TRAIN_COUNT, VALIDATION_IDS, PreparedData, data, now, sha, write_json
import integration
import validate_video as video
import validation_delivery as delivery
from validation_queue import verify_video
from validation_speed import ValidationSpeed

from objectives import camera_tensor, observed_tensor
from quality import appearance, decode_prediction, edge_image_support, geometry_scores, luminance
from train import geometry_for_window, make_model

sys.path.append(str(ROOT / "experiments/pointcloud_vace/adapter_gt_v2_roof"))
import roof

VAE_REFERENCES = ("2825", "5954")


def directory(run, epochs):
    return run / ".work" / f"validation_{epochs}"


def final_dir(run, epochs):
    return run / f"交付_{epochs}轮"


def _paths(run, epochs, i, building):
    name = f"{i:02d}_ID{building}.mp4"
    result = final_dir(run, epochs)
    return result / "Prediction" / name, result / "验证对照" / name


def verify_prediction(path, identity):
    with av.open(str(path)) as container:
        if container.metadata.get("comment") != identity or len(container.streams) != 1:
            raise ValueError(f"Incorrect video identity: {path}")
        stream = container.streams.video[0]
        codec = stream.codec_context
        from av.video.reformatter import ColorPrimaries, ColorRange, ColorTrc, Colorspace
        if (codec.name != "h264" or codec.profile != "High" or codec.format.name != "yuv420p"
                or stream.average_rate != 16 or codec.colorspace != Colorspace.ITU709
                or codec.color_range != ColorRange.MPEG or codec.color_primaries != ColorPrimaries.BT709
                or codec.color_trc != ColorTrc.BT709):
            raise ValueError(f"Unexpected prediction format: {path}")
        count = 0
        for frame in container.decode(stream):
            if (frame.is_corrupt or (frame.width, frame.height) != (832, 480) or frame.pts is None
                    or frame.pts * frame.time_base * 16 != count):
                raise ValueError(f"Invalid prediction frame {count} in {path}")
            count += 1
        if count != 81:
            raise ValueError(f"Expected 81 decoded prediction frames: {path}")


def _completion(run, epochs, prepared):
    recorded = data.read_json(run / f"training_{epochs}.json")
    if recorded["updates"] != epochs * TRAIN_COUNT or recorded["dataset"] != prepared.summary["schedule_sha256"]:
        raise ValueError("Training completion is missing its exact budget or dataset identity")
    for weight in recorded["weights"].values():
        if sha(run / weight["file"]) != weight["sha256"]:
            raise ValueError(f"Weight changed: {weight['file']}")
    return recorded


def _identity(run, epochs, building, completion):
    digest = completion["weights"]["structure"]["sha256"]
    return f"GT-V2-STRUCTURE|{run.name}|{epochs}|{building}|{digest}|81f-30s-cfg5-seed20261018"


def _model(run, completion):
    model, adapter = make_model()
    pipe = model[0]
    values = {key: load_file(str(run / value["file"])) for key, value in completion["weights"].items()}
    pipe.structure_adapter.requires_grad_(True)
    for name, parameter in pipe.vace.named_parameters():
        if ".lora_" in name:
            parameter.requires_grad_(True)
    integration.load_increments_state(pipe, values["adapter"], values["lora"])
    pipe.eval().requires_grad_(False)
    adapter.load_state_dict(values["structure"], strict=True)
    return model, adapter.eval().requires_grad_(False)


def _case_ready(run, epochs, building):
    return (directory(run, epochs) / building / "rendered.json").is_file()


def rendered(run, epochs):
    root = directory(run, epochs)
    return (all(_case_ready(run, epochs, building) for building in VALIDATION_IDS)
            and all((root / f"vae_{building}.npy").is_file() for building in VAE_REFERENCES))


def _publish(run, epochs, i, building, identity, verify=False):
    local = directory(run, epochs) / building
    pred, triptych = _paths(run, epochs, i, building)
    pairs = ((local / "prediction.mp4", pred, verify_prediction),
             (local / "comparison.mp4", triptych, verify_video))
    for src, dst, check in pairs:
        if dst.exists():
            if verify:
                check(dst, identity)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(src, dst)


def _save_geometry(path, building, observed):
    def array(tensor):
        return tensor.detach().float().cpu().numpy()

    midpoint, scale = observed["midpoint"], observed["scale"]
    weight = building.point_weights(hard=True).bool()
    np.savez(path, vertices=array(building.vertices) * scale + midpoint,
             pairs=building.pairs.T.cpu().numpy(), edge_scores=array(building.edge_logits.softmax(-1)),
             corner_scores=array(building.vertex_logits.sigmoid()), points=array(building.points[weight]) * scale + midpoint)


def render(run, epochs, progress):
    """Only the sampling/codec-reference phase needs the project's GPU lock."""
    prepared = PreparedData(run)
    completion = _completion(run, epochs, prepared)
    root = directory(run, epochs)
    root.mkdir(parents=True, exist_ok=True)
    model, adapter = _model(run, completion)
    pipe, core, _ = model
    policy = {}
    for i, building in enumerate(VALIDATION_IDS, 1):
        progress.stop()
        identity = _identity(run, epochs, building, completion)
        if _case_ready(run, epochs, building):
            _publish(run, epochs, i, building, identity, verify=True)
            continue
        case = root / building
        # A sealed case reuses verified outputs. Unsealed media was interrupted
        # before both clips finished and belongs only to this run.
        if case.exists():
            shutil.rmtree(case)
        case.mkdir()
        row = delivery.source_row(building)
        observed, = prepared.read(building, "observed")
        frames = list(range(81))
        observed_geometry = geometry_for_window(observed, frames)
        cam = camera_tensor(dict(centers=observed["centers"], rotations=observed["rotations"],
                                 intrinsics=observed["intrinsics"], midpoint=observed["midpoint"],
                                 scale=observed["scale"]), pipe.device)
        progress.update(announce=True, stage="validation", building=building, videos=i-1, sampling_step=0)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            framework = adapter(observed_tensor(observed, pipe.device))
            hints = adapter.project(framework, cam, hard=True)
            _save_geometry(case / "geometry.npz", framework, observed)
        del framework

        def on_step(step):
            progress.stop()
            progress.update(sampling_step=step)

        def projection(tokens):
            value = pipe.structure_adapter.structure_proj(tokens) + hints
            return lambda _tokens: value

        partial = video.source_frames(row["partial_video"])
        started = time.monotonic()
        with ValidationSpeed(pipe, core, policy, lambda value: write_json(root / "speed.json", value)) as speed:
            prediction = video.predict_video(model, row, partial, speed, geometry=observed_geometry,
                                             make_projection=projection, on_step=on_step)
        progress.update(stage="encoding")
        # The finalized model prediction already exists before reading the GT.
        if sha(row["GT_video"]) != row["gt_video_sha256"]:
            raise ValueError(f"Validation GT changed: {building}")
        gt = video.source_frames(row["GT_video"])
        video.save_comparison(case / "comparison.mp4", building, partial, prediction, gt, identity)
        video.encode_rgb_video(case / "prediction.mp4", prediction, (832, 480), identity)
        verify_video(case / "comparison.mp4", identity)
        verify_prediction(case / "prediction.mp4", identity)
        write_json(case / "rendered.json", dict(identity=identity, seconds=time.monotonic() - started))
        _publish(run, epochs, i, building, identity)
        progress.update(announce=True, stage="validation", videos=i, sampling_step=30)
        del partial, prediction, gt, hints

    # Encode/decode the real GT with the frozen production VAE. These frames
    # are used only to interpret the loss of point detail in the final report.
    for building in VAE_REFERENCES:
        progress.stop()
        destination = root / f"vae_{building}.npy"
        if not destination.exists():
            row = delivery.source_row(building)
            gt = video.source_frames(row["GT_video"])
            with torch.inference_mode():
                latent = pipe.vae.encode(pipe.preprocess_video(gt), tiled=False, device=pipe.device)
                decoded = pipe.vae.decode(latent, tiled=False, device=pipe.device)
                frames = np.stack([np.asarray(img) for img in pipe.vae_output_to_video(decoded)])
            staged = destination.with_suffix(".tmp.npy")
            np.save(staged, frames, allow_pickle=False)
            os.replace(staged, destination)
            del latent, decoded, frames
            pipe.vae.model.clear_cache()


def _metrics(run, epochs, prepared, i, building):
    row = delivery.source_row(building)
    root = directory(run, epochs)
    observed, targets = prepared.read(building, "observed", "targets")
    gt = np.stack([np.asarray(frame)[..., 0] for frame in video.source_frames(row["GT_video"])])
    name = f"{i:02d}_ID{building}"
    baseline = BASE / "Prediction_八栋" / f"{name}_Prediction.mp4"
    previous = OUTPUTS.parent / "adapter_gt_v2_roof/run_20261120_014636/Prediction" / f"{name}.mp4"
    clips = dict(gt=gt, v2=decode_prediction(baseline), new=decode_prediction(_paths(run, epochs, i, building)[0]))
    if previous.is_file():
        clips["roof_pilot"] = decode_prediction(previous)
    vae = root / f"vae_{building}.npy"
    if vae.is_file():
        clips["vae_gt"] = np.load(vae, allow_pickle=False)
    gray = {role: luminance(rgb) for role, rgb in clips.items()}
    roof_input = {role: clip for role, clip in gray.items() if role != "gt"}
    camera, record = (data.read_json(row[key]) for key in ("camera", "record"))
    roof_values = roof.scores(building, camera, record, gray["gt"], roof_input)
    with np.load(root / building / "geometry.npz", allow_pickle=False) as shape:
        geometry = geometry_scores(shape, targets, observed["scale"])
    return dict(building=building, geometry=geometry, roof=roof_values,
                appearance={role: appearance(rgb, gray["gt"]) for role, rgb in clips.items()},
                rgb_ridge=edge_image_support(gray, targets, observed))


def _short(value):
    return "N/A" if value is None else f"{value:.3f}"


def _report(run, epochs, completion, results):
    comparison, graphs, rgb_ridges, codec, timing = [], [], [], [], []
    for result in results:
        building = result["building"]
        a, b, c = (result["appearance"][key] for key in ("v2", "new", "gt"))
        old, new = (result["roof"][key] for key in ("v2", "new"))
        comparison.append(f"| {building} | {a['point_core_fraction']:.2%} → {b['point_core_fraction']:.2%} / "
                          f"{c['point_core_fraction']:.2%} | {a['local_cv']:.3f} → {b['local_cv']:.3f} / "
                          f"{c['local_cv']:.3f} | {a['color_error']:.2f} → {b['color_error']:.2f} | "
                          f"{a['background_fill']:.2%} → {b['background_fill']:.2%} | "
                          f"{old['wall_iou']:.3f} → {new['wall_iou']:.3f} | "
                          f"{old['motion']:.2f} → {new['motion']:.2f} | {old['loop']:.2f} → {new['loop']:.2f} |")
        g = result["geometry"]
        graphs.append(f"| {building} | {g['point_count']} / {g['gt_point_count']} | "
                      f"{g['ridge_matches']} / {g['ridge_gt']} | {g['ridge_predicted']} | "
                      f"{_short(g['ridge_endpoint_error_world'])} | {g['surface_coverage_distance_world']:.3f} |")
        old, new = (result["rgb_ridge"][key] for key in ("v2", "new"))
        rgb_ridges.append(f"| {building} | {old['detected']} → {new['detected']} / {new['eligible']} | "
                          f"{_short(old['position_px'])} → {_short(new['position_px'])} | "
                          f"{_short(old['straightness_p95_px'])} → {_short(new['straightness_p95_px'])} |")
        if "vae_gt" in result["appearance"]:
            vae = result["appearance"]["vae_gt"]
            codec.append(f"| {building} | {c['point_core_fraction']:.2%} | {vae['point_core_fraction']:.2%} | "
                         f"{result['roof']['vae_gt']['loop']:.2f} |")
        record = data.read_json(directory(run, epochs) / building / "rendered.json")
        timing.append(record["seconds"])
    weights = "\n".join(f"- {key} · {item['file']} · SHA-256 {item['sha256']}"
                        for key, item in completion["weights"].items())
    reviews = []
    for epoch in range(1, epochs + 1):
        record = data.read_json(run / f"epoch_{epoch}_review.json")["windows"]
        averages = [np.mean([sample[key] for sample in record.values()]) for key in ("framework", "density", "chroma", "track")]
        reviews.append(f"| {epoch} | " + " | ".join(f"{value:.4f}" for value in averages) + " |")
    return f"""# GT V2 三维框架与点分布联合训练

固定 run 为 {run.name}。东京时间 {now()} 生成正式八栋成片和自动指标。人工逐帧结构复核待完成。当前范围覆盖原来八个验证 ID。

## 训练和资源

从完整三轮 GT V2 的 Adapter 与 VACE 注意力 LoRA 初始化。新分支从残缺实测点估计角点、屋脊及面连接，在预测面上生成整圈共享的点。点投影的深度和分类保留后方点的信息。GT OBJ、完整 GT PLY、原视频固定显示点和 GT RGB 全用于监督及推理后的评价。最后一个画面由 VACE 正常解码。

原 train 10,747 栋、每栋每轮连续 17 帧、共 {epochs} 轮与 {completion['updates']:,} 次更新。训练 seed {SEED}，batch 1，832×480，AdamW，裁剪 1。新分支、原 Adapter、VACE LoRA 的初始峰值学习率为 1e-4、3e-5、1e-5。原区域均衡的 Wan Flow Matching 与角点/连边、点分布、点预算共同监督。每八步在原 17 帧 latent 的 224×224 视窗可反传解码一次，监督亮度、色度、多尺度占据以及近、远帧的同点变化。

训练记录累计 {completion['elapsed_seconds']/3600:.2f} 小时，PyTorch 峰值保留显存 {completion['peak_vram_bytes']/(1024**3):.2f} GiB。八条预测与三栏成片编码、核验累计 {sum(timing)/60:.1f} 分钟。每条 81 帧、832×480、16 fps、30 步、CFG 5、shift 5、seed 20261018，H.264 High BT.709。独立媒体与三栏均已逐帧核验。

{weights}

## 每轮固定验证窗口

八栋同一组窗口、同一加噪条件，表中均值用于观察趋势。正式视频的生成从固定随机噪声开始。

| 轮次 | 结构损失 | 密度损失 | 色偏损失 | 跨视角损失 |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(reviews)}

## 正式成片

深色点芯比例按灰度低于 150 计算。密度 CV 用 GT 前景覆盖的 16×16 小区。色差指 GT 点纹区 RGB 最大通道差。墙占据、共同可见屋顶点的邻帧灰度差、首尾屋面差沿用旧屋顶指标。箭头为旧 V2 → 本轮，斜线右侧为 GT。逐栋机器可读指标也保存了点纹连通分量、二维间距及旧屋顶增量对照。

| 建筑 | 深色点芯 | 密度 CV | 点区色差 | 背景误填 | 墙占据 IoU | 屋顶邻帧差 | 屋顶首尾差 |
| --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(comparison)}

| 建筑 | 预测点 / GT 显示点 | 匹配屋脊 / GT 屋脊 | 预测屋脊 | 匹配段端点偏差 | 完整表面覆盖距离 |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(graphs)}

上表是内部三维框架与点集的结果，距离以数据坐标单位计。匹配阈值为 0.20 与观测半径 2% 的较大者。漏检和增边可由各列核对。

| 建筑 | 视频中检出 / 可测视角段 | 线位置绝对偏差 px | 预测线自身直度 P95 px |
| --- | --- | --- | --- |
{chr(10).join(rgb_ridges)}

视频直度是按 GT 可测的投影邻域做点纹梯度估计，并用预测自己拟合线的垂距。位置匹配独立计分。无点纹证据的视角段不评分，可测中消失的段计漏检。稀疏点无法辨认的屋脊需人工复看。

## 原 GT 的 VAE 参照

| 建筑 | 原 GT 点芯比例 | 原 GT 经 VAE 点芯比例 | 原 GT 经 VAE 首尾屋顶差 |
| --- | ---: | ---: | ---: |
{chr(10).join(codec)}

每栋仅一个推理种子。原八栋的结果不能代替新留出集上的泛化评估。
"""


def review(run, epochs, progress):
    prepared = PreparedData(run)
    completion = _completion(run, epochs, prepared)
    root = directory(run, epochs)
    progress.update(announce=True, stage="scoring", videos=len(VALIDATION_IDS))
    results = []
    for i, building in enumerate(VALIDATION_IDS, 1):
        progress.stop()
        identity = _identity(run, epochs, building, completion)
        _publish(run, epochs, i, building, identity, verify=True)
        metrics = root / building / "metrics.json"
        if not metrics.exists():
            write_json(metrics, _metrics(run, epochs, prepared, i, building))
        results.append(data.read_json(metrics))
        progress.update(building=building, scored=i)
    report = _report(run, epochs, completion, results)
    places = (final_dir(run, epochs) / "实验报告.md",
              ROOT / "upload" / f"{run.name[4:12]}_GT视频V2_三维框架_{run.name[13:]}_{epochs}轮_实验报告.md")
    for path in places:
        staged = path.with_suffix(".tmp")
        staged.write_text(report, encoding="utf-8")
        os.replace(staged, path)
    progress.update(announce=True, stage="complete", report=str(places[0]),
                    visual_review="pending", videos=len(VALIDATION_IDS))


if __name__ == "__main__":
    from run import Paused, Progress

    directory_name, target_epoch, owner_fd, gpu_fd = sys.argv[1:]
    work = Path(directory_name)
    expected = ((owner_fd, work / ".run.lock"), (gpu_fd, ROOT / "logs/.wan_a6000_gpu.lock"))
    for descriptor, path in expected:
        inherited, original = os.fstat(int(descriptor)), path.stat()
        if (inherited.st_dev, inherited.st_ino) != (original.st_dev, original.st_ino):
            raise ValueError("Validation worker requires the inherited run and GPU leases")
    status = Progress(work, "starting", action="validate", target_epochs=int(target_epoch))
    status.update(announce=True)
    try:
        render(work, int(target_epoch), status)
    except Paused:
        sys.exit(75)
