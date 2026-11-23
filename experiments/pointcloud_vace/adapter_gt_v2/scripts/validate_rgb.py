"""Render and score the original eight native V2 orbits after the RGB continuation."""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

import data
from common import ROOT, now, sha, write_json
import validate_video as video
import validation_delivery as delivery
from rgb_recipe import IDS, TOTAL, PARALLAX_OUT
from validation_queue import verify_video

BASE = ROOT / "outputs/pointcloud_vace/adapter_gt_v2" / delivery.RUN_NAME
OUTPUT = "验证_八栋"


class Paused(Exception):
    pass


def stop_if_requested(run):
    if (run / "stop.request").exists():
        raise Paused()


def progress(run, stage, completed, building=None, step=0, **detail):
    write_json(run / "validation_progress.json", dict(time=now(), stage=stage, completed=completed,
               total=len(IDS), building=building, step=step, steps=video.STEPS, **detail))


def completion(run):
    done = data.read_json(run / "training_complete.json")
    if done["optimizer_updates"] != TOTAL:
        raise ValueError("Eight-building validation requires the 32,241-update weights")
    for role in ("adapter", "lora"):
        path = run / ("交付" if run.parent == PARALLAX_OUT else "交付_训练完成") / f"{role}_final.safetensors"
        if sha(path) != done[f"{role}_sha256"]:
            raise ValueError(f"Incompatible final {role} weight")
        done[f"{role}_final"] = str(path)
    return done


def case_dir(run, i, building):
    return run / ("交付" if run.parent == PARALLAX_OUT else OUTPUT) / f"{i:02d}_ID{building}"


def identity(run, building, done):
    return "|".join(("GT-V2-RGB", run.name, building, done["adapter_sha256"], done["lora_sha256"],
                     "832x480-81f-30s-cfg5-shift5-seed20261018"))


def verify_case(folder, tag):
    verify_video(folder / "Prediction.mp4", tag, comparison=False)
    verify_video(folder / "三栏对比.mp4", tag)


def render(run):
    from validation_speed import ValidationSpeed

    done = completion(run)
    out = case_dir(run, 1, IDS[0]).parent
    out.mkdir(exist_ok=True)
    work = run / ".validation"
    work.mkdir(exist_ok=True)
    speed_file = work / "speed.json"
    speed_policy = data.read_json(speed_file) if speed_file.exists() else {}
    if run.parent == PARALLAX_OUT and not speed_policy:
        speed_policy = dict(selected="accelerated", source="V2 RGB 20261120 paired CFG release")
        write_json(speed_file, speed_policy)
    model = None
    for i, building in enumerate(IDS, 1):
        stop_if_requested(run)
        tag = identity(run, building, done)
        folder = case_dir(run, i, building)
        if folder.exists():
            progress(run, "verifying", i - 1, building, 30)
            verify_case(folder, tag)
            progress(run, "verified", i, building, 30)
            continue
        progress(run, "loading", i - 1, building)
        if model is None:
            cfg = data.read_json(run / "config.json") if run.parent == PARALLAX_OUT else delivery.model_config()
            model = video.load_validation_model(run, release=(cfg, done))
        row = delivery.source_row(building)
        partial = video.source_frames(row["partial_video"])
        started = time.monotonic()
        last_tick = None
        samples = deque(maxlen=5)
        progress(run, "sampling", i - 1, building)

        def step(n):
            nonlocal last_tick
            stop_if_requested(run)
            tick = time.monotonic()
            if last_tick is not None:
                samples.append(tick - last_tick)
            last_tick = tick
            sec = sum(samples) / len(samples) if samples else None
            progress(run, "sampling", i - 1, building, n, seconds_per_step=sec,
                     eta_seconds=None if sec is None else sec * (30 * (len(IDS) - i + 1) - n))

        def projection(tokens):
            # Every step and both guidance branches see the same frozen projection.
            projected = model[0].structure_adapter.structure_proj(tokens)
            return lambda unused_tokens: projected

        with ValidationSpeed(model[0], model[1], speed_policy, lambda policy: write_json(speed_file, policy)) as speed:
            pred = video.predict_video(model, row, partial, speed, on_step=step, make_projection=projection)
        progress(run, "encoding", i - 1, building, 30)
        # Read complete GT only after the final model frames exist.
        if sha(row["GT_video"]) != row["gt_video_sha256"]:
            raise ValueError(f"Validation reference changed: {building}")
        gt = video.source_frames(row["GT_video"])
        with tempfile.TemporaryDirectory(dir=work) as tempdir:
            stage = Path(tempdir) / "ready"
            stage.mkdir()
            video.encode_rgb_video(stage / "Prediction.mp4", pred, (video.WIDTH, video.HEIGHT), tag)
            video.save_comparison(stage / "三栏对比.mp4", building, partial, pred, gt, tag)
            progress(run, "verifying", i - 1, building, 30)
            verify_case(stage, tag)
            os.rename(stage, folder)  # Both completed videos become visible together.
        write_json(work / f"seconds_{building}.json", {"seconds": time.monotonic() - started})
        progress(run, "verified", i, building, 30)
        del partial, pred, gt
    write_json(run / "validation_rendered.json", dict(time=now(), completed=len(IDS), ids=list(IDS),
               adapter_sha256=done["adapter_sha256"], lora_sha256=done["lora_sha256"]))


def scoring_modules():
    # The structural labels and measurements are shared analysis code; validation
    # loads no structure predictor and supplies only original observed V2 inputs.
    experiments = Path(__file__).resolve().parents[2]
    for path in (experiments / "adapter_gt_v2_structure/scripts", experiments / "adapter_gt_v2_roof"):
        if str(path) not in sys.path:
            sys.path.append(str(path))
    import mesh_labels
    import quality
    import roof
    return mesh_labels, quality, roof


def score(run, i, building, *, rgb_baseline=None, extra_metrics=None):
    from scipy import ndimage

    labels, quality, roof = scoring_modules()
    row = delivery.source_row(building)
    camera, record = (data.read_json(row[k]) for k in ("camera", "record"))
    gt = np.stack([np.asarray(f)[..., 0] for f in video.source_frames(row["GT_video"])])
    partial = np.stack([np.asarray(f)[..., 0] for f in video.source_frames(row["partial_video"])])
    path = case_dir(run, i, building) / "Prediction.mp4"
    paths = {"new": path, "old": BASE / "Prediction_八栋" / f"{i:02d}_ID{building}_Prediction.mp4"}
    if rgb_baseline is not None:
        paths["rgb"] = rgb_baseline / "Prediction" / f"{i:02d}_ID{building}.mp4"
    gray = {"gt": gt.astype(np.float32)}
    appearance = {"gt": {}}
    known = partial < 220
    background = ~ndimage.maximum_filter(gt < 220, size=(1, 5, 5))
    for role, media in paths.items():
        rgb = quality.decode_prediction(media)
        gray[role] = np.stack(roof.video_luma(media))
        appearance[role] = {"background_fill": float((gray[role][background] < 220).mean())}
        dark = gray[role] < 150
        appearance[role]["core_rgb"] = rgb[dark].mean(axis=0).tolist() if dark.any() else None
        appearance[role]["known_mae"] = float(np.abs(gray[role][known] - partial[known]).mean())
        # Compare low-pass adjacent frame changes to actual GT camera motion.
        prev_pred = prev_gt = prev_fg = None
        error, pixels = 0., 0
        for f in range(81):
            blurred = ndimage.gaussian_filter(gray[role][f], 2.)
            target = ndimage.gaussian_filter(gray["gt"][f], 2.)
            fg = ndimage.maximum_filter(gt[f] < 220, size=9)
            if f:
                support = fg | prev_fg
                excess = np.abs((blurred - prev_pred) - (target - prev_gt))
                error += excess[support].sum(dtype=np.float64)
                pixels += int(support.sum())
            prev_pred, prev_gt, prev_fg = blurred, target, fg
        appearance[role]["motion_residual"] = float(error / pixels) if pixels else None
        del rgb
    dark_gt = gt[gt < 150]
    appearance["gt"]["core_rgb"] = [float(dark_gt.mean())] * 3 if dark_gt.size else None
    roof_scores = roof.scores(building, camera, record, gray["gt"], {role: gray[role] for role in paths})
    vertices, faces, normals = labels.read_obj(roof.MESHES / f"{building}.obj")
    targets = labels.structural_edges(vertices - camera["world_to_local_origin"], faces, normals)
    centers, rotations, intrinsics = data.camera_arrays(camera)
    observed = dict(centers=centers, rotations=rotations, intrinsics=intrinsics)
    ridges = quality.edge_image_support(gray, targets, observed, frames=range(81),
                                        common_support=rgb_baseline is not None)
    results = dict(building=building, appearance=appearance, roof=roof_scores, ridges=ridges)
    if extra_metrics is not None:
        results.update(extra_metrics(row, camera, record, gray, partial, targets, roof))
    return results


def display(value, spec=".2f"):
    return "无可测段" if value is None else format(value, spec)


def rgb_text(values):
    return "无深色像素" if values is None else " / ".join(f"{x:.1f}" for x in values)


def case_report(result):
    title = f"建筑 {result['building']}"
    old, new, gt = (result["appearance"][k] for k in ("old", "new", "gt"))
    ro, rn, rg = (result["roof"][k] for k in ("old", "new", "gt"))
    eo, en = (result["ridges"][k] for k in ("old", "new"))
    text = [
        f"## {title}", "",
        "| 成片指标 | 原三轮 V2 | 本轮 | GT |", "| --- | ---: | ---: | ---: |",
        f"| 可测屋脊段检出数 / GT 可测视角段 | {eo['detected']}/{eo['eligible']} | {en['detected']}/{en['eligible']} | — |",
        f"| 屋脊位置误差 px | {display(eo['position_px'])} | {display(en['position_px'])} | 0 |",
        f"| 屋脊直线拟合垂距 P95 px | {display(eo['straightness_p95_px'])} | {display(en['straightness_p95_px'])} | {display(result['ridges']['gt']['straightness_p95_px'])} |",
        f"| 可辨投影长度 / GT 可测长度 | {display(eo['readable_length_px_ratio'],'.1%')} | {display(en['readable_length_px_ratio'],'.1%')} | {display(result['ridges']['gt']['readable_length_px_ratio'],'.1%')} |",
        f"| 段内断开长度占比 | {display(eo['interior_gap_ratio'],'.1%')} | {display(en['interior_gap_ratio'],'.1%')} | — |",
        f"| 对同一屋脊相邻可测视角的显隐切换 | {eo['visibility_flips']}/{eo['adjacent_eligible']} | {en['visibility_flips']}/{en['adjacent_eligible']} | — |",
        f"| 低通邻帧变化相对 GT 的误差 /255 | {display(old['motion_residual'])} | {display(new['motion_residual'])} | 0 |",
        f"| 000/080 屋面亮度差 /255 | {ro['loop']:.2f} | {rn['loop']:.2f} | {rg['loop']:.2f} |",
        f"| 屋面深色点芯比例 | {ro['deep_ink']:.2%} | {rn['deep_ink']:.2%} | {rg['deep_ink']:.2%} |",
        f"| 深色像素的平均 RGB | {rgb_text(old['core_rgb'])} | {rgb_text(new['core_rgb'])} | {rgb_text(gt['core_rgb'])} |",
        f"| 屋面 16 px 小区点纹覆盖的变异系数 | {ro['cv']:.3f} | {rn['cv']:.3f} | {rg['cv']:.3f} |",
        f"| 墙面 8 px 小区占据 IoU | {ro['wall_iou']:.3f} | {rn['wall_iou']:.3f} | 1 |",
        f"| 远离 GT 点纹的背景误填 | {old['background_fill']:.2%} | {new['background_fill']:.2%} | 0 |",
        f"| 原输入可见点区相对输入的亮度误差 /255 | {old['known_mae']:.2f} | {new['known_mae']:.2f} | 参照为输入 |"
    ]
    return "\n".join(text) + "\n"


def write_report(path, report):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(report, encoding="utf-8")
    os.replace(tmp, path)


def review(run):
    rendered = data.read_json(run / "validation_rendered.json")
    done = completion(run)
    if rendered["ids"] != list(IDS) or any(rendered[k] != done[k] for k in ("adapter_sha256", "lora_sha256")):
        raise ValueError("Rendered videos belong to a different training release")
    progress(run, "scoring", 8, scored=0)
    sections = []
    for i, building in enumerate(IDS, 1):
        stop_if_requested(run)
        progress(run, "scoring", 8, building, 30, scored=i - 1)
        metrics_file = run / ".validation" / f"metrics_{building}.json"
        if not metrics_file.is_file():
            write_json(metrics_file, score(run, i, building))
        report = case_report(data.read_json(metrics_file))
        write_report(case_dir(run, i, building) / "简报.md", report)
        sections.append(report)
        progress(run, "scoring", 8, building, 30, scored=i)
    codec = []
    for building in ("2825", "5954"):
        ref = data.read_json(run / "references" / f"vae_{building}.json")["vae"]
        codec.append(f"| {building} | {ref['cv']:.3f} | {ref['motion']:.3f} | {ref['deep_ink']:.2%} | {ref['loop']:.2f} |")
    first = data.read_json(run / "first_update.json")
    recipe = data.read_json(run / "config.json")
    # Preserve actual measurements and configuration from this run, including
    # scheduler, first counted update, resources, source identities and prompt.
    report = f"""# GT V2 稀疏点纹和屋脊辅助监督 · 原八栋验证

运行 {run.name} 在 {now()} 完成了 32,241 次优化后的八条全轨迹成片及自动量化。训练为原 10,747 栋 × 三轮，512×512、17 帧、batch 1。初始权重为 GT V2 三轮交付的原八视角 Adapter 与 rank/alpha 8 的 LoRA。本轮使用 Wan2.1-VACE-1.3B 原生预测，损失包含 17 帧 FM、低噪声五帧因果前缀的点纹、结构过渡与时序辅助。mesh 结构线只用于计算训练区域和离线量化。输入沿用残缺点、其渲染视频、编辑 mask 和原文本。

八栋均为验证集。每栋使用原相机和像素尺寸 832×480、帧 000–080、16 fps、30 步、CFG 5、shift 5、seed 20261018。Prediction 直接编码 VACE 解码的原始 RGB，三栏为条件 / 预测 / GT。八对 MP4 在发布前完整解码核验。最终 Adapter SHA-256 为 {done['adapter_sha256']}，LoRA 为 {done['lora_sha256']}。固定正提示词如下，负提示词为空。

> {recipe['prompt']}

## 数值的读法

每条原生预测成片与同栋旧三轮成片均在交付编码后读取。屋脊量化只统计原 GT 稀疏点纹在投影线附近可辨的部位。可辨长度按线中心 72% 的 12 个位置估算，段内缺口统计仍有两端证据的中断。同一段连续两帧均可测才计显隐切换。投影附近的纹理梯度是二维代理，视觉直度和断裂仍需结合原片复核。点芯采用灰度 <150，点纹采用 <220，已知区以残缺输入的点纹像素为准。跨帧数值表示相对 GT 实际变化的差异。各栋一个种子，人工逐帧复核待进行。

## 复用的 GT 经 VAE 编解码参照

来自上次屋顶实验 run_20261120_014636 的两条原相机 GT 结果。本次直接引用既有实测。

| 建筑 | 屋面点纹 CV | 共同可见点相邻变化 /255 | 屋面深色比例 | 000/080 屋面差 /255 |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(codec)}

{chr(10).join(sections)}

## 训练配置与首步记录

```json
{json.dumps(dict(config=recipe, first_update=first, completion=done), ensure_ascii=False, indent=2)}
```
"""
    path = run / OUTPUT / "实验报告.md"
    archive = ROOT / "upload" / f"{run.name[4:12]}_GT视频V2_RGB辅助_{run.name[13:]}_实验报告.md"
    write_report(path, report)
    write_report(archive, report)
    write_json(run / "validation_complete.json", dict(time=now(), completed=len(IDS), frames=648,
               prediction_videos=8, comparison_videos=8, report=str(path), upload=str(archive),
               building_reports=8, visual_review="pending"))
    progress(run, "complete", 8, step=30, scored=8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--phase", choices=("render", "report", "all"), default="all")
    args = parser.parse_args()
    run = args.run.resolve()
    try:
        if args.phase in ("render", "all"):
            render(run)
        if args.phase in ("report", "all"):
            review(run)
    except Paused:
        return 75
    return 0


if __name__ == "__main__":
    sys.exit(main())
