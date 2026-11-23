"""CPU-only paired review of the eight rendered orbits and the two existing releases."""
import argparse
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "adapter_gt_v2/scripts"))
import data
from common import ROOT, now, write_json
from rgb_recipe import IDS
import validate_rgb as validation
from tracks import STARTS, fixed_pairs, measure, patch, summary


def distribution(building, camera, gray, partial, roof):
    """Compare arrangements at the same dark-pixel count inside editable roof/wall regions."""
    result = {role: dict(roof_clump=[], wall_gap=[]) for role in gray}
    frames = range(0, 80, 2)
    for f, (surface, _) in zip(frames, roof.surface_views(building, camera, frames)):
        missing = ~ndimage.maximum_filter(partial[f] < 239, size=3)
        for kind, key in ((1, "roof_clump"), (2, "wall_gap")):
            area = ndimage.binary_erosion(surface == kind, iterations=2) & missing
            bbox = ndimage.find_objects(area.astype(np.uint8))
            if not bbox:
                continue
            rect = bbox[0]
            mask = area[rect]
            ids = np.flatnonzero(mask)
            count = int((gray["gt"][f][rect][mask] < 150).sum())
            if count < 2:
                continue
            # Same deterministic tiebreaker for every method at a given view.
            shuffled = np.random.default_rng(f).permutation(ids)
            for role, clip in gray.items():
                ordered = np.argsort(clip[f][rect].ravel()[shuffled], kind="stable")
                selected = np.zeros(mask.shape, dtype=bool)
                selected.flat[shuffled[ordered[:count]]] = True
                if kind == 1:
                    labels, _ = ndimage.label(selected, structure=np.ones((3, 3)))
                    sizes = np.bincount(labels.ravel())[1:]
                    value = sizes[sizes >= 8].sum() / count
                else:
                    value = np.quantile(ndimage.distance_transform_edt(~selected)[mask], .95)
                result[role][key].append(float(value))
    return {r: {k: float(np.mean(v)) if v else None for k, v in d.items()} for r, d in result.items()}


def hidden_locations(building, camera, targets, roof):
    if building == "2825":
        from mesh_labels import RIDGE
        segments = targets["vertices"][targets["edges"][targets["kinds"] == RIDGE]]
        f = np.linspace(.15, .85, 9)[None, :, None]
        return ((1-f)*segments[:, :1] + f*segments[:, 1:]).reshape(-1, 3)
    if building == "358":
        vertices, triangles, tags = roof.mesh_triangles(building, camera)
        faces = vertices[triangles[tags == 2]]
        # Interior wall samples, in the same local coordinate frame as observations.
        bary = [[a/8, b/8, 1-(a+b)/8] for a in range(1, 7) for b in range(1, 8-a)]
        return np.einsum("ij,kjl->kil", bary, faces).reshape(-1, 3)
    return np.empty((0, 3))


def ncc(first, second):
    if first is None or second is None:
        return None
    a, b = first - first.mean(), second - second.mean()
    if min(a.std(), b.std()) < 2:
        return None
    return float((a*b).mean() / (a.std() * b.std()))


def hidden_motion(clips, partial, uv, z, probes, camera, start, building, roof):
    """Contrast-normalized local evidence for the rear motion vs nearby front-wall motion.

    No generated point is assigned a hidden GT identity. Eligibility depends only on
    GT, cameras and the original partial. An unreadable prediction stays in the count.
    """
    C, R, K = data.camera_arrays(camera)
    frame = start + 4
    begin, depth = data.project(probes, C[start], R[start], K)
    end, end_depth = data.project(probes, C[frame], R[frame], K)
    tree = cKDTree(uv[0])
    used = set()
    out = {role: dict(total=0, readable=0, favors_rear=0) for role in clips}
    _, surface_depth = next(roof.surface_views(building, camera, [start]))
    for xy, dest, d0, d1 in zip(begin, end, depth, end_depth):
        if min(d0, d1) <= 0:
            continue
        before = patch(partial[start], xy, 7)
        after = patch(partial[frame], dest, 7)
        if before is None or after is None:
            continue
        ix, iy = np.rint(xy).astype(int)
        if building == "358" and not surface_depth[iy, ix] + .5 < d0:
            continue
        gt = clips["gt"]
        gt0, gt4 = patch(gt[start], xy, 7), patch(gt[frame], dest, 7)
        if any((p < 220).sum() > .35*(g < 220).sum() for p, g in ((before, gt0), (after, gt4))):
            continue
        ids = np.asarray(tree.query_ball_point(xy, 56), dtype=int)
        # A foreground reference remains in front at both ends.
        ids = ids[(z[0, ids] + .5 < d0) & (z[-1, ids] + .5 < d1)]
        if not len(ids):
            continue
        near = ids[np.linalg.norm(uv[0, ids]-xy, axis=1).argmin()]
        front_dest = xy + uv[-1, near] - uv[0, near]
        if np.linalg.norm(dest - front_dest) < 4:
            continue
        real, flat = ncc(gt0, gt4), ncc(gt0, patch(gt[frame], front_dest, 7))
        tile = (ix // 12, iy // 12)
        if real is None or flat is None or real < .25 or real-flat < .1 or tile in used:
            continue
        used.add(tile)
        for role, clip in clips.items():
            row = out[role]
            row["total"] += 1
            a = patch(clip[start], xy, 7)
            actual = ncc(a, patch(clip[frame], dest, 7))
            other = ncc(a, patch(clip[frame], front_dest, 7))
            if actual is not None and other is not None:
                row["readable"] += 1
                row["favors_rear"] += int(actual >= .25 and actual-other >= .1)
    return out


def extra_metrics(row, camera, record, gray, partial, targets, roof):
    building = row["id"]
    points = data.selected_partial(row, camera, record)
    center = (points.min(0) + points.max(0)) / 2
    C, R, K = data.camera_arrays(camera)
    probes = hidden_locations(building, camera, targets, roof)
    roles = dict(partial=partial, **{k: v for k, v in gray.items() if k != "gt"})
    tracks = {k: [] for k in roles}
    hidden = {}
    for start in STARTS:
        frames = list(range(start, start+5))
        projections = [data.project(points, C[f], R[f], K) for f in frames]
        uv, z = (np.stack(items) for items in zip(*projections))
        middle = np.einsum("ij,ij->i", center - C[frames], R[frames, 2])
        pairs = fixed_pairs(uv, z, middle)
        for role, clip in roles.items():
            tracks[role].append(measure(clip[start:start+5], uv, pairs))
        if len(probes):
            measured = hidden_motion(gray, partial, uv, z, probes, camera, start, building, roof)
            for role, values in measured.items():
                totals = hidden.setdefault(role, dict.fromkeys(values, 0))
                for key, n in values.items():
                    totals[key] += n
    return dict(tracks={k: summary(v) for k, v in tracks.items()}, hidden=hidden,
                spacing=distribution(building, camera, gray, partial, roof))


def number(value, fmt=".2f"):
    return "无可测样本" if value is None else format(value, fmt)


def case_section(metrics):
    lines = [f"## 建筑 {metrics['building']}", "",
             "| 成片 | 点芯 | 屋面连片 | 墙面空隙 px | 可辨脊长 | 共同可辨位置垂距 P95 px | 段内断开 | 同机位亮度差 | 已知区亮度误差 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    roles = (("old", "旧 V2"), ("rgb", "前次 RGB"), ("new", "本轮"), ("gt", "GT"))
    for role, name in roles:
        roof, spacing, ridge = (metrics[k][role] for k in ("roof", "spacing", "ridges"))
        mae = metrics["appearance"][role].get("known_mae")
        cells = (name, number(roof["deep_ink"], ".2%"), number(spacing["roof_clump"], ".2%"),
                 number(spacing["wall_gap"]), number(ridge["readable_length_px_ratio"], ".1%"),
                 number(ridge["common_straightness_px"]), number(ridge["interior_gap_ratio"], ".1%"),
                 number(roof["loop"]), number(mae) if mae is not None else "—")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", f"直度比较覆盖 {ridge['common_segments']} 个三版成片都能辨认的视角段。", "",
              "| 成片 | 点芯平均 RGB | 墙面占据 IoU | 背景误填 | 相邻变化残差 /255 | 屋脊显隐切换/可比 |",
              "|---|---:|---:|---:|---:|---:|"]
    for role, name in roles[:3]:
        a, r = metrics["appearance"][role], metrics["ridges"][role]
        lines.append(f"| {name} | {validation.rgb_text(a['core_rgb'])} | "
                     f"{metrics['roof'][role]['wall_iou']:.3f} | {a['background_fill']:.2%} | "
                     f"{a['motion_residual']:.2f} | {r['visibility_flips']}/{r['adjacent_eligible']} |")
    lines += ["", "| 成片 | 通过/固定点对 | 首帧近/远可见 | 近/远跟丢 | 近/远位移误差 px | 差分误差 px |",
              "|---|---:|---:|---:|---:|---:|"]
    for role, name in (*roles[:3], ("partial", "残缺输入参照")):
        t = metrics["tracks"][role]
        a, b = t["near"], t["far"]
        lines.append(f"| {name} | {t['passed']}/{t['total']} | {t['seen'][0]}/{t['seen'][1]} | "
                     f"{a['lost']}/{b['lost']} | {number(a['median_px'])}/{number(b['median_px'])} | "
                     f"{number(t['diff']['median_px'])} |")
    if metrics["hidden"]:
        lines += ["", "缺口局部运动与近处运动的配对比较（支持正确相对运动 / 可辨 / GT 有效片段位置）："]
        for role, name in roles:
            v = metrics["hidden"][role]
            lines.append(f"- {name}　{v['favors_rear']} / {v['readable']} / {v['total']}")
    return "\n".join(lines)


def overview(results):
    lines = ["## 八栋汇总", "",
             "成片指标按有效建筑等权平均。跟踪通过数按全部固定点对汇总。", "",
             "| 成片 | 屋面深芯 | 缺区屋面连片 | 缺区墙面空隙 px | 屋脊可辨长度 | 视差通过/固定点对 |",
             "|---|---:|---:|---:|---:|---:|"]
    for key, label in (("old", "旧 V2"), ("rgb", "前次 RGB"), ("new", "本轮"), ("gt", "GT")):
        values = []
        for group, metric, fmt in (("roof","deep_ink",".2%"), ("spacing","roof_clump",".2%"),
                                   ("spacing","wall_gap",".2f"), ("ridges","readable_length_px_ratio",".1%")):
            valid = [r[group][key][metric] for r in results if r[group][key][metric] is not None]
            values.append(number(np.mean(valid) if valid else None, fmt))
        tracking = "—"
        if key != "gt":
            passed = sum(r["tracks"][key]["passed"] for r in results)
            total = sum(r["tracks"][key]["total"] for r in results)
            tracking = f"{passed}/{total}"
        lines.append("| " + " | ".join([label, *values, tracking]) + " |")
    return "\n".join(lines)


def review(run):
    cfg = data.read_json(run / "config.json")
    done = validation.completion(run)
    rendered = data.read_json(run / "validation_rendered.json")
    if rendered["ids"] != list(IDS) or any(rendered[k] != done[k] for k in ("adapter_sha256", "lora_sha256")):
        raise ValueError("Rendered orbits do not match the completed weights")
    results = []
    for i, building in enumerate(IDS, 1):
        validation.stop_if_requested(run)
        validation.progress(run, "scoring", 8, building, 30, scored=i-1)
        path = run / ".validation" / f"parallax_{building}.json"
        if not path.exists():
            result = validation.score(run, i, building, rgb_baseline=Path(cfg["rgb_baseline"]),
                                      extra_metrics=extra_metrics)
            write_json(path, result)
        results.append(data.read_json(path))
    first = data.read_json(run / "first_update.json")
    ordinary = data.read_json(run / "first_fm.json")
    text = f"""# 原像素尺度点纹与前后层视差 · 八栋配对实验

{run.name}，从 {cfg['source_weights']['adapter_final']} 与配套 LoRA 开始。原 10,747 栋 × 三轮，共 {done['optimizer_updates']:,} 次正式更新，RGB 命中 {done['aux_updates']:,} 次。单卡、batch 1、832×480×17、rank/alpha 8/8、新 AdamW、原连续循环窗口偏移 13。17 帧 FM，实际 sigma 在 [0.10,0.60] 时以一次前向解码前 5 帧。颜色覆盖、点对、观测、结构、短时系数分别为 .05/.03/.02/.10/.02，前 1,000 步渐增。原 GT 和观测分别采样。完整 mesh 仅供监督和评估。

首个 RGB 更新显存 {first['peak_allocated_bytes']/2**30:.2f} GiB，首个普通更新 {ordinary['allocated_bytes']/2**30:.2f} GiB，全程分配峰值 {done['peak_vram_bytes']/2**30:.2f} GiB。训练耗时 {done['elapsed_seconds']/3600:.2f} 小时。输出为原相机、seed 20261018、CFG 5、30 步、81 帧的原生 VACE 视频。人工成片复核待进行。

屋面点芯按灰度 <150。屋面连片与墙面空隙仅在缺失区计算，等暗像素数后分别统计 ≥8 像素连片比例和到暗纹距离的第 95 百分位。复用既有原尺寸 GT→VAE 的 81 帧记录，2825 与 5954 的屋面深色比例分别为 7.10% 和 10.50%（20261120 RGB 报告）。

五帧片段起点 014、021、038、060。固定观测点对的四帧差分真值 ≥4 px，首帧锁定后按图像局部归一化相关跟踪，排除模糊匹配与冲突。近、远各自位移误差 ≤3 px 且差分误差 ≤2 px 计通过；所有跟丢保留在分母中。残缺输入是跟踪器的可测性参照。缺口另外测 2825 折脊与 358 后墙，在 GT 能区分深层运动和近处运动的稀疏观测区域比较两种位移的图块相关，是二维区域证据。分母为零的项目无可测样本。

提示词：{cfg['prompt']}

"""
    text += overview(results) + "\n\n" + "\n\n".join(case_section(m) for m in results) + "\n"
    report = run / "交付" / "实验报告.md"
    archive = ROOT / "upload" / f"{run.name[4:]}_GT视频V2_点纹与视差_实验报告.md"
    validation.write_report(report, text)
    validation.write_report(archive, text)
    write_json(run / "validation_complete.json", dict(time=now(), completed=8, frames=648,
               prediction_videos=8, comparison_videos=8, report=str(report), upload=str(archive),
               visual_review="pending"))
    validation.progress(run, "complete", 8, step=30, scored=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    try:
        review(parser.parse_args().run.resolve())
    except validation.Paused:
        raise SystemExit(75)
