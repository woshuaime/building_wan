#!/usr/bin/env python3
"""CPU rendering and evaluation, allowed only after all candidate sets are frozen."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

from common import ROOT, CASES, D_RUN, SAMPLES, now, sha, write
from geometry import BG, read_cloud, render_ids

W, H = 832, 480
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def decode(path):
    raw = subprocess.check_output(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-threads", "1",
        "-i", str(path), "-map", "0:v:0", "-vsync", "0", "-threads", "1",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
    return np.frombuffer(raw, np.uint8).reshape(81, H, W, 3)


class Encoder:
    def __init__(self, path, width=W, height=H, lossless=True):
        self.path = path
        self.process = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-n",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", "16",
            "-i", "pipe:0", "-an", "-c:v", "libx264rgb" if lossless else "libx264",
            "-preset", "veryfast", "-crf", "0" if lossless else "18", "-threads", "2",
            *([] if lossless else ["-pix_fmt", "yuv420p"]), "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)

    def add(self, frame):
        self.process.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())

    def close(self):
        self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError(f"Video encoding failed: {self.path}")


def coverage(case, out, generation):
    # The first recovery of hidden point coordinates occurs here, in the
    # evaluation-only process, after hash verification of ALL 3 frozen cases.
    record = json.loads((case / "case.json").read_text())
    identity = json.loads((out / "input_identity.json").read_text())
    world, _ = read_cloud(Path(identity["source"]))
    hidden_ids = np.asarray(record["hidden_original_point_ids"], dtype=int)
    hidden = world[hidden_ids] - np.asarray(record["world_to_local_origin"])
    data = np.load(out / "candidates.npz")
    all_points = data["all_points"]
    kept = data["kept_indices"]
    results, distances = {}, {}
    for name, points in (("all_candidates", all_points), ("retained_candidates", all_points[kept])):
        d = cKDTree(points).query(hidden)[0] if len(points) and len(hidden) else np.full(len(hidden), np.inf)
        distances[name] = d
        results[name] = dict(count=len(points), hidden_measured_points=len(hidden),
            within_m={str(t): int((d <= t).sum()) for t in (.1, .2, .3)},
            median_m=float(np.median(d)) if len(d) and np.isfinite(d).all() else None,
            p90_m=float(np.quantile(d,.9)) if len(d) and np.isfinite(d).all() else None)
    wall_planes = [p for p in generation["surfaces"] if p["kind"] == "wall" and "normal" in p]
    plane_d = np.min([abs((hidden-np.asarray(p["origin"])) @ np.asarray(p["normal"])) for p in wall_planes], axis=0) if wall_planes and len(hidden) else np.full(len(hidden), np.inf)
    results["estimated_wall_plane_error"] = dict(median_m=float(np.median(plane_d)) if len(plane_d) and np.isfinite(plane_d).all() else None,
        p90_m=float(np.quantile(plane_d,.9)) if len(plane_d) and np.isfinite(plane_d).all() else None,
        within_0_1m=int((plane_d<=.1).sum()), within_0_2m=int((plane_d<=.2).sum()), within_0_3m=int((plane_d<=.3).sum()))
    with (out / "withheld_point_distances.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["original_point_id", "nearest_all_candidate_m", "nearest_retained_candidate_m", "nearest_estimated_wall_plane_m"])
        for i, a, b, c in zip(hidden_ids, distances["all_candidates"], distances["retained_candidates"], plane_d):
            writer.writerow([int(i), *[float(d) if np.isfinite(d) else "NA" for d in (a,b,c)]])
    write(out / "candidate_coverage.json", results)
    return results


def camera_list(case):
    cameras = json.loads((case / "reference/camera_poses.json").read_text())
    if len(cameras) != 81:
        raise ValueError("Expected the original 81-frame trajectory")
    for camera in cameras:
        camera["R"] = np.array([camera["right"], -np.asarray(camera["up"]), camera["forward"]])
        camera["position"] = np.asarray(camera["position"])
    return cameras


def render(points, colors, camera):
    ids, _ = render_ids(points, camera)
    return np.vstack([colors, BG])[ids]


def contact_sheet(frames, path):
    thumb_w, thumb_h = 416, 240
    sheet = Image.new("RGB", (thumb_w*9, (thumb_h+20)*9), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(FONT, 14)
    for i, frame in enumerate(frames):
        x, y = i%9*thumb_w, i//9*(thumb_h+20)
        image = Image.fromarray(frame).resize((thumb_w,thumb_h))
        sheet.paste(image,(x,y+20)); draw.text((x+3,y+1), f"{i:03d}",fill="black",font=font)
    sheet.save(path)


def run_case(display, name, run, evaluator):
    case, out = CASES/name, run/"cases"/name
    generation = json.loads((out/"generation.json").read_text())
    counts = coverage(case,out,generation)
    observed = np.load(out/"observed_input.npz")
    merged = np.load(out/"merged_points.npz")
    cameras = camera_list(case)
    (out/"raw_frames").mkdir()
    (out/"observed_frames").mkdir()
    fixed_video = Encoder(out/"fixed_3d.mp4")
    input_video = Encoder(out/"observed_input.mp4")
    fixed_frames, observed_frames = [], []
    measured_count = len(observed["points"])
    visible_change_pixels = 0
    for i, camera in enumerate(cameras):
        a = render(observed["points"],observed["colors"],camera)
        b = render(merged["points"],merged["colors"],camera)
        Image.fromarray(a).save(out/"observed_frames"/f"{i:03d}.png")
        Image.fromarray(b).save(out/"raw_frames"/f"{i:03d}.png")
        fixed_video.add(b); input_video.add(a)
        fixed_frames.append(b); observed_frames.append(a)
        old_ids, _ = render_ids(observed["points"], camera)
        new_ids, _ = render_ids(merged["points"], camera)
        surviving = (old_ids >= 0) & (new_ids < measured_count) & (new_ids >= 0)
        visible_change_pixels += int(np.count_nonzero(np.any(a[surviving] != b[surviving],axis=-1)))
    fixed_video.close(); input_video.close()
    fixed_frames, observed_frames = np.stack(fixed_frames), np.stack(observed_frames)
    # Re-render the final view independently from the same merged NPZ.
    independent_last = render(merged["points"],merged["colors"],cameras[80])
    assert np.array_equal(independent_last, fixed_frames[80])
    closure = dict(independently_rendered=True, lossless_frame_000_080_equal=bool(np.array_equal(fixed_frames[0],fixed_frames[80])),
        full_frame_mae=float(abs(fixed_frames[0].astype(float)-fixed_frames[80]).mean()),
        different_pixels=int(np.count_nonzero(np.any(fixed_frames[0] != fixed_frames[80],axis=-1))),
        rgb_xyz_of_original_points_unchanged=bool(np.array_equal(merged["points"][:measured_count], observed["points"]) and np.array_equal(merged["colors"][:measured_count],observed["colors"])),
        unoccluded_original_point_changed_pixels=visible_change_pixels,
        note="New sprites can physically occlude an old point; old XYZ/RGB are never edited")
    write(out/"closure_and_preservation.json",closure)
    for label, frame in (("000",fixed_frames[0]),("080",independent_last)):
        Image.fromarray(frame).save(out/f"independent_{label}.png")
    if not np.array_equal(decode(out/"fixed_3d.mp4"), fixed_frames):
        raise RuntimeError("New video is not RGB lossless")
    target = decode(case/"reference/observed_target.mp4")
    damaged = decode(case/"reference/damaged.mp4")
    alignment = dict(frames=81, rendered_P_equal_original_damaged=bool(np.array_equal(observed_frames,damaged)),
        max_abs_rgb_difference=int(abs(observed_frames.astype(np.int16)-damaged.astype(np.int16)).max()))
    write(out/"input_render_alignment.json",alignment)
    if not alignment["rendered_P_equal_original_damaged"]:
        raise RuntimeError(f"Original camera/sprite alignment mismatch: {name}")
    d_path = D_RUN/"cases"/name/"raw_vace.mp4"
    d_frames = decode(d_path)
    original_input = decode(case/"inputs/vace_video.mp4")
    comparison = Encoder(out/"input_D_fixed_reference.mp4",4*W,H+32,False)
    font=ImageFont.truetype(FONT,19)
    overview = Image.new("RGB",(4*W,(H+32)*5),"white")
    key_frames=(0,16,32,48,64)
    for i in range(81):
        panel=Image.new("RGB",(4*W,H+32),"white"); draw=ImageDraw.Draw(panel)
        for col,(label,frame) in enumerate(zip(("Observed input (gray edit region)","Current D-3000","Fixed 3D points","Measured reference (naturally incomplete)"),
                (original_input[i],d_frames[i],fixed_frames[i],target[i]))):
            panel.paste(Image.fromarray(frame),(col*W,32))
            draw.text((col*W+9,5),f"{display} {name}  {i:03d}  {label}",font=font,fill="black")
        comparison.add(panel)
        if i in key_frames: overview.paste(panel,(0,key_frames.index(i)*(H+32)))
    comparison.close(); overview.save(out/"four_column_key_frames.png")
    contact_sheet(fixed_frames,out/"fixed_all81_contact_sheet.png")
    # Same supplied score() and masks/tracks; no alternative metric definitions.
    fixed_metrics = evaluator.evaluate(case,out/"fixed_3d.mp4")
    d_metrics = evaluator.evaluate(case,d_path)
    stored_D=json.loads((D_RUN/"cases"/name/"metrics.json").read_text())
    if d_metrics != stored_D:
        raise RuntimeError(f"D metric reproduction mismatch: {name}")
    write(out/"metrics_fixed.json",fixed_metrics); write(out/"metrics_D_recomputed.json",d_metrics)
    result=dict(display=display,case=name,generated=generation["generated_candidates"],retained=generation["retained_candidates"],
        observed=measured_count,merged=len(merged["points"]),coverage=counts,metrics_fixed=fixed_metrics,metrics_D=d_metrics,
        closure=closure,alignment=alignment,comparison_video=str(out/"input_D_fixed_reference.mp4"))
    write(out/"result.json",result)
    print(now(),"CASE_COMPLETE",name,json.dumps(result,ensure_ascii=False),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--run",type=Path,required=True)
    run=parser.parse_args().run.resolve()
    freeze=json.loads((run/"generation_all_frozen.json").read_text())
    for case in freeze["cases"]:
        for name,digest in case["files"].items():
            if sha(run/"cases"/case["case"]/name) != digest:
                raise ValueError("Frozen candidate or observed-input identity changed")
    for path,digest in freeze["source_sha256"].items():
        if sha(ROOT/path) != digest:
            raise ValueError(f"Source changed since generation freeze: {path}")
    write(run/"evaluation_reference_access.json",dict(time=now(),all_three_generation_hashes_verified=True,
        generation_frozen_at=freeze["time"],generation_modified_after_scoring=False))
    source=ROOT/"data/pointcloud_vace/微调任务/evaluate.py"
    spec=importlib.util.spec_from_file_location("uploaded_point_evaluator",source)
    evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator);evaluator.decode=decode
    rows=[run_case(display,name,run,evaluator) for display,name in SAMPLES]
    write(run/"results.json",dict(time=now(),results=rows,evaluator_path=str(source),evaluator_sha256=sha(source)))


if __name__ == "__main__":
    main()
