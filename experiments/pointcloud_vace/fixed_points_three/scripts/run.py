#!/usr/bin/env python3
"""Freeze all observed-only candidates before starting a separate scoring process."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from common import ROOT, CASES, D_RUN, SAMPLES, PYTHON, now, sha, write, prepare_observed
from candidates import generate, SETTINGS, in_box


def ply(path, xyz, rgb):
    with path.open("w") as stream:
        stream.write("ply\nformat ascii 1.0\nelement vertex %d\n" % len(xyz))
        stream.write("property double x\nproperty double y\nproperty double z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(xyz, rgb):
            stream.write("%.10f %.10f %.10f %d %d %d\n" % (*p, *c))


def generate_all(run):
    if (run / "generation_all_frozen.json").exists():
        raise RuntimeError("Candidates already frozen; do not regenerate after reference access")
    records = []
    for display, name in SAMPLES:
        out = run / "cases" / name
        out.mkdir(parents=True, exist_ok=True)
        if not (out / "observed_input.npz").exists():
            prepare_observed(CASES / name, out)
        with np.load(out / "observed_input.npz") as loaded:
            observed = {k: loaded[k] for k in loaded.files}
        xyz, rgb, keep, audit = generate(observed["points"], observed["colors"],
            observed["box_center"], observed["box_axes"], observed["box_half"])
        assert in_box(xyz, observed["box_center"], observed["box_axes"], observed["box_half"]).all()
        np.savez_compressed(out / "candidates.npz", all_points=xyz, all_colors=rgb, kept_indices=keep)
        merged = np.vstack([observed["points"], xyz[keep]])
        colors = np.vstack([observed["colors"], rgb[keep]])
        assert np.array_equal(merged[:len(observed["points"])], observed["points"])
        assert np.array_equal(colors[:len(observed["colors"])], observed["colors"])
        np.savez_compressed(out / "merged_points.npz", points=merged, colors=colors,
            original_ids=np.r_[observed["original_ids"], np.full(len(keep), -1, int)],
            world_origin=observed["world_origin"])
        ply(out / "candidates_all_world.ply", xyz + observed["world_origin"], rgb)
        ply(out / "merged_fixed_world.ply", merged + observed["world_origin"], colors)
        audit.update(display=display, case=name, time=now(), observed_points=len(observed["points"]),
                     merged_points=len(merged), measured_xyz_rgb_unchanged=True)
        write(out / "generation.json", audit)
        files = {p.name: sha(p) for p in (out / "observed_input.npz", out / "candidates.npz",
                 out / "merged_points.npz", out / "generation.json")}
        records.append(dict(display=display, case=name, generated=len(xyz), kept=len(keep), files=files))
        print(now(), "GENERATED", name, "proposals", len(xyz), "kept", len(keep), flush=True)
    source = {str(p.relative_to(ROOT)): sha(p) for p in Path(__file__).parent.glob("*.py")}
    source.update({"experiments/pointcloud_vace/adapter_d/scripts/geometry.py":
                   sha(ROOT / "experiments/pointcloud_vace/adapter_d/scripts/geometry.py")})
    write(run / "config.json", dict(started_at=now(), python=PYTHON, device="CPU only", gpu_tasks=0,
        algorithm="observed-plane and exterior-scan-strip continuation v1", settings=SETTINGS,
        samples=SAMPLES, cases=str(CASES), comparison_D_run=str(D_RUN), frames=81, width=832,
        height=480, fps=16, renderer="existing geometry.render_ids; radius=2; back-to-front sprites",
        background=[237, 240, 244], main_video_codec="libx264rgb CRF0 RGB lossless",
        comparison_first_column="original inputs/vace_video.mp4 (gray editable area)",
        observed_only_render="observed_input.mp4; verified against reference/damaged.mp4 after freeze",
        evaluation="uploaded evaluate.py unchanged; only decoder thread count limited to 1",
        mesh_used=False, withheld_coordinates_used_for_generation=False,
        references_used_for_generation=False, source_sha256=source))
    write(run / "generation_all_frozen.json", dict(time=now(), cases=records,
        reference_access_allowed_from_now=True, condition="All 3 candidate sets and selection frozen",
        hidden_xyz_accessed_during_generation=False, source_sha256=source))
    print(now(), "ALL_THREE_GENERATIONS_FROZEN", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.is_dir():
        raise ValueError("Create an independent timestamped output directory first")
    started = time.monotonic()
    write(run / "status.json", dict(status="generating_observed_only", time=now()))
    try:
        generate_all(run)
        # A fresh process is the first one permitted to load reference videos
        # or recover withheld coordinates. It verifies every candidate digest.
        subprocess.run([PYTHON, str(Path(__file__).with_name("render_evaluate.py")), "--run", str(run)], check=True)
        write(run / "status.json", dict(status="computed_awaiting_static_review", exit_code=0,
              time=now(), elapsed_seconds=round(time.monotonic()-started, 3)))
    except BaseException as exc:
        write(run / "status.json", dict(status="failed", exit_code=1, time=now(), error=repr(exc),
              elapsed_seconds=round(time.monotonic()-started, 3)))
        raise


if __name__ == "__main__":
    main()
