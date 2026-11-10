"""CPU-only media and the supplied evaluator; called strictly after raw prediction saves."""
import argparse
import importlib.util
from common import *


def postprocess(run, name):
    import numpy as np
    from PIL import Image, ImageDraw
    from diffsynth.utils.data import save_video
    from vace_p01_results import validate_video
    c = verify_frozen(run)
    case = next(r for r in c["cases"] if r["name"] == name)
    out, source = Path(case["output_dir"]), Path(case["case_dir"])
    if not (out / "raw_result.json").exists():
        raise ValueError("Reference RGB/metrics must not be read before raw predictions are retained")
    identity = json.loads((source / "case.json").read_text())
    for name in ("observed_target.mp4", "damaged.mp4", "visible_tracks.json"):
        if sha(source / "reference" / name) != identity["file_hashes"]["reference/" + name]:
            raise ValueError("Uploaded evaluation reference changed: " + name)
    input_rgb, mask = decode(case["inputs"]["vace_video.mp4"]), decode(case["inputs"]["vace_video_mask.mp4"])
    target = decode(source / "reference/observed_target.mp4")
    prediction = []
    for i in range(81):
        with Image.open(out / "raw_frames" / f"{i:03d}.png") as frame:
            if frame.size != (832, 480):
                raise ValueError("Raw PNG dimensions changed")
            prediction.append(frame.convert("RGB"))
    panels = []
    for i, (rgb, m, p, t) in enumerate(zip(input_rgb, mask, prediction, target, strict=True)):
        panel = Image.new("RGB", (3328, 512), "white")
        draw = ImageDraw.Draw(panel)
        for j, (label, frame) in enumerate(zip(
                ("Original gray-filled input", "White=generate mask", "Raw D-3000 prediction", "Observed reference (evaluation only)"),
                (Image.fromarray(rgb), Image.fromarray(m), p, Image.fromarray(t)))):
            panel.paste(frame, (832 * j, 32))
            draw.text((832 * j + 8, 8), f"{name} / {i:03d} / {label}", fill="black")
        panels.append(panel)
        if i in (0, 20, 40, 60, 80):
            panel.save(out / f"comparison_{i:03d}.png")
    save_video(panels, str(out / "input_mask_D_target.mp4"), fps=16, quality=9, ffmpeg_params=["-threads", "1"])
    video_result = validate_video(out / "input_mask_D_target.mp4", expected_size=(3328, 512), expected_frames=81)
    write(out / "comparison_result.json", video_result)
    overview = Image.new("RGB", (5 * 416, 4 * 264), "white")
    draw = ImageDraw.Draw(overview)
    for column, i in enumerate((0, 20, 40, 60, 80)):
        for row, (label, pixels) in enumerate(zip(("Input", "Mask", "Raw D3000", "Observed reference"),
                (input_rgb[i], mask[i], np.asarray(prediction[i]), target[i]))):
            frame = Image.fromarray(pixels).resize((416, 240), Image.Resampling.LANCZOS)
            overview.paste(frame, (416 * column, 264 * row + 24))
            draw.text((416 * column + 5, 264 * row + 5), f"{label} / {i:03d}", fill="black")
    overview.save(out / "overview.png")
    spec = importlib.util.spec_from_file_location("uploaded_pointcloud_evaluator", c["evaluator"])
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    # Identical RGB24 decoding, explicitly one ffmpeg output thread for this host.
    # All metric definitions, thresholds, point-track supports and formulas stay supplied.
    evaluator.decode = decode
    metrics = evaluator.evaluate(source, out / "raw_vace.mp4")
    write(out / "metrics.json", metrics)
    write(out / "evaluation_provenance.json", dict(time=now(), evaluator=c["evaluator"],
        evaluator_sha256=sha(c["evaluator"]), prediction=str(out / "raw_vace.mp4"),
        metric_input="decoded original raw prediction MP4; no compositing", ffmpeg_output_threads=1,
        reference_read_only_after_raw_save=True, withheld_observed_points=case["withheld_observed_points"],
        zero_new_observations_means_recovery_metrics_null=case["withheld_observed_points"] == 0,
        layout_downsampling="overview display only; raw/model resolution unchanged"))
    event(out, "complete_awaiting_visual_review", case=name, raw_saved=True, metrics_saved=True, comparison_saved=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("case")
    args = parser.parse_args()
    postprocess(args.run.resolve(), args.case)
