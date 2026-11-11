"""Make one two-seed comparison and five keyframes; leave visual claims for review."""
import argparse
import numpy as np
from PIL import Image, ImageDraw
from runtime import *


def postprocess(run):
    from diffsynth.utils.data import save_video
    from vace_p01_results import validate_video
    c = read(run / "config.json")
    with np.load(run / "cache/clip.npz") as archive:
        inputs = archive["video"].copy()
    predictions, metrics = {}, []
    for task in c["tasks"]:
        group, seed = task["group"], task["seed"]
        pixels = decode(output_path(run, group, seed))
        if len(pixels) != 17:
            raise ValueError("A formal video does not contain 17 frames")
        predictions[group, seed] = pixels
        metrics.append(read(result_path(run, group, seed)))
    write(run / "metrics.json", dict(time=now(), tasks=metrics, complete_videos=6,
                                     endpoint_metric="not applicable: different camera poses", visual_review="pending"))
    panels = []
    for i, source_frame in enumerate(c["frame_indices"]):
        panel = Image.new("RGB", (3328, 1024), "white")
        draw = ImageDraw.Draw(panel)
        for row, seed in enumerate((20261018, 20261019)):
            frames = [inputs[i]] + [predictions[group, seed][i] for group in ("A", "B", "C")]
            labels = ("Gray-filled input", "A: D3000, no reference", "B: observed reference", "C: completed draft reference")
            for column, (pixels, label) in enumerate(zip(frames, labels)):
                x, y = 832 * column, 512 * row
                panel.paste(Image.fromarray(pixels), (x, y + 32))
                draw.text((x + 8, y + 8), f"{label} | seed {seed} | source {source_frame:03d}", fill="black")
        panels.append(panel)
        if i in (0, 1, 2, 8, 16):
            panel.save(run / "comparisons/keyframes" / f"position_{i:02d}_source_{source_frame:03d}.png")
    comparison = run / "comparisons/comparison.mp4"
    save_video(panels, str(comparison), fps=16, quality=9, ffmpeg_params=["-threads", "1", "-movflags", "+faststart"])
    write(run / "comparisons/result.json", validate_video(comparison, expected_size=(3328, 1024), expected_frames=17))
    event(run, "comparison_saved", raw_videos=6, comparison_videos=1, keyframes=5,
          visual_review="pending", report="deferred until user follow-up")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    postprocess(parser.parse_args().run.resolve())
