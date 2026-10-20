"""CPU-only rendering of four minimal deliverables from real 81-frame predictions."""

import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image

from vace_p01_results import (
    FPS,
    HEADER_HEIGHT,
    SIZE,
    comparison_panel,
    require,
    save_png_exclusive,
    validate_video,
)


FRAME_COUNT = 81
REPRESENTATIVE_FRAMES = (0, 20, 40, 60, 80)
FINAL_FILENAMES = (
    "01_补全结果.mp4",
    "02_补前补后对比.mp4",
    "03_补前补后总览.png",
    "04_四栏验证.png",
)
COMPARISON_SIZE = (SIZE[0] * 2, SIZE[1] + HEADER_HEIGHT)
OVERVIEW_SIZE = (SIZE[0] * 5, (SIZE[1] + HEADER_HEIGHT) * 2)
VALIDATION_SIZE = (SIZE[0] * 4, SIZE[1] + HEADER_HEIGHT)


def validate_frame(image, role, index):
    require(isinstance(image, Image.Image), f"{role} B{index:03} must be a PIL image")
    require(image.mode == "RGB" and image.size == SIZE,
            f"{role} B{index:03} must be RGB 512x512 without resizing")
    image.load()


def validate_frames(damaged, masks, prediction, gt_B020):
    for role, images in (("Damaged", damaged), ("ModelMask", masks), ("Prediction", prediction)):
        require(len(images) == FRAME_COUNT, f"{role} must contain exactly 81 frames, got {len(images)}")
        for index, image in enumerate(images):
            validate_frame(image, role, index)
            if role == "ModelMask":
                pixels = np.asarray(image)
                require(np.isin(pixels, (0, 255)).all()
                        and np.array_equal(pixels[:, :, 0], pixels[:, :, 1])
                        and np.array_equal(pixels[:, :, 1], pixels[:, :, 2]),
                        f"ModelMask B{index:03} must have identical binary RGB channels")
    validate_frame(gt_B020, "GT", 20)


def write_video_exclusive(path, size, frames):
    require(importlib.util.find_spec("imageio_ffmpeg") is not None,
            "Existing imageio_ffmpeg is required; no installation attempted")
    import imageio_ffmpeg

    with Path(path).open("xb"):
        pass
    writer = imageio_ffmpeg.write_frames(
        str(path), size, fps=FPS, quality=8, codec="libx264", macro_block_size=16,
        ffmpeg_log_level="error", output_params=["-threads", "1"],
    )
    frame_count = 0
    try:
        writer.send(None)
        for image in frames:
            require(image.mode == "RGB" and image.size == size, "Video frame mode or dimensions changed")
            frame_count += 1
            require(frame_count <= FRAME_COUNT, "Video encoder received more than 81 frames")
            writer.send(np.asarray(image).tobytes())
    finally:
        writer.close()
    require(frame_count == FRAME_COUNT, f"Video encoder received {frame_count} frames instead of 81")


def comparison_frames(damaged, prediction):
    for index, (before, after) in enumerate(zip(damaged, prediction)):
        yield comparison_panel(((f"Damaged B{index:03}", np.asarray(before)),
                                (f"VACE Prediction B{index:03}", np.asarray(after))))


def overview_image(damaged, prediction):
    before = comparison_panel(tuple((f"Damaged B{index:03}", np.asarray(damaged[index]))
                                    for index in REPRESENTATIVE_FRAMES))
    after = comparison_panel(tuple((f"VACE Prediction B{index:03}", np.asarray(prediction[index]))
                                   for index in REPRESENTATIVE_FRAMES))
    overview = Image.new("RGB", OVERVIEW_SIZE, "white")
    overview.paste(before, (0, 0))
    overview.paste(after, (0, before.height))
    return overview


def four_column_image(damaged, masks, prediction, gt_B020):
    return comparison_panel((("Damaged B020", np.asarray(damaged[20])),
                             ("ModelMask B020", np.asarray(masks[20])),
                             ("VACE Prediction B020", np.asarray(prediction[20])),
                             ("GT B020", np.asarray(gt_B020))))


def validate_final_results(destination):
    destination = Path(destination)
    require(destination.is_dir() and not destination.is_symlink(), "Expected a private result directory")
    require(sorted(path.name for path in destination.iterdir()) == sorted(FINAL_FILENAMES),
            "Final output directory must contain exactly the four specified files")
    for name in FINAL_FILENAMES:
        path = destination / name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size > 0,
                f"Final output must be a nonempty regular file: {name}")
    videos = {
        FINAL_FILENAMES[0]: validate_video(destination / FINAL_FILENAMES[0], SIZE, expected_frames=FRAME_COUNT),
        FINAL_FILENAMES[1]: validate_video(destination / FINAL_FILENAMES[1], COMPARISON_SIZE,
                                          expected_frames=FRAME_COUNT),
    }
    images = {}
    for name, size in ((FINAL_FILENAMES[2], OVERVIEW_SIZE), (FINAL_FILENAMES[3], VALIDATION_SIZE)):
        with Image.open(destination / name) as image:
            image.load()
            require(image.format == "PNG" and image.mode == "RGB" and image.size == size,
                    f"Final PNG is unreadable or has unexpected dimensions: {name}")
            images[name] = dict(width=image.width, height=image.height, decode_complete=True)
    return dict(frames=FRAME_COUNT, resolution=SIZE, fps=FPS, videos=videos, images=images,
                files=list(FINAL_FILENAMES), GT_usage="post_inference_B020_four_column_only",
                raw_prediction_postprocessed=False)


def render_results(destination, damaged, masks, prediction, gt_B020):
    validate_frames(damaged, masks, prediction, gt_B020)
    destination = Path(destination)
    require(not destination.is_symlink(), "Refuse a symlink result directory")
    if destination.exists():
        require(destination.is_dir() and not any(destination.iterdir()),
                "Refuse to overwrite or remove files in a nonempty result directory")
    else:
        destination.mkdir()
    write_video_exclusive(destination / FINAL_FILENAMES[0], SIZE, iter(prediction))
    write_video_exclusive(destination / FINAL_FILENAMES[1], COMPARISON_SIZE,
                          comparison_frames(damaged, prediction))
    save_png_exclusive(destination / FINAL_FILENAMES[2], overview_image(damaged, prediction))
    save_png_exclusive(destination / FINAL_FILENAMES[3], four_column_image(damaged, masks, prediction, gt_B020))
    return validate_final_results(destination)
