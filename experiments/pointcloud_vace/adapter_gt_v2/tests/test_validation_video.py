"""CPU regressions for the preview's player format and the three panel colors."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
import sys

import av
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import validate_video
from validation_queue import verify_video


class ComparisonPlayback(TestCase):
    def test_rgb_to_bt709_delivery_roundtrip(self):
        width, height = validate_video.WIDTH, validate_video.HEIGHT
        # Include both neutral range endpoints and saturated color patches. A
        # grayscale-only fixture would miss BT.601/709 and RGB channel swaps.
        values = np.tile(np.array([0, 24, 128, 235, 255], np.uint8), (height, 1))
        gray = Image.fromarray(values).resize((width, height), Image.Resampling.NEAREST).convert("RGB")
        rgb = Image.new("RGB", (width, height), (195, 65, 20))
        rgb.paste((15, 40, 205), (width // 2, 0, width, height))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "example.mp4"
            count = validate_video.FRAMES
            validate_video.save_comparison(path, "test", [gray] * count, [rgb] * count, [gray] * count, "test")
            verify_video(path, "test")
            with av.open(str(path)) as container:
                image = next(container.decode(video=0)).to_ndarray(format="rgb24")
            y = validate_video.HEADER + height // 2
            for panel in (0, 2):
                for i, value in enumerate((0, 24, 128, 235, 255)):
                    x = width * panel + (2 * i + 1) * width // 10
                    np.testing.assert_allclose(image[y, x], [value] * 3, atol=3, rtol=0)
            for x, expected in ((width // 4, (195, 65, 20)), (3 * width // 4, (15, 40, 205))):
                np.testing.assert_allclose(image[y, width + x], expected, atol=3, rtol=0)

    def test_color_cast_rejected_even_with_compatible_stream_metadata(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "incorrect.mp4"
            size = (validate_video.WIDTH, validate_video.HEIGHT)
            gray = [Image.new("RGB", size, (235, 235, 235))] * validate_video.FRAMES
            pink = [Image.new("RGB", size, (250, 98, 251))] * validate_video.FRAMES
            validate_video.save_comparison(path, "test", gray, gray, pink, "test")
            with self.assertRaisesRegex(ValueError, "color cast"):
                verify_video(path, "test")

    def test_old_rgb_profile_is_rejected_before_publication(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.mp4"
            with av.open(str(path), "w") as container:
                container.metadata["comment"] = "test"
                stream = container.add_stream("libx264rgb", rate=16)
                stream.width, stream.height, stream.pix_fmt = 64, 32, "rgb24"
                for packet in stream.encode(av.VideoFrame.from_image(Image.new("RGB", (64, 32)))):
                    container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            with self.assertRaisesRegex(ValueError, "yuv420p"):
                verify_video(path, "test")
