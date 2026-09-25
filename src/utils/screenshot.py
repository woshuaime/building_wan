import os
from datetime import datetime
import numpy as np
from PIL import Image


class ScreenshotManager:
    """截图管理器"""

    def __init__(self, output_dir=None):
        self.output_dir = output_dir or self._get_default_output_dir()
        self.ensure_output_dir()

    def _get_default_output_dir(self):
        """获取默认输出目录"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return os.path.join(os.getcwd(), f'scans_{timestamp}')

    def ensure_output_dir(self):
        """确保输出目录存在"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def set_output_dir(self, directory):
        """设置输出目录"""
        self.output_dir = directory
        self.ensure_output_dir()

    def save_screenshot(self, image_data, track_index, photo_index):
        """保存截图

        Args:
            image_data: 图像数据
            track_index: 轨道索引 (1-4)
            photo_index: 照片索引 (1-16)
        """
        filename = f'track{track_index}_{photo_index:02d}.png'
        filepath = os.path.join(self.output_dir, filename)

        counter = 1
        while os.path.exists(filepath):
            filename = f'track{track_index}_{photo_index:02d}_{counter}.png'
            filepath = os.path.join(self.output_dir, filename)
            counter += 1

        image_data.save(filepath, 'PNG')
        return filepath

    def save_screenshot_from_array(self, img_array, track_index, photo_index):
        """保存 numpy 数组图像（线程安全，用于异步存图）

        Args:
            img_array: RGB numpy array (H, W, 3), dtype=uint8
            track_index: 轨道索引 (1-4)
            photo_index: 照片索引 (1-16)
        """
        filename = f'track{track_index}_{photo_index:02d}.png'
        filepath = os.path.join(self.output_dir, filename)

        counter = 1
        while os.path.exists(filepath):
            filename = f'track{track_index}_{photo_index:02d}_{counter}.png'
            filepath = os.path.join(self.output_dir, filename)
            counter += 1

        # PyVista screenshot arrays are already top-to-bottom in image order.
        pil_img = Image.fromarray(img_array)
        pil_img.save(filepath, 'PNG')
        return filepath

    def save_array_to_path(
        self,
        img_array,
        filepath,
        flip=False,
        compress_level=None,
    ):
        """Save a numpy RGB array to an exact path."""
        parent = os.path.dirname(filepath)
        if parent and not os.path.exists(parent):
            os.makedirs(parent)
        data = np.flipud(img_array) if flip else img_array
        pil_img = Image.fromarray(data)
        save_options = {}
        if compress_level is not None:
            save_options["compress_level"] = max(
                0,
                min(9, int(compress_level)),
            )
        pil_img.save(filepath, 'PNG', **save_options)
        return filepath

    def get_output_dir(self):
        """获取输出目录"""
        return self.output_dir
