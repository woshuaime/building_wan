# 3D Scanner - 模型虚拟拍摄工具

基于 **PySide6 + PyVista + VTK** 的桌面应用：导入 OBJ 三维模型，沿着 4 条圆形轨道，由离屏渲染器自动拍摄 32 / 64 / 96 张多角度照片，用于 3D 重建。

---

## 安装与运行

```bash
pip install -r requirements.txt
cd src
python main.py
```

打包：`pyinstaller 3D_Scanner.spec`（产物在 `dist/3D_Scanner/`，无控制台窗口）。

依赖：`numpy`、`Pillow`、`pyvista`、`vtk`、`PySide6`、`pyvistaqt`。

---

## 使用方法

1. 右侧 **Project** 分组点击 **导入 OBJ**，选择模型。
2. **调整模式** 分组切换：默认检视（旋转相机）/ 调整模型 / 调整轨道（点彩色按钮拖拽圆环改半径）。
3. **Capture** 分组选择档位（32 / 64 / 96 张）→ **拍摄所有照片**。

> 切换出"调整模型"模式 / 开始拍摄时，会自动调用 `_solidify_model_transform()` 把视觉旋转固化到 mesh 顶点，保证拍摄数据与视觉一致。

---

## 关键参数

```python
CAMERA_FOV       = 45.0
CAMERA_NEAR      = 0.01
CAMERA_FAR       = 1000.0
CAPTURE_SIZE_PX  = 1024          # 改 2048 提升画质
MAX_TRIANGLES    = 80000         # 超限自动抽稀
INITIAL_ELEV     = 22.0
INITIAL_AZIM     = 135.0
```

轨道半径 = `模型最大尺寸 × 3.5`（`compute_orbit_radius`）。拍摄相机朝向 `focal_point=(0,0,0)`、`up=(0,0,1)`，使用**平行投影**。

---

## 4 条轨道

| 编号 | 名称 | 平面 | UI 颜色 |
| --- | --- | --- | --- |
| 1 | 水平 | XY（z 固定） | 红 `#E63B36` |
| 2 | 竖直 | XZ（y 固定） | 蓝 `#1E87E6` |
| 3 | 左斜 | 绕 X 轴 +45° | 绿 `#42A047` |
| 4 | 右斜 | 绕 X 轴 −45° | 橙 `#FA8C00` |

4 条轨道共圆心，公共交点 `(r, 0, 0)` 就是拍摄第一帧的相机位置。

---

## 输出

```
<save_root>/
└── model00000/             ← 自动递增，五位数字
    ├── track1_01.png       ← 轨道 N，第 ii 张
    ├── ...
    ├── track4_16.png
    └── camera_poses.json   ← 每帧 target_pos / real_pos
```

`camera_poses.json` 格式：

```json
{
  "frames": [
    { "file_path": "track1_01.png", "target_pos": [x, y, z], "real_pos": [x, y, z] }
  ]
}
```

---

## 项目结构

```
src/
├── main.py                 # 入口
├── core/
│   ├── obj_loader.py       # OBJLoader（解析 v/vn/f）
│   ├── track.py            # CircularTrack + TrackSystem
│   └── mesh_solid.py       # 多边形三角化 + 抽稀
├── ui/
│   └── tk_widget.py        # PyVistaView + MainWindow + 拍摄流程
└── utils/
    ├── screenshot.py       # ScreenshotManager（PNG 写入）
    └── model_folder.py     # 输出目录递增
```

主要类 / 函数：

| 名字 | 位置 | 作用 |
| --- | --- | --- |
| `OBJLoader.load()` | `core/obj_loader.py` | 解析 OBJ |
| `TrackSystem.get_all_camera_positions()` | `core/track.py` | 收集 4 条轨道的全部相机位置 |
| `TrackSystem.get_intersection_point()` | `core/track.py` | 公共交点（拍摄第一帧位置） |
| `PyVistaView.load_obj()` | `ui/tk_widget.py` | 加载 OBJ → 对齐 → 居中 → 重建场景 |
| `PyVistaView.capture_frame()` | `ui/tk_widget.py` | 离屏渲染一帧 |
| `MainWindow._on_capture()` | `ui/tk_widget.py` | 拍摄入口 |
| `MainWindow._capture_one()` | `ui/tk_widget.py` | 递归拍一张 + 调度下一张 |
| `ScreenshotManager.save_screenshot_from_array()` | `utils/screenshot.py` | numpy → PNG |

---

## 数据集生成准则

项目后续会从“模型拍照工具”升级为“模型补全数据集生成工具”。完整流程和设计约束见：

- [docs/DATASET_WORKFLOW.md](docs/DATASET_WORKFLOW.md)
- [docs/workflow.md](docs/workflow.md)

最重要的硬规则：

```text
prepare and freeze once, split frozen mesh, render frozen triplet without transform
```

也就是：普通 OBJ 模式负责摆正、居中、对齐底面和朝向，并导出冻结后的 `aligned_complete.obj`；Blender 只基于这个冻结模型切分出 `complete.obj`、`incomplete.obj`、`removed.obj`；Dataset Mode 只读取三元组，不再 normalize、recenter、rescale、reorient，也不根据任何一个 mesh 重新计算 transform。这样 `complete_rgb`、`incomplete_rgb`、`missing_mask`、`visible_mask` 才能在像素和几何上严格对齐。

当前 Dataset Mode 默认使用 `prealigned_triplet`：选择一个同时包含 `complete.obj`、`incomplete.obj`、`removed.obj` 的文件夹，软件只做读取和批量渲染。如果内部需要 OBJ 到 PyVista 的固定坐标转换，也必须对三者完全相同地应用：

```text
(x, y, z) -> (z, x, y)
```

因此 Dataset Mode 输出的 `camera_poses.json` 使用的是软件拍摄坐标，不是原始 OBJ 语义坐标。四类输出 `complete_rgb`、`incomplete_rgb`、`missing_mask`、`visible_mask` 必须共享同一组相机位姿。

Dataset Mode 的 `camera_poses.json` 会额外记录相机模型：

```json
{
  "camera_model": {
    "focal_point": [0, 0, 0],
    "view_up": [0, 0, 1],
    "fov": 45.0,
    "near": 0.01,
    "far": 1000.0,
    "parallel_projection": true,
    "parallel_scale": 7.5,
    "image_size": [1024, 1024]
  },
  "frames": []
}
```

Dataset Mode 还会输出 `metadata.json`，记录输入三元组路径、坐标规则、相机模型、输出目录和保存失败列表。

## License

MIT
