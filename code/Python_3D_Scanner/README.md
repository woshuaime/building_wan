# Python_3D_Scanner：原始设计与持续更新

最近补记：2026-10-23（UTC+08:00，北京时间）。本文件持续保留原扫描器设计、采集参数、文件格式和数据准则，以及当前源码缺失的核对。历史正文已恢复在原文件内，后续进展按日期追加；归档仅作备份，无需离开本文件才能阅读旧记录。

阅读顺序：[此前累计正文](#history-records) → [2026-10-23阶段更新](#update-20261023) → [本次历史恢复及生效约定](#history-repair)。旧段落的“当前”“下一步”、命令和PID按其记录时点理解；较晚的更正与实际完成证据在后文，旧命令不因此重新执行。

截至本次修复：Pilot8系列及Complex12的C/U/S各300更新、五方法60条81帧已完成。本次仅修复文档；[实验入口](../../experiments/README.md)和[文档索引](../../docs/DOCUMENTATION_INDEX.md)仍可直接使用。

当前事实：src/和两份GUI工作流源码缺失；下方安装/运行命令保留为原设计记录，不代表现在已恢复GUI。

<a id="history-records"></a>

## 一、此前累计正文：此次整理前长期保留的内容

以下完整保留原有记录顺序、参数、命令、过程和当时结论，只调整标题层级。原记录中的未完成状态与后来完成状态共同构成历史，不能用后者替换前者。

<!-- RESTORED_HISTORY_BEGIN -->
### 3D Scanner - 模型虚拟拍摄工具

> 2026-10-14 工作副本核对：下文为历史扫描器设计和操作说明。当前目录缺少 `src/`、`docs/DATASET_WORKFLOW.md` 和 `docs/workflow.md`，不能按本页命令直接运行 GUI。当前可开发的训练/验证入口见项目根目录 [README](../../README.md)、[当前状态](../../PROJECT_CONTEXT.md) 和 [操作手册](../../TRAINING.md)。恢复采集工具时需取回原始源码并独立验证，不能根据文档重造历史实现后声称已经恢复。

基于 **PySide6 + PyVista + VTK** 的桌面应用：导入 OBJ 三维模型，沿着 4 条圆形轨道，由离屏渲染器自动拍摄 32 / 64 / 96 张多角度照片，用于 3D 重建。

---

#### 安装与运行

```bash
pip install -r requirements.txt
cd src
python main.py
```

打包：`pyinstaller 3D_Scanner.spec`（产物在 `dist/3D_Scanner/`，无控制台窗口）。

依赖：`numpy`、`Pillow`、`pyvista`、`vtk`、`PySide6`、`pyvistaqt`。

---

#### 使用方法

1. 右侧 **Project** 分组点击 **导入 OBJ**，选择模型。
2. **调整模式** 分组切换：默认检视（旋转相机）/ 调整模型 / 调整轨道（点彩色按钮拖拽圆环改半径）。
3. **Capture** 分组选择档位（32 / 64 / 96 张）→ **拍摄所有照片**。

> 切换出"调整模型"模式 / 开始拍摄时，会自动调用 `_solidify_model_transform()` 把视觉旋转固化到 mesh 顶点，保证拍摄数据与视觉一致。

---

#### 关键参数

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

#### 4 条轨道

| 编号 | 名称 | 平面 | UI 颜色 |
| --- | --- | --- | --- |
| 1 | 水平 | XY（z 固定） | 红 `#E63B36` |
| 2 | 竖直 | XZ（y 固定） | 蓝 `#1E87E6` |
| 3 | 左斜 | 绕 X 轴 +45° | 绿 `#42A047` |
| 4 | 右斜 | 绕 X 轴 −45° | 橙 `#FA8C00` |

4 条轨道共圆心，公共交点 `(r, 0, 0)` 就是拍摄第一帧的相机位置。

---

#### 输出

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

#### 项目结构

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

#### 数据集生成准则

项目后续会从“模型拍照工具”升级为“模型补全数据集生成工具”。以下历史工作流文档当前缺失，需要从原采集项目恢复：

- `docs/DATASET_WORKFLOW.md`（缺失）
- `docs/workflow.md`（缺失）

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

#### License

MIT
<!-- RESTORED_HISTORY_END -->

<a id="update-20261023"></a>

## 二、2026-10-23：阶段更新与当前资料补全

下面继续保留357b09f阶段补充的结果、参数和操作说明。该阶段曾把历史移到归档、部分原文页改成导航；这一文档组织做法已由第三节的用户更正取代，实验事实和既有检查结果保留。

<!-- RETAINED_UPDATE_BEGIN -->
### Python_3D_Scanner：现存训练工具与扫描器历史

更新：2026-10-23（UTC+09:00）。当前检出的可维护内容主要是training工具、历史配置和测试。**扫描器GUI源码src/缺失，不能按旧版说明直接运行或打包GUI。**

#### 当前实际保留的内容

| 位置 | 作用与状态 |
| --- | --- |
| [training/](training/README.md) | 数据划分、片段准备、项目训练/验证入口和历史配置 |
| [tests/](tests/) | 训练工具测试以及依赖旧GUI源码的测试 |
| [requirements.txt](requirements.txt) | 历史扫描器依赖清单，不是当前scj训练环境的安装清单 |
| src/ | 当前缺失，不能运行旧main.py |
| docs/DATASET_WORKFLOW.md、docs/workflow.md | 当前缺失，旧链接不再作为可用说明入口 |

项目总体状态见 [根README](../../README.md)、[PROJECT_CONTEXT](../../PROJECT_CONTEXT.md)；当前BuildingFiller条件补全专用实验见 [experiments](../../experiments/README.md)。Complex12专用源码已位于experiments/vace/complex12_rev1/scripts，不在本目录。

#### 历史扫描器设计

旧说明描述PySide6、PyVista、VTK的桌面OBJ导入与离屏渲染工具，支持四条轨道、相机位置导出和多视角图像采集。这些是原始设计记录，不是本次验证过的可运行能力。

历史数据生成强调同一建筑的几何、相机和渲染参数一致，再生成GT/Damaged/Exact配对。当前VACE实验使用已经交接、验证的BuildingFiller数据包，不依赖从缺失GUI重新生成数据，也不通过文档推测重造原始采集实现。

原版详细参数、四轨道设计、文件格式与数据准则完整保留于 [更新前README](../../docs/history/20261023_before_md_refresh/code/Python_3D_Scanner/README.md)。归档中的安装、运行与打包命令仅用于历史追溯。

#### 当前环境与边界

服务器项目：/mnt/windowsE/chuanjun/building_wan。训练解释器：/home/shi/miniconda3/envs/scj/bin/python；PyTorch2.11.0+cu128，单卡A6000。不要为阅读此目录安装GUI依赖或覆盖已有训练环境。

若后续明确要恢复扫描器，应先从原始来源取回src及工作流，再单独验证OBJ导入、相机、渲染、输出和测试。当前任务没有完成该恢复，也没有运行依赖缺失src的GUI测试。

#### 维护入口

- [training说明](training/README.md)：现存脚本的职责、历史数据划分与运行注意事项。
- [TRAINING](../../TRAINING.md)：当前服务器操作、状态读取与CPU检查。
- [实验台账](../../EXPERIMENTS.md)：区分T2V历史和VACE已完成实验。
- [目录规则](../../docs/PROJECT_STRUCTURE.md)：新想法进入experiments，成果保留在独立run。
<!-- RETAINED_UPDATE_END -->

<a id="history-repair"></a>

## 三、2026-10-23：恢复原文件内的持续历史

用户明确要求长期MD保留“从以前到现在”的更新。先前仅保留当前摘要、让读者另找归档的处理不符合要求；本次将上述旧正文直接恢复到原文件，同时保留后续更新，没有回退实验结果或把完成状态改回执行中。

以后在本文件按日期追加阶段进展、失败与修复、参数变更、结论和核查范围。需要纠正旧结论时，说明原判断、后来的证据及新结论；原记录继续保留。可以在顶部维护简短状态和阅读索引，不能用摘要或归档链接替换历史正文。

本次修复依据原文件备份及Git记录恢复，没有补造新的运行经历；原数据、权重、视频、指标、冻结报告和现有非Markdown工作保持原样。此次检查与恢复范围见[文档索引](../../docs/DOCUMENTATION_INDEX.md)中的后续补记。
