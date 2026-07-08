# 3D Scanner - 规范文档

> 设计规范 + 验收标准。冲突时**以 `src/` 代码为准**。

---

## 1. 项目概述

桌面应用，导入 OBJ 三维模型后通过 4 条圆形轨道的虚拟相机自动拍摄 32 / 64 / 96 张照片，输出用于 3D 重建（NeRF / 3DGS / Photogrammetry）的图像序列。

- **GUI**: PySide6（Qt Fusion 风格）
- **3D**: PyVista + VTK（`pyvistaqt.QtInteractor` 嵌入）
- **打包**: PyInstaller（`3D_Scanner.spec`，无控制台窗口）

---

## 2. UI 布局

`QHBoxLayout` 左右分栏：

- **左侧**（弹性宽）：PyVista 3D 视口。
- **右侧**（固定 340 px）：**Project** / **调整模式** / **Capture** 三个分组 + 底部状态栏。

### 交互模式（`InteractionMode`）

| 模式 | 鼠标行为 | 切换时的副作用 |
| --- | --- | --- |
| `VIEWING`（默认检视） | 全局相机旋转 | — |
| `MODEL_ADJUSTING` | 仅旋转 `pickable=True` 的模型 actor，视觉中心强制锁定在轨道中心 | 切出时调 `_solidify_model_transform()` 把视觉旋转固化到顶点 |
| `TRACK_ADJUSTING` | 全局相机旋转；在选中圆环上拖拽改半径 | — |

`_on_capture()` 启动拍摄时也会再调用一次 `_solidify_model_transform()`。

---

## 3. 核心功能

### OBJ 导入

`PyVistaView.load_obj()` → `OBJLoader.load()` → **RANSAC 地面法向对齐到 +Z** → Yaw 对齐到 +X → 居中到原点 → `_obj_to_pv_coords()` 坐标系交换 → `build_pyvista_mesh()` 三角化（超 80000 面自动抽稀）→ 轨道半径 = `max(extent) × 3.5`。

#### 规范化流程（4 步）

1. **居中**：几何中心平移到原点
2. **地面法向对齐**：RANSAC 拟合地面平面，旋转使法向对齐到 +Z（建筑竖直站立）
3. **Yaw 对齐**：绕 Z 轴旋转，使 X-Y 平面主方向对齐到 +X
4. **缩放**：缩放到 TARGET_EXTENT（默认 12.0）以内

#### 地面识别策略

采用双层识别机制，优先使用 `mesh_bottom_boundary`：
- **第一层**（优先）：扫描真实三角面片，筛选 Z 值最低、法向与 +Z 轴对齐的面片，合并为底面
- **第二层**：RANSAC 点云平面拟合（8 次迭代）

#### 导入姿态验收

导入后自动验证，满足以下条件才允许继续：
- 中心误差 ≤ 1e-5
- 顶面轴误差 ≤ 1e-6
- 顶面朝相机出发点对齐度 ≥ 0.999
- 底面平行度 ≥ 0.999
- 规范化来源为 `ground_normalized` 且未使用 fallback

### UI 辅助元素

- **YZ 垂直基准面**：灰色半透明平面（opacity=0.4），用于可视化建筑脚底板对齐
- **视锥体线框**：黑色取景框，显示当前相机的拍摄范围
- **轨道公共交点**：橙色圆点，标识拍摄第一帧位置

### 初始观察视角

导入后默认使用概览视角：
- 俯仰角：24°
- 方位角：128°
- 距离：轨道半径 × 3.0（最小 24.0）

### 4 条轨道（`TrackSystem`）

| `plane` | 平面 |
| --- | --- |
| `horizontal` | XY（z 固定） |
| `vertical` | XZ（y 固定） |
| `diag_left` | 绕 X 轴 +45° |
| `diag_right` | 绕 X 轴 −45° |

共圆心 = 原点；公共交点 `(r, 0, 0)` 即拍摄第一帧位置。

### 虚拟相机 / 拍摄

- 每轨 8 / 16 / 24 张（总 32 / 64 / 96 三档）
- **UI 预览起点**：离观察者最近的点（`_best_start_angle`，基于当前相机方位角）
- **实际拍摄起点**：固定朝向（elev=0°, azim=0°，正对 +X 方向）
- `focal_point=(0,0,0)`，`up=(0,0,1)`，**平行投影**
- FOV 45°，截图 1024×1024
- 渲染：主线程离屏 Plotter（`_cap_plotter`）；存图：`ThreadPoolExecutor`
- 帧调度：`QTimer.singleShot(10, _capture_one)` 非阻塞递归

### 输出

```
<save_root>/model00000/trackN_II.png + camera_poses.json
```

---

## 4. 关键技术决策

- **主线程渲染 + 线程池存图**：VTK 不允许跨线程。
- **`_cap_plotter` 生命周期**：创建 → 复用 → 销毁，全部拍摄出口都走 `_cleanup_capture()`。
- **`user_matrix` 固化**：调整模型视觉旋转必须固化到顶点，否则拍摄数据与视觉脱节。
- **离屏相机参数重应用**：`capture_frame` 在 `render()` 后再 set 一次（PyVista 状态机会回滚部分设置）。
- **MODEL_ADJUSTING 中心锁定**：用户拖拽旋转模型时，视觉中心强制锁定在轨道中心，只允许旋转不允许平移。

---

## 5. 验收标准

### 功能

- [x] OBJ 导入 + 自动对齐（RANSAC 地面法向）+ 三轴居中
- [x] 导入姿态自动验收（不满足则报错）
- [x] 4 条轨道正确显示（4 种平面方向 + UI 颜色）
- [x] YZ 垂直基准面可视化
- [x] 视锥体线框可视化
- [x] 3 种交互模式可切换（VIEWING / MODEL_ADJUSTING / TRACK_ADJUSTING）
- [x] 单条轨道可拖拽改半径
- [x] 32 / 64 / 96 张档位可选
- [x] 照片存为 `<root>/model00000/trackN_II.png`
- [x] 同步生成 `camera_poses.json`（`target_pos` + `real_pos`）
- [x] 拍摄中可点击 **停止** 中断
- [x] 异常有 `crash.log`

### 工程

- [x] PyInstaller 打包成单 EXE 无控制台
- [x] 离屏 Plotter 创建前清理旧实例
- [x] 关闭主窗口时统一释放资源

---

## 6. 关键参数参考

```python
# 规范化
TARGET_EXTENT = 12.0                    # 规范化后最大边长
RANSAC_DISTANCE_THRESHOLD = 0.05       # RANSAC 距离阈值
RANSAC_ITERATIONS = 1000               # RANSAC 迭代次数

# 相机
CAMERA_FOV = 45.0                      # 视场角（度）
CAMERA_NEAR = 0.01                     # 近裁剪面
CAMERA_FAR = 1000.0                    # 远裁剪面
CAPTURE_SIZE_PX = 1024                 # 截图分辨率

# 轨道
ORBIT_MARGIN = 1.8                     # 轨道半径系数
MAX_TRIANGLES = 80000                  # 三角面数上限

# 初始视角
INITIAL_OVERVIEW_ELEV = 24.0           # 导入后俯仰角
INITIAL_OVERVIEW_AZIM = 128.0          # 导入后方位角
INITIAL_OVERVIEW_DISTANCE_FACTOR = 3.0 # 导入后距离系数
```
