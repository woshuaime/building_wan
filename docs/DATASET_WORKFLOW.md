# Dataset Workflow

本文档记录本项目从“模型拍照工具”升级为“模型补全数据集生成工具”的核心思路、流程和硬性约束。后续实现数据集功能时，以本文档作为架构准则。

## 项目目标

当前软件的目标不只是给 OBJ 模型拍多视角图片，而是为“模型补全 + 多视角重建”制造训练数据。

目标训练流程：

```text
完整建筑 OBJ
-> 制造几何一致的残缺模型
-> 渲染残缺多视角图片
-> 渲染完整目标图片和缺失区域 mask
-> 训练中间补全模型，把每张残缺图片补完整
-> 将补全后的多视角图片导入开源多视角重建软件
-> 重建出完整模型
```

## 核心不变量

最重要的原则：

```text
prepare and freeze once, split frozen mesh, render frozen triplet without transform
```

中文解释：

```text
普通模式负责摆正并冻结 aligned_complete.obj；
Blender 只基于 aligned_complete.obj 制造残缺；
Dataset Mode 只读取冻结后的 complete / incomplete / removed；
禁止 normalize、recenter、rescale、reorient 三元组中的任何 mesh。
```

## Dataset Mode 输入模式

数据集模式需要明确区分“准备/冻结”和“读取/渲染”的职责，避免 Dataset Mode 重新改动已经冻结的三元组。

### prealigned_triplet

当前默认实现此模式。

```text
普通模式输入：原始 complete.obj
普通模式输出：aligned_complete.obj
Blender 输入：aligned_complete.obj
Blender 输出：complete.obj / incomplete.obj / removed.obj
三者状态：共享同一个冻结后的标准世界坐标系
Dataset Mode 行为：只读取三元组并渲染，不重新计算 transform
```

在这个模式下，Dataset Mode 不能对 `complete.obj`、`incomplete.obj`、`removed.obj` 中任何一个 mesh 重新计算中心、朝向、尺度或摆正矩阵。允许的唯一坐标处理是固定的 OBJ -> PyVista 坐标约定转换，并且必须对三者完全一致地执行。

普通模式的准备/冻结阶段必须保证：

```text
建筑底面平行灰色虚拟平面；
建筑顶面朝向四轨交叉的相机出发点。
模型中心在四轨中心。
```

Blender 阶段必须保证：

```text
complete.obj / incomplete.obj / removed.obj
共享同一个 Blender 世界坐标系；
不要把 incomplete 或 removed 单独移动到原点；
不要单独应用不同缩放、旋转或导出重定位。
```

当前实现约定：

```text
输入 OBJ 语义坐标 -> 软件拍摄坐标
(x, y, z) -> (z, x, y)
```

这一步是全局坐标约定转换，不是归一化、重居中、重缩放，也不是针对某一个 mesh 的独立 transform。`camera_poses.json` 记录的是转换后的软件拍摄坐标。如果后续训练或重建代码需要回到原始 OBJ 语义坐标，必须显式使用这个固定映射的逆变换。

### shared_triplet

非默认实验模式，仅在明确需要时使用。

```text
三者状态：共享原始坐标系，但尚未进入软件拍摄坐标
软件行为：只根据 complete.obj 计算一次刚体对齐 M，同一个 M 应用于三者
```

当前默认 Dataset Mode 不走这个模式。

### raw_triplet

后续再扩展此模式。

```text
Blender 输入：原始 raw_complete.obj
Blender 输出：raw complete / raw incomplete / raw removed
三者状态：共享原始坐标系
软件行为：未来如需支持，也必须显式选择，不能作为默认 Dataset Mode
```

默认流程下不使用 raw_triplet。

原因：

- `complete_mesh`、`incomplete_mesh`、`removed_mesh` 必须严格共享同一个世界坐标系。
- 被切掉的几何原来在哪里，切完后仍然应该在原来的位置。
- 如果残缺模型切完后重新计算中心、重新缩放或重新摆正，残缺图、完整图和 mask 会发生错位。
- 训练时要求同一个像素位置在所有监督图中具有同一个几何含义。

错误流程：

```text
complete.obj 导入后归一化
incomplete.obj 再导入后重新归一化
removed.obj 再导入后重新归一化
=> 三者坐标不一致，mask 和 RGB 不再严格对应
```

正确流程：

```text
普通模式导入原始 complete.obj
-> 自动/手动摆正并冻结 aligned_complete.obj
-> Blender 基于 aligned_complete.obj 切出 complete/incomplete/removed
-> Dataset Mode 原样读取冻结三元组
-> 三者共享同一个拍摄坐标系、同一个轨道中心、同一组相机位姿
```

## 导入摆放准则

每个完整 OBJ 导入时应建立统一的规范坐标系：

- 模型中心放在四轨中心。
- 建筑底面平行于灰色虚拟平面。
- 建筑顶面朝向四轨交叉的相机出发点。
- 四轨中心是模型和相机系统的公共世界中心。
- 自动导入时就完成摆正，不依赖后续手工修正。

这些准则对应当前软件的用户语义：

```text
MeshLab 打开时建筑顶面朝屏幕使用者
=> 本软件导入后建筑顶面朝四轨交叉的相机出发点
```

## 数据集生成流程

推荐架构流程：

```text
1. 读取共享坐标系下的 complete.obj / incomplete.obj / removed.obj
2. 只根据 complete.obj 计算一次刚体对齐 M
3. 记录 alignment_report，scale 固定为 1.0
4. 同一个 M 同时应用到三者
5. 得到：
   - complete_mesh
   - incomplete_mesh
   - removed_mesh
6. 生成统一相机轨道和 camera_poses
7. 对每个 camera_pose 渲染：
   - complete_rgb
   - incomplete_rgb
   - missing_mask
   - visible_mask
8. 导出图片、OBJ、camera_poses.json、metadata.json
```

其中第 2 步只能使用 `complete.obj` 计算变换，不能把 `incomplete_mesh` 或 `removed_mesh` 再送回普通导入流程单独对齐。

## 推荐输出结构

```text
dataset_root/
└── model_0026_damage_000/
    ├── complete.obj
    ├── incomplete.obj
    ├── removed.obj
    ├── complete_rgb/
    │   ├── track1_01.png
    │   └── ...
    ├── incomplete_rgb/
    │   ├── track1_01.png
    │   └── ...
    ├── missing_mask/
    │   ├── track1_01.png
    │   └── ...
    ├── visible_mask/
    │   ├── track1_01.png
    │   └── ...
    ├── camera_poses.json
    └── metadata.json
```

## Mask 渲染准则

mask 不应依赖 RGB 图片相减。

推荐方式是几何渲染：

- `visible_mask`：用同一相机位姿渲染 `incomplete_mesh` 的可见区域。
- `missing_mask`：用同一相机位姿渲染 `removed_mesh` 在完整场景中的可见缺失区域。

注意：`missing_mask` 不能简单把 `removed_mesh` 单独白色渲染完事。更严谨的做法是让完整场景参与深度遮挡，只把被移除几何的可见像素标白。这样可以避免把当前相机看不见的背面、内部面错误标成缺失区域。

同一帧必须使用完全相同的相机参数：

```text
complete_rgb    使用 camera_pose_i
incomplete_rgb  使用 camera_pose_i
missing_mask    使用 camera_pose_i
visible_mask    使用 camera_pose_i
```

## 与中间补全模型的衔接

本数据集不只是为了保存多视角图片，而是为了支撑后续“缺失识别 + 图像补全”的中间模型训练。数据集中已有的关键字段正好对应后续训练监督：

```text
incomplete_rgb  残缺建筑图像，作为模型主要输入
complete_rgb    完整建筑目标图像，监督补全结果
missing_mask    真实缺失区域，监督缺失区域识别，也可作为 baseline 的 oracle mask
visible_mask    残缺模型可见区域，用于区分可见建筑区域和背景/不可见区域
camera_poses    多视角相机位姿，用于组织视角序列和后续几何一致性约束
```

最终系统推理流程应是：

```text
incomplete_rgb
    ↓
Missing Region Detector
    ↓
pred_missing_mask
    ↓
Inpainting Model
    ↓
completed_rgb
```

也就是：

```text
先识别缺失区域，再根据缺失区域进行图像补全。
```

实验开发顺序可以和最终推理顺序不同。为了降低难度，推荐先使用数据集中已有的真实 `missing_mask` 做补全 baseline：

```text
incomplete_rgb + GT missing_mask
    ↓
Inpainting Model
    ↓
completed_rgb
```

这一阶段用于验证：如果缺失区域已经知道，补全模型是否能把建筑补好。它可称为：

```text
GT-mask inpainting baseline
Oracle-mask inpainting baseline
```

随后再训练缺失区域识别模型：

```text
incomplete_rgb
    ↓
Missing Region Detector
    ↓
pred_missing_mask
```

最后比较两种补全结果：

```text
1. 使用 GT missing_mask 的补全结果
2. 使用 pred_missing_mask 的补全结果
```

这样可以定位误差来源：

```text
GT mask 效果好，但 pred mask 效果差：
    问题主要在缺失区域识别。

GT mask 效果也不好：
    问题主要在补全模型。
```

AnimateDiff 在本项目中不负责识别缺失区域。它更适合用于：

```text
将同一建筑、同一轨道或相邻轨道的多视角图片视作伪视频序列，
增强补全结果在外观、纹理、局部结构上的跨视角一致性。
```

但需要注意：

```text
AnimateDiff 不能严格保证三维几何一致性。
```

如果要进一步增强几何一致性，后续应考虑引入：

```text
camera pose
depth map
normal map
edge map
visible_mask
multi-view projection constraint
```

大模型或 VLM 的作用也不是替代像素级 `missing_mask` 监督，而是生成建筑结构语义提示，例如缺失部位、建筑风格、材料、屋顶或墙面延续方向等，用于辅助补全模型生成更合理的建筑结果。

完整中间模型设计见：

```text
docs/workflow.md
```

## camera_poses.json 应记录的信息

Dataset Mode 当前 `camera_poses.json` 已记录相机模型、保存失败列表和逐帧输出路径：

```text
camera_model:
    focal_point
    view_up
    fov
    near / far
    parallel_projection
    parallel_scale
    image_size

failed_outputs:
    保存失败的图片路径和错误信息

frames:
    frame_id
    track_index
    photo_index
    target_pos
    real_pos
    camera
    outputs:
        complete_rgb
        incomplete_rgb
        missing_mask
        visible_mask
```

这些信息可以保证后续训练、评估、MVS 重建和论文复现实验都能追溯。

## metadata.json 应记录的信息

建议每个样本记录：

- 原始 OBJ 路径或模型 ID
- 共同对齐报告 `alignment_report`
- 对齐矩阵、平移量和 `scale=1.0`
- 残缺生成方法
- 残缺参数
- 随机种子
- 删除面数量
- 残缺面积比例
- 输出图片分辨率
- 相机轨道配置
- 软件版本或 git commit

## 残缺生成建议

残缺应优先在三维几何空间里制造，而不是在二维图片上随机画 mask。

可选方法：

- 连通面片删除：实现简单，适合 MVP。
- 球体、盒体、平面切割：几何解释清楚，适合论文表达。
- 结构感知破损：更接近真实建筑破损，但实现复杂。

论文中建议称为：

```text
基于三维几何约束的参数化退化方法
Geometry-Guided Synthetic Incompletion
Controlled Random Degradation
```

这样可以解释“随机”的合理性：随机不是无约束乱删，而是在受控参数范围内生成多样化、可复现、几何一致的残缺样本。

## 实现边界

当前 UI 中的普通导入流程仍然可以用于查看和手工调整单个模型。

但数据集批量生成时，应使用新的数据集流程：

```text
complete / incomplete / removed 共享原始坐标
-> 只根据 complete 计算一次整体对齐
-> 同一个对齐变换作用到三者
-> 直接渲染 complete/incomplete/removed
```

不要把切割后的 `incomplete.obj`、`removed.obj` 当成新模型重新导入普通 UI 流程。

## 验收标准

一个数据集样本应满足：

- 完整模型、残缺模型、被移除模型三者坐标严格一致。
- 模型中心保持在四轨中心。
- 建筑底面平行于灰色虚拟平面。
- 建筑顶面朝向相机出发点。
- 同一帧的四类输出图片像素对齐。
- `missing_mask` 表示当前相机真正可见的缺失区域。
- `visible_mask` 表示当前相机可见的残缺模型区域。
- 所有随机过程可通过 `seed` 和参数复现。
