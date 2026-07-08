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
normalize once, split after normalization, never renormalize split meshes
```

中文解释：

```text
完整模型只标准化一次；
在标准化后的同一个坐标系里切割；
切出来的残缺部分和被移除部分只继承坐标；
切割后绝不重新摆正、绝不重新居中、绝不重新缩放。
```

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
complete.obj 导入后归一化一次
-> 在这个坐标系里切割
-> incomplete_mesh = 删除一部分面之后的剩余几何
-> removed_mesh = 被删除的那部分几何
-> 三者共享同一个坐标系、同一个轨道中心、同一组相机位姿
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
1. 读取完整 OBJ
2. 对完整 OBJ 做一次标准化
3. 记录 normalization_matrix 和 normalization_report
4. 在标准化后的 mesh 上制造残缺
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

其中第 4 步必须在标准化后的坐标系中完成，不能把 `incomplete_mesh` 或 `removed_mesh` 再送回普通导入流程。

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

当前 `camera_poses.json` 只记录了 `file_path`、`target_pos`、`real_pos`，后续数据集版本建议扩展为：

- `file_path`
- `track_index`
- `frame_index`
- `camera_position`
- `focal_point`
- `view_up`
- `fov`
- `resolution`
- `near`
- `far`
- `camera_to_world`
- `world_to_camera`

这些信息可以保证后续训练、评估、MVS 重建和论文复现实验都能追溯。

## metadata.json 应记录的信息

建议每个样本记录：

- 原始 OBJ 路径或模型 ID
- 标准化矩阵 `normalization_matrix`
- 标准化报告 `normalization_report`
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
完整模型导入并标准化一次
-> 直接在内存 mesh 上切割
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
