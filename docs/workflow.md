# Workflow

本文档记录本项目从“建筑 OBJ 数据集生成”到“中间补全模型”再到“多视角重建”的整体流程。它和 `DATASET_WORKFLOW.md` 的分工是：

- `DATASET_WORKFLOW.md`：重点说明数据怎么造、mask 怎么渲染、坐标系如何保持一致。
- `workflow.md`：重点说明造好的数据如何训练中间补全模型，以及最终系统如何推理。

## 总体目标

本项目的最终目标不是单纯给 OBJ 模型拍照，而是构建一个面向建筑模型补全与多视角重建的数据和方法流程：

```text
完整建筑 OBJ
-> 参数化三维退化，生成 incomplete / removed 几何
-> 渲染 incomplete_rgb / complete_rgb / missing_mask / visible_mask / camera_poses
-> 训练中间补全模型
-> 输出 completed_rgb
-> 将补全后的多视角图像输入 MVS / SfM 软件
-> 重建完整建筑模型
```

核心数据准则仍然是：

```text
prepare and freeze once, split frozen mesh, render frozen triplet without transform
```

也就是普通模式先把原始完整模型摆正并冻结为 `aligned_complete.obj`；Blender 基于冻结模型切出 `complete.obj`、`incomplete.obj`、`removed.obj`；Dataset Mode 只读取冻结后的三元组并批量渲染，不能重新 normalize、recenter、rescale 或 reorient。

## 中间补全模型设计

数据集生成完成后，下一阶段目标是训练一个中间补全模型，用于从残缺建筑多视角图像中恢复完整建筑外观。

中间模型的最终目标不是简单地对单张图片做修复，而是：

```text
输入：残缺建筑多视角图像 incomplete_rgb
输出：
1. 预测缺失区域 pred_missing_mask
2. 补全后的多视角图像 completed_rgb
```

训练时使用数据集中已有的监督信号：

```text
incomplete_rgb  残缺建筑图像
complete_rgb    完整建筑目标图像
missing_mask    真实缺失区域 mask
visible_mask    残缺模型可见区域 mask
camera_poses    多视角相机位姿
```

其中 `missing_mask` 用于监督缺失区域识别，`complete_rgb` 用于监督图像补全。

## 最终系统流程

最终推理时，中间模型应按照以下顺序工作：

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

因此最终系统逻辑一定是：

```text
先识别建筑缺失区域
再根据缺失区域进行图像补全
```

不能把顺序理解反。缺失区域识别和图像补全可以是两个模型，也可以后续整合成一个端到端框架，但逻辑上仍然是“识别 -> 补全”。

## 实验开发顺序

虽然最终系统是“先识别，再补全”，但为了降低实验难度，开发时可以分阶段进行。

推荐实验顺序：

```text
阶段一：GT-mask inpainting baseline
阶段二：missing region detection
阶段三：predicted-mask full pipeline
阶段四：VLM/LLM semantic prompt guidance
阶段五：AnimateDiff multi-view consistency
```

### 阶段一：GT-mask 补全 baseline

第一阶段先不训练缺失区域识别模型，而是直接使用数据集中已有的真实 `missing_mask`。

输入：

```text
incomplete_rgb
GT missing_mask
```

输出：

```text
completed_rgb
```

监督目标：

```text
complete_rgb
```

流程：

```text
incomplete_rgb + GT missing_mask
    ↓
Inpainting Model
    ↓
completed_rgb
```

这一阶段用于回答一个基础问题：

```text
如果缺失区域已经知道，补全模型能不能把建筑补好？
```

如果使用真实 mask 时补全效果都不好，说明主要问题在补全模型；此时不应该先纠结缺失区域识别。

该阶段可称为：

```text
GT-mask inpainting baseline
Oracle-mask inpainting baseline
```

### 阶段二：缺失区域识别模型

第二阶段训练缺失区域识别模型，用于从残缺图像中预测缺失区域。

输入：

```text
incomplete_rgb
```

输出：

```text
pred_missing_mask
```

监督目标：

```text
GT missing_mask
```

流程：

```text
incomplete_rgb
    ↓
Missing Region Detector
    ↓
pred_missing_mask
```

该任务本质上是图像分割任务。

可选模型：

```text
U-Net
DeepLab
SegFormer
Mask2Former
SAM / SAM2 辅助分割
```

当前建议先从简单稳定的 U-Net 或 SegFormer 开始，不建议一开始就把 SAM 或大模型作为唯一的缺失区域识别方法。

评价指标：

```text
IoU
Dice
Precision
Recall
```

### 阶段三：预测 mask 的完整补全流程

第三阶段将缺失识别模型和补全模型连接起来，形成真实推理流程。

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

这一阶段需要分别比较：

```text
1. 使用 GT missing_mask 的补全效果
2. 使用 pred_missing_mask 的补全效果
```

判断方式：

```text
如果 GT mask 效果好，但 pred mask 效果差：
    问题主要在缺失区域识别模型。

如果 GT mask 效果也不好：
    问题主要在补全模型。
```

### 阶段四：引入大模型 / VLM 语义提示

大模型或视觉语言模型不直接替代像素级 mask 监督。

它的主要作用是：

```text
理解建筑缺失区域的语义
生成更明确的补全提示词
约束补全结果符合建筑结构规律
```

输入：

```text
incomplete_rgb
missing_mask 或 pred_missing_mask
```

输出：

```text
architecture-aware prompt
```

示例输出：

```json
{
  "missing_part": "upper-right roof corner and wall surface",
  "building_style": "gray concrete building",
  "structure_hint": "continue the roof plane and vertical wall boundary",
  "material_hint": "use the same wall texture, lighting, and perspective",
  "inpaint_prompt": "restore the missing upper-right roof corner and wall surface, preserving the gray material, straight facade edges, roof plane, perspective, and lighting"
}
```

补全模型使用：

```text
incomplete_rgb
missing_mask / pred_missing_mask
architecture-aware prompt
```

输出：

```text
completed_rgb
```

大模型在这里承担语义辅助作用，不是主要的像素级监督来源。

### 阶段五：AnimateDiff 多视角一致性

AnimateDiff 不适合作为缺失区域识别模型。

它在本项目中的合理作用是：

```text
增强多视角补全结果的一致性
```

本项目有同一建筑的多视角图像，例如：

```text
track1_01.png
track1_02.png
track1_03.png
track1_04.png
```

这些图像可以按相机轨迹组织成伪视频序列：

```text
multi-view images as pseudo-video frames
```

因此，可以将同一建筑的有序多视角图像输入 AnimateDiff-based inpainting 模型，使补全结果在不同视角之间更加一致。

目标是避免：

```text
第一张图补成红色屋顶
第二张图补成灰色屋顶
第三张图窗户数量变化
第四张图墙面纹理不一致
```

推荐论文表述：

```text
将按相机轨迹排序的多视角渲染图像视为伪视频序列，
引入 AnimateDiff 的时序一致性先验，
用于增强建筑图像补全过程中的跨视角一致性。
```

需要注意：

```text
多视角序列不等同于真实视频。
AnimateDiff 只能缓解外观、纹理、风格和局部结构的不一致。
它不能严格保证三维几何一致性。
```

如果后续需要进一步增强几何一致性，可以考虑引入：

```text
camera pose
depth map
normal map
edge map
visible_mask
multi-view projection constraint
```

## 最小可行实现路线

### MVP-1：数据读取器

先实现数据读取器，确保每一帧可以正确读取：

```text
incomplete_rgb/track*_*.png
complete_rgb/track*_*.png
missing_mask/track*_*.png
visible_mask/track*_*.png
camera_poses.json
metadata.json
```

一个训练 batch 建议包含：

```python
{
    "incomplete": Tensor[B, V, 3, H, W],
    "complete": Tensor[B, V, 3, H, W],
    "missing_mask": Tensor[B, V, 1, H, W],
    "visible_mask": Tensor[B, V, 1, H, W],
    "camera_pose": Tensor[B, V, ...],
    "prompt": List[str],
}
```

其中：

```text
B = batch size
V = 同一建筑的视角数量
H, W = 图像高度和宽度
```

### MVP-2：GT-mask 单视角补全

先实现：

```text
incomplete_rgb + GT missing_mask -> completed_rgb
```

目标是验证补全模型本身是否有效。

### MVP-3：缺失区域识别

再实现：

```text
incomplete_rgb -> pred_missing_mask
```

目标是实现自动识别建筑缺失区域。

### MVP-4：完整 pipeline

连接两个模型：

```text
incomplete_rgb
    ↓
pred_missing_mask
    ↓
completed_rgb
```

### MVP-5：大模型提示

加入 VLM / LLM，为补全模型生成建筑结构相关提示词。

### MVP-6：AnimateDiff 多视角一致性

最后再把多视角图像组织成伪视频序列，引入 AnimateDiff 增强不同视角之间的补全一致性。

## 推荐方法名称

可选名称：

```text
LLM-Guided Multi-view Architectural Inpainting
AnimateDiff-assisted Multi-view Building Completion
Geometry-Guided and LLM-Assisted Multi-view Building Image Completion
```

更偏论文任务定义的名称：

```text
Reconstruction-Oriented Multi-View Image Completion
```

## 推荐论文方法表述

本文提出一种面向建筑模型重建的多视角图像补全框架。首先，基于三维几何退化生成残缺建筑模型，并通过统一相机轨迹渲染残缺图像、完整目标图像以及缺失区域 mask。随后，训练缺失区域检测网络从残缺图像中预测可见缺失区域。为了增强补全结果的结构合理性，引入视觉语言模型分析缺失区域的建筑语义，并生成结构感知的补全提示。最后，将按相机轨迹排序的多视角图像视为伪视频序列，利用 AnimateDiff 的时序一致性先验增强跨视角补全结果的一致性，从而得到适用于多视角重建的完整图像序列。

## 关键结论

最终系统流程：

```text
先识别缺失区域，再补全。
```

实验开发顺序：

```text
可以先用真实 missing_mask 做补全 baseline，再训练缺失识别模型。
```

AnimateDiff 的作用：

```text
不是识别缺失区域，而是增强多视角补全结果的一致性。
```

大模型的作用：

```text
不是替代 mask 监督，而是生成建筑结构语义提示，辅助补全模型生成更合理的建筑结果。
```
