# 工作日志与技术参考

本文档用于沉淀本项目讨论中的关键结论、技术路线、架构准则和每日推进记录。它不是最终论文文本，而是项目过程中的“技术记忆”。

## 更新规则

- 每天自动整理一次当天对话和工作进展。
- 优先记录已经定下来的技术决策，不记录零散闲聊。
- 每条日志尽量包含：背景、结论、原因、后续动作。
- 如果某个想法还没验证，标记为“待验证”，不要写成既定事实。
- 如果出现重大架构变更，同步更新本文档的“长期技术参考”部分。

## 长期技术参考

### 项目总目标

本项目从一个 OBJ 模型多视角拍照工具，逐步升级为一个用于“残缺建筑模型补全 + 多视角重建”的数据集和实验平台。

整体工作流：

```text
完整建筑 OBJ
-> 制造残缺 OBJ 和 removed OBJ
-> 渲染残缺多视角图片、完整目标图片、mask、camera pose
-> 训练中间图像补全模型
-> 将补全后的多视角图片输入 MVS / SfM 软件
-> 重建完整建筑模型
```

### 核心数据准则

最重要的不变量：

```text
prepare and freeze once, split frozen mesh, render frozen triplet without transform
```

含义：

- 普通 OBJ 模式负责 Prepare / Alignment：导入原始完整模型，自动或手动摆正、居中、对齐底面和朝向，并导出 `aligned_complete.obj`。
- Blender 基于 `aligned_complete.obj` 制造残缺，并导出 `complete.obj`、`incomplete.obj`、`removed.obj`。
- `complete_mesh`、`incomplete_mesh`、`removed_mesh` 必须共享同一个冻结后的世界坐标系。
- Dataset Mode 只读取冻结三元组并批量渲染，禁止重新 normalize、recenter、rescale、reorient 三元组中的任何 mesh。
- 如果软件内部确实需要 OBJ -> PyVista 固定坐标转换，必须对 `complete.obj`、`incomplete.obj`、`removed.obj` 完全相同地应用。
- 切掉的几何原本在哪里，冻结三元组中仍应保持相对完整模型的位置。

原因：

- 训练数据需要像素级和几何级严格对齐。
- `complete_rgb`、`incomplete_rgb`、`missing_mask`、`visible_mask` 必须来自同一个 camera pose。
- 如果 Dataset Mode 重新计算 transform，mask 会和图像错位，后续 MVS 重建也会受影响。

模式区分：

- `prealigned_triplet`：默认 Dataset Mode。三者已经来自冻结后的 `aligned_complete.obj`，Dataset Mode 不做旋转、平移、缩放或重定向。
- `shared_triplet`：非默认实验模式。只有明确需要时才允许使用，不能替代冻结三元组默认流程。

### 导入摆放准则

完整 OBJ 导入时应满足：

- 模型中心位于四轨中心。
- 建筑底面平行于灰色虚拟平面。
- 建筑顶面朝向四轨交叉的相机出发点。
- 自动导入时就完成摆正。

用户语义：

```text
MeshLab 默认打开时建筑顶面朝屏幕使用者
=> 本软件导入后建筑顶面朝四轨交叉的相机出发点
```

### 数据集建议输出

推荐每个样本输出：

```text
model_xxxx_damage_xxx/
├── complete.obj
├── incomplete.obj
├── removed.obj
├── complete_rgb/
├── incomplete_rgb/
├── missing_mask/
├── visible_mask/
├── camera_poses.json
└── metadata.json
```

### Mask 生成原则

不要用 RGB 图片简单相减生成 mask。

推荐做法：

- `visible_mask`：渲染当前相机下 `incomplete_mesh` 的可见区域。
- `missing_mask`：渲染当前相机下 removed 几何真正可见的缺失区域。
- 更严谨时，完整场景应参与深度遮挡，只把 removed 几何的可见像素标白。

### 拍摄保存与打包准则

拍摄流程必须保证“进度、图片文件、camera pose”三者一致：

- 异步保存图片时，不能在保存队列完成前清理 capture manager。
- 正常完成、用户停止、异常退出三个分支都应等待已提交的保存任务落盘，再写入 `camera_poses.json`。
- `_save_image_async` 这类后台任务应接收固定的 capture manager 引用，避免读取被主线程清空后的可变状态。

Windows EXE 打包准则：

- 默认使用项目实际运行环境 `D:\scj\envs\ad\python.exe` 构建。
- 使用 PyInstaller onedir 输出，交付时必须保留 `dist\3D_Scanner` 整个目录，不能只拷贝单个 exe。
- PySide6 / Shiboken DLL 搜索路径需要通过 runtime hook 提前加入。
- 必须过滤旧 ICU DLL（例如 `icuuc.dll`、`icudt58.dll`），避免外部 Anaconda 目录中的旧 ICU 覆盖 Windows/System32 ICU，导致 `PySide6.QtWidgets` 启动失败。

### 中间补全模型定位

中间模型不是直接生成 3D，而是补全多视角图片。

输入：

```text
incomplete_rgb
missing_mask
visible_mask
camera_poses
可选 depth / normal / edge 等几何条件
```

输出：

```text
completed_rgb
```

训练目标：

```text
让补全后的多视角图片不仅单张好看，还能保持多视角一致，并服务于后续 MVS 重建。
```

### 中间模型最终流程与实验顺序

最终系统流程必须是：

```text
incomplete_rgb
-> Missing Region Detector
-> pred_missing_mask
-> Inpainting Model
-> completed_rgb
```

也就是先识别建筑缺失区域，再根据缺失区域补全图像。不要把 AnimateDiff 或 inpainting 模型直接理解成缺失区域识别器。

实验开发可以先走更稳的分阶段路线：

```text
1. GT-mask / Oracle-mask inpainting baseline
2. missing region detection
3. predicted-mask full pipeline
4. VLM / LLM semantic prompt guidance
5. AnimateDiff multi-view consistency
```

阶段一可以直接使用数据集已有的 `missing_mask` 作为 oracle mask，先验证：

```text
incomplete_rgb + GT missing_mask -> completed_rgb
```

这样可以把问题拆开判断：

- 如果 GT mask 下补全效果不好，主要问题在补全模型。
- 如果 GT mask 效果好但 pred mask 效果差，主要问题在缺失区域识别模型。

### AnimateDiff 的角色

AnimateDiff 不应被当成完整方案，也不负责直接识别缺失区域，而是中间模型中的一个一致性组件。

它适合解决：

```text
将按相机轨迹排序的多视角渲染图像视为伪视频序列，
增强补全结果在外观、纹理、局部结构上的跨视角一致性
```

它不能单独保证：

```text
四条轨道之间的三维几何一致性
```

因此如果要提高几何一致性，仍需要 camera pose、depth、normal、edge、visible_mask 或 multi-view projection constraint 等几何条件。

因此更完整的方向应是：

```text
Inpainting backbone
+ AnimateDiff motion module
+ mask condition
+ geometry condition
+ camera pose condition
+ multi-view consistency design
+ reconstruction-oriented evaluation
```

### VLM / LLM 的角色

大模型或视觉语言模型不替代像素级 `missing_mask` 监督。

它更合理的作用是：

```text
理解建筑缺失区域的语义
生成结构感知补全提示
约束补全结果符合建筑结构规律
```

例如生成：

```json
{
  "missing_part": "upper-right roof corner and wall surface",
  "structure_hint": "continue the roof plane and vertical wall boundary",
  "material_hint": "use the same wall texture, lighting, and perspective"
}
```

在训练和推理中，VLM / LLM 提供的是语义辅助，不是主要监督来源。

### 冲 CCF-A 的研究定位

如果只是把 LaMa / Stable Diffusion Inpainting / AnimateDiff / ControlNet / MVS 拼起来，贡献偏工程。

更有研究价值的定位：

```text
Reconstruction-Oriented Multi-View Image Completion
面向三维重建的多视角一致图像补全
```

核心问题：

```text
如何从残缺建筑的多视角图像中，生成既视觉合理、又多视角几何一致、还能被 MVS 重建成完整模型的补全图像？
```

潜在贡献：

- 定义残缺建筑多视角补全服务于三维重建的新任务。
- 构建严格配对的数据集：complete / incomplete / removed / mask / camera pose。
- 设计 camera-aware、geometry-guided、reconstruction-oriented 的多视角补全框架。
- 使用图像指标和最终三维重建指标共同评估。

## 当前阶段重点

当前不急着继续修软件拍照交互，而是优先想清楚中间补全模型和整体实验路线。

短期建议：

1. 先用脚本生成数据集，不一定依赖完整 GUI。
2. 跑通单帧 inpainting baseline。
3. 再研究 AnimateDiff 如何处理一条轨道的连续视角补全。
4. 后续加入 camera pose、depth、normal、edge、多视角一致 loss。
5. 最终用 MVS 重建结果作为重要评估。

## 每日工作日志

### 2026-07-04

背景：

- 明确项目目标从“OBJ 拍照工具”转向“残缺建筑补全数据集与实验平台”。
- 用户希望记录对话重点，作为后续技术参考和论文思路沉淀。

已确定结论：

- 完整模型只标准化一次，切割后得到的 `incomplete_mesh` 和 `removed_mesh` 必须继承同一个坐标系。
- 切掉的几何原本在哪里，切完后仍然应和完整模型对齐。
- 数据集可以优先用脚本生成，当前更重要的是研究中间补全模型路线。
- AnimateDiff 可以作为连续视角补全的组件，但不能构成完整科研贡献。
- 冲 CCF-A 需要把问题提升为“面向三维重建的多视角一致图像补全”，而不是简单工程组装。

技术判断：

- 单帧 inpainting 只能作为 baseline。
- AnimateDiff 适合增强轨道内连续帧一致性。
- 还需要 mask、camera pose、depth/normal/edge 几何条件、多视角一致机制和 MVS 导向评估。

后续动作：

- 继续细化中间模型架构。
- 设计数据集脚本输出格式。
- 设计 baseline 和 ablation 实验。
- 后续自动更新本文档，持续追加工作日志和技术决策。

### 2026-07-05

背景：

- 每日工作日志自动更新任务触发。

已确定结论：

- 无新增技术决策。

技术判断：

- 当前仍沿用 2026-07-04 已确定的路线：先稳定数据准则和中间补全模型设计，再推进脚本化数据集生成与实验验证。

后续动作：

- 保持每日自动整理，后续只追加新的技术决策、架构变化、实验路线、风险和待办。

### 2026-07-06

背景：

- 继续修复 GUI 拍摄与交付问题，重点处理 64 张拍摄提前结束、EXE 打包后 `QtWidgets` 启动失败两个问题。

已确定结论：

- 64 张拍摄实际只保存到第 2/3 条轨道的根因不是轨道逻辑，而是异步保存线程尚未完成时主线程提前 `_cleanup_capture()`，导致后台 `_save_image_async()` 读到已清空的 `_capture_mgr` 并跳过后续图片。
- 拍摄流程已改为：正常完成、停止、异常分支都先等待保存线程池完成，再写 `camera_poses.json`，最后清理 capture 状态。
- 后台保存函数改为接收固定的 capture manager 引用，避免依赖主线程中的可变 `self._capture_mgr`。
- EXE 交付采用 PyInstaller onedir 模式，输出目录为 `dist\3D_Scanner\3D_Scanner.exe`。
- 打包失败弹窗 `DLL load failed while importing QtWidgets` 的根因是 PyInstaller 收进了外部 Anaconda 目录中的旧 ICU DLL（`icuuc.dll`、`icudt58.dll`），覆盖了系统 ICU，导致 PySide6 的 QtWidgets 加载失败。
- 已在 spec 中加入 PySide6/Shiboken runtime hook，并对 PyInstaller 的手动 binaries 与 `Analysis.binaries` 做二次过滤，排除旧 ICU DLL。
- `build.bat` 固定优先使用 `D:\scj\envs\ad\python.exe` 构建，减少环境漂移。

技术判断：

- 拍摄输出必须以“图片文件实际落盘完成”为完成标准，不能只以 GUI 进度条达到 100% 为准。
- Windows 打包时，当前机器的 PATH 会影响 PyInstaller 自动收集 DLL；后续遇到 Qt/Open3D/VTK DLL 问题时，应优先检查是否混入了非当前环境的 DLL。
- 当前 EXE 已验证能启动到正常主窗口标题 `3D Scanner - 模型拍摄工具`，但仍建议后续用 EXE 做一次完整导入 OBJ 和拍摄 smoke test。

风险和待验证：

- PyInstaller 输出目录约 1GB，后续如果需要分发给他人，应再做体积优化和干净机器验证。
- 构建日志中有 Qt SQL 驱动的可选依赖缺失警告，当前软件不使用 Qt SQL，暂不影响启动；若后续引入数据库功能需要重新评估。
- 当前目录不是 git 仓库，无法用 git 状态追踪本次改动，后续建议纳入版本管理或至少保留打包前后文件清单。

后续动作：

- 用正式 EXE 进行一次“导入 OBJ -> 调整视图/模型 -> 选择 64/96 张 -> 完整拍摄输出”的端到端检查。
- 检查 `camera_poses.json` 数量与 `track*_*.png` 数量是否一致，作为后续自动化 smoke test 的基础。
- 评估是否需要压缩打包体积，或为交付增加 README/使用说明。

### 2026-07-07

背景：

- 用户提供了 ChatGPT 中整理出的“中间补全模型设计”文本，希望同步到项目文档中。
- 当前数据集结构已经包含 `incomplete_rgb`、`complete_rgb`、`missing_mask`、`visible_mask` 和 `camera_poses`，可以支撑后续缺失识别和图像补全流程。

已确定结论：

- 新增 `docs/workflow.md`，用于承接“总工作流 + 中间补全模型设计”，和 `docs/DATASET_WORKFLOW.md` 分工：前者讲训练/推理流程，后者讲数据如何生成。
- `docs/DATASET_WORKFLOW.md` 新增“与中间补全模型的衔接”，明确每类数据在训练中的用途。
- `README.md` 增加 `docs/workflow.md` 入口。
- 最终系统流程明确为：`incomplete_rgb -> Missing Region Detector -> pred_missing_mask -> Inpainting Model -> completed_rgb`，即先识别缺失区域，再补全图像。
- 实验开发顺序明确为：先使用 GT `missing_mask` 做 Oracle-mask inpainting baseline，再训练缺失区域识别模型，最后接入预测 mask 的完整 pipeline。
- AnimateDiff 不负责识别缺失区域，它的定位是将按相机轨迹排序的多视角图像当作伪视频序列，增强补全结果的跨视角一致性。
- VLM / LLM 不替代像素级 `missing_mask` 监督，它的定位是生成建筑结构语义提示，辅助补全模型生成更合理的建筑结果。

技术判断：

- 先做 GT-mask baseline 可以把问题拆开：GT mask 下补不好说明补全模型弱；GT mask 好但 pred mask 差说明缺失区域识别弱。
- 缺失区域识别本质是分割任务，后续可以从 U-Net 或 SegFormer 这类稳定模型开始，再考虑 SAM/SAM2 辅助。
- AnimateDiff 可以缓解外观、纹理和局部结构的不一致，但不能严格保证三维几何一致性；后续仍需要 camera pose、depth、normal、edge、visible_mask 或多视角投影约束。

风险和待验证：

- 多视角图像作为伪视频序列并不等同于真实视频，AnimateDiff 的收益需要通过重建结果和跨视角一致性指标验证。
- VLM 生成的语义提示可能不稳定，需要设计可复现的 prompt 模板或缓存机制。
- 当前只是文档和路线整理，还未实现数据读取器、baseline、缺失检测或 AnimateDiff 接入。

后续动作：

- 实现数据读取器，统一读取 `incomplete_rgb`、`complete_rgb`、`missing_mask`、`visible_mask`、`camera_poses.json` 和 `metadata.json`。
- 先实现 `incomplete_rgb + GT missing_mask -> completed_rgb` 的单视角 baseline。
- 再实现 `incomplete_rgb -> pred_missing_mask` 的缺失区域识别模型。
- 最后接入 predicted-mask pipeline、VLM 语义提示和 AnimateDiff 多视角一致性。

### 2026-07-09

背景：

- 继续把软件从普通 OBJ 拍照工具改造成 Dataset Mode 数据集生成工具。
- 用户重新澄清数据来源：普通模式先负责摆正并冻结完整模型，Blender 再基于冻结后的 `aligned_complete.obj` 制造三元组。
- 用户明确要求继续遵循普通拍照模式的摆放准则：模型中心在四轨中心、底面平行灰色虚拟平面、顶面朝四轨交叉相机出发点。

已确定结论：

- Dataset Mode 默认模式最终确认为 `prealigned_triplet` / frozen triplet。
- 数据集输入仍是一组三元组：`complete.obj`、`incomplete.obj`、`removed.obj`。
- 三元组必须共享同一个冻结后的标准世界坐标系。
- Dataset Mode 禁止 normalize、recenter、rescale、reorient，禁止根据 `complete.obj`、`incomplete.obj` 或 `removed.obj` 重新计算 transform。
- 如果内部需要 OBJ -> PyVista 固定坐标转换，只允许对三者应用同一个固定转换。
- `shared_triplet` 降级为非默认实验路径；默认导入数据集不走该模式。
- Dataset Mode 输出四类图片：`complete_rgb`、`incomplete_rgb`、`missing_mask`、`visible_mask`，同一帧共享同一相机位姿和相机参数。
- `missing_mask` 表示当前视角下 removed 几何真正可见的缺失区域；`visible_mask` 表示当前视角下 incomplete 主体的可见区域。
- `camera_poses.json` 需要记录 `camera_model`、逐帧 `camera`、四路输出路径和 `failed_outputs`；`metadata.json` 记录数据模式、坐标规则、`alignment_report`、源文件和保存失败列表。

架构变化：

- 新增 `src/core/dataset_loader.py`，提供三元组读取；默认 `prealigned_triplet` 只 parse 三个 OBJ，不改动坐标。
- `src/core/obj_loader.py` 新增 `parse_obj_file()`，用于 Dataset Mode 原样解析 OBJ，不走普通模式的缩放标准化。
- `src/ui/tk_widget.py` 新增 Dataset Mode 导入与拍摄分支；普通 OBJ 模式仍走原来的 `OBJLoader.load()` 路径。
- Dataset Mode 禁止“调整模型”，避免用户单独旋转三元组中的某一部分导致 mask 错位。
- Dataset Mode 拍摄时复用四个离屏 renderer，避免每张图新建 Plotter 导致卡顿。
- Dataset Mode 中如果截图返回 `None`，现在直接视为失败，不再静默生成空白训练样本。
- 保存失败会写入 `camera_poses.json` 和 `metadata.json` 的 `failed_outputs`，方便后续数据清洗。
- 文档 `README.md`、`docs/DATASET_WORKFLOW.md`、`docs/workflow.md` 已同步为“普通模式冻结模型，Dataset Mode 只读冻结三元组”新准则。

技术判断：

- 普通模式和 Dataset Mode 必须职责分离：普通模式用于 Prepare / Alignment，Dataset Mode 只用于批量渲染冻结三元组。
- 不在 Dataset Mode 中做缩放、旋转或平移，能最大限度保证 Blender 切出的 `incomplete` 与 `removed` 不发生相对错位。
- mask 与 RGB 是否像素级对齐，核心取决于三元组共享冻结坐标、同一固定 OBJ->PyVista 转换、同一 camera pose 和同一渲染参数。

风险和待验证：

- 此前实现过 `shared_triplet` 默认路径并经 sub agent 审查到 94/100；用户最终确认默认流程应改为冻结三元组只读，因此需要以新准则重新验证。
- 普通模式导出 `aligned_complete.obj` 的功能还需要明确实现或确认现有“保存位置”是否已经等价于冻结导出。
- 目前完成了 `py_compile` 和临时 cube 三元组 smoke test；仍需用真实建筑三元组做 GUI 导入、拍摄、mask 对齐检查。

后续动作：

- 用真实冻结三元组文件夹测试 `prealigned_triplet` 导入和四路输出。
- 检查 `complete_rgb` / `incomplete_rgb` / `missing_mask` / `visible_mask` 是否像素级对齐。
- 批量扫描真实 OBJ，统计哪些模型底面识别失败或姿态验收失败。
- 后续编写三元组生成脚本：从 complete 切出 incomplete 和 removed，并记录随机种子与切割参数。

## 待办池

- [ ] 设计数据集生成脚本的最小版本。
- [ ] 确定残缺生成策略：连通面删除、球体切割、盒体切割或平面切割。
- [ ] 用真实数据验证 `camera_poses.json` 的 `camera_model`、逐帧 `camera`、四路输出路径和 `failed_outputs` 是否满足训练/重建读取需求。
- [ ] 继续完善 `metadata.json` 字段，补充残缺生成方法、随机种子、切割参数和软件版本/git commit。
- [ ] 设计单帧 inpainting baseline。
- [ ] 研究 AnimateDiff 如何接收 mask 和残缺图条件。
- [ ] 研究 camera pose / depth / normal 条件如何注入补全模型。
- [ ] 设计最终 MVS 重建评估指标。
- [ ] 用正式 EXE 做导入 OBJ 与 64/96 张拍摄端到端 smoke test。
- [ ] 增加拍摄输出一致性检查：图片数量、轨道分布、`camera_poses.json` 条目数必须一致。
- [ ] 评估 PyInstaller 输出体积优化和干净机器启动验证。
- [ ] 实现中间模型数据读取器，支持同一建筑多视角 batch。
- [ ] 实现 GT-mask / Oracle-mask inpainting baseline。
- [ ] 实现缺失区域识别模型并用 IoU、Dice、Precision、Recall 评估。
- [ ] 对比 GT mask 与 pred mask 的补全结果，定位误差来源。
- [ ] 设计 VLM / LLM 建筑结构语义提示模板。
- [ ] 将多视角图像组织为伪视频序列，验证 AnimateDiff 对跨视角一致性的收益。

## 日志模板

```text
### YYYY-MM-DD

背景：

- 

已确定结论：

- 

技术判断：

- 

风险和待验证：

- 

后续动作：

- 
```
