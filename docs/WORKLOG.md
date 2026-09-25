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

### 视频领域微调准则

- 建筑领域 LoRA 是中间阶段：先学习“白色背景下旋转建筑模型”的视频分布，不等同于最终的残缺补全模型。
- 训练按 20 个视频 smoke test、200 个视频试训、全部数据训练逐级扩展；前一级未验证通过时，不直接启动下一级。
- 基础模型与 LoRA 的验证必须使用固定 prompt、固定随机种子和相同推理参数，保留可比较输出。
- 当前建筑视频训练不需要音频；音频库只是 DiffSynth 通用训练入口的启动依赖。
- 12GB GPU 上先采用低分辨率、短帧和 CPU offload 验证链路，再根据显存与耗时逐级提高到目标配置。
- 当前 81 帧源视频应拆为五个有一帧重叠的 17 帧窗口，避免 DiffSynth 在 `num_frames=17` 时只读取前 17 帧而遗漏后续轨迹。
- 双 GPU DDP 不会合并两张显卡的显存；Windows 双卡训练使用 Gloo 同步，并采用“单卡预缓存 T5/VAE + 双卡训练 DiT LoRA”的两阶段架构。
- 正式论文实验必须保留 `train/val/test=1912/239/239` 独立划分。全库五段为 11950 个片段，但正式训练集五段应为 9560 个，不能把验证集和测试集并入训练。
- 当前 `balanced` 是按源视频序号在五个窗口间均衡分配，每个视频固定选择一段；它不会自动随 epoch 轮换。完整轨迹训练应显式使用 `all`，或另行实现 epoch 轮换。
- checkpoint 选择以固定条件下的视觉验证为准，不能只看 loss 或默认选择最后一步；现有 200 视频试验中 `step-100` 优于 `step-200`。

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

### 冻结完整模型导出结构

普通模式导出的冻结完整模型用于进入 Blender 切割阶段。导出时应保持以下约定：

- 导出入口与拍照输出分离：`导出位置` 只负责 aligned complete OBJ，`保存位置` 只负责照片 / 数据集图片。
- 导出坐标必须是 OBJ 语义坐标，不是 PyVista / 软件世界坐标。
- 软件内部正向固定转换为 `pv=(obj_z,obj_x,obj_y)`；导出时反向转换为 `obj=(pv_y,pv_z,pv_x)`。
- 导出前必须固化当前普通模式模型姿态，确保用户手动调整后的视觉姿态已经写回顶点。
- 当前导出目录结构约定：

```text
export_root/
├── aligned_complete/
│   ├── 1.obj
│   ├── 2.obj
│   └── ...
└── aligned_complete_json/
    ├── 1.json
    ├── 2.json
    └── ...
```

其中 `N.obj` 与 `N.json` 一一对应，从 `1` 开始自动递增。

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

### 2026-07-10

背景：

- 继续补齐“普通模式冻结完整模型 -> Blender 切割 -> Dataset Mode 只读三元组”的前置工具链。
- 用户希望普通模式不仅能拍照，也能稳定导出已对齐的完整模型，且导出目录和照片保存目录分离。
- 用户明确导出文件需要按序号组织，便于批量处理多个完整模型。

已确定结论：

- Dataset Mode 默认逻辑继续保持 `prealigned_triplet`，不启用 `shared_triplet`，不改变三元组只读规则。
- Dataset Mode 元数据文件名统一为 `metadata.json`，不再使用 `dataset_meta.json`。
- `prealigned_triplet` 的 `alignment_report` 必须明确记录未应用任何几何对齐：
  - `normalize_applied=false`
  - `recenter_applied=false`
  - `rescale_applied=false`
  - `reorient_applied=false`
  - `geometry_based_alignment_applied=false`
- 普通模式新增“导出已对齐完整模型”能力，导出前先固化当前 actor 视觉变换，再把 PyVista 坐标反向转换回 OBJ 语义坐标。
- `导出位置` 与 `保存位置` 分离：前者只负责 `aligned_complete` 导出，后者继续负责照片 / 数据集输出。
- 冻结完整模型导出结构最终确定为：

```text
export_root/
├── aligned_complete/
│   ├── 1.obj
│   ├── 2.obj
│   └── ...
└── aligned_complete_json/
    ├── 1.json
    ├── 2.json
    └── ...
```

架构变化：

- `src/ui/tk_widget.py` 新增普通模式导出入口：
  - `PyVistaView.export_aligned_complete(filepath, meta_path=None)`
  - `MainWindow._on_export_aligned()`
  - `MainWindow._on_export_folder()`
- 新增反向固定转换 `_pv_to_obj_coords()`，与 `_obj_to_pv_coords()` 成对：
  - 正向：`pv=(obj_z,obj_x,obj_y)`
  - 反向：`obj=(pv_y,pv_z,pv_x)`
- Project 面板新增 `导出位置` 按钮，独立记忆 aligned complete 的导出目录。
- `导出已对齐完整模型` 会自动扫描下一个可用编号，写入 `aligned_complete/N.obj` 和 `aligned_complete_json/N.json`。
- 导出 meta 记录 `exported_from_normal_mode`、`coordinate_space`、`inverse_fixed_conversion`、`intended_next_step`、源 OBJ 路径、顶点数和面数。

技术判断：

- `aligned_complete.obj` 必须使用 OBJ 语义坐标导出；如果直接导出 PyVista 坐标，Dataset Mode 后续再次执行 OBJ -> PyVista 固定转换会导致方向错乱。
- “导出位置”和“保存位置”必须分离，避免用户把准备阶段的冻结模型输出和拍照 / 数据集输出混在一起。
- 按编号分目录保存比连续覆盖单个 `aligned_complete.obj` 更适合批量整理与后续 Blender 处理。

风险和待验证：

- 当前导出顶点来源为 `self._mesh.points`，面来源为 `self._faces`。由于普通显示 mesh 可能通过 `MAX_TRIANGLES` 抽稀显示面，长期更稳妥的顶点来源应评估改为完整顶点缓存 `self._verts_base`。
- 需要用真实大模型验证导出的 `aligned_complete/N.obj` 在 Blender 中可正常打开，并且基于它切出的三元组回到 Dataset Mode 后与普通模式冻结姿态一致。
- 需要确认导出的 OBJ 面数、顶点数是否与普通模式导入后的完整几何一致。

后续动作：

- 用真实建筑 OBJ 测试：导入普通模式 -> 自动/手动摆正 -> 导出 `aligned_complete/N.obj` -> Blender 打开。
- 基于导出的 `N.obj` 制造一组三元组，再用 Dataset Mode 导入，检查 `complete_rgb`、`incomplete_rgb`、`missing_mask`、`visible_mask` 是否对齐。
- 评估并实现导出顶点源从 `self._mesh.points` 切换到 `self._verts_base`，避免显示抽稀逻辑影响完整几何导出。

### 2026-07-13

背景：

- 自动工作日志更新任务触发。

已确定结论：

- 无新增技术决策。

### 2026-07-14

背景：

- 用户提出为项目构建训练代码与验证代码。
- 已只读检查现有训练流程文档、仓库结构、样例输出、Python 环境和 GPU 环境，尚未修改训练相关文件。

已确定结论：

- 当前仓库仍以数据生成软件为主体，尚无独立训练框架、训练入口或验证入口。
- 仓库中的 `model00023`、`model00024` 是普通模式照片样例，不具备 Dataset Mode 训练所需的 `complete_rgb`、`incomplete_rgb`、`missing_mask`、`visible_mask` 四路配对结构。
- 现有文档定义的实验顺序保持不变：先做 GT-mask / Oracle-mask 图像补全 baseline，再训练缺失区域识别模型，最后比较 GT mask 与预测 mask 驱动的补全结果。
- 本轮尚未确定是只实现 GT-mask baseline，还是同时实现缺失区域识别与串联验证；也尚未取得真实 Dataset Mode 数据集路径。

技术判断：

- 训练与验证框架应读取 Dataset Mode 的冻结配对输出，并按建筑或残缺实例划分 train / val / test，不能按单帧随机拆分导致同一模型视角泄漏到不同集合。
- 当前软件运行环境为 Python 3.8，机器检测到 RTX 3060 12GB GPU；训练环境建议与 PyVista / PySide6 软件环境分离，具体 Python、PyTorch 和 CUDA 版本待用户确认后落地。
- 在没有真实四路配对样本前，可以搭建代码和合成 smoke test，但不能把合成测试通过视为真实训练数据链路已经验收。

风险和待验证：

- 训练任务范围未确认，直接同时实现多个模型可能扩大首阶段工作量并掩盖数据读取问题。
- 尚未验证真实 `missing_mask` 的二值范围、图片分辨率、帧命名完整性及 `camera_poses.json` 与四路图片的一一对应关系。
- 需要确认真实数据集根目录，以及是否允许建立独立训练环境。

后续动作：

- 用户确认训练范围、真实数据集路径和独立训练环境后，先实现数据索引与泄漏安全的数据集划分。
- 搭建 GT-mask 补全训练、验证、指标与检查点保存的最小闭环，再实现缺失区域识别和串联验证。
- 使用真实样本执行端到端验证；若真实数据尚未生成，则先用合成小数据做 smoke test，并明确区分两类测试结果。

### 2026-07-15

背景：

- 自动工作日志更新任务触发。

已确定结论：

- 无新增技术决策。

### 2026-07-16

背景：

- 自动工作日志更新任务触发。

已确定结论：

- 无新增技术决策。

### 2026-07-17

背景：

- 自动工作日志更新任务触发。

已确定结论：

- 无新增技术决策。

### 2026-07-18

背景：

- 自动工作日志更新周期内无新的实质工作结果。

已确定结论：

- 无新增技术决策。

### 2026-07-19

背景：

- 用户查询现有项目文档的名称与分工，已确认 `WORKLOG.md`、`workflow.md` 和 `DATASET_WORKFLOW.md` 的用途。

已确定结论：

- 无新增技术决策。

### 2026-07-20

- 无新增技术决策。

### 2026-07-21

- 无新增技术决策。

### 2026-07-22

- 无新增技术决策。

### 2026-07-23

- 无新增技术决策。

### 2026-07-24

- 无新增技术决策。

### 2026-07-25

- 无新增技术决策。

### 2026-07-26

- 无新增技术决策。

### 2026-07-27

- 无新增技术决策。

### 2026-07-28

- 无新增技术决策。

### 2026-09-01

背景：
- 四轨照片已整理为面向视频微调的数据：`src/video_wan81` 中共有 2390 个 MP4，已检查为 480x480、每段 81 帧，并生成 `metadata.csv`。
- 当前阶段从数据生成转入建筑领域视频模型微调，目标是先让基础模型学习“白色背景下旋转建筑模型”的领域分布；这一步不是最终的残缺补全训练。

已确定结论：
- 采用 DiffSynth-Studio 的 Wan2.1-T2V-1.3B LoRA 训练链路，仓库位于 `D:/code/DiffSynth-Studio`。
- 训练环境独立放在 `D:/scj/envs/diffsynth`，使用 Python 3.10、PyTorch 2.5.1+cu121；CUDA 已验证可用。
- 先进行 20 个视频的 smoke test，再依次扩展到 200 个和全部 2390 个视频。
- smoke test 使用 `metadata_smoke.csv`，初始配置为 256x256、17 帧、LoRA rank 8、1 epoch，并启用模型/优化器 CPU offload 与梯度检查点 offload。
- 训练指定 GPU 1，GPU 0 保留给其他图形任务；`CUDA_VISIBLE_DEVICES=1` 后，训练进程内部显示为设备 0 属于正常重编号。
- 当前建筑视频不需要音频；训练脚本仅因通用初始化而依赖 `librosa`，已在训练环境中安装 `librosa 0.11.0`。

架构与环境处理：
- `accelerate.exe` 保留了已失效的旧环境路径，不能直接使用；改为通过 `D:/scj/envs/diffsynth/python.exe -m accelerate.commands.launch` 启动训练。
- 已确认训练入口为 `examples/wanvideo/model_training/train.py`，LoRA 官方参考配置为 `examples/wanvideo/model_training/lora/Wan2.1-T2V-1.3B.sh`。

风险和待验证：
- 尚未确认 Wan2.1-T2V-1.3B 基础权重已完成下载，也尚未确认首轮 smoke test 能完整输出 LoRA 权重。
- 12GB 显存下 480x480、81 帧训练可能非常慢或显存不足；必须根据 256x256、17 帧 smoke test 的显存和耗时再逐级提高配置。
- 当前 prompt 基本相同，只适合验证建筑领域适配链路；后续需要更有区分度、可复现的建筑描述模板。
- 建筑领域 LoRA 不能替代后续基于 incomplete、mask、camera pose 的条件补全模型。

后续动作：
- 运行 20 视频 smoke test，确认模型下载、数据读取、loss、显存占用、checkpoint 和 CSV 日志均正常。
- smoke test 成功后编写验证脚本，用固定 prompt 和随机种子对基础模型/LoRA 输出做并排比较。
- 再扩展到 200 视频试训，确认训练稳定后才启动全量 2390 视频训练。

### 2026-09-02

背景：

- 完成 20 视频 smoke、200 视频单卡试训以及 Windows 双 RTX 3060 的训练链路验证。
- 发现 81 帧视频在 DiffSynth 使用 `num_frames=17` 时只读取前 17 帧，后 64 帧没有参与原 200 视频试训。

已确定结论：

- 200 视频试验中，`step-100` 的建筑形态、连续视角变化和几何稳定性最佳；`step-200` 出现细长、低细节退化，说明当前 `1e-4` 学习率偏高且后期过拟合，下一轮建议从 `5e-5` 起步。
- 81 帧视频按 `1-17 / 17-33 / 33-49 / 49-65 / 65-81` 拆为五段，相邻段共享一帧。
- 全部 2390 视频拆五段得到 11950 个理论片段；正式训练必须沿用冻结划分，只训练 `1912×5=9560` 个片段，val/test 各为 1195 个片段。
- `balanced` 当前只让不同建筑均匀分布到五个时间位置，不会在 epoch 间自动轮换；参数确认后应使用 `all` 完整覆盖训练轨迹。
- 双卡采用 Gloo file rendezvous 的真实 DDP，但两张 12GB RTX 3060 不会合并为 24GB，因此不直接把 81 帧作为单个训练样本。
- 双卡训练采用 split-cache：第一阶段单 GPU 缓存 T5/VAE 输出，第二阶段双 GPU 同步训练 DiT LoRA。

架构与实现：

- 新增 `training/prepare_temporal_clips.py`，支持 `all` 和 `balanced` 两种 17 帧切片策略，并验证 81 帧完整覆盖。
- 新增 `training/check_dual_gpu.py`，已验证两张 GPU 的 DDP 梯度校验值均为 80.0，梯度同步成立。
- 新增 `training/wan_train_entry.py`，解决 Windows 下标准 Accelerate 多卡依赖 NCCL/libuv 的问题。
- 新增 `training/train_split_domain_lora.py`，编排 cache/train 两阶段并记录 launcher manifest。
- 新增 `dual_3060_smoke.json` 和尚未运行的 `dual_3060_smoke_384.json`。
- 已生成详细交接文档 `docs/HANDOFF_2026-09-02.md`。

验证结果：

- 256×256、5 个切片的双卡 smoke 成功，生成 `step-3.safetensors`；训练阶段两张 GPU 均有实际显存占用。
- 训练工具 `py_compile` 通过，`tests.test_training_tools` 共 10 项测试全部通过。
- 200 视频人工审查记录位于 `training/reviews/building_pilot_200.json`，当前最佳权重为 `building_pilot_200_lora/step-100.safetensors`。

风险和待验证：

- 384×384 双卡配置尚未 dry-run 或实际运行。
- `wan_train_entry.py` 最后加入的跨 rank 平均 loss 日志已通过语法和单元测试，但尚未重新进行真实双卡 smoke 验证。
- 当前训练 caption 基本完全相同，容易使模型学习“平均建筑”；后续应设计可复现且有区分度的建筑描述。
- 当前领域 LoRA 只证明建筑与局部旋转先验学习，不能证明残缺区域补全能力。

后续动作：

- 先重新跑 256 双卡 smoke，核验跨 rank 平均 loss 日志。
- 再测试 384 smoke 的显存和速度。
- 随后使用 500 个 train 建筑的 `balanced` 切片做 `5e-5` 参数试验；参数稳定后再对 1912 个 train 视频使用 `all` 生成 9560 个片段训练。

### 2026-09-03

- 无新增技术决策。

### 2026-09-04

- 无新增技术决策。

### 2026-09-08

- 无新增技术决策。

### 2026-09-09

- 无新增技术决策。

### 2026-09-10

- 无新增技术决策。

### 2026-09-11

- 无新增技术决策。

### 2026-09-12

- 无新增技术决策。

### 2026-09-13

- 无新增技术决策。

### 2026-09-14

- 无新增技术决策。

### 2026-09-15

- 无新增技术决策。

### 2026-09-16

- 无新增技术决策。

### 2026-09-17

- 无新增技术决策。

### 2026-09-18

- 无新增技术决策。

### 2026-09-21

- 无新增技术决策。

### 2026-09-22

- 无新增技术决策。

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
