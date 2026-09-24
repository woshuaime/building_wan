# Project Context

## 1. Current Goal

- 在学校服务器上使用单张 A100 80GB 从头训练 Wan2.1-T2V-1.3B 建筑领域 LoRA。
- 使用完整81帧旋转视频，使模型学习一般建筑模型的外观和完整环绕变化，而不是只学习少量视角。
- 保持正式配置为384×384、81帧、LoRA rank 16、8 epochs。
- 使用逐样本BF16视频latent和单份共享BF16文本context降低共享存储占用，避免旧缓存约170GB的预计规模。
- 训练后按阶段权重生成固定条件的视频，判断外观记忆、视角覆盖和时序稳定性。
- A6000单轨实验在另一台实验室服务器上进行；验证入口为 `scripts/validate_wan81_a6000.py`，使用该实验的统一prompt比较基础模型与最新LoRA。

## 2. Current Status

已完成：

- 服务器目录、数据、基础模型、tokenizer、训练入口和Slurm脚本已落地。
- `building_v1` 固定划分已存在：train 1912、val 239、test 239、pilot 200，划分交叉审计为0。
- 单卡A100环境已在真实Slurm作业内确认可用，BF16和DiffSynth导入曾验证通过。
- Slurm脚本已简化为资源申请后调用独立训练脚本。
- 训练名称、Slurm job name、日志前缀和当前输出目录统一为 `building_wan`。
- `librosa` 缺失问题已解决，当前版本0.11.0。
- 缓存逻辑已升级为 `compact-sft-shared-context-bf16-v2`：每个样本只保存BF16视频latent和必要标量，相同prompt的BF16文本context全局只保存一份。
- 按真实Wan张量形状进行序列化模拟：单样本约1,549,837字节，共享context约4,196,325字节，1912份预计约2.764GiB；相比上一版10.229GiB再减少72.98%。
- 2026-09-14按用户明确要求删除了旧版残缺的222份缓存，释放约20GB；原始数据、模型和checkpoint未删除。

正在进行：

- 正式训练已完成，正在准备固定提示词和种子的阶段权重推理验证。

尚未完成：

- 尚未生成本次A100实验的阶段验证视频和最终效果结论。

## 3. Environment

- Server/user：`u2025171963@workstation`
- OS/kernel：Linux `4.18.0-80.7.1.el8_0.x86_64`
- Workspace：`/home/share/CHUANJUN/building_wan`
- Conda env：`mv2v`
- Python：3.10.20，`/home/u2025171963/miniconda3/envs/mv2v/bin/python`
- PyTorch：2.10.0+cu128；CUDA runtime：12.8
- GPU：Slurm `gpu` 分区的 `gpu1`，2×A100可供节点调度；本项目只申请1×A100 80GB。
- 登录节点未分配GPU，当前登录会话中CUDA不可用是预期行为。

## 4. Important Paths

- Dataset：`/home/share/CHUANJUN/building_wan/data/video_wan81`
- Train metadata：`/home/share/CHUANJUN/building_wan/data/splits/building_v1/metadata_train.csv`
- Compact cache：`/home/share/CHUANJUN/building_wan/cache/wan21_t2v_13b_building_384x384_f81_train`
- DiT：`/home/share/CHUANJUN/building_wan/models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors`
- T5：`/home/share/CHUANJUN/building_wan/models/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors`
- VAE：`/home/share/CHUANJUN/building_wan/models/Wan-Series-Converted-Safetensors/Wan2.1_VAE.safetensors`
- Tokenizer：`/home/share/CHUANJUN/building_wan/models/Wan2.1-T2V-1.3B/google/umt5-xxl`
- Training entry：`/home/share/CHUANJUN/building_wan/code/Python_3D_Scanner/training/wan_train_entry.py`
- Train script：`/home/share/CHUANJUN/building_wan/scripts/train_wan81.sh`
- Slurm script：`/home/share/CHUANJUN/building_wan/scripts/submit_wan81.sh`
- Interactive launcher：`/home/share/CHUANJUN/building_wan/scripts/start_wan81.sh`
- Current output：`/home/share/CHUANJUN/building_wan/checkpoints/building_wan`
- Logs：`/home/share/CHUANJUN/building_wan/logs/slurm`

## 5. Confirmed Configuration

- Model：Wan2.1-T2V-1.3B
- Resolution：384×384
- Frames：81
- Dataset：1912个 `building_v1` train样本，dataset repeat 1
- GPU/process：1×A100，1个训练进程
- LoRA base/targets：`dit`；`q,k,v,o,ffn.0,ffn.2`
- LoRA rank：16
- Learning rate：2e-5
- Weight decay：0.01
- Epochs：8
- Save steps：400
- Gradient accumulation：1
- Gradient checkpointing：启用
- Cache：逐样本BF16视频latent + 单份共享BF16文本context，格式 `compact-sft-shared-context-bf16-v2`
- Training offload：T5和VAE
- Cache offload：DiT
- Dataset workers：缓存4，训练2
- Expected sample visits/optimizer steps：1912×8 = 15,296（单进程、accumulation 1）
- Slurm：8 CPUs、192GB RAM、最长10天、comment `videoedit2025`

## 6. Important Decisions

- 舍弃双RTX 3060作为新正式方案，改用单卡A100 80GB。
- 旧双3060实验权重保留作历史证据，但当前A100实验不resume它们。
- 当前A100实验从step 0、epoch 0开始。
- 当前实验名称固定为 `building_wan`，不使用Markdown形式的反斜杠。
- Slurm脚本只申请资源，训练参数集中在 `train_wan81.sh`。
- 同一缓存/输出目录禁止并发写入，提交前必须检查队列。
- 缓存采用共享文本context的紧凑BF16格式；训练读取时在项目层恢复原三元组，训练配置和最终BF16输入不变。
- 未经用户明确指令，不执行 `sbatch`，不自动resume，不删除旧checkpoint。
- 新的后续实验需要用户确认名称并使用独立输出目录，不能自动覆盖 `building_wan`。

## 7. Changes Made

- `scripts/submit_wan81.sh`：整理为单卡A100的简洁Slurm脚本，任务名为 `building_wan`。
- `scripts/train_wan81.sh`：集中当前正式参数；使用绝对Python路径；增加路径、GPU、缓存数量和输出冲突检查；接入共享文本context缓存及v2格式标记。
- `scripts/start_wan81.sh`：提交后自动跟踪stdout/stderr；支持传入Job ID重新连接现有任务；Slurm状态查询临时超时时继续显示日志，不再误判任务结束。
- `scripts/validate_wan81.sh`和`submit_wan81_validation.sh`：使用本地基础模型和固定验证配置，在单张A100上生成base及阶段LoRA对比视频。
- `scripts/validate_wan81_a6000.py`：为单轨A6000实验按实际最高step生成独立验证配置；默认只做预检，`--run` 才在训练结束后生成base与LoRA对比视频。
- `code/Python_3D_Scanner/training/wan_train_entry.py`：支持单进程直接运行；样本缓存只保存必要BF16视频latent，并原子保存/加载单份共享BF16文本context；拒绝多prompt误用共享context。
- `code/Python_3D_Scanner/training/validate_domain_lora.py`：新增服务器本地模型文件和tokenizer路径支持，可关闭低显存磁盘offload以使用A100推理。
- `AGENTS.md`：新增长期项目规则。
- `PROJECT_CONTEXT.md`：新增当前阶段上下文和新会话接手流程。
- `EXPERIMENTS.md`：新增历史与计划实验台账。
- `TRAINING.md`：新增服务器训练操作手册。
- `SESSION_HANDOFF.md`：新增本轮工作交接摘要，供下一次Codex会话快速接手。
- 已删除：缓存目录中旧版残缺的222个 `.pth` 文件；不可恢复，但这些不是模型权重或原始数据。

## 8. Known Issues

- Job 1155实测共享context正确单独保存为BF16约4.20MB，样本文件不含重复context；但Accelerate在`forward`返回后把样本latent转回FP32，实际单份约3.10MB，完整预计约5.52GiB，而不是模拟的2.764GiB。训练读取时仍会转为BF16，影响存储量但不改变训练输入精度。
- 作业日志会出现 `Can't initialize NVML`；它影响监控接口，但此前CUDA计算本身可用。
- 内核4.18低于Accelerate提示的推荐5.5，存在潜在挂起风险；当前无权限升级，需靠日志与Slurm状态观察。
- 服务器没有 `quota` 和 `lfs` 命令，无法直接查询个人配额；共享文件系统整体空间充足不等于个人配额已确认。
- 项目根目录及两个代码目录都不是Git worktree；没有可用分支、commit和 `git diff`。建议后续补建版本控制或从真实仓库clone。
- 当前A100输出目录尚无权重，正式效果未知。

## 9. Failed / Rejected Approaches

- 方案：继续使用本地双RTX 3060正式训练。
  - 为什么不用：81帧完整视频训练受显存与速度约束，旧实验只覆盖较短片段，用户认为视角变化不足。
  - 替代方案：单张A100 80GB、384×384、81帧、8 epochs。
- 方案：A100实验resume旧双3060 LoRA。
  - 为什么不用：旧实验参数和目标不同，用户明确要求新A100实验使用新权重。
  - 替代方案：从基础模型step 0开始新的 `building_wan` 实验。
- 方案：使用 `mv2v_dual` 环境。
  - 为什么不用：它是旧双卡环境，且早期作业出现DiffSynth导入问题。
  - 替代方案：`mv2v` 的绝对Python路径。
- 方案：把原始81帧、噪声和重复上下文全部写入每个缓存。
  - 为什么不用：单份约97MB，完整数据预计约170GB，写到约210/221份时出现 `torch.save` iostream错误。
  - 第一阶段替代方案：`compact-sft-bf16-v1`，预计约10.23GiB。
- 方案：在1912个缓存中重复保存相同prompt的BF16文本context。
  - 为什么不用：相同的约4.2MB context重复1912次，占上一版缓存约73%。
  - 替代方案：`compact-sft-shared-context-bf16-v2`，逐样本只存视频latent，context只存一份，预计约2.764GiB。
- 方案：同时提交两个作业写同一个缓存目录。
  - 为什么不用：作业1149和1150并发生成相同缓存，均失败，并可能导致文件竞争或损坏。
  - 替代方案：提交前检查 `squeue`，同一实验只保留一个写入作业。
- 方案：把示例任务号写死后直接 `tail`。
  - 为什么不用：实际Job ID不同，日志文件不存在。
  - 替代方案：使用 `sbatch --parsable` 返回的真实Job ID，或运行 `start_wan81.sh`。

## 10. Next Steps

1. 新会话先按下方启动规则完成只读检查和状态汇报。
2. 用户明确要求开始提交验证后，确认`squeue`为空并提交一个`building_wan_val`作业。
3. 生成base、step-4000、step-8000、step-12000和step-15296五个固定条件的384×384、81帧视频。
4. 运行视频诊断和contact sheet，对比视角覆盖、结构稳定、建筑外观以及是否过拟合。
5. 后续重建缓存前修复Accelerate输出回转，使latent真正以BF16落盘；现有缓存可正常用于训练，不需为本次验证重建。
6. 验证状态变化后更新本文件和 `EXPERIMENTS.md`。

## 11. Last Updated

- Date：2026-09-16T01:35:05+08:00
- Git branch：不可用；当前目录不是Git worktree
- Git commit：不可用；当前目录不是Git worktree

## New Codex Session Startup

新Codex对话开始后，第一步必须：

1. 阅读 `AGENTS.md`。
2. 阅读 `PROJECT_CONTEXT.md`。
3. 阅读 `TRAINING.md`。
4. 阅读 `EXPERIMENTS.md`。
5. 执行 `git status` 和 `git diff`；若仍非Git项目，如实记录不可用。
6. 查看当前分支、最近commit；若不可用，不得猜测。
7. 执行 `pwd` 并核对workspace目录。
8. 查看 `squeue -u u2025171963`、缓存数量、checkpoint和最近日志。

完成后先向用户汇报：当前项目目标、完成进度、关键配置、未解决问题和下一步建议。在完成以上步骤前，不直接修改代码。

## 12. Latest Handoff Update

- 本次handoff核验时间：2026-09-16T01:35:05+08:00。
- Job 1155于2026-09-14T23:20:08+08:00以`COMPLETED / 0:0`结束，总耗时13:04:29；当前Slurm队列为空。
- 正式训练完成15,296个optimizer steps，保存step-400至step-15296共39份LoRA，最终loss为0.0207207。
- 已生成loss曲线`outputs/validation/building_wan/loss_curve.png`。
- 已准备服务器验证配置及单卡Slurm脚本，预检通过；尚未提交验证作业、尚无验证视频。
- 当前目录仍不是Git worktree，无法提供分支、commit或diff。
- 本轮完整交接摘要见 `SESSION_HANDOFF.md`。

## 13. A6000 Batch Training Variant

- 新增 `scripts/train_wan81_a6000.sh` 和 `scripts/submit_wan81_a6000.sh`，不修改旧A100脚本，也不复用已有 `checkpoints/building_wan` 输出目录。
- A6000脚本默认使用 `/mnt/windowsE/chuanjun/building_wan`，可通过 `BUILDING_WAN_ROOT`、`WAN_PYTHON`、`WAN_DATASET_ROOT`、`WAN_METADATA` 和 `WAN_OUTPUT_ROOT` 覆盖路径。
- 默认训练配置保持384×384、81帧、LoRA rank 16、学习率2e-5、8 epochs；默认物理batch为2、梯度累积为2，有效batch为4。
- A6000脚本默认数据改为 `data/single_orbit_videos`，清单改为 `configs/single_orbit_same_prompt_v1/metadata_train.csv`，缓存改为 `cache/wan21_t2v_13b_single_orbit_same_prompt_384x384_f81_train`，输出改为 `checkpoints/building_wan_a6000_single_orbit`。
- 当前单轨试验先使用一条统一 prompt，固定划分为训练1992、验证234、测试118；划分清单和 manifest 位于 `configs/single_orbit_same_prompt_v1`。
- `wan_train_entry.py` 增加缓存样本的真实批处理：按样本拼接latent和text context，并为每个样本独立采样时间步、噪声和flow-matching权重。当前共享 context 适用于这轮统一 prompt；后续逐栋 prompt 必须改用新的按样本/按文本去重缓存并重建缓存。
- A6000物理batch和梯度累积可分别用 `WAN_TRAIN_BATCH_SIZE` 与 `WAN_GRADIENT_ACCUMULATION_STEPS` 调整；显存不足时首轮将物理batch降为1再保留累积步数。
- 尚未提交Slurm作业，尚未实测A6000显存峰值；正式提交前需要在服务器执行 `bash -n`、入口编译检查、缓存检查和A6000 GPU型号检查。
