# Project Rules

## 1. Project Overview

- 本项目使用建筑模型旋转视频对 Wan2.1-T2V-1.3B 进行领域 LoRA 训练，使模型学习一般建筑物的外观与完整环绕视角；当前阶段不是针对某一种特殊建筑做分类或结构分析。
- `code/Python_3D_Scanner` 包含三维模型采集、数据准备、实验管理、验证和项目训练入口。
- `code/DiffSynth-Studio` 是第三方 Wan 训练与推理引擎。项目逻辑优先放在 `Python_3D_Scanner`，除非没有可行替代，不直接改第三方核心代码。
- 当前训练流程是：固定数据划分 → 81帧视频预处理为“逐样本BF16视频latent + 单份共享BF16文本context”的紧凑缓存 → 单卡 A100 LoRA 训练 → 保存阶段权重 → 用固定提示词和种子生成对比视频。
- 核心技术栈为 Python、PyTorch、Accelerate、DiffSynth-Studio、Wan2.1、LoRA 和 Slurm。

## 2. Environment

- 服务器系统：Linux，主机 `workstation`，内核 `4.18.0-80.7.1.el8_0.x86_64`。
- Slurm GPU 分区：`gpu`；节点 `gpu1`；节点提供 2 张 A100。当前项目每次只申请 1 张 A100、启动 1 个训练进程。
- Conda 环境：`mv2v`。不要再使用旧的双卡环境 `mv2v_dual`。
- Python：`/home/u2025171963/miniconda3/envs/mv2v/bin/python`，版本 3.10.20。
- 已确认 PyTorch：2.10.0+cu128；PyTorch CUDA runtime：12.8。
- 已确认关键包：Accelerate 1.12.0、Transformers 4.57.0、Safetensors 0.7.0、librosa 0.11.0。
- 登录节点没有分配 CUDA，`torch.cuda.is_available()` 返回 False 属于正常现象。A100/CUDA 检查必须在 Slurm 作业内完成。
- 作业日志已确认 GPU 型号为 `NVIDIA A100 80GB PCIe`，可见显存约 79.3GiB。
- 服务器 NVML 初始化存在警告，可能影响 `nvidia-smi` 监控，但此前 Slurm 作业内 PyTorch CUDA 和 BF16 运算已确认可用。

## 3. Important Paths

- 项目根目录：`/home/share/CHUANJUN/building_wan`
- 项目代码：`/home/share/CHUANJUN/building_wan/code/Python_3D_Scanner`
- DiffSynth-Studio：`/home/share/CHUANJUN/building_wan/code/DiffSynth-Studio`
- 数据目录：`/home/share/CHUANJUN/building_wan/data/video_wan81`
- 固定训练清单：`/home/share/CHUANJUN/building_wan/data/splits/building_v1/metadata_train.csv`
- 当前缓存：`/home/share/CHUANJUN/building_wan/cache/wan21_t2v_13b_building_384x384_f81_train`
- 基础模型：`/home/share/CHUANJUN/building_wan/models`
- 项目训练入口：`/home/share/CHUANJUN/building_wan/code/Python_3D_Scanner/training/wan_train_entry.py`
- 启动脚本：`/home/share/CHUANJUN/building_wan/scripts`
- 当前 A100 输出：`/home/share/CHUANJUN/building_wan/checkpoints/building_wan`
- Slurm 日志：`/home/share/CHUANJUN/building_wan/logs/slurm`

## 4. Code Editing Rules

- 修改前先读取本文件、`PROJECT_CONTEXT.md`、`TRAINING.md` 和 `EXPERIMENTS.md`，并检查实际路径和当前任务状态。
- 项目脚本和 `code/Python_3D_Scanner` 可以按任务修改；修改 `code/DiffSynth-Studio` 前必须确认项目层无法解决。
- `models/`、`data/` 和已有 `checkpoints/` 默认只读，不得擅自修改、删除或覆盖。
- 不允许凭聊天记忆猜测服务器路径；必须在文件系统中核实。
- 大改前检查 Git 状态和 diff。当前服务器副本不是 Git worktree，应明确报告 Git 检查不可用，不能伪造分支或 commit。
- 不删除旧实验或旧 checkpoint。只有用户明确指定精确目标时才允许删除，并在删除后报告范围和可恢复性。
- 不覆盖已有实验输出。当前实验名由用户确认后使用，不要自动改名。

## 5. Training Rules

- 当前正式方案固定为单卡 A100、单训练进程；不使用不必要的 DDP 或双卡配置。
- 正式脚本使用 `mv2v` 环境的绝对 Python 路径，因此 Slurm 脚本无需交互式 `conda activate`。
- Slurm 脚本只负责资源申请和调用训练脚本；模型路径与训练参数放在独立训练脚本中。
- GPU 分区提交必须包含 `#SBATCH --comment=videoedit2025`。
- stdout 和 stderr 必须分别写入 `logs/slurm/%x-%j.out` 与 `%x-%j.err`。
- 一个缓存目录或一个输出目录同一时间只能有一个写入作业。提交前先检查 `squeue`，禁止重复提交同一实验。
- 新实验使用独立输出目录，不自动 resume 旧权重；只有用户明确要求时才允许 resume。
- 当前 A100 实验从 step 0、epoch 0 开始，不加载旧双3060 LoRA。
- 不运行 `sbatch`，除非用户明确说“开始提交”或同等明确指令。

## 6. Validation Rules

重要修改后至少执行：

- `bash -n` 检查所有相关 Shell 脚本。
- `python -m py_compile` 或等价导入检查。
- 检查数据、模型、tokenizer、入口和输出路径存在性。
- 核对训练参数真正传入下游入口，而不只是写在未读取的环境变量中。
- 检查缓存数量、格式标记和输出目录是否冲突。
- 查看 `squeue`，防止重复任务写同一目录。
- 如果是 Git 项目，检查 `git status` 和 `git diff`；如果不是，明确记录无法检查。
- 未完成上述检查时，不提交长时间训练任务。

## 7. Communication Rules

- 不得声称执行了实际未执行的命令、训练或验证。
- 不确定的信息标记为“待确认”，并说明验证方式。
- 不自动删除文件，不自动覆盖 checkpoint，不自动启动正式训练。
- 区分“缓存生成”“正式训练”和“推理验证”，汇报时明确当前所处阶段。
- 报错时先给出任务状态和直接失败原因，再区分无害警告与致命错误。

## 8. Context Maintenance

每轮重要工作结束前执行 handoff update：检查任务状态、实际改动和 Git 状态；更新 `PROJECT_CONTEXT.md`；正式实验状态变化时更新 `EXPERIMENTS.md`；运行方法变化时更新 `TRAINING.md`；只有长期规则变化时更新本文件。
