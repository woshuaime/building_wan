# Project Rules

## 1. 项目范围

本项目用 Wan2.1-T2V-1.3B 的 LoRA 学习灰色建筑 mesh 的外观、完整环绕视角和视频时序稳定性。当前主要实验是实验室服务器上的单轨统一 prompt 训练；逐栋独立 prompt 和 Cap3D 描述属于后续实验，不能混入本轮缓存。

目录职责：

- code/Python_3D_Scanner：数据划分、缓存、训练入口和验证入口。
- code/DiffSynth-Studio：第三方训练与推理引擎。项目层能解决的问题不要直接改第三方核心。
- scripts：训练、续训、验证和服务器启动脚本。
- configs：固定的数据划分和实验配置。

## 2. 当前服务器环境

当前 A6000 服务器项目根目录：

    /mnt/windowsE/chuanjun/building_wan

当前使用的 Conda 环境和 Python：

    /home/shi/miniconda3/envs/scj
    /home/shi/miniconda3/envs/scj/bin/python

已确认的运行条件：

- GPU：1 张 NVIDIA RTX A6000，约 47.4 GiB 显存。
- PyTorch：2.11.0+cu128。
- 训练使用单进程、单卡，不使用 DDP。
- A6000 训练脚本通过 WAN_PYTHON 覆盖 Python 路径；在服务器上应显式指定 scj，再执行脚本。

A100 的 mv2v 环境和 /home/share/CHUANJUN/building_wan 路径只属于历史实验，不能当作当前 A6000 实验的默认路径。

## 3. 当前实验配置

building_wan_a6000_single_orbit 使用：

- data/single_orbit_videos
- configs/single_orbit_same_prompt_v1
- 训练 1992、验证 234、测试 118
- 384×384、81 帧
- 物理 batch 2、梯度累积 2、有效 batch 4
- LoRA rank 16、学习率 2e-5、8 epochs
- dataset_num_workers=0
- 共享文本 context 的 compact-sft-shared-context-bf16-v2 缓存

本轮所有视频使用同一条 prompt，因此可以共用一份文本 context。以后使用逐栋 prompt 时，必须重新设计缓存并重新生成，不能直接复用本轮共享 context。

## 4. 文件和数据规则

- data/、models/、cache/、已有 checkpoints/ 和 outputs/ 默认只读，不擅自删除。
- 新实验必须使用独立输出目录，不能覆盖历史权重。
- 训练崩溃后先保留旧输出，再把续训结果写到独立目录。
- .safetensors LoRA 文件不能用 --resume_from_checkpoint 当作完整训练状态加载；续训必须使用 --lora_checkpoint。
- 同一缓存目录或输出目录不能被两个进程同时写入。
- 不把 Cap3D prompt、测试集或验证集混入本轮训练。

## 5. 训练和故障处理规则

- 未经用户明确要求，不执行 sbatch、不启动正式训练、不启动验证推理。
- A6000 原训练曾因 free(): invalid next size (normal) 在约 2742 个 batch 处退出。当前修复是单进程数据读取（worker 设为 0），并从已验证可读的 step-2400.safetensors 续训。
- 续训脚本是 scripts/resume_wan81_a6000.sh，默认输出到 checkpoints/building_wan_a6000_single_orbit_resumed，不能覆盖原目录。
- 续训前必须确认没有旧训练进程；推荐在 tmux 中运行，断开 SSH 后任务仍可继续。
- 黄色警告、NVML 提示和低 GPU 利用率不能直接当成训练失败；以 Python traceback、进程状态和输出文件为准。

## 6. 修改和验证

修改前：

1. 读取本文件、PROJECT_CONTEXT.md、TRAINING.md 和 EXPERIMENTS.md。
2. 检查 git status、当前分支和最近提交。
3. 核对实际服务器路径、Python 路径、缓存数量和输出目录。

修改后至少执行：

    git diff --check
    bash -n scripts/train_wan81_a6000.sh scripts/resume_wan81_a6000.sh
    /home/shi/miniconda3/envs/scj/bin/python -m py_compile code/Python_3D_Scanner/training/wan_train_entry.py scripts/validate_wan81_a6000.py
    git status --short

不得声称执行了没有执行的训练、验证或服务器命令。

## 7. Git 和交接

本地仓库是 D:\building_wan，远程仓库是 git@github.com:woshuaime/building_wan.git。提交文档或代码后，先检查 diff，再推送 main，最后在服务器执行 git pull --ff-only origin main。不要用强制推送覆盖他人的提交。

重要工作完成后同步更新：

- PROJECT_CONTEXT.md：当前真实状态和下一步。
- EXPERIMENTS.md：实验台账。
- TRAINING.md：可执行命令和排错方法。
- SESSION_HANDOFF.md：下一次会话需要知道的事实。

chuanjun.code-workspace 只描述本地 Windows 工作区，不写服务器 Linux 绝对路径；除非本地目录结构发生变化，否则保持不动。
