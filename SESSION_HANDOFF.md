# Session Handoff

## 1. 更新时间和当前任务

- 更新时间：2026-09-25
- 本地仓库：D:\building_wan
- 远程仓库：git@github.com:woshuaime/building_wan.git
- 服务器项目：/mnt/windowsE/chuanjun/building_wan
- 当前任务：恢复 A6000 单轨统一 prompt 训练，并在完成后生成固定条件对比视频

## 2. 环境

- Conda 环境：scj
- Python：/home/shi/miniconda3/envs/scj/bin/python
- PyTorch：2.11.0+cu128
- GPU：NVIDIA RTX A6000，约 47.4 GiB
- 训练方式：单卡、单进程，推荐放在 tmux 中运行

## 3. 当前实验

- 实验名：building_wan_a6000_single_orbit
- 数据：data/single_orbit_videos
- 固定划分：configs/single_orbit_same_prompt_v1
- 样本：训练 1992、验证 234、测试 118
- 统一 prompt：训练、验证、测试使用同一条建筑旋转 prompt
- 配置：384×384、81 帧、物理 batch 2、梯度累积 2、有效 batch 4、LoRA rank 16、8 epochs
- 缓存：cache/wan21_t2v_13b_single_orbit_same_prompt_384x384_f81_train
- 原输出：checkpoints/building_wan_a6000_single_orbit
- 续训输出：checkpoints/building_wan_a6000_single_orbit_resumed_exact

## 4. 已完成的工作

1. 建立本地 Git 仓库并与 GitHub main 同步。
2. 增加单轨统一 prompt 的固定训练、验证、测试划分。
3. 增加真实 batch 合并和每样本独立噪声/时间步处理。
4. 设置 A6000 默认物理 batch 2、梯度累积 2。
5. 使用共享文本 context 的紧凑缓存，缓存 1992 个样本已完成。
6. 增加 A6000 验证脚本和独立输出目录。
7. 原训练崩溃后把训练 worker 默认改为 0。
8. 增加跳过已完成 batch、保留全局 step 的续训脚本。
9. 修复 LoRA checkpoint 加载：续训使用 --lora_checkpoint。

最新已同步的代码提交为：

    8cfa348 Load LoRA checkpoints correctly when resuming

## 5. 原训练状态

原训练日志：

    logs/a6000_single_orbit_20260924_172511.log

训练约在第 2742 个 batch 处因以下错误退出：

    free(): invalid next size (normal)
    中止（核心转储）

没有观察到 GPU OOM。缓存已完整，step-2400.safetensors 已通过 safetensors 完整性检查。不要删除原缓存或原 checkpoint。

## 6. 下一步的正确命令

先确认没有旧训练进程，然后在 tmux 中执行：

    cd /mnt/windowsE/chuanjun/building_wan
    tmux new -s wan-resume
    WAN_PYTHON=/home/shi/miniconda3/envs/scj/bin/python bash scripts/resume_wan81_a6000.sh 2>&1 | tee logs/a6000_single_orbit_resume_$(date +%Y%m%d_%H%M%S).log

脚本默认从原目录的 step-2400.safetensors 读取 LoRA，按原物理 batch 2 换算当前 batch 1 的数据游标，并把新结果写到独立 resumed_exact 目录。不要改用 --resume_from_checkpoint，也不要把 resumed 输出写回原目录。

训练查看：

    watch -n 2 nvidia-smi
    tail -F logs/a6000_single_orbit_resume_*.log

## 7. 训练完成后的验证

先预检：

    /home/shi/miniconda3/envs/scj/bin/python scripts/validate_wan81_a6000.py --checkpoint-dir checkpoints/building_wan_a6000_single_orbit_resumed_exact

预检无误后，需要生成视频时再执行：

    /home/shi/miniconda3/envs/scj/bin/python scripts/validate_wan81_a6000.py --checkpoint-dir checkpoints/building_wan_a6000_single_orbit_resumed_exact --run

验证只比较基础模型与最新 LoRA 的固定 prompt、固定 seed 视频。需要人工检查建筑体块、360 度视角、几何变形、闪烁和时序稳定性。

## 8. 交接规则

- 不自动启动训练或验证。
- 不删除 cache、data、models、checkpoints、outputs。
- 不覆盖已有 checkpoint。
- 不并发写同一个缓存或输出目录。
- 不把 Cap3D 独立 prompt 混入本轮统一 prompt 训练。
- 修改后更新 PROJECT_CONTEXT.md、EXPERIMENTS.md、TRAINING.md 和本文件。
- chuanjun.code-workspace 只描述本地工作区，本轮无需改动。

## 2026-09-28：原生崩溃诊断续跑

resumed_exact 在日志 step 4225 后发生 free(): invalid next size (fast)；
batch 1 / accumulation 4 / worker 0 未消除故障，根因尚未定位。
step-4000.safetensors 的 600 个张量已加载并通过有限值检查，历史输出保留。
诊断从该权重开始：INITIAL_STEPS=4000，SKIP_BATCHES=6400
（前次跳过 4800，加上 checkpoint 保存前新增的 4000-2400）。
这是权重续训和样本预算换算，不恢复 optimizer/RNG，也不保证相同 shuffle 顺序。
WAN_DEBUG_GDB=1 启用 GDB 全线程原生堆栈、py-bt、共享库清单及
PYTHONFAULTHANDLER/TORCH_SHOW_CPP_STACKTRACES；默认不开启。
诊断保持原训练参数，不修改系统 core_pattern，不删除历史产物。
独立输出：checkpoints/building_wan_a6000_single_orbit_diag_20260928
独立日志：logs/a6000_diag_20260928.log
启动后需现场确认 loss 更新；本记录不代表崩溃已修复。