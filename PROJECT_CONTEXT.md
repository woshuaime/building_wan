# Project Context

## 1. 当前目标

在实验室服务器上使用单张 NVIDIA RTX A6000 训练 Wan2.1-T2V-1.3B 建筑领域 LoRA。第一轮只使用单轨建筑视频和一条统一 prompt，让模型先学习灰色建筑 mesh 的体块、环绕视角和时序稳定性。逐栋独立 prompt 暂留给后续实验。

## 2. 当前真实状态

已完成：

- 本地仓库已建立，远程为 git@github.com:woshuaime/building_wan.git。
- 服务器项目已放在 /mnt/windowsE/chuanjun/building_wan。
- 单轨固定划分已生成：训练 1992、验证 234、测试 118。
- 统一 prompt 的 384×384、81 帧缓存已完整生成，缓存样本数为 1992。
- A6000 原训练已经进入正式训练阶段，并保存了 step-2400.safetensors。
- step-2400.safetensors 已通过 safetensors 完整性读取检查。
- 已定位续训时的 checkpoint 类型问题，并改用 --lora_checkpoint。

当前未完成：

- 原训练在约第 2742 个 batch 处退出，尚未完成 8 epochs。
- 尚未完成从 step-2400 开始的续训。
- 尚未生成 A6000 的 base/LoRA 对比视频，因此暂不能下视觉效果结论。

## 3. A6000 实验配置

- 实验名：building_wan_a6000_single_orbit
- GPU：1× NVIDIA RTX A6000，约 47.4 GiB
- 环境：/home/shi/miniconda3/envs/scj
- Python：/home/shi/miniconda3/envs/scj/bin/python
- PyTorch：2.11.0+cu128
- 数据：data/single_orbit_videos
- 划分：configs/single_orbit_same_prompt_v1
- 训练样本：1992
- 验证样本：234
- 测试样本：118
- 分辨率：384×384
- 帧数：81
- 物理 batch：2
- 梯度累积：2
- 有效 batch：4
- LoRA rank：16
- 学习率：2e-5
- 训练轮数：8
- 数据 worker：0
- 输出目录：checkpoints/building_wan_a6000_single_orbit_all（平铺后原 segment 目录已删除）

本轮共享文本 context 只适用于统一 prompt。以后改为逐栋 prompt 时，必须重建按 prompt 设计的缓存。

## 4. 服务器重要路径

项目根目录：

    /mnt/windowsE/chuanjun/building_wan

单轨数据和清单：

    data/single_orbit_videos
    configs/single_orbit_same_prompt_v1/metadata_train.csv
    configs/single_orbit_same_prompt_v1/metadata_val.csv
    configs/single_orbit_same_prompt_v1/metadata_test.csv

缓存：

    cache/wan21_t2v_13b_single_orbit_same_prompt_384x384_f81_train

模型：

    models/Wan2.1-T2V-1.3B
    models/Wan-Series-Converted-Safetensors

训练、续训和验证：

    scripts/train_wan81_a6000.sh
    scripts/resume_wan81_a6000.sh
    scripts/validate_wan81_a6000.py

原训练日志：

    logs/a6000_single_orbit_20260924_172511.log

## 5. 故障与修复

原训练退出信息：

    free(): invalid next size (normal)
    中止（核心转储）

这不是 GPU OOM。缓存已经完整，且 step-2400.safetensors 可读。最可能原因是训练阶段多进程数据读取和 native 内存释放冲突，因此 A6000 脚本已把 dataset_num_workers 默认改为 0。

之后又出现过：

    ValueError: Cannot load checkpoint ... 600 keys are unexpected

原因是把 LoRA 权重当成完整训练状态使用了 --resume_from_checkpoint。正确做法是使用 --lora_checkpoint；该修复已经提交并同步到代码。

## 6. 正确续训方式

不要删除原目录，也不要把续训结果写回原目录。当前默认续训命令：

~~~bash
cd /mnt/windowsE/chuanjun/building_wan
WAN_PYTHON=/home/shi/miniconda3/envs/scj/bin/python \
  bash scripts/resume_wan81_a6000.sh
~~~

默认读取：

    checkpoints/building_wan_a6000_single_orbit_all/step-2400.safetensors

默认按 checkpoint 原训练物理 batch 2 换算缓存游标：step-2400 在当前物理 batch 1 下跳过 4800 个 batch，初始全局 step 设为 2400，结果写入：

    checkpoints/building_wan_a6000_single_orbit_all/building_wan_a6000_single_orbit_resumed_exact

如果实际要从其他 step 续训，需要设置 WAN_RESUME_FROM_CHECKPOINT、WAN_RESUME_CHECKPOINT_STEP、WAN_RESUME_CHECKPOINT_BATCH_SIZE 和 WAN_RESUME_INITIAL_STEPS。脚本会按当前物理 batch 自动换算跳过的缓存 batch。

注意：单轨 8 epochs 已于 2026-09-29 跑完，权重平铺在 checkpoints/building_wan_a6000_single_orbit_all/ 下（step-400 ~ 11200 共 28 个）。续训前建议用 pause_wan81_a6000.sh 记录暂停点，再用 resume_from_pause.sh --start 启动，而不是直接跑本节的默认命令。

## 7. 验证方式

先做只读预检（默认目录已指向合并后的最新一轮）：

~~~bash
cd /mnt/windowsE/chuanjun/building_wan
/home/shi/miniconda3/envs/scj/bin/python scripts/validate_wan81_a6000.py \
  --checkpoint-dir checkpoints/building_wan_a6000_single_orbit_all
~~~

确认预检通过后，用户明确要求生成视频时再加 --run：

~~~bash
/home/shi/miniconda3/envs/scj/bin/python scripts/validate_wan81_a6000.py \
  --checkpoint-dir checkpoints/building_wan_a6000_single_orbit_all --run
~~~

验证只比较基础模型和最新 LoRA 的固定 prompt、固定 seed 视频，不计算验证集自动指标。需要人工观看建筑体块、360 度完整性、几何变形、闪烁和时序稳定性。

## 8. A100 历史实验

A100 实验 building_wan 已独立完成：384×384、81 帧、1912 个样本、8 epochs，15,296 个 optimizer steps，最终 loss 约 0.02072。它使用旧服务器路径 /home/share/CHUANJUN/building_wan 和 mv2v 环境，仅作为历史实验，不与当前 A6000 输出混用。

## 9. Git 与工作区

本地工作区：D:\building_wan。

本地文档或代码修改后：

~~~powershell
git diff --check
git status --short
git add <files>
git commit -m "描述本次修改"
git push origin main
~~~

服务器同步：

~~~powershell
ssh lab-codex "cd /mnt/windowsE/chuanjun/building_wan && git pull --ff-only origin main"
~~~

chuanjun.code-workspace 只描述本地 Windows 文件夹和 DiffSynth 子目录，没有加入服务器绝对路径；因此本轮保持不变。

## 10. 下一步

1. 单轨 8 epochs 已跑完（2026-09-29 02:25，最后权重 step-11200），无需再启动续训。
2. 生成 base/LoRA 固定条件对比视频，权重目录用合并后的 checkpoints/building_wan_a6000_single_orbit_all/。
3. 人工检查建筑体块、360 度视角完整性、几何变形、闪烁和时序稳定性。
4. 观看视频后再决定是否进行逐栋 prompt 实验。
5. 历史段（step-400 ~ 2400、2800 ~ 3600）如需对比也可一并生成。

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

## 11. 2026-09-28（续）：epoch4 续训、暂停续跑机制、目录清理

- diag_20260928 诊断跑未产出可用权重（目录已清空）。
- epoch4_resume_20260928_171131 在约 step 3927 处以退出码 134（SIGABRT）崩溃，日志 logs/a6000_epoch4_resume_20260928_171131.log。
- epoch4_resume_20260928_174226 从同一更早权重重启，产出 step-2800/3200/3600 后被暂停脚本手动停止，未复现 free() 崩溃，日志 logs/a6000_epoch4_resume_20260928_174226.log。
- 已用固定 prompt、固定 seed 生成 base 与 step-800/2400/3200/3600 的 LoRA 对比视频，保存在 outputs/validation/9.28/。
- 新增 scripts/pause_wan81_a6000.sh（记录运行中训练的最新 checkpoint 和数据游标到 logs/pause_state_*.json）和 scripts/resume_from_pause.sh（读取记录后按当前物理 batch 换算并重启训练）。
- 用暂停记录，以 epoch4_resume_20260928_174226/step-3600.safetensors 为 --lora_checkpoint 续跑，新输出目录 checkpoints/building_wan_a6000_single_orbit_resumed_20260928_190750，训练进程仍在运行（已产出 step-4000…6800）。
- 应用户要求清理历史 checkpoints，删除了 resumed、resumed_b1、resumed_exact、diag_20260928（空）、resumed_epoch4_20260928（空）、building_train_1912_all_256_cooled_lora（实际已不在服务器）共 6 项，不可恢复。
- 清理后 checkpoints/ 只剩 4 项：building_wan（多栋历史）、building_wan_a6000_single_orbit（单轨 400–2400）、resumed_epoch4_20260928_174226（2800–3600）、resumed_20260928_190750（当前正在跑）。
- 本文档第 6、7 节中出现的 resumed_exact 目录已不存在；实际续训应以最新的 pause/resume 流程为准，输出目录以每次暂停记录生成的新目录名为准，不再固定写死 resumed_exact。

## 12. 2026-09-29：单轨 8 epochs 跑完、checkpoints 平铺

- 训练于 2026-09-29 02:25 自然跑完 8 个 epochs，最后权重 step-11200，进程正常退出，无崩溃。
- 单轨三段输出（400–2400、2800–3600、4000–11200）已合并到 checkpoints/building_wan_a6000_single_orbit_all/，随后按用户要求把权重全部平铺到该目录下，三个子目录已删除。
- 现在该目录内是 28 个权重，step-400 ~ step-11200 连续，约 1.2G；checkpoints/ 顶层只有 building_wan（多栋历史）和 building_wan_a6000_single_orbit_all。
- 三段元数据改名保留，避免互相覆盖：loss_from_original_step400_2400.csv / loss_from_epoch4_step2800_3600.csv / loss_from_final_step4000_11200.csv，以及对应的 training_args_from_*.json 和 run_manifest_from_*.json。
- 平铺后目录里没有额外的分段信息，追溯某个 step 属于哪一段要看上面的元数据文件或日志时间戳。
- 脚本默认路径已同步：validate_wan81_a6000.py、compare_lora_a6000.sh 指向平铺后的 building_wan_a6000_single_orbit_all/；resume_wan81_a6000.sh、pause_wan81_a6000.sh 的历史默认输出仍带该前缀。
- 下一步：用最新 step-11200 或若干中间 step 做固定 prompt、固定 seed 的 base/LoRA 对比视频，人工检查体块、环绕完整性、几何变形、闪烁和时序稳定性。