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
- 输出目录：checkpoints/building_wan_a6000_single_orbit

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

    checkpoints/building_wan_a6000_single_orbit/step-2400.safetensors

默认跳过前 2400 个缓存 batch，初始全局 step 设为 2400，结果写入：

    checkpoints/building_wan_a6000_single_orbit_resumed

如果实际要从其他 step 续训，需要同时设置 WAN_RESUME_FROM_CHECKPOINT、WAN_RESUME_SKIP_BATCHES 和 WAN_RESUME_INITIAL_STEPS。

## 7. 验证方式

续训完成后先做只读预检：

~~~bash
cd /mnt/windowsE/chuanjun/building_wan
/home/shi/miniconda3/envs/scj/bin/python scripts/validate_wan81_a6000.py \
  --checkpoint-dir checkpoints/building_wan_a6000_single_orbit_resumed
~~~

确认预检通过后，用户明确要求生成视频时再加 --run：

~~~bash
/home/shi/miniconda3/envs/scj/bin/python scripts/validate_wan81_a6000.py \
  --checkpoint-dir checkpoints/building_wan_a6000_single_orbit_resumed --run
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

1. 在服务器确认没有残留训练进程和占用 GPU 的任务。
2. 在 tmux 中执行正确的续训命令。
3. 观察 resumed 输出目录中的 loss.csv 和 checkpoint。
4. 训练结束后先预检，再生成 base/LoRA 对比视频。
5. 观看视频后再决定是否进行逐栋 prompt 实验。
