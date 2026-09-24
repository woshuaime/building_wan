# Experiments

本文档记录各轮实验的配置、结果和失败原因。历史实验保留，不把不同 GPU、数据划分或输出目录混在一起。

## Experiment: building_train_1912_all_256_cooled_lora

- 时间：2026-09-04 至 2026-09-08
- 目的：在本地双 RTX 3060 上验证建筑 LoRA 训练链路。
- 配置：256×256、17 帧片段、LoRA rank 8、1 epoch、学习率 5e-5。
- 数据：1912 个源视频切成 9560 个片段。
- 结果：完成 4780 个 optimizer steps，末步 loss 约 0.02147。
- 结论：阶段视频显示视角覆盖不足、变化较少，不作为当前实验的初始化权重。
- 输出：checkpoints/building_train_1912_all_256_cooled_lora
- 处理：权重保留，不删除、不覆盖。

## Experiment: building_wan_a100_preparation_failures

- 时间：2026-09-13 至 2026-09-14
- 目的：验证 A100 环境并生成 384×384、81 帧缓存。
- 配置：单卡 A100、building_v1 train 1912 个样本。
- 失败原因：早期任务分别遇到 DiffSynth 导入路径、librosa 缺失，以及两个任务同时写同一缓存目录；最后在约 210/221 份时出现 torch.save iostream 错误。
- 处理：按用户要求删除旧版残缺缓存中的 222 个中间文件；原始数据、模型和历史 checkpoint 未删除。
- 结论：A100、CUDA 和 BF16 可用，但这些准备任务没有训练结果。

## Experiment: building_wan

- 时间：2026-09-14
- 目的：在单卡 A100 上训练一般建筑物的 81 帧环绕视频 LoRA。
- 配置：384×384、81 帧、1912 个样本、LoRA rank 16、学习率 2e-5、8 epochs、每 400 步保存。
- 结果：Job 1155 正常完成 15,296 个 optimizer steps，最终 loss 约 0.0207207，保留 step-400 至 step-15296 共 39 份 LoRA。
- 验证：已准备固定 prompt、seed 和阶段权重验证配置；当时尚未生成视频。
- 输出：checkpoints/building_wan
- 说明：这是历史 A100 结果，不是当前 A6000 续训来源。

## Experiment: building_wan_a6000_single_orbit

- 开始时间：2026-09-23
- 目的：在单张 RTX A6000 上先用单轨建筑视频和统一 prompt 训练；逐栋独立 prompt 留到后续。
- GPU：1× NVIDIA RTX A6000，约 47.4 GiB
- 环境：/home/shi/miniconda3/envs/scj
- 数据：data/single_orbit_videos
- 固定划分：configs/single_orbit_same_prompt_v1
- 样本数：训练 1992、验证 234、测试 118
- 配置：384×384、81 帧、物理 batch 2、梯度累积 2、有效 batch 4、LoRA rank 16、学习率 2e-5、8 epochs。
- Prompt：训练、验证和测试使用同一条统一建筑旋转 prompt；本轮不使用 Cap3D 逐栋描述。
- 缓存：compact-sft-shared-context-bf16-v2；每个样本保存视频 latent，共享文本 context 单独保存一份。
- 原输出：checkpoints/building_wan_a6000_single_orbit

### 当前状态

- 1992/1992 个缓存样本已完成。
- 原训练已进入正式训练阶段，并保存了 step-2400.safetensors。
- 训练约在第 2742 个 batch 处退出，日志为 logs/a6000_single_orbit_20260924_172511.log。
- 退出信息为 free(): invalid next size (normal)，不是 GPU OOM。
- 复查确认 step-2400.safetensors 可以被 safetensors.safe_open 正常读取。
- 训练脚本已将 dataset_num_workers 默认改为 0，降低多进程 native 内存释放导致崩溃的风险。
- 续训脚本已改为使用 --lora_checkpoint，避免把 LoRA 权重误当成完整训练状态。
- 续训输出将写入 checkpoints/building_wan_a6000_single_orbit_resumed。
- 续训和推理验证尚未重新启动。

### 续训和验证

正确续训：

~~~bash
cd /mnt/windowsE/chuanjun/building_wan
WAN_PYTHON=/home/shi/miniconda3/envs/scj/bin/python \
  bash scripts/resume_wan81_a6000.sh
~~~

续训完成后先预检：

~~~bash
/home/shi/miniconda3/envs/scj/bin/python scripts/validate_wan81_a6000.py \
  --checkpoint-dir checkpoints/building_wan_a6000_single_orbit_resumed
~~~

确认预检通过并明确要生成视频后，加上 --run。验证只做固定条件的 base/LoRA 对比，不自动计算量化指标。

## 实验纪律

- 不删除 cache、data、models、checkpoints 或 outputs。
- 不把续训写入原输出目录。
- 不同时启动两个写同一目录的进程。
- 没有视频验证前，不声称模型效果已经确认。
