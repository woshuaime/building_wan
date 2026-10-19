# Experiments

更新时间：2026-10-18。新增 BuildingFiller P01 数据接收检查和原始 VACE 基线计划；尚未执行 VACE 推理。历史串行测试已清理且不再自动续跑。各硬件、划分、权重来源分开记录。历史报告与本次核对结果不一致时，明确标记待核实，不补造训练结果。

## 当前准备：P01 源 465 / VACE-1.3B（2026-10-18）

- 来源为独立 BuildingFiller 项目，经实验室 Windows 主机 `D:\P01_465` 交接。服务器已有 `data/buildingfiller_import/P01_465/` ZIP、SHA256 及收尾说明；本轮没有执行上传。
- 源/家族 465，稳定 ID `b465_2e6fc746`，A1 Medium，删除表面积 5.679235%，人工接受；仅 P01 的 512 子集在本次服务器独立验证。ZIP 为 1,319,395 bytes，SHA256 `5a966a269866a1423c1efa38937fcce6ec6984f8fda78c35395abccab83d5bb8`。
- ZIP CRC、262 项文件 SHA256、243 PNG 实际解码、相机/元数据绑定、三角色 81 帧对应和三对闭合通过。随包校验器返回 passed。27 张全黑 Exact 合法保留；训练发布状态仍为 false。
- 审计报告/解压副本独立保存在 `outputs/buildingfiller_handoff/P01_465_20261018/`；冻结配置 `configs/vace_p01_465_17f_v1.json` 选择连续 005–021，17 帧、512×512、16 fps、seed 20261018、30 步、CFG 5，计划只用原始正式 VACE、不加载 T2V LoRA。
- CPU 检查了二值 Exact → RGB ModelMask 列表和 Damaged 白区填 RGB127 的实际预处理合同，张量 `[1,3,17,512,512]`、Mask 值域 0/1；没有落盘这些派生输入或加载权重。GT-oracle Mask 是本基线已知条件，GT RGB 仅供评估。
- CUDA/A6000 可用，DiffSynth 2.1.5 pipeline 离线导入通过；项目和可访问缓存未发现 VACE 权重，其他用户缓存不可读取。缺权重路径、项目基线执行器、派生输入及用户 GPU 确认，**没有生成补全结果或评价指标**。

操作与限制见 [P01 交接与 VACE 方案](docs/BUILDINGFILLER_P01_VACE_HANDOFF.md)。以下历史 T2V 台账保持独立。

## 历史 T2V 记录（原有 Git 版本）

以下保留本次同步前的 T2V 记录，其训练终态和操作描述不代表当前状态。当前目标、检查结果和运行授权以本文件开头的 P01 章节及其交接方案为准。此前 T2V 审计及训练代码修改仍留在服务器工作区，未纳入本次提交。

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
- 原输出：checkpoints/building_wan_a6000_single_orbit（2026-09-29 合并到 checkpoints/building_wan_a6000_single_orbit_all/ 下）

### 当前状态

- 1992/1992 个缓存样本已完成。
- 原训练已进入正式训练阶段，并保存了 step-2400.safetensors。
- 训练约在第 2742 个 batch 处退出，日志为 logs/a6000_single_orbit_20260924_172511.log。
- 退出信息为 free(): invalid next size (normal)，不是 GPU OOM。
- 复查确认 step-2400.safetensors 可以被 safetensors.safe_open 正常读取。
- 训练脚本已将 dataset_num_workers 默认改为 0，降低多进程 native 内存释放导致崩溃的风险。
- 续训脚本已改为使用 --lora_checkpoint，避免把 LoRA 权重误当成完整训练状态。
- 续训输出将写入独立目录 checkpoints/building_wan_a6000_single_orbit_resumed_exact；脚本会按 checkpoint 原 batch 自动换算数据游标，并写入 run_manifest.json。
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
  --checkpoint-dir checkpoints/building_wan_a6000_single_orbit_resumed_exact
~~~

确认预检通过并明确要生成视频后，加上 --run。验证只做固定条件的 base/LoRA 对比，不自动计算量化指标。

## 实验纪律

- 不删除 cache、data、models、checkpoints 或 outputs。
- 不把续训写入原输出目录。
- 不同时启动两个写同一目录的进程。
- 没有视频验证前，不声称模型效果已经确认。

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

## 2026-09-28（续）：暂停/续跑机制、epoch4 续训、目录清理

- 原生崩溃诊断跑（diag_20260928）未产出可用权重，目录内容已确认为空。
- 新增 checkpoint 类型续训 epoch4_resume_20260928_171131：从更早权重续训，在约 step 3927 处以退出码 134（SIGABRT）崩溃，日志 logs/a6000_epoch4_resume_20260928_171131.log 显示 GDB 捕获到原生 SIGABRT 信号。
- 新增续训 epoch4_resume_20260928_174226：从同一更早权重重启，产出 step-2800/3200/3600 后被暂停脚本手动停止，未复现 free() 崩溃。日志：logs/a6000_epoch4_resume_20260928_174226.log。
- 用该轮 step-3200/3600 权重与 step-2400/800/base 生成了固定 prompt、固定 seed 对比视频，保存在 outputs/validation/9.28/（A-step3200.mp4、A-step3600.mp4、B-step2400.mp4、B-step800.mp4、base.mp4）。
- 新增暂停/续跑脚本 scripts/pause_wan81_a6000.sh 和 scripts/resume_from_pause.sh：暂停脚本记录正在运行训练的最新 checkpoint 和游标到 logs/pause_state_*.json；续跑脚本读取该记录后重新计算 skip_train_batches / initial_train_steps 并重启训练，不恢复 optimizer/RNG。
- 用暂停记录以 epoch4_resume_20260928_174226 的 step-3600.safetensors 为 --lora_checkpoint 续跑，新输出目录 checkpoints/building_wan_a6000_single_orbit_resumed_20260928_190750，日志 logs/a6000_resume_from_pause_20260928_191930.log。该训练进程（pid 630785）仍在运行，已产出 step-4000/4400/4800/5200/5600/6000/6400/6800。
- 应用户要求清理历史 checkpoints 目录，删除以下 6 个（含 2 个空目录），原始 checkpoint 文件不可恢复：
  - checkpoints/building_wan_a6000_single_orbit_resumed
  - checkpoints/building_wan_a6000_single_orbit_resumed_b1
  - checkpoints/building_wan_a6000_single_orbit_resumed_exact
  - checkpoints/building_wan_a6000_single_orbit_diag_20260928（空）
  - checkpoints/building_wan_a6000_single_orbit_resumed_epoch4_20260928（空，注意和保留的 `_174226` 不是同一目录）
  - checkpoints/building_train_1912_all_256_cooled_lora（本地 3060 早期验证跑，实际已不在服务器上）
- 清理后 checkpoints/ 只保留 4 项：building_wan（A100 多栋历史结果）、building_wan_a6000_single_orbit（单轨 400–2400）、building_wan_a6000_single_orbit_resumed_epoch4_20260928_174226（2800–3600）、building_wan_a6000_single_orbit_resumed_20260928_190750（当时正在跑）。
- 本文档和 PROJECT_CONTEXT.md、TRAINING.md、SESSION_HANDOFF.md 中此前提到的 resumed_exact 已不存在；续训输出目录以每次暂停记录生成的新目录名为准，不再固定写死 resumed_exact。以下路径在 2026-09-29 的目录合并后统一加了 building_wan_a6000_single_orbit_all/ 前缀。

## 2026-09-29：训练跑完 8 epochs、checkpoints 合并并平铺

- 训练于 2026-09-29 02:25 自然跑完 8 个 epochs，最后一个权重 step-11200，进程正常退出，无崩溃。
- 单轨三段的输出先合并到 checkpoints/building_wan_a6000_single_orbit_all/，随后应用户要求把全部权重平铺到该目录下，三个子目录已删除。
- 平铺后目录内共 28 个权重，step-400 ~ step-11200 连续无缺口，每 400 步一个，约 1.2G。
- 三段各自的元数据改名保留，避免 loss.csv / training_args.json 互相覆盖：
  - loss_from_original_step400_2400.csv、training_args_from_original_step400_2400.json
  - loss_from_epoch4_step2800_3600.csv、training_args_from_epoch4_step2800_3600.json、run_manifest_from_epoch4_step2800_3600.json
  - loss_from_final_step4000_11200.csv、training_args_from_final_step4000_11200.json、run_manifest_from_final_step4000_11200.json
- 注意：权重平铺后没有文件能单独标识「某个 step 属于哪一段」，追溯来源只能靠上面的元数据和 log 时间戳。
- 路径变更同步到脚本：validate_wan81_a6000.py 和 compare_lora_a6000.sh 的默认 checkpoint 目录都指向平铺后的 building_wan_a6000_single_orbit_all/；resume_wan81_a6000.sh 和 pause_wan81_a6000.sh 的历史默认输出路径仍带 building_wan_a6000_single_orbit_all/ 前缀。
- 单轨 8 epochs 已跑完，续训脚本的历史默认目标 resumed_exact 依然不存在，不要再按旧路径直接启动续训。