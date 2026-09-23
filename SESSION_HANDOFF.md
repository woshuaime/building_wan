# Session Handoff

## 1. Handoff Snapshot

- Updated：2026-09-16T01:35:05+08:00
- Workspace：`/home/share/CHUANJUN/building_wan`
- User/host：`u2025171963@workstation`
- Active Slurm jobs：0
- Current cache：1912/1912，约5.6GiB，完成标记存在
- Current A100 checkpoints：39份，step-400至step-15296
- Training state：Job 1155已完成，推理验证配置已准备但尚未提交

## 2. Work Actually Completed This Session

1. 核实学校服务器的项目、数据、模型、训练入口、Conda环境、Slurm资源和日志路径。
2. 将正式方案从旧双RTX 3060切换为单张A100 80GB、单训练进程。
3. 将Slurm职责收敛为资源申请和调用独立训练脚本；训练参数集中到 `scripts/train_wan81.sh`。
4. 将任务名、日志前缀和当前A100实验输出名称统一为纯文本 `building_wan`。
5. 修复并验证当前环境依赖：使用 `mv2v`，`librosa` 已可正常导入。
6. 审计旧缓存：发现单份约97MB，其中约71MB是重复保存的81帧原始画面，完整缓存预计接近170GB。
7. 实现 `compact-sft-bf16-v1`：只保存正式SFT训练必需的 `input_latents`、正向 `context` 和必要标量，并按训练实际使用的BF16落盘。
8. 用旧样本验证紧凑缓存：单份由97,074,029字节降到5,744,709字节，减少94.08%；预计1912份约10.23GiB；进入训练前的BF16张量逐元素一致。
9. 按用户明确要求删除旧版残缺的222个 `.pth` 缓存，释放约20GB；未删除原始视频、基础模型、日志或历史checkpoint。
10. 建立跨Codex会话上下文机制：`AGENTS.md`、`PROJECT_CONTEXT.md`、`EXPERIMENTS.md`、`TRAINING.md` 和本文件。
11. 将缓存进一步升级为 `compact-sft-shared-context-bf16-v2`：1912个样本只存各自BF16视频latent，相同prompt的BF16文本context只存一份；模拟完整缓存由10.229GiB降到2.764GiB，再减少72.98%。
12. 用户提交Job 1155后确认单张A100、CUDA和缓存生成正常；共享context已正确去重。实测Accelerate把返回的latent回转为FP32，完整缓存预计约5.52GiB，但训练时仍转BF16。
13. 修复`start_wan81.sh`：可传Job ID重新连接，且`squeue`临时超时不会终止日志显示；对Job 1155短时连接测试通过且未影响任务。
14. Job 1155完成15,296步正式训练，最终状态`COMPLETED / 0:0`，最终loss 0.0207207。
15. 准备A100固定条件验证：base、step-4000、step-8000、step-12000和step-15296；新增本地模型路径支持、验证配置和Slurm脚本，dry-run通过。

## 3. Files Modified or Created

- `scripts/submit_wan81.sh`
  - 单卡A100 Slurm资源配置；job name为 `building_wan`；stdout/stderr写入独立Slurm日志。
- `scripts/train_wan81.sh`
  - 保存全部正式训练参数、实际模型与数据路径、路径/GPU/缓存/输出冲突检查。
  - 缓存阶段调用共享文本context紧凑缓存模式，并使用 `.compact-sft-shared-context-bf16-v2` 格式标记。
- `scripts/start_wan81.sh`
  - 捕获真实Job ID并持续显示stdout/stderr；支持传Job ID重新连接；Slurm查询临时超时时保持跟踪。
- `code/Python_3D_Scanner/training/wan_train_entry.py`
  - 支持单GPU单进程直接运行。
  - `--compact-sft-cache`在数据处理时裁剪无用字段，每个样本不再重复保存context。
  - 新增`--shared-text-context-path`，原子保存单份共享context，并在训练加载时恢复原输入结构；多prompt数据会被拒绝。
- `AGENTS.md`
  - 长期项目、环境、路径、编辑、训练、验证和沟通规则。
- `PROJECT_CONTEXT.md`
  - 当前阶段状态、确认配置、决策、问题、失败方案和下一步。
- `EXPERIMENTS.md`
  - 旧双3060实验、A100准备失败记录及待开始的 `building_wan` 实验。
- `TRAINING.md`
  - 当前服务器环境检查、Slurm提交、日志、取消与故障排查说明。
- `SESSION_HANDOFF.md`
  - 本轮实际工作的浓缩交接记录。

## 4. Deleted Material

- 精确删除目标：`cache/wan21_t2v_13b_building_384x384_f81_train` 下旧版残缺的222个 `.pth` 文件。
- 删除后：0个 `.pth`，目录约16KB。
- 可恢复性：不可恢复；这些文件是失败任务生成的中间缓存，不是原始数据或模型权重。
- 其余旧实验和checkpoint均保留。

## 5. Confirmed Current Training Configuration

- Model：Wan2.1-T2V-1.3B
- Dataset：`building_v1` train，1912个视频
- Resolution/frames：384×384，81帧
- GPU/process：1×A100 80GB，1个进程
- LoRA：DiT，targets `q,k,v,o,ffn.0,ffn.2`，rank 16
- Optimizer settings：learning rate 2e-5，weight decay 0.01
- Schedule：8 epochs，save every 400 steps，gradient accumulation 1
- Memory：gradient checkpointing启用；训练阶段offload T5和VAE
- Cache：逐样本BF16视频latent + 单份共享BF16文本context，格式 `compact-sft-shared-context-bf16-v2`
- Start：step 0 / epoch 0，不resume旧双3060权重
- Output：`/home/share/CHUANJUN/building_wan/checkpoints/building_wan`

## 6. Validation Already Completed

- 三个Shell脚本通过 `bash -n`。
- `wan_train_entry.py` 通过Python编译检查。
- `tests.test_training_tools` 共12项测试通过。
- 当前 `mv2v` 环境执行 `pip check` 无损坏依赖。
- DiffSynth、librosa和主要训练依赖已成功导入。
- v2紧凑缓存通过共享context保存/恢复、不同context拒绝、结构及序列化大小检查；模拟1912份约2.764GiB。
- A100缓存和正式训练已端到端完成；尚未生成阶段对比视频。

## 7. Current Problems and Risks

- 正式训练已完成，当前没有活动作业。
- 实际样本latent以FP32落盘，预计完整缓存约5.52GiB；这是存储优化未完全达到预期，不影响训练时的BF16输入。
- 当前A100输出目录没有checkpoint，尚无新模型效果可评估。
- NVML初始化警告仍可能出现，主要影响监控；应以PyTorch CUDA检测和实际计算为准。
- Linux内核4.18低于Accelerate建议的5.5，存在潜在挂起风险。
- 无 `quota`/`lfs` 命令，个人存储配额仍待学校平台侧确认。
- 当前服务器项目不是Git worktree，无法提供可信分支、commit或diff。
- Job 1149和1150曾并发写同一缓存并失败；以后同一实验绝不能重复提交。

## 8. Next Steps

1. 下一次会话先阅读 `AGENTS.md`、`PROJECT_CONTEXT.md`、`SESSION_HANDOFF.md`、`TRAINING.md` 和 `EXPERIMENTS.md`。
2. 只读检查workspace、Slurm队列、缓存数量、输出目录和最近日志。
3. 用户明确要求开始提交验证后，提交一个`building_wan_val`作业。
4. 检查五个固定条件视频及元数据，生成contact sheet和视频诊断。
5. 评价完整旋转、建筑外观、结构稳定、时序一致性和阶段过拟合。
6. 验证状态变化后更新 `PROJECT_CONTEXT.md`、`SESSION_HANDOFF.md` 和 `EXPERIMENTS.md`。

## 9. Guardrails for the Next Session

- 不自动执行 `sbatch`，除非用户明确授权开始提交。
- 不自动删除缓存、原始数据、模型或checkpoint。
- 不使用旧 `mv2v_dual` 环境，不启动双卡训练。
- 不resume旧双3060权重。
- 不同时启动两个写同一缓存或输出目录的任务。
- 不把NVML警告误判为CUDA训练必然失败。
- 不在阶段对比视频生成前声称模型视觉效果已经确认。

## 10. This Handoff Update

- 已核实Job 1155完整成功、checkpoint和loss记录齐全。
- 已准备并验证推理配置及脚本，尚未执行`sbatch`。
- 已生成loss曲线；视觉效果仍需A100推理视频确认。
- 项目仍不是Git worktree，无法检查branch、commit或diff。
