# Session Handoff

更新时间：2026-10-18。用户当前要求 BuildingFiller 源 465 数据交接与原始 VACE 最小补全实验准备；完成环境与数据检查后暂停，等待明确确认再启动 GPU 推理。历史训练继续授权仍不生效，不自动训练或补训。

## 优先接手：P01 / VACE 准备

- 先读 [P01 交接与 VACE 方案](docs/BUILDINGFILLER_P01_VACE_HANDOFF.md)。实际执行端是 SSH Linux `vccip`，用户 `scj`，不能把 Windows 主机的 D 盘或 C 盘 SSH 配置当成服务器文件。
- 服务器 `data/buildingfiller_import/P01_465/` 已有 ZIP、SHA256 和 `Stage1_Closeout_Report.md`。本轮仅读取原件，未执行上传；Windows 端源路径/SSH 客户端/别名检查命令已提供，尚未在 Windows 实测。
- ZIP SHA256 `5a966a269866a1423c1efa38937fcce6ec6984f8fda78c35395abccab83d5bb8`；262 项哈希、ZIP CRC、243 PNG 完整解码、81 帧对齐/相机绑定、三对首尾闭合及随包校验均 passed。Exact 046–072 共 27 张全黑，不能删除。
- 审计报告、联系图和独立解压副本在 `outputs/buildingfiller_handoff/P01_465_20261018/`；原数据不改。只检查了 P01 512 子集，不把来源记录的全部八例当成本机逐例检查。
- DiffSynth 2.1.5 VACE pipeline 离线导入通过，A6000 CUDA 可用、总显存 47.4 GiB。项目模型目录及当前用户缓存未找到 VACE；其他用户 `/home/shi/.cache` 无读取权限。现有 T2V DiT 不含 VACE 分支。
- 计划 `configs/vace_p01_465_17f_v1.json`：源 005–021 连续 17 帧、512×512、16 fps、原始 VACE-1.3B、不加载旧 LoRA。RGB Mask 列表/0–1 范围和填灰条件帧的 CPU 内存检查已通过；派生 Mask 与条件帧尚未落盘，没有加载权重、调用 VAE 或推理。
- 下一步先确认可读正式 VACE 权重路径，落实项目专用执行器和独立输入保存，再等待用户确认 GPU 推理。禁止自动下载/安装/升级或直接运行会下载模型与数据的 vendor 示例。GT 仅评估，Exact 明确是 GT-oracle；training_ready=false 保持。
- 新增只读审计工具 `scripts/audit_buildingfiller_package.py`，仅显式传 `--output-dir` 才写入全新审计目录，已有目录拒绝。未改第三方核心或 T2V 数据、缓存、历史权重/视频；未提交或同步远程。
- 本轮检查：git diff --check、训练/续训 bash -n、训练/验证/新审计入口 py_compile 通过，scripts/tests 36 项全部通过。真实包只读复验、错误预期 SHA256 拒绝、已有输出目录拒绝、冻结配置与 17 帧映射一致检查通过；已有审计报告哈希未变。PowerShell 仅审阅，未在 Windows 执行。

以下为 10-15 及更早 T2V 诊断交接，不能当作启动当前 VACE 或历史训练的授权。

## 历史 T2V 记录（原有 Git 版本）

以下保留本次同步前的 T2V 记录，其训练终态和操作描述不代表当前状态。当前目标、检查结果和运行授权以本文件开头的 P01 章节及其交接方案为准。此前 T2V 审计及训练代码修改仍留在服务器工作区，未纳入本次提交。

## 1. 更新时间和当前任务

- 更新时间：2026-09-28
- 本地仓库：D:\building_wan
- 远程仓库：git@github.com:woshuaime/building_wan.git
- 服务器项目：/mnt/windowsE/chuanjun/building_wan
- 当前任务：单轨 8 epochs 已跑完（最后权重 step-11200，checkpoints/building_wan_a6000_single_orbit_all/），下一步做固定条件 base/LoRA 对比视频并人工评估

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
- 原输出：checkpoints/building_wan_a6000_single_orbit_all/step-2400.safetensors
- 续训输出：统一放在 checkpoints/building_wan_a6000_single_orbit_all/ 下；2026-09-29 已把三段输出合并到这个父目录

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

    /home/shi/miniconda3/envs/scj/bin/python scripts/validate_wan81_a6000.py --checkpoint-dir checkpoints/building_wan_a6000_single_orbit_all

预检无误后，需要生成视频时再执行：

    /home/shi/miniconda3/envs/scj/bin/python scripts/validate_wan81_a6000.py --checkpoint-dir checkpoints/building_wan_a6000_single_orbit_all --run

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

## 9. 2026-09-28（续）：当前正在运行的训练、暂停续跑机制、目录清理

- diag_20260928 未产出可用权重；epoch4_resume_20260928_171131 在约 step 3927 处以退出码 134（SIGABRT）崩溃。
- epoch4_resume_20260928_174226 从同一更早权重重启，产出 step-2800/3200/3600 后被暂停脚本手动停止，未复现 free() 崩溃。
- 用该轮权重生成了对比视频，存于 outputs/validation/9.28/（A-step3200.mp4、A-step3600.mp4、B-step2400.mp4、B-step800.mp4、base.mp4）。
- 新增 scripts/pause_wan81_a6000.sh 和 scripts/resume_from_pause.sh：可在训练运行中记录暂停点（写入 logs/pause_state_*.json）并从该点续跑，不需要等崩溃。
- 用暂停记录以 epoch4_resume_20260928_174226/step-3600.safetensors 为 --lora_checkpoint 续跑，当前正在运行：checkpoints/building_wan_a6000_single_orbit_resumed_20260928_190750（pid 630785），日志 logs/a6000_resume_from_pause_20260928_191930.log，已产出 step-4000…6800。**交接时先确认该进程状态，不要重复启动第二个训练写同一目录。**
- 应用户要求清理了历史 checkpoints，删除 resumed、resumed_b1、resumed_exact、diag_20260928（空）、resumed_epoch4_20260928（空）、building_train_1912_all_256_cooled_lora（实际已不在服务器）共 6 项，不可恢复。第 3、6、7 节中提到的 resumed_exact 目录已不存在，续训和验证都不要再假设这个目录存在。
- 清理后 checkpoints/ 只剩 4 项：building_wan、building_wan_a6000_single_orbit、building_wan_a6000_single_orbit_resumed_epoch4_20260928_174226、building_wan_a6000_single_orbit_resumed_20260928_190750（当时）。

## 10. 2026-09-29：单轨 8 epochs 跑完、checkpoints 合并

- 训练于 2026-09-29 02:25 自然跑完 8 个 epochs，最后权重 step-11200，无崩溃。
- 单轨全部权重（400–2400、2800–3600、4000–11200 三段）已平铺到 checkpoints/building_wan_a6000_single_orbit_all/，原三个子目录已删除。平铺后该目录共 28 个权重（step-400 ~ 11200），上面的三段来源已无法从文件名区分。
- 三段元数据改名保留：loss_from_original_step400_2400.csv、loss_from_epoch4_step2800_3600.csv、loss_from_final_step4000_11200.csv，以及对应的 training_args_from_*.json / run_manifest_from_*.json。
- resume_wan81_a6000.sh、pause_wan81_a6000.sh、compare_lora_a6000.sh、validate_wan81_a6000.py 的默认 checkpoint 路径已同步为 building_wan_a6000_single_orbit_all/。
- 下一步：跑 base/LoRA 固定条件对比视频。推荐用平铺后的目录（自动取最新 step-11200）：

    /home/shi/miniconda3/envs/scj/bin/python scripts/validate_wan81_a6000.py \
      --checkpoint-dir checkpoints/building_wan_a6000_single_orbit_all --run
