# Training

更新时间：2026-10-18。当前仅完成 BuildingFiller P01 交接检查与 VACE 基线准备，不启动训练或 GPU 推理。当前 A6000 项目路径为 `/mnt/windowsE/chuanjun/building_wan`。历史串行测试已按要求清理。本文件中的启动命令仅供用户再次明确授权训练/推理后使用；代码维护本身不授权启动。

## 当前阶段：BuildingFiller / VACE，只读检查

Windows PowerShell 完整复制命令及三机路径边界见 [P01 交接](docs/BUILDINGFILLER_P01_VACE_HANDOFF.md)。Linux 已有同包并验证通过，当前无需再次上传。新工具默认不写数据或输出：

```bash
cd /mnt/windowsE/chuanjun/building_wan
/home/shi/miniconda3/envs/scj/bin/python -B scripts/audit_buildingfiller_package.py \
  data/buildingfiller_import/P01_465/BuildingFiller_P01_465_512_v1.zip \
  --expected-sha256 5a966a269866a1423c1efa38937fcce6ec6984f8fda78c35395abccab83d5bb8
```

可加 `--output-dir <全新审计目录>` 保存报告和解压副本，已有目录拒绝。实际报告位于 `outputs/buildingfiller_handoff/P01_465_20261018/`。17 帧冻结参数在 `configs/vace_p01_465_17f_v1.json`；VACE 权重和项目专用执行器尚待落实，不直接运行会下载模型/示例数据的第三方示例。GPU 推理等待用户明确确认。下文是历史 T2V 入口，不用于 P01 条件补全。

## 历史 T2V 记录（原有 Git 版本）

以下保留本次同步前的 T2V 记录，其训练终态和操作描述不代表当前状态。当前目标、检查结果和运行授权以本文件开头的 P01 章节及其交接方案为准。此前 T2V 审计及训练代码修改仍留在服务器工作区，未纳入本次提交。

本文档记录当前实验室服务器上的 A6000 单轨训练、续训和验证方法。当前项目根目录为 /mnt/windowsE/chuanjun/building_wan，环境为 scj。

## 1. 环境和路径

服务器 Python：

    /home/shi/miniconda3/envs/scj/bin/python

检查环境：

    /home/shi/miniconda3/envs/scj/bin/python --version
    /home/shi/miniconda3/envs/scj/bin/python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
    /home/shi/miniconda3/envs/scj/bin/python -m pip check

当前 GPU 应为 1 张 NVIDIA RTX A6000。登录节点没有 GPU 时，torch.cuda.is_available() 为 False 不代表训练节点不可用。

## 2. A6000 单轨配置

训练脚本：

    scripts/train_wan81_a6000.sh

续训脚本：

    scripts/resume_wan81_a6000.sh

验证脚本：

    scripts/validate_wan81_a6000.py

默认参数：

    数据：data/single_orbit_videos
    清单：configs/single_orbit_same_prompt_v1/metadata_train.csv
    划分：训练1992、验证234、测试118
    分辨率：384×384
    帧数：81
    物理 batch：2
    梯度累积：2
    有效 batch：4
    LoRA rank：16
    训练轮数：8
    dataset_num_workers：0
    原输出：checkpoints/building_wan_a6000_single_orbit_all（step-400 ~ 11200 平铺在同一层）

统一 prompt 训练会使用一份共享文本 context。逐栋 prompt 必须使用新的缓存设计，不能复用本轮缓存。

## 3. 只读检查

    cd /mnt/windowsE/chuanjun/building_wan
    bash -n scripts/train_wan81_a6000.sh scripts/resume_wan81_a6000.sh
    /home/shi/miniconda3/envs/scj/bin/python -m py_compile code/Python_3D_Scanner/training/wan_train_entry.py scripts/validate_wan81_a6000.py
    find cache/wan21_t2v_13b_single_orbit_same_prompt_384x384_f81_train -type f -name '*.pth' | wc -l
    ps -ef | grep wan_train_entry.py | grep -v grep || true

缓存完整时应为 1992 个 .pth，并同时存在共享文本 context 文件。

## 4. 原训练故障

原训练曾在约第 2742 个 batch 退出：

    free(): invalid next size (normal)
    中止（核心转储）

这不是显存不足。完整缓存仍可复用，step-2400.safetensors 已确认可读。最可能原因是多进程数据读取触发 native 内存释放问题，因此当前训练和续训都默认 dataset_num_workers=0。

曾出现的 checkpoint 报错：

    600 keys are unexpected

原因是把 LoRA 权重传给了 --resume_from_checkpoint。LoRA 续训必须使用 --lora_checkpoint；--resume_from_checkpoint 只适合完整训练状态。

## 5. 正确续训

不要删除或覆盖原输出。建议先进入 tmux：

    tmux new -s wan-resume
    cd /mnt/windowsE/chuanjun/building_wan
    WAN_PYTHON=/home/shi/miniconda3/envs/scj/bin/python bash scripts/resume_wan81_a6000.sh 2>&1 | tee logs/a6000_single_orbit_resume_$(date +%Y%m%d_%H%M%S).log

脚本默认：

- 从 checkpoints/building_wan_a6000_single_orbit_all/step-2400.safetensors 读取 LoRA。
- 根据 checkpoint 产生时的物理 batch 2，自动换算当前物理 batch 1 的缓存游标（默认跳过 4800 个 batch）。
- 把初始全局 step 设为 2400。
- 默认物理 batch 1、梯度累积 4、worker 0，有效 batch 仍为 4。
- 把新权重写入独立的 checkpoints/building_wan_a6000_single_orbit_all/building_wan_a6000_single_orbit_resumed_exact，目录非空时拒绝启动。
- 在输出目录写入 run_manifest.json，并用目录锁防止重复启动。

如果使用其他 checkpoint，必须同时设置：

    WAN_RESUME_FROM_CHECKPOINT=/path/to/step-N.safetensors WAN_RESUME_CHECKPOINT_STEP=N WAN_RESUME_CHECKPOINT_BATCH_SIZE=2 WAN_RESUME_INITIAL_STEPS=N WAN_PYTHON=/home/shi/miniconda3/envs/scj/bin/python bash scripts/resume_wan81_a6000.sh

查看训练：

    tmux attach -t wan-resume
    tail -F logs/a6000_single_orbit_resume_*.log
    watch -n 2 nvidia-smi

断开 tmux 使用 Ctrl+B，再按 D；不要用 Ctrl+C，除非要停止训练。

## 6. 续训完成后的验证

先只读预检：

    cd /mnt/windowsE/chuanjun/building_wan
    /home/shi/miniconda3/envs/scj/bin/python scripts/validate_wan81_a6000.py --checkpoint-dir checkpoints/building_wan_a6000_single_orbit_all

预检通过且确认要生成视频后：

    /home/shi/miniconda3/envs/scj/bin/python scripts/validate_wan81_a6000.py --checkpoint-dir checkpoints/building_wan_a6000_single_orbit_all --run

结果目录类似：

    outputs/validation/building_wan_a6000_single_orbit_resumed/step-N

脚本比较基础模型和最新 LoRA 的固定 prompt、seed、384×384、81 帧视频。验证集用于路径和 prompt 一致性检查，不会自动给出建筑质量分数；需要人工检查体块、环绕完整性、几何变形、闪烁和时序稳定性。

## 7. 状态和排错

    ps -ef | grep wan_train_entry.py | grep -v grep
    du -sh cache/wan21_t2v_13b_single_orbit_same_prompt_384x384_f81_train
    find checkpoints/building_wan_a6000_single_orbit_all -maxdepth 2 -name '*.safetensors' | sort -V | tail
    tail -n 100 logs/a6000_single_orbit_resume_*.log

NVML 初始化警告、黄色感叹号和短时低 GPU 利用率不一定是错误；优先检查 traceback、进程是否仍在、loss.csv 是否增长和 checkpoint 是否保存。不要在原因未确认时重复启动第二个训练进程。

## 8. A100 历史命令
A100 的 /home/share/CHUANJUN/building_wan 和 mv2v 流程已完成并保留在历史文档中。当前 A6000 训练不要使用 A100 的脚本、环境或输出目录。

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

## 9. 2026-09-28（续）：暂停/续跑脚本和当前运行状态

diag_20260928 未产出可用权重（目录已清空并删除）。epoch4_resume_20260928_171131 在约 step 3927 处以退出码 134（SIGABRT）崩溃。epoch4_resume_20260928_174226 从同一更早权重重启，产出 step-2800/3200/3600 后被暂停脚本手动停止，未复现崩溃，并已生成对比视频 outputs/validation/9.28/。

新增两个脚本，用于安全暂停正在跑的训练并从暂停点续跑，不需要等到崩溃或跑满：

    bash scripts/pause_wan81_a6000.sh            # 只记录当前最新 checkpoint 和游标，训练继续跑
    bash scripts/pause_wan81_a6000.sh --stop      # 记录后停止训练进程
    bash scripts/resume_from_pause.sh             # 只打印将要执行的续跑命令
    bash scripts/resume_from_pause.sh --start     # 实际重启训练

`resume_from_pause.sh` 默认读取 logs/ 下最新的 pause_state_*.json，也可用 `--state` 指定。重启前会检查是否已有 wan_train_entry.py 进程在跑，避免重复写同一目录。

该轮已于 2026-09-29 02:25 自然跑完 8 个 epochs（最后权重 step-11200，无崩溃）。输出目录为 checkpoints/building_wan_a6000_single_orbit_all（--lora_checkpoint 取自 epoch4_resume_20260928_174226/step-3600.safetensors），日志 logs/a6000_resume_from_pause_20260928_191930.log。

本文档第 5、6 节中提到的默认续训目标 resumed_exact 已被删除，不再是当前有效的默认输出目录；请按 pause/resume 流程生成的新目录名核对，不要凭旧文档假设目录仍存在。

checkpoints/ 已按用户要求合并为两项：building_wan（多栋历史结果）和 building_wan_a6000_single_orbit_all（单轨全部三段，step-400 ~ 11200，共 28 个权重）。resumed、resumed_b1、resumed_exact、diag_20260928、resumed_epoch4_20260928（空）、building_train_1912_all_256_cooled_lora（实际已不在服务器）已在 2026-09-28 按用户要求删除，不可恢复。
## 10. 2026-09-29：checkpoints 平铺后的路径

单轨全部权重已平铺到 checkpoints/building_wan_a6000_single_orbit_all/，step-400 ~ step-11200 共 28 个，直接放在该目录下，没有子目录。脚本默认路径已同步。

检查全部权重：

    find checkpoints/building_wan_a6000_single_orbit_all -maxdepth 1 -name '*.safetensors' | sort -V

验证（默认目录已指向平铺后的目录，会自动选最新 step-11200）：

    /home/shi/miniconda3/envs/scj/bin/python scripts/validate_wan81_a6000.py --checkpoint-dir checkpoints/building_wan_a6000_single_orbit_all

    /home/shi/miniconda3/envs/scj/bin/python scripts/validate_wan81_a6000.py --checkpoint-dir checkpoints/building_wan_a6000_single_orbit_all --run

训练已跑完 8 个 epochs，当前无需续训；若以后要续训，先用 pause_wan81_a6000.sh 记录，再用 resume_from_pause.sh --start，不要直接跑第 5 节的默认命令（历史默认目标 resumed_exact 不存在）。

分段元数据：平铺后三段各自的 loss 和参数记录改名为 loss_from_original_step400_2400.csv、loss_from_epoch4_step2800_3600.csv、loss_from_final_step4000_11200.csv，以及对应的 training_args_from_*.json 和 run_manifest_from_*.json。
