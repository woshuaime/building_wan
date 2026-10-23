# Project Rules

更新：2026-10-23（UTC+09:00）。当前事实见 [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)，文档清单见 [docs/DOCUMENTATION_INDEX.md](docs/DOCUMENTATION_INDEX.md)。以下工作规则继续有效；历史实验参数仅适用于其对应run。

## 1. 项目范围

当前项目研究BuildingFiller残缺建筑序列的VACE条件补全。八例基线、CPU缺口回贴、P04/P06两例17帧LoRA及八例81帧扩展评估均已完成；最新Complex12 Rev1也已完成C/U/S各300更新及五方法60条81帧轨迹。接手先从experiments/README.md查看现有run和成果，不自动另起GPU任务。

历史Wan2.1-T2V-1.3B LoRA用于灰色建筑外观、环绕视角和视频时序学习；单轨统一prompt、逐栋prompt和Cap3D的划分及缓存规则仍适用于各自历史实验，不能混入VACE条件补全实验。

目录职责：

- experiments：每个研究想法独立目录，放专用代码、配置入口、结果链接和说明；新运行按docs/PROJECT_STRUCTURE.md组织。
- code/Python_3D_Scanner：数据划分、缓存、训练入口和验证入口。
- code/DiffSynth-Studio：第三方训练与推理引擎。项目层能解决的问题不要直接改第三方核心。
- scripts：已有公共适配器、兼容启动入口、训练/验证工具和CPU回归。Complex12专用源码位于experiments/vace/complex12_rev1/scripts。
- configs：固定的数据划分和实验配置。

## 2. 当前服务器环境

当前 A6000 服务器项目根目录：

    /mnt/windowsE/chuanjun/building_wan

当前使用的 Conda 环境和 Python：

    /home/shi/miniconda3/envs/scj
    /home/shi/miniconda3/envs/scj/bin/python

已确认的运行条件：

- GPU：1张NVIDIA RTX A6000，标称48GB；历史监控显示约47.99GiB总显存，实际空闲量以运行时查询为准。
- PyTorch：2.11.0+cu128。
- 训练使用单进程、单卡，不使用 DDP。
- A6000 训练脚本通过 WAN_PYTHON 覆盖 Python 路径；在服务器上应显式指定 scj，再执行脚本。

A100 的 mv2v 环境和 /home/share/CHUANJUN/building_wan 路径只属于历史实验，不能当作当前 A6000 实验的默认路径。

## 3. 当前VACE实验与历史T2V配置

已完成的两例VACE smoke配置是 `configs/vace_roof_smoke_v1.json`，详细说明见 `docs/VACE_P04_P06_LORA_SMOKE.md`。以下预算只属于该历史实验；Complex12使用自己run中的冻结配置，不能混用：

- 仅P04/source1227和P06/source1848，各固定B000–B016，512×512，GT/Damaged/Exact同帧；B080不参与选窗。
- GT用于官方FlowMatch训练监督；Damaged填灰127与二值ModelMask用于VACE条件。推理不读取GT，不使用GT首尾帧或视频初始化。
- 新建vace分支LoRA，rank=alpha=16，目标q/k/v/o/ffn.0/ffn.2；只优化A/B参数，冻结DiT仍保留反向传播。
- 恒定5e-5 AdamW、wd0、batch/accum1、clip1、workers0、checkpointing、seed20261018；300次实际更新，两例交替各150，仅保存100/300。BF16骨干/autocast和FP32 LoRA参数。
- 顺序为两例17帧Base、300更新训练、两例LoRA-100、两例LoRA-300、CPU制图；最多6次17帧推理，各阶段fresh进程，LoRA只加载到vace、倍率1，不累加融合。
- 原test/composite_check及数据、模型保持只读；输出在独立 `outputs/vace/lora_roof_smoke/run_<时间>/`，只保存指定权重、配置、loss与两例各四项可视化。
- `run_vace_roof_smoke.sh` 无参数或 `--dry-run` 只做CPU准备；用户执行 `--run` 或 `--tmux` 才启动GPU。首轮已有完整结果，不得将启动命令当作日常状态查询或自动再跑一轮。
- OOM、非有限值、断梯度或进程异常立即停止，不自动扩预算、降规格、重试、续训或重跑81帧。此LoRA学过P04/P06，不能冒充无泄漏初始化或泛化实验。

历史 `building_wan_a6000_single_orbit` 使用：

- data/single_orbit_videos
- configs/single_orbit_same_prompt_v1
- 训练 1992、验证 234、测试 118
- 384×384、81 帧
- 原训练物理 batch 2、梯度累积 2；当前脚本和续训默认 batch 1、累积 4，有效 batch 均为 4
- LoRA rank 16、学习率 2e-5、8 epochs
- dataset_num_workers=0
- 共享文本 context 的 compact-sft-shared-context-bf16-v2 缓存；格式名不保证 latent 的落盘 dtype，训练时转为 BF16

该历史T2V实验所有视频使用同一条prompt，因此可以共用一份文本context。以后使用逐栋prompt时必须重新设计并生成缓存；当前VACE任务不使用这份旧T2V缓存。

## 4. 文件和数据规则

- data/、models/、cache/、已有 checkpoints/ 和 outputs/ 默认只读，不擅自删除。
- 新实验必须使用独立输出目录，不能覆盖历史权重。
- 新想法先建experiments/<任务>/<想法>/，新产物放outputs/<任务>/<想法>/<run>/。已有大型结果不为整理而搬动；runs链接只导航，不复制成果。正式源码不再放进outputs。
- 训练崩溃后先保留旧输出，再把续训结果写到独立目录。
- .safetensors LoRA文件不能用--resume_from_checkpoint当作完整训练状态；历史T2V入口的增量续训使用--lora_checkpoint。当前VACE smoke入口没有完整状态resume，不将旧T2V参数套用到本轮。
- 同一缓存目录或输出目录不能被两个进程同时写入。
- 不把 Cap3D prompt、测试集或验证集混入本轮训练。

## 5. 训练和故障处理规则

- 未经用户明确要求，不执行 sbatch、不启动正式训练、不启动验证推理。
- A6000 原训练曾因 free(): invalid next size (normal) 在约 2742 个 batch 处退出；worker 0 下仍复现过原生崩溃，根因未确认，不能声称已修复。
- 续训脚本是 scripts/resume_wan81_a6000.sh，默认读取平铺目录内的 step-2400.safetensors，输出到独立的 checkpoints/building_wan_a6000_single_orbit_resumed_<时间>。可用 --dry-run 只读检查；不得把它当作自动下一步。
- 最高 checkpoint、日志 step 和 optimizer update 不是同一概念。当前 logger step 每个物理 batch 加一，续训必须从来源参数换算样本预算，不恢复 optimizer/RNG/shuffle。
- 历史T2V训练完成证据待核实，29个固定条件验证视频已经生成，见PROJECT_CONTEXT.md；不要凭旧交接自动补训或重新推理。该说明不覆盖当前VACE run的真实更新与完成状态。
- 续训前必须确认没有旧训练进程；推荐在 tmux 中运行，断开 SSH 后任务仍可继续。
- 黄色警告、NVML 提示和低 GPU 利用率不能直接当成训练失败；以 Python traceback、进程状态和输出文件为准。
- GPU计算利用率与显存占用比例分开解释；显存约8GiB不表示没有GPU训练。文档记录运行状态时附核查时刻，区分CPU测试、更新已发生、进程正常退出和视觉效果确认。

## 6. 修改和验证

修改前：

1. 读取本文件、PROJECT_CONTEXT.md、TRAINING.md 和 EXPERIMENTS.md。
2. 检查 git status、当前分支和最近提交。
3. 核对实际服务器路径、Python 路径、缓存数量和输出目录。

修改后至少执行：

    git diff --check
    bash -n scripts/train_wan81_a6000.sh scripts/resume_wan81_a6000.sh
    /home/shi/miniconda3/envs/scj/bin/python -m py_compile code/Python_3D_Scanner/training/wan_train_entry.py scripts/validate_wan81_a6000.py
    /home/shi/miniconda3/envs/scj/bin/python -m unittest discover -s scripts/tests -v
    git status --short

不得声称执行了没有执行的训练、验证或服务器命令。

GPU任务运行时，CPU回归可显式设置 `CUDA_VISIBLE_DEVICES=''`，不启动额外CUDA计算；不要为文档维护中断、重启或修改正在运行的训练配置。

## 7. Git 和交接

本地仓库是 D:\building_wan，远程仓库是 git@github.com:woshuaime/building_wan.git。提交文档或代码后，先检查 diff，再推送 main，最后在服务器执行 git pull --ff-only origin main。不要用强制推送覆盖他人的提交。

用户于2026-10-20设定持续工作要求：**每次用户实际启动GPU训练或推理后，都主动更新项目文档并进行Git记录，不必再次等待用户提醒或另问是否提交。**

- 每次后续接手、阶段结束或失败时，先核对实际run、进程/终端、配置、更新数、退出码和已有成果。运行中写带时刻的快照，结束后更新为真实成功/失败，不把保存权重或loss下降单独当作成功。
- 同步PROJECT_CONTEXT.md、EXPERIMENTS.md、TRAINING.md和SESSION_HANDOFF.md；必要时更新README与专项说明，记录参数、数据身份/帧号、权重和结果路径、耗时/显存、效果及未验证事项。
- 完成相关检查后，审查差异，只提交本轮相关代码、配置和文档，再按既有main流程推送origin；不得把不相关未提交工作一并打包。核实GitHub账号、远程和提交结果，不强推。
- 模型、LoRA权重、原始数据、视频/图片成果、临时文件、日志和环境安装产物默认不入Git；在文档中记录实际保存位置。该持续授权不等于允许自动启动新GPU任务、追加训练预算或修改冻结参数。

重要工作完成后同步更新：

- PROJECT_CONTEXT.md：当前真实状态和下一步。
- EXPERIMENTS.md：实验台账。
- TRAINING.md：可执行命令和排错方法。
- SESSION_HANDOFF.md：下一次会话需要知道的事实。

chuanjun.code-workspace 只描述本地 Windows 工作区，不写服务器 Linux 绝对路径；除非本地目录结构发生变化，否则保持不动。

根目录 PROJECT_CONTEXT.md / TRAINING.md / EXPERIMENTS.md / SESSION_HANDOFF.md 是长期累计文档，必须在原文件内保留历史正文并追加最新进展。docs/ 原迁移记录也已恢复原文及后续更新；归档和 .pre_git_sync_20260924/ 作为额外备份。扫描器 src/ 当前缺失，不能把历史 GUI 操作描述当作已验证能力。

## 8. Complex12完成记录与文档维护

- Complex12 Rev1的固定run为outputs/ComplexSingleOrbit_Rev1_Pilot_20261022；C/U/S各300真实optimizer更新、B/L/C/U/S共60条81帧、4860张原始预测、12视频/12总览及指标报告已经完成。日常查看experiments/vace/complex12_rev1/scripts/status.py，不把resume.sh当状态查询。
- 训练CS01/03/05/06/07/09/10/11，验证CS02/04/08/12；17帧连续窗训练、81帧评估，训练seed20261022、推理seed20261018。C/U/S使用共同新初始化，各自新优化器与相同随机序列；L仅为旧P04/P06参考。
- C从40更新恢复完整optimizer/RNG到300；B/CS04保存完整预测后原生退出−6，独立验证后保留有效成果。这些事实不等于原生故障根因已经解决，也不授权重复成功任务。
- 专用源码、数据身份、loss映射、指标口径与局限见experiments/vace/complex12_rev1/README.md及原run。不能将历史T2V的weights-only续训或两例smoke预算套到Complex12。
- 当前状态与历史过程统一在根目录四份长期文档；docs/PROJECT_CONTEXT、EXPERIMENTS、HANDOFF、PROGRESS保留早期正文、后续更新和根入口。专题和实验README负责细节，run中的报告/配置/日志作为冻结证据。
- 文档全面维护针对项目自有当前说明，不改第三方文档、原始数据包说明、已有run报告和历史快照。更正历史事实时另记补充和日期，不把原日志改成成功记录。
- 2026-10-23的Markdown更新保留现有非Markdown工作区修改，只提交相关文档；没有训练、推理、依赖更新或外部worker调用。完成证据与检查以docs/DOCUMENTATION_INDEX.md和实际Git记录为准。

## 9. 2026-10-23用户更正：长期MD必须在原处持续积累

用户明确要求一直存在的MD保留“从以前到现在”的更新。此前ccfe24d/357b09f将部分长期文档压缩成当前摘要、把历史仅放归档的处理不符合这一要求；本次恢复原文件正文并保留后续更新。

- README、PROJECT_CONTEXT、EXPERIMENTS、TRAINING、SESSION_HANDOFF和AGENTS持续维护；可以更新顶部状态，但不得以当前摘要替换历次正文。
- 新阶段按日期追加，保留原实验目的、方案、参数、执行、失败、排错、结果、结论和未解决事项。已有文档的记录顺序不因整理而丢失。
- 后来完成不删除“当时尚未完成”的快照；后来发现错误，应另记原判断、纠正依据和新结论。过时命令注明适用时期，不能当新启动指令。
- docs/history或Git只作备份与追溯，不能成为读者查看原有历史的唯一途径；不再把长期文档改成仅有跳转链接的入口页。
- 新想法仍用独立experiments目录和run；目录整理不能顺带删减主文档的演进记录。

### 已形成的规则沿革

| 时期 | 规则背景与持续保留的内容 |
| --- | --- |
| 2026-09的T2V阶段 | 独立划分、统一prompt缓存、权重增量续训及物理batch步号口径 |
| 2026-10-14/15审计诊断 | 进程身份、输出隔离、真实退出与原生故障未确认的边界 |
| 2026-10-20 VACE两例及八例评估 | 固定预算、GT隔离、不自动重试；GPU执行后主动更新文档与Git |
| 2026-10-22/23 Complex12 | 该run单独授权、配对初始化/采样、真实300更新与完整恢复状态 |
| 2026-10-23目录整理 | 想法、公共代码、run分开；大型成果不搬动 |
| 2026-10-23本次纠正 | 长期MD原文恢复，今后追加更新，不用摘要或归档替代历史正文 |
