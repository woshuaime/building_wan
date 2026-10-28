# BuildingFiller建筑补全项目与演进记录

最近补记：2026-10-23（UTC+08:00，北京时间）。本文件持续保留项目发展路线、原有项目介绍、结果入口和后续阶段更新。历史正文已恢复在原文件内，后续进展按日期追加；归档仅作备份，无需离开本文件才能阅读旧记录。

阅读顺序：[此前累计正文](#history-records) → [2026-10-23阶段更新](#update-20261023) → [本次历史恢复及生效约定](#history-repair)。旧段落的“当前”“下一步”、命令和PID按其记录时点理解；较晚的更正与实际完成证据在后文，旧命令不因此重新执行。

截至本次修复：Pilot8系列及Complex12的C/U/S各300更新、五方法60条81帧已完成。本次仅修复文档；[实验入口](experiments/README.md)和[文档索引](docs/DOCUMENTATION_INDEX.md)仍可直接使用。

## 项目发展时间线

| 阶段 | 已保留的过程与结果 |
| --- | --- |
| 2026-09-04—09-08：双3060 | 1912视频、17帧片段的T2V实验；参数、更新数和视角覆盖问题保留 |
| 2026-09：A100 | 缓存准备失败、导入/依赖问题及之后正式训练，39份保留权重 |
| 2026-09-23—09-29：A6000单轨 | 统一prompt划分与缓存、反复原生崩溃、续训与权重平铺、29个验证视频；终态证据仍待核实 |
| 2026-10-14—10-15：审计与诊断 | 游标/进程/验证工具修复，真实GPU诊断及按要求停止清理；根因未确认 |
| 2026-10-18—10-19：BuildingFiller P01 | 数据交接、图像/Mask/相机验证、VACE权重核查和执行器准备 |
| 2026-10-20：Pilot8 | 八例Base、CPU Composite、P04/P06 LoRA17、LoRA300八例81帧；中途失败、授权重跑和最终观察都保留 |
| 2026-10-22—10-23：Complex12 Rev1 | 配对方案、B/CS04保存后异常、B/L完成、C中断后恢复、C/U/S各300更新、60条轨迹与最终评测 |
| 2026-10-23：结构与文档 | 实验目录整理；先前把长期MD缩成摘要的做法在本次纠正，恢复原文并续记最新状态 |

<a id="history-records"></a>

## 一、此前累计正文：此次整理前长期保留的内容

以下完整保留原有记录顺序、参数、命令、过程和当时结论，只调整标题层级。原记录中的未完成状态与后来完成状态共同构成历史，不能用后者替换前者。

<!-- RESTORED_HISTORY_BEGIN -->
### 建筑视频 Wan LoRA 与 VACE 条件补全

P01–P08已完成原始VACE-1.3B的81帧补全、三方可视化和CPU缺口回贴。当前研究推进到P04/P06的17帧条件补全LoRA小样本验证：检查模型能否学会训练题上的完整表面恢复。

首轮已由用户在服务器终端执行并正常完成。2026-10-20 17:05:32（北京时间）核查：300次真实更新、两例各150次、6次17帧推理全部完成，100/300步LoRA与两例各四项成果齐全，总耗时749.247秒。五个代表视角中P04屋顶凹槽和P06开口均明显改善；完整视频时序和进一步人工验收待查看。当前先查看结果，不重复启动。

详细数据关系、参数、运行证据、成果说明和判断标准见 [P04/P06 LoRA实验说明](docs/VACE_P04_P06_LORA_SMOKE.md)；日常操作见 [TRAINING.md](TRAINING.md)。

#### 开发入口

- [项目规则](AGENTS.md)：编辑、训练授权、数据保护及检查要求。
- [当前状态](PROJECT_CONTEXT.md)：以实际产物为依据，区分历史报告和已核对事实。
- [操作手册](TRAINING.md)：当前 A6000 预检、续训和验证命令。
- [实验台账](EXPERIMENTS.md)：双 3060、A100 和 A6000 的独立实验记录。
- [会话交接](SESSION_HANDOFF.md)：下一次开发的入口与未解决问题。
- [BuildingFiller P01 与 VACE 准备](docs/BUILDINGFILLER_P01_VACE_HANDOFF.md)：三机边界、PowerShell SCP 命令、实际数据验证与 17 帧最小实验方案。
- [P04/P06 VACE LoRA实验](docs/VACE_P04_P06_LORA_SMOKE.md)：当前两例17帧训练、Base/100/300公平比较、GT监督路径、显存说明及结果检查。
- [VACE 权重与执行历史](docs/VACE_P01_EXECUTION.md)：官方模型、八例81帧执行、P01复用和既往故障证据。
- [2026-10-14 审计](docs/PROJECT_AUDIT_20261014.md)：已确认的问题、修改及验证范围。

#### 代码与数据流

历史 T2V：建筑渲染视频 → 固定 train/val/test 划分 → 视频 latent/文本 context 缓存 → DiT LoRA 训练 → 固定 prompt/seed 视频对比 → 人工检查与像素诊断。

已完成VACE基线：BuildingFiller原始包 → 真实样本/帧号验收 → ModelMask适配 → 原始VACE整序列81帧 → `outputs/vace/test/01–08/`。每例四文件包含原始预测和Before/Prediction/GT对照。

已完成CPU诊断：原始预测 + Damaged + 原推理ModelMask → 只替换缺口的Composite → `outputs/vace/composite_check/01–08/`。保留区域改善，P04凹槽和P06黑洞仍在。

当前条件训练：P04/P06原始PNG连续17帧 → GT监督目标 + Damaged/Mask/文本条件 → 同一份vace LoRA → 同条件17帧Base/LoRA-100/LoRA-300比较 → `outputs/vace/lora_roof_smoke/run_<时间>/`。旧T2V LoRA和缓存不参与。

- `code/Python_3D_Scanner/training/`：项目训练入口、历史数据工具、验证和报告工具。
- `code/DiffSynth-Studio/`：第三方引擎；项目逻辑优先放在项目层。
- `scripts/`：当前 A6000 入口与历史 A100/Slurm 入口。不要仅凭文件名选择环境。
- `configs/vace_roof_smoke_v1.json`：当前两例LoRA固定窗口、训练预算、推理参数和文本。
- `configs/single_orbit_same_prompt_v1/`：历史T2V冻结划分，1992/234/118，共2344个视频。
- `data/`、`models/`、`cache/`、`checkpoints/`、`outputs/`：原始或生成产物，默认只读。
- `logs/`：运行日志和历史暂停记录；旧记录中的路径可能已迁移。
- `docs/`：审计、兼容入口和 Claude worker 说明；当前状态统一指向根目录文档。
- `.pre_git_sync_20260924/`：历史快照，不作为当前运行入口。

当前 A6000 环境使用 `/home/shi/miniconda3/envs/scj/bin/python`。扫描器的 `requirements.txt` 属于历史 Windows GUI 环境，不是训练环境完整依赖清单。

扫描器 README 描述的桌面采集、OBJ 冻结与三元组渲染是历史设计；当前工作副本没有 `code/Python_3D_Scanner/src/` 和对应数据集工作流文档。需要从原始采集项目恢复并审查后再开发这部分，不能据此声称桌面工具可运行。

本轮LoRA已由用户手动授权并启动；文档维护只读取进度，不重启现有任务。训练题效果需结合真实可视化判断，不能写成独立测试成绩或81帧一致性结论。
<!-- RESTORED_HISTORY_END -->

<a id="update-20261023"></a>

## 二、2026-10-23：阶段更新与当前资料补全

下面继续保留357b09f阶段补充的结果、参数和操作说明。该阶段曾把历史移到归档、部分原文页改成导航；这一文档组织做法已由第三节的用户更正取代，实验事实和既有检查结果保留。

<!-- RETAINED_UPDATE_BEGIN -->
### BuildingFiller 建筑序列补全研究

文档核对日期：2026-10-23（UTC+09:00）。本仓库当前研究 **给定残缺建筑、Exact缺损位置和文本时，VACE能否恢复完整建筑序列**。历史Wan T2V实验学习外观与环绕生成，单独保留。

最新Complex12 Rev1实验已经完成：C/U/S各300次真实优化器更新；B/L/C/U/S共60条81帧轨迹、4860张原始预测、12条同步视频及12张总览。当前文档维护不启动训练或推理。

#### 从哪里开始

| 目的 | 入口 |
| --- | --- |
| 直接观看最新十二栋结果 | [视频、总览、指标和结论](outputs/ComplexSingleOrbit_Rev1_Pilot_20261022/00_看结果/README_从这里开始.md) |
| 了解各项研究想法的关系 | [实验总目录](experiments/README.md) |
| 核对当前完成状态和遗留问题 | [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) |
| 查看参数、方法与实验结论 | [EXPERIMENTS.md](EXPERIMENTS.md) |
| 查状态、使用CPU检查和定位故障 | [TRAINING.md](TRAINING.md) |
| 接手工作或新建实验 | [SESSION_HANDOFF.md](SESSION_HANDOFF.md)、[目录约定](docs/PROJECT_STRUCTURE.md) |
| 查所有项目文档及历史范围 | [文档索引](docs/DOCUMENTATION_INDEX.md) |

#### 当前结果说明什么

普通LoRA C在Complex12验证集的缺损区MAE比Base降低64.21%，四栋验证建筑均有改善。S的平均缺损/结构指标稍好于U，但验证MAE仅降低1.80%、结构Sobel仅降低0.19%，且主要由CS12贡献；部分屋檐与端墙弯折仍在，保留区变化和全图闭环误差增加。

这支持建筑补全LoRA在本数据分布上的迁移；尚不能认定结构加权稳定胜出，不能推断完整三维恢复或高频闪烁已经消除。完整口径、反例和训练/验证差异见 [中文报告](outputs/ComplexSingleOrbit_Rev1_Pilot_20261022/00_看结果/最终实验结论.md)。

#### 项目内的实验关系

1. **Pilot8 Base**：八栋原始VACE-1.3B完整81帧补全。
2. **CPU Composite**：只把Base预测贴入Exact，诊断已知区域变化；不是新的模型预测。
3. **LoRA17**：P04/P06各17帧、300更新，验证训练链路与训练题恢复。
4. **LoRA81**：同一LoRA-300扩展到八例81帧，不新增训练。
5. **Complex12 Rev1**：独立8栋训练/4栋验证，比较普通、均匀缺损、结构带加权；旧LoRA仅作L参考。
6. **历史T2V**：双3060、A100和A6000单轨实验；不混用其初始化、划分或缓存。

#### 代码、实验与结果如何分开

| 目录 | 职责 | 维护原则 |
| --- | --- | --- |
| [experiments/](experiments/README.md) | 一个想法一夹，专用源码、配置入口、runs链接 | 新想法不覆盖旧run |
| [scripts/](scripts/README.md) | 公共适配器、兼容入口、服务器工具及CPU测试 | 已有跨实验导入路径保留 |
| code/Python_3D_Scanner/ | 项目训练入口和历史数据工具 | 当前GUI源码缺失，见该目录README |
| code/DiffSynth-Studio/ | 第三方引擎 | 优先在项目层实现变化 |
| [configs/](configs/README.md) | 既有冻结配置与划分 | 历史配置不随新想法改写 |
| data/、models/ | 原始包、数据与基础模型 | 默认只读 |
| cache/、checkpoints/ | 历史共享缓存及权重 | 新实验不能混用未确认来源 |
| [outputs/](outputs/README.md) | 视频、预测、指标、实验权重、运行记录 | 保持已有物理路径 |
| logs/ | 运行和诊断记录 | 原生异常记录保留 |
| [docs/](docs/README.md) | 专题说明、文档索引及历史快照 | 当前事实与历史证据分开 |

Complex12的13个专用Python脚本位于experiments/vace/complex12_rev1/scripts/；原run/scripts保留兼容链接。runs链接只导航，Git不会沿链接打包大型成果。

#### 数据与模型边界

训练：同帧GT作为FlowMatch监督目标；Damaged白区填灰127、二值Exact转ModelMask、文本作为VACE条件。推理：只读Damaged、Exact和文本，GT与结构图不进入条件。评估：模型原始预测相对GT比较，已知区域相对Damaged比较，真实预测000/080单独计闭环。

Exact是配对数据提供的缺损定位，本项目尚未评估自动检测缺口。Complex12的000–079是80个独立视角，080输入重复首帧；预测080保留真实模型输出，不复制000。

#### 当前环境与完成标准

服务器项目根为/mnt/windowsE/chuanjun/building_wan；解释器为/home/shi/miniconda3/envs/scj/bin/python。实际环境为Python3.10.13、PyTorch2.11.0+cu128、CUDA runtime12.8、DiffSynth2.1.5、PEFT0.17.1，单卡A6000。旧A100的mv2v和Windows采集环境只属于历史记录。

完成必须同时有真实更新/退出证据、可读预测、统一评测与视觉结论；启动脚本、保存权重或loss下降都不能单独代替完成。原生崩溃根因仍未确认。研究建议不构成自动启动下一轮实验的授权，具体规则见 [AGENTS.md](AGENTS.md)。
<!-- RETAINED_UPDATE_END -->

<a id="history-repair"></a>

## 三、2026-10-23：恢复原文件内的持续历史

用户明确要求长期MD保留“从以前到现在”的更新。先前仅保留当前摘要、让读者另找归档的处理不符合要求；本次将上述旧正文直接恢复到原文件，同时保留后续更新，没有回退实验结果或把完成状态改回执行中。

以后在本文件按日期追加阶段进展、失败与修复、参数变更、结论和核查范围。需要纠正旧结论时，说明原判断、后来的证据及新结论；原记录继续保留。可以在顶部维护简短状态和阅读索引，不能用摘要或归档链接替换历史正文。

本次修复依据原文件备份及Git记录恢复，没有补造新的运行经历；原数据、权重、视频、指标、冻结报告和现有非Markdown工作保持原样。此次检查与恢复范围见[文档索引](docs/DOCUMENTATION_INDEX.md)中的后续补记。

## 四、2026-10-24：可带回家阅读的实验报告

本次补记采用UTC+09:00。根目录新增 [upload/](upload/README.md)，用于保存每一组实验的独立详细报告，方便复制到另一台机器分析下一步。

首份为 [Complex12 Rev1十二栋单轨完整实验报告](upload/20261022_Complex12_Rev1_十二栋单轨实验报告.md)。正文直接包括数据身份、五种方法、损失与采样、实际参数、运行异常和恢复、训练/验证数值、十二栋视觉观察、问题帧与下一步建议；只复制这一份MD也能理解实验。原视频、权重和指标留在原run，未搬动。

Complex12仍为C/U/S各300真实更新、五方法60条81帧已完成。本次仅整理报告和同步文档。以后每组实验完成后都在upload新增对应报告，并在原长期MD内继续追加历史，不能用报告入口替代旧正文。详细约定见 [upload/README.md](upload/README.md) 与 [AGENTS.md](AGENTS.md) 第10节。


## 2026-10-26：OriginalHard 结果与效果复核

[最新观看入口](outputs/vace/complex12_originalhard_eval/run_20261026_012033/00_看结果/README_从这里开始.md)位于outputs/vace/complex12_originalhard_eval/run_20261026_012033/00_看结果，不在旧Complex12目录。已有权重五方法60条81帧（40复用、20新推理），0新训练，12视频/12图/指标与结论齐全。

CS09/11改善明显，CS10/12仍有错误连接和坡面，CS12首尾形状不一致；S没有稳定压过C/U。可单独复制的详细[OriginalHard报告](upload/20261026_Complex12_OriginalHard_已有权重评测报告.md)已补齐，保留参数、真实执行、逐栋数值、反例帧和下一步建议。旧成果及本文件历史原样保留。


## 2026-10-26：C_mix 固定配比实验启动与后续交接

2026-10-26T22:26:41+08:00：普通C_mix已实际发生33/300更新，后续队列为16条推理、配对指标与视频。入口：[C_mix实验](experiments/vace/complex12_cmix_pilot/README.md) / [结果目录](outputs/vace/complex12_cmix_pilot/run_20261026_222008/00_看结果/README_从这里开始.md)。按用户最新要求，启动后结束对话，视觉复核及最终报告待另行通知。


## 2026-10-27：C_mix 完整评测、视觉复核与收尾

按用户后续通知，C_mix已完成报告与视觉收尾。真实数量为一个普通C_mix的300更新（250 Rev1/50 OriginalHard）、16条81帧/1296张原始PNG、32视频、6张固定Hard问题图及1张新增Rev1反例。此前33/300与等待复核的内容是当时快照，继续保留。

直接打开[结果入口](outputs/vace/complex12_cmix_pilot/run_20261026_222008/00_看结果/README_从这里开始.md)，优先看Hard CS10、Hard CS12，再看Rev1 CS10的B020条纹反例。[便携完整报告](upload/20261027_Complex12_C_mix_混合训练与配对评测报告.md)与run报告内容相同，内含训练/数据/参数、完整指标、逐栋结构观察、执行记录与结果路径。

本轮未达到同时改善Hard并保持Rev1的目标：Hard CS10连接面、CS12折面/首尾仍未解决；Rev1有10/12栋缺口MAE退步，CS10从0.009775升到0.127054并出现GT没有的条纹。建议保留旧C参考。下一步仅建议当前入口纯Rev1的小范围复现对照，未启动。本次收尾没有新GPU训练/推理、覆盖旧run或搬动大成果。


## 2026-10-27：Orbit17线性/循环配对实验已启动

核查时间：2026-10-27T20:16:32+09:00（UTC+09:00）。用户已授权线性组和循环组都执行，并要求确认实际开始训练后结束本轮对话。当前status=running，C_lin17已完成22/600次真实AdamW更新，C_ring17为0/600（待串行执行），推理0/40；日志已有有限loss及参数实际变化。这是启动快照，不是两组完成或效果验证。

固定run：`/mnt/windowsE/chuanjun/building_wan/outputs/vace/complex12_orbit17_pair/run_20261027_200842`。专用源码与说明：[Orbit17实验](experiments/vace/complex12_orbit17_pair/README.md)；[结果导航](outputs/vace/complex12_orbit17_pair/run_20261027_200842/00_看结果/README_从这里开始.md)。tmux=`complex12_rev1:orbit17_pair`（%3），runner PID 2469437，GDB worker PID 2469491；源码提交`1a82bd5`。

这轮比较普通C的线性17帧窗和跨079→000的循环窗；每组600更新，300/600分预算配对，Rev1训练、Hard只评测。自动计算与交付计划为40条81帧/3240原PNG、48视频，完成后等待后续视觉复核及报告。所有旧实验历史继续保留。


## 2026-10-29：Orbit17配对实验复核与最终报告完成

计算于2026-10-27T23:41:29+09:00结束，用户随后通知复核/报告收尾；10月28日至29日完成本次复核。实际run为`outputs/vace/complex12_orbit17_pair/run_20261027_200842`。两组各600真实AdamW更新、40条81帧/3240原PNG、48视频、7固定问题图和4张最终七行探针总览齐全；另保留4张早期探针图与1张覆盖图。45个队列worker均单次启动、EXIT 0；线性300→600为计划内完整optimizer/RNG恢复，循环连续0→600，没有失败后回放或新增GPU任务。

[详细便携报告](upload/20261029_Complex12_Orbit17_线性与循环窗口配对实验报告.md)；[直接看结果](outputs/vace/complex12_orbit17_pair/run_20261027_200842/00_看结果/README_从这里开始.md)。报告直接列出方案、执行、四组与逐栋指标、300/600配对和固定问题帧。主要结论：循环通常降低缺口像素误差，但未稳定改善闭环；Hard CS12显著改善（两组600都有收益），CS10关键连接仍错。Rev1均值改善但ring仍有4栋MAE退步。历史正文和前轮C_mix失败结论保留。

下一步仅建议第二训练随机序列的600更新线性/循环配对复验，检验像素收益与首尾代价是否可重复；尚未执行或排队。Hard未训练建筑仅CS12，是开发验证，不能夸大泛化。旧run、权重、原始数据/预测和已有报告保持不变；本次收尾不自动开启任何训练、推理或新模块。
