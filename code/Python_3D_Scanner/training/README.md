# 训练工具：早期使用记录与持续更新

最近补记：2026-10-23（UTC+08:00，北京时间）。本文件持续保留Windows smoke/pilot、数据划分、早期命令和后续服务器使用说明。历史正文已恢复在原文件内，后续进展按日期追加；归档仅作备份，无需离开本文件才能阅读旧记录。

阅读顺序：[此前累计正文](#history-records) → [2026-10-23阶段更新](#update-20261023) → [本次历史恢复及生效约定](#history-repair)。旧段落的“当前”“下一步”、命令和PID按其记录时点理解；较晚的更正与实际完成证据在后文，旧命令不因此重新执行。

截至本次修复：Pilot8系列及Complex12的C/U/S各300更新、五方法60条81帧已完成。本次仅修复文档；[实验入口](../../../experiments/README.md)和[文档索引](../../../docs/DOCUMENTATION_INDEX.md)仍可直接使用。

<a id="history-records"></a>

## 一、此前累计正文：此次整理前长期保留的内容

以下完整保留原有记录顺序、参数、命令、过程和当时结论，只调整标题层级。原记录中的未完成状态与后来完成状态共同构成历史，不能用后者替换前者。

<!-- RESTORED_HISTORY_BEGIN -->
### 建筑视频 LoRA 实验工具

> 2026-10-14：下文记录早期 Windows smoke/pilot 流程，并非当前 A6000 的默认路径或当前阶段。当前统一 prompt 单轨实验、只读预检、28 份权重及29个对比视频见根目录 [PROJECT_CONTEXT](../../../PROJECT_CONTEXT.md)、[TRAINING](../../../TRAINING.md) 和 [EXPERIMENTS](../../../EXPERIMENTS.md)。部分历史 src 路径当前缺失，不要直接复制旧命令。

本目录保存项目自己的实验配置和入口。`DiffSynth-Studio` 只作为第三方训练/推理引擎，不在其官方示例中维护项目逻辑。

#### 当前阶段

`smoke.json` 对应已经完成的20视频链路测试：

- Wan2.1-T2V-1.3B
- 256x256
- 17帧
- LoRA rank 8
- 1 epoch / 20 steps
- GPU 1

这一阶段只验证训练链路和领域 LoRA 是否可加载，不代表最终建筑补全效果。

#### 1. 配置预检

在 `D:\code\Python_3D_Scanner` 中运行：

```bat
D:\scj\envs\diffsynth\python.exe training\train_domain_lora.py --config training\configs\smoke.json
```

不带 `--run` 时只检查数据、路径和参数，并打印实际训练命令，不会占用 GPU。

#### 2. 启动训练

```bat
D:\scj\envs\diffsynth\python.exe training\train_domain_lora.py --config training\configs\smoke.json --run
```

如果输出目录已经包含权重，脚本会主动停止以防覆盖。确认确实需要重跑时，额外添加 `--allow-existing`。

当前 smoke 权重已经生成，无需再次训练。

#### 3. 验证预检

```bat
D:\scj\envs\diffsynth\python.exe training\validate_domain_lora.py --config training\configs\smoke.json
```

该命令确认基础模型、step-10、step-20和输出路径，不会加载模型。

#### 4. 生成公平对比视频

```bat
D:\scj\envs\diffsynth\python.exe training\validate_domain_lora.py --config training\configs\smoke.json --run
```

脚本只加载一次基础模型，并使用完全相同的 prompt、negative prompt、seed和推理参数依次生成：

```text
outputs/validation/building_smoke/base.mp4
outputs/validation/building_smoke/step-10.mp4
outputs/validation/building_smoke/step-20.mp4
outputs/validation/building_smoke/validation_metadata.json
```

如果中途中断，可用 `--only step-20 --run` 只生成指定结果。已有文件默认不会覆盖，覆盖时添加 `--overwrite`。

#### 5. 汇总结果

```bat
D:\scj\envs\diffsynth\python.exe training\summarize_run.py --config training\configs\smoke.json
```

输出 `outputs/validation/building_smoke/experiment_summary.json`，记录训练参数、loss摘要、checkpoint和验证状态。

#### 判断标准

首先检查工程正确性：三个视频均成功输出，元数据中的 prompt、seed、尺寸、帧数完全一致，LoRA加载无报错。

然后观察模型变化：是否更像单体建筑模型、是否保持白色干净背景、是否出现连续旋转、几何形状是否跨帧稳定。20步结果即使没有明显改善也不代表路线失败，它只用于 smoke test。

通过上述检查后，再建立固定 train/val/test 划分和200视频试验配置，不直接开始2390视频全量训练。

#### building_v1 固定划分

生成命令：

```bat
D:\scj\envs\diffsynth\python.exe training\prepare_splits.py
```

默认使用随机种子 `20260902`，按 `video_manifest.json` 中的 `source_obj` 分组，而不是按单个视频随机拆分。已冻结的 `building_v1` 结果为：

```text
train:     1912
val:        239
test:       239
pilot_200:  200（仅从train中选择）
```

输出位于 `training/data_splits/building_v1`。`split_manifest.json` 记录输入文件SHA-256、随机种子、样本数和集合交叉审计；`split_index.csv` 记录每个视频对应的源OBJ和集合。已有划分默认不覆盖，重建时必须显式添加 `--overwrite`。

#### 200视频试训

先做配置预检：

```bat
D:\scj\envs\diffsynth\python.exe training\train_domain_lora.py --config training\configs\pilot_200.json
```

确认后启动一轮200步训练：

```bat
D:\scj\envs\diffsynth\python.exe training\train_domain_lora.py --config training\configs\pilot_200.json --run
```

该实验保持256x256、17帧、rank 8和相同学习率，只改变训练样本规模，便于与20视频smoke实验比较。权重在第50、100和200步保存。

训练完成后生成公平对比：

```bat
D:\scj\envs\diffsynth\python.exe training\validate_domain_lora.py --config training\configs\pilot_200.json --run
```
<!-- RESTORED_HISTORY_END -->

<a id="update-20261023"></a>

## 二、2026-10-23：阶段更新与当前资料补全

下面继续保留357b09f阶段补充的结果、参数和操作说明。该阶段曾把历史移到归档、部分原文页改成导航；这一文档组织做法已由第三节的用户更正取代，实验事实和既有检查结果保留。

<!-- RETAINED_UPDATE_BEGIN -->
### 项目训练工具：职责、历史配置与当前入口

更新：2026-10-23（UTC+09:00）。本目录是项目自己的数据准备与训练/验证工具，DiffSynth-Studio作为第三方引擎。旧Windows smoke/pilot已是历史阶段，当前最新完成实验是 [Complex12 Rev1](../../../experiments/vace/complex12_rev1/README.md)。

#### 1. 文件职责

| 文件/组 | 用途 |
| --- | --- |
| [wan_train_entry.py](wan_train_entry.py) | 项目Wan训练入口、缓存与LoRA接入，A6000公共脚本调用它 |
| [train_domain_lora.py](train_domain_lora.py)、[train_split_domain_lora.py](train_split_domain_lora.py) | 读取历史实验配置并组织训练 |
| [validate_domain_lora.py](validate_domain_lora.py) | 历史配置驱动的Base/LoRA比较 |
| [prepare_splits.py](prepare_splits.py) | source_obj分组划分 |
| [prepare_temporal_clips.py](prepare_temporal_clips.py) | 历史长视频片段准备 |
| [experiment_config.py](experiment_config.py) | 配置读取、路径与参数 |
| [summarize_run.py](summarize_run.py) | 汇总已存在训练和验证记录 |
| run_formal_pipeline.py、watch_formal_pipeline.py、monitor_formal_pipeline_30m.py | 历史流程执行/监控入口，不作为当前状态查询默认命令 |
| audit_experiment_completion.py、build_experiment_report_assets.py、build_final_experiment_report.py、build_full_experiment_pdf.py | 历史完成检查、图表与报告 |
| analyze_validation_videos.py、check_dual_gpu.py | 历史视频分析与双卡检查 |
| set_windows_pagefile.ps1、word_soffice_proxy.cs | Windows历史环境辅助，当前Linux不使用 |

仅看文件名不能判断只读。数据准备、报告和监控脚本可能生成文件或触发流程；文档维护不执行这些历史入口。

#### 2. 历史配置集合

[configs/](configs/)现有14个JSON：

| 配置 | 历史目的 |
| --- | --- |
| smoke.json、pilot_200.json | 20视频链路测试、200训练视频试验 |
| dual_3060_smoke.json、dual_3060_smoke_384.json、dual_3060_smoke_lossavg.json | 双3060不同尺寸/损失汇总对照 |
| train_1912_all_256_cooled.json、train_1912_all_384_cooled.json | building_v1全训练集配置 |
| train_500_balanced_384.json、train_500_balanced_384_power100.json | 500视频子集及功耗配置历史 |
| cache_4780_half0_256.json、cache_4780_half1_256.json、cache_4726_half0_remaining_256.json、cache_9060_remaining_384.json | 历史分段缓存记录 |
| building_wan_a100_validation.json | A100历史验证配置 |

这些配置包含各自历史路径，不能当作当前scj环境的默认参数。新想法的配置进入对应experiments目录，不覆盖这里的历史配置。

#### 3. 三类数据划分不能混用

| 实验 | 划分 | 关键区别 |
| --- | --- | --- |
| building_v1 | 1912训练、239验证、239测试；pilot200来自训练 | seed20260902，按source_obj分组 |
| A6000统一prompt单轨 | 1992训练、234验证、118测试 | [冻结清单](../../../configs/single_orbit_same_prompt_v1/)，所有视频同prompt |
| VACE条件补全 | Pilot8两例固定17帧；Complex12为8训练/4验证 | GT/Damaged/Exact配对，独立配置与损失 |

building_v1的历史划分工具和数据入口保留在 [data_splits/](data_splits/)。生成或覆盖划分是新的数据变更，不能在本次文档维护中自动执行。早期Windows逐栋prompt与后来的统一prompt缓存策略也不能混用。

#### 4. 缓存与更新口径

统一prompt单轨缓存有1992个.pth，标记compact-sft-shared-context-bf16-v2。共享文本context只对相同prompt成立；改为逐栋文本必须新建缓存设计。latent落盘dtype按实际数据判断，不能从缓存名推断。

T2V历史logger每个物理batch增加step，累积中的batch也计数。当前默认batch1/accum4与原batch2/accum2的有效batch相同，但文件步号不能直接互换。weights-only续训不恢复optimizer/RNG/shuffle，须从来源参数换算已处理样本。

Complex12的300是明确的真实optimizer update，恢复另有完整状态；不能把上述T2V续训规则照搬过去。VACE不使用T2V缓存、旧T2V LoRA或Cap3D prompt。

#### 5. 当前操作与验证

当前Python为/home/shi/miniconda3/envs/scj/bin/python。A6000公共入口及只读状态查询统一看 [TRAINING.md](../../../TRAINING.md)。最新VACE专用实现位于 [Complex12 scripts](../../../experiments/vace/complex12_rev1/scripts/)，两例LoRA实现仍在项目scripts中。

旧train_domain_lora/validate_domain_lora使用--run区分实际执行；输出保护、--allow-existing、--overwrite等能力不等于获得覆盖历史结果的授权。旧脚本完整Windows操作说明保留在 [归档](../../../docs/history/20261023_before_md_refresh/code/Python_3D_Scanner/training/README.md)，不再列为建议启动步骤。

项目规定的CPU回归为scripts/tests；本目录相邻tests/test_training_tools.py另有12项历史训练工具测试。依赖缺失src的GUI测试未通过当前文档任务验证，不能声称整个扫描器测试集通过。本次只维护Markdown，不改现存未提交训练代码。
<!-- RETAINED_UPDATE_END -->

<a id="history-repair"></a>

## 三、2026-10-23：恢复原文件内的持续历史

用户明确要求长期MD保留“从以前到现在”的更新。先前仅保留当前摘要、让读者另找归档的处理不符合要求；本次将上述旧正文直接恢复到原文件，同时保留后续更新，没有回退实验结果或把完成状态改回执行中。

以后在本文件按日期追加阶段进展、失败与修复、参数变更、结论和核查范围。需要纠正旧结论时，说明原判断、后来的证据及新结论；原记录继续保留。可以在顶部维护简短状态和阅读索引，不能用摘要或归档链接替换历史正文。

本次修复依据原文件备份及Git记录恢复，没有补造新的运行经历；原数据、权重、视频、指标、冻结报告和现有非Markdown工作保持原样。此次检查与恢复范围见[文档索引](../../../docs/DOCUMENTATION_INDEX.md)中的后续补记。
