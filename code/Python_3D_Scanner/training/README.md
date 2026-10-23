# 项目训练工具：职责、历史配置与当前入口

更新：2026-10-23（UTC+09:00）。本目录是项目自己的数据准备与训练/验证工具，DiffSynth-Studio作为第三方引擎。旧Windows smoke/pilot已是历史阶段，当前最新完成实验是 [Complex12 Rev1](../../../experiments/vace/complex12_rev1/README.md)。

## 1. 文件职责

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

## 2. 历史配置集合

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

## 3. 三类数据划分不能混用

| 实验 | 划分 | 关键区别 |
| --- | --- | --- |
| building_v1 | 1912训练、239验证、239测试；pilot200来自训练 | seed20260902，按source_obj分组 |
| A6000统一prompt单轨 | 1992训练、234验证、118测试 | [冻结清单](../../../configs/single_orbit_same_prompt_v1/)，所有视频同prompt |
| VACE条件补全 | Pilot8两例固定17帧；Complex12为8训练/4验证 | GT/Damaged/Exact配对，独立配置与损失 |

building_v1的历史划分工具和数据入口保留在 [data_splits/](data_splits/)。生成或覆盖划分是新的数据变更，不能在本次文档维护中自动执行。早期Windows逐栋prompt与后来的统一prompt缓存策略也不能混用。

## 4. 缓存与更新口径

统一prompt单轨缓存有1992个.pth，标记compact-sft-shared-context-bf16-v2。共享文本context只对相同prompt成立；改为逐栋文本必须新建缓存设计。latent落盘dtype按实际数据判断，不能从缓存名推断。

T2V历史logger每个物理batch增加step，累积中的batch也计数。当前默认batch1/accum4与原batch2/accum2的有效batch相同，但文件步号不能直接互换。weights-only续训不恢复optimizer/RNG/shuffle，须从来源参数换算已处理样本。

Complex12的300是明确的真实optimizer update，恢复另有完整状态；不能把上述T2V续训规则照搬过去。VACE不使用T2V缓存、旧T2V LoRA或Cap3D prompt。

## 5. 当前操作与验证

当前Python为/home/shi/miniconda3/envs/scj/bin/python。A6000公共入口及只读状态查询统一看 [TRAINING.md](../../../TRAINING.md)。最新VACE专用实现位于 [Complex12 scripts](../../../experiments/vace/complex12_rev1/scripts/)，两例LoRA实现仍在项目scripts中。

旧train_domain_lora/validate_domain_lora使用--run区分实际执行；输出保护、--allow-existing、--overwrite等能力不等于获得覆盖历史结果的授权。旧脚本完整Windows操作说明保留在 [归档](../../../docs/history/20261023_before_md_refresh/code/Python_3D_Scanner/training/README.md)，不再列为建议启动步骤。

项目规定的CPU回归为scripts/tests；本目录相邻tests/test_training_tools.py另有12项历史训练工具测试。依赖缺失src的GUI测试未通过当前文档任务验证，不能声称整个扫描器测试集通过。本次只维护Markdown，不改现存未提交训练代码。
