# 会话交接

核对日期：2026-10-23（UTC+09:00）。本次任务为“全部项目Markdown细致更新”，没有启动训练、推理、外部worker或新实验。

## 1. 接手顺序

1. 阅读[AGENTS.md](AGENTS.md)、[当前状态](PROJECT_CONTEXT.md)与[操作手册](TRAINING.md)。
2. 从[实验目录](experiments/README.md)确认目标是Pilot8、Complex12还是历史T2V。
3. 核对git status/log及对应run的真实状态，不能从历史快照的PID或“下一步”直接行动。
4. 查看[完整文档索引](docs/DOCUMENTATION_INDEX.md)区分当前说明、专题历史和冻结档案。

## 2. 最新成果已经完成

Complex12的C/U/S均300更新，B/L/C/U/S各12条81帧；60轨迹、4860PNG、12视频、12总览、指标及中文报告齐全。progress.json为complete，无待自动续跑任务。

[结果入口](outputs/ComplexSingleOrbit_Rev1_Pilot_20261022/00_看结果/README_从这里开始.md)提供12栋身份、媒体链接与训练/验证表。S的平均数略领先U，不代表结构全面改善；细部弯折、保留区变化、闭环代价均已写入[报告](outputs/ComplexSingleOrbit_Rev1_Pilot_20261022/00_看结果/最终实验结论.md)。

## 3. 重要路径与兼容关系

| 内容 | 位置 |
| --- | --- |
| Complex12唯一专用源码 | experiments/vace/complex12_rev1/scripts/ |
| 固定run | outputs/ComplexSingleOrbit_Rev1_Pilot_20261022/ |
| 新浏览入口 | experiments/vace/complex12_rev1/runs/20261022/，目录链接 |
| 原scripts兼容路径 | 固定run/scripts，指向新源码 |
| 当前冻结配置 | 固定run/resolved_config.json，配置ID以真实JSON为准 |
| 三组正式权重 | 固定run/checkpoints/C、U、S/step300.safetensors |
| 恢复入口 | 固定run/resume.sh；已完成时直接返回 |
| 历史L权重 | outputs/vace/lora_roof_smoke/run_20261020_165237_150325_2048720/training/lora_update0300.safetensors |

旧resume.sh和结果路径不因源码迁移失效。新想法必须新建实验目录和run，不能改写当前完成标记或冻结配置。历史LoRA和原始数据不复制进新源码目录。

## 4. 必须延续的事实

- C在挂载恢复后从40次更新继续到300，完整恢复optimizer/RNG；新U/S各自从共同新初始化开始。
- B/CS04在保存完整预测后退出−6，经独立验证后复用；不能称全部60个worker都正常退出。
- 旧P04/P06 LoRA学过这两栋，不能当无泄漏初始化；CS06/source1251不是P06/source1848。
- 60条Complex12轨迹有连续逐帧图复核，不声称实时高频闪烁已被量化或三维恢复已验证。
- 原生崩溃根因尚未定位；历史T2V完整预算结束仍缺独立证据。
- 现有1992个T2V缓存为.pth；共享文本context可复用的前提是同一prompt。
- 扫描器GUI源码缺失；Windows命令和旧Slurm流程不能当当前A6000操作。

## 5. 本次文档更新与原有工作区

接手时已有历史T2V代码、诊断工具和若干Markdown的未提交改动。用户此次授权更新所有项目Markdown，因此相关文档纳入本次维护；已有非Markdown代码保持原样，不混入文档提交。

旧迁移页、扫描器README及早期VACE操作原文保留在[本次归档](docs/history/20261023_before_md_refresh/README.md)；上一轮四份根文档的字节级快照继续保留。冻结run报告、原始数据包说明、第三方文档和工具档案不改写为“当前状态”。

目录整理提交为ccfe24d。文档更新后的提交以实际git log为准；不把Git工作区HEAD当作既有run的唯一源码证明。

## 6. 验证与下一步

CPU检查按TRAINING执行；原6466个成果文件和历史权重保护，不为更新文档重新计算大权重哈希或跑GPU。文档核对范围与本次实际检查结果记录在[文档索引](docs/DOCUMENTATION_INDEX.md)。

下一轮实验尚未授权。当前只交付已有结果与完整文档；报告中的配对多种子建议不能自动转为执行任务。
