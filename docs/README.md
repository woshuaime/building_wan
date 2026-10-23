# 文档阅读入口

更新：2026-10-23（UTC+09:00）。当前实验已经推进到Complex12 Rev1全部完成；旧迁移与故障记录按历史保留。

## 先看什么

| 想了解的内容 | 文档 |
| --- | --- |
| 项目用途与主要成果 | [根README](../README.md) |
| 历次上下文、环境变化与最新状态 | [PROJECT_CONTEXT](../PROJECT_CONTEXT.md) |
| 从早期至今的实验、失败、结果与结论 | [EXPERIMENTS](../EXPERIMENTS.md) |
| 历次操作/排错及当前状态查询与恢复 | [TRAINING](../TRAINING.md) |
| 历次会话记录与最新交接 | [SESSION_HANDOFF](../SESSION_HANDOFF.md) |
| 按研究想法打开结果 | [experiments](../experiments/README.md) |
| 所有维护文档与更新记录 | [DOCUMENTATION_INDEX](DOCUMENTATION_INDEX.md) |
| 工作规则与目录边界 | [AGENTS](../AGENTS.md)、[PROJECT_STRUCTURE](PROJECT_STRUCTURE.md) |

## 专题资料

| 文档 | 适用范围 |
| --- | --- |
| [P01数据交接](BUILDINGFILLER_P01_VACE_HANDOFF.md) | 已接收包、身份、Exact规则和历史来源 |
| [VACE模型与执行入口](VACE_P01_EXECUTION.md) | 正式权重、本地组件、Pilot8到Complex12的关系 |
| [P04/P06 LoRA17](VACE_P04_P06_LORA_SMOKE.md) | 两例训练题的配置、6次推理、参数/梯度证据与历史17帧观察 |
| [原生稳定性诊断](NATIVE_STABILITY_20261014.md) | 2026-10-14/15的真实故障与诊断范围，根因未确认 |
| [项目审计](PROJECT_AUDIT_20261014.md) | 当时的工具修复、测试和完成状态限制 |
| [Claude worker](CLAUDE_WORKER.md) | 本地可选工具实现和授权边界，不表示已调用API |

docs下的 [PROJECT_CONTEXT](PROJECT_CONTEXT.md)、[EXPERIMENTS](EXPERIMENTS.md)、[PROGRESS](PROGRESS.md)、[HANDOFF](HANDOFF.md)已恢复原始迁移/过程正文并保留后续补记。根目录四份长期文档同样包含从以前到现在的记录；当前执行状态读其中最新日期的补记，无需为查看旧内容跳到归档。

## 历史与冻结证据

- [结构整理前归档](history/20261023_before_structure/README.md)：保留根文档的完整旧过程。
- [本次Markdown更新前归档](history/20261023_before_md_refresh/README.md)：保留旧迁移快照、P01长记录、worker和扫描器说明。
- .pre_git_sync_20260924、第三方README、数据包README以及run内最终报告/日志保持原样。
- [Complex12正式报告](../outputs/ComplexSingleOrbit_Rev1_Pilot_20261022/00_看结果/最终实验结论.md)是该run交付证据；后续发现或新实验应另作记录，不静默改写历史结果。

文档里的PID、空闲显存、tmux窗口、旧主机和时间只在各自核查时刻有效。历史“待完成”不表示现在需要重跑；先查对应run/progress及实验README。

## 2026-10-23后续补记：正文历史已恢复

前一轮把长期文档改成当前状态摘要/导航页，虽保存备份，仍不符合用户的持续记录要求。本次把历史直接恢复到原文件，最新参数、结果和已完成状态继续保留，并按日期说明两次文档处理及本次纠正。此后新增进展追加到原文，归档仅作备份。
