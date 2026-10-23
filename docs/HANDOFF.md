# 会话交接：历次记录与后续更新

最近补记：2026-10-23（UTC+08:00，北京时间）。本文件持续保留原交接模板、当时的委派与复核要求，以及后续实际工作边界。历史正文已恢复在原文件内，后续进展按日期追加；归档仅作备份，无需离开本文件才能阅读旧记录。

阅读顺序：[此前累计正文](#history-records) → [2026-10-23阶段更新](#update-20261023) → [本次历史恢复及生效约定](#history-repair)。旧段落的“当前”“下一步”、命令和PID按其记录时点理解；较晚的更正与实际完成证据在后文，旧命令不因此重新执行。

截至本次修复：Pilot8系列及Complex12的C/U/S各300更新、五方法60条81帧已完成。本次仅修复文档；[实验入口](../experiments/README.md)和[文档索引](../docs/DOCUMENTATION_INDEX.md)仍可直接使用。

<a id="history-records"></a>

## 一、此前累计正文：此次整理前长期保留的内容

以下完整保留原有记录顺序、参数、命令、过程和当时结论，只调整标题层级。原记录中的未完成状态与后来完成状态共同构成历史，不能用后者替换前者。

<!-- RESTORED_HISTORY_BEGIN -->
### GPT ↔ Claude Handoff

This is the durable task contract for delegated execution. GPT fills the pending section before invoking Claude and reviews the returned section afterward.

#### Pending delegation

- Status: none
- Objective:
- Why this change is needed:
- Allowed paths:
- Forbidden paths/content:
- Commands allowed:
- Training allowed: no
- Success criteria:
- Validation required:
- Logs/results to preserve:
- On failure, return:

#### Claude return

- Status: not run
- Files changed:
- Commands run:
- Validation results:
- Failures/warnings:
- Work not completed:
- Git status/diff/commit: Check the actual Git worktree; the repository is established. Current handoff is [SESSION_HANDOFF.md](../SESSION_HANDOFF.md).
- Suggested next step for GPT review:

#### GPT review

- Scope compliance:
- Diff/code review:
- Configuration review:
- Runtime/log review:
- Secret scan:
- Accepted/rejected/follow-up:
<!-- RESTORED_HISTORY_END -->

<a id="update-20261023"></a>

## 二、2026-10-23：阶段更新与当前资料补全

下面继续保留357b09f阶段补充的结果、参数和操作说明。该阶段曾把历史移到归档、部分原文页改成导航；这一文档组织做法已由第三节的用户更正取代，实验事实和既有检查结果保留。

<!-- RETAINED_UPDATE_BEGIN -->
### 会话交接：兼容入口

更新：2026-10-23（UTC+09:00）。下一次接手先读 [根SESSION_HANDOFF.md](../SESSION_HANDOFF.md)，再读 [AGENTS](../AGENTS.md)、[PROJECT_CONTEXT](../PROJECT_CONTEXT.md)、[TRAINING](../TRAINING.md)、[EXPERIMENTS](../EXPERIMENTS.md)。

当前项目目录为/mnt/windowsE/chuanjun/building_wan，Python为/home/shi/miniconda3/envs/scj/bin/python。实验按 [experiments](../experiments/README.md)归类，大结果未搬动；Complex12的13个专用脚本已移入其独立实验目录。

Pilot8与Complex12的现有GPU任务均已结束，不根据旧交接继续训练或推理。工作区仍有历史非Markdown修改，文档更新不将它们混入提交。真实文件和Git状态应在下一会话重新核对。

旧迁移交接逐字保存在 [历史快照](history/20261023_before_md_refresh/docs/HANDOFF.md)。其中旧盘符、缺环境或无Git等说明仅属历史，不覆盖当前服务器事实。
<!-- RETAINED_UPDATE_END -->

<a id="history-repair"></a>

## 三、2026-10-23：恢复原文件内的持续历史

用户明确要求长期MD保留“从以前到现在”的更新。先前仅保留当前摘要、让读者另找归档的处理不符合要求；本次将上述旧正文直接恢复到原文件，同时保留后续更新，没有回退实验结果或把完成状态改回执行中。

以后在本文件按日期追加阶段进展、失败与修复、参数变更、结论和核查范围。需要纠正旧结论时，说明原判断、后来的证据及新结论；原记录继续保留。可以在顶部维护简短状态和阅读索引，不能用摘要或归档链接替换历史正文。

本次修复依据原文件备份及Git记录恢复，没有补造新的运行经历；原数据、权重、视频、指标、冻结报告和现有非Markdown工作保持原样。此次检查与恢复范围见[文档索引](../docs/DOCUMENTATION_INDEX.md)中的后续补记。
