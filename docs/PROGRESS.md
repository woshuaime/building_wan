# 项目进度：历次记录与后续更新

最近补记：2026-10-23（UTC+08:00，北京时间）。本文件持续保留迁移期间的已完成、进行中、未完成事项，以及后来完成状态。历史正文已恢复在原文件内，后续进展按日期追加；归档仅作备份，无需离开本文件才能阅读旧记录。

阅读顺序：[此前累计正文](#history-records) → [2026-10-23阶段更新](#update-20261023) → [本次历史恢复及生效约定](#history-repair)。旧段落的“当前”“下一步”、命令和PID按其记录时点理解；较晚的更正与实际完成证据在后文，旧命令不因此重新执行。

截至本次修复：Pilot8系列及Complex12的C/U/S各300更新、五方法60条81帧已完成。本次仅修复文档；[实验入口](../experiments/README.md)和[文档索引](../docs/DOCUMENTATION_INDEX.md)仍可直接使用。

<a id="history-records"></a>

## 一、此前累计正文：此次整理前长期保留的内容

以下完整保留原有记录顺序、参数、命令、过程和当时结论，只调整标题层级。原记录中的未完成状态与后来完成状态共同构成历史，不能用后者替换前者。

<!-- RESTORED_HISTORY_BEGIN -->
### 历史迁移进度（兼容入口）

2026-10-14：当前状态统一维护在根目录 [SESSION_HANDOFF.md](../SESSION_HANDOFF.md)，操作规则见 [AGENTS.md](../AGENTS.md)，开发审计见 [PROJECT_AUDIT_20261014.md](PROJECT_AUDIT_20261014.md)。仓库及 scj 环境已建立，单轨权重和验证产物已存在；不要把下面的迁移初期快照当作当前事实或执行指令。

<details>
<summary>迁移初期历史原文（保留证据，已失效）</summary>

### Progress

#### Current phase

Migration/bootstrap on the laboratory A6000 server.

#### Completed

- Confirmed the working copy path, SSH identity, Ubuntu kernel, and RTX A6000 hardware identity.
- Confirmed this copy is not currently a Git worktree.
- Confirmed no Slurm commands, project Conda environment, Claude SDK, or MCP Python SDK are currently available in the non-interactive shell.
- Added a dependency-free project MCP worker scaffold for GPT-to-Claude delegation.
- Added project routing/safety rules and A6000-oriented context/handoff/experiment documents.
- Added secret and generated-artifact ignore rules.
- Validated the worker's MCP initialize, tool listing, and non-mutating dry-run responses without calling Claude.
- Confirmed Codex discovers the project-level `claude_worker` server as enabled.
- Confirmed the project secret scan finds no Anthropic key material.

#### In progress

- Repository origin and branch strategy are not decided.
- The final Python/CUDA environment has not been migrated or validated.
- Claude API credentials/connectivity are intentionally not live-tested yet.

#### Not completed

- No repository has been initialized or cloned.
- No dependency installation has been performed.
- No GPU framework/CUDA tensor test has been run.
- No training command or long-running process has been started.
- Copied paths and scripts have not yet been fully ported from school A100/Slurm assumptions to direct A6000 execution.

#### Next phase

1. Decide whether this directory becomes a new repository or is replaced/connected to an existing remote repository.
2. Create or migrate the A6000 Python environment and verify PyTorch/CUDA.
3. Set `ANTHROPIC_API_KEY` in the user/SSH environment (never in the project).
4. Run the worker's non-mutating live API smoke test, then let GPT review the result.
5. Audit training paths/configs and create an A6000 short-smoke-test launcher before any long training.

</details>
<!-- RESTORED_HISTORY_END -->

<a id="update-20261023"></a>

## 二、2026-10-23：阶段更新与当前资料补全

下面继续保留357b09f阶段补充的结果、参数和操作说明。该阶段曾把历史移到归档、部分原文页改成导航；这一文档组织做法已由第三节的用户更正取代，实验事实和既有检查结果保留。

<!-- RETAINED_UPDATE_BEGIN -->
### 进度查询：兼容入口

更新：2026-10-23（UTC+09:00）。项目当前摘要见 [PROJECT_CONTEXT](../PROJECT_CONTEXT.md)，跨实验完成证据见 [EXPERIMENTS](../EXPERIMENTS.md)。

Complex12 Rev1已完成C/U/S各300更新、60/60条轨迹和全部12栋媒体/指标。没有需要自动继续的GPU阶段。读取 [progress.json](../outputs/ComplexSingleOrbit_Rev1_Pilot_20261022/progress.json) 或从项目根执行：

```bash
/home/shi/miniconda3/envs/scj/bin/python experiments/vace/complex12_rev1/scripts/status.py
```

不要用resume.sh或训练/推理启动脚本查询进度。命令语义与历史实验状态读取见 [TRAINING](../TRAINING.md)。

更新前进度全文保存在 [历史快照](history/20261023_before_md_refresh/docs/PROGRESS.md)。历史轮次、PID、GPU占用和“下一步”只属于记录时刻，不能据此自动重启任务。
<!-- RETAINED_UPDATE_END -->

<a id="history-repair"></a>

## 三、2026-10-23：恢复原文件内的持续历史

用户明确要求长期MD保留“从以前到现在”的更新。先前仅保留当前摘要、让读者另找归档的处理不符合要求；本次将上述旧正文直接恢复到原文件，同时保留后续更新，没有回退实验结果或把完成状态改回执行中。

以后在本文件按日期追加阶段进展、失败与修复、参数变更、结论和核查范围。需要纠正旧结论时，说明原判断、后来的证据及新结论；原记录继续保留。可以在顶部维护简短状态和阅读索引，不能用摘要或归档链接替换历史正文。

本次修复依据原文件备份及Git记录恢复，没有补造新的运行经历；原数据、权重、视频、指标、冻结报告和现有非Markdown工作保持原样。此次检查与恢复范围见[文档索引](../docs/DOCUMENTATION_INDEX.md)中的后续补记。
