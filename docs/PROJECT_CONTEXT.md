# 项目上下文：迁移记录与后续更新

最近补记：2026-10-23（UTC+08:00，北京时间）。本文件持续保留早期项目上下文、架构决策和迁移假设，以及后来实际环境的更正。历史正文已恢复在原文件内，后续进展按日期追加；归档仅作备份，无需离开本文件才能阅读旧记录。

阅读顺序：[此前累计正文](#history-records) → [2026-10-23阶段更新](#update-20261023) → [本次历史恢复及生效约定](#history-repair)。旧段落的“当前”“下一步”、命令和PID按其记录时点理解；较晚的更正与实际完成证据在后文，旧命令不因此重新执行。

截至本次修复：Pilot8系列及Complex12的C/U/S各300更新、五方法60条81帧已完成。本次仅修复文档；[实验入口](../experiments/README.md)和[文档索引](../docs/DOCUMENTATION_INDEX.md)仍可直接使用。

<a id="history-records"></a>

## 一、此前累计正文：此次整理前长期保留的内容

以下完整保留原有记录顺序、参数、命令、过程和当时结论，只调整标题层级。原记录中的未完成状态与后来完成状态共同构成历史，不能用后者替换前者。

<!-- RESTORED_HISTORY_BEGIN -->
### 历史迁移上下文（兼容入口）

2026-10-14：当前状态统一维护在根目录 [PROJECT_CONTEXT.md](../PROJECT_CONTEXT.md)，操作规则见 [AGENTS.md](../AGENTS.md)，开发审计见 [PROJECT_AUDIT_20261014.md](PROJECT_AUDIT_20261014.md)。仓库及 scj 环境已建立，单轨权重和验证产物已存在；不要把下面的迁移初期快照当作当前事实或执行指令。

<details>
<summary>迁移初期历史原文（保留证据，已失效）</summary>

### Project Context

#### Stable goal

Fine-tune and evaluate Wan2.1-T2V-1.3B with LoRA so that the model learns the appearance and complete rotating views of general building models.

#### Current execution environment

- Current working copy: `/mnt/windowsE/chuanjun/building_wan`
- Current SSH host/user: `shi@vccip`
- Target GPU: NVIDIA RTX A6000 (hardware detected through PCI/driver metadata)
- Operating system: Ubuntu Linux 6.8
- Scheduler: no Slurm commands detected on this host; do not use copied school-cluster `sbatch` instructions here.
- Python environment: not migrated yet. The only currently confirmed interpreter is `/usr/bin/python3`.
- Internet/API path: the SSH session inherits the existing reverse-proxy environment. Claude API connectivity is not considered verified until a safe API smoke test succeeds.

The root-level `PROJECT_CONTEXT.md`, `TRAINING.md`, `SESSION_HANDOFF.md`, and `EXPERIMENTS.md` were copied from the former school A100/Slurm environment. They are historical migration evidence, not instructions for the A6000 host.

#### Main modules

- `code/Python_3D_Scanner`: project-owned collection, preparation, training, validation, and experiment logic.
- `code/DiffSynth-Studio`: third-party Wan training/inference engine; modify only when the project layer cannot solve the issue.
- `data/`, `models/`, `cache/`, `checkpoints/`, `outputs/`: large or generated assets; read-only by default.

#### Agent architecture

- GPT Codex is the sole main agent, planner, reviewer, risk owner, and final decision-maker.
- Claude API is an execution worker invoked only after explicit user approval.
- The project-level MCP server `claude_worker` exposes one delegation tool to Codex and runs inside the SSH workspace.
- Claude receives bounded project file and command tools. It cannot write to protected artifact directories through its file tool, cannot run destructive commands, and cannot train unless GPT explicitly enables training for that delegation.
- Every Claude execution result returns to GPT for diff/config/log/result review.

#### Architecture decisions

1. Use a small project-local stdio MCP server instead of a second chat window or a large orchestration framework.
2. Use only Python standard-library code at bootstrap time, so the integration can be validated before the project Conda environment exists.
3. Read credentials only from environment variables. Never store API keys in the repository, handoff, or logs.
4. Preserve copied school-cluster experiment records as history while creating new A6000-specific progress and experiment records here.
5. Until Git is created or cloned, report Git state as unavailable; never fabricate a branch, commit, or diff.

</details>
<!-- RESTORED_HISTORY_END -->

<a id="update-20261023"></a>

## 二、2026-10-23：阶段更新与当前资料补全

下面继续保留357b09f阶段补充的结果、参数和操作说明。该阶段曾把历史移到归档、部分原文页改成导航；这一文档组织做法已由第三节的用户更正取代，实验事实和既有检查结果保留。

<!-- RETAINED_UPDATE_BEGIN -->
### 项目状态：兼容入口

更新：2026-10-23（UTC+09:00）。当前状态统一维护在 [根目录PROJECT_CONTEXT.md](../PROJECT_CONTEXT.md)。

最新事实：Pilot8 Base、Composite、两例LoRA17、八例LoRA81均已完成；Complex12 Rev1的C/U/S各300真实更新、五方法60条81帧预测也已完成。当前A6000/scj环境和正式VACE权重已被实际使用，旧文档中的“环境尚未迁移”“模型未到位”不再描述现在。

本文件不重复维护第二套状态。数据/模型、输出位置、未解决故障和后续边界请读根文档；实际结果从 [experiments](../experiments/README.md)进入。

更新前原文逐字保存在 [历史快照](history/20261023_before_md_refresh/docs/PROJECT_CONTEXT.md)，其中时间、路径、未完成状态仅代表当时，不触发新训练或重跑。
<!-- RETAINED_UPDATE_END -->

<a id="history-repair"></a>

## 三、2026-10-23：恢复原文件内的持续历史

用户明确要求长期MD保留“从以前到现在”的更新。先前仅保留当前摘要、让读者另找归档的处理不符合要求；本次将上述旧正文直接恢复到原文件，同时保留后续更新，没有回退实验结果或把完成状态改回执行中。

以后在本文件按日期追加阶段进展、失败与修复、参数变更、结论和核查范围。需要纠正旧结论时，说明原判断、后来的证据及新结论；原记录继续保留。可以在顶部维护简短状态和阅读索引，不能用摘要或归档链接替换历史正文。

本次修复依据原文件备份及Git记录恢复，没有补造新的运行经历；原数据、权重、视频、指标、冻结报告和现有非Markdown工作保持原样。此次检查与恢复范围见[文档索引](../docs/DOCUMENTATION_INDEX.md)中的后续补记。
