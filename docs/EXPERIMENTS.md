# 实验记录：早期模板与后续更新

最近补记：2026-10-23（UTC+08:00，北京时间）。本文件持续保留早期实验组织模板、迁移记录和后来完成实验的衔接。历史正文已恢复在原文件内，后续进展按日期追加；归档仅作备份，无需离开本文件才能阅读旧记录。

阅读顺序：[此前累计正文](#history-records) → [2026-10-23阶段更新](#update-20261023) → [本次历史恢复及生效约定](#history-repair)。旧段落的“当前”“下一步”、命令和PID按其记录时点理解；较晚的更正与实际完成证据在后文，旧命令不因此重新执行。

截至本次修复：Pilot8系列及Complex12的C/U/S各300更新、五方法60条81帧已完成。本次仅修复文档；[实验入口](../experiments/README.md)和[文档索引](../docs/DOCUMENTATION_INDEX.md)仍可直接使用。

<a id="history-records"></a>

## 一、此前累计正文：此次整理前长期保留的内容

以下完整保留原有记录顺序、参数、命令、过程和当时结论，只调整标题层级。原记录中的未完成状态与后来完成状态共同构成历史，不能用后者替换前者。

<!-- RESTORED_HISTORY_BEGIN -->
### 历史迁移实验模板（兼容入口）

2026-10-14：当前状态统一维护在根目录 [EXPERIMENTS.md](../EXPERIMENTS.md)，操作规则见 [AGENTS.md](../AGENTS.md)，开发审计见 [PROJECT_AUDIT_20261014.md](PROJECT_AUDIT_20261014.md)。仓库及 scj 环境已建立，单轨权重和验证产物已存在；不要把下面的迁移初期快照当作当前事实或执行指令。

<details>
<summary>迁移初期历史原文（保留证据，已失效）</summary>

### Experiments

#### Migration note

The root-level `EXPERIMENTS.md` records experiments performed on the former school A100/Slurm cluster. Preserve it as historical evidence. Do not relabel those runs as A6000 results.

No laboratory A6000 training experiment has been run from this working copy yet.

#### A6000 experiment template

- Date/time:
- Objective:
- Host/GPU:
- Git commit or explicit "Git unavailable":
- Python/PyTorch/CUDA:
- Model/config:
- Dataset/split:
- Batch size / accumulation:
- Learning rate / schedule:
- Input checkpoint:
- Output path:
- Log path:
- Short smoke test result:
- Peak GPU memory:
- Runtime:
- Loss summary:
- NaN: yes/no/not checked
- OOM: yes/no/not checked
- Checkpoint result:
- Final status:
- Failure cause:
- Useful observations:

</details>
<!-- RESTORED_HISTORY_END -->

<a id="update-20261023"></a>

## 二、2026-10-23：阶段更新与当前资料补全

下面继续保留357b09f阶段补充的结果、参数和操作说明。该阶段曾把历史移到归档、部分原文页改成导航；这一文档组织做法已由第三节的用户更正取代，实验事实和既有检查结果保留。

<!-- RETAINED_UPDATE_BEGIN -->
### 实验台账：兼容入口

更新：2026-10-23（UTC+09:00）。完整当前台账统一维护在 [根目录EXPERIMENTS.md](../EXPERIMENTS.md)。

当前主要链条为Pilot8 Base → CPU Composite诊断 → P04/P06 LoRA17 → 八例LoRA81 → Complex12 Rev1配对实验。各项已经完成；Composite没有训练，Complex12 C/U/S没有从旧L继续训练。

T2V历史与VACE条件补全分开记录：单轨28权重/29视频保留，但完整8 epochs终态仍待核实。不要将这一历史限制误套到已有300真实更新证据的Complex12。

按想法查看 [experiments目录](../experiments/README.md)。本文件旧Windows/迁移时期台账逐字保存在 [历史快照](history/20261023_before_md_refresh/docs/EXPERIMENTS.md)，不再与当前结果混排。
<!-- RETAINED_UPDATE_END -->

<a id="history-repair"></a>

## 三、2026-10-23：恢复原文件内的持续历史

用户明确要求长期MD保留“从以前到现在”的更新。先前仅保留当前摘要、让读者另找归档的处理不符合要求；本次将上述旧正文直接恢复到原文件，同时保留后续更新，没有回退实验结果或把完成状态改回执行中。

以后在本文件按日期追加阶段进展、失败与修复、参数变更、结论和核查范围。需要纠正旧结论时，说明原判断、后来的证据及新结论；原记录继续保留。可以在顶部维护简短状态和阅读索引，不能用摘要或归档链接替换历史正文。

本次修复依据原文件备份及Git记录恢复，没有补造新的运行经历；原数据、权重、视频、指标、冻结报告和现有非Markdown工作保持原样。此次检查与恢复范围见[文档索引](../docs/DOCUMENTATION_INDEX.md)中的后续补记。
