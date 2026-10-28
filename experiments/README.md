# 按研究想法查看实验

更新：2026-10-23（UTC+09:00）。每个目录回答一个明确研究问题，保存专用代码、配置入口和结果说明。runs/是原有产物的目录链接，视频、图片、权重和缓存保持原位置。

## 1. 最新成果

[Complex12 Rev1结果入口](vace/complex12_rev1/runs/20261022/00_看结果/README_从这里开始.md)包含12栋建筑的视频、关键帧、训练/验证指标及中文结论。C/U/S各300更新和五方法60条完整轨迹已完成，没有待自动续跑任务。

## 2. VACE建筑补全实验

| 实验目录 | 研究内容 | 实际完成量 | 直接看结果 |
| --- | --- | --- | --- |
| [pilot8_base](vace/pilot8_base/README.md) | 原始VACE八例基线 | 八例81帧，各四成果 | [Base](vace/pilot8_base/runs/base81/) |
| [pilot8_composite](vace/pilot8_composite/README.md) | 仅缺口回贴的CPU诊断 | 八例各三成果，无新增推理 | [Composite](vace/pilot8_composite/runs/cpu_check/) |
| [pilot8_lora17](vace/pilot8_lora17/README.md) | P04/P06两个固定训练题 | 300更新，100/300两份LoRA，六次17帧推理 | [训练与比较](vace/pilot8_lora17/runs/20261020/) |
| [pilot8_lora81](vace/pilot8_lora81/README.md) | 旧LoRA-300整圈与迁移 | 八例81帧，32项成果；Base复用 | [LoRA81](vace/pilot8_lora81/runs/20261020/) |
| [complex12_rev1](vace/complex12_rev1/README.md) | 普通/均匀/结构加权配对比较 | 三组各300更新，60轨迹、4860PNG、12视频、12总览 | [结果与结论](vace/complex12_rev1/runs/20261022/00_看结果/README_从这里开始.md) |

实验关系：Base暴露凹槽/黑洞 → Composite确认仅回贴不足 → 两例LoRA17验证可学习 → LoRA81检验整圈和迁移 → Complex12在独立数据上比较C/U/S。箭头描述历史研究顺序，不表示自动执行下一阶段。

L是旧两例LoRA参考；Complex12的C/U/S是共同新初始化，不能说它们由L继续训练。P04/P06是Pilot8编号，不是Complex12的CS04/CS06。

## 3. 历史T2V生成

| 实验目录 | 内容 | 保留证据与限制 |
| --- | --- | --- |
| [single_orbit](t2v/single_orbit/README.md) | A6000统一prompt单轨 | 1992/234/118划分、1992缓存、28权重和29比较视频；8 epochs终态待核实 |
| [early_experiments](t2v/early_experiments/README.md) | Windows、双3060、A100 | A10039份权重；双3060权重目录当前缺失，历史环境独立 |

T2V学习外观与环绕生成，VACE学习Damaged/Mask条件补全；不共享LoRA、训练目标或文本缓存。公共组件从 [scripts](../scripts/README.md) 和 [训练工具](../code/Python_3D_Scanner/training/README.md)查找。

## 4. 新想法与新运行

从 [_template](_template/README.md)建立说明：先写问题、对照、数据身份、变化变量、预算和验证口径。不同研究问题进入新的idea目录；同一想法不同运行使用新的run_id。

`experiments/<vace或t2v>/<idea>/`保存专用源码、配置和说明；`outputs/<任务>/<idea>/<run_id>/`保存这一次生成的权重、预测、指标、日志及00_看结果。只创建实际需要的内容，不创建虚假完成目录或复制大结果。具体见 [目录规范](../docs/PROJECT_STRUCTURE.md)。

已有特殊命名目录保持不动。Linux可通过runs链接查看；其他环境不支持链接时，查各实验README记录的原始物理路径。GitHub仓库默认不含模型和媒体，链接不意味着这些成果上传到了GitHub。

## 5. 状态和文档

全局当前状态见 [PROJECT_CONTEXT](../PROJECT_CONTEXT.md)，跨实验结论见 [EXPERIMENTS](../EXPERIMENTS.md)，操作见 [TRAINING](../TRAINING.md)，接手见 [SESSION_HANDOFF](../SESSION_HANDOFF.md)。本次Markdown更新范围和检查记录见 [文档索引](../docs/DOCUMENTATION_INDEX.md)。

## 2026-10-24：每组实验的独立报告

日期采用UTC+09:00。研究入口仍由本目录按想法组织，新增的 [upload/](../upload/README.md) 专门保存可复制到其他机器阅读的详细实验报告。

[Complex12 Rev1完整报告](../upload/20261022_Complex12_Rev1_十二栋单轨实验报告.md) 已整理完成，写清B/L/C/U/S的区别、C/U/S各300真实更新、60条完整轨迹、异常恢复和训练/验证结果。实验本身没有重跑，结果仍在原run。

以后每组实验收尾时，除了更新本目录对应README和根长期MD，还需新增对应upload报告并登记索引；较早实验仍保留各自原入口，不能用新报告索引删去原研究过程。格式见 [upload要求](../upload/README.md)。


## 2026-10-26：OriginalHard 已有权重评测（执行中）

[新实验入口](vace/complex12_originalhard_eval/README.md)使用已训练的B/L/C/U/S、完整81帧；本轮训练更新0。CS01–CS08输入完全一致，40条预测复用；CS09–CS12新缺损需20条推理。run为 `/mnt/windowsE/chuanjun/building_wan/outputs/vace/complex12_originalhard_eval/run_20261026_012033`，最终仍提供十二栋五方法同步对比。当前运行快照见根PROJECT_CONTEXT，完成结论待实际复核后追加。


## 2026-10-26：OriginalHard 完成与复核补记

此前“执行中”是启动时快照，现已完成40条复用+20条新推理、五方法60条81帧、12视频/12总览、指标与中文报告，新训练0。入口为[OriginalHard实验](vace/complex12_originalhard_eval/README.md)，[完整报告](../upload/20261026_Complex12_OriginalHard_已有权重评测报告.md)。

新缺损CS09/11明显改善，CS10/12连接形状仍不可靠；S没有稳定优于C/U。四栋中3栋旧训练、1栋旧验证，完整8/4身份汇总另列。下一步仅建议困难结构缺损覆盖对照，未启动。


## 2026-10-26：C_mix 固定配比实验启动与后续交接

[C_mix实验](vace/complex12_cmix_pilot/README.md)已获授权，固定300更新、250/50混合及16条新推理。2026-10-26T22:26:41+08:00真实更新33/300；run为`/mnt/windowsE/chuanjun/building_wan/outputs/vace/complex12_cmix_pilot/run_20261026_222008`。计算和媒体自动执行，报告/视觉复核依用户要求等待通知，尚无效果结论。


## 2026-10-27：C_mix 完整评测、视觉复核与收尾

[C_mix实验](vace/complex12_cmix_pilot/README.md)已完成：300更新（250/50）、16条81帧、1296原PNG、32视频、6固定图+1反例，完整静态轨迹复核与[便携报告](../upload/20261027_Complex12_C_mix_混合训练与配对评测报告.md)齐全。实际位置`/mnt/windowsE/chuanjun/building_wan/outputs/vace/complex12_cmix_pilot/run_20261026_222008`，直接打开[看结果](../outputs/vace/complex12_cmix_pilot/run_20261026_222008/00_看结果/README_从这里开始.md)。

Hard CS10连接、CS12形状/闭环未解决；Rev1有10/12栋MAE退步，CS10条纹最明显，当前保留旧C参考。建议中的纯Rev1复现对照未执行；不再自动展开新实验。


## 2026-10-27：Orbit17线性/循环配对实验已启动

核查时间：2026-10-27T20:16:32+09:00（UTC+09:00）。用户已授权线性组和循环组都执行，并要求确认实际开始训练后结束本轮对话。当前status=running，C_lin17已完成22/600次真实AdamW更新，C_ring17为0/600（待串行执行），推理0/40；日志已有有限loss及参数实际变化。这是启动快照，不是两组完成或效果验证。

新入口：[Orbit17线性/循环配对](vace/complex12_orbit17_pair/README.md)，实际run为`/mnt/windowsE/chuanjun/building_wan/outputs/vace/complex12_orbit17_pair/run_20261027_200842`。

两组普通C各600更新，Rev1训练、Hard只评测，300各4探针、600各16输入，共40轨迹/3240PNG。队列包含两组训练、指标及48视频；后续人工视觉复核与独立upload报告等待用户通知。旧实验结果不搬动。


## 2026-10-29：Orbit17配对实验复核与最终报告完成

Orbit17线性/循环配对实验已完成：两组各600、40条81帧/3240原PNG、48视频，静态完整帧复核与最终报告齐全。[详细便携报告](../upload/20261029_Complex12_Orbit17_线性与循环窗口配对实验报告.md)；[直接看结果](../outputs/vace/complex12_orbit17_pair/run_20261027_200842/00_看结果/README_从这里开始.md)。

ring600同预算MAE 13/16更低，但全图首尾15/16更差；CS12主要折面两组600均改善，CS10连接未解决。不是稳定闭环成功结论。源码仍在vace/complex12_orbit17_pair/scripts，大结果未移动；启动快照继续保留。
