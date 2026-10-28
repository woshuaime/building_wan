# 运行产物存储与结果入口

更新：2026-10-23（UTC+09:00）。日常请从 [experiments](../experiments/README.md)按研究问题进入。这里保留成果的真实物理路径；已有视频、PNG、指标、权重和冻结配置没有搬动。

## 已有目录

| 物理目录 | 内容与状态 | 对应说明 |
| --- | --- | --- |
| ComplexSingleOrbit_Rev1_Pilot_20261022/ | C/U/S各300更新，60条81帧、4860PNG、12视频/12图，全部完成 | [Complex12](../experiments/vace/complex12_rev1/README.md) |
| vace/test/ | 八例Base原始输出及对照，各四成果 | [Base](../experiments/vace/pilot8_base/README.md) |
| vace/composite_check/ | 八例CPU硬边界回贴，各三成果 | [Composite](../experiments/vace/pilot8_composite/README.md) |
| vace/lora_roof_smoke/ | 两例17帧训练、100/300权重、六次对照 | [LoRA17](../experiments/vace/pilot8_lora17/README.md) |
| vace/lora300_eval81/ | 同一LoRA-300八例81帧，32项成果 | [LoRA81](../experiments/vace/pilot8_lora81/README.md) |
| vace/P01_465_81f_20261019_171510_1691517/ | P01最早81帧结果，后续Base来源之一 | [P01说明](../docs/VACE_P01_EXECUTION.md) |
| buildingfiller_handoff/ | P01接收、解包与验收材料 | [交接说明](../docs/BUILDINGFILLER_P01_VACE_HANDOFF.md) |
| validation/validation_all_single_orbit/ | T2V Base及28权重的29比较视频 | [单轨T2V](../experiments/t2v/single_orbit/README.md) |
| validation/building_wan/、validation/9.28/ | 早期loss等结果及比较 | [早期T2V](../experiments/t2v/early_experiments/README.md) |

## 直接查看最新实验

打开 [Complex12的00_看结果](ComplexSingleOrbit_Rev1_Pilot_20261022/00_看结果/README_从这里开始.md)，可查看CS01–CS12同步视频、关键帧总览、训练/验证指标及 [中文结论](ComplexSingleOrbit_Rev1_Pilot_20261022/00_看结果/最终实验结论.md)。原始预测在predictions，科研指标在metrics，正式step300在checkpoints/C、U、S。

原run/scripts现在是指向experiments/vace/complex12_rev1/scripts的兼容链接。原resume.sh、冻结配置和完成记录未迁移。已完成任务不通过删标记或改配置强行重跑。

## 新运行的目录职责

`outputs/<任务>/<想法>/<run_id>/`内部按需使用：

| 子目录/文件 | 内容 |
| --- | --- |
| 00_看结果/ | 面向观看的README、视频、总览、简明指标和结论 |
| checkpoints/ | 正式权重与必要恢复状态 |
| predictions/ | 原始模型预测，供统一评估 |
| metrics/ | 逐帧/逐栋指标、训练/验证汇总、loss |
| logs/ | 必要执行和错误记录 |
| resolved_config.json、progress.json | 冻结运行参数与真实进度 |
| resume.sh、采样表等 | 实际需要的可恢复入口与配对依据 |

一个新研究问题先建experiments目录；同一想法不同运行用新的run_id。源码不再正式存放outputs；现存兼容链接不作为新代码存储惯例。

历史成果默认只读。文档更新不重写run中的最终报告、损失映射、日志和配置，也不删除失败记录。模型、媒体、日志、缓存默认不入Git；本README是导航文件，不改变产物保护规则。

## 2026-10-24补充：可携带报告与冻结结果的区别

日期采用UTC+09:00。用户新增 [upload/](../upload/README.md) 用于每组实验的独立详细Markdown，首份为 [Complex12报告](../upload/20261022_Complex12_Rev1_十二栋单轨实验报告.md)。可携带报告在正文复述必要方法、参数、结果与解释，方便另一台机器阅读；本目录中的原始权重、预测、指标、日志和冻结报告继续作为实验证据保留。

本次没有搬动、覆盖或复制既有大成果，也没有用新的说明去改写原run的失败记录。以后新想法的run继续按既定outputs规范隔离，收尾时额外写upload报告并更新长期MD。


## 2026-10-26：最新 OriginalHard 结果位置

直接打开[OriginalHard 的00_看结果](vace/complex12_originalhard_eval/run_20261026_012033/00_看结果/README_从这里开始.md)或[评审图册](vace/complex12_originalhard_eval/run_20261026_012033/00_看结果/评审图册.html)。物理路径：`/mnt/windowsE/chuanjun/building_wan/outputs/vace/complex12_originalhard_eval/run_20261026_012033/00_看结果/`。

这里是已完成的已有权重评测：40复用+20新推理、60条81帧、12同步视频/12总览、指标和[最终中文结论](vace/complex12_originalhard_eval/run_20261026_012033/00_看结果/最终实验结论.md)。没有新训练或新checkpoint。CS09/11明显改善，CS10/12结构仍不理想。完整可携带报告见[upload](../upload/20261026_Complex12_OriginalHard_已有权重评测报告.md)。

旧Rev1结果仍在上文原目录，未搬动或覆盖。本轮按新想法规范放在outputs/vace/complex12_originalhard_eval下，不能只在outputs第一层寻找。


## 2026-10-27：C_mix 完整评测、视觉复核与收尾

最新C_mix结果在[00_看结果](vace/complex12_cmix_pilot/run_20261026_222008/00_看结果/README_从这里开始.md)，物理路径：`/mnt/windowsE/chuanjun/building_wan/outputs/vace/complex12_cmix_pilot/run_20261026_222008/00_看结果/`。优先Hard CS10/CS12，再看Rev1 CS10新增B020反例；[中文最终报告](vace/complex12_cmix_pilot/run_20261026_222008/00_看结果/最终实验结论.md)，[便携报告](../upload/20261027_Complex12_C_mix_混合训练与配对评测报告.md)。

实际300更新（250/50）、16条81帧/1296PNG、32视频、6固定Hard图+1新增反例全部交付，metrics含逐栋配对、分组及Rev1退步明细。Hard结构未获可靠改善，Rev1 CS10明显退步；完成交付与效果达标分开记录。

这一组按既定目录规则独立放在outputs/vace/complex12_cmix_pilot/run_20261026_222008内，旧Rev1与OriginalHard仍在原处，没有覆盖或搬动。


## 2026-10-27：Orbit17线性/循环配对实验已启动

核查时间：2026-10-27T20:16:32+09:00（UTC+09:00）。用户已授权线性组和循环组都执行，并要求确认实际开始训练后结束本轮对话。当前status=running，C_lin17已完成22/600次真实AdamW更新，C_ring17为0/600（待串行执行），推理0/40；日志已有有限loss及参数实际变化。这是启动快照，不是两组完成或效果验证。

本轮新产物均在`/mnt/windowsE/chuanjun/building_wan/outputs/vace/complex12_orbit17_pair/run_20261027_200842`，源码在[Orbit17实验](../experiments/vace/complex12_orbit17_pair/README.md)。直接入口：[00_看结果](vace/complex12_orbit17_pair/run_20261027_200842/00_看结果/README_从这里开始.md)。两组计划各600更新、40预测/3240PNG及48视频；入口按真实阶段逐步生成。原有大型结果保持原位置，不复制到新run。


## 2026-10-29：Orbit17配对实验复核与最终报告完成

新增Orbit17完成入口：[详细便携报告](../upload/20261029_Complex12_Orbit17_线性与循环窗口配对实验报告.md)；[直接看结果](vace/complex12_orbit17_pair/run_20261027_200842/00_看结果/README_从这里开始.md)。

实际目录`vace/complex12_orbit17_pair/run_20261027_200842/`，两组各600，40×81帧、3240PNG、48视频；旧C复用，不重跑已成功任务。复核结论：循环组像素通常更好，闭环不普遍更好；Hard CS12主要折面改善，CS10连接仍错。大成果和旧run未搬移；这是原run最终收尾，不新建重复GPU实验。
