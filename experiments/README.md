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
