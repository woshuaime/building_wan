# 按研究想法查看实验

每个目录回答一个明确问题。代码和说明放在实验目录；runs/中的链接打开原有运行产物，不复制视频、权重或缓存。

## VACE建筑补全

| 文件夹 | 研究内容 | 状态 | 直接看结果 |
| --- | --- | --- | --- |
| [vace/pilot8_base](vace/pilot8_base/README.md) | 八例原始Base补全 | 完成 | [八例结果](vace/pilot8_base/runs/base81/) |
| [vace/pilot8_composite](vace/pilot8_composite/README.md) | CPU保留区回贴诊断 | 历史完成 | [回贴结果](vace/pilot8_composite/runs/cpu_check/) |
| [vace/pilot8_lora17](vace/pilot8_lora17/README.md) | P04/P06两例17帧LoRA训练 | 完成 | [训练与比较](vace/pilot8_lora17/runs/20261020/) |
| [vace/pilot8_lora81](vace/pilot8_lora81/README.md) | 旧LoRA-300八例81帧扩展评估 | 完成 | [八例结果](vace/pilot8_lora81/runs/20261020/) |
| [vace/complex12_rev1](vace/complex12_rev1/README.md) | 十二栋C/U/S配对实验 | 全部完成 | [视频、指标和结论](vace/complex12_rev1/runs/20261022/00_看结果/README_从这里开始.md) |

## 历史T2V生成

| 文件夹 | 内容 | 状态 |
| --- | --- | --- |
| [t2v/single_orbit](t2v/single_orbit/README.md) | A6000统一prompt单轨训练及29个比较视频 | 结果保留；完整训练终态待核实 |
| [t2v/early_experiments](t2v/early_experiments/README.md) | 双3060、A100阶段 | 历史记录，独立环境 |

公共实现继续从 [scripts](../scripts/README.md) 查找；Complex12专用源码已实际归到自己的scripts/目录。

## 增加新的想法

从 [_template/README.md](_template/README.md) 复制说明，先写清研究问题、对照、数据划分、预算和结果位置。新想法使用新目录；同一想法的不同运行分别放run，不覆盖成功结果。目录规则见 [PROJECT_STRUCTURE.md](../docs/PROJECT_STRUCTURE.md)。

runs/链接供浏览；不存在的新run不会预先伪造。Linux服务器可以直接打开，其他检出环境也可通过各实验README中的原结果路径定位。
