# 实验台账

更新日期：2026-10-23（UTC+09:00）。完整实验入口见 [experiments/README.md](experiments/README.md)。

| 实验 | 研究问题 | 已完成结果 / 状态 |
| --- | --- | --- |
| [Pilot8 Base](experiments/vace/pilot8_base/README.md) | 原始VACE能否补全八栋建筑 | 八例81帧基线与四项可视化 |
| [Pilot8 Composite](experiments/vace/pilot8_composite/README.md) | 保留输入已知区域能改善多少 | CPU硬边界回贴诊断；不是模型原始输出，也没有新训练 |
| [P04/P06 LoRA17](experiments/vace/pilot8_lora17/README.md) | 两个训练题能否学会表面补全 | 300更新，Base/LoRA100/LoRA300共6次17帧比较 |
| [Pilot8 LoRA81](experiments/vace/pilot8_lora81/README.md) | 17帧收益能否扩展到整圈和未见建筑 | 八例81帧，32项成果；复用Base |
| [Complex12 Rev1](experiments/vace/complex12_rev1/README.md) | 普通、均匀缺损、结构带加权的配对比较 | C/U/S各300更新；五方法60条完整轨迹；已完成 |
| [T2V单轨历史](experiments/t2v/single_orbit/README.md) | 统一文本下的灰色建筑与环绕生成 | 保留28权重、29对比视频；完整训练终态待核实 |
| [T2V早期历史](experiments/t2v/early_experiments/README.md) | 双3060、A100阶段 | 独立历史环境及权重，不能混入VACE |

## Complex12 Rev1 最终结论

8栋训练、4栋验证分开统计；每栋000–079按有效像素汇总，跨栋等权，000/080另计闭环。五种方法为 B=Base、L=旧LoRA参考、C=普通、U=均匀缺损加权、S=结构区域加权。

| 验证集方法 | 缺损MAE↓ | 结构Sobel↓ | 保留区变化↓ | 全图闭环↓ |
| --- | ---: | ---: | ---: | ---: |
| B | 0.031004 | 0.018646 | 0.002426 | 0.003007 |
| L | 0.013269 | 0.014869 | 0.001737 | 0.002736 |
| C | 0.011097 | 0.013642 | 0.001844 | 0.002973 |
| U | 0.010417 | 0.013313 | 0.001914 | 0.003521 |
| S | 0.010229 | 0.013288 | 0.001924 | 0.003579 |

C相对Base改善明确。S缺损均值略领先U，验证优势集中于CS12；细部屋檐与跨屋脊残留没有稳定消除，保留区和全图闭环存在代价。12栋观察、所有指标与局限见 [最终报告](outputs/ComplexSingleOrbit_Rev1_Pilot_20261022/00_看结果/最终实验结论.md)。

C/U/S实际优化器步数均300；3份step300、60条81帧预测、12视频、12总览真实可读。B/CS04保存后原生异常的有效输出经独立复核保留；C在服务器恢复后从40更新续到300。历史异常未被隐藏或认定根治。

## 结构整理记录

2026-10-23将Complex12的13个专用脚本迁出结果目录，建立按想法分类的实验入口与结果链接。原视频、图像、指标、权重、数据、缓存和冻结配置保持原位置与内容；本次没有GPU实验。

全部旧台账逐字归档于 [整理前EXPERIMENTS.md](docs/history/20261023_before_structure/EXPERIMENTS.md)，包括早期失败、CPU诊断、授权重跑和硬件实验。旧过程记录不再与当前完成状态混排。
