# 运行产物存储

日常请从 [实验目录](../experiments/README.md) 进入。这里保留已有成果的物理路径，模型权重、视频、图像、指标和缓存没有搬动。

| 原目录 | 所属想法 / 入口 |
| --- | --- |
| ComplexSingleOrbit_Rev1_Pilot_20261022/ | [十二栋配对实验](../experiments/vace/complex12_rev1/README.md) |
| vace/test/ | [八例Base](../experiments/vace/pilot8_base/README.md) |
| vace/composite_check/ | [CPU回贴诊断](../experiments/vace/pilot8_composite/README.md) |
| vace/lora_roof_smoke/ | [P04/P06 17帧LoRA](../experiments/vace/pilot8_lora17/README.md) |
| vace/lora300_eval81/ | [八例81帧LoRA](../experiments/vace/pilot8_lora81/README.md) |
| vace/P01_465_81f_20261019_171510_1691517/ | [P01早期完整轨迹](../experiments/vace/pilot8_base/README.md) |
| buildingfiller_handoff/ | [Pilot8数据交接](../experiments/vace/pilot8_base/README.md) |
| validation/validation_all_single_orbit/ | [历史T2V单轨](../experiments/t2v/single_orbit/README.md) |
| validation/building_wan/、validation/9.28/ | [早期T2V](../experiments/t2v/early_experiments/README.md) |

新运行按outputs/<任务>/<想法名>/<run编号>/保存。每个run内部区分00_看结果、checkpoints、predictions、metrics、logs及冻结配置，创建实际需要的部分。未来专用源码放experiments，不在本目录继续开发。

Complex12原scripts路径现在是兼容链接；原resume.sh、配置与结果仍有效。
