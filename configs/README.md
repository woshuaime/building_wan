# 已有冻结配置与划分

| 文件/目录 | 用途 |
| --- | --- |
| [vace_p01_465_17f_v1.json](vace_p01_465_17f_v1.json) | 早期P01 17帧参数来源；81帧由对应入口显式派生 |
| [vace_p01_model_sources_v1.json](vace_p01_model_sources_v1.json) | VACE基础模型与编码组件来源 |
| [vace_roof_smoke_v1.json](vace_roof_smoke_v1.json) | P04/P06各17帧、300更新的冻结配置 |
| [single_orbit_same_prompt_v1/](single_orbit_same_prompt_v1/) | 历史T2V的1992/234/118划分 |

Complex12配置仍在原run的resolved_config.json，从 [实验configs入口](../experiments/vace/complex12_rev1/configs/pilot_20261022.json) 查看。

新想法的配置放在对应experiments/<任务>/<想法>/configs/，实际运行另保存resolved_config。历史配置不随新想法覆盖。
