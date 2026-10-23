# Pilot8：原始VACE八例基线

问题：原始VACE-1.3B能否恢复BuildingFiller八栋建筑的缺口？已完成P01–P08全部81帧；本目录不表示新的运行。

## 入口

| 内容 | 路径 |
| --- | --- |
| 八例正式成果 | [runs/base81](runs/base81/) |
| P01最早81帧成果 | [runs/p01_first81](runs/p01_first81/) |
| P01数据交接检查 | [runs/p01_handoff](runs/p01_handoff/) |
| 八例调度与恢复 | [vace_building_test.py](../../../scripts/vace_building_test.py) |
| 81帧公共推理 | [vace_p01_81f.py](../../../scripts/vace_p01_81f.py) |
| 数据包适配 | [buildingfiller_pilot8.py](../../../scripts/buildingfiller_pilot8.py) |
| 既有启动入口 | [run_vace_building_test.sh](../../../scripts/run_vace_building_test.sh) |
| 原17帧参数来源 | [配置](../../../configs/vace_p01_465_17f_v1.json) |

以上项目层链接均从本实验目录相对定位。正式81帧由既有入口派生，不能把原17帧配置直接当成81帧配置。输出保持在 outputs/vace/test/01–08，每例四文件。

八例对应：P01/465、P02/763、P03/871、P04/1227、P05/1259、P06/1848、P07/2297、P08/939。512×512、16FPS、30步、CFG5、seed20261018，无LoRA；合法全黑Mask帧保留，GT仅用于推理后比较。

[CPU回贴诊断](../pilot8_composite/README.md) 是另一项历史分析；[LoRA81](../pilot8_lora81/README.md)复用了这批Base结果。详细历史见 [原实验台账](../../../docs/history/20261023_before_structure/EXPERIMENTS.md)。
