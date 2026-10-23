# BuildingFiller 建筑补全研究

日常从 [实验目录](experiments/README.md) 进入：一个研究想法一个文件夹，分别查看代码、配置、运行记录和结果。

最新完成的是 [Complex12 Rev1 十二栋单轨小试](experiments/vace/complex12_rev1/README.md)：C/U/S 各300次更新，五方法共60条81帧轨迹、4860张原始预测、12条同步视频及12张总览。当前没有待续跑的实验。

[直接看十二栋结果](outputs/ComplexSingleOrbit_Rev1_Pilot_20261022/00_看结果/README_从这里开始.md) · [中文结论](outputs/ComplexSingleOrbit_Rev1_Pilot_20261022/00_看结果/最终实验结论.md)

普通 LoRA 在验证集明显优于 Base；结构加权 S 相比均匀加权 U 的平均收益较小，仍有屋檐弯折、保留区变化和首尾不一致。详细结论以报告和逐栋结果为准。

## 目录职责

| 目录 | 放什么 |
| --- | --- |
| [experiments/](experiments/README.md) | 各研究想法的入口、专用代码、配置入口及结果链接 |
| [scripts/](scripts/README.md) | 已有公共工具、共享适配器、兼容启动入口和CPU回归 |
| code/Python_3D_Scanner/ | 历史训练入口及数据处理工具 |
| code/DiffSynth-Studio/ | 第三方引擎 |
| [configs/](configs/README.md) | 已有冻结配置与历史数据划分 |
| data/、models/ | 原始数据和基础模型 |
| cache/、checkpoints/ | 历史公共缓存和权重，既有路径保留 |
| [outputs/](outputs/README.md) | 运行产物的物理存储；大文件保留原位置 |
| logs/ | 运行日志及故障记录 |
| [docs/](docs/README.md) | 专题说明与历史文档 |

以后新想法在 experiments/vace/ 或 experiments/t2v/ 下建目录；同一想法多次运行分别保存 run，具体规范见 [目录约定](docs/PROJECT_STRUCTURE.md)。既有结果、权重、数据及冻结配置没有重新生成或搬家。

## 项目文档

- [当前状态](PROJECT_CONTEXT.md)：已经完成什么、还存在哪些限制。
- [实验台账](EXPERIMENTS.md)：实验之间的关系及结果入口。
- [操作手册](TRAINING.md)：只读查看、CPU检查和明确授权后的运行入口。
- [交接说明](SESSION_HANDOFF.md)：接手顺序和兼容路径。
- [项目规则](AGENTS.md)：训练授权、数据保护和Git约定。

当前服务器为单卡 A6000，Python 位于 /home/shi/miniconda3/envs/scj/bin/python。扫描器历史README描述的GUI源目录目前缺失，不代表桌面采集程序可直接运行。
