# 历史T2V：双3060与A100阶段

更新：2026-10-23（UTC+09:00）。这里保存历史来源和成果入口，不表示当前服务器应执行这些训练。

## 阶段与数据

| 阶段 | 历史数据/参数 | 已知结果与局限 |
| --- | --- | --- |
| Windows smoke/pilot | 20视频smoke、200视频pilot；256×256、17帧、rank8 | 训练工具与配置仍保留，具体旧产物不能仅凭文档认定存在 |
| 双RTX3060 | 1912训练视频，9560片段；256×256、17帧、rank8、lr5e-5、1 epoch | 历史记录4780更新、末loss0.02147；对应权重目录当前不在服务器 |
| A100 building_wan | building_v1训练1912，384×384、81帧、rank16、lr2e-5、8 epochs、累积1 | 历史记录15296更新、末loss0.0207207；当前保留39份权重 |

building_v1历史划分为1912/239/239，seed20260902，按source_obj分组；pilot200只从训练集合选择。它与后来的单轨1992/234/118不同，不能合并计数，也不能套用VACE的CS编号。

## 已有位置

- [A100权重](../../../checkpoints/building_wan/)保留step400–15296，共39份。
- [loss等结果](runs/building_wan/)对应outputs/validation/building_wan。
- [早期比较视频](runs/comparison_0928/)对应outputs/validation/9.28。
- [项目训练工具](../../../code/Python_3D_Scanner/training/README.md)保留smoke、pilot、双3060及A100配置。
- 完整历史记录见 [台账归档](../../../docs/history/20261023_before_structure/EXPERIMENTS.md)。

A100的/home/share/CHUANJUN/building_wan与mv2v只作为历史环境记录；当前A6000项目位于/mnt/windowsE/chuanjun/building_wan，Python为/home/shi/miniconda3/envs/scj/bin/python。

## 使用这些历史结果时

先确认具体文件、来源配置、数据身份、帧数和文本再比较。旧T2V LoRA不能叠加到VACE分支，也不作为Complex12 C/U/S初始化。最高文件步号的口径应按对应历史训练实现解释，不能直接跨实验比较。

历史smoke/pilot的Windows命令已经归档；本文不提供默认重跑指令，不自动执行Slurm、安装依赖或恢复缺失权重。当前完整补全成果从 [VACE实验总入口](../../README.md) 查看。
