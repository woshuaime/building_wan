# BuildingFiller 建筑序列补全研究

文档核对日期：2026-10-23（UTC+09:00）。本仓库当前研究 **给定残缺建筑、Exact缺损位置和文本时，VACE能否恢复完整建筑序列**。历史Wan T2V实验学习外观与环绕生成，单独保留。

最新Complex12 Rev1实验已经完成：C/U/S各300次真实优化器更新；B/L/C/U/S共60条81帧轨迹、4860张原始预测、12条同步视频及12张总览。当前文档维护不启动训练或推理。

## 从哪里开始

| 目的 | 入口 |
| --- | --- |
| 直接观看最新十二栋结果 | [视频、总览、指标和结论](outputs/ComplexSingleOrbit_Rev1_Pilot_20261022/00_看结果/README_从这里开始.md) |
| 了解各项研究想法的关系 | [实验总目录](experiments/README.md) |
| 核对当前完成状态和遗留问题 | [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) |
| 查看参数、方法与实验结论 | [EXPERIMENTS.md](EXPERIMENTS.md) |
| 查状态、使用CPU检查和定位故障 | [TRAINING.md](TRAINING.md) |
| 接手工作或新建实验 | [SESSION_HANDOFF.md](SESSION_HANDOFF.md)、[目录约定](docs/PROJECT_STRUCTURE.md) |
| 查所有项目文档及历史范围 | [文档索引](docs/DOCUMENTATION_INDEX.md) |

## 当前结果说明什么

普通LoRA C在Complex12验证集的缺损区MAE比Base降低64.21%，四栋验证建筑均有改善。S的平均缺损/结构指标稍好于U，但验证MAE仅降低1.80%、结构Sobel仅降低0.19%，且主要由CS12贡献；部分屋檐与端墙弯折仍在，保留区变化和全图闭环误差增加。

这支持建筑补全LoRA在本数据分布上的迁移；尚不能认定结构加权稳定胜出，不能推断完整三维恢复或高频闪烁已经消除。完整口径、反例和训练/验证差异见 [中文报告](outputs/ComplexSingleOrbit_Rev1_Pilot_20261022/00_看结果/最终实验结论.md)。

## 项目内的实验关系

1. **Pilot8 Base**：八栋原始VACE-1.3B完整81帧补全。
2. **CPU Composite**：只把Base预测贴入Exact，诊断已知区域变化；不是新的模型预测。
3. **LoRA17**：P04/P06各17帧、300更新，验证训练链路与训练题恢复。
4. **LoRA81**：同一LoRA-300扩展到八例81帧，不新增训练。
5. **Complex12 Rev1**：独立8栋训练/4栋验证，比较普通、均匀缺损、结构带加权；旧LoRA仅作L参考。
6. **历史T2V**：双3060、A100和A6000单轨实验；不混用其初始化、划分或缓存。

## 代码、实验与结果如何分开

| 目录 | 职责 | 维护原则 |
| --- | --- | --- |
| [experiments/](experiments/README.md) | 一个想法一夹，专用源码、配置入口、runs链接 | 新想法不覆盖旧run |
| [scripts/](scripts/README.md) | 公共适配器、兼容入口、服务器工具及CPU测试 | 已有跨实验导入路径保留 |
| code/Python_3D_Scanner/ | 项目训练入口和历史数据工具 | 当前GUI源码缺失，见该目录README |
| code/DiffSynth-Studio/ | 第三方引擎 | 优先在项目层实现变化 |
| [configs/](configs/README.md) | 既有冻结配置与划分 | 历史配置不随新想法改写 |
| data/、models/ | 原始包、数据与基础模型 | 默认只读 |
| cache/、checkpoints/ | 历史共享缓存及权重 | 新实验不能混用未确认来源 |
| [outputs/](outputs/README.md) | 视频、预测、指标、实验权重、运行记录 | 保持已有物理路径 |
| logs/ | 运行和诊断记录 | 原生异常记录保留 |
| [docs/](docs/README.md) | 专题说明、文档索引及历史快照 | 当前事实与历史证据分开 |

Complex12的13个专用Python脚本位于experiments/vace/complex12_rev1/scripts/；原run/scripts保留兼容链接。runs链接只导航，Git不会沿链接打包大型成果。

## 数据与模型边界

训练：同帧GT作为FlowMatch监督目标；Damaged白区填灰127、二值Exact转ModelMask、文本作为VACE条件。推理：只读Damaged、Exact和文本，GT与结构图不进入条件。评估：模型原始预测相对GT比较，已知区域相对Damaged比较，真实预测000/080单独计闭环。

Exact是配对数据提供的缺损定位，本项目尚未评估自动检测缺口。Complex12的000–079是80个独立视角，080输入重复首帧；预测080保留真实模型输出，不复制000。

## 当前环境与完成标准

服务器项目根为/mnt/windowsE/chuanjun/building_wan；解释器为/home/shi/miniconda3/envs/scj/bin/python。实际环境为Python3.10.13、PyTorch2.11.0+cu128、CUDA runtime12.8、DiffSynth2.1.5、PEFT0.17.1，单卡A6000。旧A100的mv2v和Windows采集环境只属于历史记录。

完成必须同时有真实更新/退出证据、可读预测、统一评测与视觉结论；启动脚本、保存权重或loss下降都不能单独代替完成。原生崩溃根因仍未确认。研究建议不构成自动启动下一轮实验的授权，具体规则见 [AGENTS.md](AGENTS.md)。
