# Python_3D_Scanner：现存训练工具与扫描器历史

更新：2026-10-23（UTC+09:00）。当前检出的可维护内容主要是training工具、历史配置和测试。**扫描器GUI源码src/缺失，不能按旧版说明直接运行或打包GUI。**

## 当前实际保留的内容

| 位置 | 作用与状态 |
| --- | --- |
| [training/](training/README.md) | 数据划分、片段准备、项目训练/验证入口和历史配置 |
| [tests/](tests/) | 训练工具测试以及依赖旧GUI源码的测试 |
| [requirements.txt](requirements.txt) | 历史扫描器依赖清单，不是当前scj训练环境的安装清单 |
| src/ | 当前缺失，不能运行旧main.py |
| docs/DATASET_WORKFLOW.md、docs/workflow.md | 当前缺失，旧链接不再作为可用说明入口 |

项目总体状态见 [根README](../../README.md)、[PROJECT_CONTEXT](../../PROJECT_CONTEXT.md)；当前BuildingFiller条件补全专用实验见 [experiments](../../experiments/README.md)。Complex12专用源码已位于experiments/vace/complex12_rev1/scripts，不在本目录。

## 历史扫描器设计

旧说明描述PySide6、PyVista、VTK的桌面OBJ导入与离屏渲染工具，支持四条轨道、相机位置导出和多视角图像采集。这些是原始设计记录，不是本次验证过的可运行能力。

历史数据生成强调同一建筑的几何、相机和渲染参数一致，再生成GT/Damaged/Exact配对。当前VACE实验使用已经交接、验证的BuildingFiller数据包，不依赖从缺失GUI重新生成数据，也不通过文档推测重造原始采集实现。

原版详细参数、四轨道设计、文件格式与数据准则完整保留于 [更新前README](../../docs/history/20261023_before_md_refresh/code/Python_3D_Scanner/README.md)。归档中的安装、运行与打包命令仅用于历史追溯。

## 当前环境与边界

服务器项目：/mnt/windowsE/chuanjun/building_wan。训练解释器：/home/shi/miniconda3/envs/scj/bin/python；PyTorch2.11.0+cu128，单卡A6000。不要为阅读此目录安装GUI依赖或覆盖已有训练环境。

若后续明确要恢复扫描器，应先从原始来源取回src及工作流，再单独验证OBJ导入、相机、渲染、输出和测试。当前任务没有完成该恢复，也没有运行依赖缺失src的GUI测试。

## 维护入口

- [training说明](training/README.md)：现存脚本的职责、历史数据划分与运行注意事项。
- [TRAINING](../../TRAINING.md)：当前服务器操作、状态读取与CPU检查。
- [实验台账](../../EXPERIMENTS.md)：区分T2V历史和VACE已完成实验。
- [目录规则](../../docs/PROJECT_STRUCTURE.md)：新想法进入experiments，成果保留在独立run。
