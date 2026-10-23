# 项目目录约定

2026-10-23整理：按研究想法组织工作入口，运行产物保持原位置。此次没有训练、推理、重编码或改动模型计算。

## 三层关系

1. 项目公共层：scripts/的适配器、检查工具、兼容入口，以及code/中的训练基础和第三方引擎。
2. 实验层：experiments/<vace或t2v>/<想法名>/，放研究问题、专用代码和配置入口。
3. 运行层：outputs/<任务>/<想法名>/<run编号>/，放某一次实际运行的冻结配置、权重、预测、指标、日志和可视化。

已有结果采用既有路径，不为统一名字迁移大型成果。experiments中的runs链接把物理结果归到对应想法下；一个成果只保留一份。

## 以后新增实验

```text
experiments/vace/<idea>/
├── README.md
├── scripts/                 # 专用实现
├── configs/                 # 实验配置
└── runs/
    └── <run_id> -> 对应的outputs目录

outputs/vace/<idea>/<run_id>/
├── 00_看结果/
├── checkpoints/
├── predictions/
├── metrics/
├── logs/
├── resolved_config.json
└── progress.json
```

具体run只创建实际需要的目录。训练缓存、恢复状态、采样表等放在该run内；不是每个实验都必须具备所有产物。没有可用方法就写明未执行，不造空结果。

更换研究问题或损失设计时新建idea目录；相同想法的重复运行使用新run编号。数据划分、初始化、种子和预算先写清楚，再在授权范围内运行。成功结果的配置及完成记录不覆盖。

共享能力先复用项目层；只有某个实验使用的脚本放在该实验scripts/。不在outputs里继续开发正式源码。历史scripts入口保留，以兼容跨实验导入、已有测试和命令，不做本轮无关的计算逻辑重构。

## 本次具体变化

| 原位置 | 现在的位置或用途 |
| --- | --- |
| outputs/ComplexSingleOrbit_Rev1_Pilot_20261022/scripts/下13个Python脚本 | experiments/vace/complex12_rev1/scripts/ |
| 原Complex12 scripts路径 | 指向新源码目录的兼容链接，原resume.sh不变 |
| Pilot8 Base、Composite、LoRA17、LoRA81结果 | 分别从各自experiments目录的runs链接打开，物理结果不变 |
| Complex12视频、预测、指标、权重、配置 | 仍在原ComplexSingleOrbit_Rev1_Pilot_20261022目录 |
| 四份根文档的旧详细过程 | docs/history/20261023_before_structure/逐字归档 |
| 根README及四份状态文档 | 简洁的当前事实、实验索引、操作与交接 |

Complex12代码显式区分PROJECT、SCRIPTS和RUN：源码位置变化不会将输出写到experiments。当前代码服务这一已完成run；下一项想法不能直接复用该固定RUN。

## Git与历史记录

代码、实验说明、配置来源和结果链接入Git。数据、模型、视频、预测图、权重、日志、缓存与临时文件不入Git。runs目录里的链接只记录目标路径，Git不会沿链接打包成果。

已在Git中的旧run/resolved_config.json、mask_loss_mapping.md、resume.sh作为历史来源保留；它们不是未来把新源码放回outputs的理由。已有冻结配置保持唯一内容，实验configs入口指向原文件。

Linux服务器支持这些目录链接；平台不支持链接时，可用实验README所列的原始结果位置和新的源码路径。当前chuanjun.code-workspace仍只描述Windows工作区，未写入Linux绝对路径。

旧日志/配置中的实际时间、路径、PID、提交号保留原记录。它们只代表当时，不覆盖新的完成状态；当前状态统一看根目录PROJECT_CONTEXT.md及对应run/progress.json。

## 本次检查记录

- 298项项目CPU回归通过（50.447秒），8项Complex12专用CPU检查通过（2.110秒）。
- 规定shell语法、Python编译和git diff --check通过。
- 新旧两个状态入口均从项目外的工作目录正常读取complete、60/60、C/U/S各300；未启动模型。
- 20份当前导航文档的159个本地链接有效。
- 6466个已有成果文件的inode、大小及mtime与整理前一致；不重复计算大型文件哈希。
