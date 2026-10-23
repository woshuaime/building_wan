# 项目目录与文档维护约定

更新：2026-10-23（UTC+09:00）。按研究想法组织工作入口，已有大型成果保持物理位置。目录整理与本次Markdown维护均未启动训练、推理或媒体重编码。

## 1. 公共代码、实验和运行产物

| 层次 | 位置 | 职责 |
| --- | --- | --- |
| 公共层 | scripts/、code/Python_3D_Scanner/training/ | 已有适配器、通用工具、兼容入口、回归 |
| 第三方引擎 | code/DiffSynth-Studio/ | 供应商源码，项目层能解决的问题不改其核心 |
| 实验层 | experiments/<任务>/<想法>/ | 一个研究问题的说明、专用实现、配置入口、runs链接 |
| 运行层 | outputs/<任务>/<想法>/<run_id>/ | 一次实际运行的权重、预测、指标、日志、可视化与冻结参数 |
| 公共数据与模型 | data/、models/、cache/、历史checkpoints/ | 保留原始来源和已有大文件，不因归类搬动 |

实验代表研究问题，run代表一次执行。更换损失设计或研究问题新建idea；相同想法的重复实验新建run。多个方法可以作为同一配对实验的子组，但数据、初始化、预算和评测关系应写清楚。

## 2. 新实验建议结构

```text
experiments/vace/<idea>/
├── README.md
├── scripts/
├── configs/
└── runs/
    └── <run_id> -> 对应outputs目录

outputs/vace/<idea>/<run_id>/
├── 00_看结果/
├── checkpoints/
├── predictions/
├── metrics/
├── logs/
├── resolved_config.json
└── progress.json
```

只创建实际需要的内容，必要恢复状态、采样表、缓存和resume入口归到对应run。没有执行的方法写明未执行，不制造空预测、假权重或完成状态。模板见 [实验模板](../experiments/_template/README.md)。

专用源码在experiments维护，公共能力从scripts/code复用。现有公共入口不为目录整洁而随意迁移；跨实验导入和已有测试应保持可用。

## 3. 已有目录如何归类

| 原位置 | 当前导航/维护位置 | 处理 |
| --- | --- | --- |
| outputs/vace/test | experiments/vace/pilot8_base/runs/base81 | 链接；结果不搬 |
| outputs/vace/composite_check | experiments/vace/pilot8_composite/runs/cpu_check | 链接；标明CPU回贴 |
| outputs/vace/lora_roof_smoke/run_20261020_165237_150325_2048720 | experiments/vace/pilot8_lora17/runs/20261020 | 链接 |
| outputs/vace/lora300_eval81/run_20261020_194300_145372 | experiments/vace/pilot8_lora81/runs/20261020 | 链接 |
| outputs/ComplexSingleOrbit_Rev1_Pilot_20261022 | experiments/vace/complex12_rev1/runs/20261022 | 链接 |
| Complex12原scripts下13个Python | experiments/vace/complex12_rev1/scripts | 源码已迁移；原scripts保留兼容链接 |
| Complex12原resolved_config.json | experiments/vace/complex12_rev1/configs/pilot_20261022.json | 配置链接，唯一内容仍在原run |
| outputs/validation下历史结果 | experiments/t2v各实验runs | 链接 |

Complex12代码显式区分PROJECT、SCRIPTS、RUN，原resume.sh仍可定位源码与输出。当前RUN固定服务已完成实验，新想法不能直接复用其输出目录、配置或完成标记。

## 4. 文档分工与优先级

| 文档 | 负责内容 |
| --- | --- |
| 根README.md | 项目目的和阅读入口 |
| 根PROJECT_CONTEXT.md | 当前事实、环境和未解决事项 |
| 根EXPERIMENTS.md | 实验台账、横向结果和结论边界 |
| 根TRAINING.md | 服务器操作、状态读取、恢复和检查 |
| 根SESSION_HANDOFF.md | 下一会话需要知道的事实 |
| AGENTS.md | 项目工作规则与授权边界 |
| experiments各README | 该想法的数据、配置、代码、结果和局限 |
| docs专项说明 | 数据交接、技术细节、历史故障和工具 |
| docs/DOCUMENTATION_INDEX.md | 文档全量清单、更新范围与检查 |
| run内报告/日志 | 该次执行的冻结证据，后续文档不倒改 |

docs旧PROJECT_CONTEXT/EXPERIMENTS/HANDOFF/PROGRESS现在只作兼容导航，不再维护第二套“当前状态”。若旧日志或归档与当前总览冲突，应按记录时间和适用实验解释；不能删去旧失败记录来制造一致。

## 5. 历史保护与Git

原始数据、模型、权重、视频、预测图、日志和缓存默认只读且不入Git。runs链接仅记录目标路径，Git不会沿链接打包成果。其他平台不支持链接时，从实验README的原路径定位；GitHub上无法读取服务器媒体属于部署差异。

chuanjun.code-workspace只描述Windows工作区，不加入服务器绝对路径。已跟踪的旧run配置、损失映射和resume文件作为历史来源保留，不代表未来把源码放回outputs。

本次全面更新针对32份项目维护Markdown。第三方文档、数据包说明、冻结run报告、.pre_git_sync与既有历史快照不改写；9份容易误导的旧说明逐字保存到 [更新前归档](history/20261023_before_md_refresh/README.md)。完整范围见 [文档索引](DOCUMENTATION_INDEX.md)。

## 6. 结构整理与本次文档检查的区别

此前结构整理已经通过298项项目CPU回归、8项Complex12专用CPU检查、两个状态入口兼容检查、159个本地导航链接及6466个成果文件元数据比对。这些是该次结构整理的记录，不冒充本次重新执行结果。

本次Markdown维护另执行项目规定检查、当前文档链接核对、归档原文核对及已有成果/非Markdown修改保护检查，具体结果统一记录在 [文档索引](DOCUMENTATION_INDEX.md)。文档任务不需要新的GPU验证或全项目哈希审计。
