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
| 根README.md | 项目发展时间线、原有介绍、后续更新和阅读入口 |
| 根PROJECT_CONTEXT.md | 历次目标/环境/状态、当前事实和未解决事项 |
| 根EXPERIMENTS.md | 完整实验过程、失败/修复、横向结果和结论演变 |
| 根TRAINING.md | 各时期操作与故障处理、当前状态读取/恢复/检查 |
| 根SESSION_HANDOFF.md | 历次会话交接、用户约定和最新接手事实 |
| AGENTS.md | 当前工作规则、授权边界和规则沿革 |
| experiments各README | 该想法的数据、配置、代码、结果和局限 |
| docs专项说明 | 数据交接、技术细节、历史故障和工具 |
| docs/DOCUMENTATION_INDEX.md | 文档全量清单、更新范围与检查 |
| run内报告/日志 | 该次执行的冻结证据，后续文档不倒改 |

docs旧PROJECT_CONTEXT/EXPERIMENTS/HANDOFF/PROGRESS恢复早期正文并追加后来更正，当前状态仍以根文档最新补记为准。此前“只作兼容导航”的做法已按用户要求纠正。原文件必须能读到历史；若旧记录与后来结论不同，应保留原判断及其后续更正，不能删去失败记录来制造一致。

## 5. 历史保护与Git

原始数据、模型、权重、视频、预测图、日志和缓存默认只读且不入Git。runs链接仅记录目标路径，Git不会沿链接打包成果。其他平台不支持链接时，从实验README的原路径定位；GitHub上无法读取服务器媒体属于部署差异。

chuanjun.code-workspace只描述Windows工作区，不加入服务器绝对路径。已跟踪的旧run配置、损失映射和resume文件作为历史来源保留，不代表未来把源码放回outputs。

本次全面更新针对32份项目维护Markdown。第三方文档、数据包说明、冻结run报告、.pre_git_sync与既有历史快照不改写；9份容易误导的旧说明逐字保存到 [更新前归档](history/20261023_before_md_refresh/README.md)。完整范围见 [文档索引](DOCUMENTATION_INDEX.md)。

## 6. 结构整理与本次文档检查的区别

此前结构整理已经通过298项项目CPU回归、8项Complex12专用CPU检查、两个状态入口兼容检查、159个本地导航链接及6466个成果文件元数据比对。这些是该次结构整理的记录，不冒充本次重新执行结果。

本次Markdown维护另执行项目规定检查、当前文档链接核对、归档原文核对及已有成果/非Markdown修改保护检查，具体结果统一记录在 [文档索引](DOCUMENTATION_INDEX.md)。文档任务不需要新的GPU验证或全项目哈希审计。

## 7. 2026-10-23后续修正：恢复原文件里的历史正文

前述结构整理与第一轮Markdown维护虽然保留了归档，但将长期MD缩成当前摘要，未满足用户要求。现从整理前原文与Git恢复四份根长期记录、根README及九份旧专题/工具说明，后续完成结果继续保留在同一文件。

长期MD采用“阅读索引＋此前累计正文＋按日期追加更新”。历史只调整标题层级以便阅读，不删除参数、命令、失败、过程和当时结论；日期或阶段不同的状态可以并存，由后续补记说明演变。归档仅作备份，不代替正文。

未来目录整理只改变工作入口；新想法可以新建实验目录，但主文档中的项目发展史继续累积。维护摘要、修正链接或增加索引时，必须核对原有记录仍完整存在，不能重复本次错误。
