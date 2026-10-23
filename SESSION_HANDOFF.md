# 会话交接

更新日期：2026-10-23（UTC+09:00）。Complex12 Rev1已全部完成；本次按用户要求整理目录，没有新GPU任务。

## 先看这些

1. [AGENTS.md](AGENTS.md) 与 [当前状态](PROJECT_CONTEXT.md)。
2. [实验目录](experiments/README.md)，按研究想法选择入口。
3. [Complex12最终报告](outputs/ComplexSingleOrbit_Rev1_Pilot_20261022/00_看结果/最终实验结论.md) 和原run的progress.json。

C/U/S各300更新、60条轨迹、4860张预测、12视频/12总览齐全。无需继续、补训或重跑。本轮报告只建议将来验证配对重复稳定性，没有执行授权。

## 整理后的路径

- Complex12唯一专用源码：experiments/vace/complex12_rev1/scripts/，共13个Python文件。
- 原outputs/ComplexSingleOrbit_Rev1_Pilot_20261022/scripts为指向上述目录的兼容链接；原resume.sh无需改动。
- experiments/vace/complex12_rev1/runs/20261022链接到原run；其他实验也有明确结果入口。
- 公共适配器及旧启动命令仍在scripts；项目主训练入口和第三方引擎仍在code。
- 大文件、冻结配置与结果物理位置保持原样。不要把runs链接再复制成第二份数据。
- 新想法/新run规范见 [PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)。

四份根文档原文已保留到 [整理前历史](docs/history/20261023_before_structure/README.md)。历史PID、运行中快照和“下一步”不能当作当前事实。

## 尚未解决的事实

B/CS04保存完整81帧后原生异常，输出经独立验证复用；C在挂载恢复后从40更新续到300。没有认定原生崩溃根因已修复。

历史A6000 T2V仍不能独立确认完整8 epochs结束，不自动补预算。旧P04/P06 LoRA只能作历史参考，不能用作无泄漏的新初始化。

## 工作副本保护

接手时已有未提交的历史T2V、诊断工具及docs迁移快照改动；本次整理不打包这些无关改动。此次Git只收实验目录整理、Complex12结束时的专用源码、当前文档与历史归档。

按 [TRAINING.md](TRAINING.md) 进行CPU检查。整理后的实际检查结果与提交以本次Git记录为准，不启动GPU验证目录变化。

本次已通过298项项目CPU回归、8项Complex12检查、规定语法/编译/diff检查及新旧状态入口检查；6466个成果文件元数据不变，159个当前导航链接有效。
