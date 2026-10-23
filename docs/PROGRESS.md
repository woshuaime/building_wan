# 进度查询：兼容入口

更新：2026-10-23（UTC+09:00）。项目当前摘要见 [PROJECT_CONTEXT](../PROJECT_CONTEXT.md)，跨实验完成证据见 [EXPERIMENTS](../EXPERIMENTS.md)。

Complex12 Rev1已完成C/U/S各300更新、60/60条轨迹和全部12栋媒体/指标。没有需要自动继续的GPU阶段。读取 [progress.json](../outputs/ComplexSingleOrbit_Rev1_Pilot_20261022/progress.json) 或从项目根执行：

```bash
/home/shi/miniconda3/envs/scj/bin/python experiments/vace/complex12_rev1/scripts/status.py
```

不要用resume.sh或训练/推理启动脚本查询进度。命令语义与历史实验状态读取见 [TRAINING](../TRAINING.md)。

更新前进度全文保存在 [历史快照](history/20261023_before_md_refresh/docs/PROGRESS.md)。历史轮次、PID、GPU占用和“下一步”只属于记录时刻，不能据此自动重启任务。
