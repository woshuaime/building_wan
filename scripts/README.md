# 项目公共工具与兼容入口

新的实验专用实现放在 [experiments/](../experiments/README.md)。本目录保留已有跨实验适配器、命令入口、历史工具和CPU测试，避免因移动共享模块破坏旧命令。

| 类别 | 主要文件 | 对应实验 |
| --- | --- | --- |
| BuildingFiller包适配与检查 | buildingfiller_pilot8.py、audit_buildingfiller_package.py | [Pilot8 Base](../experiments/vace/pilot8_base/README.md) |
| VACE模型/条件/81帧公共实现 | vace_p01.py、vace_p01_environment.py、vace_p01_81f*.py | 多个VACE实验复用 |
| 八例基线调度 | vace_building_test.py、vace_building_reuse.py、run_vace_building_test.sh | [Pilot8 Base](../experiments/vace/pilot8_base/README.md) |
| 两例17帧LoRA | vace_roof_*.py、run_vace_roof_smoke.sh | [LoRA17](../experiments/vace/pilot8_lora17/README.md) |
| 八例81帧LoRA评估 | vace_lora300_eval81.py、run_vace_lora300_eval81.sh | [LoRA81](../experiments/vace/pilot8_lora81/README.md) |
| 结果重排和CPU回贴 | reorganize_vace_results.py、composite_vace_results.py | [Composite历史诊断](../experiments/vace/pilot8_composite/README.md) |
| A6000历史训练、暂停、续训、验证 | train_wan81_a6000.sh、resume_wan81_a6000.sh、wan_a6000_state.py、validate_wan81_a6000.py | [T2V单轨](../experiments/t2v/single_orbit/README.md) |
| 原生退出/权重检查 | wan_gdb.py、check_wan_lora.py、diagnose_wan_a6000.py | 既有故障诊断 |
| A100/Slurm历史入口 | submit_wan81*.sh、train_wan81.sh等 | [早期T2V](../experiments/t2v/early_experiments/README.md) |
| CPU回归 | tests/ | 按项目规则执行 |

Complex12专用源码已经移到 [experiments/vace/complex12_rev1/scripts/](../experiments/vace/complex12_rev1/scripts/)。只读状态见 [操作手册](../TRAINING.md)；GPU命令不是状态查询工具。
