# 公共适配器、兼容入口与CPU检查

更新：2026-10-23（UTC+09:00）。这里保留跨实验代码和已有命令入口。新实验专用源码放experiments；Complex12的13个脚本已经位于 [专属目录](../experiments/vace/complex12_rev1/scripts/)。

## 1. 工具地图

| 类别 | 主要文件 | 说明 |
| --- | --- | --- |
| BuildingFiller包适配 | buildingfiller_pilot8.py、audit_buildingfiller_package.py | 身份、帧序、相机和数据完整性 |
| VACE基础加载与条件 | vace_p01.py、vace_p01_environment.py、vace_p01_81f*.py | 本地模型、Exact/填灰、完整序列公共实现 |
| 八例Base调度 | vace_building_test.py、vace_building_reuse.py | 成功前缀复用、串行推理与交付 |
| 两例LoRA17 | vace_roof_*.py | 两固定训练题、LoRA优化、阶段监督与制图 |
| 八例LoRA81 | vace_lora300_eval81.py | 训练题先行、迁移批次及CPU收尾 |
| CPU展示 | reorganize_vace_results.py、composite_vace_results.py | 既有结果布局重排、独立Composite诊断 |
| A6000 T2V | train_wan81_a6000.sh、resume_wan81_a6000.sh、wan_a6000_state.py、validate_wan81_a6000.py、validate_all_single_orbit.py | 历史训练/续训/验证，不是当前VACE入口 |
| 暂停与恢复 | pause_wan81_a6000.sh、resume_from_pause.sh | 历史T2V进程及样本游标管理 |
| 原生退出与权重诊断 | wan_gdb.py、check_wan_lora.py、diagnose_wan_a6000.py | 退出状态和已有权重检查；不证明根因已解决 |
| A100/Slurm历史 | submit_wan81*.sh、train_wan81.sh等 | 旧环境专用，本机不默认执行 |
| 可选外部worker | claude_worker_mcp.py、configure_claude_key.sh | [使用边界](../docs/CLAUDE_WORKER.md)，本次未调用API |
| CPU回归 | [tests/](tests/) | 项目规定的回归检查入口 |

## 2. 默认行为不能混淆

| 入口 | 无参数/正式模式 | CPU或只读分支 |
| --- | --- | --- |
| run_vace_p01_465_17f.sh | 无参数启动GPU | --dry-run作预检 |
| run_vace_p01_465_81f.sh | 无参数启动GPU，--tmux为可见会话 | --dry-run作预检 |
| run_vace_building_test.sh | 无参数启动批量GPU；--resume复用有效前缀 | --dry-run预检；--prepare含P01复制/复用，并非纯读取 |
| run_vace_roof_smoke.sh | --run/--tmux才启动GPU | 无参数或--dry-run为CPU准备 |
| run_vace_lora300_eval81.sh | --run-trained/--run-transfer启动GPU | 无参数/--dry-run预检；--render-only是CPU制图 |
| train_wan81_a6000.sh | 正式训练入口 | 不假设通用--dry-run存在 |
| resume_wan81_a6000.sh | 正式weights-only续训入口 | 单独--dry-run为预检 |
| Complex12 status.py | 读取现有状态 | 不调用模型或启动任务 |

预检可能完整读取大权重，CPU准备/制图也可能写文件；“不使用GPU”不等于“完全只读”。当前VACE成果都已完成，日常直接打开文件或用status.py。不要为了查看进度重复启动。

## 3. 共享条件与结果边界

VACE实际推理条件只使用Damaged派生图、二值ModelMask和文本。GT/结构标签仅供监督、生成后的展示或指标，不进入推理。原始模型输出与Composite分开保存；常规制图不得把GT贴回预测。

T2V的共享prompt缓存、历史LoRA和续训游标只适用于对应历史实验。单个.safetensors不能当完整optimizer/RNG恢复状态。新想法不得直接改当前run固定输出路径来覆盖旧成果。

## 4. 修改后检查

从项目根目录按 [TRAINING](../TRAINING.md)执行规定shell语法、Python编译、scripts/tests及git diff检查。CPU测试可设置CUDA_VISIBLE_DEVICES为空，避免附带CUDA计算。测试中的模拟模型和临时视频不能计作真实训练或推理成果。

本次Markdown更新不改这里已有的未提交代码，也不把它们混入文档提交。当前测试结果和文档链接检查记录见 [文档索引](../docs/DOCUMENTATION_INDEX.md)。
