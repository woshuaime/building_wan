# 历史T2V：A6000单轨统一prompt

目标是灰色建筑外观与环绕生成，属于历史T2V任务，与当前VACE缺损补全分开。

- 数据：data/single_orbit_videos，训练1992、验证234、测试118；384×384、81帧。
- [冻结划分](../../../configs/single_orbit_same_prompt_v1/)，统一文本context缓存1992个样本。
- [已有29个比较视频](runs/validation/)，原路径 outputs/validation/validation_all_single_orbit/。
- [28份保留权重](../../../checkpoints/building_wan_a6000_single_orbit_all/)；logger step不是optimizer update。
- [训练入口](../../../scripts/train_wan81_a6000.sh)、[续训入口](../../../scripts/resume_wan81_a6000.sh)、[验证入口](../../../scripts/validate_wan81_a6000.py)保持原路径。

历史报告的完整8 epochs终态仍缺充分证据，不自动补训或验证。多次原生崩溃的根因尚未确认；CPU与串行诊断不证明已经根治。完整参数、分段权重和故障过程见 [历史台账](../../../docs/history/20261023_before_structure/EXPERIMENTS.md) 及 [历史操作手册](../../../docs/history/20261023_before_structure/TRAINING.md)。
