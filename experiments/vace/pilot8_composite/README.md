# Pilot8：CPU缺口回贴诊断

这项历史检查把Base预测只贴回原Exact缺口，其余位置保留Damaged，用来区分“缺口恢复”与“已知区域被模型改动”。它没有训练或加载新模型。

- [八例结果](runs/cpu_check/)：原位置 outputs/vace/composite_check/01–08。
- [CPU实现](../../../scripts/composite_vace_results.py)。
- [原始Base](../pilot8_base/README.md)。

主要观察：保留区可通过回贴保持，但P04凹槽和P06黑洞仍存在，说明仅后处理不能解决缺口生成问题。这里的视频是Composite，不作为LoRA或Complex12的原始模型输出；后续实验没有采用这种回贴。
