# P04/P06：17帧VACE LoRA小样本

问题：两个固定训练题是否能学会表面恢复？首轮已完成300次实际优化器更新，以及两例Base、LoRA-100、LoRA-300共6次17帧比较。

- 数据仅P04/source1227、P06/source1848，各B000–B016。
- VACE分支rank=alpha16，scale1，学习率5e-5，训练/推理seed20261018；冻结骨干。
- [固定配置](../../../configs/vace_roof_smoke_v1.json)，[训练入口](../../../scripts/vace_roof_smoke.py)，[损失与更新实现](../../../scripts/vace_roof_training.py)。
- [既有启动脚本](../../../scripts/run_vace_roof_smoke.sh)，仅查看文件不会运行；不重复启动。
- [结果和权重](runs/20261020/)：outputs/vace/lora_roof_smoke/run_20261020_165237_150325_2048720。
- [专项记录](../../../docs/VACE_P04_P06_LORA_SMOKE.md)。

P04凹槽、P06开口在17帧中明显改善。它们是训练题，不是独立泛化样本。训练出的LoRA-300随后用于 [八例81帧评估](../pilot8_lora81/README.md)，也作为Complex12的L参考；C/U/S没有以它为初始化。
