# Pilot8：LoRA-300完整81帧评估

问题：P04/P06的17帧收益能否扩展到整圈，并迁移到其余六栋？已完成八例81帧与32项成果，没有新训练。

- [八例成果](runs/20261020/)：outputs/vace/lora300_eval81/run_20261020_194300_145372。
- [执行实现](../../../scripts/vace_lora300_eval81.py)，[既有启动入口](../../../scripts/run_vace_lora300_eval81.sh)。
- 权重唯一来自 [P04/P06 LoRA17](../pilot8_lora17/README.md) 的LoRA-300，vace分支、scale1。
- Base复用 [八例原始结果](../pilot8_base/README.md)。512×512、81帧、16FPS、30步、CFG5、seed20261018；没有Composite。

P04屋顶凹槽全环绕持续填平；P06黑洞全程闭合，仍有轻微痕迹和轮廓波动。迁移六例中P05/P07/P08明显改善，P01/P02有限，P03整体持平且部分后段变差。

首次P04推理后的制图失败曾导致经用户授权补跑一次；最终八例均正常完成。完整旧记录见 [实验台账归档](../../../docs/history/20261023_before_structure/EXPERIMENTS.md)。
