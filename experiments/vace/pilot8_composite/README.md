# Pilot8：CPU缺口回贴诊断

更新：2026-10-23（UTC+09:00）。**历史检查已完成**，八例各三个成果。本项用已有Base预测判断错误来自缺损区还是已知区域，未训练、未加载新模型、未增加GPU推理。

## 为什么有这轮检查

Base会同时改变缺口、原有墙面和背景。把非缺损区域还原为Damaged后，可以观察：保留区变干净是否足以改善整体，以及缺口内的凹槽、黑洞是否仍然存在。因此它是定位问题的一次后处理实验，不是一份额外LoRA训练结果。

公式：`Composite = Exact白区 × Base预测 + Exact黑区 × Damaged`。沿用原始二值Exact，不扩大、不羽化，不使用GT作为回贴像素。所有帧与原Base一一对应，建筑尺度不变。

## 输入、结果与边界

- 数据身份、512×512/81帧/16FPS与 [原始Base实验](../pilot8_base/README.md)相同，编号01–08不变。
- [八例成果](runs/cpu_check/)的物理位置为 `outputs/vace/composite_check/01–08/`。
- [CPU实现](../../../scripts/composite_vace_results.py)读取现有预测和原始Damaged/Exact；GT仅用于并排比较。
- 原始Base的四文件目录保持独立，Composite结果不能覆盖或冒充原始预测。

| 每例文件 | 内容 |
| --- | --- |
| 01_仅缺口补全.mp4 | 仅缺口取Base，其余取Damaged的81帧视频 |
| 02_四方视频对比.mp4 | Before、Base、Composite、GT同步对比 |
| 03_关键帧对比.png | 同原始视角比较上述四种画面 |

## 实际结论

回贴能够保留原本正确的已知区域，但P04屋顶凹槽/台阶和P06黑洞仍在缺损内部。仅后处理没有让模型学会正确补全，因此随后开展的是P04/P06的VACE LoRA训练。

后续 [LoRA17](../pilot8_lora17/README.md)、[LoRA81](../pilot8_lora81/README.md)与 [Complex12](../complex12_rev1/README.md)展示的都是原始模型预测，未采用这里的回贴。比较它们时应选对应Base原始结果，不能用Composite代替Base来改变评价口径。

本目录保留这次诊断的来源与结论，不需要再运行、删除或合并到新的正式结果中。
