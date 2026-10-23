# Pilot8：LoRA-300完整81帧评估

更新：2026-10-23（UTC+09:00）。研究问题：P04/P06的17帧收益能否扩展到完整环绕，并迁移到其他六栋？**八例81帧已经完成，最终32项成果，没有新增训练。**

## 唯一权重与公平条件

只使用 [P04/P06 LoRA17](../pilot8_lora17/README.md)的300更新权重：

`/mnt/windowsE/chuanjun/building_wan/outputs/vace/lora_roof_smoke/run_20261020_165237_150325_2048720/training/lora_update0300.safetensors`

加载到vace分支、scale1，不叠加LoRA-100、T2V或其他增量。每例从原始VACE新加载。Base全部复用 [八例原始结果](../pilot8_base/README.md)，没有重跑Base。

81帧B000–B080、512×512、16FPS、30步、CFG5、seed20261018；正负文本、Damaged填灰127、二值ModelMask与Base相同。保留合法全黑帧，无Mask扩大/羽化、无Composite或GT条件。B080使用真实预测。

## 执行与结果位置

先完成训练题P04/P06，检查整圈改善仍成立，再运行P01/P02/P03/P05/P07/P08。首次P04推理成功但CPU制图报PIL/NumPy类型错误，临时预测被清理；用户授权修复后补跑一次。因此最终结果是八条轨迹，实际推理尝试为九次，不隐去失败消耗。

两个成功批次耗时369.2秒和1129.4秒，共1498.6秒；此数不含首次失败尝试。记录的推理峰值allocated17.141GiB/reserved25.832GiB。最终两批均正常退出。

[八例成果](runs/20261020/)物理位置：

`outputs/vace/lora300_eval81/run_20261020_194300_145372/`

| 每例文件 | 内容 |
| --- | --- |
| 01_微调后补全.mp4 | 512×512/16FPS，81帧原始LoRA预测 |
| 02_Base_微调后_GT_对比.mp4 | Before、Base、LoRA-300、GT四栏同步 |
| 03_关键帧总览.png | 四行五列，同帧比较 |
| 04_Mask验证.png | Before、ModelMask、Base、LoRA、GT五栏 |

编号01–08和源建筑映射严格沿用Base，视频和PNG均完成读取核验。原始Base、Composite及17帧目录未被覆盖。

## 八例观察

| 样本 | 源编号 | 与训练关系 | 完整81帧观察 |
| --- | ---: | --- | --- |
| P01 | 465 | 未训练 | 改善有限，部分视角与Base接近 |
| P02 | 763 | 未训练 | 改善有限，不能宣称所有结构都提升 |
| P03 | 871 | 未训练 | 整体相近，部分后半圈可能变差 |
| P04 | 1227 | 训练题扩展 | 屋顶凹槽整圈明显减轻，表面连续性更好 |
| P05 | 1259 | 未训练 | 明显迁移收益，早期小凹陷和闭环误差仍在 |
| P06 | 1848 | 训练题扩展 | 黑洞整圈持续闭合，仍有轻微痕迹与形状波动 |
| P07 | 2297 | 未训练 | 明显改善，局部灰斑和边缘误差未完全消失 |
| P08 | 939 | 未训练 | 有明显迁移改善，细部形状仍需观察 |

结果支持“保持VACE骨干、扩大建筑补全LoRA对照”的后续研究方向；不证明几何正确性、高频闪烁消失或全部未见建筑均改善。当前没有自动追加训练。后来的 [Complex12 Rev1](../complex12_rev1/README.md)是独立配对实验，数据、配置、结果另存。

## 实现与操作边界

- [vace_lora300_eval81.py](../../../scripts/vace_lora300_eval81.py)：批次、worker、CPU制图与结果验证。
- [run_vace_lora300_eval81.sh](../../../scripts/run_vace_lora300_eval81.sh)：仅转发参数到Python入口。
- 无参数或--dry-run执行CPU预检；--run-trained和--run-transfer会启动对应GPU批次。
- --render-only配合既有output-root，只从保留原始视频完成CPU制图，不重新推理。已有完整结果不需要执行它。
- 历史故障和授权补跑过程见 [原台账](../../../docs/history/20261023_before_structure/EXPERIMENTS.md)；不要用重新启动代替读取结果。
