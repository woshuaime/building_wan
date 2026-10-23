# P04/P06：17帧VACE LoRA小样本

更新：2026-10-23（UTC+09:00）。研究问题：固定两个训练题，能否让VACE分支学习恢复表面？**首轮已经完成300次真实优化器更新、六次17帧推理，以及两例各四项可视化。**

## 数据和训练范围

| 样本 | 源编号 | 缺损档位 | 固定原始帧 | 更新次数 |
| --- | ---: | --- | --- | ---: |
| P04 | 1227 | A2 Medium | B000–B016 | 150 |
| P06 | 1848 | A3 Medium | B000–B016 | 150 |

在B000–B079枚举连续17帧窗口，按原始Exact可见面积选最大窗，并列取最早；两例均得到B000–B016。B080不参与。两例按固定次序交替，各150更新，无新增样本、随机换窗或验证集混入。

GT只作官方FlowMatch监督目标；Damaged白区填灰127和二值ModelMask作条件，文本沿用 [Base](../pilot8_base/README.md)的中文正负描述。训练和推理均保持原始像素空间，不扩Mask、不做Composite。推理不读取GT作为条件。

## 参数和实际执行

| 参数 | 值 |
| --- | --- |
| 初始化 | 原始Wan2.1-VACE-1.3B，新建vace LoRA，无旧增量 |
| 目标 | q/k/v/o/ffn.0/ffn.2；150个Linear、300个A/B张量 |
| LoRA | rank=alpha16、scale1、10,936,320可训练参数 |
| 精度 | BF16骨干/autocast，FP32 LoRA |
| 优化器 | AdamW，恒定5e-5，wd0，clip1 |
| 批量/累积/workers | 1/1/0，gradient checkpointing |
| 预算/保存 | 300真实更新；只保存100、300 |
| 随机性 | 训练与推理seed20261018 |
| 推理 | 17帧、512×512、16FPS、30步、CFG5 |

顺序为两例Base → 300更新训练 → 两例LoRA-100 → 两例LoRA-300 → CPU制图。所有GPU阶段使用fresh进程，避免100/300融合叠加；每个阶段退出0。训练早期与最终记录确认梯度有效和LoRA参数变化。

[冻结配置](../../../configs/vace_roof_smoke_v1.json)、[主流程](../../../scripts/vace_roof_smoke.py)、[训练实现](../../../scripts/vace_roof_training.py)、[详细专项记录](../../../docs/VACE_P04_P06_LORA_SMOKE.md)。

## 权重与成果

[运行目录](runs/20261020/)物理路径：

`outputs/vace/lora_roof_smoke/run_20261020_165237_150325_2048720/`

| 位置 | 作用 |
| --- | --- |
| [training/lora_update0100.safetensors](runs/20261020/training/lora_update0100.safetensors) | 100更新LoRA，43,780,800 bytes |
| [training/lora_update0300.safetensors](runs/20261020/training/lora_update0300.safetensors) | 300更新LoRA，43,780,800 bytes |
| [training/experiment.json](runs/20261020/training/experiment.json) | 配置、代码来源、阶段退出、运行状态 |
| [training/loss.csv](runs/20261020/training/loss.csv) | 300次实际更新记录 |
| 04/、06/ | 各两个视频、两个PNG |

每例四文件：01_微调后补全.mp4、02_缺损_原模型_微调后_GT_对比.mp4、03_训练进展对比.png、04_Mask验证.png。进展图展示Before/Base/100/300/GT五行及B000/004/008/012/016；Mask图P04取B003、P06取B006。整个run共12个指定文件。

仅LoRA权重不能恢复optimizer/RNG；此历史入口没有完整状态resume。当前结果已完整，不应再执行 [启动脚本](../../../scripts/run_vace_roof_smoke.sh)的--run或--tmux。

## 效果与限制

| Mask内MAE，0–255 | Base | LoRA-100 | LoRA-300 |
| --- | ---: | ---: | ---: |
| P04 | 27.2225 | 1.7813 | 1.5860 |
| P06 | 46.3986 | 1.3737 | 1.3068 |

关键帧里P04凹槽和P06开口明显变为连续表面；300相对100的RGB误差继续下降，但不等于所有几何/时序方面都更好。这是训练题效果，不能当作独立泛化证据。原experiment.json中的pending_user_review保留，不能用自动完成状态代替用户验收。

全流程749.247秒；训练550.933秒，峰值allocated15.139GiB/reserved15.393GiB，均为当时实测。后续 [八例81帧](../pilot8_lora81/README.md)单独验证了整圈与迁移；这份LoRA-300也作为 [Complex12](../complex12_rev1/README.md)的L参考，但C/U/S未以它初始化。
