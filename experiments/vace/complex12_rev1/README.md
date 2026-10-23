# Complex12 Rev1：十二栋单轨配对实验

更新：2026-10-23（UTC+09:00）。研究问题：普通LoRA C、均匀缺损加权U和结构带加权S，哪种方式对完整环绕建筑补全更有效？

**已全部完成。** C/U/S各300次真实优化器更新；B/L/C/U/S各12条81帧，共60条轨迹、4860张原始预测PNG、12视频、12总览。当前没有待续跑任务，也未自动启动下一实验。

## 1. 看结果与追溯证据

| 内容 | 入口 |
| --- | --- |
| 直接观看12栋 | [结果说明](runs/20261022/00_看结果/README_从这里开始.md)、[可播放图册](runs/20261022/00_看结果/评审图册.html) |
| 中文观察与结论 | [最终实验结论](runs/20261022/00_看结果/最终实验结论.md) |
| 科研指标 | [训练汇总](runs/20261022/metrics/train_summary.csv)、[验证汇总](runs/20261022/metrics/val_summary.csv)、[逐栋](runs/20261022/metrics/building_metrics.csv)、[逐帧](runs/20261022/metrics/frame_metrics.csv) |
| 原始模型预测 | [predictions](runs/20261022/predictions/) |
| 正式权重 | [C](runs/20261022/checkpoints/C/step300.safetensors)、[U](runs/20261022/checkpoints/U/step300.safetensors)、[S](runs/20261022/checkpoints/S/step300.safetensors) |
| 配置与方法 | [冻结配置](configs/pilot_20261022.json)、[原任务书](runs/20261022/task_brief.txt)、[损失映射](runs/20261022/mask_loss_mapping.md)、[共同采样表](runs/20261022/shared_sampling_table.csv) |
| 实际状态与日志 | [progress.json](runs/20261022/progress.json)、[logs](runs/20261022/logs/)、[恢复入口](runs/20261022/resume.sh) |

物理RUN_ROOT为 `/mnt/windowsE/chuanjun/building_wan/outputs/ComplexSingleOrbit_Rev1_Pilot_20261022/`。本页runs只是指向该目录的链接，没有复制或迁移成果。原run/scripts是到本实验scripts的兼容链接；正式源码只在本实验维护。

## 2. 数据身份与划分

数据根：`data/buildingfiller_import/Complex12/BuildingFiller_Complex12_512_Rev1_20261022/`。只使用Rev1，原始数据只读。

| 编号 | 源编号 | 身份 | 形体与关注处 |
| --- | ---: | --- | --- |
| CS01 | 304 | 训练 | 矩形平屋顶 |
| CS02 | 90 | 验证 | 长条平屋顶 |
| CS03 | 1019 | 训练 | 单坡屋顶 |
| CS04 | 688 | 验证 | 双坡屋顶 |
| CS05 | 986 | 训练 | 错位双坡 |
| CS06 | 1251 | 训练 | 分段坡面、跨屋脊 |
| CS07 | 1432 | 训练 | L形平屋顶 |
| CS08 | 1805 | 验证 | 高低错层平屋顶 |
| CS09 | 24 | 训练 | 主屋面与附属体，Rev1局部缺损 |
| CS10 | 923 | 训练 | 窄长高翼坡屋面，Rev1局部缺损 |
| CS11 | 346 | 训练 | 交叉多翼，Rev1局部缺损 |
| CS12 | 1344 | 验证 | L形多翼，Rev1局部缺损 |

每栋固定35°俯仰、360°单轨、512×512、000–080共81帧。输入000与080闭合重复；预测080由模型真实生成，不能复制000。训练与常规指标排除080；闭环指标保留它。

Exact合法全黑段：CS04的017–028、CS05的009–030、CS09的009–025、CS10的045–058、CS11的003–020；帧不删除、不扩Mask。17帧训练窗可以包含全黑帧，只要求整个窗的Exact至少有一个有效像素。结构带可见性不参与抽样筛选。

首次验收已检查5832张PNG、5991项清单、36个模型及72项闭环关系，详见原package_verify.log。文档维护不重新做大规模哈希。

## 3. 五种方法与配对设计

| 方法 | 初始化/监督 |
| --- | --- |
| B | 原始Wan2.1-VACE-1.3B，无LoRA |
| L | 已有P04/P06的LoRA-300，仅作历史参考 |
| C | 共同新LoRA初始化，普通FlowMatch损失 |
| U | 同一新初始化，缺损区均匀增加权重 |
| S | 同一新初始化，结构区域增加权重 |

L训练过Pilot8的P04/source1227和P06/source1848，各B000–B016；这里CS06/source1251是另一栋建筑。L的旧中文训练文本和两例数据与C/U/S不配对。C/U/S从新初始化开始，各建新优化器，未使用L权重。

C/U/S共享300行采样表及初始化、噪声、时间步序列。每次先均匀选训练建筑，再均匀选合法连续17帧窗口；起点0–63，只使用000–079。先裁图像窗口再编码VAE，不能从81帧latent切出训练段。无随机几何/颜色增强。

## 4. 标签、损失和模型输入

M表示原始Exact的缺损区域。结构标签E由raw_visible以7×7膨胀后与Exact相交得到；这是训练损失标签的构造，**不扩大推理ModelMask**。

| 组别 | latent网格权重 |
| --- | --- |
| C | `w = 1` |
| U | `w = 1 + (2 × sum(E) / sum(M)) × M` |
| S | `w = 1 + 2 × E` |

E/M空间以8×8最大池化映射，17帧时间对应[0]、[1–4]、[5–8]、[9–12]、[13–16]，得到1×5×64×64标签；target为1×16×5×64×64。先按通道求误差，再按权重归一化每个样本，最后沿用官方时间步权重。空标签或权重强度为零时退回原MSE。

VACE条件的64通道Mask使用空间重排与nearest-exact时间映射，不等同上述loss标签网格。正式条件仅含Damaged、二值ModelMask、文本；白区填RGB127，黑区保留原像素。GT只作训练监督和评测，结构图也不进入正式推理。

训练只缓存冻结文本context、GT latent和VACE条件编码，不缓存可训练VACE分支的输出。冻结DiT仍保留对LoRA的反向链。

## 5. 固定训练与推理参数

| 项目 | 实际值 |
| --- | --- |
| 基础模型 | Wan-AI/Wan2.1-VACE-1.3B，revision574e6a744642ce3bee319afc31496b88bde8aac4 |
| LoRA位置 | vace；q/k/v/o/ffn.0/ffn.2，150个Linear、300个A/B张量、10,936,320参数 |
| LoRA设置 | rank=alpha=16，dropout0、bias none，FP32 LoRA / BF16骨干及autocast |
| 优化器 | AdamW恒定5e-5，wd0，betas0.9/0.999、eps1e-8 |
| 训练预算 | C/U/S各300真实optimizer.step；batch1、accum1、clip1、workers0 |
| 内存与时间步 | gradient checkpointing；无CPU激活offload；官方1000时间步、采样范围0–1 |
| 随机性 | 训练seed20261022；推理seed20261018 |
| 正式推理 | 81帧、512×512、16FPS、30步、CFG5、cfg_merge=false |
| 其他推理项 | sigma_shift5、vace_scale1、denoising_strength1、CPU随机数设备 |
| VAE | tiled=true，tile_size30×52、tile_stride15×26 |
| LoRA加载 | scale1；每个worker新加载Base，只融合当次一种LoRA |
| 保存与评测 | 100/200恢复点、latest周期恢复状态、300正式权重；仅step300进入比较 |

正向文本：

> A complete untextured gray building with continuous roof surfaces and walls, viewed during a smooth 360-degree orbit against a plain light-gray background.

负向文本：

> 彩色纹理，窗户装饰，新增建筑，建筑形变，结构闪烁，背景变化，文字，水印

合法零时间步权重可能产生单步零loss，不因此删除或追加更新。仅LoRA safetensors不是完整恢复状态；本run的恢复还读取优化器、RNG和进度。C实际从40更新恢复后完成300，U/S使用各自新优化器完成。

## 6. 指标和交付口径

- RGB范围[0,1]；常规误差统计000–079。每栋按有效像素汇总，跨栋等权；8训练和4验证严格分开。
- 缺损区MAE/MSE以GT为参考，PSNR由该栋汇总MSE计算。空区标N/A，零MSE的PSNR为Infinity。
- 结构误差：灰度0.299R+0.587G+0.114B，3×3 Sobel除8、reflect边界，E区域内对Gx/Gy绝对差取均值。
- 保留区变化：Exact外预测与Damaged的MAE。闭环：预测000/080的全图和Exact000区域差异。
- 不把邻帧视角变化误认为闪烁，不用闭环MAE代替完整几何一致性。
- 逐帧5760行、逐栋72行包含Damaged基线；配对差分C−B、S−C、S−U共36行。各训练loss为300行。

每栋视频包含Before、GT、B、L、C、U、S及空白格，2×4布局，81帧/16FPS；每面板512×512另加32像素标题，成片2048×1088。关键帧图7行×5列，B000/020/040/060/080，共2560×3808。PNG与视频均来自原始预测；没有GT回贴、主体单独缩放、强制闭环或调色。

## 7. 实际结论与逐栋风险

C相对B的训练/验证缺损MAE降低71.00%/64.21%，说明普通LoRA的改善能够迁移到这4栋验证建筑。S相对C的验证MAE降低7.82%；S相对U仅降低1.80%，结构Sobel仅降低0.19%，验证MAE优势主要由CS12贡献。

S的缺损均值最好，L的保留区变化和全图闭环均值最好。S相比C在全部12栋的保留区与全图闭环均有代价；验证均值分别增加4.35%和20.38%。不能只看缺损MAE宣布S全面胜出。

| 样本 | 完整轨迹观察 |
| --- | --- |
| CS01 | Base前后段墙面补丁明显减少，C/U/S仍有柔和阴影差异 |
| CS02 | 35–68段黑槽被填平；L的缺损MAE最好，S与C/U接近 |
| CS03 | 屋面孔洞与墙面黑槽减轻；远侧屋顶细线仍在，S不优于C/U |
| CS04 | 33–53段屋顶/端墙折痕减轻，40–50仍可见；S结构提升有限 |
| CS05 | 早期3–7帧浅凹未完全消失；固定五帧可能漏掉，U局部更好 |
| CS06 | 跨脊9–26段亮缝弯曲，14–22仍未恢复正确直线；S结构误差高于C |
| CS07 | L形缺口闭合，S的B010有小灰斑，局部边缘仍波动 |
| CS08 | Base已有较完整轮廓，15–23段连接凸起减少但残留，增益有限 |
| CS09 | 0–6浅凹、36–44多余折线与后段阴影仍在；S结构误差比B高4.75% |
| CS10 | Base/L黑洞在C/U/S的完整81帧中消除，是明显改善例；C的MAE最低 |
| CS11 | 36–62段大折痕减少，细部连接和阴影仍有误差；L的MAE最低 |
| CS12 | 前段补丁减少，27–44弯曲折痕和长脊波动仍在；S数值提升最明显 |

观察来自完整81帧接触表及关键帧核对，不等同实时播放验收或高频闪烁定量测量。四栋验证、单种子和二维灰度误差不能支持三维几何精确恢复或统计显著性。下一步建议先保持VACE骨干，用更充分的建筑/种子对照验证加权收益；这只是建议，没有启动。

## 8. 耗时、退出与状态读取

| 阶段 | 已记录worker耗时 | 含义 |
| --- | ---: | --- |
| C训练完成进程 | 1570.050秒 | 从40更新恢复后的进程计时，不含中断前40步和停机等待 |
| U训练 | 675.368秒 | 当前完整训练worker |
| S训练 | 667.404秒 | 当前完整训练worker |
| B / L各12次推理 | 2217.205 / 2263.578秒 | 各worker计时合计 |
| C / U / S各12次推理 | 2296.492 / 2303.893 / 2300.669秒 | 各worker计时合计 |

三组训练记录峰值allocated为15,986,839,552 bytes，reserved为16,223,567,872 bytes。LoRA推理峰值分别为18,406,674,944和27,736,932,352 bytes；这些是历史阶段峰值，不是当前空闲显存，也不能跨阶段相加。

B/CS04在保存完整预测后原生退出−6，独立复核后保留有效输出；其余推理worker和最终三组训练完成进程退出0。历史原生故障根因仍未确认。最终progress完成时间2026-10-23 02:59:37 +08:00（03:59:37 +09:00）。

从项目根目录只读查看：

```bash
/home/shi/miniconda3/envs/scj/bin/python experiments/vace/complex12_rev1/scripts/status.py
```

不要用resume.sh作为状态查询，也不要更改成功run的冻结配置。源码中RUN固定指向此run；新想法需新目录和独立运行配置。

## 9. 源码职责

| 文件 | 职责 |
| --- | --- |
| [complex_common.py](scripts/complex_common.py) | 数据、标签映射、路径和配置 |
| [complex_train.py](scripts/complex_train.py) | C/U/S优化、保存和完整状态恢复 |
| [complex_infer.py](scripts/complex_infer.py) | 单方法/单建筑81帧预测 |
| [complex_evaluate.py](scripts/complex_evaluate.py) | 指标与同步媒体 |
| [run_pilot.py](scripts/run_pilot.py) | 串行阶段调度与成功任务复用 |
| [status.py](scripts/status.py) | 只读进度 |
| prepare.py、native_exit.py | 初始准备、退出处理 |
| review_frames.py、finalize.py、write_final_report.py | CPU复核和交付 |
| test_metrics.py、test_pairing.py | 八项专用CPU检查 |

共享VACE组件仍从项目scripts引用；不修改第三方骨干来实现目录整理。
