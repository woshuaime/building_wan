# P04/P06 VACE条件补全LoRA小样本验证

本说明记录两例训练题的过拟合验证。它检查条件补全训练链路是否真实工作、生成表面能否更接近同帧GT；两例均参与训练，结果不能作为独立测试或81帧环绕一致性证据。

## 1. 当前运行与证据

最终核查时间：**2026-10-20 17:05:32 +08:00**。首轮已正常完成；当前查看结果和人工验收，不再次启动训练。

| 项目 | 已核查事实 |
| --- | --- |
| 项目目录 | `/mnt/windowsE/chuanjun/building_wan` |
| Python | `/home/shi/miniconda3/envs/scj/bin/python` |
| Run | `outputs/vace/lora_roof_smoke/run_20261020_165237_150325_2048720/` |
| tmux | `vace_p01_17f:3.0`，pane `%3` |
| Base | P04/P06两次17帧推理均完成，阶段退出0 |
| 训练 | loss.csv为update1–300，两例各150次；训练worker正常退出0 |
| 权重 | 100/300步各43,780,800 bytes，均为真实vace LoRA增量 |
| 后评估 | 六次17帧推理全部完成，Base/train/lora100/lora300均exit_code0 |
| 最终媒体 | 04/06各两视频、两PNG，验证通过；加training共12文件 |
| 整体状态 | runtime为complete/complete；终端打印COMPLETED，总耗时749.247秒 |
| 待人工确认 | 完整17帧时序/闪烁和用户视觉验收；runtime为pending_user_review |

本轮由用户在VS Code Remote-SSH终端启动，Codex本次只读检查已有运行和成果、维护文档。训练前5步终端逐次确认有限loss、有效梯度、optimizer仅含LoRA及参数实际变化；最终training_result也确认gradient_check_passed和parameters_changed为true。100/300阶段均实际加载150个vace目标层的LoRA。

中期17:00:09曾记录210更新、两例各105次和100步权重；该记录仅为过程证据，已被以上最终状态更新，不再作为待完成任务。

### 1.1 运行耗时与峰值

| 阶段 | 耗时秒 | Peak allocated GiB | Peak reserved GiB | 退出码 |
| --- | ---: | ---: | ---: | ---: |
| 两例Base | 61.052 | 15.864719 | 17.308594 | 0 |
| 训练300更新 | 550.933 | 15.138875 | 15.392578 | 0 |
| 两例LoRA-100 | 63.372 | 15.866717 | 17.384766 | 0 |
| 两例LoRA-300 | 64.277 | 15.866717 | 17.384766 | 0 |

全流程749.247秒，约12分29秒；除上表计算阶段外还含子进程调度与CPU收尾。每个阶段峰值从本阶段加载前重置，包含加载/编码，不代表仅训练稳态的显存。

### 1.2 已查看的结果与量化参考

终端汇总以下RGB误差，未额外保存指标报告：

| 样本/指标 | Base | LoRA-100 | LoRA-300 |
| --- | ---: | ---: | ---: |
| P04，Mask内对GT的MAE | 27.2225 | 1.7813 | 1.5860 |
| P06，Mask内对GT的MAE | 46.3986 | 1.3737 | 1.3068 |
| P04，Mask外对Damaged的变化 | 0.7613 | 0.5765 | 0.4826 |
| P06，Mask外对Damaged的变化 | 0.6816 | 0.5301 | 0.4475 |

数值单位为0–255。按17帧相应区域的所有像素和RGB通道汇总；Mask内参考完整GT，Mask外参考Damaged。Mask外包含大量背景，较低平均值不能保证每处建筑边界都没有副作用。

已抽看两例五行总览的B000/B004/B008/B012/B016：

- P04同条件17帧Base仍有明显屋顶/墙顶凹陷，100/300步均变为更连续的完整表面，更接近GT。
- P06同条件17帧Base仍有开口/凹陷，但内部比旧81帧黑洞更浅；100/300步在已查看视角表现为连续表面，改善不只是把洞涂亮。
- 100步已出现明显改善，300步RGB误差再小幅下降；仅凭这些关键帧不能量化整个视频的几何误差或宣布300步在所有方面都优于100步。

当前结论是“训练链路正常，已查看的两例训练题关键帧视觉改善”。未逐帧人工播放完整17帧，也未做三维几何重建或独立测试，闪烁/时序和用户验收仍待确认。experiment.json的pending_user_review不因文档抽查而被修改。

## 2. 为什么选这两例

原始VACE-1.3B的八例81帧结果已在 `outputs/vace/test/01–08/`。CPU硬边界回贴位于 `outputs/vace/composite_check/01–08/`，采用白Mask取预测、黑Mask取Damaged。

| 样本 | 来源 | 基线问题 | 回贴后的观察 |
| --- | --- | --- | --- |
| P04 | source1227，A2 Medium | 屋顶凹槽、补丁或台阶 | 未损区域变干净，但凹槽和台阶仍在 |
| P06 | source1848，A3 Medium | 深色洞口、表面没有闭合 | 保留区域恢复，但黑洞仍在 |

当前先学习这两道题，控制训练预算和输入变量。既有81帧预测只提供问题线索；不能截取其中17帧冒充本轮单次17帧Base。

## 3. 数据来源与固定窗口

只读数据包：

~~~
/mnt/windowsE/chuanjun/building_wan/data/buildingfiller_import/P02_P08_20261020/BuildingFiller_Pilot8_AllRes_v1.zip
~~~

包内根为 `BuildingFiller_Pilot8_AllRes_v1/`。通过manifest、sample_meta和cameras核对真实身份，原始PNG位于 `samples/P04/512/` 与 `samples/P06/512/` 下的 `gt/`、`damaged/`、`exact/`。

| 样本 | stable_id | sample_id | 原始帧号 | 17帧Exact白像素总数 |
| --- | --- | --- | --- | --- |
| P04 | `b1227_bd506b7f` | `sample_0003` | B000–B016 | 139,483 |
| P06 | `b1848_33a0a1a4` | `sample_0005` | B000–B016 | 206,570 |

选窗规则：在B000–B079上枚举起点0–63的64个连续17帧窗口，累加原始Exact白像素数，取最大值；并列取更小起点。B080为闭合重复帧，不参与。两例选定后不随机换窗口。

GT、Damaged、Exact使用完全相同的原始索引、512×512像素空间和相机视角。无随机裁剪、翻转、调色或单独主体缩放。P04/P06原81帧数据均无全黑Exact；加载器仍保留合法全黑Mask的语义，不人为造白区。

## 4. 监督目标和模型条件如何接入

| 角色 | 实际来源 | 本地接口 | 用途 |
| --- | --- | --- | --- |
| 干净训练目标 | 同帧原始完整GT | `input_video` → `input_latents` | GT经VAE编码、加训练噪声，计算官方FlowMatch监督 |
| 条件图像 | 原始Damaged，Mask白区填RGB127 | `vace_video` | 提供可见残缺建筑及上下文 |
| 条件Mask | 原始Exact二值复制为相同RGB通道 | `vace_video_mask` | 白区生成，黑区保留上下文 |
| 文本 | 成功VACE冻结配置的统一正向描述 | `prompt` → T5 context | 两例共享相同文本条件 |

复用 `scripts/vace_p01.py:adapt_images`：Exact须为L模式0/255，Damaged须为RGB，空间保持512×512；白区填灰127，黑区像素不变。不扩张、收缩、反转或羽化Mask。

真实 `WanVideoUnit_VACE` 会读取视频与Mask，构造inactive/reactive两路冻结VAE latent及64个Mask通道，得到96通道VACE上下文。它不是仅在表格里写了Mask字段。该上下文只含冻结编码与Mask，不是可训练VACE分支的输出。

GT进入训练目标编码是正确监督路径。正式推理worker使用 `include_gt=False`，不读GT图像，也不传GT视频、首帧、尾帧或reference image。Raw Prediction和Composite不作为训练答案。

Exact来自完整/缺损几何配对，配置明确记录 `mask_is_GT_oracle=true`；本轮没有验证从真实残缺图像自动预测Mask的能力。

## 5. 模型、文本与版本

- 原始模型：`Wan-AI/Wan2.1-VACE-1.3B`，revision `574e6a744642ce3bee319afc31496b88bde8aac4`。
- 正式DiT/VACE文件：`models/Wan2.1-VACE-1.3B/<revision>/diffusion_pytorch_model.safetensors`。
- T5/VAE：`models/Wan-Series-Converted-Safetensors/` 下既有转换组件。
- Tokenizer：`models/Wan2.1-T2V-1.3B/google/umt5-xxl`；仅复用文本组件，不加载旧T2V LoRA或缓存。
- 实际环境：PyTorch2.11.0+cu128、DiffSynth2.1.5、PEFT0.17.1、safetensors0.7.0。
- DiffSynth是项目内vendor源码，没有独立Git提交号；实际run的experiment.json记录项目HEAD、相关项目脚本和vendor源码的小文件哈希及包版本。不能仅用未提交工作区的HEAD反推全部实际代码。

文本来源是 `configs/vace_p01_465_17f_v1.json`，训练正向描述与全部三个推理阶段保持逐字相同。

正向：

> 白色背景中一栋完整的灰色无纹理建筑三维模型，规则连续的墙面和屋顶，保持建筑形状与相机轨迹一致，固定俯仰视角平滑环绕，结构稳定，光照一致。

负向：

> 彩色纹理，窗户装饰，新增建筑，建筑形变，结构闪烁，背景变化，文字，水印

训练沿用标准SFT正向条件，负向文本只用于各推理阶段的CFG；推理CFG5不是训练损失的设置。

## 6. 固定训练参数

配置源：`configs/vace_roof_smoke_v1.json`。本次run另将真正使用的配置保存到唯一的 `training/experiment.json`，运行过程中不修改其config部分。

| 参数 | 本轮固定值 |
| --- | --- |
| 初始化 | 原始VACE，新建LoRA，不加载旧增量 |
| LoRA基础模块 | `vace` |
| 目标层后缀 | `q,k,v,o,ffn.0,ffn.2` |
| 匹配规模 | 150个Linear，300个A/B张量，10,936,320参数 |
| rank / alpha / 内置缩放 | 16 / 16 / 1 |
| dropout / bias | 0 / none |
| RSLoRA / DoRA | false / false |
| 初始化 | 标准PEFT零增量初始化 |
| 精度 | BF16骨干及autocast，FP32可训练LoRA参数 |
| 优化器 | AdamW，lr5e-5恒定，wd0，betas0.9/0.999，eps1e-8 |
| batch / 梯度累积 / workers | 1 / 1 / 0 |
| 总更新数 | 300次真实optimizer.step，P04/P06交替各150次 |
| checkpoint | 只在update100、300保存LoRA |
| 梯度裁剪 | max_grad_norm1.0 |
| gradient checkpointing / CPU激活offload | true / false |
| 训练seed | 20261018，进程初始化一次 |
| 训练scheduler | 本地官方Wan FlowMatch，1000训练时间步，采样范围0–1 |
| VAE分块 | tiled=true，tile_size30×52，tile_stride15×26 |

FP32 LoRA参数用于保留小学习率产生的更新，骨干计算仍用BF16混合精度。DiT参数冻结，但其计算保留反向链；不能将整个DiT放进no_grad或detach条件提示。

冻结T5的正向context、GT VAE latent、Damaged/Mask VAE context在本次进程内复用，随后将T5/VAE移CPU；不缓存可训练VACE输出，不建立新的全量磁盘缓存。

前5次真实更新累计检查至少一次有效梯度和参数变化，不要求每个LoRA张量首步都非零。官方scheduler存在一个零权重时间步，偶遇单步零loss/梯度不直接判失败。非有限loss、断链、非有限梯度或参数会停止；每步写入并flush loss.csv。

## 7. 公平比较与串行执行

| 顺序 | 工作 | 新增推理次数 |
| --- | --- | --- |
| 1 | 原始VACE分别生成P04/P06的17帧Base | 2 |
| 2 | 同一份vace LoRA训练300更新，保存100/300 | 0 |
| 3 | 训练worker正常退出后，使用LoRA-100推理两例 | 2 |
| 4 | fresh原始模型加载LoRA-300，推理两例 | 2 |
| 5 | CPU生成两例各四成果、验证后结束 | 0 |

全部推理固定17帧、512×512、16FPS、seed20261018、30步、CFG5、sigma_shift5、vace_scale1、denoising_strength1、cfg_merge=false、BF16、tiledVAE和CPU随机数设备。

推理LoRA倍率为1，仅加载到vace。full模式的本地LoRA加载会融合到基础权重，clear_lora不能撤销已融合增量，因此Base、100、300各使用独立GPU进程重新加载原始模型；禁止先融合100再叠300。

训练worker必须正常退出0且有完成标记，parent才进入评估。权重文件存在、更新循环结束或loss较低都不能单独当作整轮完成。整个首轮上限为6次17帧推理及一次300更新训练，不包含81帧重跑。

## 8. 输出文件的含义

~~~
run_20261020_165237_150325_2048720/
├── training/
│   ├── lora_update0100.safetensors
│   ├── lora_update0300.safetensors
│   ├── experiment.json
│   └── loss.csv
├── 04/
│   ├── 01_微调后补全.mp4
│   ├── 02_缺损_原模型_微调后_GT_对比.mp4
│   ├── 03_训练进展对比.png
│   └── 04_Mask验证.png
└── 06/
    └── 同样四项可视化
~~~

上述结构已实际产出并核查，共12文件；四个视频均17帧/16FPS、yuv420p/BT.709，四张PNG可读。原始视频512×512、比较2048×544，进展图2560×2720、Mask图2560×544。程序在发布前已完整解码并检查RGB/帧序；文档维护又只读复核视频计帧和PNG读取。

- 两个safetensors只包含vace LoRA增量，不能恢复optimizer、RNG或完整训练状态。
- experiment.json记录固定配置、输入路径、原始帧号、代码/模型版本、阶段退出码、实际更新数和可获得的显存峰值；训练进度通常每10步更新。
- loss.csv仅update/sample/loss/lr，逐更新flush；CSV可能比JSON多几步。异常退出极短写入窗口内只能确认已落盘的更新，不从进度条推测额外更新。
- 01为LoRA-300的17帧原始预测MP4，不进行Mask回贴或美化。
- 02为Before | Base | LoRA-300 | GT四栏同步视频。
- 03为Before / Base / LoRA-100 / LoRA-300 / GT五行，列为片段索引0、4、8、12、16，对应原始B000/B004/B008/B012/B016。
- 04取本段白区面积最大的视角：P04为B003，P06为B006，展示Before | ModelMask | Base | LoRA-300 | GT。

视频使用H.264 High/yuv420p/BT.709有限范围及faststart，防止旧RGB4:4:4交付在播放器中紫绿偏色。PNG保留原像素，MP4允许正常有损编码量化；面板只加标题，不单独缩放主体或调色。

Base/100/300的逐帧数组只保存在本轮独占系统临时目录供跨进程CPU制图，正常完成或受控失败后清理；不永久保留输入副本、Mask PNG、逐帧预测、额外日志或报告。原test/composite_check不改。

## 9. 查看、重连与失败处理

本轮已回到shell，可重连查看完成记录：

~~~bash
tmux attach-session -t vace_p01_17f
~~~

然后按Ctrl+b，再按3；Ctrl+b再按d仅分离，不中止计算。只读获取末尾：

~~~bash
tmux capture-pane -p -t vace_p01_17f:3.0 -S -60
~~~

查看记录，不调用训练器：

~~~bash
/home/shi/miniconda3/envs/scj/bin/python -B - <<'PY'
import csv
import json
from pathlib import Path
run = Path('/mnt/windowsE/chuanjun/building_wan/outputs/vace/lora_roof_smoke/run_20261020_165237_150325_2048720')
experiment = json.loads((run / 'training/experiment.json').read_text())
print(json.dumps(experiment['runtime'], indent=2, ensure_ascii=False))
with (run / 'training/loss.csv').open(newline='') as stream:
    rows = list(csv.DictReader(stream))
print('Recorded updates:', len(rows))
print('Last record:', rows[-1] if rows else None)
PY
~~~

入口无参数或--dry-run仅做CPU准备；--run前台执行，--tmux首次启动或重连通用会话。本轮已有完整结果，不能再次执行启动命令充当状态查询或自动重跑训练。

OOM、非有限值、断梯度、原生退出或媒体失败时，parent打印失败阶段/原因/退出码并保留已存LoRA。没有自动重试、降分辨率、改帧数、改rank/lr或续训。当前入口不支持完整状态resume，重新执行会创建全新run并重新做Base和训练，应先看实际失败原因再决定。

## 10. GPU利用率和显存怎么读

用户16:54:19截图显示GPU-Util100%、进程SM约99%、功耗298/300W、显存8229MiB/47.99GiB（16.7%）。16.7%是容量占比；GPU-Util是采样期间设备执行任务的活动比例，两者不能互换。

17帧/batch1、少量LoRA梯度及AdamW状态、gradient checkpointing、冻结T5/VAE编码后移CPU共同降低显存。冻结骨干参数仍需参与计算与反向链，显存少不表示模型没训练，也不应为填满显存改动本轮固定参数。

100%活动率和接近功耗上限支持当时处于高负载，但不等于理论FLOPS达到100%。截图中的86°C仅为一次观测，未核查时钟/限频原因，不据此声称已限频或需要调整硬件。

Base阶段实测峰值为15.865GiB allocated /17.309GiB reserved，训练阶段为15.139/15.393GiB。PyTorch allocated、reserved与监控工具进程占用的含义不同，阶段峰值也包含加载与冻结编码；不能从单帧8GiB截图推断峰值。

## 11. 如何判断有没有学会

先确认程序链路：实际更新300、每例150、两份LoRA可加载、训练worker正常退出、六次推理与两例媒体检查完成。再看可视化，分别回答：

1. 同条件17帧Base是否也有凹槽或黑洞；不能用旧81帧结果代替这一步。
2. P04在100/300步后屋顶是否接近平面、墙顶衔接是否连续，还是只把凹槽变浅。
3. P06洞口是否真正闭合为与GT接近的表面，还是仅颜色变亮。
4. Mask外原本正确的墙面、屋顶和背景有没有更差；本轮交付是原始预测，不回贴遮住副作用。
5. 片段内部是否有轮廓变化、闪烁；即使改善，也只说明这一段训练题的表现。

终端的Mask内RGB误差和已知区域变化是参考值，不能自动证明结构闭合。FlowMatch损失每步随机噪声与时间步不同，允许波动；既不要求单调下降，也不设loss阈值宣布成功。

允许结论为实现正常且视觉改善、实现正常但未明显学会、或训练未正常完成。300步只是首轮预算，没有改善不自动追加训练，也不能直接断言LoRA无效或必须换模块。

## 12. 代码入口与验证范围

| 文件 | 职责 |
| --- | --- |
| `configs/vace_roof_smoke_v1.json` | 固定源身份、选窗、文本、训练和推理配置 |
| `scripts/vace_roof_data.py` | 原始PNG读取、身份/相机/窗口与CPU预检 |
| `scripts/vace_roof_cpu_check.py` | meta真实层匹配、实际VACE条件Unit的CPU语义检查 |
| `scripts/vace_roof_training.py` | 官方FlowMatch、vace LoRA优化、早期梯度检查及增量保存 |
| `scripts/vace_roof_smoke.py` | 串行子进程、GPU锁、唯一配置状态、失败停止和CPU制图调度 |
| `scripts/vace_roof_results.py` | 两视频/两PNG、逐帧对齐和播放器兼容校验 |
| `scripts/run_vace_roof_smoke.sh` | 指定scj Python、离线依赖模式和可恢复终端 |

准备阶段294项CPU回归通过，包含真实tiny Wan/VACE与官方损失的反向链、GT隔离、Mask真用、LoRA增量重载、异常停止和媒体编码；这些测试使用临时合成数据/小随机模型。本次文档维护再次以空CUDA_VISIBLE_DEVICES运行294项CPU测试通过（45.607秒），语法、编译、diff检查通过；没有额外GPU计算。正式GPU执行证据独立记录在第1节，不能把CPU测试当正式训练结果。

本轮后续保持停止边界：不自动扩大数据、跑81帧、增加Depth/相机条件、换14B或追加训练。后续正式训练与测试划分不得将已经学过P04/P06的LoRA当作无泄漏初始化。
