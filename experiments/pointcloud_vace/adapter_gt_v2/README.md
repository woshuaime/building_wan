# GT 视频 V2：八视角结构 Adapter＋VACE LoRA，完整训练划分 3 epoch

本实验把**原样残缺点云视频＋同一残缺点集的八视角结构**映射到完整几何 GT 点云视频。从官方 Wan2.1-VACE-1.3B 新建 Adapter 和 LoRA，保留原 D 组结构与注入位置，不加载 D-3000 增量、旧优化器或弱监督目标。任务由独立后台程序串行完成导入、测速、选型、缓存、正式训练和最终保存；本入口不连接 validation/test 推理、视频展示或旧 `evaluate.py`。

本文件说明实现和运行方法，不代表某个 run 已完成导入、测速或正式更新。实际阶段、已消费样本及选型结果以该 run 的文件为准。

## 路径与启动

项目根目录为 `/mnt/windowsE/chuanjun/building_wan`，已有 Python 为 `/home/shi/miniconda3/envs/scj/bin/python`。输入目录为 `data/pointcloud_vace/pointcloud_video_pairs_v2/`，实测原点云复用 `data/pointcloud_vace/sample-data-organized/pointcloud/`。新代码位于本实验 `scripts/`；缓存写入 `cache/pointcloud_vace/adapter_gt_v2/`，每个 run 使用独立的 `outputs/pointcloud_vace/adapter_gt_v2/run_<时间戳>/`。

创建配置与启动使用如下入口；`RUN` 应替换为本次确定的绝对路径，续跑时保留原路径：

```bash
PYTHON=/home/shi/miniconda3/envs/scj/bin/python
SCRIPTS=/mnt/windowsE/chuanjun/building_wan/experiments/pointcloud_vace/adapter_gt_v2/scripts
RUN='/mnt/windowsE/chuanjun/building_wan/outputs/pointcloud_vace/adapter_gt_v2/run_<时间戳>'

"$PYTHON" "$SCRIPTS/configure.py" --run "$RUN"
"$PYTHON" "$SCRIPTS/run.py" launch "$RUN"
```

启动回执写入 `launch.json`，包含实际 tmux 会话、PID 及查看/停止/续跑命令。会话名为 `adapter_gt_v2_<时间戳>`：

```bash
tmux attach -t 'adapter_gt_v2_<时间戳>'
"$PYTHON" "$SCRIPTS/run.py" status "$RUN"
"$PYTHON" "$SCRIPTS/run.py" stop "$RUN"
"$PYTHON" "$SCRIPTS/run.py" resume "$RUN"
```

`status` 只读取状态；`resume` 使用同一 run 和同一 schedule。不要为刷新进度另起 `launch`。tmux 中按 `Ctrl+B`、再按 `D` 可离开查看而保持后台运行。程序不依赖 Codex 在线、SSH 连接、后续消息或人工放行文件。

大文件、临时文件及实际使用的 Hugging Face、Torch、XDG、CUDA、Triton 等缓存均指向项目所在 E 盘挂载目录；不修改 `HOME`。原数据、预训练模型及历史结果保持只读。

## 导入检查与固定划分

只按 `manifest.csv` 的 `id,partial_video,GT_video,camera,record,split_use` 读取正式数据，保留 `GT_video` 大小写。`check_20261115/` 的预览不会进入训练，也不执行上传包中的旧 Windows 脚本。

| 原始 split_use | 栋数 | 本轮用途 |
| --- | ---: | --- |
| train | 10,747 | 全部训练 |
| validation | 905 | 等待用户另行发起验证 |
| test | 661 | 保留独立测试 |
| diagnostic_only | 147 | 不训练 |
| exclude_boundary | 3,421 | 不训练 |
| unpaired_GT_only | 9 | 不训练 |

导入核对全部 15,890 条清单、15,881 残缺视频与 15,890 GT 视频（合计 31,771）、相机/record 数量、`records.video_bytes`、编码/尺寸/帧率/帧数、ID、split、来源与原点云头部点数。找回原文件后写 `import_map.json`，不改写来源 JSON 的 Windows 路径；实际训练清单和固定窗口分别写入 `train_manifest.json`、`schedule.json`。损坏或缺失文件明确报错，不静默删样本。

若存在本次指定上传 ZIP，复用已解压的最终目录，只补缺失文件；逐成员检查归档 CRC，并核对解压文件的大小和 CRC。全部导入检查通过后才删除这次上传 ZIP，保留解压数据，不删除历史压缩包。CRC 是文件完整性检查，不是视觉效果验收。

全量容器头部核查可能约需十几分钟，另有 ZIP/解压文件 CRC 的磁盘读取时间；这只是准备阶段的量级参考，不是已测出的 GPU 训练耗时。`preparing`、`autotuning`、`caching` 均不表示正式 optimizer 更新已经开始。当前细节写入 `progress.json`、`import_progress.json` 并打印到 tmux。导入期间也响应 `stop.request`。

## 灰度视频、窗口与结构输入

输入固定为 832×480、81 帧、16 fps，点色 `[80,80,80]`、背景 `[240,240,240]`，不缩放、不重新渲染。上传元数据为 `gray`、全范围、CRF 0；当前 FFmpeg/PyAV 将实际 H264 High 4:4:4 Predictive 文件报告为 `yuvj420p`、`color_range=pc`。解码直接复制全范围原始 Y 平面，再复制为三通道送模型，避免有限范围颜色转换。后续解码继续定位损坏文件。

固定 seed 为 `20261115`。每个 epoch 中每栋训练建筑恰好出现一次，以 `random.Random(seed+epoch)` 打乱原 train 清单顺序。对字符串建筑 ID，定义：

```text
u = int(SHA256(UTF8(str(seed) + ':' + id)), 16) % 64
start = (u + 21 * epoch) % 64          # epoch 为 0、1、2
frame_ids = start ... start+16
```

partial、GT、目标相机共用同一组索引，只使用 000–079，排除作为闭环复制的 080。同栋三个 epoch 使用三个不同窗口，总量严格为 `10,747×3=32,241`。batch 1 时为 32,241 更新；batch 2 时每 epoch 最后单样本不丢弃，共 16,122 更新。恢复及改 batch 均以已提交样本游标为准，不把更新数当样本数。

八视角结构只取 `records.selected_source_point_indices.partial` 指定的实测显示点，保留原文件点序。世界 XYZ 在 float64 下减 `cameras.world_to_local_origin` 后才转换模型精度；`GT_PLY_local_to_world_offset` 不参与 partial 坐标处理。本轮不读取 mesh 或完整 GT PLY，也不依赖旧任务目录。

使用冻结 `camera_poses_local` 的 000/010/020/030/040/050/060/070 构建 416×240 深度/有效图，保持 50° 垂直视场和上传相机方向。图像索引坐标的主点为 `(415.5,239.5)`；缩小后的像素坐标使用 `(uv+.5)/2-.5`，同步保持内参和像素中心约定。相机不按包围盒重算。锚点沿用 D 组 FPS/密度定义；显示点少于 128 时确定性重复有效锚点，固定形状以支持不同建筑组合的真实 batch 2。

## 条件、目标与损失

`vace_video` 是未经填灰的 partial 17 帧；mask 显式全 1，允许整幅生成。保留官方 inactive/reactive 编码和拼接语义，全 1 mask 不代表删除 partial 条件。GT 仅进入监督目标与区域权重，不作为条件、参考首帧、结构输入或初始视频；训练目标不回贴 partial，不用 partial/GT 像素相减定义缺口。

沿用原版 Flow Matching 速度目标和 scheduler 时间权重。GT 灰度 `<239` 定义前景，用 7×7 方形结构元素向外膨胀 3 像素；其余为背景。先对 8×8 空间块取面积平均，再按 VAE 的 `第0帧；1–4；5–8；9–12；13–16` 聚合，得到两个区域的 `[2,5,60,104]` 分数面积。该网格不同于 Adapter 的 `[5,30,52]` token 网格。前景与背景 MSE 各按面积归一化后以 1:1 合并，空区域跳过，再乘该样本 scheduler 权重，最后 batch 内平均。区域权重不进入模型条件。

固定正向文本保存为 `prompt.txt`，negative 为空：

> A static building point cloud made of individually distinguishable sparse neutral gray dots on a uniform light gray background. The complete roof and walls are represented by separate dots with visible gaps between them. The camera moves smoothly around the building while the three-dimensional arrangement of the dots remains stationary. The point size, neutral gray appearance, and camera motion match the input point cloud video.

## 训练参数与自动选速

原版骨干、VACE、VAE、T5 冻结；只训练新八视角 Adapter 和 15 个 VACE 控制块 self/cross attention 的 q/k/v/o LoRA，rank=alpha=8，保持零残差初始化。实际 Adapter/LoRA/冻结参数数量记录在 `optimizer_ownership.json`。增量仍需完整反向传播；零投影导致上游首步梯度为零时，在后续完整更新检查梯度链路。

- 单 A6000、单进程；batch 在 1/2 中实测选择，累积固定 1。
- BF16 骨干/autocast；Adapter、LoRA、AdamW 状态为 FP32。
- AdamW：lr `1e-4`，betas `(0.9,0.999)`，eps `1e-8`，weight decay 0，clip norm 1。
- 按已消费样本的前 2% 线性 warmup，随后 cosine 降至 `1e-5`；batch 变化不重复 LR 进度，不自动放大学习率。
- Flow Matching 1,000 时间步，shift 5，vace_scale 1；每样本独立时间步和噪声。
- VAE 初始 latent tile_size `[30,52]`、stride `[15,26]`；当前实现记录对应像素轴序与实际设置。
- 初始加载 workers 0；不训练 EMA，不启用多卡、DeepSpeed 或 8-bit optimizer。

流水线先用多栋真实 train 窗口（含较多显示点的建筑）测缓存编码成本、空间及数值一致性，然后按独立子进程串行比较 batch 2＋完整 checkpointing、batch 1＋关闭/部分 checkpointing、batch 1＋完整 checkpointing；余量允许时再测 batch 2 的部分/关闭方案。每个候选至少 5 次热身、20 次稳定计时，覆盖 backward、clip、AdamW 首次状态分配和 CUDA 同步。

整卡预算为 `min(40 GiB,实际总显存−6 GiB)`；同时记录 torch allocated/reserved 峰值与 NVML 整卡峰值。合格条件包含数值有限、梯度链路、真实 batch 2 独立相机/样本等价检查，以及兼容 SDPA 注意力路径。按 samples/s 和**缓存准备＋训练＋保存的预计总时间**选择，差距小于 5% 取峰值显存较低者。选型只表示本轮实测候选中的最快者，不代表穷尽所有设置。

`autotune.json`、`autotune_candidates/` 与 `selected_config.json` 保存实测结果及失败原因；未选出前不能把某个候选写成最终配置。测速更新不计入正式 epoch，训练前恢复同一份干净初始化与 RNG。

## 缓存边界

T5 固定文本只编码一次；八视角原始深度/有效图、锚点、选中 partial 点及相机可缓存。几何缓存只持久保存八个源视角，不为全部窗口保存目标深度；每次按 partial 点与冻结目标相机重建该窗口深度，控制磁盘需求。

可选冻结 latent 缓存严格编码最终 17 帧窗口，不从整段 81 帧 latent 直接切片。target latent、partial reactive latent、区域面积及共享 inactive 编码均记录身份；共享 inactive 先实际编码，不能假定为零。低精度缓存与在线路径进行读回及小批重新编码数值核对。若全量缓存的预计总成本更高或空间不足，选在线编码，保留文本/几何缓存。

缓存键纳入源文件、建筑/帧索引、相机/record/渲染、模型与 VAE 设置、prompt、条件和损失版本；临时文件完整写入后原子发布，按需加载。Adapter CNN、视图融合与投影输出从不缓存后 detach；正式每步都重新计算并参与梯度。

## 恢复、产物与完成边界

项目公共 GPU 锁为 `logs/.wan_a6000_gpu.lock`，另有本 run 独占锁；GPU 被其他任务使用时写 `waiting_for_gpu`，不会终止别人。正式第一次成功更新保存完整恢复点，此后每 200 更新或 5 分钟（先到者）、epoch 结束及最终结束保存。恢复状态包含两类增量、AdamW、RNG、样本/epoch 游标、LR 进度、配置、schedule/数据身份，保留 latest、previous 和三个 epoch 结束状态，不重复存冻结基座。

`stop` 写请求，训练在完整更新边界保存并停止 supervisor；不会被自动恢复当作故障重启。OOM 结束当前 worker，由 supervisor 从已完整提交的恢复点使用已测合格的更省显存候选续跑，先增加 checkpointing、再从 batch 2 降至 1，最多自动恢复两次。非显存错误记录真实阶段、建筑及窗口，不用持续降 batch 掩盖问题。异常写 `failure.json` 和真实退出码，保留已有产物。

run 包含导入映射/清单/schedule、`config.json`、源码/基座身份与文本、选型结果、缓存索引与耗时、`train.log`、`loss.jsonl`、`progress.json`。`checkpoints/` 包含 Adapter/LoRA 初始化、`adapter_epoch1/2/3.safetensors`、对应 LoRA、`resume_epoch1/2/3.pt`、`resume_latest.pt`、`resume_previous.pt`，以及 `adapter_final.safetensors`、`lora_final.safetensors`。具体文件位置和完成量以本 run 实际内容为准。

只有 32,241 个计划样本全部提交、三个 epoch 完成且最终权重保存成功，才写 `training_complete.json` 并标记 completed。loss 下降和文件可读不等于视频效果通过。**后台自行完成 3 epoch；本次及训练结束时均不执行验证，完成后等待用户另行发起。**

## 检查与参考

CPU 回归入口：

```bash
CUDA_VISIBLE_DEVICES='' /home/shi/miniconda3/envs/scj/bin/python -m unittest discover \
  -s /mnt/windowsE/chuanjun/building_wan/experiments/pointcloud_vace/adapter_gt_v2/scripts -p 'test_*.py' -v
```

接口依据：[官方 VACE inactive/reactive 实现](https://github.com/ali-vilab/VACE/blob/main/vace/models/wan/wan_vace.py)、[官方基座](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B)、[PyTorch 2.11 checkpoint](https://docs.pytorch.org/docs/2.11/checkpoint.html)、[PyTorch 2.11 SDPA](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html)。记录实际本地基座身份，优先已有 revision `574e6a744642ce3bee319afc31496b88bde8aac4`；原基座 SHA256 为 `c46a6f5f7d32c453c3983bbc59761ea41cd02ad584fb55d1a7ee2b76145847a2`。这些链接供接口核对，不要求下载新版或升级环境。

## 2026-11-18：同一 run 的缓存执行加速

用户询问并要求处理缓存过慢。固定 run 仍为 `outputs/pointcloud_vace/adapter_gt_v2/run_20261117_225803`；原训练配置、初始化、划分、窗口、全部 32,241 样本、原缓存和历史选型记录均继续使用。原 supervisor 在完整缓存文件边界主动停止，退出130；此时正式 optimizer 更新仍为0。该退出表示有序切换执行程序。

新增 `scripts/fast_vae.py` 只调整同一九块空间 tile 的执行批量、复制和拼接位置，不改 tile_size/stride、时序编码、模型权重或训练定义。`scripts/cache_acceleration.py` 在同一 GPU 锁和 tmux 中串行测试原编码、GPU拼接、2/3/9块批量；各候选5次热身和20次稳定计时，每份结果与原始缓存按BF16 `rtol=.005, atol=.005`核对，区域权重和身份要求精确相同，未通过者排除。不同候选为独立进程；不并发争用GPU，不放宽显存预算。原编码也是候选，不能承诺加速倍数。

一次真实窗口的临时启动检查中，GPU拼接与原缓存完全相同，编码3.731秒；3块批量为3.454秒，但最大latent偏差0.0390625，未通过既定逐元素数值阈值，不能据此直接采用。该单窗时间不代表稳定吞吐；最终结果由后台写入 `cache_optimization/results.json` 和 `selected.json`。GPU输入复制优化也须经过后续全候选检查。

新增源码身份及副本单独写入 `cache_optimization/source_identity.json`；原 `config.json`、`source/` 与原始 `autotune.json/selected_config.json` 不覆盖。已发布缓存直接复用；不重新做已完成的全量导入，不把临时测速算进正式更新。所选编码完成剩余缓存后，自动调用原正式训练/恢复代码；仍不执行效果验证。运行时缓存OOM可退回已测合格原编码，保留所有完整缓存文件。

本 run 的 `stop.sh` 保持原用法，`resume.sh` 指向新的独立缓存/训练 supervisor；使用这两个入口可保留执行选择和既有进度。tmux 会话仍为 `adapter_gt_v2_20261117_225803`，新增窗口 `cache-speed`。关闭查看终端不影响后台继续执行。


## 2026-11-18　整帧 VAE 缓存重建与训练交接

本轮原有 9 个重叠分块在每个窗口对 GT、partial 各编码一次。三栋真实训练数据 1163、7161、6497 的短测中，每个窗口两路编码合计约 3.65–3.81 秒；保持分块的 CUDA Graph 为 3.54–3.55 秒并逐值一致。整帧一次编码同一段 17 帧，三栋耗时 1.53–1.69 秒，特征相对旧分块的 L2 差为 0.84%–0.90%。用户选择用整帧重新生成全部窗口。

东京时间 04:14 接管原 run。原分块 supervisor 在完整缓存边界退出 130；2,893 个分块文件及其配置保存在旧目录。新格式配置身份为 `0636d11cf38ac7030ac1bff39030a5af661493e7fe1c2837489e8dfe7d64bc19`，缓存目录为 `cache/pointcloud_vace/adapter_gt_v2/0636d11cf38ac7030ac1bff39030a5af661493e7fe1c2837489e8dfe7d64bc19/`。共享文本沿用相同 prompt 的已冻结结果，公共 inactive 用全零输入重新经整帧 VAE 编码。CPU 线程提前准备下一个窗口，每次保持一个预取任务。GT 目标与 reactive 条件分别编码，训练读取时检查新身份、帧号、精度、形状以及有限值。八视角几何、三个 epoch、原始 AdamW 初始化及 32,241 个样本的顺序仍使用原 run 的记录。

东京 04:15 已完成 25 个不同真实 train 窗口的端到端检查，包含点数较多的建筑、5 次预热与 20 次稳定计时。准备、编码、原子写盘及读回合计 **2.116 秒/窗口**；按此估算全量缓存约 **19.0 小时**，预计窗口文件约 67.5 GiB。整卡峰值 5.17 GiB，分配和预留峰值为 3.47/4.26 GiB，整卡预算 40 GiB。1163、1640、3 三栋的 GT latent 与全部 96 个 VACE 条件通道，同官方 `tiled=False` 路径比对的最大绝对误差均为 0。正式更新为 0，自动程序用新缓存复核此前合格的训练配置，再生成其余缓存并进入训练。资格检查写入的 25 个窗口直接计入整帧缓存。

`cache_execution_v2/` 保存四组临时编码对照；`cache_fullframe/` 保存生效配置、原始输入指纹、正式编码核对、测速、训练选型及源码副本（均位于 run 内）。根 `config.json`、`autotune.json`、`selected_config.json` 为最初分块方案，整帧选型的 `selected_config.json` 位于 `cache_fullframe/`。本轮运行代码位于 `scripts/cache_encoder.py`、`scripts/fullframe_cache.py` 和 `scripts/run_fullframe.py`。

```bash
tmux attach -t adapter_gt_v2_20261117_225803
# 会话内切换至 fullframe 窗口
# 从项目根目录只读查看当前阶段
cat outputs/pointcloud_vace/adapter_gt_v2/run_20261117_225803/progress.json
```

本轮继续使用 run 自身的 `stop.sh` 和 `resume.sh`。执行入口将在独立子进程中为正式训练及恢复加载同一份整帧配置，采用完整 optimizer/RNG 检查点并以样本游标计数。正式效果待用户另行发起验证。本次启动检查包括 298 项项目 CPU 回归、19 项数据/流水线检查和 17 项模型/缓存检查。


## 2026-11-18　按用户要求清理旧分块缓存

东京 04:26:14，实际删除初始分块配置 `263dc3b4dea8590925b2551d2440a96b228a78ef5af8a6456b562a221c08e8db` 下 `latents/` 中的 2,893 个窗口，释放约 6.51 GB。保留该配置目录的 `shared.pt`（5,194,533 字节），供整帧需要重新校验时复用文本条件。整帧缓存写入独立目录，清理结束时该目录已落盘 231 个窗口。此前分块路径和已落盘数量属于历史记录，旧分块窗口已无可复用副本。


## 2026-11-19　A6000 正式训练提速实测

整帧缓存于东京 00:25 完成，共 32,241 个窗口。正式训练采用单卡 batch 1、关闭梯度重算的 B 配置。第 1,221 次更新前最近 300 步的平均墙钟耗时为 2.608 秒，其中训练日志内的步骤计时为 2.143 秒。此前最慢两条相邻记录相隔约 165 秒，因此用启动以来均速推算曾显示将近 30 小时。观察机械盘上 9 次正式进度发布，临时文件创建到原子替换平均耗时 0.445 秒。

在同一 run 锁与 GPU 锁内，从第 1,221 次更新的完整 AdamW/RNG 恢复点对相同的后续 25 个窗口分别执行 5 步预热和 20 步计时。每个候选使用独立进程与独立进度、损失文件，计时包含读取、前反向、优化器、进度发布与显存查询。正常训练环境下测得如下结果。

| 候选 | 墙钟秒/步 | 数据读取秒/步 | GPU 更新所在阶段秒/步 | 进度等记录秒/步 |
| --- | ---: | ---: | ---: | ---: |
| 原训练及记录路径 | 2.564 | 0.090 | 2.030 | 0.441 |
| 原训练计算，优化进度发布与显存查询 | 2.077 | 0.019 | 2.053 | 0.002 |
| 再启用合并的位置旋转计算 | 2.004 | 0.019 | 1.979 | 0.002 |

后测使用相同窗口，文件系统页缓存较热。约 0.44 秒的差异可以直接定位到进度等记录，不能把全部时间差归给同一项。进度优化仍以临时文件原子发布，每一步都有记录。省去强制刷盘的范围只有 progress.json 和 current_batch.json。完整检查点、最终权重和故障记录沿用原保存流程。整卡显存仍按同一预算和频率采样，训练进程内复用 NVML 句柄。

第 1,221 步上的首次端到端核对中，原方案与两个候选的 RNG 一致，但 25 步后的参数变化和梯度超过严格复算阈值。原方案本身的两次计算也有差异，25 步逐样本顺序、时间步、学习率、首个损失都一致，最大相对损失差 0.42%，总梯度范数差 5.56%。旋转算子的单独核对覆盖实际形状 1×7,800×1,536，正反向 BF16 输出逐元素一致。独立 CPU 重放 25 步进度文件与当前批次文件，50 份发布内容逐字节相同。第 1,262 步的重复测试启用 PyTorch 确定性警告，Flash Attention 仍报告非确定性反向，该轮严格比对未过。

实际证据在本 run 的 training_speed/。首次三项测速与原版重复结果在 trial_1795019701988808750/，后续对照在 reporting_ 和 strict_reporting_ 开头的目录。每个目录中的 loss.jsonl 等记录只用于测速。正式训练游标只读 run 根目录的进度、损失和 checkpoints/ 完整恢复点。

第 1,271 次更新的严格对照使用同一检查点、同一 25 窗、确定性 GPU 注意力；仅在这两个测速子进程内把缺少确定性 CUDA 实现的小型 grid_sample 反向投影放到 CPU。前 5 步预热后，原记录与优化记录的完整步长分别为 3.394 和 2.937 秒。两边 25 步的损失、梯度范数、最终 Adapter/LoRA 参数增量和梯度、Torch CPU/CUDA RNG 均一致。严格测速的绝对速度仅适用于该对照。正式运行恢复原数值路径与 GPU 投影，只采用进度写盘和显存查询优化。新的选择在 training_speed/selection.json，核对依据在 strict_reporting_1795020215436790847/qualification.json。原 B 配置及 40 GiB 预算、每次 1 个窗口、完整 AdamW/RNG 保存策略继续生效。

东京 01:48 正式进度 1,323/32,241。剔除测速已读取的首批窗口，正式第 1,297–1,321 步平均 2.515 秒/窗口，优化前最近 300 步平均 2.608 秒。按这批实际速度，剩余约 21.6 小时。整卡峰值 32.79 GiB。

机械盘首次读入某些窗口会继续增加等待。为评估这段时间，独立 CPU 进程按正式 schedule 最多提前 4 个窗口，仅读取已经存在的整帧 latent 和几何文件，让系统保留最近读过的页。正式进程继续执行自己的读取与身份核对。全程使用正式样本，不暂停计算。先开启 32 步，再关闭 32 步，各取末 25 个连续正式步的墙钟间隔；两项均值和中位数都改善至少 5% 才自动保留。证据写入 training_speed/readahead.json。

东京 01:53 已正式更新 1,439/32,241，回读第 1,402 步完整 optimizer/RNG 恢复点成功。正式训练中提前读取的第 1,365–1,389 步平均 2.089 秒，中位数 2.090；关闭后的第 1,397–1,421 步平均 2.837 秒，中位数 2.293，该段含一次计划保存。预取已自动保留。相对优化前 300 步均值 2.608 秒，近期正式每步耗时约低 20%。按开启时的实测速度估计，还需 17.9 小时，磁盘停顿和保存可能延长耗时。

继续启用后再核对正式第 1,428–1,452 步，25 个新窗口的墙钟均值为 2.107 秒。正式轨迹的样本游标连续。项目 CPU 回归 298 项、专项流水线 19 项和模型/缓存/进度检查 20 项通过。
