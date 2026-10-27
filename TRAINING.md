# 训练操作与持续更新记录

最近补记：2026-10-23（UTC+08:00，北京时间）。本文件持续保留各阶段使用过的命令、环境、续训方法、退出问题与后续操作修正。历史正文已恢复在原文件内，后续进展按日期追加；归档仅作备份，无需离开本文件才能阅读旧记录。

阅读顺序：[此前累计正文](#history-records) → [2026-10-23阶段更新](#update-20261023) → [本次历史恢复及生效约定](#history-repair)。旧段落的“当前”“下一步”、命令和PID按其记录时点理解；较晚的更正与实际完成证据在后文，旧命令不因此重新执行。

截至本次修复：Pilot8系列及Complex12的C/U/S各300更新、五方法60条81帧已完成。本次仅修复文档；[实验入口](experiments/README.md)和[文档索引](docs/DOCUMENTATION_INDEX.md)仍可直接使用。

<a id="history-records"></a>

## 一、此前累计正文：截至2026-10-22的累计记录

以下完整保留原有记录顺序、参数、命令、过程和当时结论，只调整标题层级。原记录中的未完成状态与后来完成状态共同构成历史，不能用后者替换前者。

<!-- RESTORED_HISTORY_BEGIN -->
### Training

更新时间：2026-10-22T20:00:17+08:00。Complex12 Rev1的B/L基线均已完成12/12条，共24轨迹/1944原始PNG；当前C训练实际9/300次更新，U/S尚未开始。实验尚未完成；后续固定三组step300评测。

#### 当前：Complex12 Rev1十二栋单轨小试（执行中）

- B/L阶段已齐：24条81帧/1944 PNG；L的12个worker全部exit0，逐条验证旧LoRA确实改变vace权重，scale1，来源仅P04/1227、P06/1848旧训练。C已从新共享初始化启动（此快照实际9/300更新）；U、S随后各自新建优化器并使用同表300行。100/200只留完整恢复状态，正式评测只用step300；不额外导出100/200的仅LoRA版本。实时训练更新见run/training/C_progress.json及progress.json。

- Base阶段结束快照：12条完整轨迹、972张512×512原始PNG，累计模型加载至保存2217.2秒，峰值allocated17.140GiB；11个worker正常exit0，CS04保存后exit-6但81帧均独立验证可用，没有重跑。L阶段已启动；终端会逐条打印唯一旧LoRA绝对路径、vace分支和scale1，并验证实际权重变化。

- 18:54:17 B/CS04完成30步、VAE解码、81 PNG保存及读回后，发生`free(): invalid next size (fast)`，worker退出-6、父任务停止。独立CPU再次解码全部81帧，确认配置/尺寸/数量一致，保留有效输出，不重新消耗GPU；异常事实保存在run的`logs/errors.md`和`checks/B_CS04_post_exit_validation.json`。后续只继续缺失任务，使用已有GDB退出监测器采集真实退出码和异常堆栈；4项CPU退出监测测试通过，未更换环境/模型/训练参数，未声称根因修复。

- 本轮明确授权来自2026-10-22任务书及9节补充交付规范；其C/U/S各300次训练授权仅适用于本轮，不受上轮“只推理”限制。范围固定12栋、512×512、单轨81帧、16FPS；不扩数据、不加seed/预算、不改骨干。
- Run：`/mnt/windowsE/chuanjun/building_wan/outputs/ComplexSingleOrbit_Rev1_Pilot_20261022`。数据：`/mnt/windowsE/chuanjun/building_wan/data/buildingfiller_import/Complex12/BuildingFiller_Complex12_512_Rev1_20261022`；原包SHA校验并解压，包内verify_package.py --decode-images退出0，5832张PNG/12栋、72闭环对及Rev1人工accepted已确认。原始数据保持只读；本轮脚本、缓存、配置、权重、预测和媒体都在run内。
- 划分：训练CS01/03/05/06/07/09/10/11，验证CS02/04/08/12；CS09–12沿用Rev1局部缺口。全部A2 Medium、35°仰角；000–079独立视角，080输入闭环重复，预测080真实保留。
- B=原VACE-1.3B；L=`outputs/vace/lora_roof_smoke/run_20261020_165237_150325_2048720/training/lora_update0300.safetensors`，仅迁移参考；C/U/S从同一份新初始化开始，各自新建AdamW。rank=alpha16、vace q/k/v/o/ffn.0/ffn.2、FP32 LoRA/BF16骨干、lr5e-5/wd0/clip1、batch1/accum1/checkpointing，训练seed20261022。固定300真实更新，只比较step300。
- 共同采样表300行：先均匀抽训练建筑，再抽Exact非空17帧窗口，起点0–63；保留内部黑帧，E空不排除，不使用080。共同时间步/噪声/其他随机种子，GT与条件先选17帧再编码。C原损失；S=1+2E；U=1+(2sumE/sumM)M，额外总权重相等；E=7×7膨胀raw_visible后与Exact相交。M/E映射本窗口[0]/[1–4]/[5–8]/[9–12]/[13–16]及8×8 max，实际loss网格1×5×64×64。
- 所有推理81帧/30步/CFG5/shift5/seed20261018、统一用户英文prompt、原中文negative、LoRA scale1。每条轨迹fresh进程/原始Base，只读取Damaged+Exact派生条件与文本；GT RGB/结构不进入推理，不回贴或强制闭环。
- 实际联调：CS06、CS09各C/U/S临时更新，共6次，loss有限/梯度有效/参数改变；不计入正式预算。CS09 Base81已退出0，179.3秒，峰值17.140GiB allocated，81原始PNG可读并抽看五视角；计入60条。其余仍在运行，不能写成训练/全评测成功。
- 顺序：联调→CS09 B81→其余B/L基线→C/U/S训练→三组各12条81帧→指标/可视化→视觉复核与中文结论。真实终端为`vace_p01_17f:5.0`，窗口`complex12_rev1`，pane `%5`；父PID 3148978，实时打印真实步数/耗时/已完成轨迹数。串行GPU锁，无第二个GPU任务。
- 输出入口`00_看结果/`：每栋2×4同步视频（Before/GT/B/L；C/U/S/空位）及7行5列总览；目标60轨迹/4860原始PNG、12视频/12总览。正式权重`checkpoints/C|U|S/step300.safetensors`及必要恢复状态；metrics按000–079、楼内像素加权、跨楼等权，train/val分开，闭环单列。尚未完成最终交付。
- 检查：298项项目CPU回归通过（51.000秒）；4项配对/损失/标签检查、4项指标聚合/Sobel检查通过；语法、编译、diff检查通过。仅旧L训练prompt的来源字段曾因浅拷贝写成当前英文，已纠正并记录`provenance/config_metadata_correction.json`；实际训练/推理设置没有变化，既有结果有效，无GPU重跑。

更新时间：2026-10-20 20:09:57 +09:00。LoRA-300八例81帧已经全部完成，八个worker和两个批次均退出0；当前只需查看成果，不再启动推理或训练。

#### 当前：查看已完成的LoRA-300八例81帧结果

结果根目录为 `outputs/vace/lora300_eval81/run_20261020_194300_145372/`，每例01–08恰好四项媒体。P04/P06扩展到81帧仍明显优于Base；六例迁移中P05/P07/P08明显改善，P01/P02有限，P03部分视角变差。详细判断及指标见EXPERIMENTS.md。

真实终端为已有会话 `vace_p01_17f` 的窗口4 `lora300_eval81`，pane `%4`，已回shell。以下只读查看已完成记录：

```bash
tmux attach-session -t vace_p01_17f
# 会话内 Ctrl+b，然后按 4
tmux capture-pane -p -t vace_p01_17f:4.0 -S -60
```

已执行两阶段，以下仅记录复现方式，不是新的启动指令：

```bash
WAN_PYTHON=/home/shi/miniconda3/envs/scj/bin/python bash scripts/run_vace_lora300_eval81.sh --run-trained
# P04/P06各81帧完成后，已逐帧确认改善并在终端打印结论，才执行：
WAN_PYTHON=/home/shi/miniconda3/envs/scj/bin/python bash scripts/run_vace_lora300_eval81.sh \
  --run-transfer --output-root /mnt/windowsE/chuanjun/building_wan/outputs/vace/lora300_eval81/run_20261020_194300_145372
```

首次 `--run-trained` 曾在P04推理成功后的CPU制图阶段失败；用户明确授权修复后重跑，上述成功run因此使用时间戳子目录。修复已提交 `ea51376`：制图前统一NumPy像素，01原始视频先保存；如将来CPU制图失败，可用 `--render-only 04 --output-root <对应根目录>` 从保留的01补齐三项比较，不调用模型。当前八例均完整，无需补图或重跑。

本轮入口默认无参数或 `--dry-run` 只做CPU检查，固定唯一已完成LoRA-300/vace/scale1，不包含训练分支；fresh worker串行运行并打印实际30步/耗时/显存/结果/退出码。八例Base已有同条件结果均复用。两个批次369.2/1129.4秒，峰值17.141/25.832GiB allocated/reserved；新训练0、Base补跑0，含首次失败共9次LoRA推理，成功交付8例。没有自动重试、分段、降分辨率或追加实验。

每例四文件为 `01_微调后补全.mp4`、`02_Base_微调后_GT_对比.mp4`、`03_关键帧总览.png`、`04_Mask验证.png`；四栏顺序Before/Base/LoRA-300/GT。512×512主体不独立缩放，视频81帧/16FPS、High/yuv420p/BT.709，完整解码及帧序验证通过。总览均用000/020/040/060/080，Mask图按最大可见白区选真实帧号。RGB误差只作辅助，不能代替结构/时序检查；EXPERIMENTS.md使用Base/LoRA最终原始视频双方解码后的统一指标。

#### GPU运行后的固定交接流程

用户已要求每次GPU训练/推理之后主动执行：核对真实run和退出状态 → 更新PROJECT_CONTEXT/EXPERIMENTS/TRAINING/SESSION_HANDOFF及相关专项说明 → 检查本轮差异 → Git提交并推送main。运行中写时间快照，结束后补最终结果；失败也记录，不能提前写成功。默认只提交相关代码、配置和文档，不提交模型/LoRA、数据、成果媒体或日志，也不把记录要求当作追加GPU工作的授权。详见AGENTS.md第7节。

#### 当前：P04/P06 17帧LoRA验证

详细参数、数据关系、成果布局及判断标准见 [P04/P06 LoRA实验说明](docs/VACE_P04_P06_LORA_SMOKE.md)。

##### 查看本轮结果与终端记录

本轮实际复用的是旧会话 `vace_p01_17f`，窗口3。断线后执行：

~~~bash
tmux attach-session -t vace_p01_17f
~~~

进入后按 `Ctrl+b`，松开再按 `3`。只读查看末尾进度：

~~~bash
tmux capture-pane -p -t vace_p01_17f:3.0 -S -60
~~~

实际run为 `/mnt/windowsE/chuanjun/building_wan/outputs/vace/lora_roof_smoke/run_20261020_165237_150325_2048720/`。阶段、更新数和完成状态看其 `training/experiment.json`，逐更新记录看 `training/loss.csv`；本轮均已完整记录300次更新，四个GPU阶段exit_code均0。运行中JSON训练进度约每10步更新、CSV每步flush，可能短暂相差数步。

先打开04/06各自的 `02_缺损_原模型_微调后_GT_对比.mp4`，再看 `03_训练进展对比.png` 的100/300差别。实际总耗时749.247秒，训练峰值allocated15.139GiB、reserved15.393GiB；全流程峰值15.867/17.385GiB。五个代表帧初步显示两例结构闭合改善，完整视频人工验收仍待完成。

##### CPU检查与首次启动

CPU预检（默认也是此模式，不加载正式模型或初始化CUDA、不创建实验输出）：

```bash
bash /mnt/windowsE/chuanjun/building_wan/scripts/run_vace_roof_smoke.sh --dry-run
```

以下为首次启动完整实验的命令说明；本轮已经启动，当前不要重复执行：

```bash
bash /mnt/windowsE/chuanjun/building_wan/scripts/run_vace_roof_smoke.sh --tmux
```

在已有tmux内会复用当前会话并打印准确重连命令；普通SSH终端新建/重连 `vace_roof_smoke`，断开后用 `tmux attach-session -t vace_roof_smoke`。仅前台运行可用 `--run`，无参数绝不启动GPU。不要在正在运行时重复执行。

- 固定P04/P06均B000–B016，512×512/16FPS，GT训练监督与Damaged/Mask条件隔离。配置、完整正负文本、alpha/rank和dtype位于 `configs/vace_roof_smoke_v1.json`；不得用旧T2V入口或旧LoRA。
- 单卡顺序：Base两次17帧→同一份vace LoRA交替两例300 optimizer updates→100与300权重各两次17帧→CPU比较。每次推理30步/CFG5/seed20261018，fresh阶段加载保证LoRA不累计。训练官方FlowMatchSFTLoss、只优化vace A/B，BF16混合精度加FP32 LoRA参数；冻结DiT仍保留反向链，编码完成后T5/VAE移CPU。
- 训练前5次实际更新累计确认有效梯度和参数变化；官方训练scheduler包含零权重时间步，单步零loss/梯度不直接误报失败，但持续断链/非有限值必须停。每10步显示平均loss，loss.csv逐更新flush；只有100/300保存增量，不冒充optimizer/RNG恢复状态。
- 输出独立run目录，training仅两个safetensors、experiment.json、loss.csv；04/06各原始LoRA300视频、四方视频、五行训练进展PNG和最大Mask视角五栏验证PNG。视频17帧、H.264/yuv420p/BT.709，不回贴、不用GT替换预测。异常退出保留权重，整体标FAILED；不会自动续训、重试、降分辨率或变更帧数。
- CPU294项回归通过不代表正式GPU训练通过。终端打印两例像素误差仅供参考，凹槽/洞口是否真正闭合、是否闪烁仍需查看同条件Base/100/300可视化，不按loss阈值自动宣布结构成功。

##### GPU、显存和退出状态

- 用户截图中的 `GPU-Util=100%` 表示采样期间GPU持续执行任务，`MEM=16.7%` 对应约8GiB/48GiB的显存容量占用。低显存占比不能当作低计算利用率，也不用为占满显存修改本轮参数。
- 本轮训练只为LoRA保存可训练梯度和AdamW状态，序列17帧、batch1，使用checkpointing，冻结的T5/VAE编码后移到CPU；训练稳态显存可以明显低于完整推理加载阶段。阶段记录的peak仍包括加载/编码过程，本轮实测15.139GiB allocated，不能用约8GiB的中途截图代替。冻结DiT仍需参与前向及梯度回传。
- `TRAINING UPDATES FINISHED` 只说明更新循环走完；必须再看训练子进程退出0、parent阶段complete。全流程结束还需六次推理及两例最终成果验证，不能根据两份权重存在提前宣布完整成功。
- loss.csv是每步噪声/时间步采样下的FlowMatch损失，允许波动。不能把300次optimizer update与推理30步混为一谈，也不根据loss阈值判定凹槽/黑洞已修好。
- 遇FAILED、OOM、非有限值或原生崩溃时保留run，报告stage/error/实际更新数及退出码。当前入口没有完整训练状态resume；再次执行启动命令会新建一轮Base+训练，不是继续原任务。

#### 已完成：CPU缺口硬边界合成

本轮八例已经完成，无需重跑。独立入口为 `scripts/composite_vace_results.py`，只读原始预测和交接ZIP，逐例生成三项成果，不调用模型。需要另一次CPU复核时才执行：

```bash
CUDA_VISIBLE_DEVICES='' /home/shi/miniconda3/envs/scj/bin/python -u -B /mnt/windowsE/chuanjun/building_wan/scripts/composite_vace_results.py
```

默认输出 `outputs/vace/composite_check/01–08/`；目标存在则自动使用新的时间戳根目录，绝不覆盖。每例只有合成MP4、四方MP4、四行关键帧PNG；缺输入/Mask来源无法核实则报告并跳过该例，退出码1，不重新推理补救。原推理Exact适配规则不变，GT不进入合成函数。标题和RGB视频采用兼容的H.264/yuv420p/BT.709，避免旧RGB 4:4:4播放偏色；硬边界精确性在编码前验证，不将有损MP4宣称为逐像素无损。

本轮运行位于 `vace_p01_17f:vace_composite`，终端已打印八例SUCCESS和退出0。P04/P06分别查看 `outputs/vace/composite_check/04/03_关键帧对比.png` 与 `outputs/vace/composite_check/06/03_关键帧对比.png`；内部凹槽/黑洞仍然存在，不应继续自动启动推理或边界修改。

#### 已完成操作：CPU结果重排与只读验收

只读验收现有八例及三方对比（不修改结果、不加载模型）：

```bash
CUDA_VISIBLE_DEVICES='' /home/shi/miniconda3/envs/scj/bin/python -B /mnt/windowsE/chuanjun/building_wan/scripts/reorganize_vace_results.py
```

用户授权离线重排时可追加 `--apply`，旧布局先生成并验证新成果，再安全替换；已升级布局中兼容视频只核验，旧RGB 4:4:4视频仅替换02，其他三文件不动。八例先统一预检，缺失预测/GT/Mask或错号即停止，不调用模型补救。原始MP4的SHA与文件身份始终保护。三栏为81帧/16FPS、H.264 High/yuv420p、明确BT.709有限范围、CRF12及faststart，便于普通播放器播放；禁止再使用libx264rgb/gbrp作为交付视频。FFmpeg可解码不等于浏览器显示正确，必须同时检查像素格式与色彩标签，并对三栏逐帧检查编码误差与帧序。

最终每例文件：`01_补全结果.mp4`、`02_补前_补后_GT_三栏对比.mp4`、`03_补前_补后_GT_三行总览.png`、`04_缺损_Mask_预测_GT_四栏验证.png`。旧GPU结果校验器仍按历史四文件名工作，不应对升级后的目录执行恢复推理。

#### 历史操作：保留01–03，恢复P04–P08（后续已成功，不再执行）

在 VS Code Remote-SSH 终端执行一条命令：

```bash
bash /mnt/windowsE/chuanjun/building_wan/scripts/run_vace_building_test.sh --resume --tmux
```

新建/重连 `vace_building_test`，已在 tmux 则复用当前会话；终端直接显示每例真实30步、已完成数量、实际耗时及基于已完成样本估算的批量ETA。按 `Ctrl+b` 后按 `d` 分离；新会话断线后 `tmux attach-session -t vace_building_test`。不要重复启动或另起并发。

显式resume先完整解码并核验已有连续结果及源身份，P01/P02/P03只读复用，恢复队列当前严格 `04/P04/1227 → 05/P05/1259 → 06/P06/1848 → 07/P07/2297 → 08/P08/939`。父进程/gate再核对已有四文件SHA与队列一致；全七例输入仍统一检查，随后各待跑例独立worker串行运行。全部沿用P01固定prompt、原始VACE、无LoRA、81帧512×512/16FPS/seed20261018/30步/CFG5/fullBF16，不自动改规格、重试或更换方法。

最终位置是 `outputs/vace/test/01–08/`，各成功目录只允许 `01_补全结果.mp4`、`02_补前补后对比.mp4`、`03_补前补后总览.png`、`04_四栏验证.png`。GT020仅模型返回后用于四栏，ModelMask/条件留内存；临时编码在自有系统/tmp，验收后发布，不存日志/JSON/原始预测PNG副本。

只读恢复预检：`bash /mnt/windowsE/chuanjun/building_wan/scripts/run_vace_building_test.sh --resume --dry-run`。不带resume的首次入口依然拒绝任何已有02–08；恢复也拒绝少文件、错号、非连续、软链接或额外内容，不删除或覆盖已有结果。`--prepare --resume`只验收/复用P01，不启动GPU；当前无需再次整理。

P04失败是T5完整SHA不匹配，不是OOM；当前三次独立全文件复查均正常，无法复现原根因。若再次出现，终端现在会同时显示expected/actual、绝对路径、大小与变化信息，仍立即停止；不要改批准hash、跳过校验、覆盖权重或不断重试。模型/输入/空间/其他GPU任务异常同样停止，保留此前成功结果。

#### 历史操作：P01 81 帧极简验证（已完成，不再执行）

```bash
bash /mnt/windowsE/chuanjun/building_wan/scripts/run_vace_p01_465_81f.sh --tmux
```

脚本新建/重连可见 `vace_p01_81f`，如已在 tmux 则复用现有会话并打印其重连命令，不嵌套。按 `Ctrl+b` 后按 `d` 分离；新会话断线后执行 `tmux attach-session -t vace_p01_81f`。直接不带参数运行同一脚本也可前台启动，但无 tmux 时断线可能中断。

只检查、不加载模型或保存输出：同一脚本加 `--dry-run`。已实际检查通过，不需要重复下载、上传或运行 17 帧。启动后只一次完整 81 帧/512/16FPS/seed20261018/30步/CFG5，保持同一文本与 Mask 逻辑，不分段或降规格；显存不足停止。

成功最终目录严格四文件，终端打印实际目录/耗时/峰值：`01_补全结果.mp4`、`02_补前补后对比.mp4`、`03_补前补后总览.png`、`04_四栏验证.png`。无日志/JSON/指标/输入或预测 PNG；系统临时编码文件正常退出清理。详细边界见 [VACE 执行说明](docs/VACE_P01_EXECUTION.md)。

#### 历史操作：P01 首次 17 帧（不再自动执行）

不需要下载、安装或重新传数据。已有 tmux，以下命令创建并立即进入可见会话；实验结束后保留 shell 便于查看最终结果：

```bash
tmux new-session -s vace_p01_17f 'bash /mnt/windowsE/chuanjun/building_wan/scripts/run_vace_p01_465_17f.sh; status=$?; printf "\nLauncher exit code: %s\n" "$status"; exec bash'
```

执行该命令即授权本次一次真实 17 帧实验。按 `Ctrl+b`，松开后按 `d` 仅分离查看，不中断任务。断线后 `tmux attach-session -t vace_p01_17f` 重连；已有同名会话应重连，不要另开实验或终止其他会话。

单独进入会话后也可执行 `bash /mnt/windowsE/chuanjun/building_wan/scripts/run_vace_p01_465_17f.sh`。不使用 tmux 时同一命令前台可见，但 SSH 断开可能中断。只读检查用脚本 `--dry-run`，它不会创建实验输出或加载模型。完整路径、图像布局、指标限制及失败处理见 [执行说明](docs/VACE_P01_EXECUTION.md)。

#### 2026-10-19 BuildingFiller / VACE 准备历史

Windows PowerShell 完整复制命令及三机路径边界见 [P01 交接](docs/BUILDINGFILLER_P01_VACE_HANDOFF.md)。Linux 已有同包并验证通过，当前无需再次上传。新工具默认不写数据或输出：

```bash
cd /mnt/windowsE/chuanjun/building_wan
/home/shi/miniconda3/envs/scj/bin/python -B scripts/audit_buildingfiller_package.py \
  data/buildingfiller_import/P01_465/BuildingFiller_P01_465_512_v1.zip \
  --expected-sha256 5a966a269866a1423c1efa38937fcce6ec6984f8fda78c35395abccab83d5bb8
```

可加 `--output-dir <全新审计目录>` 保存报告和解压副本，已有目录拒绝。10-18 报告位于 `outputs/buildingfiller_handoff/P01_465_20261018/`。冻结参数仍在 `configs/vace_p01_465_17f_v1.json`，独立执行器现已实现，使用方法见 [VACE 执行说明](docs/VACE_P01_EXECUTION.md)。不要直接运行会下载模型/示例数据的 vendor 示例。

当前默认只读预检和下载计划（都不会推理，缺权重的 Python 预检返回 2）：

```bash
/home/shi/miniconda3/envs/scj/bin/python -B scripts/vace_p01.py --dry-run
WAN_PYTHON=/home/shi/miniconda3/envs/scj/bin/python bash scripts/download_vace_p01.sh --dry-run
```

真实 CPU 准备目录为 `outputs/vace/P01_465_17f_baseline_cpu_prepare_20261019/`，已有目录拒绝覆盖。下载授权后才把下载脚本改为 `--download`；权重就绪后先运行 Python `--dry-run --verify-models`，GPU 另行授权后才用 `--run --output-dir <全新目录>`。下文为历史 T2V 入口，不用于 P01 条件补全。

#### 1. 环境和配置

Python：`/home/shi/miniconda3/envs/scj/bin/python`。本次确认 PyTorch 2.11.0+cu128、CUDA runtime 12.8、CUDA 可用；A100 的 mv2v 和 `/home/share/CHUANJUN/building_wan` 是历史环境。

当前默认：384×384、81 帧、batch 1、梯度累积 4、有效 batch 4、worker 0、LoRA rank 16、学习率 2e-5、计划总预算 8 epochs。原训练保存参数是 batch 2 / 累积 2 / worker 2，不能把它写成当前默认。

共享 context 仅适用于统一 prompt。完整缓存为 1992 个 `.pth`、`.shared-text-context-bf16.pt` 和 `.compact-sft-shared-context-bf16-v2`。格式名不保证 latent 全部以 BF16 落盘；训练时转为 BF16。

#### 2. 修改后的检查

```bash
cd /mnt/windowsE/chuanjun/building_wan
git diff --check
bash -n scripts/train_wan81_a6000.sh scripts/resume_wan81_a6000.sh scripts/pause_wan81_a6000.sh scripts/resume_from_pause.sh scripts/compare_lora_a6000.sh
/home/shi/miniconda3/envs/scj/bin/python -m py_compile code/Python_3D_Scanner/training/wan_train_entry.py scripts/validate_wan81_a6000.py scripts/validate_all_single_orbit.py scripts/wan_a6000_state.py scripts/single_orbit_validation.py
/home/shi/miniconda3/envs/scj/bin/python -m unittest discover -s scripts/tests -v
git status --short
```

扫描器 GUI 的三份测试依赖缺失的 `src/` 模块，不应把整个扫描器测试集报告为通过。训练工具既有测试可以单独执行：

```bash
cd code/Python_3D_Scanner
/home/shi/miniconda3/envs/scj/bin/python -m unittest discover -s tests -p test_training_tools.py -v
```

#### 3. 只读预检

以下命令不启动训练或推理，也不写 checkpoints/outputs/logs：

```bash
cd /mnt/windowsE/chuanjun/building_wan
WAN_PYTHON=/home/shi/miniconda3/envs/scj/bin/python bash scripts/resume_wan81_a6000.sh --dry-run
WAN_PYTHON=/home/shi/miniconda3/envs/scj/bin/python bash scripts/resume_from_pause.sh --state logs/pause_state_20260928_190750.json
/home/shi/miniconda3/envs/scj/bin/python scripts/validate_wan81_a6000.py
/home/shi/miniconda3/envs/scj/bin/python scripts/validate_all_single_orbit.py
```

单权重验证默认选择平铺目录最高 step-11200。全部权重入口会核对 train/val/test 的 prompt、列表哈希、路径和集合交叉。列表哈希核对不等于重新核验全部视频内容哈希或源 OBJ 身份。

`compare_lora_a6000.sh` 是历史阶段对比入口，不带 `--run` 时仍会写配置；需要真正只读检查时使用上述 Python 入口。

#### 4. 续训的单位与路径

- `.safetensors` 是 LoRA 权重，使用 `--lora_checkpoint`。不要作为完整状态传给 `--resume_from_checkpoint`。
- logger step 是处理过的物理 batch 数，含续训步号偏移，不是 optimizer updates。
- 先从 checkpoint 对应的训练参数计算累计样本，再按当前物理 batch 换算 `skip_train_batches`。训练器会打乱预分组，因此短 batch 不一定在 epoch 最后。来源 batch 不能整除样本数时，只能在完整 epoch 边界推算累计样本；切换到不能整除的新 batch 时，也只允许完整 epoch 边界的游标。其他情况拒绝猜测，不对齐的游标同样拒绝启动。
- step-2400 属于原 batch 2 段，对应 4800 样本；step-3600 属于 batch 1、skip 4800、initial 2400 段，对应 6000 样本；step-11200 属于 skip 6000、initial 3600 段，对应 13600 样本。
- 平铺 checkpoint 通过 `training_args_from_*_stepN_M.json` 匹配来源；目录迁移后的旧暂停记录只有在来源参数匹配时才能定位权重。不会仅凭同名 step 自动替换。
- 已知来源的 step/batch/skip override 必须与元数据一致；未知来源需要显式提供原 batch 和当前 skip，不猜测。
- 默认直接续训仍以平铺的 step-2400 为历史起点。它不是推荐的自动下一步，也不会自动选择最新权重。
- 默认新输出为 `checkpoints/building_wan_a6000_single_orbit_resumed_<时间>/`，位于独立顶层目录。目录已存在时启动拒绝；不要把未来结果写入平铺历史目录。
- 初始新训练也默认生成独立时间戳目录。旧/残缺缓存不会静默重建覆盖，应指定新的 `WAN_CACHE_ROOT`。该单轨入口只接受当前冻结的数据根目录和训练清单；其他数据、prompt 或划分应建立独立实验入口，避免复用同数量但不同内容的旧缓存。

当前训练完成证据不一致，见 [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)。不得因为此问题自动启动补训。

#### 5. 暂停与续跑（需用户明确要求）

```bash
# 记录运行中本项目训练的最新可读权重，训练继续运行。
bash scripts/pause_wan81_a6000.sh
# 记录后停止所选进程；多进程时必须显式指定 --pid。
bash scripts/pause_wan81_a6000.sh --pid <训练PID> --stop
# 只读打印和检查续跑计划。
WAN_PYTHON=/home/shi/miniconda3/envs/scj/bin/python bash scripts/resume_from_pause.sh --state <暂停记录.json>
# 明确批准后后台启动；启动器存活不等于 loss 已更新。
WAN_PYTHON=/home/shi/miniconda3/envs/scj/bin/python bash scripts/resume_from_pause.sh --state <暂停记录.json> --start
```

暂停入口兼容 `--output_path /目录` 和 `--output_path=/目录`，路径中可以有空格，相对路径按训练进程的 cwd 解析。当前 scj Python 缺少 pidfd 接口，权重头部预检使用 safetensors 的 NumPy 后端，不导入 PyTorch；参数和游标检查在打开权重文件前完成。停止时仅把进程控制交给 `/usr/bin/python3`；权重读取仍使用 scj。控制器核对 PID 的启动时间及项目命令，并通过 pidfd 发信号，拒绝向身份变化的进程发信号。

v2 暂停记录保存明确的累计样本数；历史 v1 记录重新读取 checkpoint 来源参数计算，避免旧单位错误。用户通过 `WAN_TRAIN_BATCH_SIZE` 和 `WAN_GRADIENT_ACCUMULATION_STEPS` 改 batch 时需自行维持所需有效 batch。若短 batch 使当前 checkpoint 的预算不可推算，暂停入口会拒绝生成记录，也不会停止进程；需使用完整 epoch 边界的 checkpoint，或先为该独立实验实现实际样本计数。暂停快照可能在终止前继续前进；仅从最后保存权重续跑，丢失其后的进度。

这是权重续跑，不恢复 optimizer/scheduler/RNG、累积梯度或 shuffle 序列。checkpoint 若不是梯度累积边界，预算换算也不构成完整状态恢复。

直接从确认的权重续训可先这样预检：

```bash
WAN_RESUME_FROM_CHECKPOINT=/mnt/windowsE/chuanjun/building_wan/checkpoints/building_wan_a6000_single_orbit_all/step-11200.safetensors WAN_PYTHON=/home/shi/miniconda3/envs/scj/bin/python bash scripts/resume_wan81_a6000.sh --dry-run
```

移除 `--dry-run` 会实际训练，只能在明确授权、核对计划并确认没有旧进程后执行。推荐在 tmux 中运行；`WAN_DEBUG_GDB=1` 仅启用原生堆栈诊断，不表示故障已修复。

#### 6. 验证推理与效果评估（需用户明确要求）

29 个固定条件视频已经生成，不应默认重跑。若要生成新的最新权重对比：

```bash
/home/shi/miniconda3/envs/scj/bin/python scripts/validate_wan81_a6000.py --run
```

两个 Python 验证入口先解析参数，再依据 `args.run` 获取共享锁；`--run`、`--ru`、`--r` 均不能绕过锁。全部权重入口支持 `--only base,step-11200`。已有完成视频在推理参数和 checkpoint 来源一致时跳过；`--overwrite` 只重新生成未完成/未验证的已有视频。参数不同应使用新输出目录；旧配置不会被替换。

像素诊断 v2（无需模型推理，但会生成新分析文件）：

```bash
/home/shi/miniconda3/envs/scj/bin/python scripts/analyze_validation_videos.py --output-dir /mnt/windowsE/chuanjun/building_wan/outputs/validation/validation_all_single_orbit/_review_v2
```

已有目录时拒绝覆盖。v2 全图前景占比使用灰度 <0.9 的像素比例；二阶像素变化没有运动补偿，不能称为纯闪烁分数。人工评估建筑体块、完整环绕、局部形变、闪烁、构图和时序稳定性；低运动量也可能代表近乎静止的视频。

#### 7. 排错与终态

检查实际进程、loss 增长、可读 checkpoint、traceback 和退出记录。NVML 警告和短时低 GPU 利用率不能单独判定失败。worker 0 未消除所有历史原生崩溃，根因仍待定位。

今后 `wan_train_entry.py` 仅在正常返回后写 `training_completion.json`；其中 `final_logger_step` 使用 batch 口径。旧运行没有该文件，不应补造成功记录。A6000 脚本使用同一 launcher 锁，进程退出时释放；绕过这些入口的进程仍需检查。

#### 8. 原生崩溃诊断与真正的进程退出

本轮 GPU 诊断已结束，用户要求不再继续跑并删除串行测试，已执行。
普通反向 GDB 基线在 166 个新增 batch 后原生崩溃；Valgrind 覆盖 196 batch
无访问错误后主动停止；GPU memcheck 覆盖 4 batch 报告 0 错误后主动停止，
未完成原计划 8 batch。串行完整测试跑完 2336 batch，Python 正常退出，
6 份新权重张量检查通过；本次未复现不能证明根因已修复。
详见 [诊断记录](docs/NATIVE_STABILITY_20261014.md)。不要自动重跑诊断。

诊断入口默认仅预检，加 --run 才真实训练；实跑使用独立日志和权重目录，
保存外层退出码、连续有限的 loss、预算步数与最终权重存在情况：

```bash
/home/shi/miniconda3/envs/scj/bin/python scripts/diagnose_wan_a6000.py --checkpoint checkpoints/building_wan_a6000_single_orbit_all/step-11200.safetensors
```

`training_completion.json` 证明引擎返回，解释器随后仍可能原生崩溃，必须同时
检查外层退出码。`completed_budget` 不等于根因确认，权重需另行读取检查张量。
缺少完成标记时，报告仍保留已经记录的 loss 数量。

旧 GDB 命令在 Python 正常退出后执行 py-bt / continue，会因可选 helper
缺失返回 1。已修复：WAN_DEBUG_GDB=1 使用 scripts/wan_gdb.py，在新输出
写 debugger_exit.json，分别记录 inferior 退出、信号与 launcher 状态；仅在
实际故障信号时采集栈，缺少 py-bt 不再影响实际退出判断。正常 exit 0、
Python exit 9、SIGABRT、SIGSEGV 的真实 CPU GDB 回归返回 0、9、134、139。
此次 GPU 长测启动早于修复，仍使用旧命令，已单独核对 exited normally，
不能说已用新版 helper 完成 GPU 验证。

仅检查既有 LoRA 的全部张量（CPU，不训练、不推理，不写报告文件）：

```bash
/home/shi/miniconda3/envs/scj/bin/python scripts/check_wan_lora.py checkpoints/building_wan_a6000_single_orbit_all/step-11200.safetensors
```

工具完整读取 BF16 payload 并检查有限值，可加 --reference 核对键名和形状。
头部可读不代表张量有限；检查通过也不证明视觉质量。

WAN_TRAINING_SEED 可设置 0 至 2**32-1 的固定种子。WAN_SERIAL_BACKWARD=1
通过 PyTorch context 将反向调度到调用线程，只支持单进程；默认值 0。
该选项在本次完整预算中未复现崩溃，仍只是候选绕过，不改为正式默认。
诊断阶段的短窗口会写 diagnostic_limit.json，不写完整预算的成功标记。
完整串行测试及两轮串行短测的权重、日志和临时监督脚本已按要求删除，
不能使用这些路径续训；历史正式权重及非串行诊断证据保留。
<!-- RESTORED_HISTORY_END -->

<a id="update-20261023"></a>

## 二、2026-10-23：阶段更新与当前资料补全

下面继续保留357b09f阶段补充的结果、参数和操作说明。该阶段曾把历史移到归档、部分原文页改成导航；这一文档组织做法已由第三节的用户更正取代，实验事实和既有检查结果保留。

<!-- RETAINED_UPDATE_BEGIN -->
### 服务器操作与验证手册

文档核对日期：2026-10-23（UTC+09:00）。本页区分只读查看、CPU准备和GPU执行。现有VACE正式实验均已完成，日常先查看结果和状态。

#### 1. 当前环境

| 项目 | 实际值 |
| --- | --- |
| 项目根 | /mnt/windowsE/chuanjun/building_wan |
| Python | /home/shi/miniconda3/envs/scj/bin/python |
| Python / torch / CUDA runtime | 3.10.13 / 2.11.0+cu128 / 12.8 |
| DiffSynth / PEFT / safetensors | 2.1.5 / 0.17.1 / 0.7.0 |
| 计算方式 | 单卡A6000，单进程，当前训练不使用DDP |
| 代码版本 | 整理结构提交ccfe24d；后续以git log实际结果为准 |

旧A100的mv2v、/home/share路径和Slurm命令不适用于本机。扫描器requirements.txt是旧GUI依赖，不是当前训练环境的安装清单。本次不安装或更新依赖。

#### 2. 日常只读查看

从项目根运行：

```bash
cd /mnt/windowsE/chuanjun/building_wan
/home/shi/miniconda3/envs/scj/bin/python -B experiments/vace/complex12_rev1/scripts/status.py
git status --short --branch
git log -3 --oneline
tmux list-sessions
```

Complex12应显示complete、60/60、C/U/S各300。状态脚本不加载模型、不导入torch、不改变progress.json。tmux会话不存在只说明终端历史未保留，不代表成果丢失；不要为查看状态重新启动训练入口。

查看旧LoRA17的实际记录：

```bash
/home/shi/miniconda3/envs/scj/bin/python -B - <<'PY'
import csv, json
from pathlib import Path
run = Path('outputs/vace/lora_roof_smoke/run_20261020_165237_150325_2048720')
record = json.loads((run / 'training/experiment.json').read_text())
with (run / 'training/loss.csv').open(newline='') as f:
    history = list(csv.DictReader(f))
print(record['runtime'])
print('Saved update records:', len(history))
PY
```

#### 3. 启动脚本的实际行为

| 入口 | 无参数行为 | 只读/CPU方式 | 对应范围 |
| --- | --- | --- | --- |
| run_vace_p01_465_17f.sh | 会进入GPU执行 | --help；--dry-run做模型/CPU预检 | 历史P01 17帧 |
| run_vace_p01_465_81f.sh | 会进入GPU执行 | --help；--dry-run | 历史P01整轨 |
| run_vace_building_test.sh | 会进入GPU批量执行 | --help；--dry-run；--prepare会复制P01 | 八例Base |
| run_vace_roof_smoke.sh | CPU准备 | --help或--dry-run；正式用--run/--tmux | 两例17帧LoRA |
| run_vace_lora300_eval81.sh | 无参数为CPU预检；--run-trained/--run-transfer启动GPU | --dry-run；--render-only仅CPU制图，见[说明](experiments/vace/pilot8_lora81/README.md) | 八例LoRA-300 |
| 原Complex12 run/resume.sh | 若未完成会恢复任务；已完成则直接返回 | 日常改用status.py | 当前固定run |
| train_wan81_a6000.sh | 正式历史T2V训练 | 阅读脚本和配置；没有通用--dry-run承诺 | A6000历史T2V |
| resume_wan81_a6000.sh | 正式历史T2V续训 | --dry-run | 平铺权重的weights-only续训 |

预检可能读取大权重哈希，CPU准备可能产生新文件，均不等于廉价状态查询。不要因文档列出命令就执行；GPU任务须有当前明确授权。历史默认输出策略与新目录规范不同，新实验要先明确新的独立run路径。

#### 4. 恢复方式不能混用

| 实验 | 可恢复内容 | 不能推断的内容 |
| --- | --- | --- |
| Complex12 | resume_latest.pt中的LoRA、AdamW、已完成行及Python/NumPy/torch/CUDA RNG | 不可跨配置、方法或初始化恢复 |
| 两例LoRA17 | 100/300的LoRA增量可供推理 | 不能恢复完整optimizer/RNG；入口无完整训练resume |
| 历史T2V | --lora_checkpoint加载LoRA并按来源参数换算样本预算 | 不恢复旧optimizer、RNG、shuffle或累积中间态 |

Complex12成功任务按配置ID和内容完整性复用。当前run已经complete；不要删除完成标记以强制重跑。历史T2V的step是logger物理batch计数，换batch时必须从来源段参数换算样本数；不把最高文件名当作训练完成证据。

#### 5. 真实进度与故障判断

在可见tmux终端运行经过授权的任务，显示样本身份、方法、真实优化器更新/去噪步数、耗时、保存状态和已完成轨迹。断开SSH后重连原会话，先确认实际进程，避免双重写同一run。

出现OOM、非有限值、断梯度或进程异常时保留输出和真实退出码，按该实验授权范围处理。已保存文件需独立验证；不能只因有权重或一张图片就标记成功。B/CS04的−6退出及历史T2V堆损坏根因仍未确认。

显存容量占比、PyTorch allocated/reserved和GPU计算利用率分别记录。nvidia-smi不在PATH、黄色告警、瞬时低利用率都不能单独判定失败。只有状态和资源查询可靠时才能决定是否启动；本次文档更新不进行GPU试算。

#### 6. 修改后规定的CPU检查

```bash
git diff --check
bash -n scripts/train_wan81_a6000.sh scripts/resume_wan81_a6000.sh
CUDA_VISIBLE_DEVICES='' /home/shi/miniconda3/envs/scj/bin/python -m py_compile code/Python_3D_Scanner/training/wan_train_entry.py scripts/validate_wan81_a6000.py
CUDA_VISIBLE_DEVICES='' /home/shi/miniconda3/envs/scj/bin/python -m unittest discover -s scripts/tests -v
git status --short
```

修改Complex12计算代码时，另运行其8项专用CPU检查：

```bash
CUDA_VISIBLE_DEVICES='' /home/shi/miniconda3/envs/scj/bin/python -m unittest discover -s experiments/vace/complex12_rev1/scripts -p 'test_*.py' -v
```

GUI测试依赖缺失src/，不能把上述CPU回归写成全部GUI能力通过。文档验证还应检查相对链接、样本映射、命令默认行为及冻结报告的口径一致性。

#### 7. 交接与Git

更新根状态、台账、操作和交接文档；记录实际run/代码位置、更新数、退出证据、结论及局限。仅提交相关文件，保留其他未提交工作。检查差异与GitHub账号后按main流程推送，并在服务器执行git pull --ff-only origin main，不强推。

模型、视频、日志与缓存留在原目录。详细旧操作和失败过程见 [文档历史索引](docs/DOCUMENTATION_INDEX.md)。
<!-- RETAINED_UPDATE_END -->

<a id="history-repair"></a>

## 三、2026-10-23：恢复原文件内的持续历史

用户明确要求长期MD保留“从以前到现在”的更新。先前仅保留当前摘要、让读者另找归档的处理不符合要求；本次将上述旧正文直接恢复到原文件，同时保留后续更新，没有回退实验结果或把完成状态改回执行中。

以后在本文件按日期追加阶段进展、失败与修复、参数变更、结论和核查范围。需要纠正旧结论时，说明原判断、后来的证据及新结论；原记录继续保留。可以在顶部维护简短状态和阅读索引，不能用摘要或归档链接替换历史正文。

本次修复依据原文件备份及Git记录恢复，没有补造新的运行经历；原数据、权重、视频、指标、冻结报告和现有非Markdown工作保持原样。此次检查与恢复范围见[文档索引](docs/DOCUMENTATION_INDEX.md)中的后续补记。

## 四、2026-10-24：Complex12可复核报告与每组实验收尾要求

日期采用UTC+09:00。新增 [Complex12完整报告](upload/20261022_Complex12_Rev1_十二栋单轨实验报告.md)，逐项记录17帧训练/81帧推理、训练seed20261022/推理seed20261018、C/U/S相同新初始化与随机序列、损失权重、GT隔离、恢复状态及实际300更新。报告中同时解释了worker耗时口径：C的1570.050秒仅是恢复后的成功进程记录，不能当作全部300更新的总成本。

本轮已完成；日常查看以下只读入口即可，不调用resume.sh重跑：

```bash
/home/shi/miniconda3/envs/scj/bin/python /mnt/windowsE/chuanjun/building_wan/experiments/vace/complex12_rev1/scripts/status.py
```

以后每组实验的收尾顺序为：核对实际进程/退出和产物、核对真实更新与评测、在upload写该组独立报告、同步长期MD和实验README、完成项目规定检查与相关Git记录。报告必须写清故障和恢复、未完成内容、训练与验证差异，不用loss下降或权重存在替代完整结果。

本次只写文档，未训练、推理、修改冻结参数或重编码媒体。新建议需要另一次实验的明确范围，不能把报告里的后续方向当成现在自动执行的命令。格式与必填信息见 [upload/README.md](upload/README.md)。


## 2026-10-26：OriginalHard 已有权重评测启动快照

核查时间：2026-10-26T01:27:12+09:00。用户已授权对上传的 OriginalHard 使用已有 B/L/C/U/S 跑完整 81 帧，并要求“旧的相同的就不用跑了”。本轮新增训练更新为 **0**，C/U/S 沿用已完成的 Rev1 step300，L 为旧 P04/P06 参考。

新实验入口：[OriginalHard](experiments/vace/complex12_originalhard_eval/README.md)。真实 run：`/mnt/windowsE/chuanjun/building_wan/outputs/vace/complex12_originalhard_eval/run_20261026_012033`。源码位于 experiments/vace/complex12_originalhard_eval/scripts，旧大结果保持原位。

逐帧核对后，CS01–CS08 的 GT/Damaged/Exact/结构图及相机记录与旧版一致，五方法共40条预测已解码后链接复用；CS09–CS12 的缺损不同，新增20条推理。最终展示60条，其中40条复用、20条新推理。新缺损中的CS09/10/11来自旧训练身份，CS12来自旧验证身份；应单独汇总四栋及3/1子组，不能称为十二栋新建筑泛化。

此刻 runner PID 1678408 实际存活，tmux 为 complex12_rev1:originalhard81；状态 running，当前 infer_L_CS09，有效轨迹 41/60（复用40，新推理1）。这是一条运行中快照，不代表全部推理、媒体或视觉结论已经完成。

本轮没有训练入口或新检查点。日常只读查看：

```bash
bash /mnt/windowsE/chuanjun/building_wan/outputs/vace/complex12_originalhard_eval/run_20261026_012033/resume.sh
```

已启动的同一入口加 --run/--resume 才会执行队列，不用它重复启动当前进程。终端显示真实去噪30步、当前方法/建筑、耗时、保存状态。成功结果复用；进程异常保留输出与退出码，不自动重试。CPU评测保持旧指标定义，另增hard_summary区分新缺损四栋及身份。


### 2026-10-26 后续交接：用户要求启动后即可结束对话

用户进一步明确“跑起来就行，就可以结束这轮对话了，不需要实时监控，我在vscode的tmux能看到过程即可”。01:31:02（UTC+09:00）核查时，runner与C/CS09 worker实际存活，C去噪17/30，40条复用、2条新增已完成；VS Code所连tmux客户端位于complex12_rev1的originalhard81窗口。

按此要求结束本轮主动跟踪，保留既有队列继续运行。队列包含剩余推理、统一指标、12栋视频/总览和复核图；计算结束状态为awaiting_visual_review。详细视觉分析、最终中文结论、upload实验报告和完成文档需下次接手核对真实产物后补齐，不能将本轮“已启动”写成“实验已全部完成”。长期文档的历史正文和旧实验产物继续保留。


## 2026-10-26：OriginalHard 完整轨迹复核与结果收尾

复核记录：2026-10-26T02:41:09+09:00（UTC+09:00）。本次接手已核对20个新worker退出0、40条复用，共60条81帧/4860 PNG，12视频/12总览和统一指标均完成；用户要求审查困难结构效果后，逐段查看CS09–CS12全部81帧联系图及六个原尺寸问题帧。新增训练更新0，本次没有启动新GPU任务。此前启动/待复核快照按原文保留。

run：`/mnt/windowsE/chuanjun/building_wan/outputs/vace/complex12_originalhard_eval/run_20261026_012033`。直接观看入口：[00_看结果](outputs/vace/complex12_originalhard_eval/run_20261026_012033/00_看结果/README_从这里开始.md)；完整可携带报告：[OriginalHard已有权重评测](upload/20261026_Complex12_OriginalHard_已有权重评测报告.md)。最终结论在run的00_看结果/最终实验结论.md，核查后progress更新为complete。完成指执行和报告完整，不表示补全质量全部合格。

CS09、CS11明显改善；CS10连接面/折线仍错，C的MAE最低；CS12错误折面及首尾形状差异明显，旧L反而最好。新缺损四栋B/L/C/U/S的MAE为0.069922/0.036078/0.036619/0.036443/0.035899。S比B降低48.66%，但比C/U只低1.97%/1.49%；S相对C保留区变化增加4.17%、全图闭环误差增加13.78%，不能称S稳定胜出。CS12的S首尾缺损区误差0.126121，L为0.029390。

身份不能混淆：CS01–CS08输入不变且五方法复用；CS09/10/11是旧训练建筑新缺损，仅CS12为旧验证建筑新缺损。不是12栋全新建筑泛化，不能用完整8/4均值掩盖困难4栋。视觉结论来自连续静态帧与原尺寸图，视频有程序全帧解码证据，本次未做实时播放或量化高频闪烁。

当前没有训练或推理待自动追加。只读状态命令仍为：

```bash
bash /mnt/windowsE/chuanjun/building_wan/outputs/vace/complex12_originalhard_eval/run_20261026_012033/resume.sh
```

不要为查看效果添加--run/--resume，不通过删完成标记触发GPU。40条旧预测是链接，导出原始PNG时须沿链接复制；大成果留在原run。本次报告与原始CSV/连续帧复核一致，指标定义未改；根文档按日期追加。

收尾实测：2026-10-26T02:42:50+09:00，progress=complete；60轨迹/4860 PNG与12总览可读，12视频完整解码证据齐全。298项CPU回归（57.214秒）及9项专用检查通过；详细检查见文档索引和run/logs/final_review.done.json。


## 2026-10-26：C_mix 固定配比实验启动与后续交接

核查时间：2026-10-26T22:26:41+08:00。用户授权一组普通C_mix的300更新、16条81帧推理及配对指标/视频，随后要求确认训练启动即可结束对话，视觉复核和最终报告等待另行通知。当前running，已记录33/300次真实优化器更新、推理0/16；这不是完成或效果确认。

实际run：`/mnt/windowsE/chuanjun/building_wan/outputs/vace/complex12_cmix_pilot/run_20261026_222008`。研究入口：[C_mix](experiments/vace/complex12_cmix_pilot/README.md)，观看入口：[00_看结果](outputs/vace/complex12_cmix_pilot/run_20261026_222008/00_看结果/README_从这里开始.md)。tmux为`complex12_rev1:cmix_pilot`（%2），runner PID 2179781，GDB worker PID 2179822。

从旧Rev1 shared_initialization及原始VACE开始，新AdamW，普通C损失与参数不变。旧300行顺序/窗口/随机字段保留；局部Random(20261026)按CS09/10/11抽20/14/16行Hard，合计Rev1 250/Hard 50。验证CS02/04/08/12任何版本不入训练。条件缓存按版本、建筑、窗口隔离；合法零scheduler权重计数、不重抽不补步。两版改变了缺损位置与连接线索，不是同一缺口单纯扩大。

队列依次完成300更新、Hard CS09–12和Rev1 CS01–12共16条/1296原始PNG、原指标与分组/逐栋新旧变化、Rev1退步项、32视频和六张固定问题帧。旧C结果复用；Hard CS01–08展示复用新Rev1；不重跑B/L/U/S。计算结束标记awaiting_visual_review，用户通知后再进行完整视觉复核、run中文报告和同内容upload报告。当前不提前写效果结论。

本轮9项专用CPU检查通过（0.658秒），298项项目CPU回归通过（50.410秒），规定shell语法/Python编译与差异检查通过；没有额外GPU试跑。旧run、数据、第三方引擎及19项无关工作区修改保持原样。冻结配置ID：`386098b4cc71d7a446b5d2dcba635e8a8a3142dcaa3e6bd8629299406fb0c545`。

只读状态：

```bash
bash /mnt/windowsE/chuanjun/building_wan/outputs/vace/complex12_cmix_pilot/run_20261026_222008/resume.sh
```

同一run中断后的已授权恢复入口（先确认没有活跃runner）：

```bash
bash /mnt/windowsE/chuanjun/building_wan/outputs/vace/complex12_cmix_pilot/run_20261026_222008/resume.sh --resume
```

恢复完整adapter/optimizer/RNG状态并执行剩余采样行，不用safetensors代替恢复状态；成功轨迹复用。当前正在运行，不再次启动恢复命令。


## 2026-10-27：C_mix 完整评测、视觉复核与收尾

本轮正式训练已经结束，后续查看结果不需要重新训练或运行队列。run为`/mnt/windowsE/chuanjun/building_wan/outputs/vace/complex12_cmix_pilot/run_20261026_222008`；[实验说明](experiments/vace/complex12_cmix_pilot/README.md)、[完整报告](upload/20261027_Complex12_C_mix_混合训练与配对评测报告.md)保留准确参数与结论。

训练实际为共同初始化+新AdamW的300更新，250/50混合；仅vace LoRA，普通C、rank/alpha16、lr5e-5、17帧、BF16骨干/FP32 LoRA。原表随机与顺序逐行保留，缓存按版本/建筑/窗口隔离。Rev1/Hard各4/3步合法零时间权重仍按预算执行；无重抽、补步或本轮中断恢复。最终step300可读且300个tensor有限，完整恢复状态另存resume_latest.pt（update300）及100/200里程碑。

现在只读查看状态：

```bash
CUDA_VISIBLE_DEVICES='' bash /mnt/windowsE/chuanjun/building_wan/outputs/vace/complex12_cmix_pilot/run_20261026_222008/resume.sh
```

不带参数仅调用status.py；`--run/--resume`是执行入口，不是状态查询。当前16条预测、32视频和报告已经完成，不要为了刷新入口重新执行GPU队列或删除完成标记。原evaluate.done.json保留当时“等待用户通知复核”的快照，最终收尾看progress.json及logs/final_review.done.json。

本次效果没有达标：Hard CS10/CS12关键结构未解决，Rev1 CS10新条纹明显；详细逐栋证据见报告。该现象不构成自动训练下一组的授权。CPU检查使用CUDA_VISIBLE_DEVICES为空，不为文档收尾触发CUDA训练。


## 2026-10-27：Orbit17线性/循环配对实验已启动

核查时间：2026-10-27T20:16:32+09:00（UTC+09:00）。用户已授权线性组和循环组都执行，并要求确认实际开始训练后结束本轮对话。当前status=running，C_lin17已完成22/600次真实AdamW更新，C_ring17为0/600（待串行执行），推理0/40；日志已有有限loss及参数实际变化。这是启动快照，不是两组完成或效果验证。

固定run：`/mnt/windowsE/chuanjun/building_wan/outputs/vace/complex12_orbit17_pair/run_20261027_200842`。专用源码与说明：[Orbit17实验](experiments/vace/complex12_orbit17_pair/README.md)；[结果导航](outputs/vace/complex12_orbit17_pair/run_20261027_200842/00_看结果/README_从这里开始.md)。tmux=`complex12_rev1:orbit17_pair`（%3），runner PID 2469437，GDB worker PID 2469491；源码提交`1a82bd5`。

两组均只用Rev1训练8栋CS01/03/05/06/07/09/10/11，验证CS02/04/08/12不变；Hard只评测。普通C、原VACE骨干、共享初始LoRA与各自新AdamW，rank/alpha16、lr5e-5、17帧、BF16骨干/FP32 LoRA等原设置保持。线性窗起点0–63，循环窗起点0–79按模80取17帧，080不入训练；跨界先同步取完整17帧再编码。原线性前300行保留；后300行用局部Random(20261027)，循环起点另用Random(20261028)，其他随机字段逐行配对。合法零scheduler权重照常计数，不重抽补步。

已启动的同一队列包含：线性0→300、四条81帧复现探针与CPU提前核对、完整状态恢复线性301→600、循环0→600、循环300四条探针、两组600各16条正式预测，最后统一CPU评测和视频。提前核对保留复现证据，不因主观效果选择性取消循环组。总预算两组各600、40条81帧、3240原PNG；计划48视频、7固定问题图和4张七行探针总览。旧C两版原预测直接复用，无新增B/L/U/S/C_mix推理。

只读状态（不加载模型）：

```bash
bash /mnt/windowsE/chuanjun/building_wan/outputs/vace/complex12_orbit17_pair/run_20261027_200842/resume.sh
```

查看真实终端：

```bash
tmux attach -t complex12_rev1
```

选择`orbit17_pair`窗口；单GPU自动串行，两组并未同时占卡训练。已授权的同run恢复入口为`bash /mnt/windowsE/chuanjun/building_wan/outputs/vace/complex12_orbit17_pair/run_20261027_200842/resume.sh --resume`，仅在确认旧runner不活跃后使用，不能用作状态查询。恢复加载完整AdamW/RNG与对应采样位置，stop-update是绝对目标；不要把300权重当完整状态或从300再追加600。异常保留最后有效恢复点和日志，不自动改规格或扩预算。

本次实测14项Orbit17专项CPU检查通过（1.466秒），298项项目CPU回归通过（50.203秒），规定shell语法/Python编译及git diff --check通过；没有额外GPU试跑。准备时覆盖图首次因服务器缺少matplotlib失败，随后改用现有Pillow生成并通过；没有安装依赖，未重复GPU任务，prepare.log保留实际错误与后续成功。旧run、原始数据、模型、第三方源码及19项既有无关工作区修改保持原样且不纳入本轮提交。

每10更新保存完整恢复状态，300/600保存LoRA，300另保留恢复点；两组状态独立。缓存按版本/建筑/完整有序frame_ids/编码设置区分，放在新run。自动队列完成后标记awaiting_visual_review；之后按用户通知进行完整轨迹视觉复核、run最终中文报告及upload同内容便携报告。此时不能预先宣称循环优于线性，也不能把视频解码当人工播放或三维验证。新预算、配比或模块不自动展开。
