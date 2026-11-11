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


## 2026-10-29：Orbit17配对实验复核与最终报告完成

计算于2026-10-27T23:41:29+09:00结束，用户随后通知复核/报告收尾；10月28日至29日完成本次复核。实际run为`outputs/vace/complex12_orbit17_pair/run_20261027_200842`。两组各600真实AdamW更新、40条81帧/3240原PNG、48视频、7固定问题图和4张最终七行探针总览齐全；另保留4张早期探针图与1张覆盖图。45个队列worker均单次启动、EXIT 0；线性300→600为计划内完整optimizer/RNG恢复，循环连续0→600，没有失败后回放或新增GPU任务。

[详细便携报告](upload/20261029_Complex12_Orbit17_线性与循环窗口配对实验报告.md)；[直接看结果](outputs/vace/complex12_orbit17_pair/run_20261027_200842/00_看结果/README_从这里开始.md)。本轮训练/推理已完成，日常只读查询仍为：

```bash
bash /mnt/windowsE/chuanjun/building_wan/outputs/vace/complex12_orbit17_pair/run_20261027_200842/resume.sh
```

不要因旧启动段落写着running或awaiting_visual_review再次执行恢复入口。各组checkpoints下step300/step600均存在；resume_latest.pt为600，resume_step300.pt保留300完整状态。线性301–600从完整状态恢复，绝对目标600；不是从safetensors恢复优化器，也不是追加另一轮600。

两组仅用Rev1训练8栋、验证4栋不变，Hard只评测；普通C、共同初始LoRA与新AdamW、600预算相同。线性起点0–63，循环模80起点0–79，完整17帧同步读取后编码，080不训练。原线性前300行保留，扩表seed20261027、循环起点seed20261028，其他逐步随机字段配对。每组600中12步合法零时间权重按表计数，不补步。循环600有127个跨界窗，跨训练8栋的000/079曝光139/135，线性为7/15。

执行耗时：线性0–300 1562.688秒、301–600 1119.953秒，循环0–600 1878.918秒；40推理合计7541.678秒，队列墙钟3小时27分26秒。缓存命中不同，不将速度差归因于方法。训练/推理PyTorch峰值allocated约15.139/17.143GiB，reserved约15.393/25.832GiB。本次无GPU任务；准备阶段matplotlib缺失已在启动前改用Pillow，原日志不改，不能宣称历史原生退出问题已修复。

本次CPU完整解码3240原PNG、48条视频及16张展示PNG；全部40条000–080通过连续静态联系图复核（缩略内容），7张固定问题图按完整512内容检查。没有实时视频播放、定量闪烁或三维验证，不能把文件可读写成模型效果全达标。原evaluate.done的待复核状态与既有启动正文保留为历史，当前交付状态以progress和logs/final_review.done.json为准。

收尾CPU检查日志为本run的logs/final_cpu_project_tests.log（298项通过）、final_cpu_orbit17_tests.log（14项通过），以及文档写入后规定检查记录。图像复核与方法结论见详细报告；后续GPU计划仅是建议。


## 2026-11-09：0号建筑点云原版VACE单次基线准备

记录时间：2026-11-09T02:20:32.991939+09:00。用户明确授权仅一次原版VACE推理，要求确认实际GPU推理后立即结束会话，不等待完成。本节为启动前准备快照，不是完成或视觉效果结论。

- 研究入口：[0号点云基线](experiments/pointcloud_vace/0_v2_base/README.md)；独立run：`/mnt/windowsE/chuanjun/building_wan/outputs/pointcloud_vace/0_v2_base/run_20261109_021750`。专用源码在experiments/pointcloud_vace/0_v2_base/scripts/run.py，复用既有官方VACE加载与参数适配，不改第三方核心。
- 输入为data/pointcloud_vace/0_v2的src_video.mp4、src_mask.mp4、prompt.txt；两视频完整解码均81帧832×480/16FPS，时间戳逐帧一致，mask严格0/255且白区全部对应RGB127。原始英文prompt要求离散稀疏彩色点，逐字使用；negative为空、无扩写，无输入修复或mask重制。
- 一次整序列masked video editing：Wan2.1-VACE-1.3B原始官方权重，无LoRA/训练/旧T2V缓存，30步、CFG5、seed20261018、shift5、VACE scale1、BF16、tiled VAE、full memory。Python为/home/shi/miniconda3/envs/scj/bin/python。启动前NVML没有其他GPU进程，RTX A6000可用；nvidia-smi命令及pynvml模块缺失，已使用项目既有ctypes NVML检查，不安装依赖。
- 独立后台父进程监督唯一worker，不重试。日志run.log、实际参数config.json、prompt副本、输入绝对路径及SHA256、run.pid/worker.pid、status.json和最终exit.json自动保存。sampling_started.json只在首个扩散步骤完成且CUDA同步后生成。
- 正常结束先保存raw_vace.mp4及81张raw_frames原始PNG，后尝试input_mask_raw.mp4及五张比较PNG；原始结果不回贴、不被可视化覆盖。无未填灰原始观测视频，不将src_video冒充完整原始视频。
- 本次只启动；是否生成分离点、是否实心墙、漂移/闪烁和已知点变化均待后续查看。未生成完成报告，不宣称模型效果成功，不启动第二次推理。旧实验run及既有无关工作区修改原样保留。

只读进度：`tail -n 40 /mnt/windowsE/chuanjun/building_wan/outputs/pointcloud_vace/0_v2_base/run_20261109_021750/run.log`；状态：`cat /mnt/windowsE/chuanjun/building_wan/outputs/pointcloud_vace/0_v2_base/run_20261109_021750/status.json`。最终视频与exit.json仅在对应阶段实际完成后出现，不通过重新执行入口查看状态。


### 2026-11-09 后台派发快照

2026-11-09T02:22:09.011787+09:00：输入与pipeline CPU预处理检查通过，官方VACE/T5/VAE及tokenizer完整SHA核验通过；298项项目CPU回归通过（62.672秒），父进程成功/原生异常/普通失败三项模拟检查、规定shell语法/Python编译及diff检查通过。实际启动独立session 3508602，后台父PID 3508602、worker PID 3508606，stdin为/dev/null、stdout/stderr重定向run.log；run.pid与worker.pid已落盘。此快照仅确认后台派发，采样验收看本run的sampling_started.json及run.log中的真实diffusion_sampling步数；退出与效果仍待实际完成。用户要求一旦确认实际GPU推理就停止主动跟踪，因此不在本轮等待完整结果。


## 2026-11-09：0号建筑点云基线完成后的查看与报告流程

本run已完成，不再执行推理入口：`outputs/pointcloud_vace/0_v2_base/run_20261109_021750`。父PID3508602、worker3508606已退出，真实返回码0；30步完成，339.2228秒，81PNG和两视频已经保存。东京02:27:48对应服务器01:27:48，截图01:28:49没有GPU进程是完成后的正常状态。

```bash
cat /mnt/windowsE/chuanjun/building_wan/outputs/pointcloud_vace/0_v2_base/run_20261109_021750/exit.json
ls -lh /mnt/windowsE/chuanjun/building_wan/outputs/pointcloud_vace/0_v2_base/run_20261109_021750/raw_vace.mp4 /mnt/windowsE/chuanjun/building_wan/outputs/pointcloud_vace/0_v2_base/run_20261109_021750/input_mask_raw.mp4
cat /mnt/windowsE/chuanjun/building_wan/outputs/pointcloud_vace/0_v2_base/run_20261109_021750/report_review_20261109_024727/final_review.json
```

原`status.json=complete_awaiting_visual_review`和环境探测中的`GPU_inference_started=false`保留为运行时/加载前快照；实际执行由sampling_started、30步日志、媒体和exit.json证明，本次静态复核状态看新增final_review.json及[报告](upload/20261109_0号建筑点云_Wan21_VACE原版基线_实验报告.md)。不要把旧字段当作重新启动理由。

用户再次明确以后每组实验完成/失败收尾，主动在根upload生成详细MD并登记索引，同步长期MD；报告写实际模型/数据/参数/执行/失败恢复/资源/结果/局限，不将文件可读写成效果通过。若要求启动即结束，在后续确认结束或接手时补报告，不持续在线监控或重跑。当前密集点阵面片、原观测点重绘与首尾不一致均已记录；未实时播放或三维验证。


## 2026-11-09：0号建筑点云V3三墙对照启动准备

快照时间：2026-11-09T17:52:02.437838+09:00。用户已明确授权本组仅一次原版VACE视频补全，并要求确认本run第1/30步完成或sampling_started.json生成后立即结束会话。实际输入与V2是不同文件，V3保留A/B选区、只增加C墙约5.72米空区（米制尺寸为用户描述，未独立测量）。无新增训练/LoRA/参数搜索授权。

研究入口：[V3三墙实验](experiments/pointcloud_vace/0_v3_base/README.md)；本run：`outputs/pointcloud_vace/0_v3_base/run_20261109_175027`。已有解压输入与data/pointcloud_upload/0_v3_input.zip的三个成员逐字节一致；视频SHA256前16位16eb946b679a056a，mask前16位db445e937f7f0e9a。两视频均81帧832×480/16FPS、逐帧时间戳对齐；mask严格0/255、白区对应RGB127，提示词为包内原文且与V2字节相同。V2白像素移除0，新增C白像素跨81帧合计487026，新增区域外RGB变化0；000/080输入和mask相同。

代码只在experiments/pointcloud_vace/0_v3_base/scripts/run.py做prepare适配，worker/采样/保存/supervise与V2相同；正式model与inference字段已与V2 run_20261109_021750/config.json逐项相等：原版Wan2.1-VACE-1.3B、无LoRA，seed20261018、30步、CFG5、shift5、VACE scale1、BF16、tiled VAE、full memory，negative为空、无扩写。实际传vace_video和vace_video_mask，080保留真实预测，不分段/抽帧/缩放；无旧训练缓存。

本快照为prepare完成、正式GPU尚未派发；原版权重/T5/VAE/tokenizer完整校验与现有scj/A6000环境检查通过，无其他GPU作业。实际派发身份看本run launch.json、run.pid/worker.pid；第1步完成证据为sampling_started.json或日志completed_steps=1，加载完成不算。后台按V2自动保存raw_vace.mp4、81 raw_frames、三栏input_mask_raw.mp4、日志、配置和真实exit.json。

效果留待输出后复核，不预写C墙补全、A/B点阵或闭环结论；后续完成或明确失败收尾时报告写`upload/20261109_0号建筑点云_V3三墙对照_Wan21_VACE原版_实验报告.md`并登记upload索引。旧V2/mesh实验和原长期历史保留，不重复已成功推理。


### 2026-11-09：V3三墙对照后台派发快照

2026-11-09T17:58:21.608660+09:00：已派发独立session 4037385，父PID 4037385、GPU worker PID 4037389，stdin=/dev/null，日志为`/mnt/windowsE/chuanjun/building_wan/outputs/pointcloud_vace/0_v3_base/run_20261109_175027/run.log`，身份另存本run launch.json/run.pid/worker.pid。本快照记录后台派发，正式采样验收以本run sampling_started.json或completed_steps=1日志为准。实际配置逐项等同V2，V3输入指纹及包内prompt已核验；298项CPU检查通过（51.218秒），规定语法/编译/diff检查通过。用户要求首步完成即结束会话，不等待完整媒体或效果；后台将自动保存本次产物及真实退出，后续再补upload详细报告，不重跑成功任务。


## 2026-11-09：V3三墙完成后的只读查看与报告

本run已成功退出，不再执行prepare/supervise：`outputs/pointcloud_vace/0_v3_base/run_20261109_175027`。父PID4037385和worker4037389已退出，真实返回0；30步/81帧齐全，18:03:56东京时间结束，对应服务器17:03:56。

```bash
cat /mnt/windowsE/chuanjun/building_wan/outputs/pointcloud_vace/0_v3_base/run_20261109_175027/exit.json
ls -lh /mnt/windowsE/chuanjun/building_wan/outputs/pointcloud_vace/0_v3_base/run_20261109_175027/raw_vace.mp4 /mnt/windowsE/chuanjun/building_wan/outputs/pointcloud_vace/0_v3_base/run_20261109_175027/input_mask_raw.mp4
cat /mnt/windowsE/chuanjun/building_wan/outputs/pointcloud_vace/0_v3_base/run_20261109_175027/report_review_20261109_185922/final_review.json
```

[完整报告](upload/20261109_0号建筑点云_V3三墙对照_Wan21_VACE原版_实验报告.md)已按要求进入upload。原status的complete_awaiting_visual_review、环境检查的GPU_inference_started=false和启动待复核MD保留当时身份；本次实际完成与静态复核看新final_review.json，不能把旧快照当重跑理由。

全81帧静态检查和V2配对显示C/A/B仍为暗底点阵面片，首尾不一致；未实时播放或三维验证。运行退出0与效果达标分别记录。启动推送8秒超时不影响后台推理，本轮核对并同步已有main提交；无追加GPU授权。


## 2026-11-10：天然残缺点云 D 组八视角 Adapter 正式训练启动交接

记录时间：2026-11-10T05:12:38.059111+09:00。用户新授权只实现并启动 D 组，确认首次真实 AdamW 更新后结束会话；八栋开发样例、evaluate.py、A/B/C 对照和视频验证均待训练后另开会话。新实验入口：[D组结构Adapter](experiments/pointcloud_vace/adapter_d/README.md)。

固定 run：`outputs/pointcloud_vace/adapter_d/run_20261110_050724`；独立 tmux `apc2mesh_D_20261110_050724`，启动 shell/随后exec的supervisor PID `129571`。为先登记真实作业身份，启动器使用一次性start.ready放行；放行后无Codex/SSH依赖。GPU worker PID在worker.pid，正式首步凭first_update.json的两组参数改变、AdamW step=1及恢复点保存确认，不能把本段准备快照或模型加载当首步。完整日志run.log、状态status.json、退出exit.json自动记录。

源包apc2mesh_adapter_full_v1.zip已在CRC/数量/任务单可读验证后删除，导入记录为data/pointcloud_vace/apc2mesh_full_v1_import.json。目标目录下有15,881份pointcloud和15,891个mesh文件，完整微调任务与sample-data-organized同级；旧V2/V3保留。仅按候选空间划分.csv中use=train的10,747栋取样，本轮固定3,000栋/3,000更新，seed20261110。原始点云作输入，mesh与一次生成的候选点只给监督；真实互补扫描证据缺失，不虚构真实补充标签。已知/真实补充/弱区分别面积归一，权重1/1/0.3，未知缺口0。首批为11613/Paide、026–042，弱候选38点。

复用V3原版Wan2.1-VACE-1.3B及scj环境，原参数全部冻结；只优化新Adapter142,929参数和VACE self/cross attention q/k/v/o rank=alpha8 LoRA2,949,120参数，共3,092,049。AdamW lr1e-4/wd0/clip1，BF16骨干/FP32增量、batch1/accum1、832×480、连续17帧、gradient checkpointing，原prompt、无旧LoRA。首步后每100步保存完整恢复状态，结束独立保存adapter_final.safetensors和lora_final.safetensors。启动OOM只按任务单依次启用CPU激活卸载、必要时13帧，实际配置与各失败尝试独立记录；首步成功后不自动重跑。

8项专项CPU检查与298项项目CPU回归通过（51.151秒）；计划3,000对点云指纹/解析及mesh配对全部通过，mesh替换不会改变RGB/mask/几何输入。PEFT原地插入接口差异已在CPU联调修正；无额外GPU试跑。包内原渲染入口引用未随包提供的旧路径，本轮训练读取/渲染适配详见实验README，未修改上传脚本或原始点。19项既有无关工作区修改保持原样。

训练完成或明确失败后，按项目规则在upload补详细实验报告并登记索引，同步长期MD；本轮启动交接不等待训练完成，不预写模型效果。只读进度：`tail -n 30 /mnt/windowsE/chuanjun/building_wan/outputs/pointcloud_vace/adapter_d/run_20261110_050724/run.log`。


## 2026-11-10：D组八视角Adapter训练完成，待视频验收

2026-11-10T14:54:28+09:00核查：固定run `outputs/pointcloud_vace/adapter_d/run_20261110_050724` 已完成3,000/3,000次真实AdamW更新、3,000栋use=train建筑，2026-11-10T13:04:49+09:00正常退出0，28382.635秒（7小时53分）。原版VACE冻结，Adapter142929＋注意力LoRA2949120参数；17帧832×480/BF16配置未降级，无OOM或重试。最终两类增量及完整恢复点已CPU读回验证，264个optimizer参数step均为3000；峰值allocated/reserved为15.109656/15.361328GiB。

[训练阶段详细实验报告](upload/20261110_点云D组_八视角结构Adapter微调_训练阶段实验报告.md)已归档。弱目标构造成功165批，当前窗口有效弱区162批（5.4%）、其中有效时间权重159批；真实互补监督0，其余主要为可靠观测区学习。结构上游首次非零梯度在第3步，累计2492批；最终全部24个Adapter/240个LoRA张量改变。该结果证明执行完成，墙面稀疏点、檐口、漂移/闭环效果均尚未验证。八栋推理、evaluate.py及其他对照仍等待用户新授权，本次状态收尾GPU任务0。

完成证据见本run training_complete.json/exit.json及completion_review_20261110_145428/completion_check.json。此前启动快照和全部历史保留；不能把待评测状态当作需要重跑训练。

启动更正：此前--hold/start.ready等待说明未实际生效（shell条件右括号前缺空格，start.ready不存在）；真实首步为05:12:03+09:00。正式训练单次执行并正常完成，本次不修改冻结源码；后续如依赖等待功能，需先修正和测试。


## 2026-11-10：D-3000八栋验收推理后台队列启动

记录时间：2026-11-10T15:12:52.791443+09:00。用户新授权生成训练后D组的八栋验收视频，要求确认进入实际采样后立即交接。新入口：[D组八栋评测](experiments/pointcloud_vace/adapter_d_eval/README.md)。唯一run `outputs/pointcloud_vace/adapter_d_eval/run_20261110_150749`，独立tmux `apc2mesh_D_eval_20261110_150749`，启动shell/随后exec的supervisor PID 459101；本段记录真实队列身份，当前等待一次性start.ready放行，实际GPU worker见worker.pid，首步验收以本run sampling_started.json为准。

只加载训练run_20261110_050724的最终Adapter与LoRA-3000，在同一原版Wan2.1-VACE-1.3B基座上串行推理eave_3230、roof_12512、roof_5670、strip_5016、wall_10416、wall_13697、wall_2534、wall_8079，共8次完整81帧调用。832×480/16FPS，30步/CFG5/seed20261018/shift5/VACE scale1，原prompt逐字使用，negative为空，无扩写或帧080替换。原骨干BF16、增量FP32。

八栋均为diagnostic_only且未进入实际3000栋训练，是开发验收、不是盲测。结构特征先剔除全部hidden_original_point_ids，只用观测P和包内原相机，不读mesh/隐藏点/目标RGB。两个输入视频均完整解码与时间对齐，648/648帧观测P重建在mask外与上传RGB逐值相同；CFG正负两路共享同一32760-token几何。7项专项CPU检查及298项项目CPU回归（50.512秒）通过，原训练源码与19项无关修改保留。

后台每栋自动保存81张raw_frames、raw_vace.mp4、四栏input_mask_D_target.mp4、5张对比PNG/overview及原evaluate.py指标；GT只在原始预测保存后由CPU读取。全队列汇总metrics_summary.json/csv及真实exit.json，完成标记awaiting_visual_review。主run.log同步采样进度，原始预测不回贴或覆盖；不重跑成功轨迹，不训练其他组。wall_10416无隐藏实测点，其恢复指标null应按指标支持量解释。

本轮首步采样确认后即结束，不等待8栋结果，不提前写效果结论。后续完成/失败及视觉复核时追加upload报告；训练阶段报告保留，不能把本启动记录当已生成视频。只读进度：`tail -n 30 -f /mnt/windowsE/chuanjun/building_wan/outputs/pointcloud_vace/adapter_d_eval/run_20261110_150749/run.log`。


## 2026-11-10：D-3000八栋视频和自动指标全部完成

本次状态核查确认run `outputs/pointcloud_vace/adapter_d_eval/run_20261110_150749` 已完成8/8条81帧推理，8个GPU worker全部exit0，无重试；全队列2026-11-10T16:02:08+09:00退出0，2839.762秒。8原始视频＋8四栏对比、648原PNG＋48展示PNG、8份指标和JSON/CSV汇总齐全，本次CPU重新完整解码16视频/1296帧及696张PNG通过。峰值allocated/reserved为18.601651/31.675781GiB。

[八栋推理与自动指标报告](upload/20261110_点云D组_八栋81帧推理与自动指标_实验报告.md)已归档。参考背景误填29.865%–87.838%，7个可计算白区首尾MAE为50.356–83.551/255；高点中心命中不能抵消背景误填。wall_10416无隐藏实测点，恢复类指标null；wall_2534首帧白区为空，白区首尾MAE无定义。当前仅完整性与自动指标复核，未人工看图/实时播放或三维跟踪，status为complete_awaiting_visual_review，不将生成完成写成视觉效果通过。

progress.json的running是最后批次快照，completed_cases已有8/8；以status.json/exit.json及新completion_check为完成依据，不重启已成功推理。原始日志/配置/视频和启动历史保留，本次新增GPU任务0。下一阶段是现有视频视觉复核；新训练或其他组仍需用户授权。


## 2026-11-11：三栋固定三维补点已交付，本轮不是训练

新run `outputs/pointcloud_vace/fixed_points_three/run_20261110_234704`，CPU完成三栋各81帧及原evaluate.py对照，退出0；不加载Wan或增量，不增加训练或D推理。源码在experiments/pointcloud_vace/fixed_points_three/scripts，[报告](upload/20261110_三栋固定三维补点_与D3000对照_实验报告.md)。复用scj Python，CUDA_VISIBLE_DEVICES为空，BLAS/OMP1线程，视频编码2线程。

只读状态与指标命令：

```bash
cat /mnt/windowsE/chuanjun/building_wan/outputs/pointcloud_vace/fixed_points_three/run_20261110_234704/status.json
cat /mnt/windowsE/chuanjun/building_wan/outputs/pointcloud_vace/fixed_points_three/run_20261110_234704/metrics_comparison.csv
cat /mnt/windowsE/chuanjun/building_wan/outputs/pointcloud_vace/fixed_points_three/run_20261110_234704/longwall_42_coverage.csv
```

run.py是执行入口，当前run已存在generation_all_frozen.json，禁止以运行它查询状态；它会拒绝再生成，避免参考揭示后覆盖候选。review.py仅后验制图，也不需用来刷新状态。旧D权重、旧GPU入口及冻结run均保持不动。若未来获复验授权，新建独立run，先剔隐藏点再拟合，全部三栋冻结后才评分；具体已执行命令和参数在实验README/报告。

关键排错事实：04只产生20候选/保留6，0.3m只覆盖10/42（保留9/42）；最终6点RGB231/238/220与背景最大差24，被原>30前景阈值排除，漏补100%包含几何及颜色原因。不能只调疏密，也不改评分阈值或擅自染黑。07没有可用扫描间距而拒绝新增，0隐藏点的恢复指标为null。

全部三栋81帧P重渲染与原damaged逐RGB相同；独立000/080一致不是复制尾帧。已通过2专项、298项目CPU测试及根规范shell/编译/diff检查。500步小模块未训练，用户看完再另行安排，不能据下一步建议自动启动。


## 2026-11-11：成品查看入口

三栋结果直接打开`/mnt/windowsE/chuanjun/building_wan/outputs/pointcloud_vace/fixed_points_three/交付_20261110_234704/`，其中只有三条对比视频和实验报告。无需运行任何训练、推理或制图脚本；旧run/status路径由归档链接保持可用。后续默认只向用户交付视频和一份报告，调试及复现材料内部归档。


## 2026-11-11：清理后不要再使用旧状态/恢复命令

用户授权删除五组已完成点云实验的中间产物，旧run.log/status.json/config.json/PID/缓存及过程checkpoint均已删除；上文基于这些文件的查询命令是历史说明，现在不再适用。直接查看各组保留的实验报告.md及视频；三栋当前交付目录仍为outputs/pointcloud_vace/fixed_points_three/交付_20261110_234704，旧run链接和.archive均已删，不要重建归档或自动重跑。

D训练仅保留checkpoints/adapter_final.safetensors和lora_final.safetensors，哈希与原报告一致；resume_latest.pt、各step完整状态和初始化增量已删除，不能声称还能完整恢复optimizer/RNG或复用原存盘初始化。最终权重仍可用于后续获授权的推理。清理时只停止了占用已结束D日志的旧tail查看进程，没有停止GPU作业。没有修改训练源码或新增训练。


## 2026-11-11：五栏视频仅为CPU拼版，查看不用执行入口

当前三栋仍直接打开outputs/pointcloud_vace/fixed_points_three/交付_20261110_234704，视频文件名已改为五栏对比；上排Origin/未填灰残缺点云，下排灰填输入/D-3000/固定三维补点。81帧/16 FPS，每栏832×480，成片2496×1136。

专用媒体工具为experiments/pointcloud_vace/fixed_points_three/scripts/reformat_comparisons.py，只读取原case视频、已有D预测与旧四栏第三栏，没有模型加载或候选生成。它在三条新成片全部完整解码后才替换旧视频，并删除临时编码文件。此次已完成替换，旧四栏已删，不能再次运行它刷新状态；不要调用run.py重建已清理的点集。H.264展示视频不代替原无损帧的评估证据。根规范的shell语法/编译/diff检查及298项CPU回归通过。


## 2026-11-12：五栏视频点数与mesh使用说明

本次只用CPU更新现有三条五栏视频的点数标题，分辨率2496×1248，每栏画面832×480不变，81帧/16 FPS。媒体工具reformat_comparisons.py新增--annotate-counts模式，仅针对原1136像素高五栏成片，在三条新视频全验证后替换同名视频；本次已完成，不再次执行刷新状态。固定数量读取保留报告，原始/残缺数量核对原点云；不会加载模型或重建点集。D三维输出点数不适用，不进行图像斑点计数。根规范的shell语法、编译、diff及298项CPU回归通过。
