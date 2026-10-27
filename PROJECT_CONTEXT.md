# 项目上下文与持续更新记录

最近补记：2026-10-23（UTC+08:00，北京时间）。本文件持续保留项目目标、历次状态、失败与修复、结果证据和仍未解决的问题。历史正文已恢复在原文件内，后续进展按日期追加；归档仅作备份，无需离开本文件才能阅读旧记录。

阅读顺序：[此前累计正文](#history-records) → [2026-10-23阶段更新](#update-20261023) → [本次历史恢复及生效约定](#history-repair)。旧段落的“当前”“下一步”、命令和PID按其记录时点理解；较晚的更正与实际完成证据在后文，旧命令不因此重新执行。

截至本次修复：Pilot8系列及Complex12的C/U/S各300更新、五方法60条81帧已完成。本次仅修复文档；[实验入口](experiments/README.md)和[文档索引](docs/DOCUMENTATION_INDEX.md)仍可直接使用。

<a id="history-records"></a>

## 一、此前累计正文：截至2026-10-22的累计记录

以下完整保留原有记录顺序、参数、命令、过程和当时结论，只调整标题层级。原记录中的未完成状态与后来完成状态共同构成历史，不能用后者替换前者。

<!-- RESTORED_HISTORY_BEGIN -->
### Project Context

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

更新时间：2026-10-20 20:09:57 +09:00。LoRA-300八例81帧评估全部完成，八个worker和两个批次均退出0；32项成果通过验收，GPU已空闲，本轮停止。

持续工作约定：用户每次实际启动GPU训练或推理后，主动核实进度/最终状态、更新项目文档，并提交推送本轮相关代码、配置和文档；不等再次提醒。数据、模型、成果媒体与日志留在原目录，不入Git。该规则已写入AGENTS.md和SESSION_HANDOFF.md。

#### 当前：LoRA-300八例81帧评估完成

- 成果根目录：`outputs/vace/lora300_eval81/run_20261020_194300_145372/`。01–08各恰好四文件：原始LoRA视频、Before/Base/LoRA-300/GT四栏视频、四行五列总览、五栏Mask图，共32项、26,961,833 bytes。没有回贴、调色或美化。
- 唯一权重仍为 `outputs/vace/lora_roof_smoke/run_20261020_165237_150325_2048720/training/lora_update0300.safetensors`，终端逐例打印绝对路径；只加载vace、scale1，每例fresh原始VACE-1.3B。固定B000–B080/512×512/16FPS/30步/CFG5/seed20261018及原正负文本。
- 八例Base均复用 `outputs/vace/test/01–08/01_补全结果.mp4`，执行代码/冻结配置及隐式默认参数已核对一致。Damaged填灰127和原Exact生成的二值ModelMask不变，P01/P02/P03各27张全黑Mask保留；GT仅在推理返回后读取。
- 先完成P04→P06，查看全部81帧缺口对照并在终端明确确认整体改善后，才按原授权执行P01→P02→P03→P05→P07→P08。两个批次分别369.2秒、1129.4秒，合计1498.6秒；八例峰值均allocated17.141GiB/reserved25.832GiB，全部worker与批次退出0。
- P04屋顶凹槽全环绕持续填平；P06黑洞在所有81帧保持闭合，后半圈未重新开洞。两者仍有轻微表面痕迹/侧向轮廓起伏，是训练题的81帧扩展验证，不能归为未见建筑迁移。
- 六例迁移中P05/P07/P08明显改善；P01/P02改善有限；P03整体基本持平、部分后段视角变差。P05前半圈仍有小凹点且首尾不完全一致，P07端面尚有灰度不均。已查看八例全部81帧局部对照、五视角总览和最大Mask图；逐例数值与限制见EXPERIMENTS.md。
- 16个视频在制图阶段均完整解码并检查帧序/面板内容，终态再次确认81帧/16FPS、原始512×512、比较2048×544、H.264 High/yuv420p/BT.709；16张PNG可读。总览统一B000/B020/B040/B060/B080，Mask图按原Mask最大白区选帧。临时查看图已清理。
- 原test/composite_check/小样本run及输入包69项文件大小和mtime不变。tmux `vace_p01_17f:4.0` / `%4`已回shell，NVML无计算/图形进程。新训练0、Base补跑0；本轮实际LoRA推理9次，包含首次制图失败后经用户明确授权重跑的P04一次。
- 后续建议保持VACE骨干不变，优先扩大建筑补全LoRA训练集并保留独立测试建筑，之后再研究几何/Depth约束；这是后续方向，不是执行授权。本轮完成即停。
- 归档检查：空CUDA_VISIBLE_DEVICES下完整298项CPU回归通过（50.927秒），规定及新增入口的语法/编译检查、git diff检查通过；原有其他未提交文件状态与接手快照一致。

#### 历史：首次81帧评估在CPU制图阶段停止

- 首次P04真实30/30步与VAE解码完成，模型加载至返回175.2秒，峰值17.141GiB allocated /25.832GiB reserved。随后 `comparison_panel` 接收到PIL对象而非NumPy数组，抛出 `TypeError: a bytes-like object is required, not 'Image'`，worker和批次均退出1，未启动P06。
- 原始MP4曾在临时目录编码完成，但失败时临时目录被清理；当时未保留预测，不能用首次推理评价视觉效果。修复统一转换制图像素类型，并先将原始MP4保存到最终样本目录；后续CPU制图失败可用 `--render-only` 完成，不需重新推理。用户回复“直接修复，然后再重跑”后，才执行上方已完成批次。
- 修复后298项CPU测试通过（53.124秒），包括81帧PIL输出制图、全部媒体解码、CPU制图失败后01保留且补图不改01。语法/编译/diff通过；没有用CPU结果冒充正式预测。

- 首次失败的实际tmux为 `vace_p01_17f:4.0`，窗口名 `lora300_eval81`，pane `%4`。18:24:35 +09:00启动快照不能覆盖其最终退出1；同一窗口随后承载了获授权的成功批次。
- 唯一权重：`outputs/vace/lora_roof_smoke/run_20261020_165237_150325_2048720/training/lora_update0300.safetensors`。该run已完成300更新，用户确认17帧有效；本轮只加载vace分支，scale1，不训练、不叠加其他LoRA。
- 新结果只写 `outputs/vace/lora300_eval81/`。顺序P04→P06，完成后先查看全环绕效果；仅当仍整体明显优于Base才执行P01/P02/P03/P05/P07/P08。每例独立进程重载原模型，异常即停，不自动重试。
- 八例Base均复用 `outputs/vace/test/01–08/01_补全结果.mp4`。执行入口和冻结配置确认81帧B000–B080、512、16FPS、30步、CFG5、seed20261018及正负文本一致；旧入口隐式cfg_merge/denoising_strength/VAE tile参数与现用显式参数一致。媒体自身不能反推seed，复用依据为既有执行证据和配置。
- Damaged/Exact用原Pilot8包，填灰127与二值Mask规则不变，全黑帧保留；GT像素只在推理返回后读取用于比较，不回贴或调色。每例四文件：原始LoRA视频、四栏81帧视频、四行五列总览、五栏Mask图。
- CPU准备已核对唯一权重metadata/结构、八例81帧输入与Base完整解码；297项CPU测试、规定语法/编译/diff检查通过。未做重复大文件SHA、依赖更新或模型改造。

#### 已完成：P04/P06 VACE条件LoRA首轮

详细说明见 [P04/P06 LoRA实验](docs/VACE_P04_P06_LORA_SMOKE.md)。当前run为 `outputs/vace/lora_roof_smoke/run_20261020_165237_150325_2048720/`，真实会话为 `vace_p01_17f`、窗口3、pane `%3`；不要误找尚未创建的 `vace_roof_smoke` 会话，也不要重复执行启动命令。

| 核查项 | 17:05:32的实际证据 |
| --- | --- |
| Base | 两例17帧均完成，worker退出0，阶段耗时61.052秒 |
| 训练 | `runtime.status=complete, stage=complete`；loss.csv连续300条，P04/P06各150条；训练worker退出0 |
| 真实梯度 | 终端前5步均确认有效梯度、LoRA参数变化、优化器仅含LoRA |
| 两份权重 | `training/lora_update0100.safetensors`、`lora_update0300.safetensors`，各43,780,800 bytes |
| 后评估 | 100/300各两次推理完成，累计6/6，四个GPU阶段均退出0 |
| 最终成果 | training四文件，04/06各四文件；视频17帧/16FPS可解码，PNG可读，自有临时目录已清理 |
| 耗时与峰值 | 总耗时749.247秒；训练阶段550.933秒，峰值15.139GiB allocated/15.393GiB reserved；全流程峰值15.867/17.385GiB |

可重连 `tmux attach-session -t vace_p01_17f`，选择窗口3查看完成记录；当前不需要再启动。已抽看两张五行总览：P04在100/300步后屋顶凹槽和台阶明显改善，P06开口表现为接近GT的连续表面，不只是变亮。该观察覆盖B000/B004/B008/B012/B016，完整17帧闪烁与人工验收仍待视频查看，runtime保留 `visual_assessment=pending_user_review`。

终端Mask内RGB MAE（0–255）为P04 Base27.2225→100步1.7813→300步1.5860，P06为46.3986→1.3737→1.3068。Mask外相对Damaged的平均变化也下降，但包含大量背景，不能替代建筑局部或几何评估；指标和视觉限制详见专门说明。

- 新固定配置 `configs/vace_roof_smoke_v1.json`；P04/source1227与P06/source1848均选B000–B016，分别为139483和206570个Exact白像素的17帧窗口。遍历B000–B079内64个连续窗口，最大总白区、并列最早，排除B080；读取真实原始PNG与相机，不用历史视频截图训练。
- 正负文本逐字继承已成功VACE冻结配置。GT仅经官方InputVideoEmbedder编码监督目标；Damaged经原 `adapt_images` 填灰127、Exact二值RGB作为真实 `vace_video_mask` 进入官方VACE条件Unit。冻结VAE/T5编码可内存复用，不缓存可训练VACE输出、不复用旧T2V缓存。
- 只训练vace的q/k/v/o/ffn.0/ffn.2 LoRA，rank=alpha=16、dropout0、biasnone、无RSLoRA/DoRA；150个Linear、300个A/B张量、10,936,320参数。BF16骨干和autocast，LoRA参数FP32以保留小更新；AdamW恒定5e-5、wd0、batch/accum1、clip1、checkpointing、workers0、seed20261018，仅300次真实更新，两例交替各150次，只存100/300。
- 实际完成的顺序为Base两次17帧→训练300更新→100权重两次17帧→300权重两次17帧→CPU制图。各GPU阶段独立进程，原始权重重新加载，LoRA倍率1仅fuse到vace，不叠加100/300，没有重跑81帧、重试或降规格。
- 准备阶段的 `--dry-run`、本地DiffSynth2.1.5接口/meta层匹配、实际512×512×17条件loader及294项CPU回归均通过；这些是启动前证据。用户随后启动的GPU实测进度见上表，不再沿用“尚未加载正式模型/未创建run”的旧状态。
- 实际输出为training两LoRA、experiment.json、loss.csv，04/06各四项可视化，共12文件；没有永久逐帧预测、输入副本或额外报告。临时预测数组已清理。该LoRA学过P04/P06，只能称训练题重建验证，不是泛化或81帧一致性证据。

#### 最新结果：仅替换缺损区域的CPU合成

- `scripts/composite_vace_results.py` 在现有tmux的 `vace_composite` 窗口实际执行，八例全部SUCCESS，耗时46.0秒、退出0、GPU推理0次。输出 `outputs/vace/composite_check/01–08/`，每例严格三文件：`01_仅缺口补全.mp4`、`02_四方视频对比.mp4`、`03_关键帧对比.png`。
- 使用已验收Pilot8包的真实源元数据、相机与512帧序列，包括身份已匹配的P01副本；复用原推理 `adapt_images` 从Exact生成二值ModelMask，不扩张、收缩、反转或羽化。白区逐像素取原预测MP4解码值，黑区取Damaged，GT只用于对照和关键帧选择；全黑帧原样保留Damaged。
- 两个视频均81帧/16FPS，分别512×512、2048×544，采用CPU H.264 High/yuv420p/BT.709，完整解码与RGB/帧序检查通过。总览2560×2176、四行五列，保留B000/B080，并优先展示最大Mask、可见Mask内RGB误差最大的视角及全黑视角；选择仅为诊断展示，不作为独立量化评估。
- 编码前黑区与Damaged完全一致，白区与Raw Prediction完全一致；MP4允许正常有损量化。P01颜色和主体保留改善，但有矩形拼接细缝；P04屋顶凹槽、P06深色洞口仍在，说明简单回贴不能恢复缺口内部结构。其他例也存在不同程度补丁边界，不能宣称三维一致性已解决。
- 原 `test/` 的32个文件SHA/大小/mtime均未变，源ZIP大小/mtime未变，自有临时目录已清理，无新日志/JSON/输入副本。272项CPU测试及规定语法/编译/diff检查通过；完成后停止，没有启动后续Mask或模型实验。

#### 已完成：八例结果升级为 Before / Prediction / GT

- 已完成八例CPU重排及后续播放偏色修复，终端P01–P08均SUCCESS、退出0、GPU推理次数0。八个三栏视频已全部替换为H.264 High/yuv420p/BT.709；独立比对修复前后32文件，只有八个02改变，八个原始预测与16张PNG全SHA不变，各目录仍恰好四文件，自有临时目录清理通过。完整266项CPU回归通过，哈希证据见EXPERIMENTS.md。
- 独立入口 `scripts/reorganize_vace_results.py`，不导入Torch/DiffSynth、不读取模型权重；默认只读验收，显式 `--apply` 才安全替换已有对比图像。缺少原始预测必须停止，没有模型回退。
- 输出仍为 `outputs/vace/test/01–08/`，每例恰好四项：原始 `01_补全结果.mp4`、`02_补前_补后_GT_三栏对比.mp4`、`03_补前_补后_GT_三行总览.png`、`04_缺损_Mask_预测_GT_四栏验证.png`。
- 原始MP4不移动、不重新编码，前后SHA256、inode、大小、mtime一致；四栏图核对本例B020源身份后仅改名，内容不变。三栏81帧/16FPS/1536×544，三行总览2560×1632，五列B000/B020/B040/B060/B080；每个主体保持512×512，不美化或GT回贴。
- 原三栏视频使用libx264rgb/gbrp（High 4:4:4 Predictive），FFmpeg像素检查通过，但用户播放器出现紫绿偏色；原先验收没有覆盖播放器兼容性。现改为CPU libx264/High/yuv420p/BT.709有限范围、CRF12与faststart。校验同时要求正确编码/像素格式/色彩标签及全部81帧各栏与原图误差受控（每通道MAE与均值偏差均≤3，99%像素绝对误差≤12/255），PNG仍逐像素相等；只允许常规编码量化，不进行调色或美化。
- 已有新版布局若使用RGB 4:4:4，`--apply`仅重新制作02三栏视频，原始预测和两张PNG的SHA/inode/mtime保持不变；已有兼容视频只读核验。交接包CRC、载荷哈希、图像、帧号和相机仍完整验收，各例原始黑Mask不改。
- 同分区、test目录以外的独占临时目录先生成并验收，再逐例替换三项衍生成果；失败回滚旧文件，回滚异常保留明确恢复路径。不得重新调用旧GPU入口处理新版文件名。

#### 历史阶段：校验失败排错，显式续跑 P04–P08（后续已成功）

- tmux实际错误是P04的 `Reused component SHA256 mismatch: text_encoder`，位于sample_preflight，尚未加载该例模型；不是CUDA OOM。P02/P03各30步与81帧解码/四文件发布已成功，实测峰值17.14GiB allocated /25.73GiB reserved；P01已有终端证据同为完整81帧成功。
- 文本编码器原件11,361,845,432bytes，连续三次独立完整 `sha256sum` 均为批准值 `d92de679881d38af9c89eff7bb1b6d6c9d96cb2b69831e4027e9ecabdd38eb23`，inode/size/mtime/ctime稳定。此前失败没有保存实际hash，暂不能复现或判断底层读取、内存或文件系统原因；不宣称硬件/磁盘问题已修复，也不认定权重持续损坏。
- `verify_file_sha256` 仍单次完整读取、任一不符即停止；新增expected/actual/path/size及读取前后stat，检测文件原地变化/替换，不自动重试或绕过校验，不修改批准hash或权重。
- 新显式命令为 `bash /mnt/windowsE/chuanjun/building_wan/scripts/run_vace_building_test.sh --resume --tmux`。先完整验收已有连续成功前缀的四文件/81帧/逐像素源身份，父进程和gate再比对四项hash，确认不变后只排队P04→P08。P01/P02/P03只读复用；普通首次启动依然拒绝已有02–08，显式恢复也拒绝不完整、错号、非连续、额外或软链接结果，不删除/覆盖任何既有文件。
- 已有 `test/01–03` 验收通过：P01与保留原件四SHA相同；P02对应source763，P03对应source871，总览五个Damaged与四栏Damaged/Mask/GT均逐像素匹配。`04–08`不存在，无失败半成品需清理。配置、文本、模型身份、81帧和四文件约束保持不变。
- 恢复入口实际 `--resume --dry-run` 通过（32.1s），列出的唯一GPU队列是P04–P08，权重全SHA、当前环境与七例CPU接口通过。完整250项CPU回归通过（本轮新增44项），语法/编译/diff检查通过；未执行resume GPU分支，原件/输入/冻结JSON未变，未提交Git。

#### 此前阶段：复用 P01，准备 P02–P08

- 已执行 `scripts/run_vace_building_test.sh --prepare`，全部七例预检通过后仅复制 P01；`test/` 当前仅有 `01/`，且恰好四个有效文件，`02–08` 尚未创建。原件为 `outputs/vace/P01_465_81f_20261019_171510_1691517/`。两个视频均 81 帧/16FPS，原始预测 512×512；总览及四栏的 Damaged/Mask/GT 与原始源465数据逐像素对应。
- 新交接包为 `data/buildingfiller_import/P02_P08_20261020/BuildingFiller_Pilot8_AllRes_v1.zip`，SHA256 `7f8eff9ddb3c05e43fe5a8623258f9dceb6fa61d547b244c4f2d441099b4fcf4`。包含全部八例和三种分辨率；CRC、6077 项载荷哈希、5832 张 PNG 实际解码、1944 个帧/相机绑定及几何绑定检查通过。包内 P01 仅核查身份，不导入或补全。
- 固定映射为 `02/P02/763`、`03/P03/871`、`04/P04/1227`、`05/P05/1259`、`06/P06/1848`、`07/P07/2297`、`08/P08/939`。P05/P07 的 family ID 不等于 source ID；以真实元数据为准。各样本均有 81 张 512×512 GT/Damaged/Exact 和配套相机。
- 有效人工验收均为 accepted，几何/渲染证据校验通过；包内历史快照的 pending/not_run 不覆盖 hash-bound 验收。`generation_evidence_verified=true`，但 `training_ready=false`、`generation_verified=false` 仍保持原义；只用于本次视觉功能验证，不能冒充训练发布或模型补全成功。
- 各自原始黑 Exact 保留：P02 为 023–049；P03 为 000–016、071–080；P04–P08 没有黑帧。不能套用 P01 的 046–072。Exact 二值白区生成、RGB 三通道、Damaged 白区填灰127，均在内存转换；模型条件无 GT RGB，GT B020 只在真实预测返回后读取。
- `scripts/vace_building_test.py` 与 P01 入口共用完整 81 帧执行核心，固定原始 VACE-1.3B、无 LoRA、B000–B080、512×512、16FPS、seed20261018、30步、CFG5、full BF16/tiled VAE，不分段或降规格。正负 prompt 逐字复用 P01 冻结配置。P01 旧实验没有 manifest，历史 seed/权重/prompt 不能仅从媒体独立反推；用户已确认成功，保留的成功执行代码和冻结配置未见参数差异，不能把这一限制伪装成独立运行溯源证据。
- 正式 VACE/T5/VAE/Tokenizer 完整哈希通过；A6000 探测时空闲约 47.13GiB、无其他 GPU 进程，项目盘约 358.8GiB。启动时全七例统一门禁，随后每例重新检查；父进程不初始化 CUDA，各样本独立子进程串行释放模型，不安装/下载/修改第三方核心。
- 用户启动命令为 `bash /mnt/windowsE/chuanjun/building_wan/scripts/run_vace_building_test.sh --tmux`，只执行 P02→P08。终端显示真实30步、当前例数、已完成数量、实际耗时和按已完成样本估算的批量 ETA。不写日志/JSON/输入或预测 PNG；自有系统临时目录编码并验收后只发布四文件。任何失败立即停止，已有结果不覆盖；部分批量成功后再次启动会拒绝既有 `02–08`，不得自动重算或跳过。

#### 历史准备：P01 81 帧极简视觉验证（已由后续真实结果更新）

- 新入口 `scripts/run_vace_p01_465_81f.sh --tmux`：前台可见或复用当前 tmux，真实 30 步及 VAE 进度，不写日志或 JSON。`--dry-run` 已实际通过完整模型 SHA256、环境和 81 帧 CPU loader 检查，没有创建实验输出。
- 复用冻结 17 帧 JSON 的 prompt/seed20261018/steps30/CFG5/BF16/512/16FPS 与同一 ModelMask 转换，仅在内存改为 source000–080/81 帧和四文件输出策略；原 JSON 字节不改。81 帧是一次整序列调用，不分段、不回退 offload、不降低规格。B080 正常进入模型，预测不复制 B000。
- 原 ZIP/81 组 Damaged+Exact/相机一致，046–072 共 27 张全黑标签原样保留。GT RGB 在模型返回后才单独读取 B020，唯一用于四栏对照，不进条件。
- VACE/T5/VAE/Tokenizer 完整官方固定哈希通过；A6000 空闲约 47.13 GiB、无其他 GPU 任务，项目盘约 358.8 GiB 可用。81 的 latent T=21、tokens21504，当前 SDPA 高效后端启用、长度合法，真实峰值尚未测，OOM 即停止而不改配置。
- 成功目录 `outputs/vace/P01_465_81f_<UTC时间>_<PID>/` 仅含 `01_补全结果.mp4`、`02_补前补后对比.mp4`、`03_补前补后总览.png`、`04_四栏验证.png`。不保存定量指标、输入副本、预测 PNG、运行报告；必要视频/图像先在本次独占系统 `/tmp` 目录编码验收，正常退出/失败清理，仅发布四个文件。原生 worker 失败由父进程接管；不是模型质量的正式定量验收。

#### 1. 目标与边界

当前任务是 Damaged + 独立适配的 ModelMask + 文本 → 原始 VACE 补全视频，GT 只用于评估。BuildingFiller 在个人 Windows 电脑独立制作数据，经实验室 Windows 主机转入 Linux 项目；Windows `D:\P01_465` 不能当作服务器路径。第一例为源 465、A1 Medium、删除表面积约 5.679%，不用历史 T2V LoRA。

历史 Wan2.1-T2V-1.3B LoRA 学习灰色、无纹理建筑 mesh 的体块、完整环绕和时序稳定性；单轨统一 prompt 的划分、缓存及权重保持独立。逐栋描述和 Cap3D 必须使用独立划分检查、缓存及输出。

BuildingFiller 已交接完整/残缺/移除 mesh 三元组与对齐 RGB/Exact 的 P01 数据；旧扫描器 GUI 源码仍缺失。当前 LoRA 对比及人工数据验收都没有证明 VACE 补全或三维重建能力。

#### 2026-10-20 17 帧准备历史（输出已由用户删除）

- 正式文件位于 `models/Wan2.1-VACE-1.3B/574e6a744642ce3bee319afc31496b88bde8aac4/diffusion_pytorch_model.safetensors`，7,146,067,848 bytes，完整 SHA256 匹配官方 `c46a6f5f7d32c453c3983bbc59761ea41cd02ad584fb55d1a7ee2b76145847a2`。1264/439 总/VACE 张量和双分支注册结构通过；T5/VAE/Tokenizer 完整哈希同样通过。
- 阶段 A 证据在 `outputs/vace_model_audit/P01_465_20261020_20261019T161614Z/`。A6000 CUDA/BF16 可用，空闲约 47.1 GiB，NVML 没有其他计算/图形进程；没有 `nvidia-smi` 命令不等于没有 GPU。tmux 3.2a、ffmpeg/ffprobe 已有，项目盘空闲约 359 GiB。
- 实际 CPU 准备在 `outputs/vace/P01_465_17f_baseline_cpu_prepare_20261020_20261019T162331/`，源 005–021、17 帧 RGB Mask/条件帧，当前 DiffSynth CPU 加载器合同通过；GT 仅另存评估，未进模型。
- `scripts/run_vace_p01_465_17f.sh` 由用户终端启动后只运行一次真实 17 帧模型调用。父进程实时转发无缓冲 stdout/stderr，保存 `run.log`/`worker.log`；真实 scheduler tqdm、VAE tile 进度、阶段状态可见，不改 vendor。失败保留真实退出码并停止，不自动重试、降规格或切换 offload。
- GPU 子进程正常退出且完成标记有效后，父进程才运行独立 `scripts/vace_p01_results.py`：验证原始 PNG/MP4，生成 B020 两栏、四栏、005/010/015/020 两行总图、同步对比视频及诊断指标。进程成功不等于补全质量或三维质量达标；没有预测时不伪造图。
- 本轮未执行 `--run`，没有 GPU 原始输出或效果结论；下一步是用户执行终端命令。冻结 JSON 的旧状态字段保留其冻结时含义，实际模型路径/哈希以运行 manifest 为准，参数字节未改。

#### 2026-10-19 权重核查与执行器历史

- 官方正式 VACE 文件为 7,146,067,848 bytes，SHA256 `c46a6f5f7d32c453c3983bbc59761ea41cd02ad584fb55d1a7ee2b76145847a2`，HF revision `574e6a744642ce3bee319afc31496b88bde8aac4`；远程约 130 KB 文件头与本地 DiT/VACE 双分支注册结构匹配。
- 扩展搜索找到 `/mnt/windowsE/szq/HuggingFaceWeights/Wan/Wan2.1-1.3B-VACE/diffusion_pytorch_model.safetensors`，但大小少 32 bytes、SHA256 为 `fcd12e2a8fab331ba0ab28dca371d90036a868287767f35677ff3f8e74db0969`。即使用官方文件头重建哈希仍不匹配，张量载荷不等同正式文件，禁止用作原始基线；其他训练/merged 权重也不使用。
- 现有 T5/VAE 全文件 SHA256 与官方 DiffSynth ModelScope 转换组件一致；Tokenizer 哈希固定，可全部复用，新增大文件仅约 7.15 GB。项目盘可用约 366 GiB。HF 下载 HEAD 200、文件头 Range 206，官方模型/公用组件元数据均可访问；未下载大模型。
- `scripts/vace_p01.py` 默认只读，`--prepare` 仅 CPU，`--run` 明确触发 GPU 子进程。冻结 JSON 逐字节 SHA256 校验，禁止擅改 005–021、512×512、16 fps、seed 20261018、30 步、CFG 5；不加载 LoRA。GPU 入口使用项目锁、校验全部权重哈希、禁用下载，并记录真实子进程退出。
- 真实准备结果在 `outputs/vace/P01_465_17f_baseline_cpu_prepare_20261019/`：17 张独立 RGB ModelMask、17 张填灰条件帧、原 Damaged/Exact 副本及单独 GT 评估目录。实际落盘文件经 DiffSynth CPU 预处理为 `[1,3,17,512,512]`，Mask 0/1，通过。没有生成预测或调用 VAE/DiT。
- 权重与网络证据在 `outputs/vace_model_audit/P01_465_20261019/`；来源与组件哈希在 `configs/vace_p01_model_sources_v1.json`。下载计划脚本 `scripts/download_vace_p01.sh` 默认不联网、不写模型目录；只执行过 dry-run。
- 当前只缺官方 VACE 文件和用户后续下载/GPU 授权。完整模型加载、真实峰值显存和生成效果未测，不承诺一定不 OOM；显式 CPU offload 是独立可选策略，不自动改变冻结参数。

#### 2026-10-18 P01 接收检查记录

- 执行环境为 SSH Linux `vccip`，用户 `scj`；不能直接检查 Windows 主机 D 盘或其 OpenSSH 配置。本轮没有执行 SCP；已有文件位于 `data/buildingfiller_import/P01_465/`。
- ZIP SHA256 `5a966a269866a1423c1efa38937fcce6ec6984f8fda78c35395abccab83d5bb8`；ZIP CRC、262 项载荷哈希、243 PNG 实际解码、81 帧三角色/相机对齐、三对首尾闭合及随包校验器均通过。Exact 为 0/255，046–072 共 27 张全黑，合法保留。
- 新审计副本及报告在 `outputs/buildingfiller_handoff/P01_465_20261018/`；原 ZIP/Exact/Damaged 未改。其他七例和 768/1024 图片不在本次服务器载荷检查范围。
- RTX A6000 CUDA 可用，总显存约 47.4 GiB、探测时空闲约 47.1 GiB；DiffSynth 2.1.5 离线导入通过。已有 T2V DiT 没有 VACE 分支。共享 T5/VAE/tokenizer 存在；`/home/shi/.cache` 无读取权限，未把模型缺失结论扩展到未检查位置。
- 冻结计划 `configs/vace_p01_465_17f_v1.json`：连续源 005–021、17 帧、512×512、16 fps、原始 VACE-1.3B、不加载 LoRA。当前 pipeline 需要 RGB 三通道 Mask 列表，直接 L 模式会报错。基于 Exact 二值化、复制通道和 Damaged 白区填 RGB127 的 CPU 内存合同检查通过，但未落盘派生输入或调用模型。
- 下一步需明确 VACE 权重位置、实现项目专用执行器并落盘独立派生输入；GPU 推理必须等待用户确认。`training_ready=false` 保留，GT-oracle Mask 必须在结果中披露。

#### 2. 2026-10-14 核对到的状态

- 项目路径：`/mnt/windowsE/chuanjun/building_wan`；Git 分支 `main`，本次修改前提交 `8a3720e`。
- Python：`/home/shi/miniconda3/envs/scj/bin/python`；本次确认 PyTorch `2.11.0+cu128`、CUDA runtime `12.8`、CUDA 可用。历史硬件记录为单张 RTX A6000，约 47.4 GiB；本次未重做完整硬件/显存测试。
- `data/single_orbit_videos` 有 2344 个视频；冻结划分 train 1992、val 234、test 118，视频路径存在，文件名集合无交叉。当前单轨划分按视频名处理；没有在本次审计中证明源 OBJ 身份无泄漏。
- 缓存递归计数为 1992 个 `.pth`，共享 context 和 v2 格式标记存在。10-15 已全量只读检查：latent 均为 FP32，shape 为 (1,16,21,48,48)，全部有限且不携带梯度；共享 context 为 BF16，shape 为 (1,512,4096)，同样有限且不携带梯度。训练输入再转为 BF16。
- `checkpoints/building_wan_a6000_single_orbit_all/` 平铺 28 份 LoRA，step-400 至 step-11200，每 400 一个。三段训练参数、loss 和两份 run manifest 改名保留。本次 28 份 LoRA 的 safetensors 头部均可读，每份 600 个张量；未重新扫描所有张量有限值。
- `checkpoints/building_wan/` 保留 A100 历史的 39 份权重，不与当前单轨结果混用。
- `outputs/validation/9.28/` 有 5 个阶段对比视频。
- `outputs/validation/validation_all_single_orbit/` 有基础模型及全部 28 份权重的 29 个视频；元数据记录 2026-09-29 17:08:36（Asia/Shanghai）全部完成，参数为 384×384、81 帧、seed 20260924、30 推理步、CFG 5、18 fps。
- 全部验证目录已有 `_review/` 截图、拼图和 v1 像素指标。v1 前景占比计算有分母问题，应标为历史诊断；v2 写入独立目录，不覆盖旧结果。

以上视频数量与元数据核对不替代本次重新解码，也不代表视觉质量已接受。

#### 3. 配置与步数口径

| 项目 | 原训练 | 后续续训 / 当前脚本默认 |
|---|---|---|
| 物理 batch | 2 | 1 |
| 梯度累积 | 2 | 4 |
| 有效 batch | 4 | 4 |
| 数据 worker | 2（保存参数） | 0 |
| 分辨率 / 帧数 | 384×384 / 81 | 相同 |
| LoRA rank / 学习率 | 16 / 2e-5 | 相同 |
| 计划训练长度 | 8 epochs | 相同的总样本预算，跳过此前预算 |

`step-N` 和 `loss.csv` 的 step 是 dataloader batch 计数，每个物理 batch 增加一次，并不是梯度累积后的 optimizer update 数。改 batch 后不能只用 `N × 当前 batch` 推算累计样本。

续训使用 `--lora_checkpoint`，恢复 LoRA 权重；不恢复 optimizer、scheduler、RNG、未提交的累积梯度或相同 shuffle 顺序。跳过 batch 表示近似样本预算续跑，不是精确恢复训练状态。

#### 4. 训练完成状态：待进一步核实

此前文档报告“2026-09-29 02:25 自然完成 8 epochs，最后 step-11200”。保留这一历史报告，但本次不能独立确认正常结束：

- 最高现存权重是 step-11200。
- 最后一段 loss 从 3601 到 11562，共 7962 行。
- 最后一段参数为 batch 1、initial 3600、skip 6000、1992 样本、8 epochs；按当前引擎语义，耗尽配置预算应到 `3600 + (1992×8 - 6000) = 13536`。
- 保留日志尾部停在 523/1992，没有独立成功退出记录；不存在历史 `training_completion.json`。

这些不一致可能来自日志/产物缺失或实际运行差异，不能推断已经完成，也不能据此认定某次崩溃。不得自动补训。今后的项目训练入口仅在引擎正常返回、最终保存完成后写 `training_completion.json`。

#### 5. 已知故障与修复范围

- 原训练在日志 batch 2742 发生 `free(): invalid next size (normal)`。这是原生内存 abort 记录，未见 GPU OOM 报错；根因未确认。
- worker 改为 0、batch 改为 1 后仍记录过 `free(): invalid next size (fast)`；不能声称 worker 0 已消除故障。
- LoRA 被误传给 `--resume_from_checkpoint` 的 600 unexpected keys 问题已通过 `--lora_checkpoint` 修复；该参数也不提供 optimizer 状态恢复。
- 本次修复平铺后的续训默认路径、累计样本/batch 换算、项目进程识别、Python 传递、启动失败报告和新输出目录隔离。
- 当前训练/续训/验证入口共享 launcher 锁；Python 验证依据解析后的 run 参数加锁，缩写参数同样受保护。旧进程仍需检查。其他绕过入口的命令不受该锁协调。
- 当前 Python 验证预检不写输出目录；只在用户明确要求 `--run` 时保存配置及生成视频。

#### 6. 历史 T2V 待办

当前优先事项以本文件前面的 P01 / VACE 接收检查为准；下列待办属于历史训练与采集器维护，不触发自动训练或推理。

1. 补齐训练终态证据与源 OBJ 身份对应关系，核实已有预算和数据防泄漏。
2. 基于已生成视频做人工检查和多 seed 对比，选择 checkpoint，避免自动选最后一份权重。
3. 评估逐栋 prompt 时设计新缓存，不复用共享 context。
4. 如维护历史采集器，先恢复并审查缺失的扫描器源代码，落实“冻结一次、切分冻结 mesh、无额外变换地渲染三元组”。当前 VACE 读取独立 BuildingFiller 包，按 P01 交接方案实现。

数据、模型、缓存、已有权重和输出保持默认只读。新实验使用独立输出目录。Windows workspace 文件保持不动。

#### 7. 本轮复审问题已修正

复审发现并修正：验证缩写参数绕过锁；打乱分组下短 batch 的样本预算推断；暂停等号参数解析。短 batch 的局部 epoch 游标现在拒绝猜测，完整 epoch 边界仍可换算；当前 1992 样本、batch 1/2 的历史换算不变。

补测发现 scj Python 的 os/signal 没有 pidfd 接口；停止入口改为使用系统 Python 的 pidfd 控制，重复核对进程启动时间和项目命令。通过临时 CPU 进程验证了只记录不终止、两种输出参数的暂停/终止、身份不匹配拒绝发信号，以及缩写验证命令在锁被占用时拒绝启动。以上早期代码审计未运行正式 GPU 训练/推理；随后获授权的 GPU 诊断见下一节。

权重头部预检改为 safetensors NumPy 后端，不加载 PyTorch，非法游标在打开权重文件前拒绝。补充严格退出码测试，避免把 native abort 当作正常的参数拒绝；正式训练的原生崩溃根因仍未确认。

#### 8. 2026-10-14 至 15：真实 GPU 诊断

此前用户授权诊断并选择跑完剩余 2336 个 batch；最新停止和清理要求已覆盖此前继续运行的授权，不得自动续跑。
GDB 基线从 step-11200 续跑，在新增第 166 个 batch 后（loss step 11366）
触发 SIGSEGV，退出码 255；检测位置为 UnsafeViewBackward0 析构/计算图释放。
该基线已结束，原始权重和历史输出未覆盖。

2026-10-15 独立 Valgrind 主机内存检查覆盖 196 个真实 batch，报告 0 个
内存访问错误；按诊断计划主动停止，退出码 143，不算完成训练预算。
CPU 运算校验 120 秒退出 0，8 GiB memtester 一轮完整通过；这些范围内未发现
硬件错误，不能据此排除所有硬件故障。主机为 i9-14900K，微码 0x12f。

同日全量抽取缓存：1992 个 latent 均为 FP32，shape (1,16,21,48,48)，
全部有限且不携带梯度；共享 BF16 context 也有限、shape (1,512,4096)。
GPU memcheck 实际覆盖 4 个完整 batch / 一次累积更新，报告 0 个内存错误；
因开销较大主动停止，原计划 8 batch 未完成。固定种子普通反向 256-batch
对照正常结束，说明故障并非每次必现。

项目入口新增可选 WAN_SERIAL_BACKWARD（默认 0）与 WAN_TRAINING_SEED。
串行完整预算测试于 10-15 06:07（Asia/Shanghai）结束：2336 个连续 batch，
logger step 11201–13536；loss 全部有限，末值 0.00538101838901639。
6 份新权重各 600 个 BF16 张量均有限，键名和形状与来源一致。
GDB 明确记录 Python exited normally；旧调试命令随后因 py-bt 不可用返回 1，
这是调试器误报，已与 Python 正常退出证据分开核对。新版 scripts/wan_gdb.py
已修复此误判；正常退出、Python 错误、SIGABRT、SIGSEGV 有 CPU 集成回归。
该 GPU 长测启动早于 GDB 修复，不能说已使用新版 helper 验证。

按用户要求，已删除完整串行测试及两轮串行短测的 3 个输出目录、8 份权重、
对应日志和临时监督脚本，共 350824306 bytes；仅保留文档结果摘要。
历史平铺 28 份权重、缓存、验证视频和非串行崩溃证据保留；step-11200
SHA256 仍为 4a085810854b550cdd912e3edebfbe74f4d9229dc67abd623fe84dc002743161。
核对清理前后均无训练进程，tmux 诊断会话已结束，没有重启任何 GPU 作业。

一次完整预算未复现不证明串行反向根治故障；首次非法写入未定位，根因仍未
确认。历史正式训练的终态也不能由本次独立 weights-only 诊断补证。
详见 [原生崩溃诊断](docs/NATIVE_STABILITY_20261014.md)。
<!-- RESTORED_HISTORY_END -->

<a id="update-20261023"></a>

## 二、2026-10-23：阶段更新与当前资料补全

下面继续保留357b09f阶段补充的结果、参数和操作说明。该阶段曾把历史移到归档、部分原文页改成导航；这一文档组织做法已由第三节的用户更正取代，实验事实和既有检查结果保留。

<!-- RETAINED_UPDATE_BEGIN -->
### 当前项目状态

核对日期：2026-10-23（UTC+09:00）。最新正式实验已结束；此次仅更新项目Markdown。实验入口统一在 [experiments/README.md](experiments/README.md)。

#### 1. 当前可确认的成果

| 实验 | 已完成范围 | 可读证据 |
| --- | --- | --- |
| Pilot8 Base | 8栋×81帧，每栋4项可视化 | [结果](outputs/vace/test/) |
| Pilot8 Composite | 8栋CPU硬边界回贴，每栋3项结果 | [结果](outputs/vace/composite_check/) |
| P04/P06 LoRA17 | 300真实更新、两例各150、6次17帧比较、100/300权重 | [run](outputs/vace/lora_roof_smoke/run_20261020_165237_150325_2048720/) |
| Pilot8 LoRA81 | 八例81帧、32项成果，复用Base | [run](outputs/vace/lora300_eval81/run_20261020_194300_145372/) |
| Complex12 Rev1 | C/U/S各300，60轨迹、4860PNG、12视频、12总览 | [完成状态](outputs/ComplexSingleOrbit_Rev1_Pilot_20261022/progress.json) |

Complex12最终状态为complete。源记录完成时刻2026-10-23T02:59:37+08:00，换算本页时区为2026-10-23T03:59:37+09:00。旧记录自身时区不改写。

#### 2. Complex12实验身份

物理run：outputs/ComplexSingleOrbit_Rev1_Pilot_20261022/。数据来自data/buildingfiller_import/Complex12/BuildingFiller_Complex12_512_Rev1_20261022/，使用Rev1的accepted索引。

- 训练：CS01/03/05/06/07/09/10/11；验证：CS02/04/08/12，未参与训练或检查点选择。
- B为Base；L为旧P04/P06 LoRA-300迁移参考；C/U/S从共同新初始化出发，各自新AdamW、相同采样和随机序列。
- C为原目标；U均匀增加缺损区权重；S增加结构带权重。U/S总额外权重相等。
- 17帧连续窗口训练，81帧整轨推理，512×512、16FPS、30步、CFG5；训练seed20261022，推理seed20261018，LoRA scale1。
- 只比较各组step300；100/200仅恢复点。预测没有回贴、调色或强制闭环。
- 冻结配置ID：e42ab127a6625942953bea743e31b89fc8e00dcd6604fdd3ac61b5935561c495。

完整参数、样本源号、损失映射、实际耗时/显存见 [Complex12实验页](experiments/vace/complex12_rev1/README.md)。

#### 3. 效果结论与证据边界

C相对Base的验证缺损MAE降低64.21%。S相对U验证MAE降低1.80%、结构Sobel降低0.19%；验证均值优势集中于CS12，其余三栋S的MAE略差于U。S的保留区变化及全图闭环误差高于C/U。

CS02、CS03、CS10的大黑槽/黑洞持续改善；CS04、CS06、CS09、CS12仍有细折线、弯曲交界或浅凹。CS07的S在B010附近有小灰斑。已查看60条轨迹的连续帧图及关键帧，不等同于实时高频闪烁测量或三维几何验证。

旧P04/P06是LoRA17的训练题；Complex12 CS06为源1251，不能混同旧P06源1848。CS09–CS12只代表复杂屋顶背景下局部屋面/邻墙补全，跨屋脊挑战主要看CS06。

#### 4. 环境、目录与历史T2V

当前项目与Python均位于本服务器；Python3.10.13、torch2.11.0+cu128、DiffSynth2.1.5、PEFT0.17.1。GPU为单张A6000；历史显存读数不能当作当前空闲容量。

Complex12源码已归到experiments/vace/complex12_rev1/scripts/，原run/scripts为兼容链接，原resume.sh不变。源码位置与结果位置分离，新想法按 [目录约定](docs/PROJECT_STRUCTURE.md) 创建。

历史A6000 T2V保留28份LoRA、29个比较视频和1992个.pth样本缓存；A100目录保留39份LoRA。A6000最后loss step11562、最高权重11200与推算预算末步13536不同，完整8 epochs自然结束仍待证实。logger物理batch步号不能当optimizer更新数。

#### 5. 未解决事项及停止边界

- B/CS04保存完整81帧后原生退出−6；独立验证有效输出后复用，没有重跑。原生退出根因未确认。
- 挂载恢复后C从保存的40次更新续至300；后续U/S各300。已成功轨迹未重复运行。
- 单轨T2V尚缺源OBJ级身份映射的完整泄漏核查与可靠训练终态证据。
- 扫描器src/及两份GUI工作流源文档缺失，不能据此安装运行历史GUI。
- 文档中记载的Claude默认模型字符串不是服务可用性证明；本次没有调用外部worker/API。

当前没有待自动继续的GPU任务。报告提出的配对重复实验只是建议。完整结果从 [00_看结果](outputs/ComplexSingleOrbit_Rev1_Pilot_20261022/00_看结果/README_从这里开始.md) 打开；过程快照见 [历史索引](docs/DOCUMENTATION_INDEX.md)。
<!-- RETAINED_UPDATE_END -->

<a id="history-repair"></a>

## 三、2026-10-23：恢复原文件内的持续历史

用户明确要求长期MD保留“从以前到现在”的更新。先前仅保留当前摘要、让读者另找归档的处理不符合要求；本次将上述旧正文直接恢复到原文件，同时保留后续更新，没有回退实验结果或把完成状态改回执行中。

以后在本文件按日期追加阶段进展、失败与修复、参数变更、结论和核查范围。需要纠正旧结论时，说明原判断、后来的证据及新结论；原记录继续保留。可以在顶部维护简短状态和阅读索引，不能用摘要或归档链接替换历史正文。

本次修复依据原文件备份及Git记录恢复，没有补造新的运行经历；原数据、权重、视频、指标、冻结报告和现有非Markdown工作保持原样。此次检查与恢复范围见[文档索引](docs/DOCUMENTATION_INDEX.md)中的后续补记。

## 四、2026-10-24：Complex12独立报告与持续交付约定

本次补记采用UTC+09:00；原运行日志的UTC+08:00时间保持不变。本次工作是完成实验的文档交付，未启动GPU任务。新增 [upload目录](upload/README.md) 和 [Complex12详细报告](upload/20261022_Complex12_Rev1_十二栋单轨实验报告.md)，便于用户单独复制MD回家分析。

报告依据冻结run中的resolved_config、共同采样表、C/U/S更新记录、worker完成/退出记录、指标CSV和既有连续帧视觉记录编写。正式完成状态仍为C/U/S各300更新、60条81帧、4860张预测、12视频及12总览。报告保留B/CS04退出−6但完整成果经独立检查后复用、C从update40恢复optimizer/RNG等经过，没有把异常改写为从未发生。

当前结论：C对B的缺口MAE在训练/验证集分别降低71.00%/64.21%；S对U的验证缺口MAE仅降低1.80%，四栋中仅CS12更好。S相对C的验证保留区MAE和全图闭环误差分别增加4.35%/20.38%，复杂屋脊仍有亮缝、折面和凹槽。建议先验证C/U/S多训练种子下的配对收益是否稳定；该建议未执行。

用户新增持续要求：每组实验结束都在upload保留对应的完整报告，正文自包含关键参数、结果、局限和下一步；同日同步根长期MD、对应实验README与报告索引。保留所有既有历史，旧数据、权重、媒体和冻结run报告维持原位。报告整理不等于自动传输或下一轮实验授权。


## 2026-10-26：OriginalHard 已有权重评测启动快照

核查时间：2026-10-26T01:27:12+09:00。用户已授权对上传的 OriginalHard 使用已有 B/L/C/U/S 跑完整 81 帧，并要求“旧的相同的就不用跑了”。本轮新增训练更新为 **0**，C/U/S 沿用已完成的 Rev1 step300，L 为旧 P04/P06 参考。

新实验入口：[OriginalHard](experiments/vace/complex12_originalhard_eval/README.md)。真实 run：`/mnt/windowsE/chuanjun/building_wan/outputs/vace/complex12_originalhard_eval/run_20261026_012033`。源码位于 experiments/vace/complex12_originalhard_eval/scripts，旧大结果保持原位。

逐帧核对后，CS01–CS08 的 GT/Damaged/Exact/结构图及相机记录与旧版一致，五方法共40条预测已解码后链接复用；CS09–CS12 的缺损不同，新增20条推理。最终展示60条，其中40条复用、20条新推理。新缺损中的CS09/10/11来自旧训练身份，CS12来自旧验证身份；应单独汇总四栋及3/1子组，不能称为十二栋新建筑泛化。

此刻 runner PID 1678408 实际存活，tmux 为 complex12_rev1:originalhard81；状态 running，当前 infer_L_CS09，有效轨迹 41/60（复用40，新推理1）。这是一条运行中快照，不代表全部推理、媒体或视觉结论已经完成。

推理保持512×512/81帧/16FPS/30步/CFG5/seed20261018及Rev1实际文本；GT仅用于后评测，原始预测不回贴。最终按原约定交付12视频、12总览、分身份及困难样本指标、中文结论和upload独立报告。下一步是继续现有队列和复核，不追加训练。


### 2026-10-26 后续交接：用户要求启动后即可结束对话

用户进一步明确“跑起来就行，就可以结束这轮对话了，不需要实时监控，我在vscode的tmux能看到过程即可”。01:31:02（UTC+09:00）核查时，runner与C/CS09 worker实际存活，C去噪17/30，40条复用、2条新增已完成；VS Code所连tmux客户端位于complex12_rev1的originalhard81窗口。

按此要求结束本轮主动跟踪，保留既有队列继续运行。队列包含剩余推理、统一指标、12栋视频/总览和复核图；计算结束状态为awaiting_visual_review。详细视觉分析、最终中文结论、upload实验报告和完成文档需下次接手核对真实产物后补齐，不能将本轮“已启动”写成“实验已全部完成”。长期文档的历史正文和旧实验产物继续保留。


## 2026-10-26：OriginalHard 完整轨迹复核与结果收尾

复核记录：2026-10-26T02:41:09+09:00（UTC+09:00）。本次接手已核对20个新worker退出0、40条复用，共60条81帧/4860 PNG，12视频/12总览和统一指标均完成；用户要求审查困难结构效果后，逐段查看CS09–CS12全部81帧联系图及六个原尺寸问题帧。新增训练更新0，本次没有启动新GPU任务。此前启动/待复核快照按原文保留。

run：`/mnt/windowsE/chuanjun/building_wan/outputs/vace/complex12_originalhard_eval/run_20261026_012033`。直接观看入口：[00_看结果](outputs/vace/complex12_originalhard_eval/run_20261026_012033/00_看结果/README_从这里开始.md)；完整可携带报告：[OriginalHard已有权重评测](upload/20261026_Complex12_OriginalHard_已有权重评测报告.md)。最终结论在run的00_看结果/最终实验结论.md，核查后progress更新为complete。完成指执行和报告完整，不表示补全质量全部合格。

CS09、CS11明显改善；CS10连接面/折线仍错，C的MAE最低；CS12错误折面及首尾形状差异明显，旧L反而最好。新缺损四栋B/L/C/U/S的MAE为0.069922/0.036078/0.036619/0.036443/0.035899。S比B降低48.66%，但比C/U只低1.97%/1.49%；S相对C保留区变化增加4.17%、全图闭环误差增加13.78%，不能称S稳定胜出。CS12的S首尾缺损区误差0.126121，L为0.029390。

身份不能混淆：CS01–CS08输入不变且五方法复用；CS09/10/11是旧训练建筑新缺损，仅CS12为旧验证建筑新缺损。不是12栋全新建筑泛化，不能用完整8/4均值掩盖困难4栋。视觉结论来自连续静态帧与原尺寸图，视频有程序全帧解码证据，本次未做实时播放或量化高频闪烁。

下一步建议调整为固定VACE骨干、普通C和总预算的数据覆盖对照，加入更丰富跨屋脊/屋墙连接缺损并按家族隔离验证，再判断几何/Depth是否必要。这一建议来自新的CS10/12失败证据，未启动；旧Rev1的多种子S建议和历史依据保留。

收尾实测：2026-10-26T02:42:50+09:00，progress=complete；60轨迹/4860 PNG与12总览可读，12视频完整解码证据齐全。298项CPU回归（57.214秒）及9项专用检查通过；详细检查见文档索引和run/logs/final_review.done.json。


## 2026-10-26：C_mix 固定配比实验启动与后续交接

核查时间：2026-10-26T22:26:41+08:00。用户授权一组普通C_mix的300更新、16条81帧推理及配对指标/视频，随后要求确认训练启动即可结束对话，视觉复核和最终报告等待另行通知。当前running，已记录33/300次真实优化器更新、推理0/16；这不是完成或效果确认。

实际run：`/mnt/windowsE/chuanjun/building_wan/outputs/vace/complex12_cmix_pilot/run_20261026_222008`。研究入口：[C_mix](experiments/vace/complex12_cmix_pilot/README.md)，观看入口：[00_看结果](outputs/vace/complex12_cmix_pilot/run_20261026_222008/00_看结果/README_从这里开始.md)。tmux为`complex12_rev1:cmix_pilot`（%2），runner PID 2179781，GDB worker PID 2179822。

从旧Rev1 shared_initialization及原始VACE开始，新AdamW，普通C损失与参数不变。旧300行顺序/窗口/随机字段保留；局部Random(20261026)按CS09/10/11抽20/14/16行Hard，合计Rev1 250/Hard 50。验证CS02/04/08/12任何版本不入训练。条件缓存按版本、建筑、窗口隔离；合法零scheduler权重计数、不重抽不补步。两版改变了缺损位置与连接线索，不是同一缺口单纯扩大。

队列依次完成300更新、Hard CS09–12和Rev1 CS01–12共16条/1296原始PNG、原指标与分组/逐栋新旧变化、Rev1退步项、32视频和六张固定问题帧。旧C结果复用；Hard CS01–08展示复用新Rev1；不重跑B/L/U/S。计算结束标记awaiting_visual_review，用户通知后再进行完整视觉复核、run中文报告和同内容upload报告。当前不提前写效果结论。

本轮9项专用CPU检查通过（0.658秒），298项项目CPU回归通过（50.410秒），规定shell语法/Python编译与差异检查通过；没有额外GPU试跑。旧run、数据、第三方引擎及19项无关工作区修改保持原样。冻结配置ID：`386098b4cc71d7a446b5d2dcba635e8a8a3142dcaa3e6bd8629299406fb0c545`。


## 2026-10-27：C_mix 完整评测、视觉复核与收尾

日期采用UTC+09:00；计算和媒体于服务器2026-10-26T23:43:28+08:00完成，即2026-10-27T00:43:28+09:00。用户随后通知执行此前约定的复核、报告和文档/Git收尾。此前启动快照不删除。

- 实际run：`/mnt/windowsE/chuanjun/building_wan/outputs/vace/complex12_cmix_pilot/run_20261026_222008`；[直接看结果](outputs/vace/complex12_cmix_pilot/run_20261026_222008/00_看结果/README_从这里开始.md)，[完整可携带报告](upload/20261027_Complex12_C_mix_混合训练与配对评测报告.md)。
- C_mix从Rev1共同初始化和新AdamW执行300真实更新，不是旧C继续训练；实际250次Rev1/50次Hard（CS09/10/11为20/14/16）。合法scheduler零权重为Rev1 4步/Hard 3步，照常计数、没有补步；优化器最终step范围[300,300]。
- 原训练8栋CS01/03/05/06/07/09/10/11、验证4栋CS02/04/08/12不变。512²、17帧训练、rank/alpha16、lr5e-5、普通C；300表的原随机/窗口字段不改。变化为缺损位置和连接线索，不是同洞扩大。
- 新推理先Hard CS09–12四条，再Rev1十二条；16×81帧、1296PNG、32视频均完成。每条seed20261018、30步、CFG5及旧VAE/提示词；GT不进推理，不回贴、不替换080。旧C两版结果直接复用，没有重跑B/L/C/U/S。
- 本run的1训练+16推理+1CPU评估worker全部一次START、EXIT 0，无实际恢复/重试。训练约27分06秒，计算至媒体墙钟80分26秒；训练/推理PyTorch峰值allocated约15.139/17.143GiB。
- 16条完整000–080序列按连续联系图复核；6固定问题图和新增Rev1 CS10/B020按原512内容复核。没有冒充实时视频播放、定量闪烁或三维验证。

结论：Hard训练三栋平均MAE−1.62%，但Sobel+7.86%；CS10 MAE−4.07%/Sobel+12.06%，关键连接仍错。Hard验证CS12 MAE−1.78%、MSE+9.65%、Sobel+5.01%，缺口闭环0.115625→0.117000，形状/首尾未解决。Rev1训练MAE+158.67%，验证均值−0.68%，但10/12栋MAE退步；CS10新增百叶/栅格状条纹，MAE+1199.74%。16输入known MAE及全图首尾误差均上升。

当前决策：保留旧C参考，不将本轮C_mix视为替代权重。结论只适用于250/50固定预算；Hard CS10仅14行且2行零时间权重，Hard唯一未训练建筑CS12只是开发验证。下一项仅建议当前入口纯Rev1对照复现（300更新、Rev1 CS10与Hard CS10/12三条推理），先定位显著退化来源；尚未授权/执行下一轮。所有历史数据、权重、原始预测保持不变。


## 2026-10-27：Orbit17线性/循环配对实验已启动

核查时间：2026-10-27T20:16:32+09:00（UTC+09:00）。用户已授权线性组和循环组都执行，并要求确认实际开始训练后结束本轮对话。当前status=running，C_lin17已完成22/600次真实AdamW更新，C_ring17为0/600（待串行执行），推理0/40；日志已有有限loss及参数实际变化。这是启动快照，不是两组完成或效果验证。

固定run：`/mnt/windowsE/chuanjun/building_wan/outputs/vace/complex12_orbit17_pair/run_20261027_200842`。专用源码与说明：[Orbit17实验](experiments/vace/complex12_orbit17_pair/README.md)；[结果导航](outputs/vace/complex12_orbit17_pair/run_20261027_200842/00_看结果/README_从这里开始.md)。tmux=`complex12_rev1:orbit17_pair`（%3），runner PID 2469437，GDB worker PID 2469491；源码提交`1a82bd5`。

两组均只用Rev1训练8栋CS01/03/05/06/07/09/10/11，验证CS02/04/08/12不变；Hard只评测。普通C、原VACE骨干、共享初始LoRA与各自新AdamW，rank/alpha16、lr5e-5、17帧、BF16骨干/FP32 LoRA等原设置保持。线性窗起点0–63，循环窗起点0–79按模80取17帧，080不入训练；跨界先同步取完整17帧再编码。原线性前300行保留；后300行用局部Random(20261027)，循环起点另用Random(20261028)，其他随机字段逐行配对。合法零scheduler权重照常计数，不重抽补步。

已启动的同一队列包含：线性0→300、四条81帧复现探针与CPU提前核对、完整状态恢复线性301→600、循环0→600、循环300四条探针、两组600各16条正式预测，最后统一CPU评测和视频。提前核对保留复现证据，不因主观效果选择性取消循环组。总预算两组各600、40条81帧、3240原PNG；计划48视频、7固定问题图和4张七行探针总览。旧C两版原预测直接复用，无新增B/L/U/S/C_mix推理。

本次实测14项Orbit17专项CPU检查通过（1.466秒），298项项目CPU回归通过（50.203秒），规定shell语法/Python编译及git diff --check通过；没有额外GPU试跑。准备时覆盖图首次因服务器缺少matplotlib失败，随后改用现有Pillow生成并通过；没有安装依赖，未重复GPU任务，prepare.log保留实际错误与后续成功。旧run、原始数据、模型、第三方源码及19项既有无关工作区修改保持原样且不纳入本轮提交。

每10更新保存完整恢复状态，300/600保存LoRA，300另保留恢复点；两组状态独立。缓存按版本/建筑/完整有序frame_ids/编码设置区分，放在新run。自动队列完成后标记awaiting_visual_review；之后按用户通知进行完整轨迹视觉复核、run最终中文报告及upload同内容便携报告。此时不能预先宣称循环优于线性，也不能把视频解码当人工播放或三维验证。新预算、配比或模块不自动展开。
