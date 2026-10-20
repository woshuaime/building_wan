# Project Context

更新时间：2026-10-20 20:09:57 +09:00。LoRA-300八例81帧评估全部完成，八个worker和两个批次均退出0；32项成果通过验收，GPU已空闲，本轮停止。

持续工作约定：用户每次实际启动GPU训练或推理后，主动核实进度/最终状态、更新项目文档，并提交推送本轮相关代码、配置和文档；不等再次提醒。数据、模型、成果媒体与日志留在原目录，不入Git。该规则已写入AGENTS.md和SESSION_HANDOFF.md。

## 当前：LoRA-300八例81帧评估完成

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

## 历史：首次81帧评估在CPU制图阶段停止

- 首次P04真实30/30步与VAE解码完成，模型加载至返回175.2秒，峰值17.141GiB allocated /25.832GiB reserved。随后 `comparison_panel` 接收到PIL对象而非NumPy数组，抛出 `TypeError: a bytes-like object is required, not 'Image'`，worker和批次均退出1，未启动P06。
- 原始MP4曾在临时目录编码完成，但失败时临时目录被清理；当时未保留预测，不能用首次推理评价视觉效果。修复统一转换制图像素类型，并先将原始MP4保存到最终样本目录；后续CPU制图失败可用 `--render-only` 完成，不需重新推理。用户回复“直接修复，然后再重跑”后，才执行上方已完成批次。
- 修复后298项CPU测试通过（53.124秒），包括81帧PIL输出制图、全部媒体解码、CPU制图失败后01保留且补图不改01。语法/编译/diff通过；没有用CPU结果冒充正式预测。

- 首次失败的实际tmux为 `vace_p01_17f:4.0`，窗口名 `lora300_eval81`，pane `%4`。18:24:35 +09:00启动快照不能覆盖其最终退出1；同一窗口随后承载了获授权的成功批次。
- 唯一权重：`outputs/vace/lora_roof_smoke/run_20261020_165237_150325_2048720/training/lora_update0300.safetensors`。该run已完成300更新，用户确认17帧有效；本轮只加载vace分支，scale1，不训练、不叠加其他LoRA。
- 新结果只写 `outputs/vace/lora300_eval81/`。顺序P04→P06，完成后先查看全环绕效果；仅当仍整体明显优于Base才执行P01/P02/P03/P05/P07/P08。每例独立进程重载原模型，异常即停，不自动重试。
- 八例Base均复用 `outputs/vace/test/01–08/01_补全结果.mp4`。执行入口和冻结配置确认81帧B000–B080、512、16FPS、30步、CFG5、seed20261018及正负文本一致；旧入口隐式cfg_merge/denoising_strength/VAE tile参数与现用显式参数一致。媒体自身不能反推seed，复用依据为既有执行证据和配置。
- Damaged/Exact用原Pilot8包，填灰127与二值Mask规则不变，全黑帧保留；GT像素只在推理返回后读取用于比较，不回贴或调色。每例四文件：原始LoRA视频、四栏81帧视频、四行五列总览、五栏Mask图。
- CPU准备已核对唯一权重metadata/结构、八例81帧输入与Base完整解码；297项CPU测试、规定语法/编译/diff检查通过。未做重复大文件SHA、依赖更新或模型改造。

## 已完成：P04/P06 VACE条件LoRA首轮

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

## 最新结果：仅替换缺损区域的CPU合成

- `scripts/composite_vace_results.py` 在现有tmux的 `vace_composite` 窗口实际执行，八例全部SUCCESS，耗时46.0秒、退出0、GPU推理0次。输出 `outputs/vace/composite_check/01–08/`，每例严格三文件：`01_仅缺口补全.mp4`、`02_四方视频对比.mp4`、`03_关键帧对比.png`。
- 使用已验收Pilot8包的真实源元数据、相机与512帧序列，包括身份已匹配的P01副本；复用原推理 `adapt_images` 从Exact生成二值ModelMask，不扩张、收缩、反转或羽化。白区逐像素取原预测MP4解码值，黑区取Damaged，GT只用于对照和关键帧选择；全黑帧原样保留Damaged。
- 两个视频均81帧/16FPS，分别512×512、2048×544，采用CPU H.264 High/yuv420p/BT.709，完整解码与RGB/帧序检查通过。总览2560×2176、四行五列，保留B000/B080，并优先展示最大Mask、可见Mask内RGB误差最大的视角及全黑视角；选择仅为诊断展示，不作为独立量化评估。
- 编码前黑区与Damaged完全一致，白区与Raw Prediction完全一致；MP4允许正常有损量化。P01颜色和主体保留改善，但有矩形拼接细缝；P04屋顶凹槽、P06深色洞口仍在，说明简单回贴不能恢复缺口内部结构。其他例也存在不同程度补丁边界，不能宣称三维一致性已解决。
- 原 `test/` 的32个文件SHA/大小/mtime均未变，源ZIP大小/mtime未变，自有临时目录已清理，无新日志/JSON/输入副本。272项CPU测试及规定语法/编译/diff检查通过；完成后停止，没有启动后续Mask或模型实验。

## 已完成：八例结果升级为 Before / Prediction / GT

- 已完成八例CPU重排及后续播放偏色修复，终端P01–P08均SUCCESS、退出0、GPU推理次数0。八个三栏视频已全部替换为H.264 High/yuv420p/BT.709；独立比对修复前后32文件，只有八个02改变，八个原始预测与16张PNG全SHA不变，各目录仍恰好四文件，自有临时目录清理通过。完整266项CPU回归通过，哈希证据见EXPERIMENTS.md。
- 独立入口 `scripts/reorganize_vace_results.py`，不导入Torch/DiffSynth、不读取模型权重；默认只读验收，显式 `--apply` 才安全替换已有对比图像。缺少原始预测必须停止，没有模型回退。
- 输出仍为 `outputs/vace/test/01–08/`，每例恰好四项：原始 `01_补全结果.mp4`、`02_补前_补后_GT_三栏对比.mp4`、`03_补前_补后_GT_三行总览.png`、`04_缺损_Mask_预测_GT_四栏验证.png`。
- 原始MP4不移动、不重新编码，前后SHA256、inode、大小、mtime一致；四栏图核对本例B020源身份后仅改名，内容不变。三栏81帧/16FPS/1536×544，三行总览2560×1632，五列B000/B020/B040/B060/B080；每个主体保持512×512，不美化或GT回贴。
- 原三栏视频使用libx264rgb/gbrp（High 4:4:4 Predictive），FFmpeg像素检查通过，但用户播放器出现紫绿偏色；原先验收没有覆盖播放器兼容性。现改为CPU libx264/High/yuv420p/BT.709有限范围、CRF12与faststart。校验同时要求正确编码/像素格式/色彩标签及全部81帧各栏与原图误差受控（每通道MAE与均值偏差均≤3，99%像素绝对误差≤12/255），PNG仍逐像素相等；只允许常规编码量化，不进行调色或美化。
- 已有新版布局若使用RGB 4:4:4，`--apply`仅重新制作02三栏视频，原始预测和两张PNG的SHA/inode/mtime保持不变；已有兼容视频只读核验。交接包CRC、载荷哈希、图像、帧号和相机仍完整验收，各例原始黑Mask不改。
- 同分区、test目录以外的独占临时目录先生成并验收，再逐例替换三项衍生成果；失败回滚旧文件，回滚异常保留明确恢复路径。不得重新调用旧GPU入口处理新版文件名。

## 历史阶段：校验失败排错，显式续跑 P04–P08（后续已成功）

- tmux实际错误是P04的 `Reused component SHA256 mismatch: text_encoder`，位于sample_preflight，尚未加载该例模型；不是CUDA OOM。P02/P03各30步与81帧解码/四文件发布已成功，实测峰值17.14GiB allocated /25.73GiB reserved；P01已有终端证据同为完整81帧成功。
- 文本编码器原件11,361,845,432bytes，连续三次独立完整 `sha256sum` 均为批准值 `d92de679881d38af9c89eff7bb1b6d6c9d96cb2b69831e4027e9ecabdd38eb23`，inode/size/mtime/ctime稳定。此前失败没有保存实际hash，暂不能复现或判断底层读取、内存或文件系统原因；不宣称硬件/磁盘问题已修复，也不认定权重持续损坏。
- `verify_file_sha256` 仍单次完整读取、任一不符即停止；新增expected/actual/path/size及读取前后stat，检测文件原地变化/替换，不自动重试或绕过校验，不修改批准hash或权重。
- 新显式命令为 `bash /mnt/windowsE/chuanjun/building_wan/scripts/run_vace_building_test.sh --resume --tmux`。先完整验收已有连续成功前缀的四文件/81帧/逐像素源身份，父进程和gate再比对四项hash，确认不变后只排队P04→P08。P01/P02/P03只读复用；普通首次启动依然拒绝已有02–08，显式恢复也拒绝不完整、错号、非连续、额外或软链接结果，不删除/覆盖任何既有文件。
- 已有 `test/01–03` 验收通过：P01与保留原件四SHA相同；P02对应source763，P03对应source871，总览五个Damaged与四栏Damaged/Mask/GT均逐像素匹配。`04–08`不存在，无失败半成品需清理。配置、文本、模型身份、81帧和四文件约束保持不变。
- 恢复入口实际 `--resume --dry-run` 通过（32.1s），列出的唯一GPU队列是P04–P08，权重全SHA、当前环境与七例CPU接口通过。完整250项CPU回归通过（本轮新增44项），语法/编译/diff检查通过；未执行resume GPU分支，原件/输入/冻结JSON未变，未提交Git。

## 此前阶段：复用 P01，准备 P02–P08

- 已执行 `scripts/run_vace_building_test.sh --prepare`，全部七例预检通过后仅复制 P01；`test/` 当前仅有 `01/`，且恰好四个有效文件，`02–08` 尚未创建。原件为 `outputs/vace/P01_465_81f_20261019_171510_1691517/`。两个视频均 81 帧/16FPS，原始预测 512×512；总览及四栏的 Damaged/Mask/GT 与原始源465数据逐像素对应。
- 新交接包为 `data/buildingfiller_import/P02_P08_20261020/BuildingFiller_Pilot8_AllRes_v1.zip`，SHA256 `7f8eff9ddb3c05e43fe5a8623258f9dceb6fa61d547b244c4f2d441099b4fcf4`。包含全部八例和三种分辨率；CRC、6077 项载荷哈希、5832 张 PNG 实际解码、1944 个帧/相机绑定及几何绑定检查通过。包内 P01 仅核查身份，不导入或补全。
- 固定映射为 `02/P02/763`、`03/P03/871`、`04/P04/1227`、`05/P05/1259`、`06/P06/1848`、`07/P07/2297`、`08/P08/939`。P05/P07 的 family ID 不等于 source ID；以真实元数据为准。各样本均有 81 张 512×512 GT/Damaged/Exact 和配套相机。
- 有效人工验收均为 accepted，几何/渲染证据校验通过；包内历史快照的 pending/not_run 不覆盖 hash-bound 验收。`generation_evidence_verified=true`，但 `training_ready=false`、`generation_verified=false` 仍保持原义；只用于本次视觉功能验证，不能冒充训练发布或模型补全成功。
- 各自原始黑 Exact 保留：P02 为 023–049；P03 为 000–016、071–080；P04–P08 没有黑帧。不能套用 P01 的 046–072。Exact 二值白区生成、RGB 三通道、Damaged 白区填灰127，均在内存转换；模型条件无 GT RGB，GT B020 只在真实预测返回后读取。
- `scripts/vace_building_test.py` 与 P01 入口共用完整 81 帧执行核心，固定原始 VACE-1.3B、无 LoRA、B000–B080、512×512、16FPS、seed20261018、30步、CFG5、full BF16/tiled VAE，不分段或降规格。正负 prompt 逐字复用 P01 冻结配置。P01 旧实验没有 manifest，历史 seed/权重/prompt 不能仅从媒体独立反推；用户已确认成功，保留的成功执行代码和冻结配置未见参数差异，不能把这一限制伪装成独立运行溯源证据。
- 正式 VACE/T5/VAE/Tokenizer 完整哈希通过；A6000 探测时空闲约 47.13GiB、无其他 GPU 进程，项目盘约 358.8GiB。启动时全七例统一门禁，随后每例重新检查；父进程不初始化 CUDA，各样本独立子进程串行释放模型，不安装/下载/修改第三方核心。
- 用户启动命令为 `bash /mnt/windowsE/chuanjun/building_wan/scripts/run_vace_building_test.sh --tmux`，只执行 P02→P08。终端显示真实30步、当前例数、已完成数量、实际耗时和按已完成样本估算的批量 ETA。不写日志/JSON/输入或预测 PNG；自有系统临时目录编码并验收后只发布四文件。任何失败立即停止，已有结果不覆盖；部分批量成功后再次启动会拒绝既有 `02–08`，不得自动重算或跳过。

## 历史准备：P01 81 帧极简视觉验证（已由后续真实结果更新）

- 新入口 `scripts/run_vace_p01_465_81f.sh --tmux`：前台可见或复用当前 tmux，真实 30 步及 VAE 进度，不写日志或 JSON。`--dry-run` 已实际通过完整模型 SHA256、环境和 81 帧 CPU loader 检查，没有创建实验输出。
- 复用冻结 17 帧 JSON 的 prompt/seed20261018/steps30/CFG5/BF16/512/16FPS 与同一 ModelMask 转换，仅在内存改为 source000–080/81 帧和四文件输出策略；原 JSON 字节不改。81 帧是一次整序列调用，不分段、不回退 offload、不降低规格。B080 正常进入模型，预测不复制 B000。
- 原 ZIP/81 组 Damaged+Exact/相机一致，046–072 共 27 张全黑标签原样保留。GT RGB 在模型返回后才单独读取 B020，唯一用于四栏对照，不进条件。
- VACE/T5/VAE/Tokenizer 完整官方固定哈希通过；A6000 空闲约 47.13 GiB、无其他 GPU 任务，项目盘约 358.8 GiB 可用。81 的 latent T=21、tokens21504，当前 SDPA 高效后端启用、长度合法，真实峰值尚未测，OOM 即停止而不改配置。
- 成功目录 `outputs/vace/P01_465_81f_<UTC时间>_<PID>/` 仅含 `01_补全结果.mp4`、`02_补前补后对比.mp4`、`03_补前补后总览.png`、`04_四栏验证.png`。不保存定量指标、输入副本、预测 PNG、运行报告；必要视频/图像先在本次独占系统 `/tmp` 目录编码验收，正常退出/失败清理，仅发布四个文件。原生 worker 失败由父进程接管；不是模型质量的正式定量验收。

## 1. 目标与边界

当前任务是 Damaged + 独立适配的 ModelMask + 文本 → 原始 VACE 补全视频，GT 只用于评估。BuildingFiller 在个人 Windows 电脑独立制作数据，经实验室 Windows 主机转入 Linux 项目；Windows `D:\P01_465` 不能当作服务器路径。第一例为源 465、A1 Medium、删除表面积约 5.679%，不用历史 T2V LoRA。

历史 Wan2.1-T2V-1.3B LoRA 学习灰色、无纹理建筑 mesh 的体块、完整环绕和时序稳定性；单轨统一 prompt 的划分、缓存及权重保持独立。逐栋描述和 Cap3D 必须使用独立划分检查、缓存及输出。

BuildingFiller 已交接完整/残缺/移除 mesh 三元组与对齐 RGB/Exact 的 P01 数据；旧扫描器 GUI 源码仍缺失。当前 LoRA 对比及人工数据验收都没有证明 VACE 补全或三维重建能力。

## 2026-10-20 17 帧准备历史（输出已由用户删除）

- 正式文件位于 `models/Wan2.1-VACE-1.3B/574e6a744642ce3bee319afc31496b88bde8aac4/diffusion_pytorch_model.safetensors`，7,146,067,848 bytes，完整 SHA256 匹配官方 `c46a6f5f7d32c453c3983bbc59761ea41cd02ad584fb55d1a7ee2b76145847a2`。1264/439 总/VACE 张量和双分支注册结构通过；T5/VAE/Tokenizer 完整哈希同样通过。
- 阶段 A 证据在 `outputs/vace_model_audit/P01_465_20261020_20261019T161614Z/`。A6000 CUDA/BF16 可用，空闲约 47.1 GiB，NVML 没有其他计算/图形进程；没有 `nvidia-smi` 命令不等于没有 GPU。tmux 3.2a、ffmpeg/ffprobe 已有，项目盘空闲约 359 GiB。
- 实际 CPU 准备在 `outputs/vace/P01_465_17f_baseline_cpu_prepare_20261020_20261019T162331/`，源 005–021、17 帧 RGB Mask/条件帧，当前 DiffSynth CPU 加载器合同通过；GT 仅另存评估，未进模型。
- `scripts/run_vace_p01_465_17f.sh` 由用户终端启动后只运行一次真实 17 帧模型调用。父进程实时转发无缓冲 stdout/stderr，保存 `run.log`/`worker.log`；真实 scheduler tqdm、VAE tile 进度、阶段状态可见，不改 vendor。失败保留真实退出码并停止，不自动重试、降规格或切换 offload。
- GPU 子进程正常退出且完成标记有效后，父进程才运行独立 `scripts/vace_p01_results.py`：验证原始 PNG/MP4，生成 B020 两栏、四栏、005/010/015/020 两行总图、同步对比视频及诊断指标。进程成功不等于补全质量或三维质量达标；没有预测时不伪造图。
- 本轮未执行 `--run`，没有 GPU 原始输出或效果结论；下一步是用户执行终端命令。冻结 JSON 的旧状态字段保留其冻结时含义，实际模型路径/哈希以运行 manifest 为准，参数字节未改。

## 2026-10-19 权重核查与执行器历史

- 官方正式 VACE 文件为 7,146,067,848 bytes，SHA256 `c46a6f5f7d32c453c3983bbc59761ea41cd02ad584fb55d1a7ee2b76145847a2`，HF revision `574e6a744642ce3bee319afc31496b88bde8aac4`；远程约 130 KB 文件头与本地 DiT/VACE 双分支注册结构匹配。
- 扩展搜索找到 `/mnt/windowsE/szq/HuggingFaceWeights/Wan/Wan2.1-1.3B-VACE/diffusion_pytorch_model.safetensors`，但大小少 32 bytes、SHA256 为 `fcd12e2a8fab331ba0ab28dca371d90036a868287767f35677ff3f8e74db0969`。即使用官方文件头重建哈希仍不匹配，张量载荷不等同正式文件，禁止用作原始基线；其他训练/merged 权重也不使用。
- 现有 T5/VAE 全文件 SHA256 与官方 DiffSynth ModelScope 转换组件一致；Tokenizer 哈希固定，可全部复用，新增大文件仅约 7.15 GB。项目盘可用约 366 GiB。HF 下载 HEAD 200、文件头 Range 206，官方模型/公用组件元数据均可访问；未下载大模型。
- `scripts/vace_p01.py` 默认只读，`--prepare` 仅 CPU，`--run` 明确触发 GPU 子进程。冻结 JSON 逐字节 SHA256 校验，禁止擅改 005–021、512×512、16 fps、seed 20261018、30 步、CFG 5；不加载 LoRA。GPU 入口使用项目锁、校验全部权重哈希、禁用下载，并记录真实子进程退出。
- 真实准备结果在 `outputs/vace/P01_465_17f_baseline_cpu_prepare_20261019/`：17 张独立 RGB ModelMask、17 张填灰条件帧、原 Damaged/Exact 副本及单独 GT 评估目录。实际落盘文件经 DiffSynth CPU 预处理为 `[1,3,17,512,512]`，Mask 0/1，通过。没有生成预测或调用 VAE/DiT。
- 权重与网络证据在 `outputs/vace_model_audit/P01_465_20261019/`；来源与组件哈希在 `configs/vace_p01_model_sources_v1.json`。下载计划脚本 `scripts/download_vace_p01.sh` 默认不联网、不写模型目录；只执行过 dry-run。
- 当前只缺官方 VACE 文件和用户后续下载/GPU 授权。完整模型加载、真实峰值显存和生成效果未测，不承诺一定不 OOM；显式 CPU offload 是独立可选策略，不自动改变冻结参数。

## 2026-10-18 P01 接收检查记录

- 执行环境为 SSH Linux `vccip`，用户 `scj`；不能直接检查 Windows 主机 D 盘或其 OpenSSH 配置。本轮没有执行 SCP；已有文件位于 `data/buildingfiller_import/P01_465/`。
- ZIP SHA256 `5a966a269866a1423c1efa38937fcce6ec6984f8fda78c35395abccab83d5bb8`；ZIP CRC、262 项载荷哈希、243 PNG 实际解码、81 帧三角色/相机对齐、三对首尾闭合及随包校验器均通过。Exact 为 0/255，046–072 共 27 张全黑，合法保留。
- 新审计副本及报告在 `outputs/buildingfiller_handoff/P01_465_20261018/`；原 ZIP/Exact/Damaged 未改。其他七例和 768/1024 图片不在本次服务器载荷检查范围。
- RTX A6000 CUDA 可用，总显存约 47.4 GiB、探测时空闲约 47.1 GiB；DiffSynth 2.1.5 离线导入通过。已有 T2V DiT 没有 VACE 分支。共享 T5/VAE/tokenizer 存在；`/home/shi/.cache` 无读取权限，未把模型缺失结论扩展到未检查位置。
- 冻结计划 `configs/vace_p01_465_17f_v1.json`：连续源 005–021、17 帧、512×512、16 fps、原始 VACE-1.3B、不加载 LoRA。当前 pipeline 需要 RGB 三通道 Mask 列表，直接 L 模式会报错。基于 Exact 二值化、复制通道和 Damaged 白区填 RGB127 的 CPU 内存合同检查通过，但未落盘派生输入或调用模型。
- 下一步需明确 VACE 权重位置、实现项目专用执行器并落盘独立派生输入；GPU 推理必须等待用户确认。`training_ready=false` 保留，GT-oracle Mask 必须在结果中披露。

## 2. 2026-10-14 核对到的状态

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

## 3. 配置与步数口径

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

## 4. 训练完成状态：待进一步核实

此前文档报告“2026-09-29 02:25 自然完成 8 epochs，最后 step-11200”。保留这一历史报告，但本次不能独立确认正常结束：

- 最高现存权重是 step-11200。
- 最后一段 loss 从 3601 到 11562，共 7962 行。
- 最后一段参数为 batch 1、initial 3600、skip 6000、1992 样本、8 epochs；按当前引擎语义，耗尽配置预算应到 `3600 + (1992×8 - 6000) = 13536`。
- 保留日志尾部停在 523/1992，没有独立成功退出记录；不存在历史 `training_completion.json`。

这些不一致可能来自日志/产物缺失或实际运行差异，不能推断已经完成，也不能据此认定某次崩溃。不得自动补训。今后的项目训练入口仅在引擎正常返回、最终保存完成后写 `training_completion.json`。

## 5. 已知故障与修复范围

- 原训练在日志 batch 2742 发生 `free(): invalid next size (normal)`。这是原生内存 abort 记录，未见 GPU OOM 报错；根因未确认。
- worker 改为 0、batch 改为 1 后仍记录过 `free(): invalid next size (fast)`；不能声称 worker 0 已消除故障。
- LoRA 被误传给 `--resume_from_checkpoint` 的 600 unexpected keys 问题已通过 `--lora_checkpoint` 修复；该参数也不提供 optimizer 状态恢复。
- 本次修复平铺后的续训默认路径、累计样本/batch 换算、项目进程识别、Python 传递、启动失败报告和新输出目录隔离。
- 当前训练/续训/验证入口共享 launcher 锁；Python 验证依据解析后的 run 参数加锁，缩写参数同样受保护。旧进程仍需检查。其他绕过入口的命令不受该锁协调。
- 当前 Python 验证预检不写输出目录；只在用户明确要求 `--run` 时保存配置及生成视频。

## 6. 历史 T2V 待办

当前优先事项以本文件前面的 P01 / VACE 接收检查为准；下列待办属于历史训练与采集器维护，不触发自动训练或推理。

1. 补齐训练终态证据与源 OBJ 身份对应关系，核实已有预算和数据防泄漏。
2. 基于已生成视频做人工检查和多 seed 对比，选择 checkpoint，避免自动选最后一份权重。
3. 评估逐栋 prompt 时设计新缓存，不复用共享 context。
4. 如维护历史采集器，先恢复并审查缺失的扫描器源代码，落实“冻结一次、切分冻结 mesh、无额外变换地渲染三元组”。当前 VACE 读取独立 BuildingFiller 包，按 P01 交接方案实现。

数据、模型、缓存、已有权重和输出保持默认只读。新实验使用独立输出目录。Windows workspace 文件保持不动。

## 7. 本轮复审问题已修正

复审发现并修正：验证缩写参数绕过锁；打乱分组下短 batch 的样本预算推断；暂停等号参数解析。短 batch 的局部 epoch 游标现在拒绝猜测，完整 epoch 边界仍可换算；当前 1992 样本、batch 1/2 的历史换算不变。

补测发现 scj Python 的 os/signal 没有 pidfd 接口；停止入口改为使用系统 Python 的 pidfd 控制，重复核对进程启动时间和项目命令。通过临时 CPU 进程验证了只记录不终止、两种输出参数的暂停/终止、身份不匹配拒绝发信号，以及缩写验证命令在锁被占用时拒绝启动。以上早期代码审计未运行正式 GPU 训练/推理；随后获授权的 GPU 诊断见下一节。

权重头部预检改为 safetensors NumPy 后端，不加载 PyTorch，非法游标在打开权重文件前拒绝。补充严格退出码测试，避免把 native abort 当作正常的参数拒绝；正式训练的原生崩溃根因仍未确认。

## 8. 2026-10-14 至 15：真实 GPU 诊断

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
