# 会话交接与持续更新记录

最近补记：2026-10-23（UTC+08:00，北京时间）。本文件持续保留历次会话接手事实、用户约定、阶段进度、故障处理和交接结果。历史正文已恢复在原文件内，后续进展按日期追加；归档仅作备份，无需离开本文件才能阅读旧记录。

阅读顺序：[此前累计正文](#history-records) → [2026-10-23阶段更新](#update-20261023) → [本次历史恢复及生效约定](#history-repair)。旧段落的“当前”“下一步”、命令和PID按其记录时点理解；较晚的更正与实际完成证据在后文，旧命令不因此重新执行。

截至本次修复：Pilot8系列及Complex12的C/U/S各300更新、五方法60条81帧已完成。本次仅修复文档；[实验入口](experiments/README.md)和[文档索引](docs/DOCUMENTATION_INDEX.md)仍可直接使用。

<a id="history-records"></a>

## 一、此前累计正文：截至2026-10-22的累计记录

以下完整保留原有记录顺序、参数、命令、过程和当时结论，只调整标题层级。原记录中的未完成状态与后来完成状态共同构成历史，不能用后者替换前者。

<!-- RESTORED_HISTORY_BEGIN -->
### Session Handoff

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

更新时间：2026-10-20 20:09:57 +09:00。LoRA-300八例81帧评估全部完成，32项成果有效，两个批次退出0，GPU已空闲。本轮停止，不自动启动新训练或其他实验。

#### 当前接手：八例完成，直接查看成果

- 实际结果根目录 `outputs/vace/lora300_eval81/run_20261020_194300_145372/`，01–08各四文件，共32项。原始视频512×512、比较视频2048×544，均81帧/16FPS/H.264 High/yuv420p/BT.709并完整解码通过；PNG可读，临时查看图已清理。
- 唯一权重 `outputs/vace/lora_roof_smoke/run_20261020_165237_150325_2048720/training/lora_update0300.safetensors`，vace分支、scale1。固定81帧512/16FPS/30步/CFG5/seed20261018及原正负文本；Damaged填灰127+原二值ModelMask，合法全黑帧保留，GT推理后才读，不回贴或调色。
- 先P04→P06，查看两例全部81帧后在终端明确确认改善成立，再按原授权完成P01/P02/P03/P05/P07/P08；八个worker与两个批次全部exit0。批次369.2+1129.4=1498.6秒，各例峰值allocated17.141/reserved25.832GiB。
- P04屋顶凹槽持续填平；P06黑洞在全81帧保持闭合，没有后半圈重开，侧向棱线仍轻微起伏。这是训练题扩展。六例迁移中P05/P07/P08明显改善，P01/P02有限，P03整体基本持平且部分视角变差；P05前半圈还有小凹点/首尾差异。已查看八例全81帧局部对照、总览和Mask图；详细帧号与统一解码口径指标在EXPERIMENTS.md。
- 真实终端为 `vace_p01_17f:4.0`，窗口名 `lora300_eval81`，pane `%4`，已经回shell；NVML无计算/图形进程。可只读重连查看历史，不要再次运行推理入口。八例Base全部复用，Base补跑0、新训练0。
- 原test/composite_check/小样本run及源包69项大小mtime未变。新媒体只在本轮独立目录；模型/视频/图片/临时文件不入Git。后续方向建议保持骨干、扩大建筑补全LoRA训练覆盖后再研究几何/Depth，但没有开启下一轮的授权。

#### 历史：首次制图失败及获授权重跑

- 首次P04完成30/30步与解码，175.2秒、显存峰值17.141/25.832GiB；CPU四栏制图因PIL未转NumPy而TypeError，worker及批次退出1，未启动P06。临时目录清理导致该次原始预测丢失，不能用它评估效果。
- 修复统一像素类型并先保存最终01原始预测，增加 `--render-only` CPU补图模式；298项CPU测试通过（53.124秒），修复已提交推送 `ea51376`。用户回复“直接修复，然后再重跑”后才开始上述成功批次，无未授权自动重试。
- 本轮实际LoRA推理总计9次，包含最初失败制图后的P04一次及获授权重跑；最终成功交付8例。旧失败根目录未覆盖，成功结果使用独立时间戳子目录。相关GPU过程与异常均已保留在文档中。

#### 持续约定：GPU运行后更新文档并Git记录

用户明确要求：每次用户启动GPU训练/推理之后，接手时主动核实运行事实，更新四份根入口文档及必要的专项说明，然后审查、提交并推送相关代码/配置/文档，不需要用户再次提醒。运行中只记带时刻的进度，成功/失败/中断均如实记录；不因记录工作自动另起GPU任务或续训。

GitHub身份已实测为woshuaime，origin为 `git@github-scj:woshuaime/building_wan.git`。模型/数据/结果媒体/日志不入库；其他历史未提交改动保留，不混入本轮提交。持续规则已写入AGENTS.md第7节。

#### 最新接手：LoRA首轮完成，查看成果

- 现有run：`outputs/vace/lora_roof_smoke/run_20261020_165237_150325_2048720/`。重连 `tmux attach-session -t vace_p01_17f`，选择窗口3；实际pane为 `%3`。本轮复用已有会话，不能按通用示例误找 `vace_roof_smoke` 或另开同一实验。
- 最终 `runtime.status=complete, stage=complete, actual_updates=300`；loss.csv连续300条，P04/P06各150条；推理started/completed均6。Base/train/lora100/lora300均complete、exit_code0，两份LoRA各43,780,800 bytes，04/06各四项有效成果。总耗时749.247秒。
- 终端前5步已确认有限loss、有效梯度和LoRA参数变化；training_result也记录gradient_check_passed/parameters_changed为true。训练耗时550.933秒、峰值15.139/15.393GiB allocated/reserved；全流程峰值15.867/17.385GiB。用户截图16.7%是显存占比，GPU-Util为100%，单帧8GiB不能当作训练峰值。
- 两例窗口都固定B000–B016，P04/source1227、P06/source1848；按B000–B079内17连续帧Exact总白区最大选定，排除B080。正负文本逐字继承成功VACE配置，见 `configs/vace_roof_smoke_v1.json`；Damaged/Exact原样、填灰127和二值RGB Mask沿用原适配，GT只供训练target和后验比较。
- 程序顺序Base两例17帧→共享vace LoRA训练300真实更新→100/300各两例17帧→CPU制图；fresh进程重载原始模型避免融合累加，LoRA只加载vace、倍率1。训练rank=alpha16、目标q/k/v/o/ffn.0/ffn.2、恒定5e-5 AdamW、wd0、clip1、batch1、accum1、checkpointing、seed20261018。FP32 LoRA参数/BF16骨干和autocast，冻结DiT仍回传，T5/VAE编码后移CPU。
- 只保存两份LoRA、experiment.json和loss.csv；两样本各4成果。真实进度继承终端，不写run.log/worker.log；失败停止且保留权重，CUDA OOM不自动降规格，训练worker非0退出不能算完整成功。Loss下降或Mask内RGB误差降低不能替代结构闭合视觉检查。
- CPU预检和294项回归通过，150目标层/300张量/10,936,320参数匹配；本次文档维护又以空CUDA_VISIBLE_DEVICES执行294项CPU测试通过，45.607秒。只更新文档、只读核查现有任务，没有新增GPU调用或改训练程序。
- 已抽看04/06的 `03_训练进展对比.png`：五个视角中100/300步P04屋顶和P06开口均明显更接近完整GT。完整17帧运动/闪烁和用户人工验收尚未完成，runtime仍为 `visual_assessment=pending_user_review`，不要改写成全面三维成功。先看两例的02视频，必要时结合04 Mask图定位。
- 自有临时目录已清理，成功输出共12文件，原test/composite_check未动。入口没有完整训练状态resume；再次运行启动器会创建新实验，不能用这两份增量冒充恢复点。
- 此任务是训练题过拟合验证，不是独立测试；后续不得用学过P04/P06的LoRA冒充无泄漏初始化。完成后停，不自动做81帧、Mask扩张、Depth或扩大训练。

#### 已完成记录：八例缺损区域合成

- `outputs/vace/composite_check/01–08/` 各恰好三文件，八例SUCCESS、退出0、GPU推理0次；tmux `vace_p01_17f:vace_composite` 保留实际终端进度。使用 `scripts/composite_vace_results.py`，数据从只读Pilot8 ZIP按真实身份/帧号读取，ModelMask复用原推理 `adapt_images`。
- Composite只取白区原预测、黑区Damaged，未调色、羽化或修改Mask，GT仅对照与诊断选图。全黑Mask直接保留Damaged；两个视频81帧/16FPS完整解码通过，采用High/yuv420p/BT.709。原 `test/` 32文件SHA/大小/mtime全部不变，未改源数据，临时文件已清理，272项CPU测试通过。
- P01未损区域灰度和主体保持改善，但出现可见拼接细缝；P04屋顶仍有凹槽/台阶，重点看 `04/03_关键帧对比.png` 的B003/B066；P06黑洞仍在，重点看 `06/03_关键帧对比.png` 的B006/B020。其他样本也可见补丁边界。优先研究缺口内部结构生成，边界问题次之；当前结果不能证明三维一致性或所有运动形变已解决。
- 本次已停止，不自动开展Mask扩张、GPU推理、LoRA或新下载；不要因为原test和composite_check同存就删除/替换原预测。再次执行CPU入口会新建时间戳根目录，不应无故重复生成。

#### 已完成记录：八例离线三方对比

- 八例离线重排和RGB播放偏色修复均已完成，P01–P08全部SUCCESS、退出0，GPU推理次数0；当前八个三栏视频已变为High/yuv420p/BT.709，各目录仍严格四文件。独立32文件比对证实只有八个02改变，01与两张PNG全SHA不变、GT/Mask无缺失、自有临时目录清理通过。完整266项CPU回归通过。下一步关闭旧视频预览后重新打开，避免旧媒体缓存，不再执行推理。
- 当前入口为 `scripts/reorganize_vace_results.py`；默认只读，`--apply`对旧布局进行CPU重排，已经升级的目录如果还是RGB 4:4:4则只修复02三栏视频；已为兼容格式则只验收。原始预测缺失即失败，绝不触发VACE。
- `test/01–08`的固定源对应465/763/871/1227/1259/1848/2297/939不变。原始视频保留；新版三栏视频、三行五列总览和重命名四栏图见PROJECT_CONTEXT。不得将GT或其他样本代替预测。
- CPU任务运行在已有 `vace_p01_17f` 的 `vace_layout` 窗口，真实进度和退出码在终端，不写运行日志或JSON。使用空CUDA_VISIBLE_DEVICES，入口还拒绝Torch/DiffSynth导入。
- 用户报告原RGB三栏播放紫绿偏色：旧文件实际为High 4:4:4 Predictive/gbrp，不适合当前播放器。原260项测试只证实FFmpeg解码像素，遗漏播放器兼容性。修复为libx264/High/yuv420p/BT.709有限范围，保护01与两个PNG字节及文件身份；新检查包括编码元数据、白色背景、三栏颜色/帧序和单视频替换失败回滚，不加载模型。

#### 历史接手：保留01–03，仅恢复P04–P08（后续已成功）

- 用户下一条：`bash /mnt/windowsE/chuanjun/building_wan/scripts/run_vace_building_test.sh --resume --tmux`；在现有 `vace_p01_17f` tmux中会复用该会话并打印实际重连命令，不嵌套；新会话名仍 `vace_building_test`。只读复查加 `--resume --dry-run`，不能去掉resume直接覆盖重跑。
- 原因证据在tmux历史：P04 sample_preflight，`inspect_models(...verify=True)` 报 `Reused component SHA256 mismatch: text_encoder`，未进入该例模型加载。不是OOM，P02/P03各真实81帧成功，峰值17.14GiB allocated /25.73GiB reserved。当前没有仍运行的batch/worker。
- T5原件连续三次独立全SHA256均匹配批准 `d92de679881d38af9c89eff7bb1b6d6c9d96cb2b69831e4027e9ecabdd38eb23`，11,361,845,432bytes及inode/mtime/ctime未变。原失败未记录actual，故底层原因未确定，不能宣称磁盘/RAM/读取故障已修复。仍保留严格hash门禁，任何再失败立即停，不重试“碰巧通过”、不跳过验证或更改权重。
- 新 `verify_file_sha256` 打印expected/actual/path/bytes及前后stat，单次全读取并检测替换/修改。新 `vace_building_reuse.validate_completed_sample` 全解码四文件并逐像素查源身份、前后SHA不变；显式resume只接受连续成功前缀。parent/gate比对既有文件hash和待跑队列，之后重新复核才进入新worker。
- test/01–03各恰四文件，P01四SHA等于原成功目录，P02/source763和P03/source871的五视角Damaged、四栏Damaged/Mask/GT均对应。P01–P03不重新调用模型；队列仅P04→P08。04–08当前不存在。所有CPU测试与临时编码只在自有/tmp，不能冒充新预测。
- 本轮实际执行 `--resume --dry-run` 通过32.1s，输出 `verified completed prefix=['P02','P03']`、唯一队列P04–P08；全权重SHA、CUDA环境和七例CPU loader均通过。完整250项测试通过35.630s（新增16项既有结果验收、8项严格hash、20项恢复/终端路由），规定及新脚本语法/编译/diff通过。模拟resume验证不重算01–03、gate/parent hash一致、失败即停及native139、拒非连续/缺失/错号/软链接、清理自有/tmp、parent无Torch和终端参数传递。匿名gate事件UTF8编码最大七例四hash实测2928bytes，小于4096原子写上限。未启动GPU、修改权重/原始数据/历史结果或提交Git；自有临时目录无残留。

#### 此前接手：P01复用，P02–P08批量准备历史

- 最终启动：`bash /mnt/windowsE/chuanjun/building_wan/scripts/run_vace_building_test.sh --tmux`。新建/重连可见 `vace_building_test`；已在 tmux 则复用并显示实际重连命令。断线重连用 `tmux attach-session -t vace_building_test`，不要另起七例。
- 新 ZIP 位于 `data/buildingfiller_import/P02_P08_20261020/BuildingFiller_Pilot8_AllRes_v1.zip`，SHA256 `7f8eff9ddb3c05e43fe5a8623258f9dceb6fa61d547b244c4f2d441099b4fcf4`。CRC/6077载荷哈希/5832PNG解码/相机帧号/几何/有效人工 accepted 均通过，未解压原包或修改数据。包含 P01，但只核查身份；`training_ready/generation_verified=false` 不是训练发布许可，`generation_evidence_verified=true` 不等于 VACE 补全成功。
- 固定输出顺序：`02=P02/763`、`03=P03/871`、`04=P04/1227`、`05=P05/1259`、`06=P06/1848`、`07=P07/2297`、`08=P08/939`。512 输入每角色81帧；P02黑 Exact023–049，P03为000–016和071–080，后五例无黑帧，各自保留。
- P01 原结果 `outputs/vace/P01_465_81f_20261019_171510_1691517/` 仍保留；两视频81帧/16FPS及两PNG有效，源465图像面板身份通过。已实际执行新入口 `--prepare`，复制到 `test/01` 后再次检查四文件哈希与原件一致。`test/02–08` 还不存在；本轮不生成这些预测。
- 历史 P01 无 manifest，无法从视频反推历史 seed/权重/prompt。保留执行代码和冻结配置与本轮固定参数一致，用户确认原始81帧成功；不能声称媒体独立证明了全部历史参数。
- 原始正式 VACE、本地 T5/VAE/Tokenizer 完整哈希、CUDA/BF16/高效SDPA、无其他GPU进程、空间权限、全部七例实际CPU loader 合同已通过。启动时完整门禁再做一次；parent 不初始化 CUDA，七个 fresh worker 逐个调用同一81帧核心，GT020仅预测后读取；无 LoRA、offload回退、分段或参数更改。
- `scripts/buildingfiller_pilot8.py` 负责严格包验收和内存适配；`scripts/vace_p01_reuse.py` 验证/安全复制旧结果；`scripts/vace_building_test.py` 只调度 P02–P08。原始17帧JSON SHA仍为 `7e82161e270d8a24e231feb2ca95d08ae6e02e861ce5460a241730bfbcc07080`，没有修改。
- 每个成功样本严格两个视频+两PNG，名称沿用原四文件；CPU测试假数据只在隔离/tmp。只清理本轮自有系统临时目录，失败即停止并保留此前成功输出，绝不伪成功。若启动后部分成功，入口会拒绝再次覆盖已有02–08；需要另行确认恢复策略，不能直接自动续跑。
- 本轮最终完整 `scripts/tests` 206项通过（35.474s），新增14项交接包、13项P01复用、29项批量/共享核心/终端路由CPU测试；规定及新脚本语法、编译、`git diff --check`通过。正式入口实际执行 `--help`、`--prepare`、复制后的 `--dry-run`，均成功；未执行GPU分支。首轮有一个既有GT隔离用例的MP4文件字节哈希断言波动，未复现根因；测试改为更直接的全部81帧解码像素一致、总览hash一致、四栏前三栏不变/GT栏变化后，目标连续3次及完整套件通过；没有改编码器或推理逻辑。最终仅test/01四文件，无本轮系统临时目录残留，两个原ZIP及冻结JSON哈希未变；未提交Git。

#### 历史接手：P01 81 帧、只留四文件（现已完成）

- 用户命令：`bash /mnt/windowsE/chuanjun/building_wan/scripts/run_vace_p01_465_81f.sh --tmux`。已有 tmux 会直接复用当前会话并打印实际重连命令；否则新建/重连 `vace_p01_81f`，避免重复创建实验；未安装 tmux 时前台运行、不安装。
- 81 帧预检已实际只读通过：官方模型/共享组件完整哈希、原 ZIP/81 组相机与 Damaged/Exact、原 27 黑 Mask、当前 DiffSynth CPU loader、CUDA/BF16/高效 SDPA、无其他 GPU 进程、输出空间/权限。只输出 PASS/FAIL，没有保存新的审计或 CPU 派生文件。
- 代码复用旧适配器/模型检查/本地模型加载/视频验收；旧 17 帧配置不改，只在内存派生 81 帧。一次 GPU worker、一次全序列 pipe 调用；GT B020 在预测完成后才读，B080 不复制预测 B000。
- 新 ConsoleState 仅内存/终端，真实 scheduler tqdm + VAE 阶段。匿名 pipe 给父进程传真实阶段和峰值，不落盘。父进程不创建 CUDA 上下文，避免被子进程 NVML 误认其他任务；所有预检在 worker 或显式 dry-run 内执行。
- 必要四个编码文件位于独占 `/tmp/building_wan_vace81_*`，父进程监督 native/中断并在正常失败清理；发布期 TERM/HUP 也清理。只复制四个已验收文件到全新最终目录，未用危险通配符删除其他结果。若发布遇磁盘写入失败，可能留下本次部分最终文件，必须 FAILED，不谎称四文件成功。
- 成功后只有两个 81 帧/16FPS MP4 和两张 PNG：总览000/020/040/060/080、四栏020。无 GT 回贴/美化/三维质量推断。显存峰值仅正式运行后才能报告，发生 OOM 必须停下，不自动重试或更改规格。
- 本轮验证：新脚本 `--help`/`--dry-run` 实际通过；`git diff --check`、规定及新增脚本 `bash -n`/`py_compile` 通过，完整 `scripts/tests` 150 项通过（新增46项81帧CPU测试）。真实合成81帧双视频编码/解码、GT隔离、B080、四文件名单、tmux路由、native/中断/发布期信号清理均覆盖。`outputs/vace` 仍无新子目录，系统本次临时目录无残留，原ZIP与17帧冻结JSON哈希未变；未运行GPU模型或提交Git。

#### 2026-10-20 17 帧准备历史（旧输出不再可用）

- 官方正式文件现已存在，7,146,067,848 bytes，完整 SHA256 `c46a6f5f7d32c453c3983bbc59761ea41cd02ad584fb55d1a7ee2b76145847a2`；T5/VAE/4 个 Tokenizer 同样完整校验。不要再次下载，也不要误用下方 10-19 的缺权重历史状态。
- 新审计 `outputs/vace_model_audit/P01_465_20261020_20261019T161614Z/` 包含完整模型预检、实际环境与官方依据；新 CPU 准备 `outputs/vace/P01_465_17f_baseline_cpu_prepare_20261020_20261019T162331/` 已通过 CPU loader。报告时间采用 UTC，所以时间戳可能为 10-19；会话日期为 10-20。
- `scripts/run_vace_p01_465_17f.sh` 默认前台 `--run`，只有用户亲自执行才授权一次真实实验；`--dry-run` 只完整哈希 + CPU loader，不推理。tmux 已安装、未自动创建会话。终端启动/重连/分离命令见 [执行说明](docs/VACE_P01_EXECUTION.md)。
- 真实去噪进度由 `progress_bar_cmd` 按实际 30 步计数，VAE encode/decode 包装只记录真实阶段，vendor 自带 tile 进度不改。字节流实时转发同时落 `run.log` 和 `worker.log`；中断只停止自己的子进程组。
- 输入别名为 `input_damaged/`、`input_modelmask/`、`input_condition/`、`gt_reference/`，指向本次独立副本；真实输出 `raw_prediction/` 和 MP4，兼容旧 `prediction_raw` 相对别名。GT 不进 worker，评估在真实 worker 成功之后才运行。
- 独立结果模块验证 17 PNG、视频实际解码、FPS16/512，再生成三张指定图、可选同步 MP4、metrics/validation。只有全流程保存并验收完成才打印 SUCCESS；失败保存实际退出码/阶段/日志，报告写入失败也必须 FAILED。
- 原始模型真实加载、峰值显存和补全质量尚未实测；没有预测图或质量达标结论。未改冻结 JSON、旧训练/权重/缓存、第三方核心，未提交本轮或其他既有未提交工作。
- 阶段 A 验证：`git diff --check`、规定训练/续训与新启动/下载脚本 `bash -n`、规定及全部新增 Python 文件 `py_compile` 通过，`scripts/tests` 104 项全部通过。CPU 合成 fixture 测试覆盖实时无换行/中文输出、中断只终止自己的子进程、GT 隔离、真实 MP4 编解码与错误帧数/FPS、报告写入失败不误报成功。正式启动脚本只执行了 `--help` 与 `--dry-run`，未执行默认 GPU 分支。

#### 2026-10-19 P01 / VACE 准备历史

- 先读 [VACE 执行说明](docs/VACE_P01_EXECUTION.md) 与 [P01 交接方案](docs/BUILDINGFILLER_P01_VACE_HANDOFF.md)。实际执行端是 SSH Linux `vccip`，用户 `scj`，不能把 Windows 主机的 D 盘或 C 盘 SSH 配置当成服务器文件。
- 服务器 `data/buildingfiller_import/P01_465/` 已有 ZIP、SHA256 和 `Stage1_Closeout_Report.md`。本轮仅读取原件，未执行上传；Windows 端源路径/SSH 客户端/别名检查命令已提供，尚未在 Windows 实测。
- ZIP SHA256 `5a966a269866a1423c1efa38937fcce6ec6984f8fda78c35395abccab83d5bb8`；262 项哈希、ZIP CRC、243 PNG 完整解码、81 帧对齐/相机绑定、三对首尾闭合及随包校验均 passed。Exact 046–072 共 27 张全黑，不能删除。
- 审计报告、联系图和独立解压副本在 `outputs/buildingfiller_handoff/P01_465_20261018/`；原数据不改。只检查了 P01 512 子集，不把来源记录的全部八例当成本机逐例检查。
- DiffSynth 2.1.5 可导入，A6000 总显存 47.4 GiB。官方正式文件为 7,146,067,848 bytes、SHA256 `c46a6f5f7d32c453c3983bbc59761ea41cd02ad584fb55d1a7ee2b76145847a2`；官方文件头确认 1264/439 总/VACE 张量，结构可被本地双分支加载。
- 扩展搜索找到其他目录 VACE 候选，但文件 SHA256 为 `fcd12e2a8fab331ba0ab28dca371d90036a868287767f35677ff3f8e74db0969`，即便消除文件头 metadata 差异也不匹配正式载荷，禁止用于原始基线。详细路径与证据见 `outputs/vace_model_audit/P01_465_20261019/`，不能按文件名冒充原始模型。
- `configs/vace_p01_465_17f_v1.json` 保持原字节不变：005–021、17 帧、512×512、16 fps、seed 20261018、30 步、CFG 5。新增 `scripts/vace_p01.py` 已实现只读/CPU 准备/显式 GPU 三模式，真实准备目录 `outputs/vace/P01_465_17f_baseline_cpu_prepare_20261019/` 包含独立 Mask、填灰条件帧和映射，GT 单独供评估；落盘后 CPU 加载合同检查通过，没有生成预测。
- T5/VAE 完整 SHA256 已匹配官方转换组件，Tokenizer 可复用。缺正式 VACE 单文件；官方 HF/ModelScope 元数据和 HF CDN 可访问，项目盘空闲约 366 GiB。`configs/vace_p01_model_sources_v1.json` 固定来源/哈希；下载脚本只做 dry-run，下一步须先获得下载授权，再完整预检，再另获 GPU 授权。
- `--run` 始终核验模型哈希，GPU 子进程持有继承的项目锁，外层记录真实退出码，原始 PNG 与显式可选合成分开。禁止自动安装/升级、改第三方核心、自动降帧降分辨率或自动 offload 回退。实际 GPU 路径未测，不宣称补全成功。
- Git 账号已确认为 woshuaime；此前 P01 交接提交 `a2e8b4c` 已推 main。本次新增执行器/来源配置/检查与文档尚未提交，既有 T2V 未提交工作保留。未改原数据、权重或历史输出。
- 本轮检查：git diff --check、训练/续训/下载脚本 bash -n、训练/验证/VACE 入口和新测试 py_compile 通过，scripts/tests 51 项全部通过（含新增 15 项 VACE CPU 测试）。真实包只读复验、已有输出目录拒绝且内容不变、落盘输入哈希、冻结配置与 17 帧映射、默认预检不导入 Torch 检查通过；原模型目录仍为 9 个文件，旧缓存仍为 1992 项。PowerShell 仅审阅，未在 Windows 执行；下载与真实 GPU 路径未执行。

以下为 10-15 及更早 T2V 诊断交接，不能当作启动当前 VACE 或历史训练的授权。

#### 当前事实

- 工作副本 `/mnt/windowsE/chuanjun/building_wan`，分支 main；审计前 HEAD 为 `8a3720e`。后续以 `git status` / `git log` 的实际结果为准，不把旧提交号当成已同步状态。
- scj Python `/home/shi/miniconda3/envs/scj/bin/python` 已确认 PyTorch 2.11.0+cu128、CUDA 12.8、CUDA 可用；历史硬件为单张 RTX A6000。
- 单轨视频 2344；train/val/test 为 1992/234/118，统一 prompt。1992 缓存和共享 context 存在。
- 平铺单轨权重 `checkpoints/building_wan_a6000_single_orbit_all/` 共 28 份，400–11200；A100 历史 building_wan 39 份独立保留。
- 全部单轨对比视频已经生成，29 个完成元数据及文件位于 `outputs/validation/validation_all_single_orbit/`。不用默认重新生成。
- “8 epochs 正常完成”是待核实的历史报告：现存 loss 末步 11562、最高权重 11200、按保存参数推算预算末步 13536，日志缺少完整终态。不可自动补训或补造结束标记。
- GUI 采集 `src/` 和两份数据集工作流文档缺失；不能声称扫描器完整可运行。

#### 本次修复

- `scripts/wan_a6000_state.py`：从 checkpoint 段参数计算累计样本，换算不同 batch，识别本项目进程；支持可追溯的旧记录路径迁移、v2 暂停记录、所选进程停止和续跑预检。
- `resume_wan81_a6000.sh`：修复平铺 step-2400 默认路径，加入 `--dry-run`、参数/缓存校验、新顶层输出和 launcher 锁。
- `pause_wan81_a6000.sh`、`resume_from_pause.sh`：使用上述公共工具，正确传递 WAN_PYTHON；不再无条件报告训练成功。
- `train_wan81_a6000.sh`：默认独立输出，序列化启动器，拒绝覆盖未验证缓存。
- `wan_train_entry.py`：正常返回并最终保存后写 `training_completion.json`；logger step 仍保持 batch 口径。
- Python 验证入口：只读预检、固定划分与 prompt 检查、参数/来源一致时复用完成视频。
- `analyze_validation_videos.py`：v2 全图前景占比与二阶像素变化，写独立新目录，旧结果原样保留。
- 冻结划分生成脚本拒绝静默覆盖；增加 `scripts/tests/test_a6000_tools.py`。

#### 接手顺序

先读 AGENTS.md、PROJECT_CONTEXT.md、TRAINING.md、EXPERIMENTS.md 和本文件；检查 Git 状态、实际环境、进程及产物。详细证据和验证见 `docs/PROJECT_AUDIT_20261014.md`。

优先核实训练终态、源 OBJ 防泄漏及已有视频效果；逐栋 prompt 或补全使用独立实验。不要直接运行历史 A100/Slurm 或 Windows 命令。续跑是权重与样本预算续跑，不恢复 optimizer/RNG、梯度累积或 shuffle 顺序。

#### 保护与交接

data/models/cache/已有 checkpoints/outputs 默认只读；历史数据和权重未改。只有本轮新建串行测试产物按用户要求删除，见下方清单。原有未跟踪日志、验证产物和 diffsynth.egg-info 保留。未调用 Claude worker，没有自动提交、推送或 pull；需要同步时遵循 AGENTS.md 的非强制 main 流程。

#### 复审补修与验证

- 先解析 args.run 再加锁，验证 --ru/--r 与 --run 相同，预检不获取锁。
- 短 batch 会随预分组 shuffle；来源与目标 batch 存在短组时，只在完整 epoch 边界允许推断游标，其他情况明确拒绝。不要恢复旧的“短尾总在最后”公式。
- 暂停兼容等号、分隔参数、空格和相对路径。
- scj 缺 pidfd API；通过系统 Python 控制 pidfd，核对项目命令及启动时间，避免向复用 PID 发信号。真实 CPU 测试已验证记录、身份拒绝和停止；未停止任何正式训练。
- 早期审计的新工具 23 项与既有训练工具 12 项均通过，无跳过；新增测试覆盖真实 CLI 的锁冲突与真实 CPU 进程生命周期。随后获授权的 GPU 诊断见下一节。

- 最后预检曾在原 PyTorch 权重头部读取路径退出时出现 native 内存释放错误；当前头部检查改用 NumPy 后端，并验证预检不导入 torch，错误参数必须正常以退出码 1 拒绝。此改动不证明正式训练的原生崩溃根因已修复。

#### 2026-10-15：GPU 诊断已结束，禁止自动继续

- 普通反向 GDB 基线在新增 166 batch 后（loss step 11366）触发 SIGSEGV，
  启动器退出 255；现场为 UnsafeViewBackward0 析构/计算图释放。首次非法写入
  未定位，根因未确认。logs/native_stability_20261014/baseline.log 和
  baseline_result.json 保留；旧 PID 不可复用。
- Valgrind 覆盖 196 batch，报告 0 错误后主动停止，退出 143；GPU memcheck
  覆盖 4 batch / 一次累积更新，报告 0 错误后主动停止，原计划 8 batch 未完成。
- 8 GiB memtester 一轮与 CPU 运算校验 120 秒均退出 0，不能排除所有硬件问题。
- 固定种子普通/串行短对照各 8 batch、各复测一次，均正常退出。更新前 loss
  相同，更新后同模式也有波动，最大权重差 3.0517578125e-05，未逐位一致。
  普通反向 256-batch 对照也正常退出，原生故障是间歇性的。
- 串行完整测试：10-15 03:52–06:07，seed 20261015，batch 1 / 累积 4 /
  worker 0，从历史 step-11200 权重续跑；新增 2336 batch，step 11201–13536，
  loss 连续且全部有限，末值 0.00538101838901639。6 份新权重各 600 个 BF16
  张量全有限、键名/形状匹配，Python 正常退出，无原生崩溃。这次独立
  weights-only 预算测试不能替代历史正式训练终态证据，不能证明根因修复。
- 旧 GDB 在正常退出后执行不可用的 py-bt，原始 launcher 码为 1，已与
  exited normally 证据分开核对。scripts/wan_gdb.py 已修复误判，并增加正常
  exit 0、Python exit 9、SIGABRT、SIGSEGV 的真实 CPU GDB 回归；该 GPU 长测
  启动早于修复，仍使用旧命令，不冒称新 helper 已完成 GPU 长测。
- scripts/check_wan_lora.py 通过 CPU 完整读取 BF16 payload 并检查有限值、
  可选参考键名/形状。WAN_SERIAL_BACKWARD 默认仍为 0，串行选项只是候选绕过。

#### 用户要求的串行产物清理

清理前复核 loss、完成标记、退出证据以及 6 份权重哈希和完整读取报告。
随后删除以下 3 个新建目录及对应串行日志、混合对照报告、临时监督脚本和 pyc：

- checkpoints/native_stability_20261015_serial_full
- checkpoints/native_stability_20261015_parity_serial
- checkpoints/native_stability_20261015_parity_serial_repeat

共删除 8 份权重、25 个日志/临时脚本文件，合计 350824306 bytes。
结果只保留文档摘要，原始串行报告已删除。普通反向对照、基线崩溃和其他
内存诊断证据保留。历史平铺权重仍为 28 份，step-11200 SHA256 仍为
4a085810854b550cdd912e3edebfbe74f4d9229dc67abd623fe84dc002743161，未变。
1992 个缓存和既有验证视频保留。核对时训练与监督进程均已退出，tmux 无会话；
未向旧 PID 发信号，没有重新启动 GPU 作业。后续遵守用户最新停止要求。

详细诊断见 docs/NATIVE_STABILITY_20261014.md；首次非法写入仍未定位，
不可把 GDB 误报修复、一次串行通过或 CPU/RAM 阴性结果写成原生崩溃已根治。

#### 清理后的代码检查

git diff --check、5 个项目 shell 入口 bash -n、训练/验证入口和诊断工具
py_compile 均通过。scripts/tests 的 36 项测试与既有训练工具的 12 项测试
全部通过，无跳过；只使用 CPU 和临时测试进程，没有启动 GPU 训练或推理。
<!-- RESTORED_HISTORY_END -->

<a id="update-20261023"></a>

## 二、2026-10-23：阶段更新与当前资料补全

下面继续保留357b09f阶段补充的结果、参数和操作说明。该阶段曾把历史移到归档、部分原文页改成导航；这一文档组织做法已由第三节的用户更正取代，实验事实和既有检查结果保留。

<!-- RETAINED_UPDATE_BEGIN -->
### 会话交接

核对日期：2026-10-23（UTC+09:00）。本次任务为“全部项目Markdown细致更新”，没有启动训练、推理、外部worker或新实验。

#### 1. 接手顺序

1. 阅读[AGENTS.md](AGENTS.md)、[当前状态](PROJECT_CONTEXT.md)与[操作手册](TRAINING.md)。
2. 从[实验目录](experiments/README.md)确认目标是Pilot8、Complex12还是历史T2V。
3. 核对git status/log及对应run的真实状态，不能从历史快照的PID或“下一步”直接行动。
4. 查看[完整文档索引](docs/DOCUMENTATION_INDEX.md)区分当前说明、专题历史和冻结档案。

#### 2. 最新成果已经完成

Complex12的C/U/S均300更新，B/L/C/U/S各12条81帧；60轨迹、4860PNG、12视频、12总览、指标及中文报告齐全。progress.json为complete，无待自动续跑任务。

[结果入口](outputs/ComplexSingleOrbit_Rev1_Pilot_20261022/00_看结果/README_从这里开始.md)提供12栋身份、媒体链接与训练/验证表。S的平均数略领先U，不代表结构全面改善；细部弯折、保留区变化、闭环代价均已写入[报告](outputs/ComplexSingleOrbit_Rev1_Pilot_20261022/00_看结果/最终实验结论.md)。

#### 3. 重要路径与兼容关系

| 内容 | 位置 |
| --- | --- |
| Complex12唯一专用源码 | experiments/vace/complex12_rev1/scripts/ |
| 固定run | outputs/ComplexSingleOrbit_Rev1_Pilot_20261022/ |
| 新浏览入口 | experiments/vace/complex12_rev1/runs/20261022/，目录链接 |
| 原scripts兼容路径 | 固定run/scripts，指向新源码 |
| 当前冻结配置 | 固定run/resolved_config.json，配置ID以真实JSON为准 |
| 三组正式权重 | 固定run/checkpoints/C、U、S/step300.safetensors |
| 恢复入口 | 固定run/resume.sh；已完成时直接返回 |
| 历史L权重 | outputs/vace/lora_roof_smoke/run_20261020_165237_150325_2048720/training/lora_update0300.safetensors |

旧resume.sh和结果路径不因源码迁移失效。新想法必须新建实验目录和run，不能改写当前完成标记或冻结配置。历史LoRA和原始数据不复制进新源码目录。

#### 4. 必须延续的事实

- C在挂载恢复后从40次更新继续到300，完整恢复optimizer/RNG；新U/S各自从共同新初始化开始。
- B/CS04在保存完整预测后退出−6，经独立验证后复用；不能称全部60个worker都正常退出。
- 旧P04/P06 LoRA学过这两栋，不能当无泄漏初始化；CS06/source1251不是P06/source1848。
- 60条Complex12轨迹有连续逐帧图复核，不声称实时高频闪烁已被量化或三维恢复已验证。
- 原生崩溃根因尚未定位；历史T2V完整预算结束仍缺独立证据。
- 现有1992个T2V缓存为.pth；共享文本context可复用的前提是同一prompt。
- 扫描器GUI源码缺失；Windows命令和旧Slurm流程不能当当前A6000操作。

#### 5. 本次文档更新与原有工作区

接手时已有历史T2V代码、诊断工具和若干Markdown的未提交改动。用户此次授权更新所有项目Markdown，因此相关文档纳入本次维护；已有非Markdown代码保持原样，不混入文档提交。

旧迁移页、扫描器README及早期VACE操作原文保留在[本次归档](docs/history/20261023_before_md_refresh/README.md)；上一轮四份根文档的字节级快照继续保留。冻结run报告、原始数据包说明、第三方文档和工具档案不改写为“当前状态”。

目录整理提交为ccfe24d。文档更新后的提交以实际git log为准；不把Git工作区HEAD当作既有run的唯一源码证明。

#### 6. 验证与下一步

CPU检查按TRAINING执行；原6466个成果文件和历史权重保护，不为更新文档重新计算大权重哈希或跑GPU。文档核对范围与本次实际检查结果记录在[文档索引](docs/DOCUMENTATION_INDEX.md)。

下一轮实验尚未授权。当前只交付已有结果与完整文档；报告中的配对多种子建议不能自动转为执行任务。
<!-- RETAINED_UPDATE_END -->

<a id="history-repair"></a>

## 三、2026-10-23：恢复原文件内的持续历史

用户明确要求长期MD保留“从以前到现在”的更新。先前仅保留当前摘要、让读者另找归档的处理不符合要求；本次将上述旧正文直接恢复到原文件，同时保留后续更新，没有回退实验结果或把完成状态改回执行中。

以后在本文件按日期追加阶段进展、失败与修复、参数变更、结论和核查范围。需要纠正旧结论时，说明原判断、后来的证据及新结论；原记录继续保留。可以在顶部维护简短状态和阅读索引，不能用摘要或归档链接替换历史正文。

本次修复依据原文件备份及Git记录恢复，没有补造新的运行经历；原数据、权重、视频、指标、冻结报告和现有非Markdown工作保持原样。此次检查与恢复范围见[文档索引](docs/DOCUMENTATION_INDEX.md)中的后续补记。

## 四、2026-10-24：upload报告已建立，后续会话继续遵守

本次补记采用UTC+09:00。用户要把每组实验的详细文字报告带回家分析。现已创建根目录 [upload/README.md](upload/README.md)，并完成 [Complex12 Rev1完整实验报告](upload/20261022_Complex12_Rev1_十二栋单轨实验报告.md)；该文件可独立阅读，关键参数、训练/验证表、逐栋观察和下一步建议都在正文。

Complex12实验本身仍是已完成状态：C/U/S各300更新、五方法60轨迹、4860 PNG、12视频与12总览。此次仅编写报告，没有启动GPU、复制大成果或向外部机器传输。原冻结结果位于outputs/ComplexSingleOrbit_Rev1_Pilot_20261022，报告不替代其证据。

接手应保留两个判断：普通LoRA的改善已经较大；结构加权S只有小幅平均收益，部分几何问题未解决、保留区和闭环有代价。下一项建议是多训练种子下的配对C/U/S复核，不是已经执行的新实验。B/CS04异常退出与C从update40恢复的原记录不能删。

以后每完成一组实验，在upload增加对应报告并更新索引，正文包括实际方案、执行、错误恢复、结果、局限与建议；同步四份根长期MD和实验README，继续按日期累积。未完成或失败也诚实记状态；不凭目录名自动上传，不凭建议自动开GPU。持续规则见AGENTS第10节；本次检查与Git记录见文档索引及实际提交。


## 2026-10-26：OriginalHard 已有权重评测启动快照

核查时间：2026-10-26T01:27:12+09:00。用户已授权对上传的 OriginalHard 使用已有 B/L/C/U/S 跑完整 81 帧，并要求“旧的相同的就不用跑了”。本轮新增训练更新为 **0**，C/U/S 沿用已完成的 Rev1 step300，L 为旧 P04/P06 参考。

新实验入口：[OriginalHard](experiments/vace/complex12_originalhard_eval/README.md)。真实 run：`/mnt/windowsE/chuanjun/building_wan/outputs/vace/complex12_originalhard_eval/run_20261026_012033`。源码位于 experiments/vace/complex12_originalhard_eval/scripts，旧大结果保持原位。

逐帧核对后，CS01–CS08 的 GT/Damaged/Exact/结构图及相机记录与旧版一致，五方法共40条预测已解码后链接复用；CS09–CS12 的缺损不同，新增20条推理。最终展示60条，其中40条复用、20条新推理。新缺损中的CS09/10/11来自旧训练身份，CS12来自旧验证身份；应单独汇总四栋及3/1子组，不能称为十二栋新建筑泛化。

此刻 runner PID 1678408 实际存活，tmux 为 complex12_rev1:originalhard81；状态 running，当前 infer_L_CS09，有效轨迹 41/60（复用40，新推理1）。这是一条运行中快照，不代表全部推理、媒体或视觉结论已经完成。

接手先查看该run的progress、tmux及实际PID，继续已有队列；不把终端读取超时当作失败，不重启活进程。40条旧预测是指向旧run的链接。GPU结束后仍需12栋媒体验证、新困难四栋完整轨迹复核、中文结论、upload详细报告、长期MD追加和相关Git提交。当前不要提前写complete或效果最好。


### 2026-10-26 后续交接：用户要求启动后即可结束对话

用户进一步明确“跑起来就行，就可以结束这轮对话了，不需要实时监控，我在vscode的tmux能看到过程即可”。01:31:02（UTC+09:00）核查时，runner与C/CS09 worker实际存活，C去噪17/30，40条复用、2条新增已完成；VS Code所连tmux客户端位于complex12_rev1的originalhard81窗口。

按此要求结束本轮主动跟踪，保留既有队列继续运行。队列包含剩余推理、统一指标、12栋视频/总览和复核图；计算结束状态为awaiting_visual_review。详细视觉分析、最终中文结论、upload实验报告和完成文档需下次接手核对真实产物后补齐，不能将本轮“已启动”写成“实验已全部完成”。长期文档的历史正文和旧实验产物继续保留。


## 2026-10-26：OriginalHard 完整轨迹复核与结果收尾

复核记录：2026-10-26T02:41:09+09:00（UTC+09:00）。本次接手已核对20个新worker退出0、40条复用，共60条81帧/4860 PNG，12视频/12总览和统一指标均完成；用户要求审查困难结构效果后，逐段查看CS09–CS12全部81帧联系图及六个原尺寸问题帧。新增训练更新0，本次没有启动新GPU任务。此前启动/待复核快照按原文保留。

run：`/mnt/windowsE/chuanjun/building_wan/outputs/vace/complex12_originalhard_eval/run_20261026_012033`。直接观看入口：[00_看结果](outputs/vace/complex12_originalhard_eval/run_20261026_012033/00_看结果/README_从这里开始.md)；完整可携带报告：[OriginalHard已有权重评测](upload/20261026_Complex12_OriginalHard_已有权重评测报告.md)。最终结论在run的00_看结果/最终实验结论.md，核查后progress更新为complete。完成指执行和报告完整，不表示补全质量全部合格。

CS09、CS11明显改善；CS10连接面/折线仍错，C的MAE最低；CS12错误折面及首尾形状差异明显，旧L反而最好。新缺损四栋B/L/C/U/S的MAE为0.069922/0.036078/0.036619/0.036443/0.035899。S比B降低48.66%，但比C/U只低1.97%/1.49%；S相对C保留区变化增加4.17%、全图闭环误差增加13.78%，不能称S稳定胜出。CS12的S首尾缺损区误差0.126121，L为0.029390。

身份不能混淆：CS01–CS08输入不变且五方法复用；CS09/10/11是旧训练建筑新缺损，仅CS12为旧验证建筑新缺损。不是12栋全新建筑泛化，不能用完整8/4均值掩盖困难4栋。视觉结论来自连续静态帧与原尺寸图，视频有程序全帧解码证据，本次未做实时播放或量化高频闪烁。

接手先看最终中文结论和upload完整报告。此前“补齐报告”的待办本次已处理；不要再次启动队列或重算已成功的推理。下一步尚未获GPU执行授权，最有信息量的是普通C的困难缺损覆盖对照，保持独立验证。最终检查及Git结果见文档索引和实际提交。

收尾实测：2026-10-26T02:42:50+09:00，progress=complete；60轨迹/4860 PNG与12总览可读，12视频完整解码证据齐全。298项CPU回归（57.214秒）及9项专用检查通过；详细检查见文档索引和run/logs/final_review.done.json。


## 2026-10-26：C_mix 固定配比实验启动与后续交接

核查时间：2026-10-26T22:26:41+08:00。用户授权一组普通C_mix的300更新、16条81帧推理及配对指标/视频，随后要求确认训练启动即可结束对话，视觉复核和最终报告等待另行通知。当前running，已记录33/300次真实优化器更新、推理0/16；这不是完成或效果确认。

实际run：`/mnt/windowsE/chuanjun/building_wan/outputs/vace/complex12_cmix_pilot/run_20261026_222008`。研究入口：[C_mix](experiments/vace/complex12_cmix_pilot/README.md)，观看入口：[00_看结果](outputs/vace/complex12_cmix_pilot/run_20261026_222008/00_看结果/README_从这里开始.md)。tmux为`complex12_rev1:cmix_pilot`（%2），runner PID 2179781，GDB worker PID 2179822。

从旧Rev1 shared_initialization及原始VACE开始，新AdamW，普通C损失与参数不变。旧300行顺序/窗口/随机字段保留；局部Random(20261026)按CS09/10/11抽20/14/16行Hard，合计Rev1 250/Hard 50。验证CS02/04/08/12任何版本不入训练。条件缓存按版本、建筑、窗口隔离；合法零scheduler权重计数、不重抽不补步。两版改变了缺损位置与连接线索，不是同一缺口单纯扩大。

队列依次完成300更新、Hard CS09–12和Rev1 CS01–12共16条/1296原始PNG、原指标与分组/逐栋新旧变化、Rev1退步项、32视频和六张固定问题帧。旧C结果复用；Hard CS01–08展示复用新Rev1；不重跑B/L/U/S。计算结束标记awaiting_visual_review，用户通知后再进行完整视觉复核、run中文报告和同内容upload报告。当前不提前写效果结论。

本轮9项专用CPU检查通过（0.658秒），298项项目CPU回归通过（50.410秒），规定shell语法/Python编译与差异检查通过；没有额外GPU试跑。旧run、数据、第三方引擎及19项无关工作区修改保持原样。冻结配置ID：`386098b4cc71d7a446b5d2dcba635e8a8a3142dcaa3e6bd8629299406fb0c545`。

下一次接手先核对真实progress、训练完成标记/版本计数、16个推理标记和评测退出，再做视觉复核。重点Hard CS10/B024、B058及CS12/B000、B080，同时检查Rev1逐栋退步。按用户要求本轮不持续监控到结束，最终报告等待通知；若失败则按真实原因修复，保留状态并接续，避免重复成功任务。


## 2026-10-27：C_mix 完整评测、视觉复核与收尾

接手先读[本轮完整报告](upload/20261027_Complex12_C_mix_混合训练与配对评测报告.md)及[结果入口](outputs/vace/complex12_cmix_pilot/run_20261026_222008/00_看结果/README_从这里开始.md)。真实run：`/mnt/windowsE/chuanjun/building_wan/outputs/vace/complex12_cmix_pilot/run_20261026_222008`。此前33/300与等待用户通知为历史快照；本次用户已经通知收尾，训练、评测与视觉/报告均已完成。

已完成一个C_mix的300更新（250/50）、16条81帧/1296PNG、32视频；Hard六张固定问题帧未替换，新增Rev1 CS10/B020。训练及16推理/CPU评估全部EXIT 0，无本run实际恢复/重试。全16条000–080静态连续帧复核完成；没有实时播放或三维/定量闪烁验证声明。

主要结论：CS10连接仍错（Hard MAE−4.07%、Sobel+12.06%）；CS12折面和首尾未解决（缺口首尾0.115625→0.117000）。Rev1有10/12栋MAE退步，CS10从0.009775升至0.127054并产生明显条纹。当前保留旧C参考，不能将像素均值小降写成结构恢复成功。

后续仅建议同入口纯Rev1复现对照，先确认CS10异常退步来源；本轮任务已停止，不自动执行建议、扩训练、增加预算或启动Depth。Hard只有50次样本更新且CS12是唯一未训练建筑，结论限于本次开发对照。

结果目录不搬动，旧run/数据/权重/原PNG不改；新run报告与upload报告同内容，长期MD追加保留全部历史。提交仅收尾相关Markdown，已有19项非Markdown工作区修改继续保留且不打包。最终CPU检查与Git记录见本节后续实测补记和实际Git历史。

### 本轮收尾检查实测（2026-10-27T01:46:25+09:00）

298项项目CPU回归通过（49.876秒），9项C_mix专项通过（0.706秒）；规定shell语法、Python编译、git diff --check通过。1296原PNG本次全部解码，32视频元数据核对，完整视频解码/同帧验证沿用实际生成记录；16条全81帧静态复核和七张原尺寸问题图核对已完成。无新GPU任务。

run与upload报告内容一致，新增链接有效；11份既有MD完整历史前缀保留。progress更新为complete，原评估日志的待复核快照保留；最终依据见当前run的logs/final_review.done.json。19项已有非Markdown工作不纳入本次提交。


## 2026-10-27：Orbit17线性/循环配对实验已启动

核查时间：2026-10-27T20:16:32+09:00（UTC+09:00）。用户已授权线性组和循环组都执行，并要求确认实际开始训练后结束本轮对话。当前status=running，C_lin17已完成22/600次真实AdamW更新，C_ring17为0/600（待串行执行），推理0/40；日志已有有限loss及参数实际变化。这是启动快照，不是两组完成或效果验证。

固定run：`/mnt/windowsE/chuanjun/building_wan/outputs/vace/complex12_orbit17_pair/run_20261027_200842`。专用源码与说明：[Orbit17实验](experiments/vace/complex12_orbit17_pair/README.md)；[结果导航](outputs/vace/complex12_orbit17_pair/run_20261027_200842/00_看结果/README_从这里开始.md)。tmux=`complex12_rev1:orbit17_pair`（%3），runner PID 2469437，GDB worker PID 2469491；源码提交`1a82bd5`。

已启动的同一队列包含：线性0→300、四条81帧复现探针与CPU提前核对、完整状态恢复线性301→600、循环0→600、循环300四条探针、两组600各16条正式预测，最后统一CPU评测和视频。提前核对保留复现证据，不因主观效果选择性取消循环组。总预算两组各600、40条81帧、3240原PNG；计划48视频、7固定问题图和4张七行探针总览。旧C两版原预测直接复用，无新增B/L/U/S/C_mix推理。

两组均只用Rev1训练8栋CS01/03/05/06/07/09/10/11，验证CS02/04/08/12不变；Hard只评测。普通C、原VACE骨干、共享初始LoRA与各自新AdamW，rank/alpha16、lr5e-5、17帧、BF16骨干/FP32 LoRA等原设置保持。线性窗起点0–63，循环窗起点0–79按模80取17帧，080不入训练；跨界先同步取完整17帧再编码。原线性前300行保留；后300行用局部Random(20261027)，循环起点另用Random(20261028)，其他随机字段逐行配对。合法零scheduler权重照常计数，不重抽补步。

本次实测14项Orbit17专项CPU检查通过（1.466秒），298项项目CPU回归通过（50.203秒），规定shell语法/Python编译及git diff --check通过；没有额外GPU试跑。准备时覆盖图首次因服务器缺少matplotlib失败，随后改用现有Pillow生成并通过；没有安装依赖，未重复GPU任务，prepare.log保留实际错误与后续成功。旧run、原始数据、模型、第三方源码及19项既有无关工作区修改保持原样且不纳入本轮提交。

每10更新保存完整恢复状态，300/600保存LoRA，300另保留恢复点；两组状态独立。缓存按版本/建筑/完整有序frame_ids/编码设置区分，放在新run。自动队列完成后标记awaiting_visual_review；之后按用户通知进行完整轨迹视觉复核、run最终中文报告及upload同内容便携报告。此时不能预先宣称循环优于线性，也不能把视频解码当人工播放或三维验证。新预算、配比或模块不自动展开。


## 2026-10-29：Orbit17配对实验复核与最终报告完成

接手从本节和[详细便携报告](upload/20261029_Complex12_Orbit17_线性与循环窗口配对实验报告.md)；[直接看结果](outputs/vace/complex12_orbit17_pair/run_20261027_200842/00_看结果/README_从这里开始.md)开始，之前启动时刻和C_mix结论保留。

计算于2026-10-27T23:41:29+09:00结束，用户随后通知复核/报告收尾；10月28日至29日完成本次复核。实际run为`outputs/vace/complex12_orbit17_pair/run_20261027_200842`。两组各600真实AdamW更新、40条81帧/3240原PNG、48视频、7固定问题图和4张最终七行探针总览齐全；另保留4张早期探针图与1张覆盖图。45个队列worker均单次启动、EXIT 0；线性300→600为计划内完整optimizer/RNG恢复，循环连续0→600，没有失败后回放或新增GPU任务。

同为600更新，ring对lin在16输入中MAE 13个改善、Sobel 10个改善，但known MAE 12个退步、全图首尾MAE 15个退步，Hard四栋缺口首尾全部退步。四组MAE变化分别为Rev1训练−7.73%、验证−2.68%、Hard训练身份三栋−11.98%、Hard验证CS12−12.56%；不能写成循环普遍解决闭环。

Hard CS12旧C/lin600/ring600缺口MAE为0.056055/0.023890/0.020890，主要错误折面明显缓解；缺口首尾为0.115625/0.017480/0.018376。两组600均改善，预算贡献不可归于循环独有。Hard CS10/B024连接仍错，B058局部有改善；ring相对lin MAE−17.07%仍不代表结构修复。Rev1整体均值改善，ring相对旧C仍有CS01/06/08/10四栋MAE退步；未见前轮C_mix CS10大面积条纹。lin300四探针对旧C的MAE差在±0.72%内，为近似复现，不是逐bit复现。

本轮方法保持原VACE普通C，仅比较纯Rev1线性/循环17帧，不能与C_mix的250/50训练身份混淆。最终权重在本run checkpoints/C_lin17/step600.safetensors和checkpoints/C_ring17/step600.safetensors；原预测、逐帧/逐栋、端点和分段CSV均在run。00_看结果优先Hard CS10/12、Rev1 CS10。

本次CPU完整解码3240原PNG、48条视频及16张展示PNG；全部40条000–080通过连续静态联系图复核（缩略内容），7张固定问题图按完整512内容检查。没有实时视频播放、定量闪烁或三维验证，不能把文件可读写成模型效果全达标。原evaluate.done的待复核状态与既有启动正文保留为历史，当前交付状态以progress和logs/final_review.done.json为准。

下一步仅建议第二训练随机序列的600更新线性/循环配对复验，检验像素收益与首尾代价是否可重复；尚未执行或排队。Hard未训练建筑仅CS12，是开发验证，不能夸大泛化。旧run、权重、原始数据/预测和已有报告保持不变；本次收尾不自动开启任何训练、推理或新模块。

本轮只追加相关Markdown并完成Git记录，19项此前无关代码/脚本修改继续留在工作区，不得在下次维护中顺带提交或重置。
