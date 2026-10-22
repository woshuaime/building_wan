# Session Handoff

更新时间：2026-10-22T18:59:02+08:00。Complex12 Rev1执行中，当前infer_B_CS05；5/60条输出有效，C/U/S正式更新0。CS04保存后曾原生异常退出，已独立验证81帧并复用；根因未确认，不能称退出正常或问题已修复。

## 当前：Complex12 Rev1十二栋单轨小试（执行中）

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

## 当前接手：八例完成，直接查看成果

- 实际结果根目录 `outputs/vace/lora300_eval81/run_20261020_194300_145372/`，01–08各四文件，共32项。原始视频512×512、比较视频2048×544，均81帧/16FPS/H.264 High/yuv420p/BT.709并完整解码通过；PNG可读，临时查看图已清理。
- 唯一权重 `outputs/vace/lora_roof_smoke/run_20261020_165237_150325_2048720/training/lora_update0300.safetensors`，vace分支、scale1。固定81帧512/16FPS/30步/CFG5/seed20261018及原正负文本；Damaged填灰127+原二值ModelMask，合法全黑帧保留，GT推理后才读，不回贴或调色。
- 先P04→P06，查看两例全部81帧后在终端明确确认改善成立，再按原授权完成P01/P02/P03/P05/P07/P08；八个worker与两个批次全部exit0。批次369.2+1129.4=1498.6秒，各例峰值allocated17.141/reserved25.832GiB。
- P04屋顶凹槽持续填平；P06黑洞在全81帧保持闭合，没有后半圈重开，侧向棱线仍轻微起伏。这是训练题扩展。六例迁移中P05/P07/P08明显改善，P01/P02有限，P03整体基本持平且部分视角变差；P05前半圈还有小凹点/首尾差异。已查看八例全81帧局部对照、总览和Mask图；详细帧号与统一解码口径指标在EXPERIMENTS.md。
- 真实终端为 `vace_p01_17f:4.0`，窗口名 `lora300_eval81`，pane `%4`，已经回shell；NVML无计算/图形进程。可只读重连查看历史，不要再次运行推理入口。八例Base全部复用，Base补跑0、新训练0。
- 原test/composite_check/小样本run及源包69项大小mtime未变。新媒体只在本轮独立目录；模型/视频/图片/临时文件不入Git。后续方向建议保持骨干、扩大建筑补全LoRA训练覆盖后再研究几何/Depth，但没有开启下一轮的授权。

## 历史：首次制图失败及获授权重跑

- 首次P04完成30/30步与解码，175.2秒、显存峰值17.141/25.832GiB；CPU四栏制图因PIL未转NumPy而TypeError，worker及批次退出1，未启动P06。临时目录清理导致该次原始预测丢失，不能用它评估效果。
- 修复统一像素类型并先保存最终01原始预测，增加 `--render-only` CPU补图模式；298项CPU测试通过（53.124秒），修复已提交推送 `ea51376`。用户回复“直接修复，然后再重跑”后才开始上述成功批次，无未授权自动重试。
- 本轮实际LoRA推理总计9次，包含最初失败制图后的P04一次及获授权重跑；最终成功交付8例。旧失败根目录未覆盖，成功结果使用独立时间戳子目录。相关GPU过程与异常均已保留在文档中。

## 持续约定：GPU运行后更新文档并Git记录

用户明确要求：每次用户启动GPU训练/推理之后，接手时主动核实运行事实，更新四份根入口文档及必要的专项说明，然后审查、提交并推送相关代码/配置/文档，不需要用户再次提醒。运行中只记带时刻的进度，成功/失败/中断均如实记录；不因记录工作自动另起GPU任务或续训。

GitHub身份已实测为woshuaime，origin为 `git@github-scj:woshuaime/building_wan.git`。模型/数据/结果媒体/日志不入库；其他历史未提交改动保留，不混入本轮提交。持续规则已写入AGENTS.md第7节。

## 最新接手：LoRA首轮完成，查看成果

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

## 已完成记录：八例缺损区域合成

- `outputs/vace/composite_check/01–08/` 各恰好三文件，八例SUCCESS、退出0、GPU推理0次；tmux `vace_p01_17f:vace_composite` 保留实际终端进度。使用 `scripts/composite_vace_results.py`，数据从只读Pilot8 ZIP按真实身份/帧号读取，ModelMask复用原推理 `adapt_images`。
- Composite只取白区原预测、黑区Damaged，未调色、羽化或修改Mask，GT仅对照与诊断选图。全黑Mask直接保留Damaged；两个视频81帧/16FPS完整解码通过，采用High/yuv420p/BT.709。原 `test/` 32文件SHA/大小/mtime全部不变，未改源数据，临时文件已清理，272项CPU测试通过。
- P01未损区域灰度和主体保持改善，但出现可见拼接细缝；P04屋顶仍有凹槽/台阶，重点看 `04/03_关键帧对比.png` 的B003/B066；P06黑洞仍在，重点看 `06/03_关键帧对比.png` 的B006/B020。其他样本也可见补丁边界。优先研究缺口内部结构生成，边界问题次之；当前结果不能证明三维一致性或所有运动形变已解决。
- 本次已停止，不自动开展Mask扩张、GPU推理、LoRA或新下载；不要因为原test和composite_check同存就删除/替换原预测。再次执行CPU入口会新建时间戳根目录，不应无故重复生成。

## 已完成记录：八例离线三方对比

- 八例离线重排和RGB播放偏色修复均已完成，P01–P08全部SUCCESS、退出0，GPU推理次数0；当前八个三栏视频已变为High/yuv420p/BT.709，各目录仍严格四文件。独立32文件比对证实只有八个02改变，01与两张PNG全SHA不变、GT/Mask无缺失、自有临时目录清理通过。完整266项CPU回归通过。下一步关闭旧视频预览后重新打开，避免旧媒体缓存，不再执行推理。
- 当前入口为 `scripts/reorganize_vace_results.py`；默认只读，`--apply`对旧布局进行CPU重排，已经升级的目录如果还是RGB 4:4:4则只修复02三栏视频；已为兼容格式则只验收。原始预测缺失即失败，绝不触发VACE。
- `test/01–08`的固定源对应465/763/871/1227/1259/1848/2297/939不变。原始视频保留；新版三栏视频、三行五列总览和重命名四栏图见PROJECT_CONTEXT。不得将GT或其他样本代替预测。
- CPU任务运行在已有 `vace_p01_17f` 的 `vace_layout` 窗口，真实进度和退出码在终端，不写运行日志或JSON。使用空CUDA_VISIBLE_DEVICES，入口还拒绝Torch/DiffSynth导入。
- 用户报告原RGB三栏播放紫绿偏色：旧文件实际为High 4:4:4 Predictive/gbrp，不适合当前播放器。原260项测试只证实FFmpeg解码像素，遗漏播放器兼容性。修复为libx264/High/yuv420p/BT.709有限范围，保护01与两个PNG字节及文件身份；新检查包括编码元数据、白色背景、三栏颜色/帧序和单视频替换失败回滚，不加载模型。

## 历史接手：保留01–03，仅恢复P04–P08（后续已成功）

- 用户下一条：`bash /mnt/windowsE/chuanjun/building_wan/scripts/run_vace_building_test.sh --resume --tmux`；在现有 `vace_p01_17f` tmux中会复用该会话并打印实际重连命令，不嵌套；新会话名仍 `vace_building_test`。只读复查加 `--resume --dry-run`，不能去掉resume直接覆盖重跑。
- 原因证据在tmux历史：P04 sample_preflight，`inspect_models(...verify=True)` 报 `Reused component SHA256 mismatch: text_encoder`，未进入该例模型加载。不是OOM，P02/P03各真实81帧成功，峰值17.14GiB allocated /25.73GiB reserved。当前没有仍运行的batch/worker。
- T5原件连续三次独立全SHA256均匹配批准 `d92de679881d38af9c89eff7bb1b6d6c9d96cb2b69831e4027e9ecabdd38eb23`，11,361,845,432bytes及inode/mtime/ctime未变。原失败未记录actual，故底层原因未确定，不能宣称磁盘/RAM/读取故障已修复。仍保留严格hash门禁，任何再失败立即停，不重试“碰巧通过”、不跳过验证或更改权重。
- 新 `verify_file_sha256` 打印expected/actual/path/bytes及前后stat，单次全读取并检测替换/修改。新 `vace_building_reuse.validate_completed_sample` 全解码四文件并逐像素查源身份、前后SHA不变；显式resume只接受连续成功前缀。parent/gate比对既有文件hash和待跑队列，之后重新复核才进入新worker。
- test/01–03各恰四文件，P01四SHA等于原成功目录，P02/source763和P03/source871的五视角Damaged、四栏Damaged/Mask/GT均对应。P01–P03不重新调用模型；队列仅P04→P08。04–08当前不存在。所有CPU测试与临时编码只在自有/tmp，不能冒充新预测。
- 本轮实际执行 `--resume --dry-run` 通过32.1s，输出 `verified completed prefix=['P02','P03']`、唯一队列P04–P08；全权重SHA、CUDA环境和七例CPU loader均通过。完整250项测试通过35.630s（新增16项既有结果验收、8项严格hash、20项恢复/终端路由），规定及新脚本语法/编译/diff通过。模拟resume验证不重算01–03、gate/parent hash一致、失败即停及native139、拒非连续/缺失/错号/软链接、清理自有/tmp、parent无Torch和终端参数传递。匿名gate事件UTF8编码最大七例四hash实测2928bytes，小于4096原子写上限。未启动GPU、修改权重/原始数据/历史结果或提交Git；自有临时目录无残留。

## 此前接手：P01复用，P02–P08批量准备历史

- 最终启动：`bash /mnt/windowsE/chuanjun/building_wan/scripts/run_vace_building_test.sh --tmux`。新建/重连可见 `vace_building_test`；已在 tmux 则复用并显示实际重连命令。断线重连用 `tmux attach-session -t vace_building_test`，不要另起七例。
- 新 ZIP 位于 `data/buildingfiller_import/P02_P08_20261020/BuildingFiller_Pilot8_AllRes_v1.zip`，SHA256 `7f8eff9ddb3c05e43fe5a8623258f9dceb6fa61d547b244c4f2d441099b4fcf4`。CRC/6077载荷哈希/5832PNG解码/相机帧号/几何/有效人工 accepted 均通过，未解压原包或修改数据。包含 P01，但只核查身份；`training_ready/generation_verified=false` 不是训练发布许可，`generation_evidence_verified=true` 不等于 VACE 补全成功。
- 固定输出顺序：`02=P02/763`、`03=P03/871`、`04=P04/1227`、`05=P05/1259`、`06=P06/1848`、`07=P07/2297`、`08=P08/939`。512 输入每角色81帧；P02黑 Exact023–049，P03为000–016和071–080，后五例无黑帧，各自保留。
- P01 原结果 `outputs/vace/P01_465_81f_20261019_171510_1691517/` 仍保留；两视频81帧/16FPS及两PNG有效，源465图像面板身份通过。已实际执行新入口 `--prepare`，复制到 `test/01` 后再次检查四文件哈希与原件一致。`test/02–08` 还不存在；本轮不生成这些预测。
- 历史 P01 无 manifest，无法从视频反推历史 seed/权重/prompt。保留执行代码和冻结配置与本轮固定参数一致，用户确认原始81帧成功；不能声称媒体独立证明了全部历史参数。
- 原始正式 VACE、本地 T5/VAE/Tokenizer 完整哈希、CUDA/BF16/高效SDPA、无其他GPU进程、空间权限、全部七例实际CPU loader 合同已通过。启动时完整门禁再做一次；parent 不初始化 CUDA，七个 fresh worker 逐个调用同一81帧核心，GT020仅预测后读取；无 LoRA、offload回退、分段或参数更改。
- `scripts/buildingfiller_pilot8.py` 负责严格包验收和内存适配；`scripts/vace_p01_reuse.py` 验证/安全复制旧结果；`scripts/vace_building_test.py` 只调度 P02–P08。原始17帧JSON SHA仍为 `7e82161e270d8a24e231feb2ca95d08ae6e02e861ce5460a241730bfbcc07080`，没有修改。
- 每个成功样本严格两个视频+两PNG，名称沿用原四文件；CPU测试假数据只在隔离/tmp。只清理本轮自有系统临时目录，失败即停止并保留此前成功输出，绝不伪成功。若启动后部分成功，入口会拒绝再次覆盖已有02–08；需要另行确认恢复策略，不能直接自动续跑。
- 本轮最终完整 `scripts/tests` 206项通过（35.474s），新增14项交接包、13项P01复用、29项批量/共享核心/终端路由CPU测试；规定及新脚本语法、编译、`git diff --check`通过。正式入口实际执行 `--help`、`--prepare`、复制后的 `--dry-run`，均成功；未执行GPU分支。首轮有一个既有GT隔离用例的MP4文件字节哈希断言波动，未复现根因；测试改为更直接的全部81帧解码像素一致、总览hash一致、四栏前三栏不变/GT栏变化后，目标连续3次及完整套件通过；没有改编码器或推理逻辑。最终仅test/01四文件，无本轮系统临时目录残留，两个原ZIP及冻结JSON哈希未变；未提交Git。

## 历史接手：P01 81 帧、只留四文件（现已完成）

- 用户命令：`bash /mnt/windowsE/chuanjun/building_wan/scripts/run_vace_p01_465_81f.sh --tmux`。已有 tmux 会直接复用当前会话并打印实际重连命令；否则新建/重连 `vace_p01_81f`，避免重复创建实验；未安装 tmux 时前台运行、不安装。
- 81 帧预检已实际只读通过：官方模型/共享组件完整哈希、原 ZIP/81 组相机与 Damaged/Exact、原 27 黑 Mask、当前 DiffSynth CPU loader、CUDA/BF16/高效 SDPA、无其他 GPU 进程、输出空间/权限。只输出 PASS/FAIL，没有保存新的审计或 CPU 派生文件。
- 代码复用旧适配器/模型检查/本地模型加载/视频验收；旧 17 帧配置不改，只在内存派生 81 帧。一次 GPU worker、一次全序列 pipe 调用；GT B020 在预测完成后才读，B080 不复制预测 B000。
- 新 ConsoleState 仅内存/终端，真实 scheduler tqdm + VAE 阶段。匿名 pipe 给父进程传真实阶段和峰值，不落盘。父进程不创建 CUDA 上下文，避免被子进程 NVML 误认其他任务；所有预检在 worker 或显式 dry-run 内执行。
- 必要四个编码文件位于独占 `/tmp/building_wan_vace81_*`，父进程监督 native/中断并在正常失败清理；发布期 TERM/HUP 也清理。只复制四个已验收文件到全新最终目录，未用危险通配符删除其他结果。若发布遇磁盘写入失败，可能留下本次部分最终文件，必须 FAILED，不谎称四文件成功。
- 成功后只有两个 81 帧/16FPS MP4 和两张 PNG：总览000/020/040/060/080、四栏020。无 GT 回贴/美化/三维质量推断。显存峰值仅正式运行后才能报告，发生 OOM 必须停下，不自动重试或更改规格。
- 本轮验证：新脚本 `--help`/`--dry-run` 实际通过；`git diff --check`、规定及新增脚本 `bash -n`/`py_compile` 通过，完整 `scripts/tests` 150 项通过（新增46项81帧CPU测试）。真实合成81帧双视频编码/解码、GT隔离、B080、四文件名单、tmux路由、native/中断/发布期信号清理均覆盖。`outputs/vace` 仍无新子目录，系统本次临时目录无残留，原ZIP与17帧冻结JSON哈希未变；未运行GPU模型或提交Git。

## 2026-10-20 17 帧准备历史（旧输出不再可用）

- 官方正式文件现已存在，7,146,067,848 bytes，完整 SHA256 `c46a6f5f7d32c453c3983bbc59761ea41cd02ad584fb55d1a7ee2b76145847a2`；T5/VAE/4 个 Tokenizer 同样完整校验。不要再次下载，也不要误用下方 10-19 的缺权重历史状态。
- 新审计 `outputs/vace_model_audit/P01_465_20261020_20261019T161614Z/` 包含完整模型预检、实际环境与官方依据；新 CPU 准备 `outputs/vace/P01_465_17f_baseline_cpu_prepare_20261020_20261019T162331/` 已通过 CPU loader。报告时间采用 UTC，所以时间戳可能为 10-19；会话日期为 10-20。
- `scripts/run_vace_p01_465_17f.sh` 默认前台 `--run`，只有用户亲自执行才授权一次真实实验；`--dry-run` 只完整哈希 + CPU loader，不推理。tmux 已安装、未自动创建会话。终端启动/重连/分离命令见 [执行说明](docs/VACE_P01_EXECUTION.md)。
- 真实去噪进度由 `progress_bar_cmd` 按实际 30 步计数，VAE encode/decode 包装只记录真实阶段，vendor 自带 tile 进度不改。字节流实时转发同时落 `run.log` 和 `worker.log`；中断只停止自己的子进程组。
- 输入别名为 `input_damaged/`、`input_modelmask/`、`input_condition/`、`gt_reference/`，指向本次独立副本；真实输出 `raw_prediction/` 和 MP4，兼容旧 `prediction_raw` 相对别名。GT 不进 worker，评估在真实 worker 成功之后才运行。
- 独立结果模块验证 17 PNG、视频实际解码、FPS16/512，再生成三张指定图、可选同步 MP4、metrics/validation。只有全流程保存并验收完成才打印 SUCCESS；失败保存实际退出码/阶段/日志，报告写入失败也必须 FAILED。
- 原始模型真实加载、峰值显存和补全质量尚未实测；没有预测图或质量达标结论。未改冻结 JSON、旧训练/权重/缓存、第三方核心，未提交本轮或其他既有未提交工作。
- 阶段 A 验证：`git diff --check`、规定训练/续训与新启动/下载脚本 `bash -n`、规定及全部新增 Python 文件 `py_compile` 通过，`scripts/tests` 104 项全部通过。CPU 合成 fixture 测试覆盖实时无换行/中文输出、中断只终止自己的子进程、GT 隔离、真实 MP4 编解码与错误帧数/FPS、报告写入失败不误报成功。正式启动脚本只执行了 `--help` 与 `--dry-run`，未执行默认 GPU 分支。

## 2026-10-19 P01 / VACE 准备历史

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

## 当前事实

- 工作副本 `/mnt/windowsE/chuanjun/building_wan`，分支 main；审计前 HEAD 为 `8a3720e`。后续以 `git status` / `git log` 的实际结果为准，不把旧提交号当成已同步状态。
- scj Python `/home/shi/miniconda3/envs/scj/bin/python` 已确认 PyTorch 2.11.0+cu128、CUDA 12.8、CUDA 可用；历史硬件为单张 RTX A6000。
- 单轨视频 2344；train/val/test 为 1992/234/118，统一 prompt。1992 缓存和共享 context 存在。
- 平铺单轨权重 `checkpoints/building_wan_a6000_single_orbit_all/` 共 28 份，400–11200；A100 历史 building_wan 39 份独立保留。
- 全部单轨对比视频已经生成，29 个完成元数据及文件位于 `outputs/validation/validation_all_single_orbit/`。不用默认重新生成。
- “8 epochs 正常完成”是待核实的历史报告：现存 loss 末步 11562、最高权重 11200、按保存参数推算预算末步 13536，日志缺少完整终态。不可自动补训或补造结束标记。
- GUI 采集 `src/` 和两份数据集工作流文档缺失；不能声称扫描器完整可运行。

## 本次修复

- `scripts/wan_a6000_state.py`：从 checkpoint 段参数计算累计样本，换算不同 batch，识别本项目进程；支持可追溯的旧记录路径迁移、v2 暂停记录、所选进程停止和续跑预检。
- `resume_wan81_a6000.sh`：修复平铺 step-2400 默认路径，加入 `--dry-run`、参数/缓存校验、新顶层输出和 launcher 锁。
- `pause_wan81_a6000.sh`、`resume_from_pause.sh`：使用上述公共工具，正确传递 WAN_PYTHON；不再无条件报告训练成功。
- `train_wan81_a6000.sh`：默认独立输出，序列化启动器，拒绝覆盖未验证缓存。
- `wan_train_entry.py`：正常返回并最终保存后写 `training_completion.json`；logger step 仍保持 batch 口径。
- Python 验证入口：只读预检、固定划分与 prompt 检查、参数/来源一致时复用完成视频。
- `analyze_validation_videos.py`：v2 全图前景占比与二阶像素变化，写独立新目录，旧结果原样保留。
- 冻结划分生成脚本拒绝静默覆盖；增加 `scripts/tests/test_a6000_tools.py`。

## 接手顺序

先读 AGENTS.md、PROJECT_CONTEXT.md、TRAINING.md、EXPERIMENTS.md 和本文件；检查 Git 状态、实际环境、进程及产物。详细证据和验证见 `docs/PROJECT_AUDIT_20261014.md`。

优先核实训练终态、源 OBJ 防泄漏及已有视频效果；逐栋 prompt 或补全使用独立实验。不要直接运行历史 A100/Slurm 或 Windows 命令。续跑是权重与样本预算续跑，不恢复 optimizer/RNG、梯度累积或 shuffle 顺序。

## 保护与交接

data/models/cache/已有 checkpoints/outputs 默认只读；历史数据和权重未改。只有本轮新建串行测试产物按用户要求删除，见下方清单。原有未跟踪日志、验证产物和 diffsynth.egg-info 保留。未调用 Claude worker，没有自动提交、推送或 pull；需要同步时遵循 AGENTS.md 的非强制 main 流程。

## 复审补修与验证

- 先解析 args.run 再加锁，验证 --ru/--r 与 --run 相同，预检不获取锁。
- 短 batch 会随预分组 shuffle；来源与目标 batch 存在短组时，只在完整 epoch 边界允许推断游标，其他情况明确拒绝。不要恢复旧的“短尾总在最后”公式。
- 暂停兼容等号、分隔参数、空格和相对路径。
- scj 缺 pidfd API；通过系统 Python 控制 pidfd，核对项目命令及启动时间，避免向复用 PID 发信号。真实 CPU 测试已验证记录、身份拒绝和停止；未停止任何正式训练。
- 早期审计的新工具 23 项与既有训练工具 12 项均通过，无跳过；新增测试覆盖真实 CLI 的锁冲突与真实 CPU 进程生命周期。随后获授权的 GPU 诊断见下一节。

- 最后预检曾在原 PyTorch 权重头部读取路径退出时出现 native 内存释放错误；当前头部检查改用 NumPy 后端，并验证预检不导入 torch，错误参数必须正常以退出码 1 拒绝。此改动不证明正式训练的原生崩溃根因已修复。

## 2026-10-15：GPU 诊断已结束，禁止自动继续

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

## 用户要求的串行产物清理

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

## 清理后的代码检查

git diff --check、5 个项目 shell 入口 bash -n、训练/验证入口和诊断工具
py_compile 均通过。scripts/tests 的 36 项测试与既有训练工具的 12 项测试
全部通过，无跳过；只使用 CPU 和临时测试进程，没有启动 GPU 训练或推理。
