# VACE执行：准备、故障、完成与持续更新

最近补记：2026-10-23（UTC+08:00，北京时间）。本文件持续保留P01模型准备、CPU检查、真实启动、八例Base复用与故障处理的完整过程。历史正文已恢复在原文件内，后续进展按日期追加；归档仅作备份，无需离开本文件才能阅读旧记录。

阅读顺序：[此前累计正文](#history-records) → [2026-10-23阶段更新](#update-20261023) → [本次历史恢复及生效约定](#history-repair)。旧段落的“当前”“下一步”、命令和PID按其记录时点理解；较晚的更正与实际完成证据在后文，旧命令不因此重新执行。

截至本次修复：Pilot8系列及Complex12的C/U/S各300更新、五方法60条81帧已完成。本次仅修复文档；[实验入口](../experiments/README.md)和[文档索引](../docs/DOCUMENTATION_INDEX.md)仍可直接使用。

<a id="history-records"></a>

## 一、此前累计正文：此次整理前长期保留的内容

以下完整保留原有记录顺序、参数、命令、过程和当时结论，只调整标题层级。原记录中的未完成状态与后来完成状态共同构成历史，不能用后者替换前者。

<!-- RESTORED_HISTORY_BEGIN -->
### BuildingFiller VACE 正式权重与受控执行入口

#### 当前：P04/P06条件LoRA首轮正常完成

2026-10-20当前任务为两例训练题的17帧过拟合验证。配置 `configs/vace_roof_smoke_v1.json` 固定P04/P06均B000–B016，原VACE条件填灰/Mask规则和文本不变；GT仅作训练target及评估，不进入推理条件。详细说明见 [P04/P06 LoRA实验](VACE_P04_P06_LORA_SMOKE.md)。

17:05:32北京时间已核查：run `run_20261020_165237_150325_2048720` 完成300更新、6次17帧推理，两份权重及04/06各四成果齐全，各阶段退出0。总耗时749.247秒，关键帧中两例均明显更接近完整GT，视频时序及用户人工验收仍待确认。重连 `tmux attach-session -t vace_p01_17f`、选择窗口3可看完成记录；当前不需要再次启动。完整操作见根目录TRAINING.md。

#### 已完成：八例CPU离线重排

P01–P08真实81帧结果已通过完整解码和源身份核对，后续批量成功记录已确认。当前入口为 `scripts/reorganize_vace_results.py`，默认只读，显式 `--apply` 将已有比较升级为三栏视频和三行总览；原始预测保持字节不变，四栏验证图核验后仅改名，始终每例四文件。见根目录TRAINING.md。不得重新调用GPU模型来改变展示布局，也不得用GT代替预测。

#### 历史：P02/P03已成功，显式恢复P04–P08（后续已完成）

2026-10-20用户实际启动后，P02/P03各完整81帧成功；P04在 `sample_preflight` 检查文本编码器SHA256时失败，未开始该例模型推理，不是OOM。当前01–03均已再次完整解码验收及源身份逐像素验证，P01四SHA等于保留原件，不覆盖/重跑这三例。

T5权重连续三次独立全SHA复查均匹配批准值，11,361,845,432bytes及inode/mtime/ctime稳定；原始失败没有打印actual，当前无法复现根因。不能认定磁盘/RAM故障已修复，也不因后来通过就自动绕过检查。现增加单次完整hash前后身份对比及expected/actual/path/bytes信息，任何不符立即停止，不自动重试或更换权重。

用户接下来在VS Code SSH终端执行：

```bash
bash /mnt/windowsE/chuanjun/building_wan/scripts/run_vace_building_test.sh --resume --tmux
```

只读恢复预检用 `--resume --dry-run`。显式resume会验收02起连续成功前缀，要求每例严格四文件、视频81帧/16FPS、PNG布局及Damaged/Mask/GT源身份一致，并在parent/gate及加载新样本前对比四项SHA不变；只有当前P04→P08进GPU队列。GT读取属于对已有预测的后验复核，不进入新模型条件。

无效/部分/错号/非连续/软链接或未知结果均拒绝，不自动删除、修补或覆盖。普通首次启动（无resume）仍拒绝已有02–08。成功前缀终端标REUSED，真实失败仍立即停止；不要重复执行无resume的旧命令或重跑P01–P03。本轮只做排错和CPU预检，没有启动新GPU推理。

已实际执行 `--resume --dry-run` 通过32.1s，明确复用P02/P03、唯一队列P04–P08；模型全哈希、环境、输入CPU合同均通过。完整250项CPU测试和规定语法/编译/diff检查通过；测试只用/tmp模拟模型，不能当成剩余五例已成功。01–03原件和模型、数据、冻结配置均保持不变。

#### 此前：P01 已复用，P02–P08 首次批量准备

2026-10-20 已完成新交接包验收、P01结果校验与复制、剩余七例完整CPU预检。没有启动新的GPU推理。用户在VS Code Remote-SSH终端执行：

```bash
bash /mnt/windowsE/chuanjun/building_wan/scripts/run_vace_building_test.sh --tmux
```

启动后先全七例统一门禁，再顺序运行P02→P08，绝不把P01放入GPU队列。使用当前tmux或新建/重连可见 `vace_building_test`，断线后 `tmux attach-session -t vace_building_test`。模型子进程继承终端，不后台静默运行；真实30步tqdm、VAE调用阶段、当前例数、实际耗时、实测allocated/reserved峰值及基于已完成样本耗时估算的批量ETA直接显示，不落日志。

##### 已接收数据和固定映射

新包：`data/buildingfiller_import/P02_P08_20261020/BuildingFiller_Pilot8_AllRes_v1.zip`，41,547,426 bytes，SHA256 `7f8eff9ddb3c05e43fe5a8623258f9dceb6fa61d547b244c4f2d441099b4fcf4`。sidecar、ZIP CRC、6077载荷哈希、三分辨率共5832张PNG解码、1944帧CSV/相机和24OBJ绑定均通过；不依ZIP名称猜身份、不执行包内代码、不解压到原数据。元数据固定如下：

| 输出 | Pilot/source | 类型 | 512原始全黑Exact帧 |
|---|---|---|---|
| 01 | P01/465 | A1 Medium | 046–072；仅复用既有结果 |
| 02 | P02/763 | A1 Medium | 023–049 |
| 03 | P03/871 | A1 Medium | 000–016、071–080 |
| 04 | P04/1227 | A2 Medium | 无 |
| 05 | P05/1259 | A2 Large | 无 |
| 06 | P06/1848 | A3 Medium | 无 |
| 07 | P07/2297 | A3 Large | 无 |
| 08 | P08/939 | A4 Medium | 无 |

八例有效人工验收均accepted，源身份与相机绑定完整。旧sample_meta的pending/not_run是快照，由hash-bound验收索引/事件和已验证的几何渲染证据确定当前人工状态；`generation_evidence_verified=true` 不等于 `generation_verified/training_ready=true`，后两者仍false，不据此宣称训练发布或模型补全成功。

##### P01 真实结果和复用范围

原目录 `outputs/vace/P01_465_81f_20261019_171510_1691517/` 的两个视频均81帧/16FPS，原始512×512、对比1024×544，两PNG正常读取；总览Damaged五视角及四栏Damaged/ModelMask/GT与原源465ZIP逐像素一致。已执行 `--prepare`，验收七例后仅复制这四个文件到 `outputs/vace/test/01/`；原件保留，四项SHA256一致，已有有效01后续只读复用。

旧实验按四文件要求没有manifest；不能从媒体独立反推历史seed/权重/prompt。用户确认P01原始81帧成功，保留的实际执行代码和冻结配置与请求参数一致，没有发现差异。本轮继承该代码/固定配置，而非从旧T2V文档推测文本；也不声称媒体证明了全部历史参数。

##### 逐字沿用的模型条件

正向prompt：

```text
白色背景中一栋完整的灰色无纹理建筑三维模型，规则连续的墙面和屋顶，保持建筑形状与相机轨迹一致，固定俯仰视角平滑环绕，结构稳定，光照一致。
```

负向prompt：

```text
彩色纹理，窗户装饰，新增建筑，建筑形变，结构闪烁，背景变化，文字，水印
```

两者直接取自未改字节的 `configs/vace_p01_465_17f_v1.json`，内存改成81帧。所有七例均原始正式VACE-1.3B、不加载LoRA、512×512、16FPS、seed20261018、30步、CFG5、fullBF16/tiledVAE。与P01入口共享 `execute_sample` 完整序列核心，一次pipe调用81帧，B080正常进入、预测不复制B000。Exact二值0/255复制为RGB，白区生成、Damaged对应白区填RGB127，不翻转/膨胀/制造缺口；各自黑帧保持黑。

##### 运行门禁和四文件发布

正式权重及T5/VAE/Tokenizer全文件哈希、当前DiffSynth2.1.5 CPU loader81帧、A6000CUDA/BF16/高效SDPA已实际通过；预检时空闲47.13GiB、无其他GPU任务，项目盘358.8GiB可用。24GiB启动阈值不是保证无OOM；遇资源或兼容性问题立即停止，不分段/降规格/改offload/重试。GT PNG完整性验收不构成模型输入，推理worker只读取Damaged/Exact；模型返回后才单独读取GT020制作四栏。

- 只读复查：同一脚本加 `--dry-run`，不建结果目录或加载模型。
- `--prepare`：所有检查通过后仅复制/复用P01，已经执行成功，无需再次执行；不调用模型。
- 正式父进程不初始化CUDA，先在独立gate子进程做完整检查，退出释放上下文后再串行fresh worker；共享GPU锁，遇其他任务不擅自杀进程。
- 固定 `outputs/vace/test/01–08/`，每成功目录严格 `01_补全结果.mp4`、`02_补前补后对比.mp4`、`03_补前补后总览.png`、`04_四栏验证.png`。两个视频完整解码81帧，原始无GT回贴/美化；五视角000/020/040/060/080，四栏020。
- 自有 `/tmp/building_wan_vace_remaining_*` 编码验证后才发布四文件，正常成功/失败清理，仅匿名pipe传状态，无JSON/日志/输入副本/预测PNG。SIGKILL或断电不能保证Python清理，不得因此通配删除其他临时内容。
- 已存在的02–08目录或未知test内容一律拒绝，不覆盖/删除/静默跳过。单例成功后才运行下一例，错误立即停止、保留此前成功结果；磁盘发布失败可能留下本例部分四文件，必须FAILED而不是伪成功。部分成功后的恢复需另行确认，不能重新启动覆盖已完成样本。

最终终端逐项报告P01 REUSED、P02–P08真实SUCCESS/FAILED/NOT RUN、绝对输出位置和四文件名单。技术成功不代表正式定量验收或三维补全质量通过。

本轮实际 `--help`、`--prepare`、复制后 `--dry-run` 通过，完整206项CPU测试通过（新增56项）、规定语法/编译/diff检查通过；原ZIP/冻结JSON及P01原件哈希未变，test只有01四文件，无本轮系统临时残留。没有执行 `--run` 或隐藏GPU worker；不得将CPU假模型测试当成其余七例真实结果。

#### 历史：P01 81 帧极简视觉验证（已完成，不再执行）

用户报告17帧已成功且效果基本符合预期，旧实验全部输出已由用户删除；不依赖或恢复其日志/manifest/视频，不把用户报告写成独立复核结论。本轮只完成阶段 A，未运行 GPU。

```bash
bash /mnt/windowsE/chuanjun/building_wan/scripts/run_vace_p01_465_81f.sh --tmux
```

用户运行才启动一次81帧实验。已有 tmux 复用当前会话并打印实际重连命令；否则新建/重连可见 `vace_p01_81f`，不创建重复后台任务。`Ctrl+b` 后 `d` 分离，新会话用 `tmux attach-session -t vace_p01_81f` 重连。不带参数同样脚本直接前台执行；`--dry-run` 则只检查、不加载模型或创建输出。

81帧入口复用既有模型检查/本地加载/Exact适配/CPU接口与视频验收。原17帧 JSON 原字节保留，只在内存将源范围改000–080、长度81及输出策略；512/16FPS/seed20261018/steps30/CFG5、文本/BF16/tiled VAE等不改。不进行时间分段、offload回退、降分辨率、追加实验或T2V训练。

##### 阶段 A 实际检查

- 官方 VACE 7,146,067,848 bytes、SHA256 `c46a6f5f7d32c453c3983bbc59761ea41cd02ad584fb55d1a7ee2b76145847a2`，公用组件完整哈希与本地双分支结构通过，无重复下载。真实 GPU 加载仅用户运行时发生。
- 原 ZIP SHA、81组 Damaged/Exact/相机、原二值 Mask、046–072共27全黑、当前 CPU loader `[1,3,81,512,512]` 均通过。没有读取 GT RGB 作预处理；GT020仅真实模型输出之后读取。
- A6000空闲47.13GiB、BF16可用、NVML无其他GPU进程，项目盘358.8GiB可写，视频依赖和tmux已有。81=`4n+1`、512为16倍数，长度不变；latent时间21、tokens21504，约17帧的4.2倍。
- 默认高效 PyTorch SDPA 后端启用；未使用额外 FlashAttention 包或安装依赖。显存峰值未实测，24GiB启动门槛不是无OOM承诺；如果只有数学后端/显存不足或实际OOM，明确失败后停止，不静默改变配置。

本轮完整测试150项通过，其中新增46项81帧CPU测试；覆盖真实合成81帧双视频编解码、GT隔离与B080、四文件验收、防覆盖、匿名状态/真实退出码、系统临时目录与发布期信号清理、tmux前台路由。所有合成fixture只在系统临时目录，不是模型结果。规定语法/编译/diff检查通过；正式脚本仅执行`--help`和`--dry-run`，没有新的永久实验输出或临时残留，原ZIP和冻结配置哈希未变。

##### 四文件与清理

最终目录 `outputs/vace/P01_465_81f_<UTC时间>_<PID>/` 已有即拒绝。成功必须恰有以下四个有效普通文件：

1. `01_补全结果.mp4`：模型原始81帧，512×512/16FPS，无GT回贴、无美化，B080来自prediction[80]。
2. `02_补前补后对比.mp4`：Damaged与原始预测左右同步，81帧/16FPS，单面板512原比例，仅加标题栏。
3. `03_补前补后总览.png`：000/020/040/060/080五列，第一行Damaged，第二行对应原始预测，不缩放/裁剪。
4. `04_四栏验证.png`：020的Damaged、ModelMask、原始预测、GT，尺度一致。

所有输入、派生Mask和预测帧留内存，不写输入/GT副本、预测PNG、log/JSON、metrics或额外视频。只在父进程独占 `/tmp/building_wan_vace81_*` 目录编码四个待交付文件；两MP4完整解码验证81帧/16FPS及尺寸、两PNG实际解码、名单恰四个后才发布，最终目录再验收。

父GPU监督进程没有Torch/CUDA上下文，不会被worker占用门禁误判为其他训练。worker stdout/stderr继承终端，真实去噪tqdm在实际步返回后增长，VAE真实调用阶段/耗时直接显示；匿名pipe仅传内存状态和实际GPU allocated/reserved峰值，不写状态文件。SIGINT/TERM/HUP或原生worker退出都不冒充成功，并只停止自己的进程组；自有系统临时目录正常退出/失败清理，不删除其他数据。SIGKILL/断电无法保证Python清理；不得自动通配清理其他/tmp内容。

如果复制最终文件时磁盘写入失败，可能保留本次部分最终文件，必须显示FAILED/停止阶段/实际退出码，绝不报告四文件成功。技术成功也不等同三维质量验收。只有本次用户明确启动的81帧一次模型调用，没有自动重试或追加81帧实验。

以下为17帧阶段的历史说明；其中产物路径不再代表文件仍存在，也不是当前的输出要求。

#### 历史状态：2026-10-20 17 帧阶段 A

用户已下载正式文件，本轮重新完整读取并验证：7,146,067,848 bytes，官方 SHA256 `c46a6f5f7d32c453c3983bbc59761ea41cd02ad584fb55d1a7ee2b76145847a2`，1264 张量含 439 个 VACE 张量，连续偏移覆盖文件。T5/VAE/Tokenizer 完整哈希全部通过。无需再次下载；旧冻结 JSON 的缺失状态字段仅代表冻结当时，原参数字节不变，实际路径/模型身份记录在本次运行文件中。

实际证据目录为 `outputs/vace_model_audit/P01_465_20261020_20261019T161614Z/`；当前 CUDA 空闲约 47.1 GiB，NVML 无其他计算或图形进程，项目盘约 359 GiB 可用。`nvidia-smi` 不在 PATH，可靠进程门禁改用已有 NVML 库；查询不可靠/发现其他 PID 则停止，不自动杀进程。tmux 3.2a、ffmpeg/ffprobe 及依赖已存在，未安装升级。

真实 CPU 准备目录 `outputs/vace/P01_465_17f_baseline_cpu_prepare_20261020_20261019T162331/`，完整模型预检和当前 DiffSynth CPU loader 均通过。阶段 A 没有模型加载或 GPU 推理；所有 CPU 合成视频测试只在临时目录，不是模型结果。

本轮 `git diff --check`、规定及新脚本的 `bash -n`/`py_compile` 通过，完整 `scripts/tests` 104 项通过。正式前台脚本已实际执行 `--help` 与 `--dry-run`，后者完整模型哈希和 CPU loader 通过；未执行默认 GPU 分支。报告写入失败、中断、原生异常、GT 隔离、实际视频编解码及 Before/After 面板对应均有 CPU 回归。

##### 用户在 VS Code Remote-SSH 终端执行的一条命令

```bash
tmux new-session -s vace_p01_17f 'bash /mnt/windowsE/chuanjun/building_wan/scripts/run_vace_p01_465_17f.sh; status=$?; printf "\nLauncher exit code: %s\n" "$status"; exec bash'
```

该命令会立即进入可见 tmux 会话；脚本在前台执行一次完整实验，结束后保留 shell 和终端输出。Codex 不代为启动。若已在会话中，单独执行 `bash /mnt/windowsE/chuanjun/building_wan/scripts/run_vace_p01_465_17f.sh`。

- 分离查看：`Ctrl+b`，松开后按 `d`；不是 `Ctrl+c`。
- SSH 重连后：`tmux attach-session -t vace_p01_17f`。同名会话已存在时先重连，不要重开或终止其他会话。
- 没有 tmux 的机器可直接前台执行启动脚本，不安装新软件；断线可能中断。
- 只读预检：`bash /mnt/windowsE/chuanjun/building_wan/scripts/run_vace_p01_465_17f.sh --dry-run`。此分支完整读取模型哈希并检查 CPU loader，不创建实验输出或加载权重。

##### 真实进度与可核对的退出

启动脚本使用 scj Python `-u -B`；模型子进程 stdout/stderr 按字节实时合并转发，而不是等到换行才转发，镜像保存 `run.log` 与 `worker.log`。无需 `tee`；显式启用 tqdm，日志管道不会因非 TTY 而隐藏真实进度。

`progress_bar_cmd` 接入当前 DiffSynth 实际 scheduler 的 30 步循环，只有一次真实 body 返回后才增加 completed_steps；VAE encode/decode 实例包装只报告真实调用，已有 tiled VAE 的进度也可见。`stage_events.jsonl`/`stage_state.json` 保存真实时间与最后完成阶段，不以定时器伪造进度。不改第三方核心。

新目录为 `outputs/vace/P01_465_17f_baseline_run_<UTC时间>_<PID>/`，已有目录拒绝，沿用冻结输出根目录。GPU 锁与 NVML 占用门禁、完整模型哈希、实际落盘输入 CPU 合同检查均在正式加载前执行。每个命令最多调用一次模型，不自动重试、不降帧/降分辨率、不自动改 offload。默认 full/BF16 的 24 GiB 门槛不是不 OOM 保证；实际峰值仍待首次实验。

父进程仅在真实 worker exit0、没有中断且 worker 完成标记存在后调用独立评估。任何原生退出、OOM、格式/输出数量/解码/写入错误都 FAILED，保留日志与失败现场；SIGINT/SIGTERM 只终止本次新建子进程组。只有原始 PNG、视频实际解码、比较图、评估和报告全部完成才打印 `VACE P01 17-frame inference: SUCCESS`。质量未达标仍保留真实输出，技术执行成功不等于补全质量成功。

##### 输出与图像说明

- 独立输入副本位于 `inputs/damaged/`、`inputs/exact/`、`inputs/model_mask/`、`inputs/conditioning/`；对应别名 `input_damaged/`、`input_modelmask/`、`input_condition/`。`gt_reference/` 是 `evaluation/gt/` 的相对别名，worker 不读取 GT；oracle Mask 明确披露。
- `raw_prediction/`、`raw_prediction.mp4` 为真实模型输出；旧 `prediction_raw` 名称仅为同一原件的兼容相对别名。只有显式 `--composite` 才另存 `postprocessed_prediction/` 和 MP4，默认前台脚本不合成。
- `comparison_before_after_B020.png`：source020/output015 的 Damaged 与原始预测左右对照。`comparison_before_after_grid.png`：005/010/015/020，同列对应上下两行。`comparison_4col_B020.png`：Damaged、ModelMask、原始预测、GT。
- 三图保持每个原面板 512×512，不缩放或裁剪，仅增加 32px 英文标题栏。不会借 GT 改变 After；可选 `comparison_before_after.mp4` 同步 16 FPS，使用已有依赖生成。
- `metrics.json`：Mask 内/外的原始预测对 GT、Damaged 对 GT 基线、原始预测对 Damaged 保留变化，像素加权 MAE/PSNR、白背景阈值轮廓 IoU、未运动补偿的相邻帧差。不是三维质量，也不能把相机运动直接等同闪烁；没有自动质量接受阈值。
- `results_validation.json`：17 PNG 及输入哈希/映射、MP4 的 ffprobe FPS 与 ffmpeg 完整解码17帧/512尺寸、比较图哈希。失败不伪造输出或用 GT 替代。
- `run_config.json`、`runtime.json`、`run_manifest.json`、`exit_status.json`、环境/来源/显存/阶段日志和 `result_summary.md` 保存身份、原参数、真实退出和耗时；结束时只打印实际存在的主要产物绝对路径，失败至少显示原因及日志位置。新输出分区完全不可写/写满时持久化可能失败，仍向终端明确 FAILED，不声称保存成功。

以下为 2026-10-19 的权重来源及早期接口核查历史；其中“缺权重/等待下载”不再是当前状态。

日期：2026-10-19。本轮只读核查模型、访问官方元数据及小范围文件头，并完成 CPU 数据准备和检查。没有下载完整模型、安装依赖、启动 GPU 推理或旧 T2V 训练。

#### 1. 本轮结论

唯一必须新增的大文件是原始正式版 `Wan-AI/Wan2.1-VACE-1.3B` 的 `diffusion_pytorch_model.safetensors`。文本编码器、VAE 和 Tokenizer 均可复用本项目已有文件。

新增独立入口 `scripts/vace_p01.py`，默认只读预检，`--prepare` 仅 CPU 准备，`--run` 才进入 GPU 子进程。现有实验 JSON 的 SHA256 固定为 `7e82161e270d8a24e231feb2ca95d08ae6e02e861ce5460a241730bfbcc07080`，本轮没有改动任何冻结参数。

当前仍未有通过官方哈希检查的 VACE 正式权重，所以没有实测模型加载、峰值显存或补全效果。代码入口和真实数据的 CPU 准备已通过检查，不等于 GPU 推理已验证。

#### 2. 正式文件、兼容性和空间

官方来源：[Hugging Face 模型仓库](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B/tree/main)、[ModelScope 模型仓库](https://modelscope.cn/models/Wan-AI/Wan2.1-VACE-1.3B)。冻结 HF revision：`574e6a744642ce3bee319afc31496b88bde8aac4`。

| 文件 | 官方原件大小 | 本轮是否需要下载 |
|---|---:|---|
| `diffusion_pytorch_model.safetensors` | 7,146,067,848 bytes，约 7.15 GB / 6.66 GiB | 是，唯一必需新增大文件 |
| `models_t5_umt5-xxl-enc-bf16.pth` | 11,361,920,418 bytes | 否，复用已有转换版 safetensors |
| `Wan2.1_VAE.pth` | 507,609,880 bytes | 否，复用已有 BF16 转换版 |
| `google/umt5-xxl/` 四个 tokenizer 文件 | 合计约 21.45 MB | 否，复用现有文件 |
| `config.json` / `LICENSE.txt` / `README.md` | 很小 | 可选来源记录；本地加载器不依赖新增 config.json |

官方 VACE 文件 SHA256：

```text
c46a6f5f7d32c453c3983bbc59761ea41cd02ad584fb55d1a7ee2b76145847a2
```

[官方文件详情](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B/blob/main/diffusion_pytorch_model.safetensors) 与 ModelScope API 对该大小/哈希一致。完整原始运行组件约 19.04 GB；本轮复用组件只新增约 7.15 GB。项目盘 `/mnt/windowsE` 当前可用约 366 GiB；根分区约 64 GiB。下载器要求至少两倍文件大小的空闲空间，建议预留 20 GB，目标使用项目盘。

远程官方文件头只读取约 130 KB：1264 个张量，其中 439 个 VACE 张量；825 个主干张量为 FP32，VACE 分支为 BF16。结构 MD5 为 `a61453409b67cd3246cf0c3bebad47ba`。本地 DiffSynth 2.1.5 对这个结构注册了 `wan_video_dit` 和 `wan_video_vace` 两个加载分支，维度 1536、30 层、12 heads、VACE 输入 96 通道。

注册表旧注释仍写 Preview，但本次实际核对的是正式版官方文件头，结构确实匹配；不能因此改用 Preview 参数或其他权重。普通 T2V DiT 只有 825 个键，结构/文件哈希不符，执行器会拒绝。[DiffSynth 官方示例](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/wanvideo/model_inference/Wan2.1-VACE-1.3B.py) 也选择上述正式模型。

##### 已有组件的实际核验

| 本地组件 | 大小 | SHA256 |
|---|---:|---|
| `models/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors` | 11,361,845,432 | `d92de679881d38af9c89eff7bb1b6d6c9d96cb2b69831e4027e9ecabdd38eb23` |
| `models/Wan-Series-Converted-Safetensors/Wan2.1_VAE.safetensors` | 253,806,246 | `a506bb08599a262f843f9c13c9a8c17e4ebdb07cc0e3612e19f79f29f9db1669` |

本轮实际完整读取并计算了两个本地文件的 SHA256，与官方 [DiffSynth 转换组件仓库](https://modelscope.cn/models/DiffSynth-Studio/Wan-Series-Converted-Safetensors) 的文件元数据一致；本地结构也被当前加载器识别。Tokenizer 文件哈希固定在 `configs/vace_p01_model_sources_v1.json`，其中两个大型 tokenizer 文件直接匹配 VACE 官方仓库 LFS 哈希。运行时仅传本地 `ModelConfig(path=..., skip_download=True)`，不会自动重新下载公用组件。

#### 3. 扩展搜索发现与拒绝原因

搜索了当前账号可访问的 `/mnt/windowsE`、`/mnt/windowsD`、USB 挂载点、`/home/scj`、`/home/shi` 和 `/opt`，遍历约 83,712 个目录；跳过 Git、包环境和编辑器缓存。`/home/shi`、`/opt/containerd` 无权限，一些无关视频目录有文件系统错误，不能宣称整机无遗漏。

本轮 `/mnt/windowsD` 已挂载服务器的 `/dev/sda2`，与 10-18 的空目录状态不同；仍不能据此假定它就是 Windows 主机 `D:\P01_465`。

找到候选：

```text
/mnt/windowsE/szq/HuggingFaceWeights/Wan/Wan2.1-1.3B-VACE/diffusion_pytorch_model.safetensors
```

它可读且具有相同 1264/439 张量结构，但只有 7,146,067,816 bytes，SHA256 为 `fcd12e2a8fab331ba0ab28dca371d90036a868287767f35677ff3f8e74db0969`，不等于官方文件。

该文件比官方少 32 bytes，确实缺少官方的格式 metadata；**差异不只 metadata**：将官方文件头与该本地张量载荷在内存中组合后计算 SHA256，得到 `c04bbc3bf5da73dcc8cbc22774f036feee0791bfd4fbfd518f1bd918666c4efc`，仍不匹配官方。因此不能证明它是原始正式模型，不纳入允许列表，也没有修改或复制此文件。原因未确认，不能直接认定其损坏或属于某次微调。

另发现 58 个 VACE 训练检查点，包括仅分支和 merged 模型，属于其他实验，不用于原始基线。

来源证据与官方元数据保存在新的 `outputs/vace_model_audit/P01_465_20261019/`。正式来源和允许的文件哈希固定在 `configs/vace_p01_model_sources_v1.json`。

#### 4. 网络与授权后的下载

服务器实际访问结果：HF 模型元数据 HTTP 200，权重下载端点 HEAD 200 并转到 `us.aws.cdn.hf.co`，文件头 Range GET 206；ModelScope 模型和转换组件元数据 HTTP 200。说明当前能连接官方渠道，不代表已经下载或验证了 7 GB 长连接的稳定性。

新增下载脚本默认只打印计划，不联网、不创建模型目录：

```bash
cd /mnt/windowsE/chuanjun/building_wan
WAN_PYTHON=/home/shi/miniconda3/envs/scj/bin/python bash scripts/download_vace_p01.sh --dry-run
```

**用户批准下载后，下一条执行命令：**

```bash
WAN_PYTHON=/home/shi/miniconda3/envs/scj/bin/python bash scripts/download_vace_p01.sh --download
```

固定目标为 `models/Wan2.1-VACE-1.3B/574e6a744642ce3bee319afc31496b88bde8aac4/`。已有同名目录即拒绝；先保存 `.partial`，核对完整大小和官方 SHA256 后才改为正式文件名。失败保留现场，不覆盖旧文件、不自动安装 curl、不启动推理。本轮只运行了 `--dry-run`。

##### Windows 下载与 SCP 备选

如果服务器下载失败，先保留具体错误，再由实验室 Windows PowerShell 执行以下命令。此方案未在 Windows 实测，本轮未执行。使用全新目录，Host 别名从现有 SSH 配置填写，不猜 IP、端口或账号。

```powershell
$ErrorActionPreference = 'Stop'
$Revision = '574e6a744642ce3bee319afc31496b88bde8aac4'
$FileName = 'diffusion_pytorch_model.safetensors'
$ExpectedHash = 'c46a6f5f7d32c453c3983bbc59761ea41cd02ad584fb55d1a7ee2b76145847a2'
$WeightDir = 'D:\VACE_P01_weights_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
if (Test-Path -LiteralPath $WeightDir) { throw '下载目录已存在，禁止覆盖' }
New-Item -ItemType Directory -Path $WeightDir | Out-Null
$WeightFile = Join-Path $WeightDir $FileName
& curl.exe --fail --location --retry 3 --output $WeightFile "https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B/resolve/$Revision/$FileName"
if ($LASTEXITCODE -ne 0) { throw '下载失败，保留现场；不要直接覆盖重跑' }
if ((Get-Item -LiteralPath $WeightFile).Length -ne 7146067848) { throw '权重文件大小错误' }
if ((Get-FileHash -LiteralPath $WeightFile -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ExpectedHash) { throw '官方 SHA256 不符' }
$ServerAlias = Read-Host '输入已正常工作的 SSH Host 别名'
if ($ServerAlias -notmatch '^[a-zA-Z0-9][a-zA-Z0-9._-]*$') { throw '请输入单个现有 Host 别名' }
& ssh.exe -G $ServerAlias
if ($LASTEXITCODE -ne 0) { throw '无法解析现有 SSH 配置' }
$RemoteParent = '/mnt/windowsE/chuanjun/building_wan/models/Wan2.1-VACE-1.3B'
$RemoteDirectory = "$RemoteParent/$Revision"
& ssh.exe $ServerAlias "mkdir -p -- '$RemoteParent' && mkdir -- '$RemoteDirectory'"
if ($LASTEXITCODE -ne 0) { throw '远程目录已存在或不可写，禁止覆盖' }
& scp.exe $WeightFile "${ServerAlias}:$RemoteDirectory/"
if ($LASTEXITCODE -ne 0) { throw 'SCP 失败，保留现场' }
& ssh.exe $ServerAlias "printf '%s  %s\n' '$ExpectedHash' '$RemoteDirectory/$FileName' | sha256sum -c -"
if ($LASTEXITCODE -ne 0) { throw '服务器权重 SHA256 验证失败' }
```

这只复制新下载的权重，不接触 BuildingFiller 原件或使用 Xftp。Windows 方案不需要安装新 Python 环境。后续 Linux 执行器仍会独立验证大小、SHA256 和分支结构。

#### 5. CPU 准备与真实数据检查

默认预检：

```bash
/home/shi/miniconda3/envs/scj/bin/python -B scripts/vace_p01.py --dry-run
```

默认没有 Torch/模型导入，没有 GPU 锁、下载或输出写入。缺正式权重或尚未进行完整权重哈希检查时返回 2；它不表示已经可运行 GPU。

只在新的输出目录准备输入：

```bash
/home/shi/miniconda3/envs/scj/bin/python -B scripts/vace_p01.py --prepare \
  --output-dir "outputs/vace/P01_465_17f_baseline_cpu_prepare_$(date -u +%Y%m%dT%H%M%S)"
```

本轮已实际生成的目录为 `outputs/vace/P01_465_17f_baseline_cpu_prepare_20261019/`，不要拿该已有目录重跑。包含：

- `frozen_config.json`：原 JSON 的逐字节快照，含固定 prompt、seed、步数等；`frame_map.json` 保存源 005–021 与输出 000–016、相机及原图哈希。
- `inputs/damaged/`、`inputs/exact/`：所选原 PNG 的字节不变副本。
- `inputs/model_mask/`：独立 RGB 三同通道 0/255，无膨胀、反相、缩放；`inputs/conditioning/`：Damaged 白区填 RGB127，黑区逐像素不变。
- `model_input_manifest.json`：仅有条件帧与 ModelMask 路径；`evaluation/gt/` 单独保存 GT，不传给模型或参考图。
- `model_preflight.json`、`adapter_report.json`、`prepare_report.json`、`artifacts_sha256.json`、`provenance/`：来源、转换版本、模型缺失项、哈希、CPU 环境与代码版本。
- `CPU_loader_contract.json`：从实际落盘文件重新读取并调用当前 DiffSynth 的 CPU 预处理；图像与 Mask 均为 `[1,3,17,512,512]`，Mask 为 0/1，白区条件归一化值为 `-1/255`，检查通过。

没有创建 prediction 目录或伪造预测。原 ZIP/Exact/Damaged 均未修改，GT-oracle Mask 的事实保留。

#### 6. 权重就绪后的受控 GPU 命令

先只读核验所有实际组件的完整哈希，不生成预测：

```bash
/home/shi/miniconda3/envs/scj/bin/python -B scripts/vace_p01.py --dry-run --verify-models
```

如正式权重位于其他位置，用 `--checkpoint /实际官方文件路径` 指定，仅改变本地路径，不修改冻结实验 JSON。文件必须匹配官方大小/SHA256和结构。完整哈希将顺序读取约 18.8 GB 的本地组件，冷盘可能需要数分钟。

**另外获得用户 GPU 推理授权后才执行：**

```bash
/home/shi/miniconda3/envs/scj/bin/python -B scripts/vace_p01.py --run \
  --output-dir "outputs/vace/P01_465_17f_baseline_run_$(date -u +%Y%m%dT%H%M%S)"
```

入口固定使用 005–021、512×512、16 fps、seed 20261018、30 步、CFG 5，且不加载 LoRA。`--run` 始终完整核对模型哈希，复用 `.wan_a6000_gpu.lock`，检查旧项目训练进程；输出目录已存在时拒绝。只向模型传落盘的 conditioning/ModelMask，参考图为 None，没有 GT 初始化。

所有权重和 tokenizer 均为本地路径，禁用下载。加载后必须存在 DiT、VACE、T5、VAE 四个组件。模型在单独子进程运行，父进程保留 GPU 锁并保存真实退出码：SIGSEGV/SIGABRT 即使已有完成标记也判失败；没有完成标记也不能判成功。

保存 `runtime.json`、`environment.json`、`stdout.log`、`stderr.log`、`exit_status.json`、原始 PNG 的哈希和 `peak_memory.json`。Python 异常时尽可能保存失败显存；原生崩溃时以父进程退出记录和已有文件为准。完整原始 PNG 先写 `prediction_raw/`，再生成仅供观看的 MP4。仅显式加 `--composite` 才另写 `prediction_composited/`，不能覆盖原始预测。worker 不读取 evaluation/gt。

#### 7. 显存与停止线

本轮 CUDA 探测为 A6000 总显存约 47.4 GiB，空闲约 47.1 GiB。按官方文件张量 dtype 换算，四组件统一 BF16 后的参数约 14.8 GiB，尚需激活、注意力、VAE 和 CUDA 工作区；17 帧的 5 个潜时间位置、32×32 patch 网格约 5120 tokens。当前资源有尝试空间，但没有实测峰值或成功生成证据。

默认 `--memory-mode full`，要求启动时至少 24 GiB 空闲；该门槛不是“不 OOM”的保证。若不足/OOM，记录错误后停止，不自动缩小帧数、分辨率或改精度。可由用户另外选择 `--memory-mode cpu-offload`，使用现有框架的 BF16 CPU offload，门槛 12 GiB，仍使用相同冻结参数和新的输出目录；此模式也未实测 GPU，不自动回退、不量化或升级依赖。

当前停止在正式权重下载和 GPU 推理授权之前。下载脚本仅做了 dry-run，GPU 执行路径仅做代码和 CPU 测试检查。

#### 8. 本轮验证记录

- `git diff --check`、训练/续训/下载脚本 `bash -n`、训练/验证/VACE 入口及新测试 `py_compile` 通过。
- `python -B -m unittest discover -s scripts/tests -v`：51 项全部通过，其中新增 15 项 VACE CPU 测试，覆盖帧号/哈希、Mask 语义、GT 隔离、模型缺失拒绝、项目锁冲突、原生退出误判和无完成标记拒绝。子进程失败用 CPU mock 测试，没有运行真实 GPU。
- 真实落盘输入哈希复验、已有目录拒绝且内容不变、只读预检不导入 Torch、冻结 JSON 原字节、源 005–021 检查通过。项目模型仍为 9 个文件，旧缓存仍为 1992 项；没有正式 VACE 新权重或预测目录。
- Windows 命令未在 Windows 执行，完整下载、真实模型加载、显存峰值和补全质量仍待分别授权后验证，不能把 CPU 测试当成模型补全成功。
<!-- RESTORED_HISTORY_END -->

<a id="update-20261023"></a>

## 二、2026-10-23：阶段更新与当前资料补全

下面继续保留357b09f阶段补充的结果、参数和操作说明。该阶段曾把历史移到归档、部分原文页改成导航；这一文档组织做法已由第三节的用户更正取代，实验事实和既有检查结果保留。

<!-- RETAINED_UPDATE_BEGIN -->
### VACE模型、条件与既有执行入口

更新：2026-10-23（UTC+09:00）。正式VACE-1.3B权重已在本机，Pilot8及Complex12正式实验均已完成。本页不再重复列出已经执行的“首次启动/补跑”步骤；完整早期过程见 [更新前记录](history/20261023_before_md_refresh/docs/VACE_P01_EXECUTION.md)。

#### 1. 当前模型和环境

| 组件 | 实际位置或版本 |
| --- | --- |
| 项目 | /mnt/windowsE/chuanjun/building_wan |
| Python | /home/shi/miniconda3/envs/scj/bin/python，3.10.13 |
| PyTorch / CUDA runtime | 2.11.0+cu128 / 12.8 |
| DiffSynth / PEFT / safetensors | 2.1.5 / 0.17.1 / 0.7.0 |
| VACE模型 | Wan-AI/Wan2.1-VACE-1.3B |
| revision | 574e6a744642ce3bee319afc31496b88bde8aac4 |
| 主权重 | models/Wan2.1-VACE-1.3B/<revision>/diffusion_pytorch_model.safetensors |
| T5 | models/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors |
| VAE | models/Wan-Series-Converted-Safetensors/Wan2.1_VAE.safetensors |
| Tokenizer | models/Wan2.1-T2V-1.3B/google/umt5-xxl |

批准模型身份来自 [model_sources](../configs/vace_p01_model_sources_v1.json)。主文件7,146,067,848 bytes，包含1264个张量、439个VACE张量。原正式身份校验已完成，本次没有重新下载或做全量权重哈希。

复用T5/VAE/Tokenizer不等于加载T2V骨干或旧T2V LoRA。原始VACE与每个LoRA阶段采用fresh模型进程，避免融合叠加；不能用clear_lora假定撤销已融合权重。

#### 2. 已完成的实验与范围

| 实验 | 适用条件与状态 |
| --- | --- |
| P01早期17帧 | 原B005–B021；旧结果曾由用户删除，不补造日志或恢复输出 |
| P01及八例Base81 | 81帧完整条件补全；[结果与编号](../experiments/vace/pilot8_base/README.md) |
| 八例Composite | CPU硬边界回贴诊断；[独立说明](../experiments/vace/pilot8_composite/README.md) |
| P04/P06 LoRA17 | B000–B016、300更新、六次17帧推理；[完整参数](VACE_P04_P06_LORA_SMOKE.md) |
| 八例LoRA81 | 同一LoRA-300、八例81帧、Base全复用；[观察结论](../experiments/vace/pilot8_lora81/README.md) |
| Complex12 Rev1 | 独立8训练/4验证、C/U/S各300、五方法60轨迹；[专属说明](../experiments/vace/complex12_rev1/README.md) |

Pilot8统一使用中文正向描述；Complex12使用冻结的英文正向描述。两者均512×512/16FPS/30步/CFG5/推理seed20261018，但不能因此声称所有文本与训练条件完全相同。

#### 3. 数据与条件合同

Exact为原始L模式0/255，转换相同RGB通道的ModelMask。Damaged是原始RGB，Mask白区填灰127、黑区不变。推理条件只包含这些派生图、Mask和文本；GT/结构标签仅监督或后验评价。

全黑Mask表示缺口在该视角不可见，仍保留该帧。不扩Mask、不羽化、不改变Exact、不把GT贴回预测。81帧一次生成，080始终是模型的真实输出。

VACE条件编码使用inactive/reactive冻结VAE latent及64通道Mask，形成96通道上下文。缓存冻结条件编码与缓存可训练VACE输出是不同操作；后者会破坏LoRA训练链。冻结DiT参数也必须保留向LoRA传梯度的计算路径。

#### 4. 默认命令行为

| 入口 | 默认行为/正式开关 | 只读或CPU分支 |
| --- | --- | --- |
| [P01 17帧](../scripts/run_vace_p01_465_17f.sh) | 无参数启动推理 | --dry-run预检 |
| [P01 81帧](../scripts/run_vace_p01_465_81f.sh) | 无参数启动；--tmux可见会话 | --dry-run预检 |
| [八例Base](../scripts/run_vace_building_test.sh) | 无参数运行；--resume复用有效成功前缀 | --dry-run预检，--prepare可能复制P01 |
| [LoRA17](../scripts/run_vace_roof_smoke.sh) | --run/--tmux启动 | 无参数/--dry-run为CPU准备 |
| [LoRA81](../scripts/run_vace_lora300_eval81.sh) | --run-trained/--run-transfer启动 | 无参数/--dry-run预检，--render-only制图 |
| [Complex12 status](../experiments/vace/complex12_rev1/scripts/status.py) | 只读已有状态 | 不需要运行resume.sh |

这些入口描述现有实现，不构成新一轮执行授权。部分预检会完整读取大模型，不能代替轻量状态查询；CPU准备也可能写文件。当前所有VACE任务均完成，优先打开结果。

#### 5. 媒体、异常与完成证据

Base结果布局曾由两栏升级为Before/Base/GT三栏，原始预测保持；LoRA81为Before/Base/LoRA/GT四栏；Complex12为Before/GT/B/L/C/U/S的2×4布局。同步帧号和主体比例一致，媒体可读与原始预测正确对应是完成条件之一。

历史Base P04在推理前T5哈希检查异常，后续授权继续；LoRA81首次P04推理后CPU制图异常，授权修复重跑；Complex12 B/CS04保存完整预测后原生退出−6，经独立核验保留。三种情况不同，不能合并写成“全部首次成功”或“全部是OOM”。

Complex12 C在服务器挂载恢复后从40真实更新恢复到300；U/S各300，五方法60轨迹完成。旧P01未保留完整manifest时不从视频推断seed；所有历史限制和原生故障根因仍保留。

#### 6. 维护与当前操作

配置、输出、训练代码和历史事实分别查 [configs](../configs/README.md)、[outputs](../outputs/README.md)、[scripts](../scripts/README.md) 与 [实验台账](../EXPERIMENTS.md)。当前终端状态、失败处理和规定CPU检查统一见 [TRAINING](../TRAINING.md)。

既有tmux名称和PID只是当时记录，不能保证重启后仍存在。本次没有创建终端任务、调整环境、重新训练或推理；原run报告和冻结配置不因文档整理改写。
<!-- RETAINED_UPDATE_END -->

<a id="history-repair"></a>

## 三、2026-10-23：恢复原文件内的持续历史

用户明确要求长期MD保留“从以前到现在”的更新。先前仅保留当前摘要、让读者另找归档的处理不符合要求；本次将上述旧正文直接恢复到原文件，同时保留后续更新，没有回退实验结果或把完成状态改回执行中。

以后在本文件按日期追加阶段进展、失败与修复、参数变更、结论和核查范围。需要纠正旧结论时，说明原判断、后来的证据及新结论；原记录继续保留。可以在顶部维护简短状态和阅读索引，不能用摘要或归档链接替换历史正文。

本次修复依据原文件备份及Git记录恢复，没有补造新的运行经历；原数据、权重、视频、指标、冻结报告和现有非Markdown工作保持原样。此次检查与恢复范围见[文档索引](../docs/DOCUMENTATION_INDEX.md)中的后续补记。
