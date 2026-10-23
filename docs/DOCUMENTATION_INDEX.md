# 项目Markdown总索引与更新记录

后续修复说明（2026-10-23）：本索引第1–7节保留第一轮维护记录；用户随后要求长期MD必须包含完整历史，因此已将原文恢复到原文件，详见[第8节](#8-2026-10-23后续修复历史回到原文件更新继续追加)。当前阅读无需依赖归档才能看到旧内容。

更新日期：2026-10-23（UTC+09:00）。本次按用户要求全面维护项目说明，统一到Pilot8与Complex12已经完成的真实状态，并补全数据身份、参数、结果入口、异常、结论和操作边界。

## 1. 范围与阅读顺序

更新前共识别75份Markdown：32份项目维护文档、24份历史/工具快照、12份第三方文档、1份受保护数据/模型类说明、6份run或数据交付记录。**本次更新32份维护文档**，新增本索引和归档索引，逐字归档9份旧说明；其余43份受保护文档不改写。

“全面更新”指当前项目自有说明完整一致，不把第三方README、原始数据说明、运行时报告和历史日志改成新的实验事实。没有修改训练预算、算法、冻结配置或运行结果，没有启动GPU、安装依赖或调用外部worker。

推荐顺序：[项目首页](../README.md) → [当前状态](../PROJECT_CONTEXT.md) → [实验目录](../experiments/README.md) → 对应实验README和原始结果。接手操作另看 [TRAINING](../TRAINING.md) 和 [SESSION_HANDOFF](../SESSION_HANDOFF.md)。

## 2. 根目录六份文档

| 文档 | 职责 | 本次补全 |
| --- | --- | --- |
| [README](../README.md) | 项目入口 | 当前研究问题、最新结果、实验关系与目录地图 |
| [PROJECT_CONTEXT](../PROJECT_CONTEXT.md) | 当前事实 | 完成量、配置身份、训练/验证、环境和遗留事项 |
| [EXPERIMENTS](../EXPERIMENTS.md) | 全实验台账 | Pilot8链条、Complex12完整训练/验证表、配对变化和异常 |
| [TRAINING](../TRAINING.md) | 操作手册 | 只读状态、入口实际默认行为、三类恢复、CPU检查与Git |
| [SESSION_HANDOFF](../SESSION_HANDOFF.md) | 交接 | 最新成果、唯一源码路径、历史限制及工作区保护 |
| [AGENTS](../AGENTS.md) | 项目规则 | 当前Complex12状态、文档分工与历史规则适用范围 |

## 3. 实验层九份文档

| 文档 | 本次补全 |
| --- | --- |
| [实验总目录](../experiments/README.md) | 七类实验关系、完成量、结果导航和新想法流程 |
| [新实验模板](../experiments/_template/README.md) | 数据身份、对照、预算、恢复、指标、证据和结论要求 |
| [Pilot8 Base](../experiments/vace/pilot8_base/README.md) | 八例编号、全黑Mask、逐字文本、参数及最终四文件 |
| [Pilot8 Composite](../experiments/vace/pilot8_composite/README.md) | 为什么做回贴、实际公式、三个成果及诊断结论 |
| [Pilot8 LoRA17](../experiments/vace/pilot8_lora17/README.md) | 两例选窗、300更新、权重、误差、耗时与验收边界 |
| [Pilot8 LoRA81](../experiments/vace/pilot8_lora81/README.md) | 唯一LoRA、Base复用、授权补跑、八例效果和启动行为 |
| [Complex12 Rev1](../experiments/vace/complex12_rev1/README.md) | 12源身份、损失映射、配对设计、指标、逐栋观察、耗时/退出和源码 |
| [历史T2V单轨](../experiments/t2v/single_orbit/README.md) | 1992缓存、28权重/29视频、batch/step口径与终态限制 |
| [历史早期T2V](../experiments/t2v/early_experiments/README.md) | Windows/双3060/A100参数、现存成果与缺失权重边界 |

## 4. 公共目录五份文档

| 文档 | 本次补全 |
| --- | --- |
| [configs](../configs/README.md) | 四类配置来源、单轨清单、冻结路径与文本差异 |
| [scripts](../scripts/README.md) | 公共工具地图、GPU/CPU默认行为和验证职责 |
| [outputs](../outputs/README.md) | 原成果映射、最新观看入口、新run子目录职责 |
| [Python_3D_Scanner](../code/Python_3D_Scanner/README.md) | 现存内容、缺失GUI源码与历史设计的区别 |
| [training](../code/Python_3D_Scanner/training/README.md) | 工具职责、14个历史配置、三类划分和缓存/步数语义 |

## 5. docs专题与兼容页十二份

| 文档 | 本次补全 |
| --- | --- |
| [docs首页](README.md) | 当前/专题/历史阅读导航 |
| [目录规范](PROJECT_STRUCTURE.md) | 公共/实验/run分层、已有路径映射、文档权威入口与Git |
| [P01数据交接](BUILDINGFILLER_P01_VACE_HANDOFF.md) | 已接收事实、数据身份、验收证据、条件和后续使用 |
| [VACE执行说明](VACE_P01_EXECUTION.md) | 已到位权重、组件、实验状态、入口行为及不同异常 |
| [P04/P06专项](VACE_P04_P06_LORA_SMOKE.md) | 保留完整参数/证据，明确17帧历史与后续81帧的边界 |
| [Claude worker](CLAUDE_WORKER.md) | 本地实现、未验证模型名、显式委派与凭据设置行为 |
| [原生故障记录](NATIVE_STABILITY_20261014.md) | 保留历史正文，增加日期、范围及当前衔接 |
| [旧项目审计](PROJECT_AUDIT_20261014.md) | 保留审计与测试数字，区分旧任务和本次文档工作 |
| [旧PROJECT_CONTEXT入口](PROJECT_CONTEXT.md) | 指向根当前状态，原文归档 |
| [旧EXPERIMENTS入口](EXPERIMENTS.md) | 指向根台账，原文归档 |
| [旧PROGRESS入口](PROGRESS.md) | 只读当前状态入口，原文归档 |
| [旧HANDOFF入口](HANDOFF.md) | 指向根交接，原文归档 |

## 6. 历史与原始成果保护

[更新前9份原文](history/20261023_before_md_refresh/README.md)保存旧迁移页、P01过程、worker和扫描器说明；逐字复制，不修复归档正文中的旧相对链接或重新解释当时的“当前”。正常导航使用归档索引提供的新链接。

[结构整理前归档](history/20261023_before_structure/README.md)继续保留根文档完整旧过程。.pre_git_sync、工具快照、第三方文档、数据包README及run内报告/损失映射/日志均保持原样。9份新归档与原工作区版本一致，不用Git HEAD覆盖接手时已有的文档修改。

此次不移动或重编码大型结果。旧成果及权重共6466项按inode、大小、mtime检查；接手时已有19个非Markdown修改/未跟踪代码文件按原字节保护。该检查不是对全项目内容重新做SHA256。

## 7. 实际检查记录

以下检查在2026-10-23本次文档任务中实际执行：

| 检查 | 实际结果 |
| --- | --- |
| 维护文档覆盖 | 32/32份均有实质更新；另新增本索引与归档索引 |
| 旧说明归档 | 9/9份与更新前工作区原文字节一致 |
| 当前本地链接 | 34份当前文档/索引中的361个链接全部有效；不将归档旧链接重写成新路径 |
| 项目CPU回归 | CUDA_VISIBLE_DEVICES为空；scripts/tests共298项通过，50.445秒 |
| Shell语法 | train_wan81_a6000.sh、resume_wan81_a6000.sh检查通过 |
| Python编译 | wan_train_entry.py、validate_wan81_a6000.py通过；编译缓存写系统临时目录 |
| 差异格式 | git diff --check通过 |
| 受保护Markdown | 43/43份内容保持不变 |
| 旧成果/权重 | 6466/6466项inode、大小、mtime与更新前一致 |
| 既有非Markdown工作 | 19/19份原有修改/未跟踪代码字节保持不变 |

数据编号、参数、文件名、默认命令行为及指标口径已与现有配置、脚本、CSV和冻结报告核对。仅Markdown发生本轮修改；既有代码工作留在工作区，不混入文档提交。提交身份与推送结果以实际Git记录为准，不把结构整理提交ccfe24d冒充此次文档提交。

本次结果不能代替GPU模型验证、原生故障根因定位、三维几何测量或用户逐视频视觉验收；所有运行量和图像观察引用已有实验记录。

## 8. 2026-10-23后续修复：历史回到原文件，更新继续追加

用户指出长期MD需要“从以前到现在”的更新，不能只留下当下摘要。第一轮处理把旧内容移到归档，虽然保存了原文，但破坏了原文件连续阅读的用途；现已修复。第1–7节保留第一轮维护范围与真实检查结果，其中“仅兼容入口、旧正文只在归档”的组织方式不再采用。

### 已恢复的原文件

| 原文件 | 恢复依据 | 原文件内保留的内容 |
| --- | --- | --- |
| [PROJECT_CONTEXT](../PROJECT_CONTEXT.md#history-records) | 结构整理前原文 | 项目目标、早期状态、故障、阶段结果及未解决事项 |
| [EXPERIMENTS](../EXPERIMENTS.md#history-records) | 结构整理前原文 | T2V至Complex12的完整实验过程、八例详细指标和失败记录 |
| [TRAINING](../TRAINING.md#history-records) | 结构整理前原文 | 当时操作、续训、排错及后续修正 |
| [SESSION_HANDOFF](../SESSION_HANDOFF.md#history-records) | 结构整理前原文 | 历次会话进展、用户约定、停止与清理要求 |
| [根README](../README.md#history-records) | Git的04b8d15版本 | 原项目介绍和数据流，另补完整阶段时间线 |
| [docs/PROJECT_CONTEXT](PROJECT_CONTEXT.md#history-records) | 第一轮MD更新前原文 | 早期环境、目标与架构决策 |
| [docs/EXPERIMENTS](EXPERIMENTS.md#history-records) | 同上 | 迁移说明与实验模板 |
| [docs/PROGRESS](PROGRESS.md#history-records) | 同上 | 迁移期间各阶段进度 |
| [docs/HANDOFF](HANDOFF.md#history-records) | 同上 | 早期交接与复核模板 |
| [Claude worker](CLAUDE_WORKER.md#history-records) | 同上 | 旧配置与路由做法；之后明确其被替代的部分 |
| [P01交接](BUILDINGFILLER_P01_VACE_HANDOFF.md#history-records) | 同上 | 三机来源、上传、验收、条件准备与旧方案 |
| [VACE执行](VACE_P01_EXECUTION.md#history-records) | 同上 | 模型准备、预检、运行、复用、故障及排错 |
| [扫描器README](../code/Python_3D_Scanner/README.md#history-records) | 同上 | GUI原设计、参数、文件格式和采集准则；保留当前缺源码事实 |
| [training README](../code/Python_3D_Scanner/training/README.md#history-records) | 同上 | Windows smoke/pilot操作和数据划分 |

这14份文件均在正文完整恢复原有内容，只调整标题层级以嵌入阅读结构；357b09f阶段新增内容也保留，随后追加本次纠正。原文恢复不把当时的未完成状态冒充现在；最新Complex12完成结果、参数、指标与局限继续存在。

AGENTS新增用户要求及规则沿革；目录说明、文档导航、实验模板同步更正。以后采用“此前累计正文＋按日期追加更新”，顶部可以有摘要，归档只作额外备份。历史命令按当时环境解释，历史输出缺失时如实保留记录，不为使链接存在而伪造文件。

### 本次修复检查

核查日期：2026-10-23（UTC+08:00，北京时间）。第一轮298项/50.445秒、361链接仍只代表357b09f阶段；本次修复另行执行并得到以下结果：

| 检查 | 本次实际结果 |
| --- | --- |
| 历史正文恢复 | 14/14份与来源原文一致，仅调整Markdown标题层级；命令、数字和段落完整保留 |
| 已补充资料保留 | 14/14份357b09f阶段正文完整保留，未丢失最新参数/结果 |
| 文档链接与锚点 | 34份维护文档/索引共511个本地链接通过，其中51个来自恢复的历史正文 |
| CPU回归 | 空CUDA_VISIBLE_DEVICES，298项通过，50.140秒 |
| 规定语法/编译/差异 | 两个shell入口语法、两个Python入口编译、git diff --check均通过 |
| 原成果和代码保护 | 6466项旧成果元数据、43份保护文档、9份归档原文、19份原有非Markdown代码均未改变 |
| 已完成实验状态 | 只读确认Complex12为complete、60/60轨迹、C/U/S各300更新 |

本次差异限19份Markdown：14份恢复正文，另同步项目规则、目录说明、文档导航、模板和本索引。只提交这次文档修复，已有代码修改继续留在工作区。

本次仅修复文档，不启动训练、推理、数据处理或外部worker，不改原始数据和冻结run证据。
