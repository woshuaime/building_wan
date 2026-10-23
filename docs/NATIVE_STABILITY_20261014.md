# A6000 原生崩溃诊断（2026-10-14）

文档维护更新：2026-10-23（UTC+09:00）。**以下主体是2026-10-14/15的历史诊断记录**，其中“当前主机”、PID、驱动、命令可用性和测试数量只描述当时；本次没有重新执行GPU诊断。

当前衔接：Pilot8及Complex12已完成，但不能据此宣称历史free()/SIGSEGV根因已修复。Complex12另有B/CS04保存后原生退出−6、独立验证有效输出的记录；它与下文T2V训练崩溃分别追溯。用户要求删除的串行诊断产物没有恢复，不能把历史路径当作现存成果入口。

当前环境/完成状态见 [PROJECT_CONTEXT](../PROJECT_CONTEXT.md)，操作边界见 [TRAINING](../TRAINING.md)，该T2V实验见 [说明](../experiments/t2v/single_orbit/README.md)。本次只增加上述适用范围说明，保留原诊断正文的数字、结论和未确认事项；本次CPU检查另记 [文档索引](DOCUMENTATION_INDEX.md)。

---

状态更新于 2026-10-15：所有 GPU 诊断均已结束；用户要求不再继续跑，已删除本轮串行测试产物，仅保留文档摘要。普通反向基线确实崩溃，串行完整预算未复现；首次非法写入和根因尚未确认。

## 历史证据

`logs/a6000_diag_20260928.log:1752` 主线程回溯显示：glibc
`malloc_printerr("double free or corruption (out)")` → `_int_free` →
`ViewBackward0` 析构 → `torch::autograd::deleteNode` → loss Tensor 释放。
Python 栈指向 `diffusion/runner.py:94` 下一次 loss 赋值附近。
这是堆损坏的**检测位置**，不是首次非法写入的位置；不能仅据此认定
DataLoader、checkpointing、PyTorch 特定版本或 CUDA 驱动为根因。
历史 worker 0 的崩溃证据仍成立。

## 本次独立检查

- 当前主机 vccip，RTX A6000，50,895,192,064 bytes 显存；PyTorch
  2.11.0+cu128，CUDA runtime 12.8，内核驱动 580.95.05。
- 系统无 `nvidia-smi` 命令，但 CUDA 和 NVML 均可用，不能误判为无 GPU。
- 实际 GPU 小张量前向、反向、同步正常，进程退出码 0。
- 10 个独立 CPU PyTorch 前向/反向进程正常退出。
- 10,000 次 CPU non-reentrant checkpoint/view/mean/backward 正常结束。
- 30 次 safetensors PyTorch 后端头部读取进程正常退出。
- 100 次同类读取后主动 SystemExit，均精确返回 1，stderr 只有预期消息。
  记录在 `logs/native_stability_20261014/exit_stress.jsonl`。
- `pip check` 通过。以上小测试不能替代真实模型长时间训练。
- 当前加载库清单、环境和历史主线程摘要保存在
  `logs/native_stability_20261014/`。无证据支持直接升级或降级依赖，未修改环境。

## 真实训练基线

2026-10-14 21:16（Asia/Shanghai）启动 GDB 诊断，原始训练配置保持
384×384、81 帧、rank 16、batch 1、累积 4、worker 0。
从历史平铺目录的 step-11200 权重续跑，skip 13600，initial 11200，
计划剩余 2336 个物理 batch，预期最后 logger step 13536。
这是 weights-only 续跑，不恢复旧 RNG 或 optimizer。

- 新输出：`checkpoints/native_stability_20261014_baseline/`
- 新日志：`logs/native_stability_20261014/baseline.log`
- 启动时训练 PID 68880，GDB PID 68812；这些只是本次记录，不能在后续会话
  未核对启动时间就向相同 PID 发信号。
- 初期 loss 有限、未复现是当时的进度记录；最终在第 166 个新增 batch 后崩溃，见下文。
- 原有数据、缓存、权重和输出未覆盖；没有并发启动第二个 GPU 训练。

## 后续诊断入口与验收

`scripts/diagnose_wan_a6000.py` 默认只读预检；加 `--run` 才通过既有共享锁
启动 GDB 训练，自动分配独立日志和权重目录。可用 `--checkpoint` 指定来源。
它在子进程真正退出后记录 `result.json`，检查退出码、连续且有限的 loss、
预算步数、最终权重存在和引擎完成标记。

即使写过 training_completion.json，随后 native abort 也判为失败。
`completed_budget` 仅表示该次预算与退出证据满足检查，不代表根因已确认，
也不代替权重张量可读性/有限值检测、视觉质量或多次复测。
本次已启动的基线早于该入口，退出码需由启动会话收集，不能补造。

根因确认需要最小复现或明确的非法内存访问证据，以及单变量修复前后对照。
如果基线没有复现，应记录“未复现”，继续保留未解决状态；不能以短测通过
或降低线程数直接宣称历史故障已修复。

## 基线最终结果：成功复现原生崩溃

基线新增 166 个 batch，loss 最后为 11366，随后主线程在
`UnsafeViewBackward0::~UnsafeViewBackward0` / `deleteNode` 触发 SIGSEGV。
启动会话返回退出码 255。没有完成标记，也没有新 checkpoint（未到保存周期）。
完整堆栈在 baseline.log；baseline_result.json 保存已核对的退出状态。
用户已明确选择修复后跑完剩余预算并检查退出、权重与 loss。

隔离检查：纯 PyTorch 30 层 checkpoint GPU 小模型 2000 次、缩小 Wan DiT
GPU 模型 2000 次均正常退出。加入 LoRA / Accelerate 组合继续诊断。
Valgrind 仅下载解包在 /tmp/wan-native-valgrind-20261014，不安装到系统或改 scj。
其最小 CPU 导入测试只报告动态加载器 strncmp/is_dst 的读越界上下文，
未报告写越界或重复释放；不能把这个读报告当作训练故障根因。

## 2026-10-15 继续诊断

LoRA + Accelerate + 梯度累积的缩小模型已完成 2000 次 GPU 反向并退出 0，
完整模型特有故障仍未定位。已启动独立 Valgrind 主机内存检查：
`checkpoints/native_stability_20261015_valgrind/`，日志
`logs/native_stability_20261015/valgrind_launcher.log` 和 `valgrind_training.log`。
检查器仅在临时目录解包，未更换 scj 依赖。该检查改变执行时序，因此即使
未复现也不能排除并发错误；NVIDIA ioctl 提示不能作为根因。

同日只读全量缓存审计：1992 个 latent 均为 FP32、shape (1,16,21,48,48)，
全部有限且不携带梯度；共享 context 的 (1,512,4096) BF16 张量也有限、不携带
梯度。完整审计结果在 `logs/native_stability_20261015/cache_integrity.json`。

## 2026-10-15 后续结果与完整验收

- 主机内存检查最终覆盖 196 batch，父进程退出 143 是按计划主动终止，
  Valgrind 最终报告 0 错误；停止意图另行保存，不标为完整训练成功。
- 8 GiB memtester 一轮全部通过，CPU 运算校验 120 秒通过，均退出 0。
  这是已测试范围内的阴性结果，不排除全部硬件故障。
- GPU memcheck 因开销较大缩短到 4 个完整 batch / 一次累积更新，
  报告 0 个内存错误；主动停止，原计划 8 batch 没有完成。
- step-11200 的 600 个 BF16 LoRA 张量均有限，SHA256 及完整结果在
  source_weight_integrity.json；NumPy 不支持 BF16 payload，完整读取用 PyTorch CPU。
- 同种子并行/串行各 8 batch、各复测一次，都正常退出。首次更新前的 4 个
  loss 完全一致；更新后同模式复测也出现数值波动。四组比较的最大权重差均
  3.0517578125e-05，最大 loss 差为 8.1e-06 至 2.0e-05。未通过逐位一致
  检查，但候选开关差异与同模式复测在相同量级。未改为逐位通过；含串行数据
  的报告后来按用户要求删除，本段只保留清理前核对的摘要。
- 固定种子并行反向 256-batch 对照正常结束并保存 step-11456，未再次复现。
  原始调试器返回 1；GDB 日志明确显示 Python exited normally，随后不可用的
  py-bt 命令使调试器出错。该返回码不登记成 Python 原生崩溃。

修复 GDB 启动器：scripts/wan_gdb.py 只在实际信号停机时采集堆栈，并用
exited event 记录 inferior 状态，避免正常退出后的诊断命令误判；py-bt 缺失
时保留原生栈。正常 exit 0、Python exit 9、原生 abort、segfault 四项真实 GDB
集成测试分别返回 0、9、134、139，均通过。

## 串行完整预算最终结果与用户清理要求

完整串行反向预算测试由独立 tmux 会话托管，10-15 03:52:34 启动，
06:07:29（Asia/Shanghai）引擎写入完成标记。seed 20261015，物理 batch 1、
累积 4、worker 0，384×384 / 81 帧，来源为未改动的历史 step-11200。

- 实际新增 2336 batch，loss step 11201–13536 连续；全部有限，最小 0，
  最大 0.03849076107144356，末值 0.00538101838901639。
- 11600/12000/12400/12800/13200/13536 六份权重均在 CPU 完整读取，
  每份 600 个 BF16 张量、21872640 个元素，全部有限且键名/形状与来源一致。
  最终权重 SHA256：558dd1e58ade4b623e759054c3e7bc78de8fca711b03deffb527eea8c9686879。
- 日志无 SIGSEGV / SIGABRT / 原生堆错误，GDB 明确记录 inferior exited normally。
  旧 GDB 命令在退出后执行不可用的 py-bt / continue，launcher 返回 1；
  原始失败记录未覆盖，另写核对记录区分 debugger 错误和 Python 正常退出。
  该运行启动于 GDB helper 修复之前，不能说 GPU 长测验证了新版 helper。
- 一次预算通过不等于根因确认；普通反向 256-batch 对照也通过，尚无首次
  非法写入证据，不能确定关闭反向多线程就是根治。正式默认仍为 0。

用户要求停止并删除本轮串行测试。处理时任务已正常结束，未再启动训练
或推理；核对无训练、监督进程，tmux 会话已结束。清理前重新核对连续有限
loss、退出记录和六份权重哈希与张量检查报告，随后删除：

- checkpoints/native_stability_20261015_serial_full（6 份权重）
- checkpoints/native_stability_20261015_parity_serial（1 份权重）
- checkpoints/native_stability_20261015_parity_serial_repeat（1 份权重）
- 同日诊断目录的串行日志、串行结果及完整性报告、含串行数据的对照报告、
  run_serial_full.py / verify_serial_full.py 及其 pyc，共 25 个文件。

合计 350824306 bytes。串行原始产物与报告已删除，本文为清理前核对的摘要，
上面的历史路径已不可读取。普通反向及基线崩溃证据保留。
原平铺 28 份权重、缓存和验证视频未改；来源 step-11200 SHA256 仍为
4a085810854b550cdd912e3edebfbe74f4d9229dc67abd623fe84dc002743161。
后续不得沿用本轮较早授权自动补训、重跑诊断或推理。

清理后的检查：git diff --check、shell 语法和 Python 编译检查通过；
scripts/tests 36 项与既有训练工具 12 项 CPU 回归全部通过，无跳过。
这些检查没有重新启动 GPU 作业，也不构成原生故障已修复的证据。
