> 2026-11-11 当前入口：[保留的最终结果](../../../outputs/pointcloud_vace/adapter_d/run_20261110_050724)。仅保留实验报告.md与checkpoints/adapter_final.safetensors、lora_final.safetensors；完整恢复点和初值已删。 用户已要求删除中间文件；下文旧日志、状态、配置、帧目录、归档和恢复命令仅为清理前历史，不再使用，也不据缺失文件重跑。

# 天然残缺建筑：D 组八视角结构 Adapter 微调

2026-11-10（UTC+09:00）。本轮用户只授权实现并启动 D 组；正式作业完成第一次 AdamW 参数更新及恢复点保存后即交接。八栋开发样例、evaluate.py、A/B/C 对照及 81 帧视频验证，等待训练后另行授权。本页记录启动范围，不预写训练完成或效果结论。

## 实际数据与运行位置

- 项目：`/mnt/windowsE/chuanjun/building_wan`；Python：`/home/shi/miniconda3/envs/scj/bin/python`。
- 本次 run：`outputs/pointcloud_vace/adapter_d/run_20261110_050724`。
- 源包：`data/pointcloud_upload/apc2mesh_adapter_full_v1.zip`，SHA256 `3c4ee0cda58d1733f289c74d58af395f544b0ee5ba6a084e4626cde9fc983d93`。CRC、大小、数量及任务单读取校验完成后，已于 2026-11-10T04:58:35+09:00 删除。记录在 `data/pointcloud_vace/apc2mesh_full_v1_import.json` 与本 run 的 `import.json`。
- 只解压包中 `sample-data-organized/pointcloud/`（15,881 文件）、`sample-data-organized/mesh/`（15,891 文件）及完整 `微调任务/`，三者均位于 `data/pointcloud_vace/` 下，无额外包名目录。旧 V2/V3 不变。
- 划分严格使用上传的 `微调任务/候选空间划分.csv`：train 10,747、validation 905、test 661、diagnostic_only 147、exclude_boundary 3,421。训练只取 use=train；不训练开发八栋、0 号、验证或测试建筑。
- 固定 seed=20261110，从训练候选中无放回安排 3,000 栋，每栋一次连续 17 帧；第一小批按任务单选含可信弱监督的建筑 11613（Paide），帧 026–042，其余保持固定打乱顺序。首步 timestep_index=500 保证有有效 scheduler 权重；后续时间步和噪声种子冻结在 `schedule.json`。这是固定更新预算，并非遍历全部 10,747 栋。

## 输入与监督边界

训练源是完整保留的实测 PLY/XYZ。没有人工删除实测点，也不把八栋人为遮点例改成训练集。结构分支从 P 构造八个深度/有效图、128 个最远点采样锚点及 0.5m 邻域数量；几何归一化中心、尺度、相机和粗框都由 P 确定。

缺口定位只用 P：拟合受观测支持的墙/屋顶平面，在观测平面凸包内选连续低密度框，要求其密度明显低于同平面的观测完整区域；没有可靠框的建筑仅学习可靠观测区域。原始点全部进入渲染，mask 保护实测点完整精灵。条件白区填灰 127；不根据 mesh 决定输入 mask。

没有真实互补扫描及采集条带身份记录，不把 gps_time/scan_angle 字段擅自当成独立扫描真值。弱监督只使用同栋、同观测平面的实测点片段，保留平面内点距、方向和原 RGB，平移到缺口局部坐标并由对齐 mesh 限定法向位置。要求 mesh 法向与观测面一致、距离不超过 0.3m、覆盖至少 90% 的候选框，并避开已有点；不足 8 个可信候选点时不建立弱缺口损失。每例只确定一次候选三维点，在全部目标帧重复投影；mesh、候选点与目标 RGB 不进入 Adapter 或 VACE 条件。

已知标签仅覆盖实测点与紧邻的可靠背景，不把天然空洞当作完整“留空”答案。真实补充标签本轮为空；可靠弱区同时包含点及点间背景，其余缺口全部为未知、损失权重 0。三类标签互斥，按首帧独立/其后四帧均值及 8×8 空间面积映射；分别按各自有效面积平均速度平方误差，权重 1、1、0.3，再乘官方 scheduler 时间权重。

包内旧 `make_cases.py` 引用了未随包提供的 Windows 生成器及旧文件表。新 `scripts/geometry.py` 只补训练读取/渲染，不修改上传脚本或重制开发样例。沿用包中 RGB 背景 [237,240,244]、半径 2px 的圆形点、远到近覆盖、20° 仰角、80 个独立环绕方位和 [right,-up,forward] 投影基；50° 垂直视场由已有相机/点轨迹投影校核。新建筑包围球适配边距固定 1.12，属于本轮显式记录的取景实现，不声称找回了缺失生成器的全部内部参数。

首批实测点 1,734 个，弱候选 38 个；17 帧已知/真实补充/弱区像素累计 331,943 / 0 / 18,212，粗 mask 33,086 像素。mesh 替换检查确认 RGB、mask 及所有结构输入逐值不变，未知弱损失清零。首批训练预览和标签存于本 run `checks/`，不是模型预测。

## 实际模型接入

V3 原版权重 `models/Wan2.1-VACE-1.3B/574e6a744642ce3bee319afc31496b88bde8aac4/diffusion_pytorch_model.safetensors`，SHA256 `c46a6f5f7d32c453c3983bbc59761ea41cd02ad584fb55d1a7ee2b76145847a2`。复用同一 scj 环境、T5、VAE、tokenizer 和 V3 原文 prompt，不下载或加载旧 LoRA。

`adapter.py` 实现任务单的共享 2→16→32→64 三层 CNN、可见深度校核、68→32→1 视图权重、4→64→64 锚点特征、6→64→64 相机射线 query、64 维 K/V 及投影距离偏置（σ=4 块）。全局摘要始终可读。先逐帧查询，再按 4n+1 时间组平均；17 帧 G/M 为 7,800 token，81 帧对应 32,760。输出 64→1536 的权重和偏置全零，其余常规初始化。

`integration.py` 把可选结构参数接口绑定到本次真实加载的 VaceWanModel 实例，项目层 model_fn 包装器转交给官方 `model_fn_wan_video`。复用真实 patch embedding、15 个控制块、官方 hints 与 DiT 残差路径；注入恰在 patch embedding 展平后、首个控制块前。CFG 合批时两路复用相同几何。第三方源码文件未改动。关闭结构并关闭 LoRA 可恢复原版计算；`load_increments` 可在同一基座上重新加载两份独立增量。

先冻结原模型，`trainable_models=None`，仅 VACE self_attn/cross_attn 的 q/k/v/o 加 rank=alpha=8 LoRA，再注册 Adapter。Adapter 142,929 参数，LoRA 2,949,120 参数，总计 3,092,049；AdamW 只接收这两组。第 1 步输出投影改变，上游几何层零梯度符合零初始化设计，后续梯度可以传入上游；冻结控制块和 DiT 的回传图保留。

## 固定训练配置与保存

| 项目 | 值 |
| --- | --- |
| 组别/预算 | 仅 D；3,000 次真实 AdamW 更新，无额外 GPU 试跑 |
| GPU/精度 | 单张 RTX A6000；冻结骨干 BF16，Adapter/LoRA FP32，BF16 autocast |
| 样本 | batch 1、accumulation 1、832×480、连续 17 帧；080 不训练 |
| 优化器 | AdamW，lr 1e-4，betas 0.9/0.999，eps 1e-8，wd 0，clip 1 |
| Scheduler | 原版 Wan FlowMatch，1,000 时间步、shift 5 |
| 冻结编码器 | 共享 prompt 编码一次；每窗目标/VACE 条件分开 VAE 编码，完成后 T5/VAE 移回 CPU |
| VAE | tiled，tile_size 30×52，stride 15×26 |
| 激活 | gradient checkpointing；起始不启用 CPU activation offload |
| 显存退路 | 仅首步成功前出现 CUDA OOM 才依次改为 17帧+checkpoint offload、13帧+offload；保留原尝试日志和实际配置 |
| 恢复点 | 第1步、每100步；保存模型增量、AdamW、Python/NumPy/Torch/CUDA RNG、更新数、历史、样本表/配置身份 |
| 最终权重 | `checkpoints/adapter_final.safetensors` 与 `checkpoints/lora_final.safetensors`，分别保存 |

`checkpoints/resume_step000001.pt`、各整百步的 `resume_stepXXXXXX.pt` 保留；`resume_latest.pt` 原子更新。初值及各保存步两份 safetensors 单独保留，方便后续 B/C 匹配初始化。启动 OOM 之后重新加载原始初值，成功更新以后不自动重跑或扩预算；其他错误直接记录退出。

## 启动与只读查看

```bash
/home/shi/miniconda3/envs/scj/bin/python -B experiments/pointcloud_vace/adapter_d/scripts/launch.py launch /mnt/windowsE/chuanjun/building_wan/outputs/pointcloud_vace/adapter_d/run_20261110_050724
```

该命令是正式启动入口，日常状态不要重复执行。独立 tmux 为 `apc2mesh_D_20261110_050724`，supervisor 复用 `logs/.wan_a6000_gpu.lock`，只有一个 worker；实际 PID 记入 launch.json/run.pid/worker.pid。训练完成仅写 `training_complete.json` 和真实 exit.json，不自动评测或启动其他组。

```bash
tail -n 30 /mnt/windowsE/chuanjun/building_wan/outputs/pointcloud_vace/adapter_d/run_20261110_050724/run.log
```

确需中断后恢复时，先确认无旧 worker，再由人工显式加 `--resume`；完整恢复点身份需与本 run 一致。已有成功更新不能通过普通启动入口重新播放。

## 启动前验证与后续收尾

8 项专项 CPU 检查通过，包括真实 VACE 小模型的零增量等价、冻结层回传与两步上游梯度、实际 VACE 的 240 个 LoRA 张量及精确参数数、相机投影、空视图回退和区域损失。CPU 联调发现 PEFT 在本版本原地插入而非包装模型，已据实际 API 修正；没有因此启动 GPU 试验。298 项项目 CPU 回归通过（51.151秒）。计划 3,000 对原始文件全部通过指纹/解析检查，点数范围 148–87,851；无数据修复。

本次会话以 `first_update.json` 中真实两组参数变化、AdamW step=1、CUDA 同步及恢复点落盘为成功启动证据。模型加载不算训练已开始。完成/明确失败后的完整效果检查及 upload 实验报告留给用户下一次收尾通知，历史报告和旧输出不改。


## 2026-11-10：D组八视角Adapter训练阶段完成

唯一run `outputs/pointcloud_vace/adapter_d/run_20261110_050724` 已完成3000次更新，2026-11-10 13:04:49（UTC+09）退出0；Adapter/LoRA最终权重及完整恢复点通过CPU读回核查。[训练阶段报告](../../../upload/20261110_点云D组_八视角结构Adapter微调_训练阶段实验报告.md)已归档。有效弱区仅162/3000批（5.4%），视频效果尚待验收；八栋推理、其他对照及evaluate.py未启动。状态查询不能重跑成功训练，旧启动记录保留。

启动更正：此前--hold/start.ready等待说明未实际生效（shell条件右括号前缺空格，start.ready不存在）；真实首步为05:12:03+09:00。正式训练单次执行并正常完成，本次不修改冻结源码；后续如依赖等待功能，需先修正和测试。


## 2026-11-10：D组八栋开发建筑视频验收启动

[八栋评测入口](../adapter_d_eval/README.md)。用户授权D-3000最终增量的8次81帧推理，run为outputs/pointcloud_vace/adapter_d_eval/run_20261110_150749；实际首步看sampling_started.json。沿用832×480/30步/CFG5/seed20261018，原相机与输入，结构分支剔除隐藏实测点。后台自动保存原始PNG/视频、对比与evaluate.py指标；首步即交接，完整效果与报告留后续复核。无新训练或A/B/C组预算。


## 2026-11-10：D组八栋视频与自动指标生成完成

固定run `outputs/pointcloud_vace/adapter_d_eval/run_20261110_150749` 已完成8条81帧/648原PNG、8原视频＋8对比视频及全部指标，2026-11-10 16:02:08（UTC+09）退出0，耗时47分20秒。[生成完成报告](../../../upload/20261110_点云D组_八栋81帧推理与自动指标_实验报告.md)已归档。16条视频及696张PNG重新完整解码通过；人工视觉复核待进行，自动指标存在高参考背景误填与首尾差异疑点。以status/exit判断完成，不把progress旧running字段当重跑依据；本次没有新GPU任务。
