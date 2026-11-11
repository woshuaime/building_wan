# VACE 单张参考图引导 V2：一栋、循环17帧、A/B/C配对

2026-11-12（UTC+09）。本轮用户授权实现并启动六条正式推理，确认首条带参考调用完成真实采样步后交接；完整视觉复核和报告留待后续通知。

- 唯一run：`outputs/pointcloud_vace/reference_guidance_eval/run_20261112_025414`。
- 建筑：开发例`wall_13697`，原3259点，剔除case中的24个隐藏实测点，条件仅用剩余3235点。不是盲测，无新训练。
- 循环起点78，实际帧号`78,79,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14`；中心是原始006。原080不作为额外相机。
- 选窗依据为输入/mask的缺墙可见性及实际跨越79→0；本窗口共70723个白区像素×帧，中心4525。未根据正式输出挑窗口。
- 同一Wan2.1-VACE-1.3B及D-3000最终Adapter/LoRA，rank=alpha8、scale1；17帧832×480/16FPS、30步/CFG5/shift5/VACE scale1，原prompt、negative为空，BF16骨干/FP32增量，原tiled VAE设置。
- 正式seed为20261018/20261019。每seed按B/C/A串行，共6次；先B是为了启动交接时确认参考条件真实工作。

## 三组与固定参考图

| 组 | 条件 |
| --- | --- |
| A | 现有D-3000，无额外参考图 |
| B | 中心相机的未填灰残缺观测`reference/observed.png` |
| C | 同相机补全草稿合成图`reference/reference.png` |

草稿直接取`outputs/pointcloud_vace/adapter_d_eval/run_20261110_150749/cases/wall_13697/raw_vace.mp4`的006帧，原图保存在`reference/draft.png`；白区取草稿、黑区取该case的`reference/damaged.mp4`006帧。无额外草稿GPU调用。旧预测已有暗色面片/点阵候选，照实保留作为诊断，不反复生成参考。三份参考图都是832×480，两个seed共用。

## 最小接入与帧对齐

复用`adapter_d/scripts/integration.py`、原始`load_local_vace`和`observed_points`，不修改训练或第三方源码。`prepare.py`实际保存17帧RGB/mask数组，按相同原相机索引重建目标深度与几何；八个来源相机仍为000/010/020/030/040/050/060/070，空间锚点来自同一份残缺点集。

本地原生参考图参数为`vace_reference_image`，由现有VAE/VACE编码，并使用pipeline原生参考latent前缀和输出裁除。`reference_binding.py`给结构特征及门控同时补零前缀，保证有bias的投影之后参考位置残差仍为零；真实视频部分保留原几何。17帧对应5个视频latent/7800个结构token；单参考时为6个latent/9360个token。两路CFG共享同一几何。参考图不占用真实相机帧，最终仍输出17帧。

保留原生噪声机制：B/C同seed、同形状初始化一致；A与带参考组的视频噪声不额外强行匹配。推理不读mesh、隐藏点或目标RGB，也不以GT初始化。D历史训练使用mesh弱监督这一事实保留，不将新参考称为真实答案。

## 启动与只读查询

Python固定为`/home/shi/miniconda3/envs/scj/bin/python`。准备与启动入口为本目录`scripts/prepare.py`及`scripts/run.py launch`；启动只用于本轮尚未运行的任务，不是日常状态查询。

```bash
/home/shi/miniconda3/envs/scj/bin/python experiments/pointcloud_vace/reference_guidance_eval/scripts/run.py status outputs/pointcloud_vace/reference_guidance_eval/run_20261112_025414
```

后台使用项目GPU锁、独立tmux和fresh worker串行运行，Codex/SSH结束不影响队列。完成的`.done.json`与视频被复用；普通故障修复后只继续未完成任务，不重放成功调用。首步证据为本run的`sampling_started.json`，包含实际参考/VACE latent布局、两次CFG调用和真实Adapter注入记录。

## 自动保存与后续复核

后台保存`group_A/B/C/seed_<seed>.mp4`六条原始预测，以及`comparisons/comparison.mp4`（上/下两seed，输入/A/B/C四栏）、五张边界/中心/末尾关键帧、简单mask外观测保真指标。输出不回贴，不批量保存逐帧PNG。

每条及对照视频均要求17帧/16FPS并完整解码；完成状态为`complete_awaiting_visual_review`。17帧首尾不是相同相机，不计算旧81帧首尾MAE。运行成功不代表补墙或三维恢复成功；本轮不产生XYZ点集。

用户后续通知后复核墙面连接、点风格、79→0连续性及已知结构变化，再补本run的`report.md`与upload同内容报告，按日期追加长期MD并Git收尾。当前不提前写A/B/C效果结论。旧输出及已清理的旧中间文件保持原状态。


## 本次启动证据

2026-11-12T02:55:30.865386+09:00：B/20261018实际完成首步、两路CFG调用，额外参考latent与9360个结构token对齐，真实VACE前注入已确认。tmux `vace_ref_v2_20261112_025414`，父PID 1259517，首worker 1259530。2项专项CPU检查和298项项目CPU回归通过（50.875秒），无额外GPU smoke调用。本轮到启动即交接，报告与完整视觉复核等待用户后续通知。
