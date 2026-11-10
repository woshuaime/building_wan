# D-3000：八栋开发建筑的原始视频推理与验收材料

2026-11-10（UTC+09:00）。用户已授权生成八栋 D 组验收视频，确认首栋实际进入扩散采样后结束会话，后台串行完成剩余任务。当前为启动说明，不能提前解释模型效果；A/B/C 组及新训练不在本轮预算。

## 运行身份与模型

- 正式 run：`/mnt/windowsE/chuanjun/building_wan/outputs/pointcloud_vace/adapter_d_eval/run_20261110_150749`。
- Python：`/home/shi/miniconda3/envs/scj/bin/python`；单张 A6000，复用项目 GPU 锁与独立 tmux。
- 基座：V3 同一原版 Wan2.1-VACE-1.3B，revision `574e6a744642ce3bee319afc31496b88bde8aac4`；基座 SHA256 `c46a6f5f7d32c453c3983bbc59761ea41cd02ad584fb55d1a7ee2b76145847a2`。
- 增量只取已完成训练 `outputs/pointcloud_vace/adapter_d/run_20261110_050724/checkpoints/` 的 `adapter_final.safetensors` 和 `lora_final.safetensors`，更新数 3000。SHA256 分别为 `8720f3a3d2883aaa9b985e77f74961c7669d84c71ddd652e8129e565490aaaf4`、`bcc21c2ac547781b0da6954fe3ab09998f7b57b99b6aa2741a6c3400ad181544`。
- 复用冻结训练目录中的 Adapter/LoRA 接入源码。每栋从原始基座新建模型，只加载这两份增量，检查实际活模块的24/240张量逐值等于保存权重，再冻结全部参数。原训练源码和完成 run 保留。

## 八栋输入与防泄漏

输入是上传包现有 `data/pointcloud_vace/微调任务/cases/` 的八组，按名字串行：

| 样例 | 类型 | 人为遮去的实测点 |
| --- | --- | ---: |
| eave_3230 | 跨墙/屋顶檐口 | 43 |
| roof_12512 | 屋顶 | 52 |
| roof_5670 | 屋顶 | 117 |
| strip_5016 | 墙面长条 | 42 |
| wall_10416 | 原本为空的墙面开发例 | 0 |
| wall_13697 | 墙面 | 24 |
| wall_2534 | 稀疏墙面 | 4 |
| wall_8079 | 稀疏墙面 | 10 |

八栋均为 diagnostic_only，实际训练 3000 栋中无相同建筑，不能把它们称为盲测测试集。wall_10416 没有被遮实测点，不把其恢复命中率为 null 当作失败或满分。

VACE 直接接原 `inputs/vace_video.mp4` 与 `inputs/vace_video_mask.mp4`，逐字读取各例 prompt.txt。白区生成，黑区上下文；81帧、832×480、16FPS、时间戳对应、白区RGB127及输入首尾相同均完整核验。

结构分支从原点云按 case.json 中的 hidden_original_point_ids 排除所有隐藏答案，再构造参考深度、FPS 锚点与密度。使用包内81个原相机，保留 [right,-up,forward] 和 50° FOV；点与相机同步平移到只由观测P决定的中心并共享尺度。未重建取景轨迹，不把完整原点云、mesh、目标RGB或隐藏点送进结构分支。8个参考相机为000/010/020/030/040/050/060/070。

CPU重建全部648帧，确认观测P在mask外的RGB与上传视频逐像素一致。由此核对了相机坐标、焦距、点精灵及遮点身份，而非只看视频可读。门控和结构特征采用相同4n+1布局，81帧对应 [1,32760,1]/[1,32760,64]；CFG正负两路使用同一份几何。

## 推理设置

继承 V3 实际推理参数：81帧、832×480、16FPS、30步、CFG5、seed20261018、sigma shift5、VACE scale1、原骨干BF16，增量FP32，tiled VAE 30×52/stride15×26，full memory。negative为空，无扩写、无首尾初始化、无GT视频初始化，不抽帧/分段/缩小分辨率。080是模型真实预测，原始结果不做保留点回贴。

原pipeline签名不直接接structure_tokens；本轮使用项目层固定几何model_fn，把同一份预计算G/M传入实际VACE forward。每个完整采样步必须完成两路CFG调用，且真实VACE记录已在首个控制块前注入几何。CUDA同步后才写sampling_started.json；仅权重加载不能当启动成功。

## 后台输出与错误处理

每栋在 `cases/<case>/` 自动保存：

- raw_frames/000.png–080.png：81张模型原始PNG。
- raw_vace.mp4：832×480/16FPS的原始预测，独立保留。
- input_mask_D_target.mp4：四栏输入/mask/D原始预测/实测参考，3328×512/81帧。
- comparison_000/020/040/060/080.png、overview.png：放大检查与五视角总览。
- metrics.json、evaluation_provenance.json：包内evaluate.py的指标和来源。
- prompt.txt、input_audit.json、loaded_increments.json、run.log、sampling_started.json、gpu_exit.json、raw_result.json、postprocess.log与状态。

GT/reference RGB仅在原始预测已保存、GPU worker退出后由独立CPU流程读取。复用上传evaluate.py全部原指标定义，RGB24解码显式限制ffmpeg单线程，不改阈值或计算公式；指标读取原始预测MP4，图像编码误差属于解释边界。输出背景误填、漏补、相邻可见实测点中心命中、方向点距及同视角首尾MAE；这些指标不等于三维点跟踪或人工视频验收。

全队列完成后汇总 metrics_summary.json/csv，标记complete_awaiting_visual_review并记录真实exit.json。主日志run.log同步每栋GPU日志，progress.json保存已完成例清单。一次只有一个GPU worker；每栋最多一次正式调用，不自动重试成功或失败轨迹。CPU制图失败时原始结果仍保留，错误路径清楚记录；后续可只重做CPU步骤。

独立tmux为 `apc2mesh_D_eval_20261110_150749`。实际父/子PID在launch.json/run.pid/worker.pid。启动器使用本轮已通过CPU测试的start.ready等待条件，避免训练旧启动器的右括号空格问题；放行后不依赖Codex或SSH。

## 状态与后续交接

```bash
tail -n 30 -f /mnt/windowsE/chuanjun/building_wan/outputs/pointcloud_vace/adapter_d_eval/run_20261110_150749/run.log
```

不要为刷新状态重跑启动命令。看到本run sampling_started.json中的completed_steps=1即结束本次会话，不等待8栋结束。后台将继续输出，后续用户通知收尾时再查看视频、解释指标并更新upload详细报告与长期MD。本轮已完成7项专项CPU检查（隐藏点排除、全81帧结构特征、CFG共享、冻结身份、GT读取顺序、防重复启动、后台等待条件），并按项目要求执行全局回归；实际结果见本run checks/。
