> 2026-11-11 当前入口：[保留的最终结果](../../../outputs/pointcloud_vace/0_v2_base/run_20261109_021750)。该目录保留raw_vace.mp4、input_mask_raw.mp4和实验报告.md。 用户已要求删除中间文件；下文旧日志、状态、配置、帧目录、归档和恢复命令仅为清理前历史，不再使用，也不据缺失文件重跑。

# 0号建筑点云原版VACE基线

2026-11-09（UTC+09:00）。用户授权仅一次正式推理；确认实际GPU采样后结束会话，不等待完成。新实验不沿用mesh LoRA、训练缓存或提示词。

## 问题与固定条件

观察原版VACE能否生成离散小点、是否生成实心墙、相机环绕时是否漂移/闪烁，以及原观测点是否改变。本次不训练、不搜索参数、不导出PLY、不做定量结论。

- 输入：`data/pointcloud_vace/0_v2/src_video.mp4`、`src_mask.mp4`、`prompt.txt`；逐帧完整解码，81帧、832×480、16FPS、时间戳0至5秒一致。
- Mask三通道相同、严格0/255；白区生成、黑区保留条件；每个白像素对应输入RGB(127,127,127)。000/020/040/060/080输入联系图人工静态核查。未改mask、点云、目标墙面或任何输入字节。
- 原始prompt逐字读取并复制，关闭扩写；negative prompt为空，避免旧mesh外观偏好。没有未填灰原始视频，展示仅称“填灰输入”。
- 官方Wan-AI/Wan2.1-VACE-1.3B，本地revision `574e6a744642ce3bee319afc31496b88bde8aac4`；共享原T5/VAE/tokenizer，加载checkpoint同时生成DiT及VACE分支；不加载LoRA。
- 一次81帧完整调用；832×480、16FPS、30步、CFG5、seed20261018、shift5、VACE scale1、BF16、CPU随机噪声、tiled VAE（tile30×52、stride15×26）；full memory模式，不分段、不降规格、不重试。

## 代码与保存

专用入口`scripts/run.py`复用项目`scripts/vace_p01.py`的官方权重核验、模型加载与真实`vace_video`/`vace_video_mask`参数构造。第三方引擎未修改。Python为`/home/shi/miniconda3/envs/scj/bin/python`。

本次实际run：`/mnt/windowsE/chuanjun/building_wan/outputs/pointcloud_vace/0_v2_base/run_20261109_021750`。目录创建时采用Asia/Tokyo实际时间。输入保留绝对路径及SHA256，prompt原文副本在run。准备检查不进行模型推理。

独立session中的父进程持有项目GPU锁，启动一个worker且等待真实退出。`run.pid`是父进程、`worker.pid`是GPU子进程；`run.log`保存完整实时输出。`config.json`记录最终参数、模型哈希、环境、输入逐帧检查；`sampling_started.json`仅在首个完整扩散步骤执行且CUDA同步后写入。

推理后先保存81张`raw_frames/000.png`至`080.png`及未经回贴的`raw_vace.mp4`，完整解码验证后写`raw_result.json`。之后尝试生成`input_mask_raw.mp4`和五张比较PNG；额外可视化失败单独记录，不覆盖原结果。没有观测点回贴后处理。

父进程最终写`exit.json`与`status.json`，即使worker发生原生崩溃也保留真实返回码；`complete_awaiting_visual_review`表示媒体交付完成、视觉待审，不表示点云效果成功。`failed`不自动重跑。当前完整报告与效果结论待用户后续通知及实际结果核查，不能预写完成。

## 只读查看

```bash
tail -n 40 /mnt/windowsE/chuanjun/building_wan/outputs/pointcloud_vace/0_v2_base/run_20261109_021750/run.log
cat /mnt/windowsE/chuanjun/building_wan/outputs/pointcloud_vace/0_v2_base/run_20261109_021750/status.json
ls -lh /mnt/windowsE/chuanjun/building_wan/outputs/pointcloud_vace/0_v2_base/run_20261109_021750/raw_vace.mp4 /mnt/windowsE/chuanjun/building_wan/outputs/pointcloud_vace/0_v2_base/run_20261109_021750/exit.json
```

后两项只有实际保存/退出后才存在。不要用supervise入口查询状态；同run已有attempt标记会拒绝第二次执行。


### 2026-11-09 后台派发快照

2026-11-09T02:22:09.011787+09:00：输入与pipeline CPU预处理检查通过，官方VACE/T5/VAE及tokenizer完整SHA核验通过；298项项目CPU回归通过（62.672秒），父进程成功/原生异常/普通失败三项模拟检查、规定shell语法/Python编译及diff检查通过。实际启动独立session 3508602，后台父PID 3508602、worker PID 3508606，stdin为/dev/null、stdout/stderr重定向run.log；run.pid与worker.pid已落盘。此快照仅确认后台派发，采样验收看本run的sampling_started.json及run.log中的真实diffusion_sampling步数；退出与效果仍待实际完成。用户要求一旦确认实际GPU推理就停止主动跟踪，因此不在本轮等待完整结果。


## 2026-11-09：执行完成、静态复核与独立报告

[详细报告](../../../upload/20261109_0号建筑点云_Wan21_VACE原版基线_实验报告.md)已写入统一upload目录。实际一次81帧/30步GPU推理，worker退出0，东京02:27:48结束，总耗时339.2228秒；81原PNG、原视频、三栏对比和5张关键帧均已存在。峰值allocated/reserved为18.3960/31.0273GiB。没有失败重试或第二次推理。

本次完整解码两视频和81PNG、静态复核全部81建筑区域及000/020/032/047/060/080原像素局部，结论：原版生成了暗底密集规则点阵/网格面，没有匹配输入稀疏观测点；原观测点有重绘；000/080输入/mask相同而生成白区首尾MAE75.957650/255。未做实时播放、量化闪烁或三维验证，未将点纹理认作真实三维点恢复。

新复核材料在原run的report_review_20261109_024727，最终记录final_review.json；原status的complete_awaiting_visual_review是当时快照并保留。用户要求后续每组实验主动生成upload报告、更新索引与长期MD；本轮无新增GPU任务。
