# 0号建筑点云V3三墙对照：原版VACE单次基线

2026-11-09（UTC+09:00）。用户授权仅一次正式原版Wan2.1-VACE-1.3B视频局部补全；本会话在第1/30采样步骤完成即结束，视觉和最终报告留待后续收尾。

## 输入与对照

输入固定在`data/pointcloud_vace/0_v3/`，与上传的`data/pointcloud_upload/0_v3_input.zip`逐文件字节一致。三个文件为src_video.mp4、src_mask.mp4、prompt.txt。视频SHA256为`16eb946b679a056a7227c2eb7c5217ba64cf56295393b67d88f3ee95ab56458f`，mask为`db445e937f7f0e9acbf665116f2510257af7631501066f4bb34dab21a3ef1d67`。

V3保留V2所有白区：逐帧检查移除0个白像素，81帧新增合计487026个白像素，新增区之外输入RGB变化0像素；新增区及原白区均为RGB127，mask只有0/255。提示词与V2逐字节一致。C墙约5.72米的空区来自用户描述，视频包无米制几何，本轮不独立测量。输入000/080及mask完全相同。

## 冻结配置与代码

`scripts/run.py`复制V2已跑通入口，只有prepare适配V3来源、强制V3指纹及继承V2正式config.json；worker、真实采样进度、原始保存和supervise的函数AST与V2一致。原V2入口与run不改。

配置基准：`outputs/pointcloud_vace/0_v2_base/run_20261109_021750/config.json`，完整model/inference继承，唯一文本来源为V3原prompt。单次81帧、832×480、16FPS、30步、CFG5、seed20261018、sigma shift5、VACE scale1、BF16、CPU噪声、tiled VAE(30,52)/(15,26)、full memory；negative为空、无LoRA、无扩写、无GT/首尾帧初始化。真实080预测保存，不复制000。

本次run：`/mnt/windowsE/chuanjun/building_wan/outputs/pointcloud_vace/0_v3_base/run_20261109_175027`。prepare只做输入/模型/环境检查；没有GPU试跑。supervise使用项目GPU锁、独立父进程及唯一worker，run.pid/worker.pid记录真实PID，launch.json记录独立session与命令。sampling_started.json只在真实第1步完成且CUDA同步后出现；只看到加载日志不算启动验收。

## 自动产物与后续收尾

后台自动按V2流程保存raw_frames/000–080.png、raw_vace.mp4、三栏input_mask_raw.mp4、五张关键帧PNG、config.json/prompt.txt、run.log、raw_result.json/comparison_result.json、status.json与真实exit.json；原始结果不回贴、不覆盖。可选展示失败单独记录，无自动重试或第二组预算。

只读查看：`tail -n 40 /mnt/windowsE/chuanjun/building_wan/outputs/pointcloud_vace/0_v3_base/run_20261109_175027/run.log`、`cat /mnt/windowsE/chuanjun/building_wan/outputs/pointcloud_vace/0_v3_base/run_20261109_175027/status.json`。完成或明确失败后，后续收尾报告写到`upload/20261109_0号建筑点云_V3三墙对照_Wan21_VACE原版_实验报告.md`，登记upload索引并追加长期MD；当前不预写C墙效果、A/B点阵或闭环结果。
