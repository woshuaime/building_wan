# 当前项目状态

核查日期：2026-10-23（UTC+09:00）。当前正式实验已结束，结构整理没有启动GPU训练或推理。

## 最新结果：Complex12 Rev1 已完成

- [实验入口](experiments/vace/complex12_rev1/README.md)，[结果入口](outputs/ComplexSingleOrbit_Rev1_Pilot_20261022/00_看结果/README_从这里开始.md)，[中文报告](outputs/ComplexSingleOrbit_Rev1_Pilot_20261022/00_看结果/最终实验结论.md)。
- C/U/S 各300次真实优化器更新，固定step300；B/L/C/U/S各12条完整81帧，共60条、4860张原始PNG、12条视频、12张总览。progress.json 为 complete。
- 训练8栋：CS01/03/05/06/07/09/10/11；验证4栋：CS02/04/08/12。VACE-1.3B、512×512、16FPS、30步、CFG5、推理seed20261018，原始输出没有回贴或强制闭环。
- C/U/S共同新初始化、采样与随机序列，训练seed20261022。L仅为旧P04/P06 LoRA迁移参考，未用作新三组初始化。
- C 相比 Base 的验证缺损 MAE 降低64.21%。S 相比 U 的验证 MAE 降低1.80%、结构误差降低0.19%；优势主要来自CS12，保留区及全图闭环指标退步，不能认定结构加权稳定胜出。
- 完成时刻原记录为2026-10-23T02:59:37+08:00，即2026-10-23T03:59:37+09:00。源记录保留原时区。

## 目录与当前入口

实验专用代码位于 experiments/vace/complex12_rev1/scripts/。原 outputs/ComplexSingleOrbit_Rev1_Pilot_20261022/scripts 是兼容链接；旧resume.sh和冻结配置仍指向同一结果。大文件仍在原目录。

其他VACE与历史T2V实验分列在 [experiments/README.md](experiments/README.md)。新想法按 [目录约定](docs/PROJECT_STRUCTURE.md) 新建文件夹，不能直接复用或改写旧实验的冻结结果。

## 已知限制

- B/CS04曾在完整预测保存后原生异常退出；81帧独立复核后保留复用，没有重跑。不能声称原生崩溃根因已修复。
- 挂载恢复后C从第40次更新继续至300；U/S各300，未重复已成功轨迹。
- 历史A6000 T2V保留28份权重、29个比较视频、1992个训练缓存。训练预算完整结束的证据仍不充分，不能自动补训。
- 60条Complex12轨迹已进行连续逐帧图检查；不能把它写成已量化高频闪烁或完整三维恢复。
- P04/P06旧LoRA学过这两栋；Complex12的CS06是源1251，不能混同旧P06源1848。

下一步实验尚未授权。报告建议先验证C/U/S配对重复的稳定性；当前只交付已有结果。

详细过程保留在 [整理前状态快照](docs/history/20261023_before_structure/PROJECT_CONTEXT.md)，旧快照中的“当前”“下一步”和PID均只描述当时。
