# Complex12 Rev1：十二栋单轨配对实验

问题：普通LoRA C、均匀缺损加权U和结构带加权S，哪一种实际更有效？

**已全部完成。** C/U/S各300次实际优化器更新；B/L/C/U/S各12条81帧，共60轨迹、4860张原始预测、12视频和12总览。当前没有待续跑任务。

## 直接查看

- [视频、图片、指标入口](runs/20261022/00_看结果/README_从这里开始.md)
- [可播放图册](runs/20261022/00_看结果/评审图册.html)
- [中文最终结论](runs/20261022/00_看结果/最终实验结论.md)
- [训练指标](runs/20261022/metrics/train_summary.csv) / [验证指标](runs/20261022/metrics/val_summary.csv)
- [C/U/S权重](runs/20261022/checkpoints/) / [完成状态](runs/20261022/progress.json)

普通LoRA相对Base的验证缺损MAE降低64.21%。S相对U的验证MAE降低1.80%、结构误差降低0.19%；均值优势集中于CS12，精细屋檐折线和闭环仍有问题，尚不能认定S稳定胜出。

## 本目录结构

| 位置 | 用途 |
| --- | --- |
| scripts/ | 13个专用脚本；唯一维护位置 |
| configs/pilot_20261022.json | 指向原run冻结配置的链接，不生成第二份配置 |
| runs/20261022/ | 指向原run的目录链接，不复制大文件 |
| README.md | 问题、完成状态和使用入口 |

物理结果保持在 outputs/ComplexSingleOrbit_Rev1_Pilot_20261022/。原run/scripts为指向本目录scripts的兼容链接，原resume.sh和旧命令仍有效。新的独立想法必须创建新实验目录，不能直接修改本run的冻结配置或复用其完成标记。

## 源码入口

- [complex_common.py](scripts/complex_common.py)：数据/标签/路径与冻结配置。
- [complex_train.py](scripts/complex_train.py)：C/U/S损失、更新与恢复。
- [complex_infer.py](scripts/complex_infer.py)：完整81帧原始预测。
- [complex_evaluate.py](scripts/complex_evaluate.py)：统一指标和同步媒体。
- [run_pilot.py](scripts/run_pilot.py)：串行执行与已完成复用。
- [status.py](scripts/status.py)：只读状态。
- prepare.py、review_frames.py、finalize.py、write_final_report.py、native_exit.py及两个test文件负责已有准备、复核和CPU检查。

只读状态命令（从项目根目录执行）：

```bash
/home/shi/miniconda3/envs/scj/bin/python experiments/vace/complex12_rev1/scripts/status.py
```

## 实验边界

训练CS01/03/05/06/07/09/10/11，验证CS02/04/08/12；17帧窗口训练，81帧整轨推理，512×512、16FPS、30步、CFG5，训练seed20261022、推理seed20261018。L为旧P04/P06参考；C/U/S使用共同新初始化与随机序列，各自新优化器。正式只比较step300。

实际输入只有Damaged派生条件、ModelMask和统一文本，GT与结构图只用于监督/评测。无回贴、独立裁剪或强制闭环。[原任务书](runs/20261022/task_brief.txt)、[标签映射](runs/20261022/mask_loss_mapping.md)、[共同采样表](runs/20261022/shared_sampling_table.csv)保留原样。

B/CS04保存后原生异常的有效输出已独立复核；C从40更新恢复。没有认定崩溃根因已解决，也没有自动追加预算。
