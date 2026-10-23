# 操作手册

本项目正式实验当前均已停止。先从 [实验目录](experiments/README.md) 选择研究想法，再查看该实验说明；不要仅凭旧脚本名启动任务。

## 环境

服务器项目根为 /mnt/windowsE/chuanjun/building_wan，Python为 /home/shi/miniconda3/envs/scj/bin/python，单卡RTX A6000。历史A100的mv2v环境、Slurm路径和Windows采集环境分别保留，不作当前默认环境。

## 只读查看Complex12

以下命令只读取进度，不加载模型或启动任务：

```bash
cd /mnt/windowsE/chuanjun/building_wan
/home/shi/miniconda3/envs/scj/bin/python experiments/vace/complex12_rev1/scripts/status.py
```

当前应显示complete、60/60轨迹、C/U/S均300。结果从 [00_看结果](outputs/ComplexSingleOrbit_Rev1_Pilot_20261022/00_看结果/README_从这里开始.md) 打开，视频与权重维持原路径。

原run的resume.sh仍保留。已完成状态下它直接返回完成信息；它是恢复入口，不能当作日常状态查询。只有用户授权继续一个未完成run时才执行恢复。需要新实验时，创建新的想法/运行目录，不修改这一完成run的配置或清除完成标记。

## 修改后的CPU检查

```bash
git diff --check
bash -n scripts/train_wan81_a6000.sh scripts/resume_wan81_a6000.sh
CUDA_VISIBLE_DEVICES='' /home/shi/miniconda3/envs/scj/bin/python -m py_compile code/Python_3D_Scanner/training/wan_train_entry.py scripts/validate_wan81_a6000.py
CUDA_VISIBLE_DEVICES='' /home/shi/miniconda3/envs/scj/bin/python -m unittest discover -s scripts/tests -v
git status --short
```

Complex12专用CPU回归：

```bash
CUDA_VISIBLE_DEVICES='' /home/shi/miniconda3/envs/scj/bin/python -m unittest discover -s experiments/vace/complex12_rev1/scripts -p 'test_*.py' -v
```

## 已有入口按实验查找

| 实验 | 入口说明 |
| --- | --- |
| 八例Base及P01准备 | [Pilot8 Base](experiments/vace/pilot8_base/README.md) |
| CPU硬边界回贴 | [Composite历史诊断](experiments/vace/pilot8_composite/README.md) |
| 两例17帧LoRA | [LoRA17](experiments/vace/pilot8_lora17/README.md) |
| 八例81帧LoRA评估 | [LoRA81](experiments/vace/pilot8_lora81/README.md) |
| 十二栋配对实验 | [Complex12](experiments/vace/complex12_rev1/README.md) |
| A6000历史T2V训练与续训 | [单轨T2V](experiments/t2v/single_orbit/README.md) |

## 故障和交接

先保留现有结果，确认真实进程退出码、实际优化器更新、配置和可读文件；不把权重文件名、loss下降或低显存占用直接等同于成功/失败。当前目录整理没有更改任何模型计算或损失定义，也未修复历史原生崩溃。

GPU运行结束后依项目规则更新当前状态、实验台账及交接，只提交相关代码和文档。大文件留在原结果目录。历史完整命令和故障说明见 [整理前TRAINING.md](docs/history/20261023_before_structure/TRAINING.md)，其中运行命令不构成新授权。
