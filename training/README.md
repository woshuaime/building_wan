# 建筑视频 LoRA 实验工具

本目录保存项目自己的实验配置和入口。`DiffSynth-Studio` 只作为第三方训练/推理引擎，不在其官方示例中维护项目逻辑。

## 当前阶段

`smoke.json` 对应已经完成的20视频链路测试：

- Wan2.1-T2V-1.3B
- 256x256
- 17帧
- LoRA rank 8
- 1 epoch / 20 steps
- GPU 1

这一阶段只验证训练链路和领域 LoRA 是否可加载，不代表最终建筑补全效果。

## 1. 配置预检

在 `D:\code\Python_3D_Scanner` 中运行：

```bat
D:\scj\envs\diffsynth\python.exe training\train_domain_lora.py --config training\configs\smoke.json
```

不带 `--run` 时只检查数据、路径和参数，并打印实际训练命令，不会占用 GPU。

### 分布式启动和崩溃防护

配置中的 `runtime.distributed_backend` 使用 `auto`：Windows 自动选择 Gloo，Linux 自动选择 NCCL。Linux 训练由 `torch.distributed.run --standalone` 管理进程，避免在 CUDA 进程外再嵌套 `torch.multiprocessing.spawn`。Windows 仍使用 TCP rendezvous 的兼容启动器。

训练进程默认限制 OpenMP/MKL 线程、关闭 tokenizer 并行、启用 Python/C++ 堆栈和 CUDA 异步错误报告。这样发生原生库错误时会留下 rank、C++ 栈和最后一个阶段，而不是只有一行 `core dumped`。

缓存阶段只允许单卡顺序执行。已有完整缓存会自动跳过，部分缓存必须显式使用 `--allow-existing` 继续；缓存数量不等于元数据样本数时，训练阶段会拒绝启动，避免把半成品缓存送入 DDP。

服务器端 A100 的建议顺序：

```bash
cd /home/share/CHUANJUN/building_wan
git pull --ff-only
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  training/check_dual_gpu.py --backend nccl
python training/train_split_domain_lora.py \
  --config training/configs/server_a100.json --stage cache --run
python training/train_split_domain_lora.py \
  --config training/configs/server_a100.json --stage train --run
```

`server_a100.json` 需要填写服务器自己的绝对路径，`runtime.distributed_backend` 保持 `auto`。如果缓存阶段中断，重新执行 cache 命令时加 `--allow-existing`；缓存完整后直接执行 `--stage train --run`。训练阶段不需要也不应该再启动第二个 `torchrun`。

## 2. 启动训练

```bat
D:\scj\envs\diffsynth\python.exe training\train_domain_lora.py --config training\configs\smoke.json --run
```

如果输出目录已经包含权重，脚本会主动停止以防覆盖。确认确实需要重跑时，额外添加 `--allow-existing`。

当前 smoke 权重已经生成，无需再次训练。

## 3. 验证预检

```bat
D:\scj\envs\diffsynth\python.exe training\validate_domain_lora.py --config training\configs\smoke.json
```

该命令确认基础模型、step-10、step-20和输出路径，不会加载模型。

## 4. 生成公平对比视频

```bat
D:\scj\envs\diffsynth\python.exe training\validate_domain_lora.py --config training\configs\smoke.json --run
```

脚本只加载一次基础模型，并使用完全相同的 prompt、negative prompt、seed和推理参数依次生成：

```text
outputs/validation/building_smoke/base.mp4
outputs/validation/building_smoke/step-10.mp4
outputs/validation/building_smoke/step-20.mp4
outputs/validation/building_smoke/validation_metadata.json
```

如果中途中断，可用 `--only step-20 --run` 只生成指定结果。已有文件默认不会覆盖，覆盖时添加 `--overwrite`。

## 5. 汇总结果

```bat
D:\scj\envs\diffsynth\python.exe training\summarize_run.py --config training\configs\smoke.json
```

输出 `outputs/validation/building_smoke/experiment_summary.json`，记录训练参数、loss摘要、checkpoint和验证状态。

## 判断标准

首先检查工程正确性：三个视频均成功输出，元数据中的 prompt、seed、尺寸、帧数完全一致，LoRA加载无报错。

然后观察模型变化：是否更像单体建筑模型、是否保持白色干净背景、是否出现连续旋转、几何形状是否跨帧稳定。20步结果即使没有明显改善也不代表路线失败，它只用于 smoke test。

通过上述检查后，再建立固定 train/val/test 划分和200视频试验配置，不直接开始2390视频全量训练。

## building_v1 固定划分

生成命令：

```bat
D:\scj\envs\diffsynth\python.exe training\prepare_splits.py
```

默认使用随机种子 `20260902`，按 `video_manifest.json` 中的 `source_obj` 分组，而不是按单个视频随机拆分。已冻结的 `building_v1` 结果为：

```text
train:     1912
val:        239
test:       239
pilot_200:  200（仅从train中选择）
```

输出位于 `training/data_splits/building_v1`。`split_manifest.json` 记录输入文件SHA-256、随机种子、样本数和集合交叉审计；`split_index.csv` 记录每个视频对应的源OBJ和集合。已有划分默认不覆盖，重建时必须显式添加 `--overwrite`。

## 200视频试训

先做配置预检：

```bat
D:\scj\envs\diffsynth\python.exe training\train_domain_lora.py --config training\configs\pilot_200.json
```

确认后启动一轮200步训练：

```bat
D:\scj\envs\diffsynth\python.exe training\train_domain_lora.py --config training\configs\pilot_200.json --run
```

该实验保持256x256、17帧、rank 8和相同学习率，只改变训练样本规模，便于与20视频smoke实验比较。权重在第50、100和200步保存。

训练完成后生成公平对比：

```bat
D:\scj\envs\diffsynth\python.exe training\validate_domain_lora.py --config training\configs\pilot_200.json --run
```
