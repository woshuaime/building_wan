# Training

本文档记录学校服务器上当前 `building_wan` 单卡A100训练的实际操作。所有命令都以项目根目录 `/home/share/CHUANJUN/building_wan` 为准。

## 1. 激活环境

交互式检查时可以激活：

```bash
source /home/u2025171963/miniconda3/etc/profile.d/conda.sh
conda activate mv2v
```

训练脚本直接使用以下绝对路径，因此Slurm作业不依赖交互式激活：

```text
/home/u2025171963/miniconda3/envs/mv2v/bin/python
```

不要使用旧环境 `mv2v_dual`。

## 2. 检查 Python

```bash
/home/u2025171963/miniconda3/envs/mv2v/bin/python --version
/home/u2025171963/miniconda3/envs/mv2v/bin/python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.version.cuda)"
```

登录节点没有GPU分配，因此这里的 `torch.cuda.is_available()` 可能为False。GPU可用性由训练脚本在Slurm作业内部检查。

## 3. 检查依赖

```bash
/home/u2025171963/miniconda3/envs/mv2v/bin/python -m pip check
PYTHONPATH=/home/share/CHUANJUN/building_wan/code/DiffSynth-Studio \
/home/u2025171963/miniconda3/envs/mv2v/bin/python -c "import torch, accelerate, transformers, safetensors, librosa, diffsynth; print('imports OK')"
```

## 4. 运行安全检查

```bash
cd /home/share/CHUANJUN/building_wan
bash -n scripts/train_wan81.sh scripts/submit_wan81.sh scripts/start_wan81.sh
/home/u2025171963/miniconda3/envs/mv2v/bin/python -m py_compile \
  code/Python_3D_Scanner/training/wan_train_entry.py
wc -l data/splits/building_v1/metadata_train.csv
find cache/wan21_t2v_13b_building_384x384_f81_train -type f -name '*.pth' | wc -l
squeue -u u2025171963
```

`metadata_train.csv` 应为1913行（表头加1912个样本）。提交前队列中不能已有另一个写相同缓存或输出目录的作业。

## 5. 提交 Slurm

只有用户明确要求“开始提交”后才能执行：

```bash
cd /home/share/CHUANJUN/building_wan
sbatch scripts/submit_wan81.sh
```

若希望提交后直接在当前VS Code SSH终端显示日志，可执行：

```bash
cd /home/share/CHUANJUN/building_wan
bash scripts/start_wan81.sh
```

`start_wan81.sh` 内部会执行 `sbatch`。按Ctrl+C只停止本地日志显示，不取消Slurm任务。

如果本地显示中断，但Slurm任务仍在运行，传入真实Job ID重新连接，不会再次提交：

```bash
cd /home/share/CHUANJUN/building_wan
bash scripts/start_wan81.sh <jobid>
```

跟踪器遇到`squeue`临时网络或socket超时时会继续显示日志，只有Slurm明确返回终止状态才结束。

## 6. 查看任务状态

```bash
squeue -u u2025171963
```

指定任务：

```bash
squeue -j <jobid> -o '%.18i %.24j %.2t %.10M %.10l %R'
```

## 7. 查看历史结果

```bash
sacct -j <jobid> --format=JobID,JobName,State,ExitCode,Elapsed,Start,End -X
```

`COMPLETED` 且 `ExitCode=0:0` 才表示成功完成。

## 8. 实时查看 stdout/stderr

```bash
tail -F \
  /home/share/CHUANJUN/building_wan/logs/slurm/building_wan-<jobid>.out \
  /home/share/CHUANJUN/building_wan/logs/slurm/building_wan-<jobid>.err
```

必须把 `<jobid>` 替换为 `sbatch` 返回的真实任务号，不要使用示例任务号。

## 9. 取消任务

确认任务号后执行：

```bash
scancel <jobid>
```

取消后检查：

```bash
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed -X
```

## 10. Cache 与 checkpoint

- 紧凑缓存：`/home/share/CHUANJUN/building_wan/cache/wan21_t2v_13b_building_384x384_f81_train`
- 每个 `.pth` 只保存该视频的BF16 latent和必要标量；相同prompt的BF16文本context单独保存一次。
- 完整缓存应包含1912个 `.pth`、共享context `.shared-text-context-bf16.pt` 和标记 `.compact-sft-shared-context-bf16-v2`。
- 模拟大小为：单样本约1.55MB、共享context约4.20MB、完整1912份约2.764GiB。
- 当前实验checkpoint：`/home/share/CHUANJUN/building_wan/checkpoints/building_wan`
- 保存间隔：每400步一个LoRA `.safetensors`。
- 训练结束时的最终LoRA也在该输出目录中，以实际生成的最高step文件为准。
- 历史双3060权重：`checkpoints/building_train_1912_all_256_cooled_lora`，只保留，不作为当前resume来源。

## 11. 正常阶段标志

首次生成缓存时日志包含：

```text
Cache is empty; generating compact 81-frame cache.
```

以后复用完整缓存时日志包含：

```text
Reusing complete compact-sft-shared-context-bf16-v2 cache: 1912 files.
```

进入正式训练时日志包含：

```text
Cache ready; starting LoRA training.
```

## 12. 出错后先检查

1. 用 `sacct` 确认任务最终状态和退出码。
2. 查看对应 `.err` 最后100行，再查看 `.out`。
3. 检查是否有两个作业同时写同一缓存目录。
4. 检查缓存数量、目录大小和格式标记。
5. 检查Python路径是否仍为 `mv2v`，以及DiffSynth是否能导入。
6. 区分无害警告和致命异常：NVML警告通常不是直接失败原因，Python traceback末尾才是直接原因。
7. 不要在原因未确认时连续重复提交。

常用诊断命令：

```bash
tail -n 100 logs/slurm/building_wan-<jobid>.err
tail -n 100 logs/slurm/building_wan-<jobid>.out
find cache/wan21_t2v_13b_building_384x384_f81_train -type f -name '*.pth' | wc -l
du -sh cache/wan21_t2v_13b_building_384x384_f81_train
```

## 13. A100效果验证

当前固定验证配置：

```text
code/Python_3D_Scanner/training/configs/building_wan_a100_validation.json
```

它使用同一提示词、negative prompt和seed，比较基础模型、step-4000、step-8000、step-12000和最终step-15296，均生成384×384、81帧视频。

只读预检：

```bash
cd /home/share/CHUANJUN/building_wan/code/Python_3D_Scanner
/home/u2025171963/miniconda3/envs/mv2v/bin/python \
  training/validate_domain_lora.py \
  --config training/configs/building_wan_a100_validation.json
```

只有用户明确要求开始提交验证后才能执行：

```bash
cd /home/share/CHUANJUN/building_wan
sbatch scripts/submit_wan81_validation.sh
```

验证输出目录：

```text
/home/share/CHUANJUN/building_wan/outputs/validation/building_wan
```

## 14. A6000批训练变体

A6000版本使用独立脚本和独立输出目录，默认训练数据是单轨视频，根目录是服务器上的 `/mnt/windowsE/chuanjun/building_wan`：

```bash
cd /mnt/windowsE/chuanjun/building_wan
bash -n scripts/train_wan81_a6000.sh scripts/submit_wan81_a6000.sh
```

默认参数：

```text
物理 batch：2
梯度累积：2
有效 batch：4
分辨率：384×384
帧数：81
数据：data/single_orbit_videos
清单：configs/single_orbit_same_prompt_v1/metadata_train.csv
输出：checkpoints/building_wan_a6000_single_orbit
```

训练入口会把缓存中的多个样本合并成真正的batch，并为每个样本独立采样flow-matching时间步和噪声。这一轮所有样本使用同一条 prompt，因此缓存只保存一份共享文本 context；后续逐栋 prompt 需要另行重建缓存。这样不会出现旧DataLoader只保留每批第一个样本的问题。

本轮不依赖 Cap3D 的逐栋描述。仓库中的 `configs/single_orbit_same_prompt_v1` 已包含统一 prompt 的训练、验证、测试清单；服务器同步代码后即可使用。Cap3D 生成的独立 prompt 暂不混入本轮。

显存预检或首轮运行时可以覆盖参数：

```bash
WAN_TRAIN_BATCH_SIZE=1 WAN_GRADIENT_ACCUMULATION_STEPS=4 \
  bash scripts/train_wan81_a6000.sh
```

只有用户明确要求开始提交后，才运行：

```bash
sbatch scripts/submit_wan81_a6000.sh
```

该脚本会拒绝非A6000 GPU，并拒绝混用已有checkpoint。服务器上的实际Python路径、数据路径或Slurm GRES名称不一致时，先通过对应环境变量或提交脚本头部调整。
