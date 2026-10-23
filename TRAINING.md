# 服务器操作与验证手册

文档核对日期：2026-10-23（UTC+09:00）。本页区分只读查看、CPU准备和GPU执行。现有VACE正式实验均已完成，日常先查看结果和状态。

## 1. 当前环境

| 项目 | 实际值 |
| --- | --- |
| 项目根 | /mnt/windowsE/chuanjun/building_wan |
| Python | /home/shi/miniconda3/envs/scj/bin/python |
| Python / torch / CUDA runtime | 3.10.13 / 2.11.0+cu128 / 12.8 |
| DiffSynth / PEFT / safetensors | 2.1.5 / 0.17.1 / 0.7.0 |
| 计算方式 | 单卡A6000，单进程，当前训练不使用DDP |
| 代码版本 | 整理结构提交ccfe24d；后续以git log实际结果为准 |

旧A100的mv2v、/home/share路径和Slurm命令不适用于本机。扫描器requirements.txt是旧GUI依赖，不是当前训练环境的安装清单。本次不安装或更新依赖。

## 2. 日常只读查看

从项目根运行：

```bash
cd /mnt/windowsE/chuanjun/building_wan
/home/shi/miniconda3/envs/scj/bin/python -B experiments/vace/complex12_rev1/scripts/status.py
git status --short --branch
git log -3 --oneline
tmux list-sessions
```

Complex12应显示complete、60/60、C/U/S各300。状态脚本不加载模型、不导入torch、不改变progress.json。tmux会话不存在只说明终端历史未保留，不代表成果丢失；不要为查看状态重新启动训练入口。

查看旧LoRA17的实际记录：

```bash
/home/shi/miniconda3/envs/scj/bin/python -B - <<'PY'
import csv, json
from pathlib import Path
run = Path('outputs/vace/lora_roof_smoke/run_20261020_165237_150325_2048720')
record = json.loads((run / 'training/experiment.json').read_text())
with (run / 'training/loss.csv').open(newline='') as f:
    history = list(csv.DictReader(f))
print(record['runtime'])
print('Saved update records:', len(history))
PY
```

## 3. 启动脚本的实际行为

| 入口 | 无参数行为 | 只读/CPU方式 | 对应范围 |
| --- | --- | --- | --- |
| run_vace_p01_465_17f.sh | 会进入GPU执行 | --help；--dry-run做模型/CPU预检 | 历史P01 17帧 |
| run_vace_p01_465_81f.sh | 会进入GPU执行 | --help；--dry-run | 历史P01整轨 |
| run_vace_building_test.sh | 会进入GPU批量执行 | --help；--dry-run；--prepare会复制P01 | 八例Base |
| run_vace_roof_smoke.sh | CPU准备 | --help或--dry-run；正式用--run/--tmux | 两例17帧LoRA |
| run_vace_lora300_eval81.sh | 无参数为CPU预检；--run-trained/--run-transfer启动GPU | --dry-run；--render-only仅CPU制图，见[说明](experiments/vace/pilot8_lora81/README.md) | 八例LoRA-300 |
| 原Complex12 run/resume.sh | 若未完成会恢复任务；已完成则直接返回 | 日常改用status.py | 当前固定run |
| train_wan81_a6000.sh | 正式历史T2V训练 | 阅读脚本和配置；没有通用--dry-run承诺 | A6000历史T2V |
| resume_wan81_a6000.sh | 正式历史T2V续训 | --dry-run | 平铺权重的weights-only续训 |

预检可能读取大权重哈希，CPU准备可能产生新文件，均不等于廉价状态查询。不要因文档列出命令就执行；GPU任务须有当前明确授权。历史默认输出策略与新目录规范不同，新实验要先明确新的独立run路径。

## 4. 恢复方式不能混用

| 实验 | 可恢复内容 | 不能推断的内容 |
| --- | --- | --- |
| Complex12 | resume_latest.pt中的LoRA、AdamW、已完成行及Python/NumPy/torch/CUDA RNG | 不可跨配置、方法或初始化恢复 |
| 两例LoRA17 | 100/300的LoRA增量可供推理 | 不能恢复完整optimizer/RNG；入口无完整训练resume |
| 历史T2V | --lora_checkpoint加载LoRA并按来源参数换算样本预算 | 不恢复旧optimizer、RNG、shuffle或累积中间态 |

Complex12成功任务按配置ID和内容完整性复用。当前run已经complete；不要删除完成标记以强制重跑。历史T2V的step是logger物理batch计数，换batch时必须从来源段参数换算样本数；不把最高文件名当作训练完成证据。

## 5. 真实进度与故障判断

在可见tmux终端运行经过授权的任务，显示样本身份、方法、真实优化器更新/去噪步数、耗时、保存状态和已完成轨迹。断开SSH后重连原会话，先确认实际进程，避免双重写同一run。

出现OOM、非有限值、断梯度或进程异常时保留输出和真实退出码，按该实验授权范围处理。已保存文件需独立验证；不能只因有权重或一张图片就标记成功。B/CS04的−6退出及历史T2V堆损坏根因仍未确认。

显存容量占比、PyTorch allocated/reserved和GPU计算利用率分别记录。nvidia-smi不在PATH、黄色告警、瞬时低利用率都不能单独判定失败。只有状态和资源查询可靠时才能决定是否启动；本次文档更新不进行GPU试算。

## 6. 修改后规定的CPU检查

```bash
git diff --check
bash -n scripts/train_wan81_a6000.sh scripts/resume_wan81_a6000.sh
CUDA_VISIBLE_DEVICES='' /home/shi/miniconda3/envs/scj/bin/python -m py_compile code/Python_3D_Scanner/training/wan_train_entry.py scripts/validate_wan81_a6000.py
CUDA_VISIBLE_DEVICES='' /home/shi/miniconda3/envs/scj/bin/python -m unittest discover -s scripts/tests -v
git status --short
```

修改Complex12计算代码时，另运行其8项专用CPU检查：

```bash
CUDA_VISIBLE_DEVICES='' /home/shi/miniconda3/envs/scj/bin/python -m unittest discover -s experiments/vace/complex12_rev1/scripts -p 'test_*.py' -v
```

GUI测试依赖缺失src/，不能把上述CPU回归写成全部GUI能力通过。文档验证还应检查相对链接、样本映射、命令默认行为及冻结报告的口径一致性。

## 7. 交接与Git

更新根状态、台账、操作和交接文档；记录实际run/代码位置、更新数、退出证据、结论及局限。仅提交相关文件，保留其他未提交工作。检查差异与GitHub账号后按main流程推送，并在服务器执行git pull --ff-only origin main，不强推。

模型、视频、日志与缓存留在原目录。详细旧操作和失败过程见 [文档历史索引](docs/DOCUMENTATION_INDEX.md)。
