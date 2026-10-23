# 历史迁移实验模板（兼容入口）

2026-10-14：当前状态统一维护在根目录 [EXPERIMENTS.md](../EXPERIMENTS.md)，操作规则见 [AGENTS.md](../AGENTS.md)，开发审计见 [PROJECT_AUDIT_20261014.md](PROJECT_AUDIT_20261014.md)。仓库及 scj 环境已建立，单轨权重和验证产物已存在；不要把下面的迁移初期快照当作当前事实或执行指令。

<details>
<summary>迁移初期历史原文（保留证据，已失效）</summary>

# Experiments

## Migration note

The root-level `EXPERIMENTS.md` records experiments performed on the former school A100/Slurm cluster. Preserve it as historical evidence. Do not relabel those runs as A6000 results.

No laboratory A6000 training experiment has been run from this working copy yet.

## A6000 experiment template

- Date/time:
- Objective:
- Host/GPU:
- Git commit or explicit "Git unavailable":
- Python/PyTorch/CUDA:
- Model/config:
- Dataset/split:
- Batch size / accumulation:
- Learning rate / schedule:
- Input checkpoint:
- Output path:
- Log path:
- Short smoke test result:
- Peak GPU memory:
- Runtime:
- Loss summary:
- NaN: yes/no/not checked
- OOM: yes/no/not checked
- Checkpoint result:
- Final status:
- Failure cause:
- Useful observations:

</details>
