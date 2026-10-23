# VACE模型、条件与既有执行入口

更新：2026-10-23（UTC+09:00）。正式VACE-1.3B权重已在本机，Pilot8及Complex12正式实验均已完成。本页不再重复列出已经执行的“首次启动/补跑”步骤；完整早期过程见 [更新前记录](history/20261023_before_md_refresh/docs/VACE_P01_EXECUTION.md)。

## 1. 当前模型和环境

| 组件 | 实际位置或版本 |
| --- | --- |
| 项目 | /mnt/windowsE/chuanjun/building_wan |
| Python | /home/shi/miniconda3/envs/scj/bin/python，3.10.13 |
| PyTorch / CUDA runtime | 2.11.0+cu128 / 12.8 |
| DiffSynth / PEFT / safetensors | 2.1.5 / 0.17.1 / 0.7.0 |
| VACE模型 | Wan-AI/Wan2.1-VACE-1.3B |
| revision | 574e6a744642ce3bee319afc31496b88bde8aac4 |
| 主权重 | models/Wan2.1-VACE-1.3B/<revision>/diffusion_pytorch_model.safetensors |
| T5 | models/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors |
| VAE | models/Wan-Series-Converted-Safetensors/Wan2.1_VAE.safetensors |
| Tokenizer | models/Wan2.1-T2V-1.3B/google/umt5-xxl |

批准模型身份来自 [model_sources](../configs/vace_p01_model_sources_v1.json)。主文件7,146,067,848 bytes，包含1264个张量、439个VACE张量。原正式身份校验已完成，本次没有重新下载或做全量权重哈希。

复用T5/VAE/Tokenizer不等于加载T2V骨干或旧T2V LoRA。原始VACE与每个LoRA阶段采用fresh模型进程，避免融合叠加；不能用clear_lora假定撤销已融合权重。

## 2. 已完成的实验与范围

| 实验 | 适用条件与状态 |
| --- | --- |
| P01早期17帧 | 原B005–B021；旧结果曾由用户删除，不补造日志或恢复输出 |
| P01及八例Base81 | 81帧完整条件补全；[结果与编号](../experiments/vace/pilot8_base/README.md) |
| 八例Composite | CPU硬边界回贴诊断；[独立说明](../experiments/vace/pilot8_composite/README.md) |
| P04/P06 LoRA17 | B000–B016、300更新、六次17帧推理；[完整参数](VACE_P04_P06_LORA_SMOKE.md) |
| 八例LoRA81 | 同一LoRA-300、八例81帧、Base全复用；[观察结论](../experiments/vace/pilot8_lora81/README.md) |
| Complex12 Rev1 | 独立8训练/4验证、C/U/S各300、五方法60轨迹；[专属说明](../experiments/vace/complex12_rev1/README.md) |

Pilot8统一使用中文正向描述；Complex12使用冻结的英文正向描述。两者均512×512/16FPS/30步/CFG5/推理seed20261018，但不能因此声称所有文本与训练条件完全相同。

## 3. 数据与条件合同

Exact为原始L模式0/255，转换相同RGB通道的ModelMask。Damaged是原始RGB，Mask白区填灰127、黑区不变。推理条件只包含这些派生图、Mask和文本；GT/结构标签仅监督或后验评价。

全黑Mask表示缺口在该视角不可见，仍保留该帧。不扩Mask、不羽化、不改变Exact、不把GT贴回预测。81帧一次生成，080始终是模型的真实输出。

VACE条件编码使用inactive/reactive冻结VAE latent及64通道Mask，形成96通道上下文。缓存冻结条件编码与缓存可训练VACE输出是不同操作；后者会破坏LoRA训练链。冻结DiT参数也必须保留向LoRA传梯度的计算路径。

## 4. 默认命令行为

| 入口 | 默认行为/正式开关 | 只读或CPU分支 |
| --- | --- | --- |
| [P01 17帧](../scripts/run_vace_p01_465_17f.sh) | 无参数启动推理 | --dry-run预检 |
| [P01 81帧](../scripts/run_vace_p01_465_81f.sh) | 无参数启动；--tmux可见会话 | --dry-run预检 |
| [八例Base](../scripts/run_vace_building_test.sh) | 无参数运行；--resume复用有效成功前缀 | --dry-run预检，--prepare可能复制P01 |
| [LoRA17](../scripts/run_vace_roof_smoke.sh) | --run/--tmux启动 | 无参数/--dry-run为CPU准备 |
| [LoRA81](../scripts/run_vace_lora300_eval81.sh) | --run-trained/--run-transfer启动 | 无参数/--dry-run预检，--render-only制图 |
| [Complex12 status](../experiments/vace/complex12_rev1/scripts/status.py) | 只读已有状态 | 不需要运行resume.sh |

这些入口描述现有实现，不构成新一轮执行授权。部分预检会完整读取大模型，不能代替轻量状态查询；CPU准备也可能写文件。当前所有VACE任务均完成，优先打开结果。

## 5. 媒体、异常与完成证据

Base结果布局曾由两栏升级为Before/Base/GT三栏，原始预测保持；LoRA81为Before/Base/LoRA/GT四栏；Complex12为Before/GT/B/L/C/U/S的2×4布局。同步帧号和主体比例一致，媒体可读与原始预测正确对应是完成条件之一。

历史Base P04在推理前T5哈希检查异常，后续授权继续；LoRA81首次P04推理后CPU制图异常，授权修复重跑；Complex12 B/CS04保存完整预测后原生退出−6，经独立核验保留。三种情况不同，不能合并写成“全部首次成功”或“全部是OOM”。

Complex12 C在服务器挂载恢复后从40真实更新恢复到300；U/S各300，五方法60轨迹完成。旧P01未保留完整manifest时不从视频推断seed；所有历史限制和原生故障根因仍保留。

## 6. 维护与当前操作

配置、输出、训练代码和历史事实分别查 [configs](../configs/README.md)、[outputs](../outputs/README.md)、[scripts](../scripts/README.md) 与 [实验台账](../EXPERIMENTS.md)。当前终端状态、失败处理和规定CPU检查统一见 [TRAINING](../TRAINING.md)。

既有tmux名称和PID只是当时记录，不能保证重启后仍存在。本次没有创建终端任务、调整环境、重新训练或推理；原run报告和冻结配置不因文档整理改写。
