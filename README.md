# 建筑视频 Wan LoRA 与 VACE 条件补全

P01–P08已完成原始VACE-1.3B的81帧补全、三方可视化和CPU缺口回贴。当前研究推进到P04/P06的17帧条件补全LoRA小样本验证：检查模型能否学会训练题上的完整表面恢复。

首轮已由用户在服务器终端执行并正常完成。2026-10-20 17:05:32（北京时间）核查：300次真实更新、两例各150次、6次17帧推理全部完成，100/300步LoRA与两例各四项成果齐全，总耗时749.247秒。五个代表视角中P04屋顶凹槽和P06开口均明显改善；完整视频时序和进一步人工验收待查看。当前先查看结果，不重复启动。

详细数据关系、参数、运行证据、成果说明和判断标准见 [P04/P06 LoRA实验说明](docs/VACE_P04_P06_LORA_SMOKE.md)；日常操作见 [TRAINING.md](TRAINING.md)。

## 开发入口

- [项目规则](AGENTS.md)：编辑、训练授权、数据保护及检查要求。
- [当前状态](PROJECT_CONTEXT.md)：以实际产物为依据，区分历史报告和已核对事实。
- [操作手册](TRAINING.md)：当前 A6000 预检、续训和验证命令。
- [实验台账](EXPERIMENTS.md)：双 3060、A100 和 A6000 的独立实验记录。
- [会话交接](SESSION_HANDOFF.md)：下一次开发的入口与未解决问题。
- [BuildingFiller P01 与 VACE 准备](docs/BUILDINGFILLER_P01_VACE_HANDOFF.md)：三机边界、PowerShell SCP 命令、实际数据验证与 17 帧最小实验方案。
- [P04/P06 VACE LoRA实验](docs/VACE_P04_P06_LORA_SMOKE.md)：当前两例17帧训练、Base/100/300公平比较、GT监督路径、显存说明及结果检查。
- [VACE 权重与执行历史](docs/VACE_P01_EXECUTION.md)：官方模型、八例81帧执行、P01复用和既往故障证据。
- [2026-10-14 审计](docs/PROJECT_AUDIT_20261014.md)：已确认的问题、修改及验证范围。

## 代码与数据流

历史 T2V：建筑渲染视频 → 固定 train/val/test 划分 → 视频 latent/文本 context 缓存 → DiT LoRA 训练 → 固定 prompt/seed 视频对比 → 人工检查与像素诊断。

已完成VACE基线：BuildingFiller原始包 → 真实样本/帧号验收 → ModelMask适配 → 原始VACE整序列81帧 → `outputs/vace/test/01–08/`。每例四文件包含原始预测和Before/Prediction/GT对照。

已完成CPU诊断：原始预测 + Damaged + 原推理ModelMask → 只替换缺口的Composite → `outputs/vace/composite_check/01–08/`。保留区域改善，P04凹槽和P06黑洞仍在。

当前条件训练：P04/P06原始PNG连续17帧 → GT监督目标 + Damaged/Mask/文本条件 → 同一份vace LoRA → 同条件17帧Base/LoRA-100/LoRA-300比较 → `outputs/vace/lora_roof_smoke/run_<时间>/`。旧T2V LoRA和缓存不参与。

- `code/Python_3D_Scanner/training/`：项目训练入口、历史数据工具、验证和报告工具。
- `code/DiffSynth-Studio/`：第三方引擎；项目逻辑优先放在项目层。
- `scripts/`：当前 A6000 入口与历史 A100/Slurm 入口。不要仅凭文件名选择环境。
- `configs/vace_roof_smoke_v1.json`：当前两例LoRA固定窗口、训练预算、推理参数和文本。
- `configs/single_orbit_same_prompt_v1/`：历史T2V冻结划分，1992/234/118，共2344个视频。
- `data/`、`models/`、`cache/`、`checkpoints/`、`outputs/`：原始或生成产物，默认只读。
- `logs/`：运行日志和历史暂停记录；旧记录中的路径可能已迁移。
- `docs/`：审计、兼容入口和 Claude worker 说明；当前状态统一指向根目录文档。
- `.pre_git_sync_20260924/`：历史快照，不作为当前运行入口。

当前 A6000 环境使用 `/home/shi/miniconda3/envs/scj/bin/python`。扫描器的 `requirements.txt` 属于历史 Windows GUI 环境，不是训练环境完整依赖清单。

扫描器 README 描述的桌面采集、OBJ 冻结与三元组渲染是历史设计；当前工作副本没有 `code/Python_3D_Scanner/src/` 和对应数据集工作流文档。需要从原始采集项目恢复并审查后再开发这部分，不能据此声称桌面工具可运行。

本轮LoRA已由用户手动授权并启动；文档维护只读取进度，不重启现有任务。训练题效果需结合真实可视化判断，不能写成独立测试成绩或81帧一致性结论。
