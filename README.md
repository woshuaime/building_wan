# 建筑视频 Wan LoRA 与 VACE 条件补全

当前准备 BuildingFiller 源 465 的 VACE 条件补全基线：残缺图像序列 + 独立适配 Mask + 文本 → 补全视频，GT 仅用于评估。512 交接包已在 Linux 服务器通过完整性检查；尚缺可读 VACE 权重及项目专用执行器，没有运行补全模型。历史 Wan2.1-T2V-1.3B 单轨统一 prompt LoRA 实验保持独立，不能把它当作已有补全能力。

## 开发入口

- [项目规则](AGENTS.md)：编辑、训练授权、数据保护及检查要求。
- [当前状态](PROJECT_CONTEXT.md)：以实际产物为依据，区分历史报告和已核对事实。
- [操作手册](TRAINING.md)：当前 A6000 预检、续训和验证命令。
- [实验台账](EXPERIMENTS.md)：双 3060、A100 和 A6000 的独立实验记录。
- [会话交接](SESSION_HANDOFF.md)：下一次开发的入口与未解决问题。
- [BuildingFiller P01 与 VACE 准备](docs/BUILDINGFILLER_P01_VACE_HANDOFF.md)：三机边界、PowerShell SCP 命令、实际数据验证与 17 帧最小实验方案。

## 代码与数据流

历史 T2V：建筑渲染视频 → 固定 train/val/test 划分 → 视频 latent/文本 context 缓存 → DiT LoRA 训练 → 固定 prompt/seed 视频对比 → 人工检查与像素诊断。

当前 VACE 准备：个人 Windows BuildingFiller → 实验室 Windows `D:\P01_465` → Linux `data/buildingfiller_import/P01_465` → ZIP/SHA256/图片/相机检查 → 独立 ModelMask 适配与原始 VACE 基线计划。完整性检查已完成，GPU 推理等待用户确认；不复用历史 T2V LoRA 或缓存。

- `code/Python_3D_Scanner/training/`：项目训练入口、历史数据工具、验证和报告工具。
- `code/DiffSynth-Studio/`：第三方引擎；项目逻辑优先放在项目层。
- `scripts/`：当前 A6000 入口与历史 A100/Slurm 入口。不要仅凭文件名选择环境。
- `configs/single_orbit_same_prompt_v1/`：当前冻结划分，1992/234/118，共 2344 个视频。
- `data/`、`models/`、`cache/`、`checkpoints/`、`outputs/`：原始或生成产物，默认只读。
- `logs/`：运行日志和历史暂停记录；旧记录中的路径可能已迁移。
- `docs/`：审计、兼容入口和 Claude worker 说明；当前状态统一指向根目录文档。
- `.pre_git_sync_20260924/`：历史快照，不作为当前运行入口。

当前 A6000 环境使用 `/home/shi/miniconda3/envs/scj/bin/python`。扫描器的 `requirements.txt` 属于历史 Windows GUI 环境，不是训练环境完整依赖清单。

扫描器 README 描述的桌面采集、OBJ 冻结与三元组渲染是历史设计；当前工作副本没有 `code/Python_3D_Scanner/src/` 和对应数据集工作流文档。需要从原始采集项目恢复并审查后再开发这部分，不能据此声称桌面工具可运行。

本轮 `scripts/audit_buildingfiller_package.py` 默认只读；只有显式提供新的 `--output-dir` 才保存审计副本。GPU 推理需等待用户确认。
