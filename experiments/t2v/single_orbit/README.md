# 历史T2V：A6000单轨统一prompt

更新：2026-10-23（UTC+09:00）。本实验学习灰色建筑外观和环绕视频生成，属于历史Wan2.1-T2V-1.3B任务，与VACE的Damaged/Mask条件补全分开。

## 数据、缓存和参数

| 项目 | 已核对内容 |
| --- | --- |
| 视频根 | data/single_orbit_videos，共2344条 |
| 固定划分 | 训练1992、验证234、测试118 |
| 配置入口 | [single_orbit_same_prompt_v1](../../../configs/single_orbit_same_prompt_v1/) |
| 输入尺寸 | 384×384、81帧 |
| LoRA与预算 | rank16、学习率2e-5、计划8 epochs |
| 原训练 | 物理batch2、累积2；早期workers2 |
| 当前脚本默认 | batch1、累积4、workers0；有效batch仍为4 |
| 缓存 | cache/wan21_t2v_13b_single_orbit_same_prompt_384x384_f81_train，1992个.pth |
| 缓存标记 | compact-sft-shared-context-bf16-v2 |

共享context成立的前提是所有视频使用同一条prompt。历史全量检查记录latent为FP32、形状1×16×21×48×48，shared context为BF16、1×512×4096；格式名不能证明latent落盘dtype，训练时转BF16。本次只核对目录数量，不重新扫描所有缓存张量。

逐栋prompt、Cap3D或不同划分必须另设计缓存；当前VACE不会读取这份T2V训练缓存。视频文件名集合无交叉，不等同已证明源OBJ级别完全无泄漏。

## 已有成果与完成边界

- [28份保留权重](../../../checkpoints/building_wan_a6000_single_orbit_all/)：文件步号400–11200。
- [29个比较视频](runs/validation/)：Base加保留权重比较，原位置outputs/validation/validation_all_single_orbit。
- 旧日志最后一段记录至11562；根据当时batch/skip/initial参数推算预算末步为13536，缺独立正常终止记录。
- 因此“完成全部8 epochs”仍待核实；不能用最高checkpoint或已有验证视频替代训练退出证据。
- logger step每个物理batch加一，不是optimizer update；梯度累积和历史batch变化必须计入样本预算换算。

文档更新没有补训、重新验证或改动权重；也没有给29个历史视频追加新的视觉质量结论。

## 代码、续训与故障

[训练入口](../../../scripts/train_wan81_a6000.sh)、[续训入口](../../../scripts/resume_wan81_a6000.sh)、[验证入口](../../../scripts/validate_wan81_a6000.py)保留在公共scripts。本机使用scj解释器，不能照搬A100的mv2v环境。

历史增量续训通过--lora_checkpoint加载权重；它不恢复optimizer、RNG或shuffle。默认resume从平铺目录step-2400读取，输出新时间戳目录；仅--dry-run为预检。不要把safetensors传给完整状态resume参数。

原生free()/SIGSEGV在workers0下仍出现，根因未确认。串行反向长测试没有复现，不意味着普通反向已修好；部分串行测试产物按用户要求删除，保留 [故障文档](../../../docs/NATIVE_STABILITY_20261014.md)。本任务不自动重新启动诊断。

操作与检查规则见 [TRAINING.md](../../../TRAINING.md)。详细历史参数、分段续训及失败过程保留在 [旧台账](../../../docs/history/20261023_before_structure/EXPERIMENTS.md) 和 [旧操作手册](../../../docs/history/20261023_before_structure/TRAINING.md)。
