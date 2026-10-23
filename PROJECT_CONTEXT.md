# 当前项目状态

核对日期：2026-10-23（UTC+09:00）。最新正式实验已结束；此次仅更新项目Markdown。实验入口统一在 [experiments/README.md](experiments/README.md)。

## 1. 当前可确认的成果

| 实验 | 已完成范围 | 可读证据 |
| --- | --- | --- |
| Pilot8 Base | 8栋×81帧，每栋4项可视化 | [结果](outputs/vace/test/) |
| Pilot8 Composite | 8栋CPU硬边界回贴，每栋3项结果 | [结果](outputs/vace/composite_check/) |
| P04/P06 LoRA17 | 300真实更新、两例各150、6次17帧比较、100/300权重 | [run](outputs/vace/lora_roof_smoke/run_20261020_165237_150325_2048720/) |
| Pilot8 LoRA81 | 八例81帧、32项成果，复用Base | [run](outputs/vace/lora300_eval81/run_20261020_194300_145372/) |
| Complex12 Rev1 | C/U/S各300，60轨迹、4860PNG、12视频、12总览 | [完成状态](outputs/ComplexSingleOrbit_Rev1_Pilot_20261022/progress.json) |

Complex12最终状态为complete。源记录完成时刻2026-10-23T02:59:37+08:00，换算本页时区为2026-10-23T03:59:37+09:00。旧记录自身时区不改写。

## 2. Complex12实验身份

物理run：outputs/ComplexSingleOrbit_Rev1_Pilot_20261022/。数据来自data/buildingfiller_import/Complex12/BuildingFiller_Complex12_512_Rev1_20261022/，使用Rev1的accepted索引。

- 训练：CS01/03/05/06/07/09/10/11；验证：CS02/04/08/12，未参与训练或检查点选择。
- B为Base；L为旧P04/P06 LoRA-300迁移参考；C/U/S从共同新初始化出发，各自新AdamW、相同采样和随机序列。
- C为原目标；U均匀增加缺损区权重；S增加结构带权重。U/S总额外权重相等。
- 17帧连续窗口训练，81帧整轨推理，512×512、16FPS、30步、CFG5；训练seed20261022，推理seed20261018，LoRA scale1。
- 只比较各组step300；100/200仅恢复点。预测没有回贴、调色或强制闭环。
- 冻结配置ID：e42ab127a6625942953bea743e31b89fc8e00dcd6604fdd3ac61b5935561c495。

完整参数、样本源号、损失映射、实际耗时/显存见 [Complex12实验页](experiments/vace/complex12_rev1/README.md)。

## 3. 效果结论与证据边界

C相对Base的验证缺损MAE降低64.21%。S相对U验证MAE降低1.80%、结构Sobel降低0.19%；验证均值优势集中于CS12，其余三栋S的MAE略差于U。S的保留区变化及全图闭环误差高于C/U。

CS02、CS03、CS10的大黑槽/黑洞持续改善；CS04、CS06、CS09、CS12仍有细折线、弯曲交界或浅凹。CS07的S在B010附近有小灰斑。已查看60条轨迹的连续帧图及关键帧，不等同于实时高频闪烁测量或三维几何验证。

旧P04/P06是LoRA17的训练题；Complex12 CS06为源1251，不能混同旧P06源1848。CS09–CS12只代表复杂屋顶背景下局部屋面/邻墙补全，跨屋脊挑战主要看CS06。

## 4. 环境、目录与历史T2V

当前项目与Python均位于本服务器；Python3.10.13、torch2.11.0+cu128、DiffSynth2.1.5、PEFT0.17.1。GPU为单张A6000；历史显存读数不能当作当前空闲容量。

Complex12源码已归到experiments/vace/complex12_rev1/scripts/，原run/scripts为兼容链接，原resume.sh不变。源码位置与结果位置分离，新想法按 [目录约定](docs/PROJECT_STRUCTURE.md) 创建。

历史A6000 T2V保留28份LoRA、29个比较视频和1992个.pth样本缓存；A100目录保留39份LoRA。A6000最后loss step11562、最高权重11200与推算预算末步13536不同，完整8 epochs自然结束仍待证实。logger物理batch步号不能当optimizer更新数。

## 5. 未解决事项及停止边界

- B/CS04保存完整81帧后原生退出−6；独立验证有效输出后复用，没有重跑。原生退出根因未确认。
- 挂载恢复后C从保存的40次更新续至300；后续U/S各300。已成功轨迹未重复运行。
- 单轨T2V尚缺源OBJ级身份映射的完整泄漏核查与可靠训练终态证据。
- 扫描器src/及两份GUI工作流源文档缺失，不能据此安装运行历史GUI。
- 文档中记载的Claude默认模型字符串不是服务可用性证明；本次没有调用外部worker/API。

当前没有待自动继续的GPU任务。报告提出的配对重复实验只是建议。完整结果从 [00_看结果](outputs/ComplexSingleOrbit_Rev1_Pilot_20261022/00_看结果/README_从这里开始.md) 打开；过程快照见 [历史索引](docs/DOCUMENTATION_INDEX.md)。
