# Complex12 OriginalHard：已有 B/L/C/U/S 的完整轨迹评测

建立：2026-10-26（UTC+09:00）。本实验仅做推理，使用已经完成训练的 LoRA；新增优化器更新为 **0**。实际状态以本页后续记录和 run 的 progress.json 为准。

## 1. 研究问题和数据身份

检验 Rev1 上已经训练好的普通 C、均匀缺损加权 U、结构区域加权 S，在修改前跨屋脊、交叉屋顶及屋顶连接处的 OriginalHard 缺损上能否继续有效。同时保留原始 VACE 的 B 和旧 Pilot8 LoRA 的 L 作为对照。

用户要求“旧的相同的就不用跑了”。逐帧核对表明，CS01–CS08 的 GT、Damaged、Exact、raw_visible 和相机记录与旧 Rev1 完全相同，因此复用这八栋的五方法预测，共 **40 条**。CS09–CS12 的 Damaged/Exact 不同，需要新增 **20 条** 81 帧推理。最终统一呈现十二栋、五方法、60 条轨迹，其中 40 条来自旧结果，不能把 60 条全部称为本轮新增推理。

| 样本 | 源编号 | 相对 Rev1 训练的身份 | 本轮处理 |
| --- | --- | --- | --- |
| CS01 | 304 | 训练 | 五方法复用 |
| CS02 | 90 | 验证 | 五方法复用 |
| CS03 | 1019 | 训练 | 五方法复用 |
| CS04 | 688 | 验证 | 五方法复用 |
| CS05 | 986 | 训练 | 五方法复用 |
| CS06 | 1251 | 训练 | 五方法复用 |
| CS07 | 1432 | 训练 | 五方法复用 |
| CS08 | 1805 | 验证 | 五方法复用 |
| CS09 | 24 | 训练 | OriginalHard 新缺损，跨屋脊端部 |
| CS10 | 923 | 训练 | OriginalHard 新缺损，跨屋脊与连接 |
| CS11 | 346 | 训练 | OriginalHard 新缺损，交叉双坡、多翼 |
| CS12 | 1344 | 验证 | OriginalHard 新缺损，L 形多翼交叉屋顶 |

这些是相同建筑来源上的缺损版本变化，不是十二栋全新的建筑泛化测试。新困难四栋中三栋来自旧训练身份，只有 CS12 是旧验证身份；汇总必须单列这四栋及 3/1 身份子组。OriginalHard 的困难来自缺口位置与结构信息缺失，不代表每栋缺口面积都更大。数据包中原有 human_QA:pending 记录保留，不宣称它通过了 Rev1 的人工接受流程。

## 2. 路径和成果职责

- 上传原包：`/mnt/windowsE/chuanjun/building_wan/data/buildingfiller_import/Complex12/Complex12_OriginalHard_20261025/BuildingFiller_Complex12_OriginalHard_512_20261025.zip`。
- 源码：[scripts](scripts/)。公共模型加载与预处理沿用项目现有实现。
- 本次 run：`/mnt/windowsE/chuanjun/building_wan/outputs/vace/complex12_originalhard_eval/run_20261026_012033`；导航见 [runs/run_20261026_012033](runs/run_20261026_012033/)。
- 冻结配置：[configs/run_20261026_012033.json](configs/run_20261026_012033.json)。只保存链接，真实配置在 run 内。
- 解压工作副本在该 run 的 input_data/；原始上传目录、旧数据及旧 run 保持不变。
- 旧权重来自 `/mnt/windowsE/chuanjun/building_wan/outputs/ComplexSingleOrbit_Rev1_Pilot_20261022/checkpoints/{C,U,S}/step300.safetensors`。
- L 来自 `/mnt/windowsE/chuanjun/building_wan/outputs/vace/lora_roof_smoke/run_20261020_165237_150325_2048720/training/lora_update0300.safetensors`，仅为参考，不与 C/U/S 叠加。

本轮没有新训练，不生成虚假的 checkpoints、训练 loss 或优化器恢复状态。已有权重的绝对路径与身份记录在 resolved_config.json，实际加载路径会打印到终端。预测复用通过目录链接完成，不复制大文件；导出到其他机器时应沿链接复制所需成果。

## 3. 推理与评测

保持 Rev1 的原始 VACE-1.3B、512×512、81 帧、16 FPS、30 步、CFG 5、seed 20261018、BF16、sigma_shift 5、vace_scale 1、LoRA scale 1。其余参数和实际正负文本从旧冻结配置逐项继承。

推理条件仅为 Damaged、由原始 Exact 得到的二值 ModelMask 和文本；Mask 内填灰 127，不扩大或羽化 Mask。GT 和结构标签只用于推理后的评测；不作为初始化、参考帧或条件输入。保留全部模型原始 PNG，不回贴、不调色、不强制将预测 080 替换成 000。

常规指标只使用 000–079，000/080 另算闭环。RGB 范围 [0,1]；缺损 MAE/MSE/PSNR 对 GT；保留区 MAE 对 Damaged；结构区域为 raw_visible 经 7×7 膨胀后与 Exact 相交，Sobel 核除以 8、reflect padding、灰度系数 0.299/0.587/0.114。楼内按有效像素汇总，跨楼等权；空区域 N/A，零 MSE 的 PSNR 为 Infinity。指标定义与旧 Rev1 一致。

保留 train_summary.csv 和 val_summary.csv 的 8/4 旧身份汇总，新增 hard_summary.csv 区分新缺损四栋、旧训练身份三栋、旧验证身份一栋以及输入不变八栋。配对差值包括 C−B、U−C、S−C、S−U；不预设 S 胜出。

## 4. 交付布局

run 内按原约定提供 `00_看结果/`、`predictions/`、`metrics/`、`logs/`、resolved_config.json、progress.json、resume.sh；附输入副本、必要检查记录与复核图。每栋最终观看目录仅包含：

1. `01_五方法同步对比.mp4`：2×4 布局，Before、GT、B、L / C、U、S、空位；各格内容原始 512×512，81 帧、16 FPS。
2. `02_关键帧总览.png`：七行五列，B000/B020/B040/B060/B080，所有方法同帧同视角。

入口提供 README、中文最终结论、训练/验证及困难样本指标表。收尾时另在项目 upload/ 写一份可独立阅读的完整实验报告，追加根目录长期 MD，保留历史正文。

## 5. 运行与状态

已在原有可见 tmux 会话 `complex12_rev1` 新建窗口 `originalhard81`，串行执行 CS09、CS10、CS11、CS12，每栋依次 B/L/C/U/S。所有 GPU 轨迹用独立进程。之后完成统一 CPU 评测、制图、全轨迹复核和中文报告。

只读状态命令：

```bash
bash /mnt/windowsE/chuanjun/building_wan/outputs/vace/complex12_originalhard_eval/run_20261026_012033/resume.sh
```

该入口无参数只读；`--run` 或 `--resume` 才执行未完成队列。进程仍运行时不要重复启动。成功任务检查配置及完整输出后复用；异常任务保留真实退出码和输出，不自动重试。全部计算结束后状态先为 awaiting_visual_review，报告和最终验证完成才改为 complete。

## 6. 准备阶段的实际问题与检查

首次 CPU 准备误将“Exact 外 GT 和 Damaged 必须逐像素相等”作为条件，CS01/000 因不满足而停止，当时尚未启动 GPU。独立几何渲染可在光照或边缘处产生差异，因此移除这一不适用断言，改为记录差值；预处理保留区域仍严格对照 Damaged。输入文件没有修改，首次日志保留在 logs/package_verify_attempt1.log，修正后的准备日志为 logs/package_verify.log。

原始 ZIP 与随包校验值核对一次；每栋四类输入各 81 张可读、尺寸/二值标签/首尾对应通过。40 条旧预测均解码检查后复用。旧 B/CS04 的历史原生退出 −6 及独立输出验证保留在复用记录中，不能改写成旧进程正常退出；本轮未重跑它。

项目 CPU 回归 298 项通过（50.783 秒）；专用测试覆盖指标口径、GT 隔离、合法全黑 Mask、已有权重路径及纯推理任务队列。后续运行和最终交付的真实结果继续追加到此处。
