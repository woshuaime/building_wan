# Complex12 C_mix：固定预算的轻/重缺损覆盖对照

用户于2026-10-26明确授权一组普通C_mix的300次更新和16条81帧推理；随后要求确认训练启动即可结束对话，视觉复核与最终报告等待另行通知。自动队列仍包括训练、推理、指标与视频，计算结束标记为`awaiting_visual_review`，不提前写效果结论。

当前run：`/mnt/windowsE/chuanjun/building_wan/outputs/vace/complex12_cmix_pilot/run_20261026_222008`。

[结果入口](runs/run_20261026_222008/00_看结果/README_从这里开始.md) · [冻结配置](configs/run_20261026_222008.json) · [采样表](runs/run_20261026_222008/shared_sampling_table.csv)。启动与更新证据以run的progress和日志为准。

## 问题与唯一主要变量

以旧Rev1普通C为对照，在相同300更新预算下混合关键连接处的另一缺损版本，检查OriginalHard是否改善以及Rev1是否退步。两版改变了缺损位置和连接线索，不是将同一缺口单纯扩大。

训练：CS01/03/05/06/07/09/10/11。验证：CS02/04/08/12；任何验证版本不进入训练。从旧Rev1的shared_initialization.safetensors与原始VACE-1.3B开始，新AdamW，不从旧step300追加训练。

复用旧C的FlowMatch目标、scheduler、优化与编码函数（运行时导入旧Complex12源码，旧文件不改）。rank/alpha16，vace分支，17帧512×512，lr5e-5，batch1/accum1，workers0，原BF16/FP32和梯度检查点设置不变。没有GPU预跑或临时优化步。

## 固定数据版本分配

保留旧300行采样表全部字段与顺序，仅追加data_version。局部Random(20261026)按CS09、CS10、CS11顺序抽取20、14、16行Hard；不消耗原训练随机状态。

| 建筑 | 总更新 | Rev1 | OriginalHard |
| --- | ---: | ---: | ---: |
| CS01 | 39 | 39 | 0 |
| CS03 | 34 | 34 | 0 |
| CS05 | 44 | 44 | 0 |
| CS06 | 46 | 46 | 0 |
| CS07 | 36 | 36 | 0 |
| CS09 | 40 | 20 | 20 |
| CS10 | 29 | 15 | 14 |
| CS11 | 32 | 16 | 16 |
| 合计 | 300 | 250 | 50 |

训练窗口起点0–63，末帧不超过079，合法黑Mask帧保留。缓存位于新run/cache/<data_version>/<CS>_<start>_17.pt；先同步选17帧再编码。合法零scheduler权重正常计数，在训练完成记录中按版本汇总，不重抽、不补步。完整optimizer/adapter/RNG恢复状态沿用每10更新保存、100/200里程碑；最终统一step300。

## 推理和交付

新模型先推OriginalHard CS09–12，再推Rev1 CS01–12，共16条/1296原始PNG。每条fresh Base +一份C_mix；81帧512²、16FPS、30步、CFG5、seed20261018及其余设置、实际正负文本完整沿用旧C。GT不进入推理条件，Mask预处理/内部时间映射不改，输出不回贴、不替换080。

旧C原始预测与指标直接从旧run读取，B/L/U/S不重跑。Hard CS01–08只建立对应Rev1新预测/展示的导航链接，不增加推理。

00_看结果按Rev1/OriginalHard分开，16个独立输入各有C_mix原始视频和2×2同步对比，共32视频；每个内容区域完整512²，标题单独加在图外。Hard四栋保留六张固定问题帧对比：CS09/006、CS10/024与058、CS11/073、CS12/000与080。

原指标函数直接复用Rev1实现。常规000–079，闭环000/080；楼内像素加权、跨楼等权。保存逐帧/逐栋、逐指标新旧绝对/相对差、分组汇总、Rev1退步项、CS01–08原样对照。基准为0或无有效数值时相对变化为N/A。视觉正确性需后续结合GT单独复核。

## 状态与恢复

只读查看：

```bash
bash /mnt/windowsE/chuanjun/building_wan/outputs/vace/complex12_cmix_pilot/run_20261026_222008/resume.sh
```

已获授权的同一run恢复入口（先确认没有活跃runner）：

```bash
bash /mnt/windowsE/chuanjun/building_wan/outputs/vace/complex12_cmix_pilot/run_20261026_222008/resume.sh --resume
```

无参数只读；运行入口有run/GPU互斥锁。成功轨迹解码后复用；推理完成但CPU保存标记中断时，从raw_ready恢复，不重复GPU。进程异常保留输出、日志和完整训练状态，不能把异常状态写成成功。

计算结束后不自动训练下一组。用户通知后再复核完整轨迹、撰写run中文结论及同内容upload报告，并追加长期记录。Hard只有50次更新、CS10只有14次；不改善不能直接证明普通LoRA或混合训练无效，改善也不是最优配比证明；CS12是唯一Hard未训练建筑，属于开发验证。


## 2026-10-26：C_mix 固定配比实验启动与后续交接

2026-10-26T22:26:41+08:00：已发生33/300次更新；runner PID 2179781，tmux complex12_rev1:cmix_pilot。9项专用检查及298项项目CPU回归通过；无额外GPU试跑。队列继续推理/指标/媒体，视觉结论和最终报告待用户通知。


## 2026-10-27：C_mix 完整评测、视觉复核与收尾

用户通知后已完成所有收尾，前述启动/等待说明保留为历史快照。[看结果](/mnt/windowsE/chuanjun/building_wan/outputs/vace/complex12_cmix_pilot/run_20261026_222008/00_看结果/README_从这里开始.md)，[同内容便携报告](/mnt/windowsE/chuanjun/building_wan/upload/20261027_Complex12_C_mix_混合训练与配对评测报告.md)，准确run为`/mnt/windowsE/chuanjun/building_wan/outputs/vace/complex12_cmix_pilot/run_20261026_222008`。

- 300行真实优化器更新和原采样字段全部核对；250/50、CS09/10/11 Hard各20/14/16；零scheduler权重Rev1 4步/Hard 3步，无补步。最终optimizer steps=[300,300]。
- 16/16完整81帧、1296PNG，32视频与6张固定Hard问题图完成；新增Rev1 CS10/B020反例，没有替换原反例。Hard CS01–08链接复用Rev1，不另计轨迹。
- 1训练+16推理+1评估worker全部一次START及EXIT 0；训练约27分06秒，整体计算/媒体墙钟80分26秒。本run无重试/恢复。
- 16条全000–080联系图复核、七张原尺寸图核对；视频完整解码和同帧对齐沿用生成阶段记录。本次未实时播放全部视频、未定量测闪烁或三维结构。

结果未达到双重目标：Hard CS10 MAE−4.07%而Sobel+12.06%，连接仍错；Hard CS12 MAE−1.78%、MSE+9.65%、Sobel+5.01%，缺口首尾0.115625→0.117000。Rev1有10/12栋MAE退步，CS10出现百叶状条纹，MAE0.009775→0.127054；所有16输入known MAE及全图首尾均变差。

保留旧C作为参考。后续仅建议当前入口纯Rev1、相同初始化/300预算对照，先看Rev1 CS10和Hard CS10/12三条，定位显著退化；未执行，不自动扩大预算或加模块。最终完成状态见progress.json与logs/final_review.done.json；原evaluate.done.json的待复核字段是计算结束时快照，不覆盖改写。
