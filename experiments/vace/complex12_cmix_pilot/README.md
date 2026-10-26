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
