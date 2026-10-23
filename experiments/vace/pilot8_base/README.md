# Pilot8：原始VACE八例基线

更新：2026-10-23（UTC+09:00）。研究问题：原始VACE-1.3B能否恢复BuildingFiller八栋建筑的缺口？**P01–P08全部81帧已完成**，本目录归拢已有结果，没有启动新运行。

## 数据与编号

| 结果目录 | 样本 | 源编号 | 缺损档位 | 合法全黑Exact帧 |
| --- | --- | ---: | --- | --- |
| 01 | P01 | 465 | A1 Medium | B046–B072 |
| 02 | P02 | 763 | A1 Medium | B023–B049 |
| 03 | P03 | 871 | A1 Medium | B000–B016、B071–B080 |
| 04 | P04 | 1227 | A2 Medium | 无 |
| 05 | P05 | 1259 | A2 Large | 无 |
| 06 | P06 | 1848 | A3 Medium | 无 |
| 07 | P07 | 2297 | A3 Large | 无 |
| 08 | P08 | 939 | A4 Medium | 无 |

P01独立包与八例总包已通过身份对应检查。八例包位于 `data/buildingfiller_import/P02_P08_20261020/BuildingFiller_Pilot8_AllRes_v1.zip`；P01交接说明见 [专项文档](../../../docs/BUILDINGFILLER_P01_VACE_HANDOFF.md)。全黑帧表示当前视角无可见缺损，不删除、不造白区。

## 模型与固定条件

原始Wan2.1-VACE-1.3B，无LoRA。81帧B000–B080、512×512、16FPS、30步、CFG5、seed20261018；BF16、tiled VAE、sigma_shift5、vace_scale1、denoising_strength1、cfg_merge=false、CPU随机数设备。一次生成整轨，B080使用模型的第81帧预测。

正向文本：

> 白色背景中一栋完整的灰色无纹理建筑三维模型，规则连续的墙面和屋顶，保持建筑形状与相机轨迹一致，固定俯仰视角平滑环绕，结构稳定，光照一致。

负向文本：

> 彩色纹理，窗户装饰，新增建筑，建筑形变，结构闪烁，背景变化，文字，水印

原始Exact二值0/255复制为RGB ModelMask；白区在Damaged中填灰127，黑区像素保留。无Mask扩张、羽化、独立建筑裁剪或GT推理条件。GT在生成后用于比较。81帧配置从 [早期17帧配置](../../../configs/vace_p01_465_17f_v1.json) 显式派生；文件名中的17f不能当作最终81帧长度。

## 结果与文件含义

| 内容 | 路径 |
| --- | --- |
| 八例正式结果 | [runs/base81](runs/base81/) → outputs/vace/test/ |
| P01最早81帧结果 | [runs/p01_first81](runs/p01_first81/) |
| P01数据交接检查 | [runs/p01_handoff](runs/p01_handoff/) |
| 总流程与恢复 | [vace_building_test.py](../../../scripts/vace_building_test.py) |
| 81帧公共推理 | [vace_p01_81f.py](../../../scripts/vace_p01_81f.py) |
| 数据包适配 | [buildingfiller_pilot8.py](../../../scripts/buildingfiller_pilot8.py) |

每例正式目录恰有四个成果：

| 文件 | 含义 |
| --- | --- |
| 01_补全结果.mp4 | 81帧原始Base输出，512×512/16FPS |
| 02_补前_补后_GT_三栏对比.mp4 | Before、Base、GT同步对照 |
| 03_补前_补后_GT_三行总览.png | 三行五列，B000/020/040/060/080 |
| 04_缺损_Mask_预测_GT_四栏验证.png | 同视角核对输入、ModelMask、预测和GT |

展示布局曾经CPU重排，原始预测未重新生成。旧交接文档的“两栏/两行”命名属于早期版本，当前以这里的四文件为准。

## 观察、异常与复用

P04屋顶凹槽/台阶、P06深色洞口未被Base可靠恢复，是后续两例LoRA的选择依据。八例难度与Base表现不同，不能把某个成功样本概括为全部完成精确补全。

历史P04开始前曾发生T5哈希检查异常，模型推理尚未开始；后续独立检查匹配并按用户授权继续完成。记录保留，不认定其原生或硬件根因已解决。早期P01未保留完整运行manifest，不能从视频反推种子；既有脚本、冻结参数、数据与后续复核是其可追溯范围。

[Composite](../pilot8_composite/README.md)只做CPU诊断；[LoRA81](../pilot8_lora81/README.md)复用这八例Base，未无差别重跑。日常直接查看已有结果；[启动脚本](../../../scripts/run_vace_building_test.sh)无参数会执行任务，不应用于状态检查。历史过程见 [归档台账](../../../docs/history/20261023_before_structure/EXPERIMENTS.md)。
