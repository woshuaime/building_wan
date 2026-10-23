# BuildingFiller P01数据交接与后续使用

更新：2026-10-23（UTC+09:00）。P01数据已经接收并用于完成81帧VACE实验，不需要再次上传或准备首次推理。本页说明来源、验证证据及条件语义；旧上传命令和阶段过程完整保留于 [原始记录](history/20261023_before_md_refresh/docs/BUILDINGFILLER_P01_VACE_HANDOFF.md)。

## 1. 数据来源和身份

BuildingFiller是独立数据制作项目，负责建筑几何配对、渲染、Exact、相机与质量证据；building_wan消费已交接数据，不重新生成原始数据。

| 字段 | P01身份 |
| --- | --- |
| 源/家族编号 | 465 |
| sample_id / stable_id | sample_0000 / b465_2e6fc746 |
| 缺损档位 | A1 Medium |
| 删除原表面积比例 | 0.056792346128503535 |
| 图像角色 | GT、Damaged、Exact |
| 帧序 | B000–B080，共81帧，80个不同视角 |
| 分辨率 | 512×512 |
| 合法全黑Exact | B046–B072，共27帧 |

来源记录中的human_QA=accepted、generation_evidence_verified=true与training_ready=false分别表达不同阶段状态；原sample_meta的旧pending也保留，不通过改原件让各字段看起来一致。人工接受数据不代表VACE预测达到质量要求。

## 2. 当前实际位置

| 内容 | 路径 |
| --- | --- |
| 原始ZIP与相邻SHA清单 | data/buildingfiller_import/P01_465/ |
| 原收尾记录 | 同目录Stage1_Closeout_Report.md |
| 接收审计 | [outputs/buildingfiller_handoff/P01_465_20261018](../outputs/buildingfiller_handoff/P01_465_20261018/) |
| 已解包只读载荷 | [package/BuildingFiller_P01_465_512_v1](../outputs/buildingfiller_handoff/P01_465_20261018/package/BuildingFiller_P01_465_512_v1/) |
| 随包校验器 | [verify_package.py](../outputs/buildingfiller_handoff/P01_465_20261018/package/BuildingFiller_P01_465_512_v1/verify_package.py) |

Windows的D:\P01_465及旧E盘路径只是上传来源，不是Linux可直接读取的位置。当前服务器根为/mnt/windowsE/chuanjun/building_wan；不需要因为归档包含SCP步骤而重复上传。

## 3. 已完成的首次验收

2026-10-18实际检查ZIP共1,319,395 bytes、CRC及清单通过，263个包内文件中262项被SHA256SUMS覆盖。243张PNG全部解码，GT/Damaged为RGB、Exact为L且像素仅0/255；manifest、frame_order、相机和几何身份一致。

000与080的三个图像角色及相机闭合已核对。27张全黑Exact属于合法不可见缺口视角，不视为坏帧。原审计目录保留audit.json、supplier_verify.json、environment_and_adapter_check.json、frame_selection_17.json及contact_sheet.png。

这次接收验收只证明当时P01的512子集，不把它泛化为对其他分辨率/其他建筑的验收。后续Pilot8总包和Complex12分别有自己的校验记录；本次Markdown维护没有重复全量哈希。

## 4. VACE条件适配

Damaged与Exact必须同原始帧、同尺寸和同视角。Exact白区代表生成区域，复制为RGB ModelMask；Damaged对应白区填RGB127、黑区保留。无扩张、羽化、反转或独立裁剪主体。

训练时GT可作监督target；推理时GT不读入条件，不作为首尾帧、视频初始化或reference image。GT只在预测生成后用于对比。Exact来源于配对数据，当前实验尚未验证自动检测缺口。

早期P01的17帧窗口是原始B005–B021，输出片段索引000–016；后来P04/P06 LoRA17是B000–B016，不能混同。完整81帧则保持B000–B080全部对应，预测080不复制000。

## 5. 从P01到当前实验

| 后续工作 | 当前状态/入口 |
| --- | --- |
| P01最早81帧 | [原始结果](../outputs/vace/P01_465_81f_20261019_171510_1691517/)保留 |
| 八例Base | [Pilot8 Base](../experiments/vace/pilot8_base/README.md)，P01结果复用 |
| CPU缺口回贴 | [Composite](../experiments/vace/pilot8_composite/README.md)，独立后处理诊断 |
| 两例LoRA17 | [P04/P06](VACE_P04_P06_LORA_SMOKE.md)，P01未参加训练 |
| 八例LoRA81 | [整圈评估](../experiments/vace/pilot8_lora81/README.md)，P01为迁移样本 |
| Complex12 Rev1 | [十二栋独立数据](../experiments/vace/complex12_rev1/README.md)，不可套用P编号 |

正式VACE模型已经到位并完成上述实验。旧“权重未找到”“尚未推理”只属于早期时点，不是待办。当前模型与执行入口见 [VACE说明](VACE_P01_EXECUTION.md)，全局状态见 [PROJECT_CONTEXT](../PROJECT_CONTEXT.md)。
