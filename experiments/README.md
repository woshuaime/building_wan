# 按研究想法查看实验

更新：2026-10-23（UTC+09:00）。每个目录回答一个明确研究问题，保存专用代码、配置入口和结果说明。runs/是原有产物的目录链接，视频、图片、权重和缓存保持原位置。

## 1. 最新成果

[Complex12 Rev1结果入口](vace/complex12_rev1/runs/20261022/00_看结果/README_从这里开始.md)包含12栋建筑的视频、关键帧、训练/验证指标及中文结论。C/U/S各300更新和五方法60条完整轨迹已完成，没有待自动续跑任务。

## 2. VACE建筑补全实验

| 实验目录 | 研究内容 | 实际完成量 | 直接看结果 |
| --- | --- | --- | --- |
| [pilot8_base](vace/pilot8_base/README.md) | 原始VACE八例基线 | 八例81帧，各四成果 | [Base](vace/pilot8_base/runs/base81/) |
| [pilot8_composite](vace/pilot8_composite/README.md) | 仅缺口回贴的CPU诊断 | 八例各三成果，无新增推理 | [Composite](vace/pilot8_composite/runs/cpu_check/) |
| [pilot8_lora17](vace/pilot8_lora17/README.md) | P04/P06两个固定训练题 | 300更新，100/300两份LoRA，六次17帧推理 | [训练与比较](vace/pilot8_lora17/runs/20261020/) |
| [pilot8_lora81](vace/pilot8_lora81/README.md) | 旧LoRA-300整圈与迁移 | 八例81帧，32项成果；Base复用 | [LoRA81](vace/pilot8_lora81/runs/20261020/) |
| [complex12_rev1](vace/complex12_rev1/README.md) | 普通/均匀/结构加权配对比较 | 三组各300更新，60轨迹、4860PNG、12视频、12总览 | [结果与结论](vace/complex12_rev1/runs/20261022/00_看结果/README_从这里开始.md) |

实验关系：Base暴露凹槽/黑洞 → Composite确认仅回贴不足 → 两例LoRA17验证可学习 → LoRA81检验整圈和迁移 → Complex12在独立数据上比较C/U/S。箭头描述历史研究顺序，不表示自动执行下一阶段。

L是旧两例LoRA参考；Complex12的C/U/S是共同新初始化，不能说它们由L继续训练。P04/P06是Pilot8编号，不是Complex12的CS04/CS06。

## 3. 历史T2V生成

| 实验目录 | 内容 | 保留证据与限制 |
| --- | --- | --- |
| [single_orbit](t2v/single_orbit/README.md) | A6000统一prompt单轨 | 1992/234/118划分、1992缓存、28权重和29比较视频；8 epochs终态待核实 |
| [early_experiments](t2v/early_experiments/README.md) | Windows、双3060、A100 | A10039份权重；双3060权重目录当前缺失，历史环境独立 |

T2V学习外观与环绕生成，VACE学习Damaged/Mask条件补全；不共享LoRA、训练目标或文本缓存。公共组件从 [scripts](../scripts/README.md) 和 [训练工具](../code/Python_3D_Scanner/training/README.md)查找。

## 4. 新想法与新运行

从 [_template](_template/README.md)建立说明：先写问题、对照、数据身份、变化变量、预算和验证口径。不同研究问题进入新的idea目录；同一想法不同运行使用新的run_id。

`experiments/<vace或t2v>/<idea>/`保存专用源码、配置和说明；`outputs/<任务>/<idea>/<run_id>/`保存这一次生成的权重、预测、指标、日志及00_看结果。只创建实际需要的内容，不创建虚假完成目录或复制大结果。具体见 [目录规范](../docs/PROJECT_STRUCTURE.md)。

已有特殊命名目录保持不动。Linux可通过runs链接查看；其他环境不支持链接时，查各实验README记录的原始物理路径。GitHub仓库默认不含模型和媒体，链接不意味着这些成果上传到了GitHub。

## 5. 状态和文档

全局当前状态见 [PROJECT_CONTEXT](../PROJECT_CONTEXT.md)，跨实验结论见 [EXPERIMENTS](../EXPERIMENTS.md)，操作见 [TRAINING](../TRAINING.md)，接手见 [SESSION_HANDOFF](../SESSION_HANDOFF.md)。本次Markdown更新范围和检查记录见 [文档索引](../docs/DOCUMENTATION_INDEX.md)。

## 2026-10-24：每组实验的独立报告

日期采用UTC+09:00。研究入口仍由本目录按想法组织，新增的 [upload/](../upload/README.md) 专门保存可复制到其他机器阅读的详细实验报告。

[Complex12 Rev1完整报告](../upload/20261022_Complex12_Rev1_十二栋单轨实验报告.md) 已整理完成，写清B/L/C/U/S的区别、C/U/S各300真实更新、60条完整轨迹、异常恢复和训练/验证结果。实验本身没有重跑，结果仍在原run。

以后每组实验收尾时，除了更新本目录对应README和根长期MD，还需新增对应upload报告并登记索引；较早实验仍保留各自原入口，不能用新报告索引删去原研究过程。格式见 [upload要求](../upload/README.md)。


## 2026-10-26：OriginalHard 已有权重评测（执行中）

[新实验入口](vace/complex12_originalhard_eval/README.md)使用已训练的B/L/C/U/S、完整81帧；本轮训练更新0。CS01–CS08输入完全一致，40条预测复用；CS09–CS12新缺损需20条推理。run为 `/mnt/windowsE/chuanjun/building_wan/outputs/vace/complex12_originalhard_eval/run_20261026_012033`，最终仍提供十二栋五方法同步对比。当前运行快照见根PROJECT_CONTEXT，完成结论待实际复核后追加。


## 2026-10-26：OriginalHard 完成与复核补记

此前“执行中”是启动时快照，现已完成40条复用+20条新推理、五方法60条81帧、12视频/12总览、指标与中文报告，新训练0。入口为[OriginalHard实验](vace/complex12_originalhard_eval/README.md)，[完整报告](../upload/20261026_Complex12_OriginalHard_已有权重评测报告.md)。

新缺损CS09/11明显改善，CS10/12连接形状仍不可靠；S没有稳定优于C/U。四栋中3栋旧训练、1栋旧验证，完整8/4身份汇总另列。下一步仅建议困难结构缺损覆盖对照，未启动。


## 2026-10-26：C_mix 固定配比实验启动与后续交接

[C_mix实验](vace/complex12_cmix_pilot/README.md)已获授权，固定300更新、250/50混合及16条新推理。2026-10-26T22:26:41+08:00真实更新33/300；run为`/mnt/windowsE/chuanjun/building_wan/outputs/vace/complex12_cmix_pilot/run_20261026_222008`。计算和媒体自动执行，报告/视觉复核依用户要求等待通知，尚无效果结论。


## 2026-10-27：C_mix 完整评测、视觉复核与收尾

[C_mix实验](vace/complex12_cmix_pilot/README.md)已完成：300更新（250/50）、16条81帧、1296原PNG、32视频、6固定图+1反例，完整静态轨迹复核与[便携报告](../upload/20261027_Complex12_C_mix_混合训练与配对评测报告.md)齐全。实际位置`/mnt/windowsE/chuanjun/building_wan/outputs/vace/complex12_cmix_pilot/run_20261026_222008`，直接打开[看结果](../outputs/vace/complex12_cmix_pilot/run_20261026_222008/00_看结果/README_从这里开始.md)。

Hard CS10连接、CS12形状/闭环未解决；Rev1有10/12栋MAE退步，CS10条纹最明显，当前保留旧C参考。建议中的纯Rev1复现对照未执行；不再自动展开新实验。


## 2026-10-27：Orbit17线性/循环配对实验已启动

核查时间：2026-10-27T20:16:32+09:00（UTC+09:00）。用户已授权线性组和循环组都执行，并要求确认实际开始训练后结束本轮对话。当前status=running，C_lin17已完成22/600次真实AdamW更新，C_ring17为0/600（待串行执行），推理0/40；日志已有有限loss及参数实际变化。这是启动快照，不是两组完成或效果验证。

新入口：[Orbit17线性/循环配对](vace/complex12_orbit17_pair/README.md)，实际run为`/mnt/windowsE/chuanjun/building_wan/outputs/vace/complex12_orbit17_pair/run_20261027_200842`。

两组普通C各600更新，Rev1训练、Hard只评测，300各4探针、600各16输入，共40轨迹/3240PNG。队列包含两组训练、指标及48视频；后续人工视觉复核与独立upload报告等待用户通知。旧实验结果不搬动。


## 2026-10-29：Orbit17配对实验复核与最终报告完成

Orbit17线性/循环配对实验已完成：两组各600、40条81帧/3240原PNG、48视频，静态完整帧复核与最终报告齐全。[详细便携报告](../upload/20261029_Complex12_Orbit17_线性与循环窗口配对实验报告.md)；[直接看结果](../outputs/vace/complex12_orbit17_pair/run_20261027_200842/00_看结果/README_从这里开始.md)。

ring600同预算MAE 13/16更低，但全图首尾15/16更差；CS12主要折面两组600均改善，CS10连接未解决。不是稳定闭环成功结论。源码仍在vace/complex12_orbit17_pair/scripts，大结果未移动；启动快照继续保留。


## 2026-11-09：0号建筑点云原版VACE基线

[独立实验入口](pointcloud_vace/0_v2_base/README.md)。用户授权单次81帧832×480原版VACE masked video editing，无LoRA；输入与旧mesh实验隔离。后台流程自动保存原始视频、原始PNG、配置、日志及真实退出状态；确认实际采样后结束会话，效果和完整报告待后续核查。运行快照见根目录长期MD最新补记。


## 2026-11-09：0号点云基线完成与报告补记

[0号点云基线](pointcloud_vace/0_v2_base/README.md)已完成单次原版VACE推理、81帧、无LoRA/训练，返回0；[详细报告](../upload/20261109_0号建筑点云_Wan21_VACE原版基线_实验报告.md)已归档。全部静态帧及原像素局部显示密集规则点阵面片、观测点重绘和首尾不一致，未达到自然稀疏点云补全。两视频已完整解码，未实时播放或三维验证。以后每组实验完成/失败收尾继续统一在upload主动写报告并同步长期MD，历史正文保留。


## 2026-11-09：0号点云V3三墙对照

[独立实验入口](pointcloud_vace/0_v3_base/README.md)。一次原版VACE、无LoRA，正式设置继承V2实际config。仅新增C墙生成区域、A/B输入选区保留；81帧832×480/16FPS。准备已通过，派发及首步采样以新run日志/marker为准，效果与完整报告待后续收尾，不能把V2结果当V3完成结果。


## 2026-11-09：0号点云V3三墙对照完成报告

[V3实验入口](pointcloud_vace/0_v3_base/README.md)已完成单次原版VACE、81帧/30步、无LoRA，退出0；[详细报告](../upload/20261109_0号建筑点云_V3三墙对照_Wan21_VACE原版_实验报告.md)已归档。新增C与旧A/B仍呈暗底点阵面片，V3三墙白区首尾MAE46.411138/255，未实现自然稀疏点云与一致闭环。全81建筑区域静态及V2配对复核完成，没有实时播放或三维验证，本次无新GPU任务。


## 2026-11-10：天然残缺点云 D 组结构 Adapter

[实验入口](pointcloud_vace/adapter_d/README.md)。用户仅授权八视角Adapter＋VACE注意力LoRA的D组微调，原版VACE冻结，3,000次更新，832×480连续17帧，首步成功即交接。run为outputs/pointcloud_vace/adapter_d/run_20261110_050724，首步/完成以run真实记录为准。15,881份实测点云及配对mesh已按任务单解压并删除经验证的源ZIP；mesh仅用于弱监督，未知缺口不计损失。八栋评测、其他对照及upload完成报告待后续用户通知，不能自动启动。


## 2026-11-10：D组八视角Adapter训练阶段完成

唯一run `outputs/pointcloud_vace/adapter_d/run_20261110_050724` 已完成3000次更新，2026-11-10 13:04:49（UTC+09）退出0；Adapter/LoRA最终权重及完整恢复点通过CPU读回核查。[训练阶段报告](../upload/20261110_点云D组_八视角结构Adapter微调_训练阶段实验报告.md)已归档。有效弱区仅162/3000批（5.4%），视频效果尚待验收；八栋推理、其他对照及evaluate.py未启动。状态查询不能重跑成功训练，旧启动记录保留。


## 2026-11-10：D组八栋开发建筑视频验收启动

[八栋评测入口](pointcloud_vace/adapter_d_eval/README.md)。用户授权D-3000最终增量的8次81帧推理，run为outputs/pointcloud_vace/adapter_d_eval/run_20261110_150749；实际首步看sampling_started.json。沿用832×480/30步/CFG5/seed20261018，原相机与输入，结构分支剔除隐藏实测点。后台自动保存原始PNG/视频、对比与evaluate.py指标；首步即交接，完整效果与报告留后续复核。无新训练或A/B/C组预算。


## 2026-11-10：D组八栋视频与自动指标生成完成

固定run `outputs/pointcloud_vace/adapter_d_eval/run_20261110_150749` 已完成8条81帧/648原PNG、8原视频＋8对比视频及全部指标，2026-11-10 16:02:08（UTC+09）退出0，耗时47分20秒。[生成完成报告](../upload/20261110_点云D组_八栋81帧推理与自动指标_实验报告.md)已归档。16条视频及696张PNG重新完整解码通过；人工视觉复核待进行，自动指标存在高参考背景误填与首尾差异疑点。以status/exit判断完成，不把progress旧running字段当重跑依据；本次没有新GPU任务。


## 2026-11-11：三栋固定三维补点与D-3000对照

[新实验入口](pointcloud_vace/fixed_points_three/README.md)，[详细报告](../upload/20261110_三栋固定三维补点_与D3000对照_实验报告.md)。固定run `outputs/pointcloud_vace/fixed_points_three/run_20261110_234704`，CPU完成01/eave_3230、04/strip_5016、07/wall_10416各81帧与四栏视频；三栋全部冻结候选后才读参考评分，不加载模型或新训练。首尾一致且观测保留；04候选20/保留6，0.3m覆盖10/42（保留9/42），像素漏补100%，候选位置/条带覆盖首先不足。07拒绝添加避免黑块，不代表天然缺口恢复；500步模块未执行。9视频/508PNG完整读回，243帧静态复核及详细报告完成，历史实验保持原样。


## 2026-11-12：单张参考图V2循环17帧对照启动

用户授权并要求首条正式带参考推理启动即交接。[实验入口](pointcloud_vace/reference_guidance_eval/README.md)，run=`outputs/pointcloud_vace/reference_guidance_eval/run_20261112_025414`；wall_13697、78/79/000–014，中心006，A无参考/B残缺参考/C补全草稿参考，各seed20261018/20261019，共6条17帧，原D-3000权重，无训练。B首步于2026-11-12T02:55:30.865386+09:00完成，原生参考条件与Adapter/CFG布局已共同通过实际采样。后台自动保存视频与简单对照；完整视觉复核和upload报告待用户后续通知。旧结果保持原样，历史正文继续保留。


## 2026-11-12：单张参考图V2生成阶段已完成

`outputs/pointcloud_vace/reference_guidance_eval/run_20261112_025414`已于03:01:57（UTC+09）正常退出0，397.451秒完成六条17帧A/B/C配对推理，另有一条对比视频和五张关键帧；新训练0，状态complete_awaiting_visual_review。用户本次只要求查看进度，视觉复核和正式报告仍等待后续通知；原启动快照保留，不重跑成功任务。


## 2026-11-12：单张参考图V2交付精简

用户要求按既有MD规则精简，`outputs/pointcloud_vace/reference_guidance_eval/run_20261112_025414`已实际删除参考图、关键帧、缓存、配置、日志、指标与状态等中间文件，无隐藏归档。现只留`00_对比视频.mp4`、六条`A/B/C_seed_<seed>.mp4`和`report.md`；视频原样保留，无重编码或GPU重跑。当前完成状态和参数读report.md，旧status命令/分组路径属于清理前历史。说明已同步upload，标注视觉待复核，完整效果分析留待用户通知。


## 2026-11-13：全部OBJ完整几何GT离线采样已启动

用户授权仅制作完整几何GT，输入data/pointcloud_vace/sample-data-organized/mesh全部15,890个OBJ，输出data/pointcloud_vace/mesh_gt_pointcloud同名二进制PLY；50,000面积均匀候选→官方PyTorch3D CUDA FPS16,384，XYZ double/法向float，局部float32仅用于GPU选索引。作者仓库已下载且未实例化有删除行为的CustomDataset。合成网格完整调用已通过，独立tmux mesh_gt_20261113_022544、PID1768992运行；启动快照成功77/15,890、失败0，不等待结束，不自动验证/训练。代码见experiments/pointcloud_vace/mesh_gt_sampling，日志/依赖/编译临时物在数据盘mesh_gt_runtime；本轮日志和缓存是用户明确要求的运行材料，不能套用旧视频组清理范围删除运行中作业或GT。
