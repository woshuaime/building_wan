# 冻结配置、数据划分与使用范围

更新：2026-10-23（UTC+09:00）。本目录保留已有公共参数和历史划分；新想法的配置进入对应experiments/<任务>/<想法>/configs，实际run另保存resolved_config。旧配置的来源字段与缺失状态代表冻结时刻，不因后来模型已到位而改写。

## VACE配置

| 文件 | 适用范围 | 注意事项 |
| --- | --- | --- |
| [vace_p01_465_17f_v1.json](vace_p01_465_17f_v1.json) | 早期P01/source465、B005–B021的17帧参数来源 | 81帧入口显式派生长度/范围；不能把文件名当最终运行帧数 |
| [vace_p01_model_sources_v1.json](vace_p01_model_sources_v1.json) | VACE主模型、T5、VAE、Tokenizer来源与批准身份 | 正式文件现已存在并用于完成实验；不自动下载 |
| [vace_roof_smoke_v1.json](vace_roof_smoke_v1.json) | P04/source1227、P06/source1848各B000–B016，300更新 | 与Complex12训练划分、seed、抽窗方式不同 |

VACE基础模型为Wan2.1-VACE-1.3B、revision574e6a744642ce3bee319afc31496b88bde8aac4。T5/VAE复用现存转换组件，Tokenizer复用现有目录，不等于使用T2V骨干、T2V LoRA或其缓存。

Pilot8正负中文文本从P01配置沿用。Complex12使用自己的英文正向描述及同组负向文本，不能将两个实验的文本视为完全相同。实际条件与参数以对应run的已解析配置为准。

## 历史单轨T2V划分

[single_orbit_same_prompt_v1](single_orbit_same_prompt_v1/)包含：

| 文件 | 内容 |
| --- | --- |
| metadata_train.csv | 1992训练视频 |
| metadata_val.csv | 234验证视频 |
| metadata_test.csv | 118测试视频 |
| metadata_pilot_32.csv | 32视频小规模配置入口，不是新增正式训练集 |
| split_manifest.json | seed20260924、统一prompt、文件列表/内容摘要、原Windows数据根 |

总数2344，比例约85%/10%/5%。manifest中的D盘路径是冻结来源；当前Linux实际路径由项目适配器解析。共享文本context只适用于这组相同prompt，改变prompt或划分不能仅凭样本数继续复用缓存。

building_v1的1912/239/239与这里不同，历史配置位于 [扫描器training/configs](../code/Python_3D_Scanner/training/configs/)；不并入当前VACE任务。

## Complex12配置与新运行

[Complex12配置入口](../experiments/vace/complex12_rev1/configs/pilot_20261022.json)是指向原run/resolved_config.json的链接，避免维护两份内容。样本索引、方法、采样表和真实更新见 [实验说明](../experiments/vace/complex12_rev1/README.md)。

配置修改会改变实验身份。已完成run的冻结参数不覆盖；新研究问题新建idea，相同想法的新运行新建run。文档维护不会重写上述JSON、CSV或模型身份。
