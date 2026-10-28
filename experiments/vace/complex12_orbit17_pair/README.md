# Complex12 Orbit17：线性与循环17帧窗口配对实验

2026-10-27（UTC+09:00）建立。用户已授权两组都执行，并要求确认开始训练后结束本轮对话。当前启动快照见本文末尾与根目录长期文档；计划完成量不等于已经完成。

## 研究问题与对照

在相同普通C、相同初始化、相同600次更新预算下，允许17帧窗口跨过079→000，能否改善整圈末端和首尾一致性，同时保持建筑结构及完好区域？训练只用Rev1，OriginalHard只作困难缺损评测。

此前C_mix已完成，但Hard CS10连接与CS12形状/闭环未解决，Rev1 CS10还出现新条纹。这里重新从共享初始化训练，不使用C_mix、旧C、U、S或L权重继续训练。旧C原始预测只作历史参照。

| 组 | 窗口 | 更新 | 初始化与优化器 |
| --- | --- | --- | --- |
| C_lin17 | 起点0–63，连续17帧 | 600 | 共享LoRA初始参数，新AdamW |
| C_ring17 | 起点0–79，帧号按模80排列 | 600 | 相同初始参数，另一份新AdamW |

训练帧只有000–079；080不进入任何训练窗。循环起点72为072…079、000…008，起点79为079、000…015。每个窗至少有一帧Exact非空，窗内全黑Mask帧照常保留。

## 数据、来源和随机配对

项目根为 `/mnt/windowsE/chuanjun/building_wan`。来源只读：

- Rev1 run：`outputs/ComplexSingleOrbit_Rev1_Pilot_20261022`。
- Rev1数据：`data/buildingfiller_import/Complex12/BuildingFiller_Complex12_512_Rev1_20261022`。
- Hard run：`outputs/vace/complex12_originalhard_eval/run_20261026_012033`。
- Hard数据：该run的 `input_data/BuildingFiller_Complex12_OriginalHard_512_20261025`。
- 初始LoRA：Rev1 run的 `checkpoints/shared_initialization.safetensors`。
- 原300行：Rev1 run的 `shared_sampling_table.csv`。

训练：CS01、03、05、06、07、09、10、11；验证：CS02、04、08、12。验证建筑任一版本均不进入训练。Hard CS09–11保留“训练建筑的新缺损”身份，只有CS12是未训练建筑的新缺损；不是全新盲测集。

线性组前300行保留旧表所有字段。301–600行用独立局部`Random(20261027)`，逐行依次抽建筑、该建筑合法线性起点、`timestep_index=randrange(1000)`、`noise_seed=randrange(2**31)`、`stochastic_seed=randrange(2**31)`。建筑顺序固定为上面的8栋。

循环组另用`Random(20261028)`按行抽该建筑合法循环起点；建筑、更新顺序、timestep与两个逐步seed和线性组相同。版本均为Rev1。两个CSV各600行；局部版本/窗口抽样不消耗训练随机状态。覆盖表记录300/600两个预算下每栋80视角的次数、缺口可见次数及17个局部位置次数；次数不是非零梯度贡献。

## 冻结训练与编码口径

复用旧C真实实现：Wan2.1-VACE-1.3B原骨干；仅VACE分支LoRA，rank/alpha16/16、dropout0，q/k/v/o/ffn.0/ffn.2。512×512、17帧、batch1、累积1、workers0；AdamW恒定5e-5、wd0、betas0.9/0.999、eps1e-8、clip1；BF16骨干、FP32 LoRA、原梯度检查点。训练seed20261022。

普通FlowMatch目标与原scheduler时间权重、全latent MSE不变，不加入U/S区域加权。合法零时间权重仍按采样表执行并计数，不重抽、不补步；AdamW动量可能仍使该步参数变化，因此不能把版本/窗口次数当有效梯度次数。

GT、Damaged、Exact及结构标签先按同一`frame_ids`取出完整有序17帧，再整窗编码一次。循环窗不拼接两段latent，不从81帧latent裁切；局部位置仍是0–16。GT只作训练目标。Damaged的原Exact白区填RGB127，ModelMask沿用原二值RGB与官方时间映射，不扩张、不羽化。结构标签仅供共用函数检查，普通C不使用其加权。

新缓存以“数据版本＋建筑＋完整有序frame_ids＋编码设置”区分；两组完全相同的窗口可以共享新缓存。缓存、文本编码均在新run，旧缓存不修改。VAE tiled=true、tile_size=[30,52]、tile_stride=[15,26]。

## 自动执行顺序和断点

1. CPU准备：输入基本检查、两份600行表、覆盖统计与跨界读取检查。
2. 线性组从0训练到300，保存LoRA及完整恢复状态。
3. 线性300分别对Rev1 CS10/12、Hard CS10/12运行四条81帧探针；CPU比较旧C指标和固定视角，留下提前复现检查记录。
4. 线性组恢复完整AdamW/RNG，从301继续到600，共600次，不追加另一轮600。
5. 循环组从共同初始化训练到600，保存300/600权重。
6. 循环300运行同四条探针。
7. 两组600各运行16条：先Hard CS10/12/09/11，再Rev1 CS01–12。
8. CPU生成配对指标、视频与问题图；状态改为`awaiting_visual_review`，等待后续视觉复核和报告。

提前核对是保留复现证据，不按主观效果淘汰实验组；用户要求两组都跑。运行异常会保留证据和最后有效状态并停止队列，不擅自改变预算/规格。每10次更新保存`resume_latest.pt`，300另存`resume_step300.pt`；包括LoRA、AdamW、Python/NumPy/Torch/CUDA RNG及对应历史。组间状态独立。失败的部分优化器更新不覆盖最后有效恢复点，恢复时从该点重新执行剩余采样行；若发生回放，后续交接必须如实说明。

单GPU串行、沿用既有GPU锁和tmux方式，日志显示真实更新、真实去噪进度及完成轨迹数。现有原生退出问题没有被本实验宣称修复；继续使用旧run的GDB退出码包装。

## 推理与交付量

新推理总计40条：300步2组×4探针=8；600步2组×16输入=32；合计3240张原始预测PNG。旧C直接复用，不重跑B/L/C/U/S/C_mix，不新增Hard CS01–08推理。

每条独立固定seed20261018；81帧、512×512、30步、CFG5、sigma_shift5、vace_scale1、单份LoRA scale1、denoising_strength1、cfg_merge=false、BF16、rand_device=cpu，原VAE tiled设置。复制旧C正负文本，不扩写。推理读取Damaged+Exact构造的条件与文本，绝不读取GT。预测不回贴、不调色、不以000替换080。

正向原文：

> A complete untextured gray building with continuous roof surfaces and walls, viewed during a smooth 360-degree orbit against a plain light-gray background.

负向原文：

> 彩色纹理，窗户装饰，新增建筑，建筑形变，结构闪烁，背景变化，文字，水印

`00_看结果/`分Rev1与OriginalHard：每输入2条600步补后视频和1条2×3同步视频，共48条。同步布局Before、GT、旧C / lin600、ring600、信息；每块内容完整512×512，16FPS、81帧。300步不额外做视频。

固定问题图：Hard CS09/006、CS10/024与058、CS11/073、CS12/000与080，另加Rev1 CS10/020；四个探针各一张Before/GT/旧C/lin300/lin600/ring300/ring600七行总览。Hard CS10总览用000/024/040/058/080，其余探针000/020/040/060/080。

## 评测与解释

原口径保持：Exact内MAE/MSE/PSNR，7×7结构带与Exact交集内Sobel，Exact外对Damaged的known MAE；常规000–079，首尾000/080另算。楼内按有效像素加权、跨楼等权；空区N/A、零MSE的PSNR为Infinity、零基准不强算百分比。

新增000/080各自对GT的全图/Exact MAE，并保留相互闭环误差；前000–016、中017–062、后063–079分别统计。主要配对为ring300对lin300、ring600对lin600、同组600对300（四探针）及lin300对旧C复现。600对旧C涉及预算变化，另标明，不能将它解释为纯窗口效果。

600完整评测分Rev1训练8、验证4、Hard训练身份3和验证CS12四组；逐栋列退步，单列输入未变的CS01–08。不能把Hard收益与Rev1损失合并成一个总分。低MAE或低闭环不能独立证明连接结构正确，之后需结合固定反例及完整轨迹检查屋脊、连接面、折面、轮廓和首尾形状。自动视频解码检查不等于人工播放、闪烁量化或三维验证。

## 代码和产物位置

专用源码位于本目录`scripts/`：

| 文件 | 职责 |
| --- | --- |
| complex_common.py | 本run适配、帧序列、数据与旧公共函数接口 |
| prepare.py | CPU准备、冻结设置、采样表、覆盖图、run入口 |
| complex_train.py | 两组共用普通C训练、整窗编码缓存、完整状态恢复 |
| complex_infer.py | 单LoRA独立81帧预测与真实加载路径 |
| complex_evaluate.py | 复现探针、配对/分组/分段/端点指标及结果导航 |
| media.py | 原尺寸同步图、视频编码与解码检查 |
| run_pilot.py / status.py | 串行任务队列 / 只读状态 |
| test_orbit17.py | 配对、数据隔离、恢复预算、指标和视频CPU检查 |

直接复用Rev1训练中的普通C目标、模型加载、优化器更新、参数装卸和文本编码函数；复用Rev1纯像素指标，以及项目现有VACE输入、LoRA加载、推理参数和终端进度函数。适配器将这些函数绑定到新run，旧固定300步入口不调用，不修改第三方核心或旧实验源码。

实际产物目录：`/mnt/windowsE/chuanjun/building_wan/outputs/vace/complex12_orbit17_pair/run_20261027_200842/`。

`checkpoints/<组>/`存300/600和必要恢复状态；`predictions/<组>/step<步数>/<版本>/<建筑>/`存原PNG；`metrics/`、`logs/`、`cache/`、两份采样CSV、`resolved_config.json`、`progress.json`及`resume.sh`均在此run。`runs/`和`configs/`只作导航链接，不移动或复制大结果。正式源码不放进outputs。

只读查询：

```bash
bash /mnt/windowsE/chuanjun/building_wan/outputs/vace/complex12_orbit17_pair/run_20261027_200842/resume.sh
```

用户已授权的同run任务恢复使用该入口`--resume`；不作为日常状态查询。尚在运行时不再次启动它。

训练启动后记录实际更新、进程、tmux及检查结果并提交Git。自动队列结束只标记计算/媒体完成；视觉复核、最终中文报告及同内容upload便携Markdown留待用户通知。根长期文档按日期追加，旧历史不删。任何新预算、配比或几何/时序模块均不在本轮自动展开。


## 2026-10-27：Orbit17线性/循环配对实验已启动

核查时间：2026-10-27T20:16:32+09:00（UTC+09:00）。用户已授权线性组和循环组都执行，并要求确认实际开始训练后结束本轮对话。当前status=running，C_lin17已完成22/600次真实AdamW更新，C_ring17为0/600（待串行执行），推理0/40；日志已有有限loss及参数实际变化。这是启动快照，不是两组完成或效果验证。

tmux：`complex12_rev1:orbit17_pair`；runner 2469437，GDB worker 2469491。本轮代码提交`1a82bd5`，冻结配置ID `67c161780e5f164736e8de1e16cf43db2653081475e170d96007c25a4f5c8490`。

本次实测14项Orbit17专项CPU检查通过（1.466秒），298项项目CPU回归通过（50.203秒），规定shell语法/Python编译及git diff --check通过；没有额外GPU试跑。准备时覆盖图首次因服务器缺少matplotlib失败，随后改用现有Pillow生成并通过；没有安装依赖，未重复GPU任务，prepare.log保留实际错误与后续成功。旧run、原始数据、模型、第三方源码及19项既有无关工作区修改保持原样且不纳入本轮提交。

每10更新保存完整恢复状态，300/600保存LoRA，300另保留恢复点；两组状态独立。缓存按版本/建筑/完整有序frame_ids/编码设置区分，放在新run。自动队列完成后标记awaiting_visual_review；之后按用户通知进行完整轨迹视觉复核、run最终中文报告及upload同内容便携报告。此时不能预先宣称循环优于线性，也不能把视频解码当人工播放或三维验证。新预算、配比或模块不自动展开。


## 2026-10-29：Orbit17配对实验复核与最终报告完成

计算于2026-10-27T23:41:29+09:00结束，用户随后通知复核/报告收尾；10月28日至29日完成本次复核。实际run为`outputs/vace/complex12_orbit17_pair/run_20261027_200842`。两组各600真实AdamW更新、40条81帧/3240原PNG、48视频、7固定问题图和4张最终七行探针总览齐全；另保留4张早期探针图与1张覆盖图。45个队列worker均单次启动、EXIT 0；线性300→600为计划内完整optimizer/RNG恢复，循环连续0→600，没有失败后回放或新增GPU任务。

[详细便携报告](../../../upload/20261029_Complex12_Orbit17_线性与循环窗口配对实验报告.md)；[直接看结果](../../../outputs/vace/complex12_orbit17_pair/run_20261027_200842/00_看结果/README_从这里开始.md)。本节更新此前“等待用户通知”的状态，旧段落仍是对应时刻的历史记录。

两组仅用Rev1训练8栋、验证4栋不变，Hard只评测；普通C、共同初始LoRA与新AdamW、600预算相同。线性起点0–63，循环模80起点0–79，完整17帧同步读取后编码，080不训练。原线性前300行保留，扩表seed20261027、循环起点seed20261028，其他逐步随机字段配对。每组600中12步合法零时间权重按表计数，不补步。循环600有127个跨界窗，跨训练8栋的000/079曝光139/135，线性为7/15。

同为600更新，ring对lin在16输入中MAE 13个改善、Sobel 10个改善，但known MAE 12个退步、全图首尾MAE 15个退步，Hard四栋缺口首尾全部退步。四组MAE变化分别为Rev1训练−7.73%、验证−2.68%、Hard训练身份三栋−11.98%、Hard验证CS12−12.56%；不能写成循环普遍解决闭环。

Hard CS12旧C/lin600/ring600缺口MAE为0.056055/0.023890/0.020890，主要错误折面明显缓解；缺口首尾为0.115625/0.017480/0.018376。两组600均改善，预算贡献不可归于循环独有。Hard CS10/B024连接仍错，B058局部有改善；ring相对lin MAE−17.07%仍不代表结构修复。Rev1整体均值改善，ring相对旧C仍有CS01/06/08/10四栋MAE退步；未见前轮C_mix CS10大面积条纹。lin300四探针对旧C的MAE差在±0.72%内，为近似复现，不是逐bit复现。

训练总计4561.559秒；40条推理合计7541.678秒；整个队列3小时27分26秒。训练峰值allocated/reserved 15.139/15.393GiB，推理17.143/25.832GiB。4份LoRA、600步完整恢复状态、逐步loss与采样表一致性均已核对。

本次CPU完整解码3240原PNG、48条视频及16张展示PNG；全部40条000–080通过连续静态联系图复核（缩略内容），7张固定问题图按完整512内容检查。没有实时视频播放、定量闪烁或三维验证，不能把文件可读写成模型效果全达标。原evaluate.done的待复核状态与既有启动正文保留为历史，当前交付状态以progress和logs/final_review.done.json为准。

下一步仅建议第二训练随机序列的600更新线性/循环配对复验，检验像素收益与首尾代价是否可重复；尚未执行或排队。Hard未训练建筑仅CS12，是开发验证，不能夸大泛化。旧run、权重、原始数据/预测和已有报告保持不变；本次收尾不自动开启任何训练、推理或新模块。
