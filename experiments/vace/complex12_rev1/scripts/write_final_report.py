"""Summarize completed, frozen results and the recorded visual inspection."""
import csv
from complex_common import RUN


def read(name):
    with (RUN / 'metrics' / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


summaries = {s: {r['method']: r for r in read(f'{s}_summary.csv')} for s in ('train', 'val')}
buildings = {(r['cs'], r['method']): r for r in read('building_metrics.csv')}
methods = ['B', 'L', 'C', 'U', 'S', 'Damaged']
metrics = ['masked_mae', 'masked_mse', 'masked_psnr', 'structure_sobel_l1',
           'known_mae', 'loop_full_mae', 'loop_masked_mae']


def relative(a, b, key):
    return 100 * (float(a[key]) / float(b[key]) - 1)


def summary_table(split):
    lines = ['| 方法 | 缺损 MAE↓ | 缺损 MSE↓ | PSNR dB↑ | 结构 Sobel↓ | 保留区变化↓ | 全图闭环↓ | 缺损区闭环↓ |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for method in methods:
        row = summaries[split][method]
        values = [f'{float(row[k]):.3f}' if k == 'masked_psnr' else f'{float(row[k]):.6f}' for k in metrics]
        lines.append('| ' + ' | '.join([method, *values]) + ' |')
    return '\n'.join(lines)


pair_lines = ['| 划分 | 比较 | 缺损 MAE | 结构 Sobel | 保留区变化 | 全图闭环 | 缺损区闭环 |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
for split, label in [('train', '训练8栋'), ('val', '验证4栋')]:
    for a, b in [('C', 'B'), ('S', 'C'), ('S', 'U')]:
        changes = [f'{relative(summaries[split][a], summaries[split][b], k):+.2f}%' for k in
                   ['masked_mae', 'structure_sobel_l1', 'known_mae', 'loop_full_mae', 'loop_masked_mae']]
        pair_lines.append('| ' + ' | '.join([label, f'{a} 相对 {b}', *changes]) + ' |')

building_lines = ['| 建筑 | 身份 | C−B（MAE / Sobel） | S−C（MAE / Sobel） | S−U（MAE / Sobel） | 缺损 MAE 最低方法 |',
                  '| --- | --- | ---: | ---: | ---: | --- |']
for i in range(1, 13):
    cs = f'CS{i:02}'
    identity = '训练' if buildings[cs, 'B']['split'] == 'train' else '验证'
    changes = []
    for a, b in [('C', 'B'), ('S', 'C'), ('S', 'U')]:
        changes.append(' / '.join(f'{float(buildings[cs,a][k])-float(buildings[cs,b][k]):+.6f}'
                                  for k in ('masked_mae', 'structure_sobel_l1')))
    best = min(methods[:-1], key=lambda m: float(buildings[cs, m]['masked_mae']))
    building_lines.append('| ' + ' | '.join([cs, identity, *changes, best]) + ' |')

visual = [
    ('CS01', '训练 / 304', '墙面矩形补丁明显减轻；全程保持填补，仍有屋面与墙面软灰度渐变。', 'S 与 C/U 接近，S 对 U 的缺损 MAE 仅降低0.04%。'),
    ('CS02', '验证 / 90', 'B035–B068 的 Base 黑槽在 C/U/S 中持续消失，后段未重开。', '三组视觉接近；L 的该栋 MAE 最低，S 的 MAE 略差于 U。'),
    ('CS03', '训练 / 1019', '起始屋面孔和 B036–B070 墙面黑槽均被填上；远侧檐线仍有细线与明暗变化。', 'U 的 MAE 最低；S 的 MAE 比 C/U 略差。'),
    ('CS04', '验证 / 688', '大块折面减轻，但 B033–B053 屋檐与端墙额外折痕仍在，B040–B050 明显。', 'S 未消除残留折线；相对 Base 的结构 Sobel 仅降低约0.51%。'),
    ('CS05', '训练 / 986', 'Base/L 在 B001–B006 重新开洞；新 LoRA 大幅减轻，但 B003–B007 仍有浅缺口。', 'U 局部更平顺且 MAE 最低；S 的 MAE、结构指标均差于 U。'),
    ('CS06', '训练 / 1251', '跨屋脊缺口填上；B009–B026 亮接缝仍弯曲，B052–B061 另一坡面细折痕仍在。', 'S 的 MAE 最低，但结构 Sobel 差于 C；尚未准确恢复 GT 的直线交界。'),
    ('CS07', '训练 / 1432', 'L形体量连续，Base 大屋面补丁减轻；后段仍有边缘起伏及软阴影。', 'S 的整栋指标改善，但 B010 附近小灰斑比 C/U 更明显，原尺寸已核实。'),
    ('CS08', '验证 / 1805', '高低台阶一直保持完整；B015–B023 的边缘突起减轻，细接缝及亮斑仍在。', '属于局部改善有限的样本；C/U/S 接近，U 的 MAE 最低。'),
    ('CS09', '训练 / 24', '大补丁和凹槽减轻；B000–B006 附属屋面上方浅凹及 B036–B044 额外折线仍在。', 'S 的 MAE 最低，但结构 Sobel 比 Base 高4.75%；末帧比首帧完整，仍不闭环。'),
    ('CS10', '训练 / 923', 'Base/L 在 B000–B033 的大洞、L 在 B077–B080 的黑洞均未在 C/U/S 中重现。', '三组均明显改善；C 的 MAE 最低，S 无明显额外视觉优势。'),
    ('CS11', '训练 / 346', 'B036–B062 的大折面和凹槽明显改善；B041–B048 细接缝、后段灰度变化仍在。', 'S 的结构 Sobel 最低；L 的该栋 MAE 最低，三种新 LoRA 视觉很接近。'),
    ('CS12', '验证 / 1344', '早段屋面补丁减轻；B027–B044 檐口与端墙仍弯折，后段屋脊略有波动。', 'S 的 MAE 比 C 低14.40%、比 U 低5.93%，但细部几何未明显优于二者。'),
]
visual_lines = ['| 建筑 | 身份 / 源编号 | 完整81帧的主要观察 | S 与其他方法的关系 |',
                '| --- | --- | --- | --- |']
for cs, identity, observation, comparison in visual:
    visual_lines.append(f'| [{cs}]({cs}/01_五方法同步对比.mp4) | {identity} | {observation} | {comparison} |')

report = f'''# Complex12 Rev1 十二栋单轨小试：最终实验结论

实际完成日期：2026-10-23（北京时间）。C/U/S 各完成300次真实优化器更新；B/L/C/U/S 共60条完整81帧轨迹、4860张原始预测PNG、12条同步视频、12张高清总览，以及逐帧、逐栋、训练/验证汇总指标均已生成。

## 主要发现

**建筑补全 LoRA 的收益明确，S 的额外结构收益较小。** 普通 LoRA C 相对 Base，训练集缺损 MAE 降低71.00%，验证集降低64.21%；验证集结构 Sobel 误差降低26.84%。四栋验证建筑的缺损 MAE 均改善，说明收益不限于训练建筑，但这只支持本数据分布、给定 Exact、35°单轨条件下的迁移。

**按缺损区平均 MAE/MSE/PSNR 和结构 Sobel，S 在训练与验证汇总中最好；综合所有指标尚无全面赢家。** 验证集 S 相对 U 的缺损 MAE 仅降低1.80%，结构 Sobel 仅降低0.19%，PSNR 增加0.097 dB。C/U/S 的多数实际轮廓很接近，S 没有持续消除屋脊、檐口与墙面交界的细折痕。

**S 相对 U 的验证集 MAE 优势集中在 CS12。** CS02、CS04、CS08 的 S MAE 均略高于 U；CS12 降低5.93%拉低了四栋均值。不能把汇总小幅领先解释为每栋都胜出，也不能从4栋、一次配对训练和一个推理种子声称统计显著。

**完整绕转仍有局部形变和首尾不一致。** CS02、CS03、CS10 的大黑槽/黑洞在新 LoRA 中持续改善；CS04、CS06、CS09、CS12 的细折线、弯曲接缝或浅凹槽仍随视角显露。S 在 CS07 的 B010 附近还出现小灰斑。S 的保留区变化和全图闭环误差高于 C/U，因此不能将缺损区均值领先等同于整体稳定性更好。

## 训练集与验证集指标

全部误差在 RGB [0,1] 上计算；↓越低越好、↑越高越好。常规指标只使用000–079；闭环比较真实模型输出000与080。楼内先汇总有效像素，再计算每栋指标；跨栋等权平均。PSNR 是各栋汇总 MSE 转换后的平均值。下表保留6位小数，原始精度及有效像素/帧数见 CSV。

### 训练集：8栋

{summary_table('train')}

### 验证集：4栋，主要泛化观察

{summary_table('val')}

Damaged 是输入直接作为预测的无需GPU参考。其保留区和闭环误差为0来自输入定义，不代表补全成功。所有模型在缺损区均明显优于这一参考。旧 L 的训练建筑、文本和训练过程不同，仅作迁移参考；其保留区变化、全图闭环汇总最低，CS02/CS11 的缺损 MAE 也最低，但它在 CS10 出现严重黑洞，不能凭单一均值替代逐栋检查。

### 三组关键比较

表中是两个划分均值的相对变化 `(左/右−1)×100%`；这里均为误差，负值表示改善，正值表示退步。

{chr(10).join(pair_lines)}

训练集 S 的缺损 MAE 为0.009472，验证集为0.010229，验证误差略高；C/U 也有相同方向的差异。验证集的实际黑槽修复支持迁移，但样本难度不同，不能仅以这个差值推算过拟合程度。S 的训练缺损区闭环改善较大，主要受 CS09 影响；验证集 S 的缺损区闭环比 U 反而增加3.26%，尚无稳定的时间一致性优势。

S 相对 C 的保留区 MAE 在12栋均增加，全图闭环误差也在12栋均增加。验证集对应均值从0.001844到0.001924、从0.002973到0.003579。绝对量较小，但方向一致；不能省略这项代价。保留区变化仍低于 Base，变化可能包括背景、亮度渐变和边缘细节，不能全归因于结构损坏。

## 逐栋定量差异

每个单元格是“缺损 MAE 绝对差 / 结构 Sobel 绝对差”，负值表示左方法更好；“最低方法”只按该栋缺损 MAE 排序，不表示视觉或所有指标最佳。全部7项指标的逐栋差值见 [paired_differences.csv](../metrics/paired_differences.csv)。

{chr(10).join(building_lines)}

值得单独保留的反例：C 在 CS04、CS09 的结构 Sobel 未优于 Base；S 在 CS03、CS05、CS06 的结构 Sobel 差于 C。CS09 即使 S 的缺损 MAE 较低，结构 Sobel 仍比 Base 高4.75%。平滑灰度误差下降与正确恢复折线位置并非同一结论。

## 十二栋完整轨迹观察

已逐一查看60条轨迹的000–080连续帧联系图，结合固定五关键帧及有疑点的512原尺寸帧检查。同步视频包含全部81帧、16FPS，每格512×512，按 Before、GT、B、L / C、U、S、空位排列。固定总览为 B000/B020/B040/B060/B080；未按预测效果挑选最好看的视角。

{chr(10).join(visual_lines)}

CS05 是固定五帧容易漏掉的问题：B000 看起来完整，不代表 B001–B007 没有缺口。完整轨迹检查因此保留了早段反例。CS06（源1251）才是本轮跨屋脊挑战，不是旧 Pilot8 的 P06（源1848）。CS09–CS12 的结论限定于复杂屋顶背景下局部屋面/邻墙补全，不能据此声称恢复了被大范围删除的主屋脊。

本次逐帧视觉检查可识别大洞、折痕、形状漂移和首尾差异；没有专门量化实时播放下的细微高频闪烁，因此不作“无闪烁”结论。闭环只约束首尾相似程度，不能代替整个环绕过程的一致性评估，也未把相邻相机视角的直接像素差当作闪烁指标。

## 公平性与指标口径

- 数据保持原 Complex12 Rev1，训练为 CS01/03/05/06/07/09/10/11，验证为 CS02/04/08/12；验证未参加训练或检查点选择。
- C/U/S 从同一保存的新 LoRA 初始化开始，使用各自新建的 AdamW、相同采样/噪声/时间步及随机序列；均固定比较第300次更新。唯一实验变量是普通、均匀缺损、结构带三种损失权重，不按训练 loss 跨组排序。
- 训练使用连续17帧窗口，先截取再 VAE 编码，合法全黑 Exact 帧保留。推理使用完整81帧路径，512×512、30步、CFG 5、seed 20261018、LoRA scale 1.0、同一实际文本；只输入 Damaged 派生条件、ModelMask 与文本。
- GT RGB 和结构图仅用于训练监督或推理后评测；预测未粘回已知区域，未使用 GT 回贴、调色、强制闭环，也没有将独立17帧结果拼成81帧。080保留真实模型输出。
- 结构带先对 raw_visible 做7×7膨胀再与 Exact 相交。Sobel 使用灰度系数0.299/0.587/0.114、3×3核除以8、reflect边界，取两个方向绝对误差的均值。空区域记 N/A；零 MSE 的 PSNR 保留 Infinity。完整映射见 [mask_loss_mapping.md](../mask_loss_mapping.md)。
- 像素/损失尺度覆盖率及窗口 E 空比例见 [mask_coverage.csv](../metrics/mask_coverage.csv)。合格窗口 E 空比例均为0；损失网格上的结构带平均覆盖约0.85%–1.97%。额外加权集中在较小区域，可能是效果接近的原因之一，但本轮没有验证这一因果解释。

## 下一步只建议一个实验

**做一次预先固定方案的 C/U/S 多训练种子配对重复实验。** 保持当前数据划分、VACE骨干、损失定义、300次更新及推理条件，每个训练种子内仍使用共同初始化和采样；检查 S 对 U 的小幅验证优势是否稳定，并同时观察保留区、闭环和 CS06/CS09 的几何反例。当前最需要回答的是结构加权的微小收益能否复现，现有证据还不足以优先增加结构模块或据此扩大训练预算。此项仅为建议，本轮没有启动额外种子、训练或推理。

## 完成记录与结果入口

正式交付无未完成任务。权重位于 [C step300](../checkpoints/C/step300.safetensors)、[U step300](../checkpoints/U/step300.safetensors)、[S step300](../checkpoints/S/step300.safetensors)。训练日志逐组300条，记录实际优化器更新及有限 loss/梯度；原始预测60×81张，视频逐一解码确认81帧、16FPS，并与对应原始面板核对顺序与编码偏差。逐帧常规指标5760条（6种方法含Damaged×12栋×80帧），逐栋72条，关键配对差值36条。

执行中 B/CS04 曾在完整预测保存后原生异常退出；81张输出随后独立验证可读并复用，没有为此重复推理，原生退出原因未被认定解决。恢复服务器挂载后，C 从已保存的第40次更新继续至300；U/S 各300，已成功轨迹未重跑。历史异常保留在 [logs/errors.md](../logs/errors.md)，恢复记录见 [logs/recovery_20261023.log](../logs/recovery_20261023.log)。

打开 [README_从这里开始](README_从这里开始.md) 或 [可播放评审图册](评审图册.html) 查看全部建筑；数值入口为 [训练集指标](训练集指标对比.csv)、[验证集指标](验证集指标对比.csv) 和 [metrics](../metrics/)。实际完成状态见 [progress.json](../progress.json)，配置见 [resolved_config.json](../resolved_config.json)。
'''

path = RUN / '00_看结果' / '最终实验结论.md'
path.write_text(report, encoding='utf-8')
print(path)
