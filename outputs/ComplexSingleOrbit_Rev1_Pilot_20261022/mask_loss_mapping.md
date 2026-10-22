# Complex12 Rev1：条件与损失标签

输入始终按原PNG帧号读取512×512，无裁剪、抽帧、缩放或随机增强。Exact白色表示生成区域；继承项目适配器，将白区Damaged填RGB127、黑区不变。ModelMask是Exact的RGB复制。真实加载器分别得到条件[-1,1]（填灰=-1/255）、Mask[0,1]。不扩大或羽化条件Mask。

推理读取Damaged、Exact和统一文本；不读取GT或GT结构，不回贴、调色或替换080。VACE标准条件内部将Mask按8×8重排为64通道，再nearest-exact映射时间轴；这套官方条件逻辑保持原样。

训练先同步选17连续帧，再分别编码GT目标及Damaged/ModelMask条件。缓存仅含冻结T5/VAE结果，不缓存可训练VACE输出，不使用历史T2V缓存。GT只作为FlowMatch训练target，GT结构只参与损失权重。

M_pixel = Exact/255。E_pixel = raw_visible先做7×7最大值膨胀，再AND Exact；不使用先相交的damaged_region生成结构带。

当前Wan VAE每个窗口的首帧单独编码，后续每4帧一组：局部[0]、[1,2,3,4]、[5,6,7,8]、[9,10,11,12]、[13,14,15,16]。M/E先做空间8×8 max，再在上述时间组内max，得到[1,5,64,64]，对应实际[1,16,5,64,64]目标latent。该分组是保留细信号的监督标签映射，不宣称精确覆盖因果VAE完整感受野。窗口起点即局部0，不套用全81帧时间网格。E≤M恒成立；单像素细线也保留。

C使用原FlowMatch MSE；S权重1+2E；U权重1+aM，a=2sum(E)/sum(M)。U/S额外总权重均为2sum(E)。误差先对通道平均，每样本sum(w*error)/sum(w)，随后乘原scheduler时间权重、进行batch归约。本轮batch1/accum1；M或E空时直接调用原MSE。训练目标依旧noise-latent，不增加RGB重建损失。

共同采样表保存300个实际AdamW更新：均匀选8栋训练建筑，再均匀选该栋Exact非空的17帧窗口。起点0–63、末帧≤079；内部黑帧保留，E空窗口不排除。表中时间步、噪声种子及其他随机种子逐更新共用。C/U/S从同一份新保存初始化开始，各自新建AdamW，固定step300比较。

coverage统计位于metrics/mask_coverage.csv：像素尺度使用000–079，损失尺度使用各栋所有Exact合格17帧窗口的等权平均；E空比例也在该窗口全集上统计。验证集覆盖率只作描述，不用于训练或选参。实际反向检查与映射图保存在checks/及logs/preflight.done.json。

评测仅使用原始预测PNG，常规000–079；闭环单独000/080。RGB/255，缺损区对GT，已知区对Damaged。结构灰度0.299R+0.587G+0.114B，Sobel3×3核除8、reflect边界，E_pixel内两方向绝对梯度差的平均。楼内按有效像素汇总，PSNR从汇总MSE计算；跨楼等权、训练/验证分开。空区域N/A，MSE0的PSNR为Infinity，不封顶。视频不将相邻机位像素差当作闪烁指标。
