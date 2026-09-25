"""生成 Wan2.1 建筑视频 LoRA 完整实验报告（中文宋体，英文/数字 Times New Roman）。"""
from pathlib import Path
import html, json, re
from PIL import Image as PILImage, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports' / 'Wan2.1建筑视频LoRA完整实验报告_详细版_2026-09-05.pdf'
AS = ROOT / 'outputs' / 'report_assets'
VAL = ROOT / 'outputs' / 'validation'
GEN = ROOT / 'outputs' / 'report_assets' / 'generated'
GEN.mkdir(parents=True, exist_ok=True)
pdfmetrics.registerFont(TTFont('Songti', r'C:\Windows\Fonts\STSONG.TTF'))
pdfmetrics.registerFont(TTFont('TimesNewRoman', r'C:\Windows\Fonts\times.ttf'))
pdfmetrics.registerFont(TTFont('TimesNewRoman-Bold', r'C:\Windows\Fonts\timesbd.ttf'))
NAVY, BLUE, PALE, MUTED, GRID = colors.HexColor('#17365D'), colors.HexColor('#2E74B5'), colors.HexColor('#E8EEF5'), colors.HexColor('#5E6C84'), colors.HexColor('#C9D1DB')

def mixed(s):
    s = html.escape(str(s), quote=False)
    chunks = re.split(r'([A-Za-z0-9][A-Za-z0-9 .,:;_+/=()\-\[\]%]*)', s)
    return ''.join(f'<font name="TimesNewRoman">{c}</font>' if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9 .,:;_+/=()\-\[\]%]*', c or '') else c for c in chunks if c)

styles = getSampleStyleSheet()
for n, fs, lead, col, before, after in [('Body',10.3,17,'#222222',0,7),('Small',8.3,12,'#5E6C84',0,4),('H1',16,23,'#17365D',8,8),('H2',12.5,18,'#2E74B5',6,6),('CoverTitle',24,31,'#111111',0,12),('Sub',12,18,'#5E6C84',0,12),('Caption',8.4,12,'#5E6C84',4,9)]:
    styles.add(ParagraphStyle(name=n,fontName='Songti',fontSize=fs,leading=lead,textColor=colors.HexColor(col),spaceBefore=before,spaceAfter=after,alignment=TA_CENTER if n in ('Title','Sub','Caption') else TA_LEFT,keepWithNext=n in ('H1','H2')))
styles.add(ParagraphStyle(name='TH',fontName='Songti',fontSize=8.7,leading=12,textColor=colors.white,alignment=TA_CENTER))
styles.add(ParagraphStyle(name='TD',fontName='Songti',fontSize=8.5,leading=12,textColor=colors.HexColor('#222222')))
styles.add(ParagraphStyle(name='BulletCN',parent=styles['Body'],leftIndent=13,firstLineIndent=-8,spaceAfter=4))
def P(s, st='Body'): return Paragraph(mixed(s), styles[st])
def pic(path, mw=170*mm, mh=220*mm):
    with PILImage.open(path) as im: w,h=im.size
    k=min(mw/w,mh/h); return Image(str(path), width=w*k, height=h*k)
def fig(path, cap, mw=170*mm, mh=220*mm): return [pic(path,mw,mh), P(cap,'Caption')]
def tab(headers, rows, widths):
    d=[[Paragraph(mixed(x),styles['TH']) for x in headers]]+[[Paragraph(mixed(x),styles['TD']) for x in r] for r in rows]
    t=Table(d,colWidths=widths,repeatRows=1,hAlign='LEFT'); cmd=[('BACKGROUND',(0,0),(-1,0),NAVY),('GRID',(0,0),(-1,-1),.45,GRID),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]
    for i in range(1,len(d)):
        if i%2==0: cmd.append(('BACKGROUND',(0,i),(-1,i),PALE))
    t.setStyle(TableStyle(cmd)); return t
def footer(c,doc):
    c.saveState(); c.setStrokeColor(GRID); c.line(18*mm,16*mm,192*mm,16*mm); c.setFillColor(MUTED); c.setFont('Songti',8); c.drawString(18*mm,10.5*mm,'Wan2.1 建筑视频 LoRA 实验报告'); c.setFont('TimesNewRoman',8); c.drawRightString(192*mm,10.5*mm,f'Page {doc.page}'); c.restoreState()
def make_diagrams():
    f=ImageFont.truetype(r'C:\Windows\Fonts\STSONG.TTF',28)
    # pipeline
    im=PILImage.new('RGB',(1500,420),'white'); d=ImageDraw.Draw(im); boxes=[('原始视频\n2390×81帧',40),('按 OBJ 分组\ntrain/val/test',330),('5 个 17 帧窗口\n9560 train clips',620),('缓存 T5/VAE\n落盘',910),('双卡 Gloo DDP\nLoRA 更新',1200)]
    for text,x in boxes:
        d.rounded_rectangle((x,130,x+230,285),18,fill='#E8EEF5',outline='#17365D',width=4); d.multiline_text((x+115,205),text,font=f,fill='#17365D',anchor='mm',align='center')
    for x in [270,560,850,1140]: d.line((x,208,x+55,208),fill='#2E74B5',width=7); d.polygon([(x+55,208),(x+38,198),(x+38,218)],fill='#2E74B5')
    im.save(GEN/'pipeline.png')
    # slicing
    im=PILImage.new('RGB',(1500,390),'white'); d=ImageDraw.Draw(im); d.text((40,25),'单个 81 帧视频如何变成 5 个训练片段',font=f,fill='#17365D'); d.line((100,220,1400,220),fill='#555',width=5)
    for i in range(81):
        x=100+i*16; d.line((x,205,x,235),fill='#999',width=2)
    for j,(a,b) in enumerate([(1,17),(17,33),(33,49),(49,65),(65,81)]):
        x=100+(a-1)*16; xx=100+(b-1)*16; d.rounded_rectangle((x,130,xx,185),8,fill=['#A8DADC','#BDE0FE','#CDEAC0','#FFD6A5','#DDBEA9'][j],outline='#17365D',width=2); d.text(((x+xx)//2,157),f'{a}-{b}',font=ImageFont.truetype(r'C:\Windows\Fonts\times.ttf',22),fill='#17365D',anchor='mm')
    d.text((100,275),'相邻窗口共享 1 帧，用来保持旋转连续性；训练时每次只读取 17 帧。',font=ImageFont.truetype(r'C:\Windows\Fonts\STSONG.TTF',24),fill='#333'); im.save(GEN/'slicing.png')
    # gpu evidence
    im=PILImage.new('RGB',(1200,520),'white'); d=ImageDraw.Draw(im); d.text((40,25),'正式训练 GPU 监控摘要',font=f,fill='#17365D'); vals=[('GPU 0 利用率均值',18.4,'#2E74B5'),('GPU 1 利用率均值',15.0,'#6A994E'),('GPU 0 峰值显存 GB',4.98,'#F4A261'),('GPU 1 峰值显存 GB',6.54,'#E76F51')]
    for i,(lab,v,col) in enumerate(vals):
        y=100+i*90; d.text((50,y),lab,font=ImageFont.truetype(r'C:\Windows\Fonts\STSONG.TTF',24),fill='#333'); d.rectangle((390,y,1040,y+35),fill='#eee'); d.rectangle((390,y,390+v/20*650,y+35),fill=col); d.text((1060,y),str(v),font=ImageFont.truetype(r'C:\Windows\Fonts\times.ttf',25),fill='#333')
    im.save(GEN/'gpu.png')

def build():
    make_diagrams(); loss=json.loads((AS/'loss_summary.json').read_text(encoding='utf-8')); gpu=json.loads((AS/'gpu_telemetry_summary.json').read_text(encoding='utf-8')); by={x['label']:x for x in loss['runs']}; fg={str(x['gpu']):x for x in gpu['groups'] if x['stage']=='formal_train'}
    doc=SimpleDocTemplate(str(OUT),pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,topMargin=18*mm,bottomMargin=21*mm,title='Wan2.1建筑视频LoRA完整实验报告（详细版）',author='Python_3D_Scanner experiment'); s=[]
    s += [Spacer(1,18*mm),P('实验全过程、方法解释与结果分析','Sub'),P('Wan2.1 建筑视频 LoRA 完整实验报告（详细版）','CoverTitle'),P('面向“让模型认识建筑外观”的方向 B 预研','Sub')]
    s += [tab(['项目','本轮实际情况'],[['基础模型','Wan2.1-T2V-1.3B；仅训练 LoRA 参数，基础模型冻结'],['硬件','2 张 NVIDIA GeForce RTX 3060 12GB；Windows Gloo DDP'],['原始数据','2390 个建筑旋转视频，每个视频 81 帧'],['正式训练','1912 个 train 建筑 → 9560 个 17 帧窗口；1 epoch，4780 steps'],['最终建议','step-2000 作为本轮交付基线；不是最终扫描补全模型']], [38*mm,132*mm]),Spacer(1,8*mm),P('阅读提示','H2'),P('这份报告把“做了什么、为什么这样做、观察到什么、哪些结论可以相信”分开写清楚。文中 clip 指从一个完整视频截取的时间片段；checkpoint 指某个训练步保存的模型；loss 是训练拟合指标，数值下降不等于视觉质量一定变好。','Body'),PageBreak()]
    s += [P('一、问题定义：本轮到底要解决什么','H1'),P('导师给出的阶段性目标不是立刻完成建筑扫描、缺失区域补全或可量化测绘，而是先让通用视频生成模型学会“建筑通常长什么样”。因此本轮选择方向 B：使用一批建筑旋转视频做领域适配，让模型获得建筑主体、干净背景、居中构图、连续旋转和基本块状几何的先验。','Body'),P('换句话说，本轮要验证的是“模型是否开始把建筑当成一个稳定的视觉对象”，而不是“模型是否已经精确复原某一栋建筑”。这决定了数据、caption、分辨率和评价方式。','Body'),P('总体流程','H2'),pic(GEN/'pipeline.png',170*mm,55*mm),P('图 1  从原始视频到 LoRA 训练的完整数据流。每一步都对应一个需要单独验证的风险点。','Caption'),P('为什么不能把 81 帧整段直接训练？','H2'),P('显存限制是第一原因：两张 12GB 显卡并不会自动合并成一张 24GB 显卡；每个进程仍需在自己的 GPU 上保存模型副本、激活值和梯度。视频扩散模型的显存开销随分辨率和帧数快速增长，81 帧远超本机单卡安全范围。第二个原因是工程稳定性：17 帧窗口能控制显存、缩短单步时间，也能让我们在多个时间段观察建筑旋转。第三个原因是检查点可解释：如果模型在短窗口上已经出现拉伸或漂移，直接扩大到 81 帧只会让问题更难定位。','Body'),PageBreak()]
    s += [P('二、数据集与防泄漏划分','H1'),P('原始目录为 D:\\code\\Python_3D_Scanner\\src\\video_wan81，共 2390 个建筑视频。每个视频有 81 帧，表示同一个建筑从多个角度连续旋转。为了避免“同一建筑的相邻视角同时出现在训练集和测试集”造成虚高结果，划分依据是源 OBJ 身份：同一个 OBJ 的全部视频只能进入一个集合。','Body'),tab(['集合','建筑数','视频/片段数量','用途'],[['train','1912','9560 clips','更新 LoRA 参数'],['validation','239','1195 clips','选择 checkpoint、观察泛化'],['test','239','1195 clips','保留作独立最终测试'],['合计','2390','11950 clips','5 个窗口/建筑']], [30*mm,28*mm,48*mm,64*mm]),P('每个 81 帧视频切为 5 个有重叠的 17 帧窗口：1–17、17–33、33–49、49–65、65–81。5 个窗口共覆盖 85 个位置，因此相邻窗口共享 1 帧。共享帧不是重复错误，而是为了让不同窗口之间的旋转过渡更连续。','Body'),pic(GEN/'slicing.png',170*mm,48*mm),P('图 2  单个视频的切片方式。11950 是全库理论片段数；正式训练只使用 train 的 9560 个片段。','Caption'),P('“11900 左右”这个数字从哪里来？','H2'),P('2390 × 5 = 11950。早期沟通中常把它口头说成“11900 左右”，但精确值是 11950。扣除 validation 和 test 后，真正喂给正式训练程序的是 1912 × 5 = 9560 个 train clips。这个区别非常重要：训练样本数、建筑数量和窗口数量不是同一个概念。','Body'),PageBreak()]
    s += [P('三、实验时间线：每一轮为什么要做','H1'),tab(['阶段','参数/规模','这一轮回答的问题','实际观察'],[['20 视频 smoke','仅跑通下载、LoRA 保存、推理链路','代码能否工作？','通过；不能作为质量结论'],['200 建筑 pilot','256×256，17 帧，rank 8，lr 1e-4，200 steps','学习率和 checkpoint 是否合理？','step-100 视觉最好；step-200 出现细长/退化'],['双卡 smoke 256','2×3060，Windows Gloo DDP','两张卡是否真的同步梯度？','通过；平均 loss 与显存通信正常'],['双卡 smoke 384','384×384，17 帧','更高分辨率是否能稳定运行？','通过；曾观察到 GPU1 92°C，需降温'],['500 balanced','384×384，rank 8，lr 5e-5，250 steps','中等规模下是否重复出现早停现象？','step-100 最平衡；后期几何逐渐塌陷'],['正式 9560','256×256，17 帧，rank 8，lr 5e-5，1 epoch','全量 train 是否能完整跑完？','完成 4780 steps，保存 10 个检查点']], [25*mm,49*mm,49*mm,47*mm]),P('前两轮不是“白跑”：20 视频 smoke 验证了工程闭环；200 pilot 暴露了训练过久会退化；双卡 smoke 证明分布式同步可用；500 balanced 说明早停现象不是偶然；这些证据共同决定了正式训练采用较低学习率并密集保存 checkpoint。','Body'),P('中间结果图片','H2'),pic(VAL/'building_pilot_200'/'comparison_contact_sheet.png',170*mm,70*mm),P('图 3  200 建筑 pilot 的输入/输出对比接触表。它用于判断主体居中、视角变化和几何稳定性。','Caption'),pic(VAL/'building_train_500_balanced_384_cooled'/'checkpoint_contact_sheet.png',170*mm,84*mm),P('图 4  500 建筑 balanced 试验的多 checkpoint 接触表；step-100 后继续训练并没有持续带来视觉收益。','Caption'),P('从两组接触表可以看到一致现象：训练早期主体更容易保持在画面中央，角度变化也更自然；继续训练后，loss 可能仍然下降，但建筑边缘会逐渐拉长、缩放或发生局部形变。因此正式训练必须保存多个中间点，不能只保留最后一个权重。','Body'),tab(['视觉审核项','通过标准','本轮结论'],[['主体位置','建筑主体大部分时间位于画面中央','step-100/2000 最好'],['运动连续性','相邻帧视角变化平滑，没有明显跳变','step-2000 清楚'],['几何一致性','墙体、台阶、塔楼不持续拉伸或塌陷','3000 以后风险上升'],['跨建筑泛化','未见建筑仍保持建筑外观先验','需在 A100 阶段继续扩大验证']], [38*mm,78*mm,42*mm]),PageBreak()]
    s += [P('四、正式训练：缓存、双卡与故障处理','H1'),P('正式阶段先做缓存，再训练。缓存阶段把 T5 文本编码、VAE 图像/视频编码等重复计算结果写入磁盘；训练阶段直接读取缓存，只把重点放在 DiT 的 LoRA 参数更新。这样做可以减少每一步的 CPU/GPU 预处理，避免两个进程同时下载或处理相同资源。','Body'),tab(['配置项','最终值','含义'],[['输入数据','1912 建筑 / 9560 clips','仅 train 集进入优化'],['空间分辨率','256×256','在 12GB 显存上留安全余量'],['时间长度','17 帧','单窗口连续旋转'],['LoRA rank','8','低秩可训练参数容量'],['学习率','5e-5','比 200 pilot 的 1e-4 更保守'],['训练长度','1 epoch / 4780 optimizer steps','每个 train clip 经过一次'],['保存点','500 至 4780，共 10 个','支持早停与回滚'],['分布式','Windows Gloo + file rendezvous','两进程分别绑定 GPU0/GPU1']], [42*mm,48*mm,80*mm]),P('期间处理了四类问题：Windows 环境没有可用 NCCL，因此改用 Gloo；CPU offload 与当前 torch 版本的 empty_cache 行为冲突，因此移除冲突配置；双进程同时下载模型会互相等待，因此改用本地绝对路径并跳过重复下载；同时缓存触发系统提交限制 1455，因此改成顺序缓存并把页面文件放到 D 盘。384 分辨率试验 GPU1 曾达到 92°C，正式阶段每步加入 5 秒冷却，温度回落到可接受范围。','Body'),pic(GEN/'gpu.png',170*mm,76*mm),P('图 5  正式训练 GPU 监控摘要。平均利用率受每步冷却、数据读取和同步等待影响；峰值利用率和显存峰值证明两张卡都实际参与了训练。','Caption'),PageBreak()]
    s += [P('五、训练结果：数值、视觉与硬件证据','H1'),P('loss 结果','H2'),tab(['实验','平均 loss','最后 loss','解释'],[['200 pilot','0.0444','0.0541','最后阶段高于平均值，存在退化'],['双卡 smoke 256','0.0812','0.0555','用于验证同步，不做质量比较'],['双卡 smoke 384','0.0553','0.0566','用于验证显存与分辨率'],['500 balanced','0.0365','0.0262','中期最稳定，后期仍需看视觉'],['正式 9560','0.0410','0.0215','完整跑通，loss 下降明显']], [46*mm,32*mm,32*mm,60*mm]),pic(AS/'loss_curves.png',170*mm,92*mm),P('图 6  各阶段 loss 曲线。正式曲线下降说明 LoRA 确实发生了参数更新；但 checkpoint 选择仍以固定条件下的视频视觉检查为主。','Caption'),P('双卡证据','H2'),tab(['设备','平均利用率','峰值显存','最高温度','峰值功耗'],[['GPU 0',f"{fg['0']['utilization_avg_pct']:.1f}%",f"{fg['0']['memory_max_mib']/1024:.2f} GB",f"{fg['0']['temperature_max_c']:.0f}°C",f"{fg['0']['power_max_w']:.1f} W"],['GPU 1',f"{fg['1']['utilization_avg_pct']:.1f}%",f"{fg['1']['memory_max_mib']/1024:.2f} GB",f"{fg['1']['temperature_max_c']:.0f}°C",f"{fg['1']['power_max_w']:.1f} W"]], [28*mm,36*mm,36*mm,36*mm,34*mm]),P('平均利用率只有约 15%–18% 并不等于没跑：遥测采样包含 5 秒冷却、数据读取和进程同步等待；两张卡都出现接近 100% 的瞬时利用率，并且 GPU1 峰值显存高达 6.54 GB，说明 DDP 确实在工作。','Body'),PageBreak()]
    s += [P('六、checkpoint 选择：为什么交付 step-2000','H1'),P('正式训练共保存 step-500、1000、1500、2000、2500、3000、3500、4000、4500、4780。我们对同一 prompt、negative prompt、seed 和推理步数生成验证视频，再检查三个方面：主体是否居中，17 帧旋转是否连续，建筑几何是否出现拉伸、漂移或局部变形。','Body'),pic(VAL/'building_train_1912_all_256_cooled'/'checkpoint_contact_sheet.png',135*mm,110*mm),P('图 7  正式训练 checkpoint 接触表。横向比较能看到训练进度，纵向比较能看到 17 帧内的运动连续性。','Caption'),tab(['检查点','主体/中心','运动连续性','几何稳定性','本轮判断'],[['base','偏弱','稳定但无领域适配','基线','对照'],['step-500','改善','较小变化','稳定','早期'],['step-1000','较好','偏静态','稳定','可用'],['step-2000','较好','连续旋转最清楚','稳定','推荐'],['step-3000','较好','变化明显','出现拉伸趋势','谨慎'],['step-4000','不稳定','变化大','局部变形','不推荐'],['step-4780','一般','变化仍在','尺度漂移','不推荐']], [29*mm,31*mm,33*mm,36*mm,31*mm]),P('结论：step-2000 是本轮最平衡的交付点。它不是因为 loss 最低而被选中，而是因为在“建筑主体清楚、旋转可见、几何不过度变形”三者之间取得了最好平衡。step-1000 更保守但运动幅度偏小；step-3000 以后继续优化出现过拟合/漂移迹象。','Body'),PageBreak()]
    s += [P('七、最终结论与下一步','H1'),P('本轮实验已经完成了从数据划分、窗口切片、缓存、双卡 DDP、LoRA 参数更新、checkpoint 保存到独立验证的完整闭环。可以确认：模型发生了真实的 LoRA 更新，并开始获得建筑领域的视觉先验；两张 RTX 3060 都实际参与了正式训练；9560 个 train clips 和 4780 个 optimizer steps 已完成。','Body'),P('但必须明确边界：当前结果是“建筑外观领域预研”，不是精确三维重建模型，也不是针对某一栋建筑的记忆模型。由于 caption 同质、分辨率 256×256、时间窗 17 帧，建筑细节、长时序一致性和文字条件控制仍有限。','Body'),P('建议在 A100 80GB 上按以下顺序继续：','H2')]
    for x in ['先保留 OBJ 分组，补充描述高度、对称性、台阶、孔洞、塔楼等真实几何属性；','先做 384×384、33 帧短实验，确认显存和时间一致性，再考虑 81 帧；','从 rank 16、学习率 1e-5～5e-5 开始，保存更密集 checkpoint，避免只看最后一步；','对未见过的 validation 建筑使用多 prompt、多 seed、多视角评价；','通用建筑先验稳定后，再进入 incomplete RGB、missing mask、complete RGB 的严格条件补全训练。']:
        s.append(P('• '+x,'BulletCN'))
    s += [P('可复现实验产物','H2'),tab(['产物','路径'],[['正式配置','D:\\code\\Python_3D_Scanner\\training\\configs\\train_1912_all_256_cooled.json'],['LoRA 权重','D:\\code\\DiffSynth-Studio\\models\\train\\building_train_1912_all_256_cooled_lora'],['验证视频与选择结果','D:\\code\\Python_3D_Scanner\\outputs\\validation\\building_train_1912_all_256_cooled'],['loss 与 GPU 摘要','D:\\code\\Python_3D_Scanner\\outputs\\report_assets'],['完成审计','D:\\code\\Python_3D_Scanner\\outputs\\experiment_completion_audit.json']], [45*mm,125*mm]),P('最终交付建议：使用 step-2000 作为本轮基线，并在 A100 80GB 上以相同验证协议建立下一轮对照。','Body')]
    doc.build(s,onFirstPage=footer,onLaterPages=footer); print(OUT)
if __name__=='__main__': build()
