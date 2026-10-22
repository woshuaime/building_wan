"""One implementation for raw-PNG metrics and synchronized presentation."""
import html
import math
import subprocess
import time
from PIL import ImageDraw, ImageFont
from complex_common import *

METHOD_LABELS={'Before':'Before (Damaged)','GT':'GT (Reference)','B':'B | Base','L':'L | Previous LoRA',
               'C':'C | Ordinary LoRA','U':'U | Uniform weighted','S':'S | Structure weighted'}
ERRORS=['masked_mae','masked_mse','masked_psnr','structure_sobel_l1','known_mae','loop_full_mae','loop_masked_mae']

def safe_value(value):
    if value is None:return 'N/A'
    if math.isinf(value):return 'Infinity' if value>0 else '-Infinity'
    return float(value)

def number(value):
    return None if value=='N/A' or value=='' else float(value)

def psnr(mse):
    if mse is None:return None
    return float('inf') if mse==0 else -10*math.log10(mse)

def pixel_sums(pred,gt,damaged,m,e):
    error=pred-gt
    absolute=np.abs(error).mean(axis=-1)
    squared=np.square(error).mean(axis=-1)
    outside=np.abs(pred-damaged).mean(axis=-1)
    gp=sobel(pred);gg=sobel(gt)
    structure=(np.abs(gp[0]-gg[0])+np.abs(gp[1]-gg[1]))/2
    return {'masked_pixels':int(m.sum()),'structure_pixels':int(e.sum()),'known_pixels':int((~m).sum()),
            'masked_absolute_sum':float(absolute[m].sum()),'masked_squared_sum':float(squared[m].sum()),
            'structure_absolute_sum':float(structure[e].sum()),'known_absolute_sum':float(outside[~m].sum())}

def sobel(rgb):
    gray=rgb @ np.array([0.299,0.587,0.114],dtype=np.float64)
    padded=np.pad(gray,((1,1),(1,1)),mode='reflect')
    a,b,c=padded[:-2,:-2],padded[:-2,1:-1],padded[:-2,2:]
    d,f=padded[1:-1,:-2],padded[1:-1,2:]
    g,h,i=padded[2:,:-2],padded[2:,1:-1],padded[2:,2:]
    return (-a+c-2*d+2*f-g+i)/8,(-a-2*b-c+g+2*h+i)/8

def reduced(sums):
    m,e,k=sums['masked_pixels'],sums['structure_pixels'],sums['known_pixels']
    mse=sums['masked_squared_sum']/m if m else None
    return {'masked_mae':safe_value(sums['masked_absolute_sum']/m if m else None),
            'masked_mse':safe_value(mse),'masked_psnr':safe_value(psnr(mse)),
            'structure_sobel_l1':safe_value(sums['structure_absolute_sum']/e if e else None),
            'known_mae':safe_value(sums['known_absolute_sum']/k if k else None)}

def pixels(cs,method,frame):
    if method in ('Damaged','Before','GT'):
        return np.asarray(source(cs,'gt' if method=='GT' else 'damaged',frame))
    return np.asarray(read_image(RUN/'predictions'/method/cs/f'{frame:03}.png'))

def compute_metrics(methods):
    all_frames=[];buildings=[]
    idx=index()
    for cs in IDS:
        m,e=labels(cs,0,80)
        for method in [*methods,'Damaged']:
            total=None;valid={'masked_valid_frames':0,'structure_valid_frames':0,'known_valid_frames':0}
            for frame in range(80):
                p=pixels(cs,method,frame).astype(np.float64)/255
                gt=pixels(cs,'GT',frame).astype(np.float64)/255
                d=pixels(cs,'Before',frame).astype(np.float64)/255
                sums=pixel_sums(p,gt,d,m[frame],e[frame])
                if total is None:total={k:0 for k in sums}
                for k,v in sums.items():total[k]+=v
                for role in ('masked','structure','known'):valid[role+'_valid_frames']+=int(sums[role+'_pixels']>0)
                all_frames.append({'cs':cs,'source_id':idx[cs]['source_id'],'split':idx[cs]['split'],
                    'method':method,'frame':f'{frame:03}',**reduced(sums),**sums})
            first=pixels(cs,method,0).astype(np.float64)/255
            last=pixels(cs,method,80).astype(np.float64)/255
            loop=np.abs(first-last).mean(axis=-1)
            buildings.append({'cs':cs,'source_id':idx[cs]['source_id'],'split':idx[cs]['split'],'group':idx[cs]['group'],
                'method':method,**reduced(total),'loop_full_mae':float(loop.mean()),
                'loop_masked_mae':safe_value(float(loop[m[0]].mean()) if m[0].any() else None),
                'regular_frames':80,**valid,**total,'loop_full_pixels':512*512,'loop_masked_pixels':int(m[0].sum())})
            print(f'Metrics {cs}/{method}: Mask MAE={buildings[-1]["masked_mae"]}; structure={buildings[-1]["structure_sobel_l1"]}',flush=True)
    write_csv(RUN/'metrics/frame_metrics.csv',all_frames)
    write_csv(RUN/'metrics/building_metrics.csv',buildings)
    write_csv(RUN/'metrics/damaged_baseline.csv',[r for r in buildings if r['method']=='Damaged'])
    for split in ('train','val'):
        summary=[]
        for method in [*methods,'Damaged']:
            selected=[r for r in buildings if r['split']==split and r['method']==method]
            row={'split':split,'method':method,'buildings':len(selected)}
            for metric in ERRORS:
                values=[number(r[metric]) for r in selected if number(r[metric]) is not None]
                row[metric]=safe_value(sum(values)/len(values) if values else None)
                row[metric+'_valid_buildings']=len(values)
            summary.append(row)
        write_csv(RUN/f'metrics/{split}_summary.csv',summary)
        write_csv(RUN/'00_看结果'/('训练集指标对比.csv' if split=='train' else '验证集指标对比.csv'),summary)
    differences=[]
    for cs in IDS:
        rows={r['method']:r for r in buildings if r['cs']==cs}
        for left,right in [('C','B'),('S','C'),('S','U')]:
            row={'cs':cs,'split':idx[cs]['split'],'comparison':f'{left}-{right}'}
            for metric in ERRORS:
                a,b=number(rows[left][metric]),number(rows[right][metric])
                row[metric]=safe_value(a-b if a is not None and b is not None else None)
            differences.append(row)
    write_csv(RUN/'metrics/paired_differences.csv',differences)
    return buildings

def font(size):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)

def tile(cs,method,frame):
    panel=Image.new('RGB',(512,544),'white')
    panel.paste(Image.fromarray(pixels(cs,method,frame)),(0,32))
    ImageDraw.Draw(panel).text((8,5),f'{METHOD_LABELS[method]}   B{frame:03}',fill='black',font=font(19))
    return panel

def panel(cs,frame,methods):
    order=['Before','GT',*methods]
    cols=4 if 'L' in methods else 3
    result=Image.new('RGB',(512*cols,544*2),'white')
    for pos,method in enumerate(order): result.paste(tile(cs,method,frame),((pos%cols)*512,(pos//cols)*544))
    if len(order)<cols*2:
        draw=ImageDraw.Draw(result)
        draw.text(((cols-1)*512+24,544+40),f'{cs}  B{frame:03} / B080',fill='black',font=font(26))
        draw.text(((cols-1)*512+24,544+84),'Complex12 Rev1 | 16 FPS',fill='black',font=font(21))
    return result

def encode_and_check(path,cs,methods):
    cols=4 if 'L' in methods else 3
    size=(cols*512,1088)
    ffmpeg='ffmpeg'
    if not path.exists():
        tmp=path.with_name(path.stem+'.writing.mp4')
        if tmp.exists():tmp.unlink()  # Only our incomplete CPU video; raw PNGs are immutable.
        command=[ffmpeg,'-nostdin','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s',f'{size[0]}x{size[1]}',
            '-r','16','-i','pipe:0','-an','-vf','scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p',
            '-c:v','libx264','-profile:v','high','-level:v','4.2','-crf','12','-preset','fast','-threads','2',
            '-pix_fmt','yuv420p','-color_range','tv','-colorspace','bt709','-color_trc','bt709','-color_primaries','bt709',
            '-movflags','+faststart',str(tmp)]
        p=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            for frame in range(81):p.stdin.write(np.asarray(panel(cs,frame,methods)).tobytes())
            p.stdin.close();p.stdin=None
            _,err=p.communicate(timeout=180)
            require(p.returncode==0,f'Video encoding failed: {err.decode(errors="replace")}')
        finally:
            if p.poll() is None:p.kill();p.wait()
        os.replace(tmp,path)
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
        'stream=width,height,nb_frames,r_frame_rate,pix_fmt','-of','json',str(path)],text=True))['streams'][0]
    require((info['width'],info['height'])==size and int(info['nb_frames'])==81 and info['r_frame_rate']=='16/1','Video metadata mismatch')
    p=subprocess.Popen([ffmpeg,'-nostdin','-v','error','-i',str(path),'-f','rawvideo','-pix_fmt','rgb24','pipe:1'],
                       stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        worst=0
        for frame in range(81):
            data=p.stdout.read(size[0]*size[1]*3)
            require(len(data)==size[0]*size[1]*3,'Video cannot decode all 81 frames')
            decoded=np.frombuffer(data,dtype='uint8').reshape(size[1],size[0],3)
            expected=np.asarray(panel(cs,frame,methods))
            error=float(np.abs(decoded.astype(np.int16)-expected.astype(np.int16)).mean(axis=(0,1)).max())
            worst=max(worst,error)
            require(error<=3,f'Video color/frame alignment failure {cs}/{frame}: {error}')
        require(not p.stdout.read(1),'Video has extra frames')
        err=p.stderr.read();require(p.wait()==0,f'Decoder failed: {err.decode(errors="replace")}')
    finally:
        if p.poll() is None:p.kill();p.wait()
    return {'cs':cs,'video':str(path),'frames':81,'size':list(size),'fps':16,'max_channel_encoding_mae_255':worst}

def overview(cs,methods):
    views=config()['evaluation']['keyframes']
    order=['Before','GT',*methods]
    canvas=Image.new('RGB',(512*len(views),544*len(order)),'white')
    for row,method in enumerate(order):
        for col,frame in enumerate(views):canvas.paste(tile(cs,method,frame),(col*512,row*544))
    path=RUN/'00_看结果'/cs/'02_关键帧总览.png'
    canvas.save(path)
    with Image.open(path) as im:im.load();require(im.size==canvas.size,'Overview unreadable')

def write_entry(methods,status):
    idx=index();entry=RUN/'00_看结果'
    lines=['# Complex12 Rev1 十二栋单轨小试','',f'实际状态：{status}。更新时间：{now()}。','',
        '先打开 [最终实验结论](最终实验结论.md)，或用浏览器打开 [评审图册](评审图册.html)。',
        '指标入口：[训练集8栋](训练集指标对比.csv) / [验证集4栋](验证集指标对比.csv)；验证集是主要泛化观察。','',
        '方法：B=Base；L=旧Pilot8 LoRA参考；C=普通LoRA；U=均匀缺损加权；S=结构带加权。C/U/S固定step300、共同初始化和采样。',
        '同步视频81帧、16FPS、每格原始512×512；2×4顺序Before、GT、B、L / C、U、S、空位。原始预测没有回贴、调色或强制闭环。',
        '关键帧默认B000/B020/B040/B060/B080，各方法使用相同视角、尺寸及比例。常规指标只计000–079；闭环另计000/080。','',
        '| 建筑 | 源编号 | 身份 | 类型 | 结果 |','| --- | --- | --- | --- | --- |']
    cards=[]
    for cs in IDS:
        split='训练' if cs in TRAIN else '验证'
        video=f'{cs}/01_五方法同步对比.mp4';png=f'{cs}/02_关键帧总览.png'
        lines.append(f'| {cs} | {idx[cs]["source_id"]} | {split} | {idx[cs]["visual_morphology"]} | [同步视频]({video}) · [高清总览]({png}) |')
        cards.append(f'<section id="{cs}"><h2>{cs} · {split} · source{idx[cs]["source_id"]} · {html.escape(idx[cs]["visual_morphology"])}</h2>'
            f'<video controls preload="metadata" src="{video}" style="width:100%"></video><p><a href="{png}" target="_blank">打开512原尺寸高清总览</a></p>'
            f'<a href="{png}"><img loading="lazy" src="{png}" style="width:100%"></a></section>')
    lines += ['','指标：MAE/MSE/Sobel/known/loop越低越好，PSNR越高越好；RGB范围[0,1]。楼内按有效像素汇总，跨楼等权。N/A表示区域为空，Infinity表示MSE为0。',
        '逐帧、逐栋、有效计数与C−B/S−C/S−U差值见 [metrics](../metrics/)。模型输出PNG位于 [predictions](../predictions/)。',
        'CS09–CS12只代表复杂屋顶背景下的局部屋面/邻墙补全；跨屋脊挑战主要看CS06。',
        f'未完成/失败说明：{status}。实时事实见 [progress.json](../progress.json)，续跑入口 [resume.sh](../resume.sh)。','']
    (entry/'README_从这里开始.md').write_text('\n'.join(lines),encoding='utf-8')
    document='<!doctype html><html lang="zh"><meta charset="utf-8"><title>Complex12 Rev1 评审图册</title><style>body{max-width:1500px;margin:24px auto;padding:0 20px;font-family:sans-serif;background:#f4f5f7;color:#17202b}section{background:white;padding:20px;margin:30px 0}nav{position:sticky;top:0;background:#fff;padding:12px;z-index:1}nav a{margin-right:16px}a{color:#165bba}</style>'
    document+=f'<h1>Complex12 Rev1 十二栋单轨对比</h1><p>{html.escape(status)}</p><p><a href="最终实验结论.md">中文结论</a> · <a href="训练集指标对比.csv">训练指标</a> · <a href="验证集指标对比.csv">验证指标</a></p>'
    document+='<nav>'+''.join(f'<a href="#{cs}">{cs}</a>' for cs in IDS)+'</nav>'+''.join(cards)+'</html>'
    (entry/'评审图册.html').write_text(document,encoding='utf-8')

def main():
    begin=time.monotonic();methods=config()['evaluation']['methods']
    for method in methods:
        for cs in IDS:require(valid_prediction(method,cs),'Missing raw trajectory')
    compute_metrics(methods)
    checks=[]
    for cs in IDS:
        folder=RUN/'00_看结果'/cs;folder.mkdir(exist_ok=True)
        print(f'Creating synchronized comparison video and overview: {cs}',flush=True)
        checks.append(encode_and_check(folder/'01_五方法同步对比.mp4',cs,methods))
        overview(cs,methods)
    write_json(RUN/'checks/delivery_media.json',checks)
    write_entry(methods,'60条原始轨迹、12条同步视频、12张总览及统一指标已生成；正在进行完整轨迹视觉复核和中文结论整理')
    mark_complete('evaluate',trajectories=60,png_frames=4860,videos=12,overviews=12,elapsed_seconds=time.monotonic()-begin)

if __name__=='__main__':main()
