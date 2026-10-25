"""Internal contact sheets for inspecting every frame, not final presentation."""
from PIL import ImageDraw
from complex_evaluate import *

def make_views(cs, methods):
    output = RUN / 'review' / cs
    output.mkdir(parents=True, exist_ok=True)
    path = output / f'keyframes_{"".join(methods)}.png'
    if path.exists():
        return path
    views = config()['evaluation']['keyframes']
    order = ['Before', 'GT', *methods]
    canvas = Image.new('RGB', (512 * len(views), 544 * len(order)), 'white')
    for row, method in enumerate(order):
        for col, frame in enumerate(views):
            canvas.paste(tile(cs, method, frame), (col * 512, row * 544))
    canvas.save(path)
    print(path, flush=True)
    return path

def make_sheets(cs,methods):
    # All methods use the same fixed full-frame reduction; no selected best results.
    output=RUN/'review'/cs
    output.mkdir(parents=True,exist_ok=True)
    order=['Before','GT',*methods]
    for first in range(0,81,9):
        path=output/f'all_frames_{"".join(methods)}_{first:03}_{first+8:03}.png'
        if path.exists():continue
        canvas=Image.new('RGB',(9*192,len(order)*214),'white')
        draw=ImageDraw.Draw(canvas)
        for row,method in enumerate(order):
            for col,frame in enumerate(range(first,first+9)):
                image=Image.fromarray(pixels(cs,method,frame)).resize((192,192),Image.Resampling.LANCZOS)
                canvas.paste(image,(col*192,row*214+22))
                draw.text((col*192+4,row*214+4),f'{method} B{frame:03}',fill='black',font=font(14))
        canvas.save(path)
    print(output,flush=True)

if __name__=='__main__':
    for cs in sys.argv[1:]:make_sheets(cs,config()['evaluation']['methods'])
