#!/usr/bin/env python3
"""Post-freeze diagnostic figures; never modifies generated XYZ/RGB or selection."""
from collections import Counter
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import ROOT, CASES, D_RUN, SAMPLES, now, sha, write
from geometry import read_cloud
from render_evaluate import decode, render, camera_list, FONT


def plot_panel(draw, rect, observed, hidden, proposals, final, xlabel, ylabel):
    x,y,w,h=rect
    combined=np.vstack([observed,hidden,proposals])
    lo,hi=combined.min(0),combined.max(0)
    pad=np.maximum((hi-lo)*.06,.15);lo-=pad;hi+=pad
    sx,sy=(w-75)/(hi[0]-lo[0]),(h-65)/(hi[1]-lo[1])
    def pixels(a):return (a-lo)*np.array([sx,-sy])+[x+55,y+h-35]
    draw.rectangle((x+55,y+25,x+w-20,y+h-35),outline='#888888')
    for vals,color,radius in ((observed,'#a0a0a0',2),(hidden,'#0088cc',4),(proposals,'#ffb000',3),(final,'#cc2233',4)):
        for a,b in pixels(vals):draw.ellipse((a-radius,b-radius,a+radius,b+radius),fill=color)
    font=ImageFont.truetype(FONT,15)
    draw.text((x+60,y+5),ylabel,fill='black',font=font)
    draw.text((x+70,y+h-25),f'{xlabel}: {lo[0]:.2f} .. {hi[0]:.2f} m',fill='black',font=font)
    draw.text((x+3,y+30),f'{hi[1]:.1f}',fill='black',font=font)
    draw.text((x+3,y+h-50),f'{lo[1]:.1f}',fill='black',font=font)


def review_case(display,name,run):
    out=run/'cases'/name;case=CASES/name
    generation=json.loads((out/'generation.json').read_text())
    result=json.loads((out/'result.json').read_text())
    obs=np.load(out/'observed_input.npz');cand=np.load(out/'candidates.npz');merged=np.load(out/'merged_points.npz')
    mask=decode(case/'inputs/vace_video_mask.mp4')[:,:,:,0]>127
    areas=mask.sum(axis=(1,2));frames=[start+int(np.argmax(areas[start:end])) for start,end in ((0,20),(20,40),(40,60),(60,81))]
    inputs=decode(case/'inputs/vace_video.mp4');d=decode(D_RUN/'cases'/name/'raw_vace.mp4');target=decode(case/'reference/observed_target.mp4')
    fixed=np.stack([np.asarray(Image.open(out/'raw_frames'/f'{i:03d}.png')) for i in range(81)])
    sheet=Image.new('RGB',(1280,4*348),'white');draw=ImageDraw.Draw(sheet);font=ImageFont.truetype(FONT,18)
    for row,i in enumerate(frames):
        yy,xx=np.nonzero(mask[i]);cx=int((xx.min()+xx.max())/2) if len(xx) else 416;cy=int((yy.min()+yy.max())/2) if len(yy) else 240
        cx=max(160,min(832-160,cx));cy=max(160,min(480-160,cy));box=(cx-160,cy-160,cx+160,cy+160)
        for col,(label,video) in enumerate(zip(('Input (gray edit)','D-3000','Fixed 3D','Measured reference'),(inputs,d,fixed,target))):
            sheet.paste(Image.fromarray(video[i]).crop(box),(col*320,row*348+28));draw.text((col*320+4,row*348+4),f'{i:03d} {label}',font=font,fill='black')
    sheet.save(out/'four_column_native_crops.png')
    # Color-coded point IDs are only a diagnostic, not the delivered prediction.
    palette=np.full_like(merged['colors'],150)
    metadata=generation['proposal_metadata']
    for offset,index in enumerate(cand['kept_indices']):
        palette[len(obs['points'])+offset]=[220,55,55] if metadata[index]['kind']=='roof' else [30,90,235]
    views=Image.new('RGB',(1664,960),'white')
    for j,i in enumerate(frames):
        views.paste(Image.fromarray(render(merged['points'],palette,camera_list(case)[i])),(j%2*832,j//2*480))
    views.save(out/'surface_labels_roof_red_wall_blue.png')
    colors=cand['all_colors'][cand['kept_indices']]
    foreground=(np.max(abs(colors.astype(int)-[237,240,244]),axis=1)>30) if len(colors) else np.array([],bool)
    diagnostic=dict(time=now(),key_crop_frames=frames,retained_by_surface=Counter(metadata[i]['kind'] for i in cand['kept_indices']),
        rejection_counts=Counter(r['reason'] for r in generation['rejected']),
        retained_RGB_unique=np.unique(colors,axis=0).tolist(),retained_foreground_color_points=int(foreground.sum()),
        color_threshold_note='evaluate.py foreground = max(abs(RGB-[237,240,244])) > 30',
        reference_coords_used_only_after_all_generations_frozen=True,files_sha256={})
    walls=[p for p in generation['surfaces'] if p['kind']=='wall' and 'basis' in p]
    if walls:
        wall=max(walls,key=lambda p:p['local_support'])
        origin=np.array(wall['origin']);basis=np.array(wall['basis']);normal=np.array(wall['normal']);axes=np.vstack([basis,normal])
        p=obs['points'];use=(abs((p-obs['box_center'])@obs['box_axes'].T)<=obs['box_half']+4).all(1)
        record=json.loads((case/'case.json').read_text());identity=json.loads((out/'input_identity.json').read_text());full,_=read_cloud(Path(identity['source']));hidden=full[np.array(record['hidden_original_point_ids'],int)]-obs['world_origin']
        def project(v):return (v-origin)@axes.T
        observed=project(p[use]);truth=project(hidden);allp=project(cand['all_points']);kept=project(cand['all_points'][cand['kept_indices']])
        image=Image.new('RGB',(1500,650),'white');draw=ImageDraw.Draw(image)
        draw.text((20,12),f'{display} {name}  OBS gray / hidden measurement blue (evaluation only) / pool gold / kept red',font=font,fill='black')
        for j,(a,b,labelx,labely) in enumerate(((0,1,'Observed facade tangent','Height on fitted vertical plane (m)'),(0,2,'Observed facade tangent','Normal offset from fitted plane (m)'),(1,2,'Height','Normal offset from fitted plane (m)'))):
            plot_panel(draw,(j*500,45,500,590),observed[:,[a,b]],truth[:,[a,b]],allp[:,[a,b]],kept[:,[a,b]],labelx,labely)
        image.save(out/'post_freeze_surface_coverage.png')
        diagnostic['diagnostic_plane_id']=wall['plane_id']
        diagnostic['inferred_wall_normals']=[dict(plane_id=p['plane_id'],normal=p['normal'],support=p['support'],
            local_support=p['local_support'],absolute_distance_to_ROI_center_m=float(abs((obs['box_center']-np.array(p['origin']))@np.array(p['normal']))),
            base_spacing_m=p['base_spacing'],proposed=p['proposed']) for p in walls if 'base_spacing' in p]
    for file in out.glob('*.mp4'):
        info=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate,nb_read_frames','-of','json',str(file)],text=True))
        stream=info['streams'][0]
        assert stream['nb_read_frames']=='81' and stream['r_frame_rate']=='16/1'
        subprocess.run(['ffmpeg','-nostdin','-v','error','-xerror','-threads','1','-i',str(file),'-f','null','-'],check=True)
        diagnostic.setdefault('validated_videos',[]).append(dict(path=str(file),sha256=sha(file),**stream))
    for file in out.glob('*.png'):diagnostic['files_sha256'][file.name]=sha(file)
    write(out/'post_freeze_diagnostics.json',diagnostic)
    return diagnostic


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);run=parser.parse_args().run.resolve()
    assert (run/'evaluation_reference_access.json').exists()
    rows={name:review_case(display,name,run) for display,name in SAMPLES}
    write(run/'post_freeze_diagnostics.json',dict(time=now(),cases=rows,review_source_sha256=sha(Path(__file__))))


if __name__=='__main__':main()
