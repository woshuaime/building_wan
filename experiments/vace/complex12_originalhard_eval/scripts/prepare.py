"""Read the uploaded package, bind existing weights, and validate exact reuse."""
import copy
import shutil
import zipfile
from complex_common import *


def main():
    require(not (RUN/'progress.json').exists() and not list((RUN/'logs').glob('infer_*.done.json')),
            'Preparation already completed; use status/resume')
    packet = PROJECT/'data/buildingfiller_import/Complex12/Complex12_OriginalHard_20261025'
    archive = packet/'BuildingFiller_Complex12_OriginalHard_512_20261025.zip'
    expected = Path(str(archive)+'.sha256').read_text(encoding='utf-8-sig').split()[0].lower()
    actual = sha(archive)
    require(actual == expected, 'Uploaded ZIP does not match its supplied checksum')
    destination = RUN/'input_data'
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            require((destination/item.filename).resolve().is_relative_to(destination.resolve()), 'Unsafe archive member')
        z.extractall(destination)
    data = destination/'BuildingFiller_Complex12_OriginalHard_512_20261025'
    samples = read_json(data/'dataset_index.json')['samples']
    require([s['display_id'] for s in samples] == IDS, 'Expected the original twelve fixed identities')
    original = read_json(SOURCE_RUN/'resolved_config.json')
    old_index = {s['display_id']: s for s in original['samples']}
    c = {k: copy.deepcopy(original[k]) for k in ('model','inference','evaluation','split','legacy_lora')}
    c['training'] = {k: original['training'][k] for k in ('lora_target_modules','lora_rank','lora_alpha','lora_base_model')}
    c['training'].update(enabled=False, max_updates=0, purpose='LoRA loading metadata only; no training code is scheduled')
    c.update(schema='complex12.originalhard.inference.v1', run_root=str(RUN), data_root=str(data),
             source_run=str(SOURCE_RUN), source_config_id=original['config_id'], samples=samples,
             data_version='OriginalHard_before_Rev1', created_at=now(), training_updates_this_run=0,
             package={'path':str(archive),'sha256':actual}, weights={})
    for method in ('L','C','U','S'):
        path = Path(original['legacy_lora']['path']) if method=='L' else SOURCE_RUN/'checkpoints'/method/'step300.safetensors'
        identity = sha(path)
        prior = original['legacy_lora'] if method=='L' else read_json(SOURCE_RUN/'logs'/f'train_{method}.done.json')
        require(identity == prior['sha256' if method=='L' else 'weight_sha256'], f'Changed source weight: {method}')
        c['weights'][method] = {'path':str(path),'sha256':identity,'branch':'vace','scale':1.0,'source_training_updates':300}
    c['config_id'] = hashlib.sha256(json.dumps(c, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    write_json(RUN/'resolved_config.json',c)
    mapping=[];verification=[];reused=0
    for sample in samples:
        cs=sample['display_id'];prior=old_index[cs]
        require((sample['source_id'],sample['family_id'],sample['split']) ==
                (prior['source_id'],prior['family_id'],prior['split']),f'Identity changed: {cs}')
        same={};black=[];outside_gt_difference=[]
        for role in ('gt','damaged','exact','structure_raw'):
            folder=data/sample[role]
            require(sorted(p.name for p in folder.glob('*.png')) == [f'{i:03}.png' for i in range(81)],f'Input count: {cs}/{role}')
            same[role]=all((folder/f'{i:03}.png').read_bytes()==(Path(original['data_root'])/prior[role]/f'{i:03}.png').read_bytes() for i in range(81))
        camera=read_json(data/sample['cameras']);old_camera=read_json(Path(original['data_root'])/prior['cameras'])
        same['camera']=camera==old_camera
        for frame in range(81):
            gt=np.asarray(source(cs,'gt',frame));damaged=np.asarray(source(cs,'damaged',frame))
            exact=np.asarray(source(cs,'exact',frame));raw=np.asarray(source(cs,'structure_raw',frame))
            require(np.isin(exact,[0,255]).all() and np.isin(raw,[0,255]).all(),f'Nonbinary labels: {cs}/{frame}')
            mask=exact>0
            # Separate geometry renders can differ outside Exact (lighting/edges).
            # Input preservation is against Damaged, never a GT composite.
            outside_gt_difference.append(float(np.abs(gt.astype(np.int16)-damaged.astype(np.int16))[~mask].mean()))
            if not mask.any():black.append(frame)
        for role in ('gt','damaged','exact','structure_raw'):
            require(np.array_equal(np.asarray(source(cs,role,0)),np.asarray(source(cs,role,80))),f'Input closure mismatch: {cs}/{role}')
        reusable=all(same.values())
        verification.append({'cs':cs,'same_as_rev1':same,'reusable':reusable,'black_exact_frames':black,'decoded_inputs':324,
                             'max_frame_known_GT_Damaged_mae_255':max(outside_gt_difference)})
        mapping.append({'cs':cs,'source_id':sample['source_id'],'family_id':sample['family_id'],
                        'rev1_role':sample['split'],'variant':'unchanged_control' if reusable else 'OriginalHard_changed_defect',
                        'l_training_overlap':sample['source_id'] in (1227,1848),'data_dir':str(data/sample['sample_dir'])})
        print(f'INPUT {cs} / source{sample["source_id"]}: 81 frames aligned; identical to Rev1={same}; reuse={reusable}',flush=True)
        if reusable:
            for method in c['evaluation']['methods']:
                job=f'infer_{method}_{cs}';prior_record=read_json(SOURCE_RUN/'logs'/f'{job}.done.json')
                require(prior_record['config_id']==original['config_id'],'Reuse source configuration mismatch')
                if method!='B':require(prior_record['lora']['sha256']==c['weights'][method]['sha256'],'Reuse weight mismatch')
                if prior_record['worker_exit_code']!=0:
                    require(Path(prior_record['independent_output_validation']).is_file(),'Prior abnormal exit lacks validation')
                source_folder=SOURCE_RUN/'predictions'/method/cs
                require(sorted(p.name for p in source_folder.glob('*.png'))==[f'{i:03}.png' for i in range(81)],'Incomplete source prediction')
                for frame in range(81):read_image(source_folder/f'{frame:03}.png')
                target=RUN/'predictions'/method/cs;target.parent.mkdir(exist_ok=True);target.symlink_to(source_folder,target_is_directory=True)
                mark_complete(job,method=method,cs=cs,origin='reused',frame_count=81,
                              source_prediction=str(source_folder),source_config_id=original['config_id'],
                              source_worker_exit_code=prior_record['worker_exit_code'],
                              source_validation=prior_record.get('independent_output_validation'),worker_exit_code=None)
                reused+=1
    require([r['cs'] for r in verification if not r['reusable']]==['CS09','CS10','CS11','CS12'],'Unexpected changed sample set; inspect before GPU')
    write_csv(RUN/'sample_mapping.csv',mapping)
    write_json(RUN/'checks/input_reuse_validation.json',verification)
    write_json(RUN/'progress.json',{'config_id':c['config_id'],'status':'prepared','stage':'输入与复用已核对',
               'expected_trajectories':60,'completed_trajectories':reused,'reused_trajectories':reused,
               'inferred_trajectories':0,'training_updates_this_run':0,'updated_at':now()})
    (SCRIPTS.parent/'configs'/f'{RUN.name}.json').symlink_to(RUN/'resolved_config.json')
    script=f'''#!/usr/bin/env bash
set -euo pipefail
export ORIGINALHARD_RUN_ROOT='{RUN}'
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
if [[ "${{1:-}}" == "--run" || "${{1:-}}" == "--resume" ]]; then
  exec '{PYTHON}' -u '{SCRIPTS}/run_eval.py'
fi
exec '{PYTHON}' '{SCRIPTS}/status.py'
'''
    (RUN/'resume.sh').write_text(script);(RUN/'resume.sh').chmod(0o755)
    print(f'Prepared {RUN}; reused {reused}/60; new inference required {60-reused}; training updates=0',flush=True)


if __name__=='__main__':main()
