"""Audited native O10 export using the canonical top-camera/continuous strategy."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from o10_mcap import load_mcap, v21, NUMERIC, ARM_NAMES, HAND_NAMES

TASK = 'Unscrew and remove the bottle cap'
CAMERAS = {'top':'top','left_wrist':'left_img','right_wrist':'right_img'}


def deduplicate(t, values):
    if np.any(np.diff(t)<0):
        raise ValueError('Timestamps must be sorted')
    repeated = np.flatnonzero(np.diff(t)==0)
    difference = float(np.abs(values[repeated+1]-values[repeated]).max(initial=0))
    keep = np.r_[np.diff(t)>0,True]
    return t[keep], values[keep], {'removed':int((~keep).sum()),'max_duplicate_difference':difference}


def causal_pressure(t, values, anchors, *, responsive, max_age=.05):
    index = np.searchsorted(t,anchors,side='right')-1
    clipped = np.maximum(index,0)
    age = anchors-t[clipped]
    valid = (index>=0)&(age>=0)&(age<=max_age)&responsive
    # Preserve raw bytes even for invalid samples; downstream masks control use.
    return values[clipped],valid[:,None],t[clipped],age


def convert(path, root, ep, global_start, quality):
    data = load_mcap(path)
    cleaned, audit = {}, {}
    for key in NUMERIC:
        t,v,_ = data[key]
        t,v,dups = deduplicate(t,v)
        if not np.isfinite(v).all():
            raise ValueError(f'Nonfinite numeric values: {key}')
        fixes = 0
        if key in ('rhq','rhc'):
            v,fixes = v21.remove_glitches(t,v,jump_abs=.15,glitch_ms=60.)
        if key == 'ree_r':
            v = np.unwrap(v.astype(np.float64),axis=0)
        cleaned[key] = (t,v)
        audit[key] = {**dups,'glitches_fixed':fixes}
    # Keep the top-camera timeline, trimming only unmatched numeric boundaries.
    top_t,top_frames = data['top']
    lower = max(t[0] for t,v in cleaned.values())
    upper = min(t[-1] for t,v in cleaned.values())
    top_ids = np.flatnonzero((top_t>=lower)&(top_t<=upper))
    anchors = top_t[top_ids]
    if len(anchors)<32 or np.any(np.diff(anchors)<=0):
        raise ValueError('Insufficient or nonmonotonic camera anchors')
    sampled = {k:np.column_stack([np.interp(anchors,t,v[:,j]) for j in range(v.shape[1])]).astype(np.float32)
               for k,(t,v) in cleaned.items()}
    state = np.concatenate([sampled[k] for k in ('arm_q','rhq','ree_p','ree_r')],axis=1)
    action = np.concatenate([sampled[k] for k in ('arm_cmd','rhc')],axis=1)
    tt,tv,_ = data['tactile'];tt,tv,touch_dups = deduplicate(tt,tv)
    responsive = bool(np.any(tv!=0))
    touch,valid,source_t,age = causal_pressure(tt,tv,anchors,responsive=responsive)
    n = len(anchors)
    rows = {'observation.state':state.tolist(),'action':action.tolist(),
            'timestamp':(anchors-anchors[0]).tolist(),'source_timestamp':anchors.tolist(),
            'frame_index':list(range(n)),'episode_index':[ep]*n,'index':list(range(global_start,global_start+n)),
            'task_index':[0]*n,'observation.tactile.right':touch.tolist(),
            'observation.tactile.right_valid':valid.tolist(),
            'tactile_source_timestamp':source_t.tolist(),'tactile_age_s':age.tolist()}
    image_sync = {}
    for camera,key in CAMERAS.items():
        ts,frames = data[key]
        indices = [int(i) for i in top_ids] if key=='top' else [v21.sample_nearest_idx(ts,float(t),float('inf')) for t in anchors]
        if any(i is None for i in indices):
            raise ValueError('Missing camera stream')
        delta = np.abs(ts[indices]-anchors)
        if delta.max()>.05:
            raise ValueError(f'Camera sync exceeds 50ms: {camera}')
        image_sync[camera] = float(delta.max()*1000)
        cache = {}
        def encoded(i):
            if i not in cache:
                cache[i] = v21._jpeg_bytes(v21._letterbox_224(v21._decode_rgb(frames[i])))
            return {'bytes':cache[i],'path':None}
        rows[f'observation.images.{camera}'] = [encoded(i) for i in indices]
    target = root/f'data/chunk-000/episode_{ep:06d}.parquet'
    target.parent.mkdir(parents=True,exist_ok=True)
    temp = target.with_suffix('.partial')
    table=pa.Table.from_pydict(rows)
    for column in ('observation.state','action','observation.tactile.right'):
        position=table.schema.get_field_index(column)
        table=table.set_column(position,column,table[column].cast(pa.list_(pa.float32())))
    pq.write_table(table,temp,compression='zstd')
    temp.rename(target)
    report = {'episode_index':ep,'source':str(path),'length':n,'input_top_frames':len(top_t),
              'boundary_frames_trimmed':len(top_t)-n,'numeric_cleaning':audit,
              'pressure_responsive':responsive,'pressure_valid_fraction':float(valid.mean()),
              'pressure_duplicates':touch_dups,'pressure_max_age_s':.05,
              'image_max_sync_ms':image_sync,'quality_flags':quality['flags'],
              'return_home':quality['return_home'],
              'review_decision':'Retain continuous episode; deduplicate repeated command stamps, last log-order value wins. Persistent command jumps and return-home flagged, not silently removed.',
              'fps':float(1/np.median(np.diff(anchors)))}
    (root/f'meta/convert_reports/episode_{ep:06d}.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--quality',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();root=args.out
    (root/'meta/convert_reports').mkdir(parents=True,exist_ok=True)
    records=json.loads(args.quality.read_text())['records']
    episodes=[];offset=0;excluded=[];eligible=[]
    for quality in records:
        if quality['classification']=='REJECT_REVIEW' or set(quality['reasons'])-{'non_monotonic_numeric_time'}:
            excluded.append(quality)
        else:
            eligible.append(quality)
    # Each process owns one episode and uses temporary local frame indices.
    # Finalization below sets the canonical global indices in episode order.
    with ProcessPoolExecutor(max_workers=12,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[]
        for ep,quality in enumerate(eligible):
            report_path=root/f'meta/convert_reports/episode_{ep:06d}.json'
            if report_path.exists():
                report=json.loads(report_path.read_text())
                if report['source']!=quality['path']:
                    raise ValueError('Existing export has a different source order')
                futures.append(report)
            else:
                futures.append(pool.submit(convert,Path(quality['path']),root,ep,0,quality))
        for ep,future in enumerate(futures):
            report=future if isinstance(future,dict) else future.result()
            target=root/f'data/chunk-000/episode_{ep:06d}.parquet'
            table=pq.read_table(target)
            expected=pa.array(np.arange(offset,offset+report['length']),type=pa.int64())
            table=table.set_column(table.schema.get_field_index('index'),'index',expected)
            for column in ('observation.state','action','observation.tactile.right'):
                table=table.set_column(table.schema.get_field_index(column),column,table[column].cast(pa.list_(pa.float32())))
            temporary=target.with_suffix('.finalizing')
            pq.write_table(table,temporary,compression='zstd');temporary.replace(target)
            episodes.append(report);offset+=report['length']
            print(f'Converted {ep+1}/{len(records)}: {report["length"]} frames, touch responsive={report["pressure_responsive"]}',flush=True)
    state_names=ARM_NAMES+HAND_NAMES+['ee_'+x for x in ('x','y','z','roll','pitch','yaw')]
    features={'observation.state':{'dtype':'float32','shape':[23],'names':state_names},
              'action':{'dtype':'float32','shape':[17],'names':ARM_NAMES+HAND_NAMES},
              'observation.tactile.right':{'dtype':'float32','shape':[268],'names':[f'channel_{i}' for i in range(268)]},
              'observation.tactile.right_valid':{'dtype':'bool','shape':[1],'names':['o10']}}
    for camera in CAMERAS:
        features[f'observation.images.{camera}']={'dtype':'image','shape':[224,224,3],'names':['height','width','channel']}
    info={'codebase_version':'v2.1','robot_type':'right_7dof_o10','fps':30.,'total_episodes':len(episodes),
          'total_frames':offset,'total_tasks':1,'total_chunks':1,'chunks_size':1000,
          'data_path':'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
          'features':features,'tactile_layout':'o10_268','numeric_interp_mode':'continuous',
          'action_semantics':'absolute desired joint positions, radians; filtered arm target + O10 joint_cmd',
          'tactile_semantics':'native 268 counts; causal previous packet <=50ms; all-zero episodes masked',
          'calibration':'first 16 aligned frames assumed no-contact per user instruction'}
    (root/'meta/info.json').write_text(json.dumps(info,indent=2)+'\n')
    (root/'meta/episodes.jsonl').write_text(''.join(json.dumps({'episode_index':e['episode_index'],'length':e['length'],'tasks':[TASK]})+'\n' for e in episodes))
    (root/'meta/tasks.jsonl').write_text(json.dumps({'task_index':0,'task':TASK})+'\n')
    (root/'meta/episode_sources.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in episodes))
    summary={'episodes':len(episodes),'frames':offset,'responsive_episodes':sum(e['pressure_responsive'] for e in episodes),
             'all_zero_pressure_episodes':[e['episode_index'] for e in episodes if not e['pressure_responsive']],
             'duplicates_removed':sum(sum(x['removed'] for x in e['numeric_cleaning'].values()) for e in episodes),
             'excluded':excluded,'raw_modified':False}
    (root/'meta/conversion_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
