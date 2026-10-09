"""Build the mmap cache, train-only normalization, audit, and review plots."""
import json
from pathlib import Path
import time
import numpy as np
import pyarrow.parquet as pq
from openpi.shared.normalize import NormStats,save
from openpi.vtla.cache import build_frame_cache,CachedLeRobotDataset
from openpi.vtla.pressure import o10_contact_fields

ROOT=Path('/data/open_bottle_cap_vtla_20261006/lerobot')
PROJECT=Path(__file__).resolve().parents[1]


def main():
    build_frame_cache(ROOT)
    ds=CachedLeRobotDataset(str(ROOT),30,tactile_num_frames=16,contact_supervision=True)
    ds._open();a=ds._arrays
    order=np.random.default_rng(42).permutation(len(ds._episodes));count=max(1,int(np.ceil(len(order)*.05)))
    validation=sorted(order[:count].tolist());train=sorted(order[count:].tolist())
    state_samples=[];action_samples=[];records=[];positives=negatives=unknown=0
    for ep,end in enumerate(ds._episode_ends):
        start=0 if ep==0 else ds._episode_ends[ep-1];n=end-start
        state=np.asarray(a['state'][start:end]);action=np.asarray(a['actions'][start:end]);ts=np.asarray(a['timestamp'][start:end])
        table=pq.read_table(ROOT/f'data/chunk-000/episode_{ep:06d}.parquet',columns=['index','frame_index','episode_index','source_timestamp','tactile_source_timestamp','tactile_age_s','observation.tactile.right_valid'])
        np.testing.assert_array_equal(np.asarray(table['index']),np.arange(start,end))
        np.testing.assert_array_equal(np.asarray(table['frame_index']),np.arange(n))
        np.testing.assert_array_equal(np.asarray(table['episode_index']),np.full(n,ep))
        anchors=np.asarray(table['source_timestamp']);source=np.asarray(table['tactile_source_timestamp'])
        packet_valid=np.asarray(table['observation.tactile.right_valid'].to_pylist()).reshape(-1)
        assert np.all(source[packet_valid]<=anchors[packet_valid])
        assert np.all((anchors-source)[packet_valid]<=.05)
        assert state.shape==(n,23) and action.shape==(n,17)
        assert np.isfinite(state).all() and np.isfinite(action).all() and np.all(np.diff(ts)>0)
        if ep in train:
            state_samples.append(state[15:])
            ix=np.minimum(np.arange(15,n)[:,None]+np.arange(30),n-1).reshape(-1)
            action_samples.append(action[ix])
        touch=np.asarray(a['tactile_right'][start:end]);valid=np.asarray(a['valid_right'][start:end])[:,0]
        baseline=a['baseline_right'][ep];scale=a['scale_right'][ep];usable=a['baseline_valid_right'][ep]
        deviations=np.abs((touch-baseline[None])/scale[None,:,None,None]).reshape(n,268)
        score=np.partition(deviations,-4,axis=-1)[:,-4:].mean(-1)
        known=valid&bool(usable[0])&(np.arange(n)>=15)
        positive=known&(score>=3);negative=known&(score<=1)
        positives+=int(positive.sum());negatives+=int(negative.sum());unknown+=int((~(positive|negative)).sum())
        # Check the vectorized audit against the real dataset path at boundaries.
        for local in (0,15,n-1):
            sample=ds[start+local]
            assert bool(sample['tactile_contact_valid'])==bool((positive|negative)[local])
            assert bool(sample['tactile_contact'])==bool(positive[local])
            np.testing.assert_array_equal(sample['action'][0],action[local])
            assert all(sample[f'observation.images.{c}'].shape==(224,224,3) for c in ('top','left_wrist','right_wrist'))
        records.append({'episode':ep,'frames':n,'split':'validation' if ep in validation else 'train',
                        'pressure_valid_fraction':float(valid.mean()),'contact_positive':int(positive.sum()),
                        'contact_negative':int(negative.sum()),'contact_unknown':int((~(positive|negative)).sum()),
                        'max_arm_command_jump_rad':float(np.abs(np.diff(action[:,:7],axis=0)).max()),
                        'max_hand_command_jump_rad':float(np.abs(np.diff(action[:,7:],axis=0)).max()),
                        'max_ee_position_jump_m':float(np.linalg.norm(np.diff(state[:,17:20],axis=0),axis=1).max()),
                        'fps':float(1/np.median(np.diff(ts)))})
    stats={}
    for name,parts in [('state',state_samples),('actions',action_samples)]:
        x=np.concatenate(parts).astype(np.float64)
        stats[name]=NormStats(mean=x.mean(0),std=x.std(0),q01=np.quantile(x,.01,axis=0),q99=np.quantile(x,.99,axis=0))
    save(PROJECT/'assets/bottle/open_bottle_cap',stats)
    report={'episodes':len(records),'frames':len(ds),'train_episodes':train,'validation_episodes':validation,
            'split_seed':42,'validation_fraction':.05,'normalization':'train-only; exact q01/q99; 30-step action chunks with terminal padding',
            'contact_label':'mean of four largest baseline-corrected channel deviations / episode noise; <=1 negative, >=3 positive',
            'contact_positive':positives,'contact_negative':negatives,'contact_unknown':unknown,'records':records}
    # Timing isolates frame/image loading from the model and training transforms.
    ids=np.random.default_rng(7).integers(0,len(ds),size=128);t=time.perf_counter()
    for index in ids:ds[int(index)]
    report['cached_random_sample_ms']=1000*(time.perf_counter()-t)/len(ids)
    if positives==0:raise ValueError('No positive tactile supervision after calibration')
    (PROJECT/'artifacts/data_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    plots(ds,report)
    ready={'dataset':str(ROOT),'episodes':len(records),'frames':len(ds),'audit':'artifacts/data_audit.json',
           'action_schema':'17 absolute joint targets','state_schema':'23 = arm7 + hand10 + measured EE6',
           'raw_modified':False,'prepared':True,'full_training_started':False}
    (PROJECT/'artifacts/data_ready.json').write_text(json.dumps(ready,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2),flush=True)


def plots(ds,report):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a=ds._arrays
    chosen=next(r['episode'] for r in report['records'] if r['contact_positive']>0)
    for ep in (0,chosen):
        start=0 if ep==0 else ds._episode_ends[ep-1];end=ds._episode_ends[ep];sl=slice(start,end)
        t=a['timestamp'][sl];state=a['state'][sl];actions=a['actions'][sl]
        fig,axs=plt.subplots(4,1,figsize=(13,10),sharex=True)
        axs[0].plot(t,actions[:,:7]);axs[0].set_ylabel('Arm targets (rad)')
        axs[1].plot(t,actions[:,7:]);axs[1].set_ylabel('O10 targets (rad)')
        axs[2].plot(t,state[:,17:20]);axs[2].set_ylabel('Measured EE (m)')
        axs[3].plot(t,a['tactile_right'][sl].reshape(-1,268));axs[3].set_ylabel('Raw pressure counts');axs[3].set_xlabel('Time (s)')
        fig.suptitle(f'O10 episode {ep} | all channels retained');fig.tight_layout();fig.savefig(PROJECT/f'artifacts/episode_{ep:03d}.png',dpi=140);plt.close(fig)
        sample=ds[start+len(t)//2]
        from PIL import Image
        Image.fromarray(np.concatenate([sample[f'observation.images.{c}'] for c in ('top','left_wrist','right_wrist')],axis=1)).save(PROJECT/f'artifacts/cameras_{ep:03d}.jpg')

if __name__=='__main__':main()
