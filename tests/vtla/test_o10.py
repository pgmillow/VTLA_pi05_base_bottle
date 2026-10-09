import dataclasses
import importlib.util
import io
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from openpi.vtla.pressure import calibrate_o10, o10_contact_fields


def test_sparse_contact_calibration_and_missing():
    raw=np.zeros((16,268),np.float32);valid=np.ones((16,1),bool)
    calibration=calibrate_o10(raw,valid)
    hist=np.zeros((1,16,1,268),np.float32);hist[0,-1,0,4:8]=20
    out=o10_contact_fields(hist,valid.T,*calibration)
    assert out['tactile_contact']==1 and out['tactile_contact_valid']
    missing=o10_contact_fields(hist,np.zeros((1,16),bool),*calibration)
    assert not missing['tactile_contact_valid'] and not missing['tactile_valid'].any()
    initial=o10_contact_fields(hist,valid.T,*calibration,calibrated=False)
    assert not initial['tactile_contact_valid'] and np.all(initial['tactile_baseline']==0)


def test_causal_pressure_and_repeated_stamps():
    path=str(Path(__file__).parents[2]/'data_pipeline');sys.path.insert(0,path)
    from convert_o10 import causal_pressure, deduplicate
    t,v,audit=deduplicate(np.array([0.,1.,1.,2.]),np.array([[0.],[1.],[2.],[3.]]))
    np.testing.assert_array_equal(t,[0,1,2]);np.testing.assert_array_equal(v[:,0],[0,2,3])
    assert audit=={'removed':1,'max_duplicate_difference':1.}
    raw=np.tile(np.arange(4)[:,None],(1,268))
    pressure,valid,source,age=causal_pressure(np.array([0.,.033,.066,.099]),raw,np.array([.04,.09,.2]),responsive=True)
    np.testing.assert_array_equal(pressure[:,0],[1,2,3]);np.testing.assert_array_equal(valid[:,0],[True,True,False])
    assert np.all(source<=np.array([.04,.09,.2]))


def test_native_frame_cache_matches_parquet(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image
    from openpi.vtla.cache import build_frame_cache,CachedLeRobotDataset
    (tmp_path/'meta').mkdir();(tmp_path/'data/chunk-000').mkdir(parents=True)
    (tmp_path/'meta/info.json').write_text(json.dumps({'tactile_layout':'o10_268','features':{'observation.state':{'shape':[23]},'action':{'shape':[17]}}}))
    (tmp_path/'meta/episodes.jsonl').write_text('\n'.join(json.dumps({'episode_index':e,'length':20}) for e in range(2)))
    (tmp_path/'meta/tasks.jsonl').write_text(json.dumps({'task_index':0,'task':'bottle'}))
    image=io.BytesIO();Image.fromarray(np.zeros((4,4,3),np.uint8)).save(image,format='PNG');encoded=image.getvalue()
    for ep in range(2):
        rows=[]
        for i in range(20):
            raw=np.zeros(268,np.float32);raw[:4]=0 if i<16 else 20
            rows.append({'observation.state':[ep+i/100]*23,'action':[ep+i/10]*17,
                         'timestamp':i/30,'task_index':0,'observation.tactile.right':raw.tolist(),
                         'observation.tactile.right_valid':[True],
                         **{f'observation.images.{c}':{'bytes':encoded} for c in ('top','left_wrist','right_wrist')}})
        pq.write_table(pa.Table.from_pylist(rows),tmp_path/f'data/chunk-000/episode_{ep:06d}.parquet')
    build_frame_cache(tmp_path)
    ds=CachedLeRobotDataset(str(tmp_path),4,tactile_num_frames=16,contact_supervision=True)
    end=ds[19];np.testing.assert_allclose(end['action'],np.full((4,17),1.9))
    assert end['tactile'].shape==(1,16,1,268) and end['tactile_contact']==1
    start=ds[20];assert not start['tactile_contact_valid'] and np.all(start['tactile']==0)
    np.testing.assert_allclose(start['observation.state'],np.ones(23))
    assert ds[35]['tactile_contact']==0
    assert ds[36]['tactile_contact']==1


def test_targeted_replication_preserves_other_sharding():
    import jax
    from openpi.training import sharding
    if jax.device_count()<4:
        pytest.skip('This sharding contract requires four devices')
    mesh=sharding.make_mesh(4)
    tree={'params':{'PaliGemma':{'llm':{'final_norm_1':{'Dense_0':{'kernel':jax.ShapeDtypeStruct((1024,3072),np.float32)}}}},
                    'other':jax.ShapeDtypeStruct((2048,2048),np.float32)}}
    result=sharding.fsdp_sharding(tree,mesh,replicated_paths=("['final_norm_1']['Dense_0']['kernel']",))
    assert result['params']['PaliGemma']['llm']['final_norm_1']['Dense_0']['kernel'].spec==jax.sharding.PartitionSpec()
    assert 'fsdp' in result['params']['other'].spec
