"""Validate every exported index, dimension and causal pressure timestamp."""
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
root=Path('/data/open_bottle_cap_vtla_20261006/lerobot')
meta=[json.loads(x) for x in (root/'meta/episodes.jsonl').read_text().splitlines()]
index=0;rows=0;maximum_duplicate_difference=0.;glitches=0
for episode in meta:
    ep=episode['episode_index'];n=episode['length']
    table=pq.read_table(root/f'data/chunk-000/episode_{ep:06d}.parquet',columns=['index','frame_index','episode_index','source_timestamp','tactile_source_timestamp','tactile_age_s','observation.tactile.right_valid','observation.state','action'])
    np.testing.assert_array_equal(np.asarray(table['index']),np.arange(index,index+n))
    np.testing.assert_array_equal(np.asarray(table['frame_index']),np.arange(n))
    np.testing.assert_array_equal(np.asarray(table['episode_index']),np.full(n,ep))
    assert table['observation.state'].type.value_type.bit_width==32
    assert table['action'].type.value_type.bit_width==32
    anchors=np.asarray(table['source_timestamp']);source=np.asarray(table['tactile_source_timestamp'])
    valid=np.asarray(table['observation.tactile.right_valid'].to_pylist()).reshape(-1)
    assert np.all(np.diff(anchors)>0)
    assert np.all(source[valid]<=anchors[valid]) and np.all((anchors-source)[valid]<=.05)
    for column,width in [('observation.state',23),('action',17)]:
        value=np.asarray(table[column].to_pylist());assert value.shape==(n,width) and np.isfinite(value).all()
    report=json.loads((root/f'meta/convert_reports/episode_{ep:06d}.json').read_text())
    maximum_duplicate_difference=max(maximum_duplicate_difference,max(x['max_duplicate_difference'] for x in report['numeric_cleaning'].values()))
    glitches+=sum(x['glitches_fixed'] for x in report['numeric_cleaning'].values())
    rows+=n;index+=n
result={'passed':True,'episodes':len(meta),'frames':rows,'max_duplicate_command_difference_rad':maximum_duplicate_difference,'hand_glitches_fixed':glitches,'checks':['global and local indices','float32 state23/action17','numeric finite','monotonic anchors','no future pressure for valid frames','pressure max age50ms']}
(Path(__file__).resolve().parents[1]/'artifacts/export_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
