"""Audit actual batches and augmentation RNGs without loading pi05 weights."""
import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from openpi.training import config, data_loader


def main():
    cfg = config.get_config('vtla_bottle_stage1_visual')
    loader = data_loader.create_data_loader(dataclasses.replace(cfg, num_workers=0), shuffle=True)
    rng, _ = jax.random.split(jax.random.key(cfg.seed))
    aug = {f.name: getattr(cfg.model, f.name) for f in dataclasses.fields(cfg.model)
           if f.name.startswith('image_aug_')}

    def check(key, obs, actions):
        pre, noise, time = jax.random.split(key, 3)
        prepared = model_lib.preprocess_observation(pre, obs, train=True, **aug)
        def stats(x):
            x = x.astype(jnp.float32)
            return jnp.array([jnp.all(jnp.isfinite(x)), jnp.min(x), jnp.max(x)])
        report = {'image/'+k: stats(v) for k,v in prepared.images.items()}
        report['time'] = stats(jax.random.beta(time, 1.5, 1, (actions.shape[0],)) * .999 + .001)
        report['noise'] = stats(jax.random.normal(noise, actions.shape))
        report['state'] = stats(obs.state)
        report['actions'] = stats(actions)
        return report

    fn = jax.jit(check)
    rows=[]
    for i,(obs,actions) in enumerate(loader):
        raw_bad=[jax.tree_util.keystr(p) for p,x in jax.tree_util.tree_flatten_with_path((obs,actions))[0]
                 if not np.isfinite(np.asarray(x)).all()]
        report = jax.device_get(fn(jax.random.fold_in(rng,i),obs,actions))
        row={'step':i,'raw_bad':raw_bad,'stats':{k:np.asarray(v).tolist() for k,v in report.items()}}
        row['first_frame']=int(np.asarray(loader.last_batch_meta['sample_global_index'])[0])
        rows.append(row);print(json.dumps(row),flush=True)
        Path('artifacts/staged_input_probe.json').write_text(json.dumps(rows,indent=2)+'\n')
        if i == 15: break

if __name__=='__main__':main()
