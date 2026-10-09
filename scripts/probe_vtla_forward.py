"""Isolate standard pi05 flow vs the fast/slow objective at different batches."""
import dataclasses,json,logging,time
from pathlib import Path
import jax
import numpy as np
from flax import nnx
from openpi.training import config as configs,data_loader,sharding
from openpi.models.pi0 import Pi0
import train
logging.basicConfig(level=logging.WARNING,force=True)


def main():
    jax.config.update('jax_compilation_cache_dir',str(Path.home()/'.cache/jax'))
    cfg=configs.get_config('vtla_pi05_base_bottle');mesh=sharding.make_mesh(4)
    ds=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS));rep=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
    batch=next(iter(data_loader.create_data_loader(dataclasses.replace(cfg,num_workers=0),sharding=ds,shuffle=True)))
    state,ss=train.init_train_state(cfg,jax.random.key(7),mesh,resume=False);jax.block_until_ready(state)
    report={'initial_audit':train._audit_device_parameters(state.params),'runs':[]}
    out=Path('artifacts/forward_probe.json');out.write_text(json.dumps(report,indent=2))
    @jax.jit
    def summarize(tree):
        return jax.tree.map(lambda x:(jax.numpy.all(jax.numpy.isfinite(x)),jax.numpy.min(x),jax.numpy.max(x)),tree)
    values=summarize(batch)
    print('ALL_BATCH_INPUTS_FINITE',all(bool(x) for x in jax.tree.leaves(jax.tree.map(lambda x:jax.numpy.all(jax.numpy.isfinite(x)),batch))),flush=True)
    def flow(params,batch):
        model=nnx.merge(state.model_def,params)
        return Pi0.compute_loss_and_metrics(model,jax.random.key(23),*batch,train=False)
    def paired(params,batch):
        model=nnx.merge(state.model_def,params)
        return model.compute_loss_and_metrics(jax.random.key(23),*batch,train=False)
    pflow=jax.jit(flow,in_shardings=(ss.params,ds));ppaired=jax.jit(paired,in_shardings=(ss.params,ds))
    for name,fn,size in [('flow',pflow,4),('flow',pflow,32),('flow',pflow,64),('paired',ppaired,4),('paired',ppaired,64)]:
        selected=jax.device_put(jax.tree.map(lambda x:np.asarray(x)[:size],batch),ds)
        print('START',name,size,flush=True);t=time.perf_counter()
        with sharding.set_mesh(mesh):loss,metrics=fn(state.params,selected)
        jax.block_until_ready((loss,metrics))
        row={'name':name,'batch':size,'seconds':time.perf_counter()-t,'loss_finite':bool(np.isfinite(np.asarray(loss)).all()),'loss':float(np.asarray(loss).mean()),'metrics':{k:float(v) for k,v in metrics.items()},'audit':train._audit_device_parameters(state.params)}
        report['runs'].append(row);out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(row),flush=True)
    print('PROBE_COMPLETE',flush=True)

if __name__=='__main__':main()
