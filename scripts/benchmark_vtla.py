"""Bounded real-data optimizer benchmark. No checkpoint or long training run."""
import argparse
import faulthandler
import dataclasses
import functools
import json
import logging
import os
from pathlib import Path
import time
import jax
import numpy as np
from flax import traverse_util
from openpi.training import config as configs,data_loader,sharding
import train
from probe_vtla_gradients import final_norm_shards


def inspect_bad_parameter_shards(params, audit):
    """Confirm GPU summary failures from host copies of the affected shards."""
    flat = traverse_util.flatten_dict(params.to_pure_dict(), sep='/')
    rows = []
    for bad in audit['bad']:
        for shard in flat[bad['name']].addressable_shards:
            array = np.asarray(shard.data)
            invalid = ~np.isfinite(array) | (np.abs(array) > 1e6)
            indices = np.argwhere(invalid)
            rows.append({'name':bad['name'], 'device':str(shard.device),
                         'index':str(shard.index), 'shape':list(array.shape),
                         'max_abs':float(np.max(np.abs(array))),
                         'bad_count':int(invalid.sum()),
                         'examples':[{'index':i.tolist(),'value':float(array[tuple(i)])}
                                     for i in indices[:16]]})
    return rows


def inspect_rejected_adam_shards(state, info, cfg):
    """Inspect existing moments for leaves whose candidate update was nonfinite."""
    names = train._tree_leaf_paths(state.params.filter(cfg.trainable_filter))
    bad = [name for name, ok in zip(names, np.asarray(info['_diag_update_finite']), strict=True) if not ok]
    rows = []
    for path, value in jax.tree_util.tree_flatten_with_path(state.opt_state)[0]:
        name = jax.tree_util.keystr(path)
        if not any(leaf in name for leaf in bad):
            continue
        for shard in value.addressable_shards:
            array = np.asarray(shard.data)
            invalid = ~np.isfinite(array)
            if '.nu' in name:
                invalid |= array < 0
            indices = np.argwhere(invalid)
            rows.append({'path':name, 'device':str(shard.device),
                         'index':str(shard.index), 'min':float(array.min()), 'max':float(array.max()),
                         'invalid_count':int(invalid.sum()),
                         'examples':[{'index':i.tolist(),'value':float(array[tuple(i)])} for i in indices[:8]]})
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['optimized','legacy'],default='optimized');p.add_argument('--steps',type=int,default=3)
    p.add_argument('--config',default='vtla_pi05_base_bottle')
    p.add_argument('--diagnose',action='store_true')
    p.add_argument('--audit-every-step',action='store_true')
    p.add_argument('--accumulation',type=int,default=None)
    p.add_argument('--no-prefix-reuse',action='store_true')
    p.add_argument('--fsdp-devices',type=int,default=None)
    p.add_argument('--seed',type=int,default=None)
    p.add_argument('--label',default='')
    p.add_argument('--stacktraces',action='store_true')
    args=p.parse_args();logging.basicConfig(level=logging.INFO,force=True)
    if args.stacktraces:faulthandler.dump_traceback_later(90,repeat=True)
    print('BENCHMARK_START',flush=True)
    jax.config.update('jax_compilation_cache_dir',str(Path.home()/'.cache/jax'))
    cfg=configs.get_config(args.config)
    if args.mode=='legacy':
        cfg=dataclasses.replace(cfg,model=dataclasses.replace(cfg.model,reuse_training_prefix=False),gradient_accumulation_steps=8,donate_train_buffers=False)
    if args.accumulation is not None:
        if args.accumulation<1 or cfg.batch_size%args.accumulation:
            raise ValueError('Accumulation must be a positive divisor of batch size')
        cfg=dataclasses.replace(cfg,gradient_accumulation_steps=args.accumulation,separate_gradient_update=True)
    if args.no_prefix_reuse:
        cfg=dataclasses.replace(cfg,model=dataclasses.replace(cfg.model,reuse_training_prefix=False))
    if args.fsdp_devices is not None:
        cfg=dataclasses.replace(cfg,fsdp_devices=args.fsdp_devices)
    if args.seed is not None:
        # Keep model.split_seed unchanged: only training RNG/shuffle vary.
        cfg=dataclasses.replace(cfg,seed=args.seed)
    suffix=f'_accum{args.accumulation}' if args.accumulation is not None else ''
    if args.no_prefix_reuse:suffix+='_no_prefix_reuse'
    if args.fsdp_devices is not None:suffix+=f'_fsdp{args.fsdp_devices}'
    if args.audit_every_step:suffix+='_audit'
    if args.seed is not None:suffix+=f'_seed{args.seed}'
    if args.label:
        if not args.label.replace('_','').isalnum():raise ValueError('Label must be alphanumeric or underscore')
        suffix+=f'_{args.label}'
    config_suffix = '' if args.config == 'vtla_pi05_base_bottle' else f'_{args.config}'
    out=Path(__file__).resolve().parents[1]/f'artifacts/benchmark_{args.mode}{config_suffix}{suffix}.json'
    out.unlink(missing_ok=True)
    print('Initializing four-GPU mesh',flush=True)
    mesh=sharding.make_mesh(cfg.fsdp_devices)
    print('Mesh ready; loading dataset',flush=True)
    data_sharding=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
    # This benchmark times device work, not multiprocessing startup. Production
    # retains eight workers; use the identical transforms synchronously here.
    loader=data_loader.create_data_loader(dataclasses.replace(cfg,num_workers=0),sharding=data_sharding,shuffle=True)
    print('Loader ready; obtaining first batch',flush=True)
    iterator=iter(loader);batch=next(iterator);jax.block_until_ready(batch)
    print('Batch ready; initializing pi05 weights',flush=True)
    rng,init_rng=jax.random.split(jax.random.key(cfg.seed))
    rng = sharding.replicate_via_host(rng, replicated)
    state,state_sharding=train.init_train_state(cfg,init_rng,mesh,resume=False);jax.block_until_ready(state)
    audit=train._audit_device_parameters(state.params);adam=train._audit_fresh_adam_state(state.opt_state)
    if audit['bad'] or adam['nonzero_paths']:raise ValueError('Initial state audit failed')
    parameter_count = sum(int(np.prod(x.shape)) for x in jax.tree.leaves(state.params))
    trainable_parameter_count = sum(int(np.prod(x.shape)) for x in jax.tree.leaves(state.params.filter(cfg.trainable_filter)))
    logging.info('BASE WEIGHTS AND FRESH ADAM AUDIT PASSED')
    out.with_name(out.stem+'_initial.json').write_text(json.dumps({'parameter_audit':audit,'adam_audit':adam,'replicated_paths':cfg.fsdp_replicated_paths,'final_norm_shards':final_norm_shards(state.params)},indent=2)+'\n')
    diagnostic_stages=[]
    def inspect_stage(phase,state,result):
        (loss,metrics),grads=result
        row={'phase':phase,'step':int(state.step),'loss':float(loss),'metrics':{k:float(v) for k,v in metrics.items()},
             'parameter_audit':train._audit_device_parameters(state.params),
             'gradient_audit':train._audit_device_parameters(grads),
             'final_norm_shards':final_norm_shards(state.params)}
        row['bad_parameter_host_shards'] = inspect_bad_parameter_shards(state.params, row['parameter_audit'])
        diagnostic_stages.append(row)
        out.with_name(out.stem+'_stages.json').write_text(json.dumps(diagnostic_stages,indent=2)+'\n')
        print('STAGE_DIAGNOSTICS '+json.dumps(row),flush=True)
    if args.mode=='legacy' or cfg.separate_gradient_update:
        step=train.make_accumulated_train_step(cfg,state_sharding,data_sharding,replicated,
                                             first_stage_callback=inspect_stage if args.diagnose or args.audit_every_step else None,
                                             audit_every_step=args.audit_every_step)
    else:
        step=jax.jit(functools.partial(train.train_step,cfg),in_shardings=(replicated,state_sharding,data_sharding),out_shardings=(state_sharding,replicated),donate_argnums=(1,))
    measurements=[]
    for i in range(args.steps+1):
        if args.audit_every_step:
            before={'phase':'before_gradient','step':int(state.step),
                    'parameter_audit':train._audit_device_parameters(state.params)}
            before['bad_parameter_host_shards'] = inspect_bad_parameter_shards(state.params, before['parameter_audit'])
            diagnostic_stages.append(before)
            out.with_name(out.stem+'_stages.json').write_text(json.dumps(diagnostic_stages,indent=2)+'\n')
            print('BEFORE_GRADIENT_AUDIT '+json.dumps(before),flush=True)
            if before['parameter_audit']['bad']:raise FloatingPointError('Corrupted parameter before gradient')
        started=time.perf_counter()
        with sharding.set_mesh(mesh):state,info=step(rng,state,batch)
        jax.block_until_ready((state,info));seconds=time.perf_counter()-started
        numeric_keys=('loss','flow_loss','slow_loss','fast_loss','contact_loss','fast_contact_loss',
                      'contact_gate','fast_gate','contact_accuracy','grad_norm','param_norm','param_max_abs',
                      'update_norm','skipped_update','skip_nonfinite','skip_bad_candidate')
        print('STEP_DIAGNOSTICS '+json.dumps({k:float(info[k]) for k in numeric_keys if k in info}),flush=True)
        if not np.isfinite(float(info['param_norm'])) or float(info['skipped_update'])!=0:
            failure={'info':{k:float(info[k]) for k in numeric_keys if k in info},'parameter_audit':train._audit_device_parameters(state.params)}
            failure['bad_parameter_host_shards'] = inspect_bad_parameter_shards(state.params, failure['parameter_audit'])
            failure['rejected_adam_host_shards'] = inspect_rejected_adam_shards(state, info, cfg)
            out.with_name(out.stem+'_failure.json').write_text(json.dumps(failure,indent=2)+'\n')
        train._check_step_numerics(info)
        metrics={k:float(info[k]) for k in numeric_keys if k in info}
        if metrics['skipped_update']!=0:raise FloatingPointError('Benchmark optimizer rejected update')
        measurements.append({'iteration':i,'seconds':seconds,'includes_compile':i==0,**metrics})
        report={'config':cfg.name,'mode':args.mode,'batch_size':cfg.batch_size,'accumulation':cfg.gradient_accumulation_steps,'devices':str(jax.devices()),
                'seed':cfg.seed,'split_seed':getattr(cfg.model,'split_seed',None),
                'num_train_steps':cfg.num_train_steps,'lr_schedule':dataclasses.asdict(cfg.lr_schedule),
                'optimizer':dataclasses.asdict(cfg.optimizer),'model_config':dataclasses.asdict(cfg.model),
                'parameter_count':parameter_count,'trainable_parameter_count':trainable_parameter_count,
                'model':'pi05, real O10 batch, configured trainable modules, numeric guard on','initial_parameter_audit':audit,
                'fresh_adam_audit':adam,'iterations':measurements,
                'median_steady_seconds':float(np.median([m['seconds'] for m in measurements[1:]])) if i else None,
                'data_workers':0,'production_data_workers':cfg.num_workers,
                'replicated_paths':cfg.fsdp_replicated_paths,'separate_gradient_update':cfg.separate_gradient_update,
                'fsdp_devices':cfg.fsdp_devices,'reuse_training_prefix':getattr(cfg.model,'reuse_training_prefix',False),
                'training_objective':getattr(cfg.model,'training_objective','paired'),
                'initialization':cfg.weight_loader.params_path,'xla_flags':os.environ.get('XLA_FLAGS',''),
                'control_inputs_host_replicated':True,
                'device_transfer_guard':jax.config.jax_transfer_guard_device_to_device,
                'scope':'Bounded disposable optimizer steps; no production checkpoint; separate real batches; timed device work excludes loader wait'}
        out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(measurements[-1]),flush=True)
        if i<args.steps:batch=next(iterator);jax.block_until_ready(batch)
    faulthandler.cancel_dump_traceback_later()
    print(f'BENCHMARK_COMPLETE {out}',flush=True)

if __name__=='__main__':main()
