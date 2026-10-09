#!/usr/bin/env bash
set -euo pipefail
export FASTSLOW_CONFIG="${FASTSLOW_CONFIG:-pi05_righthand_ee_fastslow_46k_b64}"
source "$(dirname -- "${BASH_SOURCE[0]}")/fastslow_env.sh"
JAX_PLATFORMS=cpu "$PYTHON_BIN" - "$FASTSLOW_CONFIG" "$@" <<'PY'
import dataclasses, datetime, hashlib, json
from pathlib import Path
from openpi.training import config
c = config.cli()
root = Path.cwd()
backbone_step = Path(c.weight_loader.backbone_path).parent.name
tactile_step = Path(c.weight_loader.tactile_path).parent.name
audit = json.loads((root / f'artifacts/initialization_{backbone_step}_touch{tactile_step}_audit.json').read_text())
if audit['bad'] or not audit['normalization_equal']:
    raise RuntimeError('Initialization audit did not pass')
if audit['weight_loader'] != dataclasses.asdict(c.weight_loader):
    raise RuntimeError('Audit does not match configured source checkpoints')
record = {'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'config':dataclasses.asdict(c), 'initialization_audit':audit,
          'effective_batch_size':c.batch_size,
          'microbatch_size':c.batch_size//c.gradient_accumulation_steps,
          'sha256':{}}
for name in ['scripts/train.py','src/openpi/training/config.py','src/openpi/training/weight_loaders.py',
             'src/openpi/fastslow/model.py','src/openpi/fastslow/data.py']:
    record['sha256'][name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
(root/'logs').mkdir(exist_ok=True)
stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
out = root/'logs'/f'{c.exp_name}.{stamp}.config.json'
out.write_text(json.dumps(record,indent=2,default=str)+'\n')
print('Configuration snapshot:',out,flush=True)
PY
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTHONUNBUFFERED=1
exec "$PYTHON_BIN" scripts/train.py "$FASTSLOW_CONFIG" "$@"
