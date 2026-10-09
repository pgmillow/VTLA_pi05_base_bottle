"""Bit-exact device transfer check, independent of model/optimizer/LR."""
import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mib', type=int, default=64)
    parser.add_argument('--host-staged', action='store_true')
    args = parser.parse_args()
    devices = jax.devices('gpu')
    reference = np.arange(args.mib * 1024 * 1024 // 4, dtype=np.uint32)
    reference = reference * np.uint32(1664525) + np.uint32(1013904223)
    mask = np.uint32(0x5A5A5A5A)
    expected = reference ^ mask
    transform = jax.jit(lambda x: jnp.bitwise_xor(x, jnp.uint32(mask)))
    output = Path('artifacts/gpu_transfer_probe_host_staged.json' if args.host_staged else 'artifacts/gpu_transfer_probe.json')
    rows = []
    for source in devices:
        original = jax.device_put(reference.copy(), source)
        jax.block_until_ready(original)
        for destination in devices:
            started = time.perf_counter()
            transferred = jax.device_put(np.asarray(original).copy() if args.host_staged else original, destination)
            # A device kernel consumes the transferred bytes; this avoids
            # accidentally validating a cached copy of the original host data.
            actual = np.asarray(transform(transferred))
            bad = np.flatnonzero(actual != expected)
            row = {'source':str(source),'destination':str(destination),'host_staged':args.host_staged,
                   'bytes':reference.nbytes,'seconds':time.perf_counter()-started,
                   'mismatched_values':int(bad.size),
                   'examples':[{'index':int(i),'expected':int(expected[i]),'actual':int(actual[i])}
                               for i in bad[:8]]}
            rows.append(row)
            output.write_text(json.dumps({'pairs':rows,'scope':'Same-device controls and peer device_put followed by bitwise XOR; no model or optimizer'},indent=2)+'\n')
            print(json.dumps(row),flush=True)
            del actual,transferred
        del original
    if any(row['mismatched_values'] for row in rows):
        raise RuntimeError('Device transfer integrity check failed')
    print('GPU_TRANSFER_PROBE_PASSED',flush=True)


if __name__ == '__main__':
    main()
