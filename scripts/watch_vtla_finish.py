"""Wait for the requested committed checkpoint, stop its owner, then resume 5k."""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from train_vtla_finish import checkpoint_counts, checkpoint_ready


def process_identity(pid):
    try:
        proc = Path(f"/proc/{pid}")
        stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        return {"start_ticks": stat[19], "cmdline": (proc / "cmdline").read_text().replace("\0", " ").strip(),
                "cwd": str((proc / "cwd").resolve())}
    except (OSError, ValueError, IndexError):
        return None


def matching_owner(actual, expected):
    return actual is not None and actual == expected


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-pid", type=int, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--additional-steps", type=int, default=5000)
    parser.add_argument("--end-lr", type=float, default=1e-5)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    checkpoint_root = Path("/data/checkpoints_vtla_bottle/vtla_bottle_direct_full/base_touch_joint_46k_seed43")
    target = checkpoint_root / str(args.checkpoint_step)
    state_path = root / "artifacts/finish_5k_status.json"
    lock = (root / "logs/finish_5k.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if state_path.exists():
        raise RuntimeError("A continuation status already exists; inspect it before any rerun")
    identity = process_identity(args.original_pid)
    if identity is None or identity["cwd"] != str(root) or not identity["cmdline"].endswith(
        "scripts/train.py vtla_bottle_direct_full"
    ):
        raise RuntimeError(f"Refusing to control unexpected process: {identity}")
    state = {"controller_pid": os.getpid(), "original_pid": args.original_pid,
             "original_identity": identity, "source_checkpoint": args.checkpoint_step,
             "checkpoint_dir": str(checkpoint_root), "additional_steps": args.additional_steps,
             "end_lr": args.end_lr, "final_checkpoint": args.checkpoint_step + args.additional_steps}

    def status(phase, **fields):
        state.update(fields)
        state.update(phase=phase, updated_at=datetime.datetime.now().astimezone().isoformat())
        tmp = state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2) + "\n")
        tmp.replace(state_path)
        print(json.dumps(state), flush=True)

    try:
        paths = ["scripts/train.py", "scripts/train_vtla_finish.py", "scripts/watch_vtla_finish.py"]
        paths += [str(p.relative_to(root)) for p in (root / "src/openpi").rglob("*.py")]
        hashes = {p: sha256(root / p) for p in paths}
        (root / "artifacts/finish_5k_source_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
        status("waiting_for_checkpoint")
        deadline = time.monotonic() + 24 * 3600
        while not checkpoint_ready(target):
            if not matching_owner(process_identity(args.original_pid), identity):
                raise RuntimeError("Original training exited before the requested checkpoint committed")
            if time.monotonic() > deadline:
                raise TimeoutError("Requested checkpoint did not commit within 24 hours")
            time.sleep(5)
        counts = checkpoint_counts(target)
        expected = args.checkpoint_step + 1
        if set(counts.values()) != {expected}:
            raise RuntimeError(f"Checkpoint counters disagree; original training left running: {counts}")
        changed = [p for p, digest in hashes.items() if sha256(root / p) != digest]
        if changed:
            raise RuntimeError(f"Source changed after arming; original training left running: {changed}")
        if not matching_owner(process_identity(args.original_pid), identity):
            raise RuntimeError("Original process changed; no signal sent")
        status("checkpoint_verified", source_counts=counts)
        # Signal only the exact training PID after the checkpoint is committed.
        os.kill(args.original_pid, signal.SIGINT)
        status("stopping_original")
        for _ in range(90):
            if not matching_owner(process_identity(args.original_pid), identity):
                break
            time.sleep(2)
        else:
            raise TimeoutError("Original training did not exit after SIGINT; refusing a concurrent restart")

        # Wait for CUDA to release this process; other users' processes are untouched.
        for _ in range(60):
            output = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                text=True, timeout=15)
            if str(args.original_pid) not in output.split():
                break
            time.sleep(2)
        else:
            raise TimeoutError("Original GPU process remains allocated")

        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES="0,1,2,3", OMP_NUM_THREADS="4", NCCL_P2P_DISABLE="1",
                   NCCL_IB_DISABLE="1", XLA_PYTHON_CLIENT_MEM_FRACTION="0.95",
                   XLA_PYTHON_CLIENT_PREALLOCATE="true", JAX_TRANSFER_GUARD_DEVICE_TO_DEVICE="disallow",
                   PYTHONUNBUFFERED="1", PYTHONPATH=f"{root}/src:{root}/packages/openpi-client/src")
        for key in ("XLA_FLAGS", "JAX_PLATFORMS", "JAX_PLATFORM_NAME"):
            env.pop(key, None)
        log = root / "logs/finish_5k_training.log"
        command = [sys.executable, str(root / "scripts/train_vtla_finish.py"),
                   "--checkpoint-step", str(args.checkpoint_step),
                   "--additional-steps", str(args.additional_steps), "--end-lr", str(args.end_lr)]
        with log.open("x") as stream:
            child = subprocess.Popen(command, cwd=root, env=env, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            status("resuming", training_pid=child.pid, training_log=str(log), command=command)
            verified = False
            while child.poll() is None:
                if not verified:
                    with log.open() as f:
                        f.seek(max(0, log.stat().st_size - 131072))
                        tail = f.read()
                    if "FINISH_RESUME_VERIFIED" in tail:
                        verified = True
                        status("training", resume_verified=True)
                time.sleep(10)
            code = child.returncode
        if code:
            raise RuntimeError(f"Continuation exited with code {code}; inspect {log}")
        final_counts = checkpoint_counts(checkpoint_root / str(state["final_checkpoint"]))
        if set(final_counts.values()) != {expected + args.additional_steps}:
            raise RuntimeError(f"Unexpected final optimizer counters: {final_counts}")
        status("complete", final_counts=final_counts, returncode=code)
    except BaseException as error:
        status("failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
