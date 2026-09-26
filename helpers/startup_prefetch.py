"""Lightweight page-cache prefetch, shared by bootstrap and EngineCore.

Never import torch/vLLM here: reads must overlap their import time.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from filelock import FileLock, Timeout


def start_prefetch(
    files: list[str], *, num_prefetch_threads: int, block_size: int,
) -> threading.Thread | None:
    if num_prefetch_threads < 1 or block_size < 1:
        raise ValueError("prefetch threads and block size must be positive")
    paths = sorted({os.path.abspath(path) for path in files})
    if not paths:
        return None
    state_dir = os.environ.get("VLLM_PREFETCH_STATE_DIR")
    if state_dir is None:
        # Also support direct loader/benchmark entrypoints without bootstrap.
        state_dir = tempfile.mkdtemp(prefix="glm53-prefetch-")
        os.environ["VLLM_PREFETCH_STATE_DIR"] = state_dir
    root = Path(state_dir)
    root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256("\0".join(paths).encode()).hexdigest()
    done = root / f"{digest}.done"
    lock = FileLock(str(root / f"{digest}.lock"), thread_local=False)
    if done.is_file():
        print(f"[WEIGHT_PREFETCH_REUSE] state=done files={len(paths)}", flush=True)
        return None
    try:
        lock.acquire(timeout=0)
    except Timeout:
        print(f"[WEIGHT_PREFETCH_REUSE] state=inflight action=load-without-wait files={len(paths)}", flush=True)
        return None
    if done.is_file():
        lock.release()
        return None

    def read_file(path: str) -> None:
        # Also coordinate overlapping shard subsets (e.g. MTP).
        key = hashlib.sha256(path.encode()).hexdigest()
        shard_done = root / f"{key}.shard-done"
        with FileLock(str(root / f"{key}.shard-lock")):
            if shard_done.is_file():
                return
            buffer = bytearray(block_size)
            with open(path, "rb", buffering=0) as stream:
                while stream.readinto(buffer):
                    pass
            shard_done.touch()

    def run() -> None:
        started = time.perf_counter()
        try:
            with ThreadPoolExecutor(max_workers=num_prefetch_threads) as pool:
                # Consume results: a failed shard must never publish a done marker.
                list(pool.map(read_file, paths))
            done.touch()
            print(f"[WEIGHT_PREFETCH_DONE] files={len(paths)} elapsed_s={time.perf_counter() - started:.3f}", flush=True)
        except Exception as exc:
            print(f"[WEIGHT_PREFETCH_ABORTED] error={type(exc).__name__}: {exc}", flush=True)
        finally:
            lock.release()

    thread = threading.Thread(target=run, name="early-weight-prefetch", daemon=True)
    try:
        thread.start()
    except BaseException:
        lock.release()
        raise
    return thread
