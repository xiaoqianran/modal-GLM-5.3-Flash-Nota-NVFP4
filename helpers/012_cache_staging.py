from __future__ import annotations

import shutil
from pathlib import Path


def _copy_tree(source: Path, destination: Path) -> int:
    if not source.exists():
        return 0
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        count += 1
    return count


def step_012_stage_cache(seed_path: str, runtime_path: str, *, name: str) -> int:
    """从持久化 seed 复制到容器本地高速目录。"""
    seed = Path(seed_path)
    runtime = Path(runtime_path)
    files = _copy_tree(seed, runtime)
    print(
        f"[012_CACHE_STAGE] name={name} files={files} seed={seed_path} runtime={runtime_path}",
        flush=True,
    )
    return files


def step_012_sync_cache(runtime_path: str, seed_path: str, *, name: str) -> int:
    """warmup 后把本地 cache 增量同步回持久化 seed。"""
    runtime = Path(runtime_path)
    seed = Path(seed_path)
    files = _copy_tree(runtime, seed)
    print(
        f"[012_CACHE_SYNC] name={name} files={files} runtime={runtime_path} seed={seed_path}",
        flush=True,
    )
    return files
