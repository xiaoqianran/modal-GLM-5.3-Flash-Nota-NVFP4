from __future__ import annotations

import shutil
import time
from pathlib import Path


_DIRTY_MARKER = ".staged-cache-dirty"


def _fingerprint(root: Path) -> tuple[tuple[str, int, int], ...]:
    if not root.exists():
        return ()
    records: list[tuple[str, int, int]] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.name == _DIRTY_MARKER:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        records.append((path.relative_to(root).as_posix(), stat.st_size, stat.st_mtime_ns))
    records.sort()
    return tuple(records)


def _copy_tree(source: Path, destination: Path) -> tuple[int, int]:
    if not source.exists():
        return 0, 0
    destination.mkdir(parents=True, exist_ok=True)
    files = 0
    total_bytes = 0
    for path in source.rglob("*"):
        if not path.is_file() or path.name == _DIRTY_MARKER:
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        try:
            total_bytes += path.stat().st_size
        except OSError:
            pass
        files += 1
    return files, total_bytes


def _format_rate(total_bytes: int, elapsed_s: float) -> float:
    if elapsed_s <= 0:
        return 0.0
    return total_bytes / 1024 / 1024 / elapsed_s


def step_012_stage_cache(seed_path: str, runtime_path: str, *, name: str) -> int:
    """从持久化 seed Volume 复制到容器本地高速目录。"""
    seed = Path(seed_path)
    runtime = Path(runtime_path)
    runtime.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    files, total_bytes = _copy_tree(seed, runtime)
    elapsed_s = time.perf_counter() - started
    print(
        "[012_CACHE_STAGE] "
        f"name={name} files={files} bytes={total_bytes} "
        f"elapsed_s={elapsed_s:.3f} mib_s={_format_rate(total_bytes, elapsed_s):.2f} "
        f"seed={seed_path} runtime={runtime_path}",
        flush=True,
    )
    return files


def step_012_sync_cache(runtime_path: str, seed_path: str, *, name: str) -> int:
    """warmup 后把本地 cache 增量同步回持久化 seed，并标记是否发生变化。"""
    runtime = Path(runtime_path)
    seed = Path(seed_path)
    before = _fingerprint(seed)
    started = time.perf_counter()
    files, total_bytes = _copy_tree(runtime, seed)
    elapsed_s = time.perf_counter() - started
    after = _fingerprint(seed)
    changed = before != after
    if changed:
        seed.mkdir(parents=True, exist_ok=True)
        (seed / _DIRTY_MARKER).write_text("1\n", encoding="utf-8")
    print(
        "[012_CACHE_SYNC] "
        f"name={name} files={files} bytes={total_bytes} changed={str(changed).lower()} "
        f"elapsed_s={elapsed_s:.3f} mib_s={_format_rate(total_bytes, elapsed_s):.2f} "
        f"runtime={runtime_path} seed={seed_path}",
        flush=True,
    )
    return files


def step_012_cache_dirty(seed_path: str) -> bool:
    return (Path(seed_path) / _DIRTY_MARKER).is_file()


def step_012_clear_dirty(seed_path: str) -> None:
    marker = Path(seed_path) / _DIRTY_MARKER
    if marker.exists():
        marker.unlink()
