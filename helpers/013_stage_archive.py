from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import time
from pathlib import Path


ARCHIVE_NAME = ".stage-cache.tar"
META_NAME = ".stage-cache-meta.json"
DIRTY_NAME = ".staged-cache-dirty"
_RESERVED = {ARCHIVE_NAME, META_NAME, DIRTY_NAME}


def _tree_manifest(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    files = 0
    total_bytes = 0
    if not root.exists():
        return digest.hexdigest(), files, total_bytes

    records: list[tuple[str, int, int]] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.name in _RESERVED:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        records.append((path.relative_to(root).as_posix(), stat.st_size, int(stat.st_mtime)))

    records.sort()
    for relative, size, mtime_ns in records:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(mtime_ns).encode("ascii"))
        digest.update(b"\n")
        files += 1
        total_bytes += size

    return digest.hexdigest(), files, total_bytes


def _read_meta(seed: Path) -> dict[str, object] | None:
    path = seed / META_NAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_meta(seed: Path, *, fingerprint: str, files: int, total_bytes: int) -> None:
    payload = {
        "fingerprint": fingerprint,
        "files": files,
        "bytes": total_bytes,
    }
    (seed / META_NAME).write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r") as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError(f"Unsafe staged cache member: {member.name}")
        tar.extractall(destination)


def _build_archive(source: Path, archive: Path) -> None:
    with tarfile.open(archive, "w") as tar:
        for item in sorted(source.iterdir()):
            if item.name in _RESERVED:
                continue
            tar.add(item, arcname=item.name, recursive=True)


def _prune_legacy_seed(seed: Path) -> int:
    removed = 0
    for item in list(seed.iterdir()):
        if item.name in _RESERVED:
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
        removed += 1
    return removed


def step_013_stage_archive(seed_path: str, runtime_path: str, *, name: str) -> bool:
    """单归档 seed -> 本地高速目录；返回是否命中归档。"""
    seed = Path(seed_path)
    archive = seed / ARCHIVE_NAME
    runtime = Path(runtime_path)
    if not archive.is_file():
        print(f"[013_ARCHIVE_MISS] name={name} seed={seed_path}", flush=True)
        return False

    if runtime.exists():
        shutil.rmtree(runtime)
    runtime.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    archive_bytes = archive.stat().st_size
    with tempfile.TemporaryDirectory(prefix=f"glm53-{name}-stage-") as temp_dir:
        local_archive = Path(temp_dir) / ARCHIVE_NAME
        shutil.copyfile(archive, local_archive)
        _safe_extract(local_archive, runtime)
    elapsed_s = time.perf_counter() - started
    fingerprint, files, total_bytes = _tree_manifest(runtime)
    mib_s = archive_bytes / 1024 / 1024 / elapsed_s if elapsed_s > 0 else 0.0
    print(
        "[013_ARCHIVE_STAGE] "
        f"name={name} files={files} bytes={total_bytes} archive_bytes={archive_bytes} "
        f"elapsed_s={elapsed_s:.3f} mib_s={mib_s:.2f} fingerprint={fingerprint[:12]}",
        flush=True,
    )
    return True


def step_013_sync_archive(runtime_path: str, seed_path: str, *, name: str) -> bool:
    """仅在本地 cache manifest 变化时重建单归档并写回 Volume。"""
    runtime = Path(runtime_path)
    seed = Path(seed_path)
    seed.mkdir(parents=True, exist_ok=True)

    fingerprint, files, total_bytes = _tree_manifest(runtime)
    previous = _read_meta(seed)
    archive = seed / ARCHIVE_NAME
    if (
        archive.is_file()
        and previous is not None
        and previous.get("fingerprint") == fingerprint
    ):
        print(
            "[013_ARCHIVE_SYNC] "
            f"name={name} changed=false files={files} bytes={total_bytes} "
            f"fingerprint={fingerprint[:12]}",
            flush=True,
        )
        return False

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f"glm53-{name}-archive-") as temp_dir:
        local_archive = Path(temp_dir) / ARCHIVE_NAME
        _build_archive(runtime, local_archive)
        archive_bytes = local_archive.stat().st_size
        temp_target = seed / f"{ARCHIVE_NAME}.tmp"
        shutil.copyfile(local_archive, temp_target)
        os.replace(temp_target, archive)

    _write_meta(seed, fingerprint=fingerprint, files=files, total_bytes=total_bytes)
    (seed / DIRTY_NAME).write_text("1\n", encoding="utf-8")
    removed = _prune_legacy_seed(seed)
    elapsed_s = time.perf_counter() - started
    print(
        "[013_ARCHIVE_SYNC] "
        f"name={name} changed=true files={files} bytes={total_bytes} "
        f"archive_bytes={archive_bytes} removed_legacy={removed} "
        f"elapsed_s={elapsed_s:.3f} fingerprint={fingerprint[:12]}",
        flush=True,
    )
    return True


def step_013_compact_legacy_seed(seed_path: str, *, name: str) -> bool:
    """一次性把旧版多文件 Volume seed 压成单个未压缩 tar。"""
    seed = Path(seed_path)
    archive = seed / ARCHIVE_NAME
    if archive.is_file():
        removed = _prune_legacy_seed(seed)
        print(
            f"[013_ARCHIVE_COMPACT_SKIP] name={name} archive_exists=true removed_legacy={removed}",
            flush=True,
        )
        return False

    fingerprint, files, total_bytes = _tree_manifest(seed)
    if files == 0:
        print(f"[013_ARCHIVE_COMPACT_SKIP] name={name} reason=empty_seed", flush=True)
        return False

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f"glm53-{name}-compact-") as temp_dir:
        local_archive = Path(temp_dir) / ARCHIVE_NAME
        _build_archive(seed, local_archive)
        archive_bytes = local_archive.stat().st_size
        shutil.copyfile(local_archive, archive)

    _write_meta(seed, fingerprint=fingerprint, files=files, total_bytes=total_bytes)
    (seed / DIRTY_NAME).write_text("1\n", encoding="utf-8")
    removed = _prune_legacy_seed(seed)
    elapsed_s = time.perf_counter() - started
    print(
        "[013_ARCHIVE_COMPACT] "
        f"name={name} files={files} bytes={total_bytes} archive_bytes={archive_bytes} "
        f"removed_legacy={removed} elapsed_s={elapsed_s:.3f} fingerprint={fingerprint[:12]}",
        flush=True,
    )
    return True

