from __future__ import annotations

import os
from collections import Counter
from pathlib import Path


_CANDIDATE_ROOTS = (
    "/root/.cache/vllm",
    "/root/.cache/flashinfer",
    "/root/.cache/torch_extensions",
    "/root/.tilelang/cache",
    "/root/.triton",
    "/root/.nv/ComputeCache",
    "/tmp/torchinductor_root",
)

_ENV_KEYS = (
    "VLLM_CACHE_ROOT",
    "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "FLASHINFER_WORKSPACE_BASE",
    "FLASHINFER_CACHE_DIR",
    "FLASHINFER_JIT_DIR",
    "FLASHINFER_CUBIN_DIR",
    "TILELANG_CACHE_DIR",
    "CUDA_CACHE_PATH",
)


def _dir_stats(root: Path) -> tuple[int, int, Counter[str]]:
    files = 0
    total_bytes = 0
    suffixes: Counter[str] = Counter()

    if not root.exists():
        return files, total_bytes, suffixes

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        files += 1
        total_bytes += size
        suffixes[path.suffix.lower() or "<none>"] += 1

    return files, total_bytes, suffixes


def step_011_discover_runtime_caches() -> dict[str, dict[str, object]]:
    """只读扫描已知 cache 根目录，为后续持久化候选提供证据。"""
    print("[011_CACHE_DISCOVERY_START]", flush=True)

    for key in _ENV_KEYS:
        value = os.environ.get(key)
        if value:
            print(f"[011_CACHE_ENV] {key}={value}", flush=True)

    results: dict[str, dict[str, object]] = {}
    roots = list(_CANDIDATE_ROOTS)

    for key in _ENV_KEYS:
        value = os.environ.get(key)
        if value and value.startswith("/") and value not in roots:
            roots.append(value)

    for raw_root in roots:
        root = Path(raw_root)
        files, total_bytes, suffixes = _dir_stats(root)
        top_suffixes = ",".join(
            f"{suffix}:{count}"
            for suffix, count in suffixes.most_common(8)
        ) or "none"
        results[raw_root] = {
            "exists": root.exists(),
            "files": files,
            "bytes": total_bytes,
            "top_suffixes": dict(suffixes.most_common(8)),
        }
        print(
            "[011_CACHE_ROOT] "
            f"path={raw_root} exists={str(root.exists()).lower()} "
            f"files={files} bytes={total_bytes} top_suffixes={top_suffixes}",
            flush=True,
        )

    print("[011_CACHE_DISCOVERY_DONE]", flush=True)
    return results
