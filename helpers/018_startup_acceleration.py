from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import threading
import time
from pathlib import Path


def step_018_prepare_startup_plan_cache(
    *,
    local_cache_root: str,
    persistent_dir: str,
) -> bool:
    """只把 vLLM startup_plan 子目录持久化，避免把整个 VLLM_CACHE_ROOT 放到 9P。"""
    local_root = Path(local_cache_root)
    persistent_root = Path(persistent_dir)
    local_root.mkdir(parents=True, exist_ok=True)
    persistent_root.mkdir(parents=True, exist_ok=True)

    had_existing_plan = any(path.is_file() for path in persistent_root.rglob("*"))
    startup_plan_link = local_root / "startup_plan"

    if startup_plan_link.is_symlink():
        current_target = startup_plan_link.resolve(strict=False)
        if current_target != persistent_root.resolve():
            startup_plan_link.unlink()
    elif startup_plan_link.exists():
        if startup_plan_link.is_dir():
            shutil.rmtree(startup_plan_link)
        else:
            startup_plan_link.unlink()

    if not startup_plan_link.exists():
        os.symlink(persistent_root, startup_plan_link, target_is_directory=True)

    print(
        "[018_STARTUP_PLAN_CACHE] "
        f"local={startup_plan_link} persistent={persistent_root} "
        f"existing={str(had_existing_plan).lower()}",
        flush=True,
    )
    return had_existing_plan


def step_018_count_startup_plan_files(persistent_dir: str) -> int:
    """统计 startup plan 文件，供首次生成后决定是否 commit Volume。"""
    root = Path(persistent_dir)
    if not root.is_dir():
        return 0
    return sum(1 for path in root.rglob("*") if path.is_file())


def step_018_start_early_weight_prefetch(
    *,
    model_path: str,
    threads: int,
    block_mib: int,
) -> threading.Thread:
    """在 vLLM 启动前后台预读全部 safetensors，让前置初始化与 9P I/O 重叠。"""
    if threads < 1:
        raise ValueError("threads must be >= 1")
    if block_mib < 1:
        raise ValueError("block_mib must be >= 1")

    root = Path(model_path)
    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    names = sorted(set(index["weight_map"].values()))
    files = [(root / name).resolve(strict=True) for name in names]
    total_bytes = sum(path.stat().st_size for path in files)
    block_size = block_mib * 1024 * 1024

    def read_file(path: Path) -> int:
        read_bytes = 0
        buffer = bytearray(block_size)
        with path.open("rb", buffering=0) as file:
            while True:
                size = file.readinto(buffer)
                if not size:
                    break
                read_bytes += size
        return read_bytes

    def run() -> None:
        started_at = time.perf_counter()
        print(
            "[018_EARLY_WEIGHT_PREFETCH_START] "
            f"files={len(files)} total_gib={total_bytes / 1024**3:.2f} "
            f"threads={threads} block_mib={block_mib}",
            flush=True,
        )
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
                read_bytes = sum(executor.map(read_file, files))
        except Exception as exc:
            print(
                "[018_EARLY_WEIGHT_PREFETCH_FAILED] "
                f"error={type(exc).__name__}: {exc}",
                flush=True,
            )
            return

        elapsed_s = time.perf_counter() - started_at
        print(
            "[018_EARLY_WEIGHT_PREFETCH_DONE] "
            f"files={len(files)} bytes={read_bytes} elapsed_s={elapsed_s:.3f} "
            f"gib_per_s={read_bytes / 1024**3 / elapsed_s:.3f}",
            flush=True,
        )

    thread = threading.Thread(
        target=run,
        name="early-weight-page-cache-prefetch",
        daemon=True,
    )
    thread.start()
    return thread
