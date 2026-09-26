from __future__ import annotations

import os
import shutil
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
