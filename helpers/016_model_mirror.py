from __future__ import annotations

import os
import shutil
import time
from pathlib import Path


def _snapshot_path(hf_cache: str, model: str, revision: str) -> Path:
    repo_dir = "models--" + model.replace("/", "--")
    return Path(hf_cache) / "hub" / repo_dir / "snapshots" / revision


def step_016_prepare_model_mirror(
    *,
    hf_cache: str,
    model: str,
    revision: str,
    destination: str = "/tmp/glm53-model",
) -> str:
    """把模型 metadata 复制到本地 /tmp；权重只建立指向 HF Volume blob 的 symlink。"""
    source = _snapshot_path(hf_cache, model, revision)
    if not source.is_dir():
        raise FileNotFoundError(f"HF snapshot not found: {source}")

    target_root = Path(destination)
    if target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    started_at = time.perf_counter()
    metadata_files = 0
    metadata_bytes = 0
    weight_links = 0

    for source_path in source.rglob("*"):
        relative = source_path.relative_to(source)
        target = target_root / relative

        if source_path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        if source_path.suffix == ".safetensors":
            blob = source_path.resolve(strict=True)
            os.symlink(blob, target)
            weight_links += 1
            continue

        shutil.copy2(source_path, target, follow_symlinks=True)
        metadata_files += 1
        metadata_bytes += target.stat().st_size

    required = (
        target_root / "config.json",
        target_root / "model.safetensors.index.json",
    )
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Model mirror missing required metadata: {missing}")
    if weight_links == 0:
        raise RuntimeError("Model mirror contains no safetensors weight links")

    elapsed_s = time.perf_counter() - started_at
    print(
        "[016_MODEL_MIRROR] "
        f"source={source} destination={target_root} "
        f"metadata_files={metadata_files} metadata_bytes={metadata_bytes} "
        f"weight_links={weight_links} elapsed_s={elapsed_s:.3f}",
        flush=True,
    )
    return str(target_root)
