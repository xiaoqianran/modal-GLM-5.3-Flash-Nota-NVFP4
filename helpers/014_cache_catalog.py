from __future__ import annotations

import importlib

CacheArtifact = importlib.import_module("helpers.009_cache_restore").CacheArtifact

PROJECT_VOLUME_NAME = "modal-GLM-5.3-Flash-Nota-NVFP4"
PROJECT_VOLUME_ROOT = "/project-volume"

HF_CACHE = f"{PROJECT_VOLUME_ROOT}/glm53-flash-nota-hf-cache"
FLASHINFER_AUTOTUNE_CACHE = (
    f"{PROJECT_VOLUME_ROOT}/glm53-flash-nota-flashinfer-autotune"
)
FLASHINFER_JIT_CACHE = f"{PROJECT_VOLUME_ROOT}/glm53-flash-nota-flashinfer-jit"
TILELANG_CACHE = f"{PROJECT_VOLUME_ROOT}/glm53-flash-nota-tilelang-cache"

STAGED_CACHE_ROOT = "/tmp/glm53-runtime-cache"
TRITON_CACHE = f"{STAGED_CACHE_ROOT}/triton"
TRITON_CACHE_SEED = f"{PROJECT_VOLUME_ROOT}/glm53-flash-nota-triton-cache"

TORCHINDUCTOR_CACHE = f"{STAGED_CACHE_ROOT}/torchinductor"
TORCHINDUCTOR_CACHE_SEED = (
    f"{PROJECT_VOLUME_ROOT}/glm53-flash-nota-torchinductor-cache"
)

CUDA_COMPUTE_CACHE = f"{STAGED_CACHE_ROOT}/cuda-compute"
CUDA_COMPUTE_CACHE_SEED = (
    f"{PROJECT_VOLUME_ROOT}/glm53-flash-nota-cuda-compute-cache"
)

GITHUB_REPO = "xiaoqianran/modal-GLM-5.3-Flash-Nota-NVFP4"
CACHE_RELEASE_TAG = "cache-b300-glm53-flash-nota-v1"

RUNTIME_CACHE_ARTIFACTS = (
    CacheArtifact(
        name="flashinfer-autotune",
        local_path=FLASHINFER_AUTOTUNE_CACHE,
        volume_name=PROJECT_VOLUME_NAME,
        release_asset="flashinfer-autotune-0.6.18-b300.tar.gz",
        required_globs=("**/autotune_configs.json",),
    ),
    CacheArtifact(
        name="flashinfer-jit",
        local_path=FLASHINFER_JIT_CACHE,
        volume_name=PROJECT_VOLUME_NAME,
        release_asset="flashinfer-jit-0.6.18-b300.tar.gz",
        required_globs=("**/*.so", "**/*.o", "**/*.cubin"),
    ),
    CacheArtifact(
        name="tilelang",
        local_path=TILELANG_CACHE,
        volume_name=PROJECT_VOLUME_NAME,
        release_asset="tilelang-b300.tar.gz",
        required_globs=("**/*.so", "**/*.cubin", "**/best_config.json"),
    ),
)

STAGED_CACHE_ARTIFACTS = (
    CacheArtifact(
        name="triton",
        local_path=TRITON_CACHE_SEED,
        volume_name=PROJECT_VOLUME_NAME,
        release_asset="triton-b300.tar.gz",
        required_globs=(".stage-cache.tar",),
    ),
    CacheArtifact(
        name="torchinductor",
        local_path=TORCHINDUCTOR_CACHE_SEED,
        volume_name=PROJECT_VOLUME_NAME,
        release_asset="torchinductor-b300.tar.gz",
        required_globs=(".stage-cache.tar",),
    ),
    CacheArtifact(
        name="cuda-compute",
        local_path=CUDA_COMPUTE_CACHE_SEED,
        volume_name=PROJECT_VOLUME_NAME,
        release_asset="cuda-compute-b300.tar.gz",
        required_globs=("cuda-compute-b300.tar",),
    ),
)

STAGED_CACHE_RUNTIME_PATHS = {
    "triton": TRITON_CACHE,
    "torchinductor": TORCHINDUCTOR_CACHE,
    "cuda-compute": CUDA_COMPUTE_CACHE,
}
STAGED_CACHE_ARCHIVE_NAMES = {
    "triton": ".stage-cache.tar",
    "torchinductor": ".stage-cache.tar",
    "cuda-compute": "cuda-compute-b300.tar",
}
STAGED_CACHE_NAMES = frozenset(STAGED_CACHE_RUNTIME_PATHS)
ARCHIVED_STAGED_CACHE_NAMES = frozenset(
    {"triton", "torchinductor", "cuda-compute"}
)
ALL_CACHE_ARTIFACTS = RUNTIME_CACHE_ARTIFACTS + STAGED_CACHE_ARTIFACTS
