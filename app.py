import importlib
import json
import os

import modal

step_001_download_model = importlib.import_module(
    "helpers.001_download_model"
).step_001_download_model
step_002_benchmark_weights = importlib.import_module(
    "helpers.002_benchmark_weights"
).step_002_benchmark_weights
step_003_start_vllm_with_model_init_observer = importlib.import_module(
    "helpers.003_model_init"
).step_003_start_vllm_with_model_init_observer
step_004_create_kv_cache_observer = importlib.import_module(
    "helpers.004_kv_cache"
).step_004_create_kv_cache_observer
step_005_create_kernel_jit_observer = importlib.import_module(
    "helpers.005_kernel_jit"
).step_005_create_kernel_jit_observer
step_006_create_cuda_graph_observer = importlib.import_module(
    "helpers.006_cuda_graph"
).step_006_create_cuda_graph_observer
step_007_start_api_ready_observer = importlib.import_module(
    "helpers.007_api_ready"
).step_007_start_api_ready_observer
step_008_run_warmup = importlib.import_module(
    "helpers.008_warmup"
).step_008_run_warmup
cache_restore_module = importlib.import_module("helpers.009_cache_restore")
CacheArtifact = cache_restore_module.CacheArtifact
cache_fingerprint = cache_restore_module.cache_fingerprint
step_009_restore_cache = cache_restore_module.step_009_restore_cache
step_010_publish_cache = importlib.import_module(
    "helpers.010_cache_publish"
).step_010_publish_cache
step_011_discover_runtime_caches = importlib.import_module(
    "helpers.011_cache_discovery"
).step_011_discover_runtime_caches
cache_staging_module = importlib.import_module("helpers.012_cache_staging")
step_012_stage_cache = cache_staging_module.step_012_stage_cache
step_012_sync_cache = cache_staging_module.step_012_sync_cache
step_012_cache_dirty = cache_staging_module.step_012_cache_dirty
step_012_clear_dirty = cache_staging_module.step_012_clear_dirty
archive_staging_module = importlib.import_module("helpers.013_stage_archive")
step_013_stage_archive = archive_staging_module.step_013_stage_archive
step_013_sync_archive = archive_staging_module.step_013_sync_archive
step_013_compact_legacy_seed = archive_staging_module.step_013_compact_legacy_seed


APP_NAME = os.getenv("GLM53_APP_NAME", "glm53-flash-nota-b300")
MODAL_WORKSPACE = os.getenv("GLM53_MODAL_WORKSPACE", "zhiyuqqq")
PUBLIC_BASE_URL = f"https://{MODAL_WORKSPACE}--{APP_NAME}-serve.modal.run"
PUBLIC_OPENAI_BASE_URL = f"{PUBLIC_BASE_URL}/v1"
MODEL = "nota-ai/GLM-5.3-Flash-Nota-NVFP4"
REVISION = "c5fc7f5ef0447ab030559bb4b08861a36dbbe847"

HF_CACHE = "/root/.cache/huggingface"
HF_VOLUME_NAME = "glm53-flash-nota-hf-cache"
FLASHINFER_AUTOTUNE_CACHE = "/root/.cache/vllm/flashinfer_autotune_cache"
FLASHINFER_AUTOTUNE_VOLUME_NAME = "glm53-flash-nota-flashinfer-autotune"
FLASHINFER_JIT_CACHE = "/root/.cache/flashinfer"
FLASHINFER_JIT_VOLUME_NAME = "glm53-flash-nota-flashinfer-jit"
TILELANG_CACHE = "/root/.tilelang/cache"
TILELANG_VOLUME_NAME = "glm53-flash-nota-tilelang-cache"
STAGED_CACHE_ROOT = "/tmp/glm53-runtime-cache"
TRITON_CACHE = f"{STAGED_CACHE_ROOT}/triton"
TRITON_CACHE_SEED = "/cache-seeds/triton"
TRITON_VOLUME_NAME = "glm53-flash-nota-triton-cache"
TORCHINDUCTOR_CACHE = f"{STAGED_CACHE_ROOT}/torchinductor"
TORCHINDUCTOR_CACHE_SEED = "/cache-seeds/torchinductor"
TORCHINDUCTOR_VOLUME_NAME = "glm53-flash-nota-torchinductor-cache"
CUDA_COMPUTE_CACHE = f"{STAGED_CACHE_ROOT}/cuda-compute"
CUDA_COMPUTE_CACHE_SEED = "/cache-seeds/cuda-compute"
CUDA_COMPUTE_VOLUME_NAME = "glm53-flash-nota-cuda-compute-cache"

GITHUB_REPO = "xiaoqianran/modal-GLM-5.3-Flash-Nota-NVFP4"
CACHE_RELEASE_TAG = "cache-b300-glm53-flash-nota-v1"
RUNTIME_CACHE_ARTIFACTS = (
    CacheArtifact(
        name="flashinfer-autotune",
        local_path=FLASHINFER_AUTOTUNE_CACHE,
        volume_name=FLASHINFER_AUTOTUNE_VOLUME_NAME,
        release_asset="flashinfer-autotune-0.6.18-b300.tar.gz",
        required_globs=("**/autotune_configs.json",),
    ),
    CacheArtifact(
        name="flashinfer-jit",
        local_path=FLASHINFER_JIT_CACHE,
        volume_name=FLASHINFER_JIT_VOLUME_NAME,
        release_asset="flashinfer-jit-0.6.18-b300.tar.gz",
        required_globs=("**/*.so", "**/*.o", "**/*.cubin"),
    ),
    CacheArtifact(
        name="tilelang",
        local_path=TILELANG_CACHE,
        volume_name=TILELANG_VOLUME_NAME,
        release_asset="tilelang-b300.tar.gz",
        required_globs=("**/*.so", "**/*.cubin", "**/best_config.json"),
    ),
)

STAGED_CACHE_ARTIFACTS = (
    CacheArtifact(
        name="triton",
        local_path=TRITON_CACHE_SEED,
        volume_name=TRITON_VOLUME_NAME,
        release_asset="triton-b300.tar.gz",
        required_globs=(".stage-cache.tar",),
    ),
    CacheArtifact(
        name="torchinductor",
        local_path=TORCHINDUCTOR_CACHE_SEED,
        volume_name=TORCHINDUCTOR_VOLUME_NAME,
        release_asset="torchinductor-b300.tar.gz",
        required_globs=(".stage-cache.tar",),
    ),
    CacheArtifact(
        name="cuda-compute",
        local_path=CUDA_COMPUTE_CACHE_SEED,
        volume_name=CUDA_COMPUTE_VOLUME_NAME,
        release_asset="cuda-compute-b300.tar.gz",
        required_globs=("**/*",),
    ),
)
STAGED_CACHE_RUNTIME_PATHS = {
    "triton": TRITON_CACHE,
    "torchinductor": TORCHINDUCTOR_CACHE,
    "cuda-compute": CUDA_COMPUTE_CACHE,
}
STAGED_CACHE_NAMES = frozenset(STAGED_CACHE_RUNTIME_PATHS)
ARCHIVED_STAGED_CACHE_NAMES = frozenset({"triton", "torchinductor"})
ALL_CACHE_ARTIFACTS = RUNTIME_CACHE_ARTIFACTS + STAGED_CACHE_ARTIFACTS

MAX_B300_CONTAINERS = 1
MAX_CONCURRENT_INPUTS = 16
SCALEDOWN_WINDOW_SECONDS = 1800

# 已实测：Modal Volume(9P) 上 prefetch-16 明显优于 default / eager。
LOAD_STRATEGY = os.getenv("GLM53_LOAD_STRATEGY", "prefetch").strip().lower()
PREFETCH_THREADS = int(os.getenv("GLM53_PREFETCH_THREADS", "16"))
PREFETCH_BLOCK_MIB = int(os.getenv("GLM53_PREFETCH_BLOCK_MIB", "16"))

if LOAD_STRATEGY not in {"default", "lazy", "prefetch", "eager"}:
    raise ValueError(f"Unsupported GLM53_LOAD_STRATEGY={LOAD_STRATEGY!r}")
if PREFETCH_THREADS < 1:
    raise ValueError("GLM53_PREFETCH_THREADS must be >= 1")
if PREFETCH_BLOCK_MIB < 1:
    raise ValueError("GLM53_PREFETCH_BLOCK_MIB must be >= 1")


volume = modal.Volume.from_name(HF_VOLUME_NAME, create_if_missing=True)
cache_volumes = {
    artifact.name: modal.Volume.from_name(
        artifact.volume_name,
        create_if_missing=True,
    )
    for artifact in ALL_CACHE_ARTIFACTS
}
cache_mounts = {
    artifact.local_path: cache_volumes[artifact.name]
    for artifact in ALL_CACHE_ARTIFACTS
}
hf_secret = modal.Secret.from_name("huggingface")
github_secret = modal.Secret.from_name("github")

download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("huggingface_hub[hf_xet]")
    .env(
        {
            "HF_HOME": HF_CACHE,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_XET_CLIENT_RETRY_MAX_ATTEMPTS": "10",
            "HF_XET_CLIENT_RETRY_MAX_DURATION": "600s",
        }
    )
    .add_local_dir("helpers", "/root/helpers")
)

cache_image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_dir("helpers", "/root/helpers")
)


runtime_image = (
    modal.Image.from_registry("vllm/vllm-openai:glm53-flash", add_python="3.12")
    .entrypoint([])
    .env(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "HF_HOME": HF_CACHE,
            "HF_HUB_OFFLINE": "1",
            "GLM53_LOAD_STRATEGY": LOAD_STRATEGY,
            "GLM53_PREFETCH_THREADS": str(PREFETCH_THREADS),
            "GLM53_PREFETCH_BLOCK_MIB": str(PREFETCH_BLOCK_MIB),
            "TILELANG_CACHE_DIR": TILELANG_CACHE,
            "TRITON_CACHE_DIR": TRITON_CACHE,
            "TRITON_CACHE_AUTOTUNING": "1",
            "TORCHINDUCTOR_CACHE_DIR": TORCHINDUCTOR_CACHE,
            "CUDA_CACHE_PATH": CUDA_COMPUTE_CACHE,
        }
    )
    .add_local_dir("helpers", "/root/helpers")
)

app = modal.App(APP_NAME)


@app.function(
    image=download_image,
    cpu=4,
    memory=65536,
    timeout=14400,
    secrets=[hf_secret],
    volumes={HF_CACHE: volume},
)
def step_001_download():
    """下载并缓存固定 revision 的模型权重；已完整缓存时直接返回。"""
    step_001_download_model(MODEL, REVISION, volume)


def _restore_cache_artifacts(artifacts) -> dict[str, str]:
    results: dict[str, str] = {}
    for artifact in artifacts:
        source = step_009_restore_cache(
            artifact,
            github_repo=GITHUB_REPO,
            release_tag=CACHE_RELEASE_TAG,
        )
        results[artifact.name] = source
        if source == "github":
            cache_volumes[artifact.name].commit()
            print(
                f"[009_CACHE_MODAL_COMMIT] name={artifact.name} source=github",
                flush=True,
            )
    return results


def _restore_runtime_caches() -> dict[str, str]:
    """CPU 部署阶段恢复全部 cache；B300 启动阶段只恢复非 staged cache。"""
    return _restore_cache_artifacts(ALL_CACHE_ARTIFACTS)


def _publish_runtime_caches() -> dict[str, bool]:
    results: dict[str, bool] = {}
    for artifact in ALL_CACHE_ARTIFACTS:
        staged_dirty = (
            artifact.name in STAGED_CACHE_NAMES
            and step_012_cache_dirty(artifact.local_path)
        )
        published = step_010_publish_cache(
            artifact,
            github_repo=GITHUB_REPO,
            release_tag=CACHE_RELEASE_TAG,
            replace_existing=staged_dirty,
        )
        results[artifact.name] = published
        if published and staged_dirty:
            step_012_clear_dirty(artifact.local_path)
            cache_volumes[artifact.name].commit()
            print(
                f"[012_CACHE_DIRTY_CLEARED] name={artifact.name}",
                flush=True,
            )
    return results


@app.function(
    image=cache_image,
    cpu=1,
    memory=1024,
    timeout=900,
    secrets=[github_secret],
    volumes=cache_mounts,
)
def step_009_restore_runtime_caches():
    """CPU 阶段：Volume 优先，缺失时从 GitHub Release 恢复运行时缓存。"""
    return _restore_runtime_caches()


@app.function(
    image=cache_image,
    cpu=1,
    memory=2048,
    timeout=1800,
    secrets=[github_secret],
    volumes=cache_mounts,
)
def step_010_publish_runtime_caches():
    """CPU 阶段：确保当前可移植缓存已有 GitHub Release 备份。"""
    return _publish_runtime_caches()


@app.function(
    image=cache_image,
    cpu=2,
    memory=4096,
    timeout=1800,
    volumes=cache_mounts,
)
def step_013_compact_staged_archives():
    """CPU 一次性迁移：把 Triton/Inductor 多文件 seed 收口为单归档。"""
    results: dict[str, bool] = {}
    for artifact in STAGED_CACHE_ARTIFACTS:
        if artifact.name not in ARCHIVED_STAGED_CACHE_NAMES:
            continue
        changed = step_013_compact_legacy_seed(
            artifact.local_path,
            name=artifact.name,
        )
        results[artifact.name] = changed
        cache_volumes[artifact.name].commit()
    return results


@app.function(
    image=runtime_image,
    gpu="B300",
    cpu=8,
    memory=32768,
    timeout=900,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    min_containers=0,
    max_containers=1,
    buffer_containers=0,
    volumes={HF_CACHE: volume},
)
def step_002_bench_weights(
    strategy: str = "prefetch",
    threads: int = 16,
    block_mib: int = 16,
):
    """仅测试 386 个 safetensors shard 的读取路径，不启动 vLLM 或 CUDA Graph。"""
    step_002_benchmark_weights(
        HF_CACHE,
        REVISION,
        strategy=strategy,
        threads=threads,
        block_mib=block_mib,
    )


@app.function(
    image=runtime_image,
    gpu="B300",
    cpu=8,
    memory=32768,
    timeout=1800,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    min_containers=0,
    max_containers=MAX_B300_CONTAINERS,
    buffer_containers=0,
    volumes={HF_CACHE: volume, **cache_mounts},
    secrets=[github_secret],
)
@modal.concurrent(max_inputs=MAX_CONCURRENT_INPUTS)
@modal.web_server(8000, startup_timeout=1800)
def serve():
    """启动单 B300 vLLM 服务，并在服务就绪后自动发送一次最小 warmup 请求。"""
    print(
        f"[PUBLIC_API] base_url={PUBLIC_BASE_URL} openai_base_url={PUBLIC_OPENAI_BASE_URL}",
        flush=True,
    )
    cache_sources = _restore_cache_artifacts(RUNTIME_CACHE_ARTIFACTS)
    cache_fingerprints_before = {
        artifact.name: cache_fingerprint(artifact)
        for artifact in RUNTIME_CACHE_ARTIFACTS
    }
    for artifact in STAGED_CACHE_ARTIFACTS:
        runtime_path = STAGED_CACHE_RUNTIME_PATHS[artifact.name]
        if artifact.name in ARCHIVED_STAGED_CACHE_NAMES:
            archived = step_013_stage_archive(
                artifact.local_path,
                runtime_path,
                name=artifact.name,
            )
            if not archived:
                step_012_stage_cache(
                    artifact.local_path,
                    runtime_path,
                    name=artifact.name,
                )
        else:
            step_012_stage_cache(
                artifact.local_path,
                runtime_path,
                name=artifact.name,
            )

    strategy = os.environ.get(
        "GLM53_LOAD_STRATEGY",
        "prefetch",
    ).strip().lower()
    prefetch_threads = int(
        os.environ.get("GLM53_PREFETCH_THREADS", "16")
    )
    prefetch_block_mib = int(
        os.environ.get("GLM53_PREFETCH_BLOCK_MIB", "16")
    )

    command = [
        "vllm",
        "serve",
        MODEL,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--served-model-name",
        MODEL,
        "--revision",
        REVISION,
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        "0.96",
        "--enforce-eager",
        "--compilation-config",
        json.dumps({"cudagraph_mode": "PIECEWISE"}),
        "--max-cudagraph-capture-size",
        "1008",
        "--kv-cache-dtype",
        "fp8",
        "--speculative-config",
        json.dumps({"method": "mtp", "num_speculative_tokens": 5}),
        "--tool-call-parser",
        "glm47",
        "--reasoning-parser",
        "glm45",
        "--enable-auto-tool-choice",
    ]

    if strategy != "default":
        command += ["--safetensors-load-strategy", strategy]

    if strategy == "prefetch":
        command += [
            "--safetensors-prefetch-num-threads",
            str(prefetch_threads),
            "--safetensors-prefetch-block-size",
            str(prefetch_block_mib * 1024 * 1024),
        ]

    print(
        "[MODEL_LOAD] "
        f"strategy={strategy} "
        f"prefetch_threads={prefetch_threads} "
        f"prefetch_block_mib={prefetch_block_mib}",
        flush=True,
    )

    # 003~006：保持单一 vLLM 进程，并按阶段旁路观测同一份真实日志。
    vllm_handle = step_003_start_vllm_with_model_init_observer(
        command,
        extra_observer_factories=[
            step_004_create_kv_cache_observer,
            step_005_create_kernel_jit_observer,
            step_006_create_cuda_graph_observer,
        ],
    )

    def warmup_once(api_ready_at: float) -> None:
        """008：API Ready 后执行真实 warmup，再固化并发现运行时缓存。"""
        step_008_run_warmup(
            model=MODEL,
            process_started_at=vllm_handle.started_at,
            api_ready_at=api_ready_at,
        )
        for artifact in STAGED_CACHE_ARTIFACTS:
            runtime_path = STAGED_CACHE_RUNTIME_PATHS[artifact.name]
            if artifact.name in ARCHIVED_STAGED_CACHE_NAMES:
                step_013_sync_archive(
                    runtime_path,
                    artifact.local_path,
                    name=artifact.name,
                )
            else:
                step_012_sync_cache(
                    runtime_path,
                    artifact.local_path,
                    name=artifact.name,
                )
            cache_volumes[artifact.name].commit()
            print(
                f"[RUNTIME_CACHE_COMMIT] name={artifact.name} path={artifact.local_path}",
                flush=True,
            )

        for artifact in RUNTIME_CACHE_ARTIFACTS:
            cache_volumes[artifact.name].commit()
            print(
                f"[RUNTIME_CACHE_COMMIT] name={artifact.name} path={artifact.local_path}",
                flush=True,
            )

            before = cache_fingerprints_before.get(artifact.name, ())
            after = cache_fingerprint(artifact)
            changed = before != after
            if changed:
                print(
                    f"[RUNTIME_CACHE_CHANGED] name={artifact.name} "
                    f"before_files={len(before)} after_files={len(after)}",
                    flush=True,
                )

            if not changed and cache_sources.get(artifact.name) != "miss":
                continue
            try:
                step_010_publish_cache(
                    artifact,
                    github_repo=GITHUB_REPO,
                    release_tag=CACHE_RELEASE_TAG,
                    replace_existing=changed,
                )
            except Exception as exc:
                print(
                    "[010_CACHE_PUBLISH_FAILED] "
                    f"name={artifact.name} error={type(exc).__name__}",
                    flush=True,
                )

        step_011_discover_runtime_caches()

    # 007/008：HTTP 200 确认 API Ready 后，立即执行最小真实 generation。
    step_007_start_api_ready_observer(
        process=vllm_handle.process,
        process_started_at=vllm_handle.started_at,
        on_ready=warmup_once,
    )
