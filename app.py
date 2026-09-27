import hashlib
import importlib
import json
import os
import threading
import time

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
cache_staging_module = importlib.import_module("helpers.012_cache_staging")
step_012_stage_cache = cache_staging_module.step_012_stage_cache
step_012_sync_cache = cache_staging_module.step_012_sync_cache
step_012_cache_dirty = cache_staging_module.step_012_cache_dirty
step_012_clear_dirty = cache_staging_module.step_012_clear_dirty
archive_staging_module = importlib.import_module("helpers.013_stage_archive")
step_013_stage_archive = archive_staging_module.step_013_stage_archive
step_013_sync_archive = archive_staging_module.step_013_sync_archive
step_013_compact_legacy_seed = archive_staging_module.step_013_compact_legacy_seed
cache_backup_state = importlib.import_module("helpers.015_cache_backup_state")
mark_cache_backup_dirty = cache_backup_state.mark_cache_backup_dirty
is_cache_backup_dirty = cache_backup_state.is_cache_backup_dirty
clear_cache_backup_dirty = cache_backup_state.clear_cache_backup_dirty
step_016_prepare_model_mirror = importlib.import_module(
    "helpers.016_model_mirror"
).step_016_prepare_model_mirror
step_017_run_generation_benchmark = importlib.import_module(
    "helpers.017_generation_benchmark"
).step_017_run_generation_benchmark
startup_acceleration_module = importlib.import_module(
    "helpers.018_startup_acceleration"
)
step_018_prepare_startup_plan_cache = (
    startup_acceleration_module.step_018_prepare_startup_plan_cache
)


APP_NAME = os.getenv("GLM53_APP_NAME", "glm53-flash-nota-b300")
MODAL_WORKSPACE = os.getenv("GLM53_MODAL_WORKSPACE", "zhiyuqqq")
PUBLIC_BASE_URL = f"https://{MODAL_WORKSPACE}--{APP_NAME}-vllmserver-serve.modal.run"
PUBLIC_OPENAI_BASE_URL = f"{PUBLIC_BASE_URL}/v1"
MODEL = "nota-ai/GLM-5.3-Flash-Nota-NVFP4"
REVISION = "c5fc7f5ef0447ab030559bb4b08861a36dbbe847"
MODEL_RUNTIME_PATH = "/tmp/glm53-model"
RUNTIME_IMAGE = "vllm/vllm-openai:glm53-flash"
# Verified from /usr/local/bin/vllm's shebang in this image. Modal's added
# /usr/local/bin/python3 has a different site-packages and cannot import vLLM.
VLLM_RUNTIME_PYTHON = "/usr/bin/python3"

PROJECT_VOLUME_NAME = "modal-GLM-5.3-Flash-Nota-NVFP4"
PROJECT_VOLUME_ROOT = "/project-volume"
HF_CACHE = f"{PROJECT_VOLUME_ROOT}/glm53-flash-nota-hf-cache"
VLLM_LOCAL_CACHE_ROOT = "/root/.cache/vllm"
VLLM_STARTUP_PLAN_CACHE = (
    f"{PROJECT_VOLUME_ROOT}/glm53-flash-nota-vllm-startup-plan"
)
VLLM_MODELINFO_CACHE = f"{PROJECT_VOLUME_ROOT}/glm53-flash-nota-vllm-modelinfos"
STARTUP_CAPABILITIES_PATH = "/root/glm53-startup-capabilities.json"
CACHE_AVAILABILITY_DIR = f"{PROJECT_VOLUME_ROOT}/glm53-cache-release-availability"
FLASHINFER_AUTOTUNE_VERSION = "0.6.18"
FLASHINFER_AUTOTUNE_ARCH = "103a"
FLASHINFER_AUTOTUNE_CACHE = (
    f"{PROJECT_VOLUME_ROOT}/glm53-flash-nota-flashinfer-autotune"
)
FLASHINFER_AUTOTUNE_RUNTIME_CACHE = (
    f"{FLASHINFER_AUTOTUNE_CACHE}/"
    f"{FLASHINFER_AUTOTUNE_VERSION}/{FLASHINFER_AUTOTUNE_ARCH}"
)
FLASHINFER_WORKSPACE_BASE = (
    f"{PROJECT_VOLUME_ROOT}/glm53-flash-nota-flashinfer-jit-workspace"
)
FLASHINFER_JIT_CACHE = f"{FLASHINFER_WORKSPACE_BASE}/.cache/flashinfer"
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
        release_asset=f"flashinfer-autotune-{FLASHINFER_AUTOTUNE_VERSION}-b300.tar.gz",
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

MAX_B300_CONTAINERS = 1
MAX_CONCURRENT_INPUTS = 16
MAX_NUM_BATCHED_TOKENS = 8192
MTP_SPECULATIVE_TOKENS = 5
MTP_DECODE_TOKENS_PER_REQUEST = MTP_SPECULATIVE_TOKENS + 1
CUDAGRAPH_CAPTURE_SIZES = tuple(
    range(
        MTP_DECODE_TOKENS_PER_REQUEST,
        MTP_DECODE_TOKENS_PER_REQUEST * MAX_CONCURRENT_INPUTS + 1,
        MTP_DECODE_TOKENS_PER_REQUEST,
    )
)
MAX_CUDAGRAPH_CAPTURE_SIZE = CUDAGRAPH_CAPTURE_SIZES[-1]
SCALEDOWN_WINDOW_SECONDS = 1800
RUNTIME_CPU_MEMORY_MIB = 307200
STARTUP_TIMEOUT_SECONDS = 1800
RUNTIME_CACHE_SYNC_ENABLED = os.getenv(
    "GLM53_RUNTIME_CACHE_SYNC",
    "0",
).strip().lower() in {"1", "true", "yes", "on"}
PRODUCTION_SERVING_PROFILE_SHA256 = (
    "4e2c2ebecbb8b6f96a7827159d99ec7d5ddfdbc81c4b26ddef4bfddc61443fd2"
)

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


def _production_serving_profile() -> dict:
    """冻结已验证的 B300 serving 配置；修改后必须显式更新指纹。"""
    return {
        "model": MODEL,
        "revision": REVISION,
        "runtime_image": RUNTIME_IMAGE,
        "gpu": "B300",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.96,
        "max_num_seqs": MAX_CONCURRENT_INPUTS,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "kv_cache_dtype": "fp8",
        "mtp_speculative_tokens": MTP_SPECULATIVE_TOKENS,
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": list(CUDAGRAPH_CAPTURE_SIZES),
        "load_strategy": LOAD_STRATEGY,
        "prefetch_threads": PREFETCH_THREADS,
        "prefetch_block_mib": PREFETCH_BLOCK_MIB,
    }


def _validate_production_serving_profile() -> str:
    """阻止关键参数被无意修改，避免 FlashInfer cache key 静默漂移。"""
    serialized = json.dumps(
        _production_serving_profile(),
        sort_keys=True,
        separators=(",", ":"),
    )
    actual = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    if actual != PRODUCTION_SERVING_PROFILE_SHA256:
        raise RuntimeError(
            "Production serving profile changed. "
            "This may invalidate FlashInfer/runtime caches. "
            f"expected={PRODUCTION_SERVING_PROFILE_SHA256} actual={actual}"
        )
    return actual


PRODUCTION_SERVING_PROFILE_FINGERPRINT = _validate_production_serving_profile()


project_volume = modal.Volume.from_name(
    PROJECT_VOLUME_NAME,
    create_if_missing=True,
)
project_mount = {PROJECT_VOLUME_ROOT: project_volume}
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
    modal.Image.from_registry(RUNTIME_IMAGE, add_python="3.12")
    .entrypoint([])
    .add_local_file(
        "helpers/019_patch_vllm_prefetch.py",
        "/tmp/019_patch_vllm_prefetch.py",
        copy=True,
    )
    .add_local_file(
        "helpers/021_patch_startup_observability.py",
        "/tmp/021_patch_startup_observability.py",
        copy=True,
    )
    .run_commands(
        "python3 /tmp/019_patch_vllm_prefetch.py",
        "python3 -c \"from pathlib import Path; "
        "p=Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py'); "
        "s=p.read_text(); "
        "old='maybe_save_startup_plan(self, kv_cache_memory_bytes_to_requested_limit)'; "
        "new='maybe_save_startup_plan(self, int(self.available_kv_cache_memory_bytes))'; "
        "assert s.count(old)==1, f'unexpected startup-plan save call count: {s.count(old)}'; "
        "p.write_text(s.replace(old,new)); "
        "print('[IMAGE_PATCH] startup plan saves actual KV cache bytes')\"",
        f"{VLLM_RUNTIME_PYTHON} /tmp/021_patch_startup_observability.py",
    )
    .env(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "HF_HOME": HF_CACHE,
            "HF_HUB_OFFLINE": "1",
            "VLLM_SERVER_DEV_MODE": "0",
            "VLLM_CACHE_ROOT": VLLM_LOCAL_CACHE_ROOT,
            "VLLM_ENABLE_STARTUP_PLAN": "1",
            "GLM53_STARTUP_PLAN_FREE_MEMORY_TOLERANCE_MIB": os.getenv(
                "GLM53_STARTUP_PLAN_FREE_MEMORY_TOLERANCE_MIB", "256"
            ),
            "GLM53_PROFILE_IMPORTS": os.getenv("GLM53_PROFILE_IMPORTS", "0"),
            "GLM53_CACHE_DISASTER_RECOVERY": os.getenv("GLM53_CACHE_DISASTER_RECOVERY", "0"),
            "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR": FLASHINFER_AUTOTUNE_RUNTIME_CACHE,
            "FLASHINFER_WORKSPACE_BASE": FLASHINFER_WORKSPACE_BASE,
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
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


@app.function(image=runtime_image, cpu=2, memory=8192, timeout=300)
def inspect_startup_runtime(smoke_test: bool = False):
    """CPU-only check of the actual image contracts; does not start vLLM/GPU."""
    from pathlib import Path
    import subprocess

    capabilities = json.loads(Path(STARTUP_CAPABILITIES_PATH).read_text())
    subprocess.run(
        [VLLM_RUNTIME_PYTHON, "-c",
         "import helpers.startup_trace, helpers.startup_prefetch; "
         "import importlib.metadata as m; print('Runtime vLLM:', m.version('vllm'))"],
        check=True,
    )
    if smoke_test:
        production_argv = ["vllm", *_build_vllm_command("/tmp/glm53-model")[3:], "--help"]
        smoke_code = (
            "import runpy, sys; import vllm.platforms; "
            "from vllm.platforms.cpu import CpuPlatform; "
            "vllm.platforms.current_platform = CpuPlatform(); "
            f"sys.argv = {production_argv!r}; "
            "runpy.run_module('helpers.020_vllm_bootstrap', run_name='__main__')"
        )
        result = subprocess.run(
            # CUDA image cannot infer a device in this CPU-only function.
            # Set CPU platform only for the diagnostic --help subprocess, but
            # parse the exact production serving argv (including new flags).
            [VLLM_RUNTIME_PYTHON, "-c", smoke_code],
            capture_output=True, text=True, timeout=180,
        )
        for line in result.stdout.splitlines():
            if "[STARTUP_TRACE]" in line:
                print(line, flush=True)
        if result.returncode:
            raise RuntimeError(f"CLI smoke failed: {result.stderr[-5000:]}")
        capabilities["cli_help_cpu_platform_smoke"] = "passed"
    print(json.dumps(capabilities, indent=2), flush=True)
    return capabilities


@app.function(
    image=download_image,
    cpu=4,
    memory=65536,
    timeout=14400,
    secrets=[hf_secret],
    volumes=project_mount,
)
def step_001_download():
    """下载并缓存固定 revision 的模型权重；已完整缓存时直接返回。"""
    step_001_download_model(MODEL, REVISION, project_volume)


def _restore_cache_artifacts(artifacts, *, serving: bool = False) -> dict[str, str]:
    results: dict[str, str] = {}
    for artifact in artifacts:
        optional_jit = artifact.name == "flashinfer-jit"
        use_deployment = serving and os.getenv("GLM53_CACHE_DISASTER_RECOVERY", "0") != "1"
        source = step_009_restore_cache(
            artifact,
            github_repo=GITHUB_REPO,
            release_tag=CACHE_RELEASE_TAG,
            availability_dir=CACHE_AVAILABILITY_DIR if optional_jit else None,
            availability_scope=RUNTIME_IMAGE,
            remote_policy="deployment" if optional_jit and use_deployment else "refresh",
        )
        results[artifact.name] = source
        if source == "github":
            project_volume.commit()
            print(
                f"[009_CACHE_MODAL_COMMIT] name={artifact.name} source=github",
                flush=True,
            )
    return results


def _restore_runtime_caches() -> dict[str, str]:
    """CPU 部署阶段恢复全部 cache；B300 启动阶段只恢复非 staged cache。"""
    results = _restore_cache_artifacts(ALL_CACHE_ARTIFACTS)
    # Commit the inventory even when no release asset was restored (negative cache).
    project_volume.commit()
    print("[009_CACHE_AVAILABILITY_COMMIT] phase=deployment", flush=True)
    return results


def _spawn_cache_backup() -> None:
    """把 GitHub 备份交给同一 Modal App 的独立 CPU function；B300 不等待。"""
    try:
        backup_runtime_caches.spawn()
        print(
            "[CACHE_BACKUP_SPAWNED] "
            f"app={APP_NAME} function=backup_runtime_caches",
            flush=True,
        )
    except Exception as exc:
        print(
            "[CACHE_BACKUP_SPAWN_FAILED] "
            f"app={APP_NAME} error={type(exc).__name__}: {exc}",
            flush=True,
        )


def _selected_cache_artifacts(artifact_names: list[str] | None):
    if artifact_names is None:
        return ALL_CACHE_ARTIFACTS
    requested = set(artifact_names)
    known = {artifact.name for artifact in ALL_CACHE_ARTIFACTS}
    unknown = requested - known
    if unknown:
        raise ValueError(f"Unknown cache artifacts: {sorted(unknown)}")
    return tuple(
        artifact for artifact in ALL_CACHE_ARTIFACTS if artifact.name in requested
    )


def _backup_runtime_caches(
    artifact_names: list[str] | None = None,
    *,
    replace_names: list[str] | None = None,
    force: bool = False,
    ensure_existing: bool = False,
) -> dict[str, bool]:
    """CPU worker implementation shared by scheduled and forced backups."""
    replace_requested = set(replace_names or ())
    results: dict[str, bool] = {}
    for artifact in _selected_cache_artifacts(artifact_names):
        generic_dirty = is_cache_backup_dirty(artifact.local_path)
        staged_dirty = (
            artifact.name in STAGED_CACHE_NAMES
            and step_012_cache_dirty(artifact.local_path)
        )
        needs_backup = (
            force
            or ensure_existing
            or artifact.name in replace_requested
            or generic_dirty
            or staged_dirty
        )

        if not needs_backup:
            results[artifact.name] = False
            print(
                "[CACHE_BACKUP_SKIP_CLEAN] "
                f"name={artifact.name}",
                flush=True,
            )
            continue

        replace_existing = (
            force
            or artifact.name in replace_requested
            or generic_dirty
            or staged_dirty
        )
        published = step_010_publish_cache(
            artifact,
            github_repo=GITHUB_REPO,
            release_tag=CACHE_RELEASE_TAG,
            replace_existing=replace_existing,
        )
        results[artifact.name] = published
        if not published:
            continue

        marker_cleared = False
        if generic_dirty:
            clear_cache_backup_dirty(artifact.local_path)
            marker_cleared = True
        if staged_dirty:
            step_012_clear_dirty(artifact.local_path)
            marker_cleared = True
        if marker_cleared:
            project_volume.commit()
    return results


@app.function(
    image=cache_image,
    cpu=1,
    memory=1024,
    timeout=900,
    secrets=[github_secret],
    volumes=project_mount,
)
def step_009_restore_runtime_caches():
    """CPU 阶段：Volume 优先，缺失时从 GitHub Release 恢复运行时缓存。"""
    return _restore_runtime_caches()


@app.function(
    image=cache_image,
    cpu=2,
    memory=8192,
    timeout=3600,
    max_containers=1,
    min_containers=0,
    buffer_containers=0,
    single_use_containers=True,
    secrets=[github_secret],
    volumes=project_mount,
    schedule=modal.Period(hours=1),
)
def backup_runtime_caches(
    artifact_names: list[str] | None = None,
    replace_names: list[str] | None = None,
):
    """每小时只检查 dirty marker；全部 clean 时立即退出，不扫描/打包 cache。"""
    return _backup_runtime_caches(
        artifact_names,
        replace_names=replace_names,
        force=False,
        ensure_existing=False,
    )


@app.function(
    image=cache_image,
    cpu=2,
    memory=8192,
    timeout=3600,
    max_containers=1,
    scaledown_window=60,
    secrets=[github_secret],
    volumes=project_mount,
)
def backup_all_force():
    """部署阶段使用：强制刷新全部已就绪 cache 到 GitHub Release。"""
    return _backup_runtime_caches(force=True)


@app.function(
    image=cache_image,
    cpu=2,
    memory=8192,
    timeout=3600,
    max_containers=1,
    min_containers=0,
    buffer_containers=0,
    single_use_containers=True,
    secrets=[github_secret],
    volumes=project_mount,
)
def backup_missing_or_dirty():
    """部署阶段：补缺失 asset；dirty 才替换已有 asset。"""
    return _backup_runtime_caches(
        force=False,
        ensure_existing=True,
    )


@app.function(
    image=cache_image,
    cpu=2,
    memory=4096,
    timeout=1800,
    volumes=project_mount,
)
def step_013_compact_staged_archives():
    """CPU 一次性迁移：把 Triton/Inductor/CUDA 多文件 seed 收口为单归档。"""
    results: dict[str, bool] = {}
    for artifact in STAGED_CACHE_ARTIFACTS:
        if artifact.name not in ARCHIVED_STAGED_CACHE_NAMES:
            continue
        changed = step_013_compact_legacy_seed(
            artifact.local_path,
            name=artifact.name,
            archive_name=STAGED_CACHE_ARCHIVE_NAMES[artifact.name],
        )
        results[artifact.name] = changed
        project_volume.commit()
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
    volumes=project_mount,
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
    image=cache_image,
    cpu=1,
    memory=1024,
    timeout=2000,
)
def step_017_bench_generation():
    """手动长输出 benchmark；不会参与 VllmServer startup。"""
    return step_017_run_generation_benchmark(
        model=MODEL,
        base_url=PUBLIC_BASE_URL,
    )


def _stage_runtime_caches() -> None:
    """B300 只读取单归档；归档缺失时本地重建，绝不逐文件读取 Volume。"""
    for artifact in STAGED_CACHE_ARTIFACTS:
        runtime_path = STAGED_CACHE_RUNTIME_PATHS[artifact.name]
        if artifact.name in ARCHIVED_STAGED_CACHE_NAMES:
            archive_name = STAGED_CACHE_ARCHIVE_NAMES[artifact.name]
            staged = step_013_stage_archive(
                artifact.local_path,
                runtime_path,
                name=artifact.name,
                archive_name=archive_name,
            )
            if not staged:
                os.makedirs(runtime_path, exist_ok=True)
                print(
                    "[STAGED_CACHE_EMPTY] "
                    f"name={artifact.name} archive={archive_name} "
                    "reason=archive_missing action=regenerate_locally",
                    flush=True,
                )
            continue
        step_012_stage_cache(
            artifact.local_path,
            runtime_path,
            name=artifact.name,
        )


def _build_vllm_command(model_path: str) -> list[str]:
    """用本地 metadata mirror 启动 serving；权重 symlink 仍指向 HF Volume blob。"""
    strategy = os.environ.get(
        "GLM53_LOAD_STRATEGY",
        "prefetch",
    ).strip().lower()
    prefetch_threads = int(os.environ.get("GLM53_PREFETCH_THREADS", "16"))
    prefetch_block_mib = int(os.environ.get("GLM53_PREFETCH_BLOCK_MIB", "16"))

    command = [
        VLLM_RUNTIME_PYTHON,
        "-m",
        "helpers.020_vllm_bootstrap",
        "serve",
        model_path,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--served-model-name",
        MODEL,
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        "0.96",
        "--max-num-seqs",
        str(MAX_CONCURRENT_INPUTS),
        "--max-num-batched-tokens",
        str(MAX_NUM_BATCHED_TOKENS),
        "--compilation-config",
        json.dumps(
            {
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": CUDAGRAPH_CAPTURE_SIZES,
            }
        ),
        "--max-cudagraph-capture-size",
        str(MAX_CUDAGRAPH_CAPTURE_SIZE),
        "--kv-cache-dtype",
        "fp8",
        "--speculative-config",
        json.dumps(
            {
                "method": "mtp",
                "num_speculative_tokens": MTP_SPECULATIVE_TOKENS,
            }
        ),
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

    if "--enable-sleep-mode" in command:
        raise RuntimeError(
            "Production serving must not enable vLLM sleep mode"
        )
    print(
        "[SLEEP_MODE_DISABLED] "
        "enable_sleep_mode=False VLLM_SERVER_DEV_MODE=0",
        flush=True,
    )

    print(
        "[MODEL_LOAD] "
        f"model_path={model_path} "
        f"strategy={strategy} "
        f"prefetch_threads={prefetch_threads} "
        f"prefetch_block_mib={prefetch_block_mib}",
        flush=True,
    )
    return command


def _sync_runtime_caches(
    cache_sources: dict[str, str],
    cache_fingerprints_before: dict[str, tuple],
) -> None:
    """只持久化真正变化的 cache；GitHub 备份交给 CPU worker。"""
    any_changed = False

    for artifact in STAGED_CACHE_ARTIFACTS:
        runtime_path = STAGED_CACHE_RUNTIME_PATHS[artifact.name]
        if artifact.name in ARCHIVED_STAGED_CACHE_NAMES:
            changed = step_013_sync_archive(
                runtime_path,
                artifact.local_path,
                name=artifact.name,
                archive_name=STAGED_CACHE_ARCHIVE_NAMES[artifact.name],
            )
        else:
            step_012_sync_cache(
                runtime_path,
                artifact.local_path,
                name=artifact.name,
            )
            changed = step_012_cache_dirty(artifact.local_path)

        if changed:
            mark_cache_backup_dirty(artifact.local_path)
            project_volume.commit()
            any_changed = True
            print(
                f"[RUNTIME_CACHE_COMMIT] name={artifact.name} "
                f"path={artifact.local_path}",
                flush=True,
            )
        print(
            "[CACHE_VOLUME_SAFE] "
            f"name={artifact.name} changed={str(changed).lower()}",
            flush=True,
        )

    for artifact in RUNTIME_CACHE_ARTIFACTS:
        before = cache_fingerprints_before.get(artifact.name, ())
        after = cache_fingerprint(artifact)
        changed = before != after

        if changed:
            mark_cache_backup_dirty(artifact.local_path)
            project_volume.commit()
            any_changed = True
            print(
                f"[RUNTIME_CACHE_CHANGED] name={artifact.name} "
                f"before_files={len(before)} after_files={len(after)}",
                flush=True,
            )
            print(
                f"[RUNTIME_CACHE_COMMIT] name={artifact.name} "
                f"path={artifact.local_path}",
                flush=True,
            )
            print(
                "[CACHE_VOLUME_SAFE] "
                f"name={artifact.name} changed={str(changed).lower()}",
                flush=True,
            )

        print(
            "[CACHE_VOLUME_SAFE] "
            f"name={artifact.name} changed={str(changed).lower()} "
            f"source={cache_sources.get(artifact.name, 'unknown')}",
            flush=True,
        )

    if any_changed:
        _spawn_cache_backup()
    else:
        print("[CACHE_BACKUP_SKIP] reason=no_runtime_cache_changes", flush=True)


def _start_runtime_cache_sync(
    cache_sources: dict[str, str],
    cache_fingerprints_before: dict[str, tuple],
) -> threading.Thread:
    """API Ready 后后台持久化 cache，不阻塞对外服务。"""

    def run() -> None:
        started_at = time.perf_counter()
        print("[CACHE_SYNC_BACKGROUND_START]", flush=True)
        try:
            _sync_runtime_caches(
                cache_sources,
                cache_fingerprints_before,
            )
        except Exception as exc:
            print(
                "[CACHE_SYNC_BACKGROUND_FAILED] "
                f"error={type(exc).__name__}: {exc}",
                flush=True,
            )
            return

        print(
            "[CACHE_SYNC_BACKGROUND_DONE] "
            f"elapsed_s={time.perf_counter() - started_at:.3f}",
            flush=True,
        )

    thread = threading.Thread(
        target=run,
        name="runtime-cache-sync",
        daemon=True,
    )
    thread.start()
    return thread


def _start_startup_plan_commit(
    before: dict[str, str], modelinfo_seed: str,
    observer: startup_acceleration_module.StartupPlanObserver,
) -> threading.Thread:
    """Persist changed valid plans and modelinfos, regardless of older cached plans."""

    def run() -> None:
        started_at = time.perf_counter()
        try:
            changed = startup_acceleration_module.changed_startup_plans(
                before, VLLM_STARTUP_PLAN_CACHE,
            )
            modelinfo_changed = startup_acceleration_module.sync_modelinfo_cache(
                local_cache_root=VLLM_LOCAL_CACHE_ROOT,
                persistent_dir=modelinfo_seed,
            )
            actual_hit, applied = observer.result()
            print(
                "[018_STARTUP_PLAN_RESULT] "
                f"actual_hit={actual_hit} "
                f"applied_fingerprints={applied} "
                f"changed_valid_plans={changed}", flush=True,
            )
            if not changed and not modelinfo_changed:
                print("[018_STARTUP_METADATA_COMMIT_SKIP] reason=unchanged", flush=True)
                return
            print("[018_STARTUP_METADATA_COMMIT_START]", flush=True)
            project_volume.commit()
        except Exception as exc:
            print(
                "[018_STARTUP_METADATA_COMMIT_FAILED] "
                f"error={type(exc).__name__}: {exc}",
                flush=True,
            )
            return
        print(
            "[018_STARTUP_METADATA_COMMIT_DONE] "
            f"elapsed_s={time.perf_counter() - started_at:.3f}",
            flush=True,
        )

    thread = threading.Thread(
        target=run,
        name="startup-plan-volume-commit",
        daemon=True,
    )
    thread.start()
    return thread


def _start_flashinfer_cache_commit(before: tuple) -> threading.Thread:
    """Persist new tuning results even when bulk runtime cache sync is disabled."""
    artifact = next(a for a in RUNTIME_CACHE_ARTIFACTS if a.name == "flashinfer-autotune")

    def run() -> None:
        try:
            after = cache_fingerprint(artifact)
            if before == after:
                print("[FLASHINFER_CACHE_COMMIT_SKIP] reason=unchanged", flush=True)
                return
            mark_cache_backup_dirty(artifact.local_path)
            project_volume.commit()
            print("[FLASHINFER_CACHE_COMMIT_DONE] trigger=autotune_saved", flush=True)
            _spawn_cache_backup()
        except Exception as exc:
            print(f"[FLASHINFER_CACHE_COMMIT_FAILED] error={type(exc).__name__}: {exc}", flush=True)

    thread = threading.Thread(target=run, name="flashinfer-cache-commit", daemon=True)
    thread.start()
    return thread


@app.cls(
    image=runtime_image,
    gpu="B300",
    cpu=8,
    memory=RUNTIME_CPU_MEMORY_MIB,
    timeout=STARTUP_TIMEOUT_SECONDS,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    min_containers=0,
    max_containers=MAX_B300_CONTAINERS,
    buffer_containers=0,
    volumes=project_mount,
    secrets=[github_secret],
)
@modal.concurrent(max_inputs=MAX_CONCURRENT_INPUTS)
class VllmServer:
    """单 B300 vLLM；每个新容器直接从持久化 cache 正常启动。"""

    @modal.enter()
    def startup(self) -> None:
        """恢复 cache、初始化 vLLM、等待 API Ready，再进行轻量 warmup。"""
        startup_started_at = time.perf_counter()
        print(
            "[RUNTIME_START] "
            f"cpu_memory_mib={RUNTIME_CPU_MEMORY_MIB}",
            flush=True,
        )
        print(
            f"[PUBLIC_API] base_url={PUBLIC_BASE_URL} "
            f"openai_base_url={PUBLIC_OPENAI_BASE_URL}",
            flush=True,
        )

        cache_sources = _restore_cache_artifacts(RUNTIME_CACHE_ARTIFACTS, serving=True)
        flashinfer_before = cache_fingerprint(RUNTIME_CACHE_ARTIFACTS[0])
        self.flashinfer_cache_commit_threads = []

        def on_flashinfer_saved() -> None:
            self.flashinfer_cache_commit_threads.append(
                _start_flashinfer_cache_commit(flashinfer_before)
            )

        cache_fingerprints_before = (
            {
                artifact.name: cache_fingerprint(artifact)
                for artifact in RUNTIME_CACHE_ARTIFACTS
            }
            if RUNTIME_CACHE_SYNC_ENABLED
            else {}
        )
        _stage_runtime_caches()
        model_path = step_016_prepare_model_mirror(
            hf_cache=HF_CACHE,
            model=MODEL,
            revision=REVISION,
            destination=MODEL_RUNTIME_PATH,
        )

        startup_plan_before = step_018_prepare_startup_plan_cache(
            local_cache_root=VLLM_LOCAL_CACHE_ROOT,
            persistent_dir=VLLM_STARTUP_PLAN_CACHE,
        )
        modelinfo_seed = startup_acceleration_module.prepare_modelinfo_cache(
            local_cache_root=VLLM_LOCAL_CACHE_ROOT,
            persistent_root=VLLM_MODELINFO_CACHE,
            capabilities_path=STARTUP_CAPABILITIES_PATH,
        )
        startup_plan_observer = startup_acceleration_module.StartupPlanObserver()

        self.vllm_handle = step_003_start_vllm_with_model_init_observer(
            _build_vllm_command(model_path),
            extra_observer_factories=[
                lambda _: startup_plan_observer,
                step_004_create_kv_cache_observer,
                lambda started_at: step_005_create_kernel_jit_observer(
                    started_at, on_flashinfer_saved=on_flashinfer_saved,
                ),
                step_006_create_cuda_graph_observer,
            ],
        )

        ready_event = step_007_start_api_ready_observer(
            process=self.vllm_handle.process,
            process_started_at=self.vllm_handle.started_at,
            timeout_s=STARTUP_TIMEOUT_SECONDS,
        )

        if not ready_event.wait(STARTUP_TIMEOUT_SECONDS):
            raise TimeoutError("vLLM did not become ready")

        api_ready_at = time.perf_counter()
        self.startup_plan_commit_thread = _start_startup_plan_commit(
            startup_plan_before, modelinfo_seed, startup_plan_observer,
        )
        step_008_run_warmup(
            model=MODEL,
            process_started_at=self.vllm_handle.started_at,
            api_ready_at=api_ready_at,
            repeats=3,
        )
        if RUNTIME_CACHE_SYNC_ENABLED:
            self.cache_sync_thread = _start_runtime_cache_sync(
                cache_sources,
                cache_fingerprints_before,
            )
        else:
            self.cache_sync_thread = None
            print(
                "[CACHE_SYNC_SKIP] "
                "reason=stable_production_cache "
                "flashinfer_cache=persist_on_save "
                "github_backup=cpu_worker",
                flush=True,
            )
        print(
            "[RUNTIME_READY] "
            f"elapsed_s={time.perf_counter() - startup_started_at:.3f}",
            flush=True,
        )

    @modal.web_server(
        8000,
        startup_timeout=STARTUP_TIMEOUT_SECONDS,
    )
    def serve(self) -> None:
        """暴露已经初始化完成的 vLLM OpenAI-compatible server。"""
        pass

    @modal.exit()
    def shutdown(self) -> None:
        """容器退出时回收 vLLM 子进程。"""
        for thread in getattr(self, "flashinfer_cache_commit_threads", []):
            thread.join(timeout=30)
        startup_plan_commit_thread = getattr(
            self,
            "startup_plan_commit_thread",
            None,
        )
        if (
            startup_plan_commit_thread is not None
            and startup_plan_commit_thread.is_alive()
        ):
            startup_plan_commit_thread.join(timeout=30)

        handle = getattr(self, "vllm_handle", None)
        if handle is None:
            return

        process = handle.process
        if process.poll() is not None:
            return

        process.terminate()
        try:
            process.wait(timeout=30)
        except Exception:
            process.kill()
