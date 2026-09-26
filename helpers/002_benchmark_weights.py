import subprocess


def step_002_benchmark_weights(
    hf_cache: str,
    revision: str,
    strategy: str = "prefetch",
    threads: int = 16,
    block_mib: int = 16,
) -> None:
    """仅测试 386 个 safetensors shard 的读取路径，不启动 vLLM 或 CUDA Graph。"""
    strategy = strategy.strip().lower()

    if strategy not in {"prefetch", "eager"}:
        raise ValueError("strategy must be 'prefetch' or 'eager'")
    if threads < 1:
        raise ValueError("threads must be >= 1")
    if block_mib < 1:
        raise ValueError("block_mib must be >= 1")

    snapshot = (
        f"{hf_cache}/hub/models--nota-ai--GLM-5.3-Flash-Nota-NVFP4/"
        f"snapshots/{revision}"
    )

    # 使用镜像原生 /usr/bin/python3，确保 eager 路径使用与 vLLM 相同的 safetensors 环境。
    benchmark_script = r"""
import concurrent.futures
import json
import sys
import time
from pathlib import Path


strategy = sys.argv[1]
threads = int(sys.argv[2])
block_size = int(sys.argv[3])
snapshot = Path(sys.argv[4])

index_path = snapshot / "model.safetensors.index.json"
index = json.loads(index_path.read_text(encoding="utf-8"))
files = [
    snapshot / name
    for name in sorted(set(index["weight_map"].values()))
]
total_bytes = sum(path.stat().st_size for path in files)

print(
    "[WEIGHT_BENCH] "
    f"strategy={strategy} "
    f"threads={threads} "
    f"block_size={block_size} "
    f"files={len(files)} "
    f"total_gib={total_bytes / 1024**3:.2f}",
    flush=True,
)


def read_file(path):
    '''按固定块大小顺序读取单个 shard，用于模拟 vLLM prefetch。'''
    read_bytes = 0

    with open(path, "rb") as file:
        while True:
            chunk = file.read(block_size)
            if not chunk:
                return read_bytes
            read_bytes += len(chunk)


started = time.perf_counter()
read_bytes = 0
tensor_count = 0
next_progress = 10

if strategy == "prefetch":
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(read_file, path) for path in files]

        for completed, future in enumerate(
            concurrent.futures.as_completed(futures),
            start=1,
        ):
            read_bytes += future.result()
            progress = completed * 100 / len(files)

            if progress >= next_progress:
                print(
                    "[WEIGHT_BENCH_PROGRESS] "
                    f"strategy={strategy} "
                    f"threads={threads} "
                    f"pct={next_progress} "
                    f"files={completed}/{len(files)} "
                    f"elapsed_s={time.perf_counter() - started:.3f}",
                    flush=True,
                )
                next_progress += 10

else:
    from safetensors.torch import load

    for index, path in enumerate(files, start=1):
        with open(path, "rb") as file:
            blob = file.read()

        read_bytes += len(blob)
        state = load(blob)
        tensor_count += len(state)
        del state, blob

        progress = index * 100 / len(files)
        if progress >= next_progress:
            print(
                "[WEIGHT_BENCH_PROGRESS] "
                f"strategy={strategy} "
                f"threads={threads} "
                f"pct={next_progress} "
                f"files={index}/{len(files)} "
                f"elapsed_s={time.perf_counter() - started:.3f}",
                flush=True,
            )
            next_progress += 10

elapsed = time.perf_counter() - started

print(
    "[WEIGHT_BENCH_RESULT] "
    f"strategy={strategy} "
    f"threads={threads} "
    f"files={len(files)} "
    f"bytes={read_bytes} "
    f"tensors={tensor_count} "
    f"elapsed_s={elapsed:.3f} "
    f"gib_per_s={read_bytes / 1024**3 / elapsed:.3f}",
    flush=True,
)
"""

    subprocess.run(
        [
            "/usr/bin/python3",
            "-c",
            benchmark_script,
            strategy,
            str(threads),
            str(block_mib * 1024 * 1024),
            snapshot,
        ],
        check=True,
    )
