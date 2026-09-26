import re
import time


_AVAILABLE_MEMORY_RE = re.compile(
    r"Available KV cache memory:\s*(-?[0-9.]+)\s+GiB",
    re.IGNORECASE,
)
_KV_CACHE_SIZE_RE = re.compile(
    r"GPU KV cache size:\s*([0-9,]+)\s+tokens",
    re.IGNORECASE,
)
_MAX_CONCURRENCY_RE = re.compile(
    r"Maximum concurrency for\s*([0-9,]+)\s+tokens per request:\s*"
    r"([0-9.]+)x",
    re.IGNORECASE,
)


class _KVCacheObserver:
    """观测 vLLM 的 KV Cache 规划结果，不改变任何缓存配置。"""

    def __init__(self, process_started_at: float) -> None:
        self.process_started_at = process_started_at
        self.planning_started_at: float | None = None
        self.available_memory_gib: float | None = None
        self.cache_tokens: int | None = None
        self.max_model_len: int | None = None
        self.max_concurrency: float | None = None

    def observe(self, line: str) -> None:
        """从 vLLM 原生日志中提取 KV Cache 容量与并发信息。"""
        now = time.perf_counter()

        available_match = _AVAILABLE_MEMORY_RE.search(line)
        if available_match and self.available_memory_gib is None:
            self.planning_started_at = now
            self.available_memory_gib = float(available_match.group(1))
            print(
                "[004_KV_MEMORY_READY] "
                f"available_gib={self.available_memory_gib:.3f} "
                f"from_process_start_s={now - self.process_started_at:.3f}",
                flush=True,
            )

        size_match = _KV_CACHE_SIZE_RE.search(line)
        if size_match and self.cache_tokens is None:
            self.cache_tokens = int(size_match.group(1).replace(",", ""))
            fields = [
                "[004_KV_CACHE_SIZE]",
                f"tokens={self.cache_tokens}",
                f"from_process_start_s={now - self.process_started_at:.3f}",
            ]
            if self.planning_started_at is not None:
                fields.append(
                    "planning_after_memory_ready_s="
                    f"{now - self.planning_started_at:.3f}"
                )
            print(" ".join(fields), flush=True)

        concurrency_match = _MAX_CONCURRENCY_RE.search(line)
        if concurrency_match and self.max_concurrency is None:
            self.max_model_len = int(
                concurrency_match.group(1).replace(",", "")
            )
            self.max_concurrency = float(concurrency_match.group(2))

            fields = [
                "[004_KV_CACHE_DONE]",
                f"available_gib={self.available_memory_gib:.3f}"
                if self.available_memory_gib is not None
                else "available_gib=unknown",
                f"tokens={self.cache_tokens}"
                if self.cache_tokens is not None
                else "tokens=unknown",
                f"max_model_len={self.max_model_len}",
                f"max_concurrency={self.max_concurrency:.2f}",
                f"from_process_start_s={now - self.process_started_at:.3f}",
            ]
            if self.planning_started_at is not None:
                fields.append(
                    "planning_after_memory_ready_s="
                    f"{now - self.planning_started_at:.3f}"
                )
            print(" ".join(fields), flush=True)


def step_004_create_kv_cache_observer(
    process_started_at: float,
) -> _KVCacheObserver:
    """创建 004 KV Cache observer，供 003 的唯一日志流统一调用。"""
    return _KVCacheObserver(process_started_at)
