import re
import time


_DYNAMO_RE = re.compile(
    r"Dynamo bytecode transform time:\s*([0-9.]+)\s*s",
    re.IGNORECASE,
)
_GRAPH_COMPILE_RE = re.compile(
    r"Compiling a graph for\s+(.+?)\s+takes\s+([0-9.]+)\s*s",
    re.IGNORECASE,
)
_TORCH_COMPILE_TOTAL_RE = re.compile(
    r"torch\.compile takes\s+([0-9.]+)\s*s\s+in total",
    re.IGNORECASE,
)
_CACHE_DIR_RE = re.compile(
    r"Using cache directory:\s*(.+?)\s+for vLLM's torch\.compile",
    re.IGNORECASE,
)
_FLASHINFER_CACHE_FILE_RE = re.compile(
    r"Using FlashInfer autotune cache file:\s*(.+)",
    re.IGNORECASE,
)
_FLASHINFER_SAVE_RE = re.compile(
    r"Saved\s+(\d+)\s+configs.*?\((\d+)\s+new,\s+"
    r"(\d+)\s+from previous config\)",
    re.IGNORECASE,
)

_KERNEL_BACKENDS = {
    "tilelang": "TileLang",
    "flashinfer": "FlashInfer",
    "cutlass": "CUTLASS",
    "trt-llm": "TRT-LLM",
    "trtllm": "TRT-LLM",
    "fused moe": "Fused-MoE",
    "fused_moe": "Fused-MoE",
    "autotune": "Autotune",
    "inductor": "Inductor",
}


class _KernelJITObserver:
    """观测 torch.compile 与 kernel/JIT 相关日志，不改变编译配置。"""

    def __init__(self, process_started_at: float) -> None:
        self.process_started_at = process_started_at
        self.compile_started_at: float | None = None
        self.dynamo_s: float | None = None
        self.graph_compile_s: list[float] = []
        self.compile_cache_dir: str | None = None
        self.compile_cache_hit = False
        self.flashinfer_cache_file: str | None = None
        self.seen_kernel_events: set[str] = set()
        self.done = False

    def _mark_compile_start(self, now: float) -> None:
        if self.compile_started_at is not None:
            return

        self.compile_started_at = now
        print(
            "[005_JIT_START] "
            f"from_process_start_s={now - self.process_started_at:.3f}",
            flush=True,
        )

    def _observe_kernel_event(self, line: str, now: float) -> None:
        lowered = line.lower()

        for needle, label in _KERNEL_BACKENDS.items():
            if needle not in lowered or label in self.seen_kernel_events:
                continue

            self.seen_kernel_events.add(label)
            print(
                "[005_KERNEL_EVENT] "
                f"backend={label} "
                f"from_process_start_s={now - self.process_started_at:.3f}",
                flush=True,
            )

    def observe(self, line: str) -> None:
        """从 vLLM 原生日志提取 compile/JIT 的可归因时间与 cache 状态。"""
        now = time.perf_counter()
        lowered = line.lower()

        self._observe_kernel_event(line, now)

        flashinfer_cache_match = _FLASHINFER_CACHE_FILE_RE.search(line)
        if flashinfer_cache_match:
            self.flashinfer_cache_file = flashinfer_cache_match.group(1).strip()
            print(
                "[005_FLASHINFER_CACHE_FILE] "
                f"path={self.flashinfer_cache_file}",
                flush=True,
            )

        flashinfer_save_match = _FLASHINFER_SAVE_RE.search(line)
        if flashinfer_save_match:
            total_configs = int(flashinfer_save_match.group(1))
            new_configs = int(flashinfer_save_match.group(2))
            previous_configs = int(flashinfer_save_match.group(3))
            print(
                "[005_FLASHINFER_AUTOTUNE_DONE] "
                f"total_configs={total_configs} "
                f"new_configs={new_configs} "
                f"previous_configs={previous_configs} "
                f"actual_cache_hit={str(previous_configs > 0).lower()} "
                f"from_process_start_s={now - self.process_started_at:.3f}",
                flush=True,
            )

        cache_match = _CACHE_DIR_RE.search(line)
        if cache_match:
            self._mark_compile_start(now)
            self.compile_cache_dir = cache_match.group(1).strip()
            print(
                "[005_COMPILE_CACHE] "
                f"path={self.compile_cache_dir}",
                flush=True,
            )

        if "directly load the compiled graph" in lowered:
            self._mark_compile_start(now)
            if not self.compile_cache_hit:
                self.compile_cache_hit = True
                print(
                    "[005_COMPILE_CACHE_HIT] "
                    f"from_process_start_s="
                    f"{now - self.process_started_at:.3f}",
                    flush=True,
                )

        if "cache the graph" in lowered:
            self._mark_compile_start(now)

        dynamo_match = _DYNAMO_RE.search(line)
        if dynamo_match:
            self._mark_compile_start(now)
            self.dynamo_s = float(dynamo_match.group(1))
            print(
                "[005_DYNAMO_DONE] "
                f"vllm_reported_s={self.dynamo_s:.3f} "
                f"from_process_start_s={now - self.process_started_at:.3f}",
                flush=True,
            )

        graph_match = _GRAPH_COMPILE_RE.search(line)
        if graph_match:
            self._mark_compile_start(now)
            shape_kind = graph_match.group(1).strip().replace(" ", "_")
            graph_s = float(graph_match.group(2))
            self.graph_compile_s.append(graph_s)
            print(
                "[005_GRAPH_COMPILE_DONE] "
                f"shape={shape_kind} "
                f"vllm_reported_s={graph_s:.3f} "
                f"from_process_start_s={now - self.process_started_at:.3f}",
                flush=True,
            )

        total_match = _TORCH_COMPILE_TOTAL_RE.search(line)
        if total_match and not self.done:
            self._mark_compile_start(now)
            self.done = True
            total_s = float(total_match.group(1))

            fields = [
                "[005_JIT_DONE]",
                f"torch_compile_s={total_s:.3f}",
                f"dynamo_s={self.dynamo_s:.3f}"
                if self.dynamo_s is not None
                else "dynamo_s=unknown",
                f"graph_compile_sum_s={sum(self.graph_compile_s):.3f}",
                f"graph_compile_count={len(self.graph_compile_s)}",
                f"compile_cache_hit={str(self.compile_cache_hit).lower()}",
                f"from_process_start_s="
                f"{now - self.process_started_at:.3f}",
            ]
            if self.compile_started_at is not None:
                fields.append(
                    "observed_compile_window_s="
                    f"{now - self.compile_started_at:.3f}"
                )

            print(" ".join(fields), flush=True)


def step_005_create_kernel_jit_observer(
    process_started_at: float,
) -> _KernelJITObserver:
    """创建 005 Kernel/JIT observer，供唯一 vLLM 日志流统一调用。"""
    return _KernelJITObserver(process_started_at)
