import re
import time


_CAPTURE_PROGRESS_RE = re.compile(
    r"Capturing CUDA graph(?:s| shapes)?"
    r"(?:\s*\((.*?)\))?:.*?"
    r"([0-9,]+)/([0-9,]+)",
    re.IGNORECASE,
)
_CAPTURE_DONE_RE = re.compile(
    r"Graph capturing finished in\s+([0-9.]+)\s+secs?"
    r"(?:,\s*took\s+([0-9.]+)\s+GiB)?",
    re.IGNORECASE,
)
_POOL_MEMORY_RE = re.compile(
    r"CUDA graph pool memory:\s*"
    r"([0-9.]+)\s+GiB\s*\(actual\),\s*"
    r"([0-9.]+)\s+GiB\s*\(estimated\)"
    r"(?:,\s*difference:\s*([0-9.]+)\s+GiB"
    r"\s*\(([0-9.]+)%\))?",
    re.IGNORECASE,
)
_MODE_RE = re.compile(
    r"cudagraph_mode(?:=|['\": ]+)\s*['\"]?([A-Z_]+)",
    re.IGNORECASE,
)
_MAX_CAPTURE_SIZE_RE = re.compile(
    r"max_cudagraph_capture_size(?:=|['\": ]+)\s*([0-9,]+)",
    re.IGNORECASE,
)

_DISABLED_MARKERS = (
    "cuda graph is disabled",
    "cuda graphs are disabled",
    "cudagraph_mode=none",
    "cudagraph_mode': 'none",
    'cudagraph_mode": "none',
    "enforce_eager=true",
    "enforce_eager': true",
    'enforce_eager": true',
)


class _CUDAGraphObserver:
    """观测 CUDA Graph capture，不改变 vLLM 的 graph 配置。"""

    def __init__(self, process_started_at: float) -> None:
        self.process_started_at = process_started_at
        self.config_mode: str | None = None
        self.max_capture_size: int | None = None
        self.capture_seen = False
        self.capture_done = False
        self.disabled = False
        self.capture_phases: dict[str, int] = {}
        self.reported_capture_s: float | None = None
        self.reported_memory_gib: float | None = None

    @staticmethod
    def _normalize_phase(raw_phase: str | None) -> str:
        if not raw_phase:
            return "unspecified"
        return (
            raw_phase.strip()
            .replace(", ", "_")
            .replace(" ", "_")
        )

    def _observe_config(self, line: str) -> None:
        if self.config_mode is None:
            mode_match = _MODE_RE.search(line)
            if mode_match:
                self.config_mode = mode_match.group(1).upper()
                print(
                    "[006_CUDAGRAPH_CONFIG] "
                    f"mode={self.config_mode}",
                    flush=True,
                )

        if self.max_capture_size is None:
            size_match = _MAX_CAPTURE_SIZE_RE.search(line)
            if size_match:
                self.max_capture_size = int(
                    size_match.group(1).replace(",", "")
                )
                print(
                    "[006_CUDAGRAPH_MAX_CAPTURE] "
                    f"size={self.max_capture_size}",
                    flush=True,
                )

    def _observe_disabled(self, line: str, now: float) -> None:
        if self.disabled or self.capture_seen:
            return

        lowered = line.lower()
        if not any(marker in lowered for marker in _DISABLED_MARKERS):
            return

        self.disabled = True
        print(
            "[006_CUDAGRAPH_DISABLED] "
            f"from_process_start_s={now - self.process_started_at:.3f}",
            flush=True,
        )

    def observe(self, line: str) -> None:
        """提取 CUDA Graph 模式、capture phases、耗时和显存占用。"""
        now = time.perf_counter()
        self._observe_config(line)
        self._observe_disabled(line, now)

        progress_match = _CAPTURE_PROGRESS_RE.search(line)
        if progress_match:
            phase = self._normalize_phase(progress_match.group(1))
            captured = int(progress_match.group(2).replace(",", ""))
            total = int(progress_match.group(3).replace(",", ""))

            if not self.capture_seen:
                self.capture_seen = True
                print(
                    "[006_CUDAGRAPH_CAPTURE_OBSERVED] "
                    f"from_process_start_s="
                    f"{now - self.process_started_at:.3f}",
                    flush=True,
                )

            previous_total = self.capture_phases.get(phase)
            if previous_total != total:
                self.capture_phases[phase] = total
                print(
                    "[006_CUDAGRAPH_PHASE] "
                    f"phase={phase} "
                    f"captured={captured} "
                    f"total={total}",
                    flush=True,
                )

        done_match = _CAPTURE_DONE_RE.search(line)
        if done_match and not self.capture_done:
            self.capture_seen = True
            self.capture_done = True
            self.reported_capture_s = float(done_match.group(1))
            if done_match.group(2) is not None:
                self.reported_memory_gib = float(done_match.group(2))

            fields = [
                "[006_CUDAGRAPH_DONE]",
                f"mode={self.config_mode or 'unknown'}",
                f"capture_s={self.reported_capture_s:.3f}",
                (
                    f"memory_gib={self.reported_memory_gib:.3f}"
                    if self.reported_memory_gib is not None
                    else "memory_gib=unknown"
                ),
                f"phase_count={len(self.capture_phases)}",
                f"capture_shapes_total={sum(self.capture_phases.values())}",
                (
                    f"max_capture_size={self.max_capture_size}"
                    if self.max_capture_size is not None
                    else "max_capture_size=unknown"
                ),
                f"from_process_start_s="
                f"{now - self.process_started_at:.3f}",
            ]
            print(" ".join(fields), flush=True)

        pool_match = _POOL_MEMORY_RE.search(line)
        if pool_match:
            actual_gib = float(pool_match.group(1))
            estimated_gib = float(pool_match.group(2))
            fields = [
                "[006_CUDAGRAPH_MEMORY]",
                f"actual_gib={actual_gib:.3f}",
                f"estimated_gib={estimated_gib:.3f}",
            ]
            if pool_match.group(3) is not None:
                fields.append(
                    f"difference_gib={float(pool_match.group(3)):.3f}"
                )
            if pool_match.group(4) is not None:
                fields.append(
                    f"difference_pct={float(pool_match.group(4)):.3f}"
                )
            print(" ".join(fields), flush=True)


def step_006_create_cuda_graph_observer(
    process_started_at: float,
) -> _CUDAGraphObserver:
    """创建 006 CUDA Graph observer，供唯一 vLLM 日志流统一调用。"""
    return _CUDAGraphObserver(process_started_at)
