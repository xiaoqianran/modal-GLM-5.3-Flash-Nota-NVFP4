import re
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


class _LineObserver(Protocol):
    def observe(self, line: str) -> None: ...


@dataclass(frozen=True)
class VLLMProcessHandle:
    """保存唯一 vLLM 子进程及其精确启动计时起点。"""

    process: subprocess.Popen[str]
    started_at: float


_WEIGHT_SECONDS_RE = re.compile(
    r"Loading weights took\s+([0-9.]+)\s+seconds",
    re.IGNORECASE,
)
_MODEL_LOADING_RE = re.compile(
    r"Model loading took\s+([0-9.]+)\s+GiB(?:\s+memory)?\s+and\s+([0-9.]+)\s+seconds",
    re.IGNORECASE,
)


class _ModelInitObserver:
    """只观测模型初始化阶段，不改变 vLLM 的加载或推理行为。"""

    def __init__(self, process_started_at: float) -> None:
        self.process_started_at = process_started_at
        self.model_started_at: float | None = None
        self.weight_started_at: float | None = None
        self.weight_done_at: float | None = None

    def observe(self, line: str) -> None:
        """识别 vLLM 原生日志，并输出 003 阶段的结构化计时。"""
        now = time.perf_counter()

        if (
            self.model_started_at is None
            and "Starting to load model" in line
        ):
            self.model_started_at = now
            print(
                "[003_MODEL_INIT_START] "
                f"from_process_start_s={now - self.process_started_at:.3f}",
                flush=True,
            )

        if (
            self.weight_started_at is None
            and (
                "Loading safetensors checkpoint shards" in line
                or "Using model weights format" in line
            )
        ):
            self.weight_started_at = now
            print(
                "[003_WEIGHT_LOAD_START] "
                f"from_process_start_s={now - self.process_started_at:.3f}",
                flush=True,
            )

        weight_match = _WEIGHT_SECONDS_RE.search(line)
        if weight_match and self.weight_done_at is None:
            self.weight_done_at = now
            fields = [
                "[003_WEIGHT_LOAD_DONE]",
                f"vllm_reported_s={float(weight_match.group(1)):.3f}",
                f"from_process_start_s={now - self.process_started_at:.3f}",
            ]
            if self.weight_started_at is not None:
                fields.append(
                    "observed_weight_phase_s="
                    f"{now - self.weight_started_at:.3f}"
                )
            print(" ".join(fields), flush=True)

        model_match = _MODEL_LOADING_RE.search(line)
        if model_match:
            memory_gib = float(model_match.group(1))
            vllm_reported_s = float(model_match.group(2))
            fields = [
                "[003_MODEL_INIT_DONE]",
                f"model_memory_gib={memory_gib:.3f}",
                f"vllm_reported_s={vllm_reported_s:.3f}",
                f"from_process_start_s={now - self.process_started_at:.3f}",
            ]
            if self.model_started_at is not None:
                fields.append(
                    "observed_model_init_s="
                    f"{now - self.model_started_at:.3f}"
                )
            if self.weight_done_at is not None:
                fields.append(
                    "post_weight_finalize_s="
                    f"{now - self.weight_done_at:.3f}"
                )
            print(" ".join(fields), flush=True)


def _stream_vllm_output(
    process: subprocess.Popen[str],
    observers: list[_LineObserver],
) -> None:
    """持续转发唯一的 vLLM 日志流，并依次交给各阶段 observer。"""
    if process.stdout is None:
        return

    for line in iter(process.stdout.readline, ""):
        print(line, end="", flush=True)
        for observer in observers:
            observer.observe(line)


def step_003_start_vllm_with_model_init_observer(
    command: list[str],
    extra_observer_factories: list[
        Callable[[float], _LineObserver]
    ] | None = None,
) -> VLLMProcessHandle:
    """启动唯一的 vLLM 进程，并把同一日志流分发给后续阶段 observer。"""
    process_started_at = time.perf_counter()
    print("[003_VLLM_PROCESS_START]", flush=True)

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    observers: list[_LineObserver] = [
        _ModelInitObserver(process_started_at),
    ]
    for factory in extra_observer_factories or []:
        observers.append(factory(process_started_at))

    threading.Thread(
        target=_stream_vllm_output,
        args=(process, observers),
        name="step-003-model-init-observer",
        daemon=True,
    ).start()

    return VLLMProcessHandle(
        process=process,
        started_at=process_started_at,
    )
