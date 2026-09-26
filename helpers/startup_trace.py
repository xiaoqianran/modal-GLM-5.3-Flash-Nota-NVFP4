"""Low-overhead startup spans. No torch/vLLM imports or changed load decisions."""
from __future__ import annotations

import contextlib
import contextvars
import functools
import json
import os
import time


_inspection = contextvars.ContextVar("glm53_inspection", default=None)


def emit(event: str, phase: str, **fields) -> None:
    origin = float(os.environ.get("GLM53_PROCESS_STARTED_AT", time.perf_counter()))
    payload = {"event": event, "phase": phase, "pid": os.getpid(),
               "from_process_start_s": round(time.perf_counter() - origin, 6), **fields}
    print("[STARTUP_TRACE] " + json.dumps(payload, sort_keys=True), flush=True)


@contextlib.contextmanager
def span(phase: str, **fields):
    started, cpu = time.perf_counter(), time.thread_time()
    emit("start", phase, **fields)
    status = "ok"
    try:
        yield
    except BaseException:
        status = "error"
        raise
    finally:
        emit("end", phase, elapsed_s=round(time.perf_counter() - started, 6),
             thread_cpu_s=round(time.thread_time() - cpu, 6), status=status, **fields)


def trace_call(phase: str):
    def decorate(function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            model_class = getattr(args[0], "class_name", None) if args else None
            token = _inspection.set(model_class) if model_class else None
            fields = {}
            if _inspection.get():
                fields["model_class"] = _inspection.get()
            if phase == "engine_entry":
                spawn = os.environ.get("GLM53_ENGINE_SPAWN_AT")
                if spawn is not None:
                    fields["spawn_to_entry_s"] = round(time.perf_counter() - float(spawn), 6)
            try:
                with span(phase, **fields):
                    result = function(*args, **kwargs)
                    if phase == "registry_cache_lookup":
                        emit("result", phase, actual_hit=result is not None, **fields)
                    return result
            finally:
                if token is not None:
                    _inspection.reset(token)
        return wrapped
    return decorate


def start_engine_process(process) -> None:
    previous = os.environ.get("GLM53_ENGINE_SPAWN_AT")
    os.environ["GLM53_ENGINE_SPAWN_AT"] = str(time.perf_counter())
    try:
        with span("engine_process_start", process_name=process.name):
            process.start()
        emit("spawned", "engine_process_start", child_pid=process.pid)
    finally:
        if previous is None:
            os.environ.pop("GLM53_ENGINE_SPAWN_AT", None)
        else:
            os.environ["GLM53_ENGINE_SPAWN_AT"] = previous


def forward_registry_trace(completed) -> None:
    """Registry captures child stdout/stderr; forward only our profiling records."""
    for data in (completed.stdout, completed.stderr):
        text = data.decode(errors="replace") if isinstance(data, bytes) else (data or "")
        for line in text.splitlines():
            if line.startswith("[STARTUP_TRACE]"):
                print(line, flush=True)
            elif line.startswith("import time:"):
                print("[REGISTRY_IMPORT_TIME] " + line, flush=True)
