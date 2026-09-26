import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from subprocess import Popen


def _probe_models_endpoint(
    url: str,
    timeout_s: float,
) -> tuple[int, int]:
    """请求 /v1/models，并返回 HTTP 状态码和模型数量。"""
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        status = int(response.status)
        payload = json.loads(response.read())

    models = payload.get("data", [])
    model_count = len(models) if isinstance(models, list) else 0
    return status, model_count


def step_007_start_api_ready_observer(
    process: Popen[str],
    process_started_at: float,
    base_url: str = "http://127.0.0.1:8000",
    timeout_s: float = 1800,
    poll_interval_s: float = 0.25,
    on_ready: Callable[[float], None] | None = None,
) -> threading.Event:
    """等待 /v1/models 首次成功，并记录 vLLM process start 到 API Ready 的时间。"""
    ready_event = threading.Event()
    models_url = base_url.rstrip("/") + "/v1/models"

    def watch() -> None:
        wait_started_at = time.perf_counter()
        deadline = wait_started_at + timeout_s
        attempts = 0

        print(
            "[007_API_READY_WAIT] "
            f"endpoint={models_url} "
            f"timeout_s={timeout_s:.1f}",
            flush=True,
        )

        while time.perf_counter() < deadline:
            attempts += 1

            return_code = process.poll()
            if return_code is not None:
                print(
                    "[007_API_READY_FAILED] "
                    f"reason=vllm_process_exited "
                    f"return_code={return_code} "
                    f"attempts={attempts} "
                    f"from_process_start_s="
                    f"{time.perf_counter() - process_started_at:.3f}",
                    flush=True,
                )
                return

            try:
                status, model_count = _probe_models_endpoint(
                    models_url,
                    timeout_s=2,
                )
                if status == 200:
                    now = time.perf_counter()
                    ready_event.set()
                    print(
                        "[007_API_READY] "
                        f"http_status={status} "
                        f"model_count={model_count} "
                        f"attempts={attempts} "
                        f"readiness_poll_s="
                        f"{now - wait_started_at:.3f} "
                        f"from_process_start_s="
                        f"{now - process_started_at:.3f}",
                        flush=True,
                    )

                    if on_ready is not None:
                        try:
                            on_ready(now)
                        except Exception as exc:
                            print(
                                "[007_ON_READY_FAILED] "
                                f"error={exc!r}",
                                flush=True,
                            )
                    return
            except (
                urllib.error.URLError,
                TimeoutError,
                json.JSONDecodeError,
            ):
                pass

            time.sleep(poll_interval_s)

        print(
            "[007_API_READY_TIMEOUT] "
            f"attempts={attempts} "
            f"timeout_s={timeout_s:.1f} "
            f"from_process_start_s="
            f"{time.perf_counter() - process_started_at:.3f}",
            flush=True,
        )

    threading.Thread(
        target=watch,
        name="step-007-api-ready-observer",
        daemon=True,
    ).start()

    return ready_event
