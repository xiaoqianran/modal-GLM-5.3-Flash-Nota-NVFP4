import json
import time
import urllib.error
import urllib.request


WARMUP_REPEATS = 3
WARMUP_MAX_TOKENS = 16


def step_008_run_warmup(
    model: str,
    process_started_at: float,
    api_ready_at: float,
    base_url: str = "http://127.0.0.1:8000",
    timeout_s: float = 300,
    repeats: int = WARMUP_REPEATS,
) -> None:
    """生产 warmup：固定 16 tokens，只触发必要的 JIT/CUDA Graph 热身。"""
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "max_tokens": WARMUP_MAX_TOKENS,
            "temperature": 0,
        }
    ).encode("utf-8")

    for index in range(1, repeats + 1):
        request = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        started_at = time.perf_counter()
        print(
            "[008_WARMUP_START] "
            f"case={index}/{repeats} max_tokens={WARMUP_MAX_TOKENS} "
            f"after_api_ready_s={started_at - api_ready_at:.3f} "
            f"from_process_start_s={started_at - process_started_at:.3f}",
            flush=True,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                status = int(response.status)
                body = response.read()
            finished_at = time.perf_counter()
            result = json.loads(body)
            usage = result.get("usage") or {}
            print(
                "[008_WARMUP_DONE] "
                f"case={index}/{repeats} http_status={status} "
                f"request_s={finished_at - started_at:.3f} "
                f"completion_tokens={usage.get('completion_tokens', 'unknown')} "
                f"from_process_start_s={finished_at - process_started_at:.3f}",
                flush=True,
            )
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
        ) as exc:
            finished_at = time.perf_counter()
            print(
                "[008_WARMUP_FAILED] "
                f"case={index}/{repeats} reason={type(exc).__name__} "
                f"request_s={finished_at - started_at:.3f} error={exc!r}",
                flush=True,
            )
            raise
