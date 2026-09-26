import json
import time
import urllib.error
import urllib.request


def step_008_run_warmup(
    model: str,
    process_started_at: float,
    api_ready_at: float,
    base_url: str = "http://127.0.0.1:8000",
    timeout_s: float = 300,
) -> None:
    """API Ready 后执行一次最小真实生成，并记录首次 generation 请求耗时。"""
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1,
            "temperature": 0,
        }
    ).encode()
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    request_started_at = time.perf_counter()
    print(
        "[008_WARMUP_START] "
        f"endpoint={url} "
        f"after_api_ready_s={request_started_at - api_ready_at:.3f} "
        f"from_process_start_s="
        f"{request_started_at - process_started_at:.3f}",
        flush=True,
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout_s,
        ) as response:
            status = int(response.status)
            body = response.read()

        finished_at = time.perf_counter()
        result = json.loads(body)
        usage = result.get("usage") or {}
        choices = result.get("choices") or []

        finish_reason = "unknown"
        content_chars = 0
        if choices and isinstance(choices[0], dict):
            finish_reason = choices[0].get("finish_reason") or "unknown"
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                content_chars = len(content)

        print(
            "[008_WARMUP_DONE] "
            f"http_status={status} "
            f"warmup_request_s="
            f"{finished_at - request_started_at:.3f} "
            f"after_api_ready_s="
            f"{finished_at - api_ready_at:.3f} "
            f"from_process_start_s="
            f"{finished_at - process_started_at:.3f} "
            f"prompt_tokens={usage.get('prompt_tokens', 'unknown')} "
            f"completion_tokens="
            f"{usage.get('completion_tokens', 'unknown')} "
            f"total_tokens={usage.get('total_tokens', 'unknown')} "
            f"finish_reason={finish_reason} "
            f"content_chars={content_chars}",
            flush=True,
        )
    except urllib.error.HTTPError as exc:
        finished_at = time.perf_counter()
        body = exc.read(1024).decode("utf-8", errors="replace")
        print(
            "[008_WARMUP_FAILED] "
            f"reason=http_error "
            f"http_status={exc.code} "
            f"warmup_request_s="
            f"{finished_at - request_started_at:.3f} "
            f"from_process_start_s="
            f"{finished_at - process_started_at:.3f} "
            f"body={body!r}",
            flush=True,
        )
    except (
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
    ) as exc:
        finished_at = time.perf_counter()
        print(
            "[008_WARMUP_FAILED] "
            f"reason={type(exc).__name__} "
            f"warmup_request_s="
            f"{finished_at - request_started_at:.3f} "
            f"from_process_start_s="
            f"{finished_at - process_started_at:.3f} "
            f"error={exc!r}",
            flush=True,
        )
