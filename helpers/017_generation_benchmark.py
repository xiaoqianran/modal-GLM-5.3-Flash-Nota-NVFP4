from __future__ import annotations

import json
import time
import urllib.request


BENCHMARK_MAX_TOKENS = 8092
BENCHMARK_TIMEOUT_S = 900
BENCHMARK_CASES = (
    (
        "logic",
        "有12枚外观相同的硬币，其中恰有1枚是假币，但不知道它比真币重还是轻。"
        "你只有一架无砝码天平和最多3次称量机会。请给出一个完整、可执行的自适应决策策略，"
        "保证在3次称量内找出假币并判断它偏重还是偏轻。",
    ),
    (
        "systems",
        "设计一个面向全球多区域部署的超大模型在线推理平台。请系统分析请求路由、GPU调度、"
        "KV/前缀/编译缓存、弹性扩缩容、故障恢复、SLO、跨区域一致性与成本控制，并说明这些目标"
        "之间的关键权衡以及你会如何做工程取舍。",
    ),
)


def step_017_run_generation_benchmark(
    *,
    model: str,
    base_url: str,
    timeout_s: float = BENCHMARK_TIMEOUT_S,
) -> list[dict[str, object]]:
    """显式手动运行 2×8092 长输出 benchmark；绝不由生产 startup 自动调用。"""
    url = base_url.rstrip("/") + "/v1/chat/completions"
    results: list[dict[str, object]] = []

    for name, prompt in BENCHMARK_CASES:
        payload = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": BENCHMARK_MAX_TOKENS,
                "temperature": 0,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        started_at = time.perf_counter()
        print(
            f"[017_BENCHMARK_START] case={name} max_tokens={BENCHMARK_MAX_TOKENS}",
            flush=True,
        )
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            status = int(response.status)
            body = response.read()
        elapsed_s = time.perf_counter() - started_at
        result = json.loads(body)
        usage = result.get("usage") or {}
        completion_tokens = int(usage.get("completion_tokens") or 0)
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        finish_reason = (
            (result.get("choices") or [{}])[0].get("finish_reason") or "unknown"
        )
        approx_tps = completion_tokens / elapsed_s if elapsed_s > 0 else 0.0
        row = {
            "case": name,
            "http_status": status,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "request_s": elapsed_s,
            "approx_completion_tps": approx_tps,
            "finish_reason": finish_reason,
        }
        results.append(row)
        print(
            "[017_BENCHMARK_DONE] "
            f"case={name} http_status={status} prompt_tokens={prompt_tokens} "
            f"completion_tokens={completion_tokens} request_s={elapsed_s:.3f} "
            f"approx_completion_tps={approx_tps:.3f} finish_reason={finish_reason}",
            flush=True,
        )

    return results
