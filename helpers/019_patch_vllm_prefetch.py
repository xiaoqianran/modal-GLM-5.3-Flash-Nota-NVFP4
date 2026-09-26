from __future__ import annotations

import ast
from pathlib import Path

SITE_PACKAGES = Path("/usr/local/lib/python3.12/dist-packages")
WEIGHT_UTILS = SITE_PACKAGES / "vllm/model_executor/model_loader/weight_utils.py"


def patch_weight_utils() -> None:
    text = WEIGHT_UTILS.read_text(encoding="utf-8")
    marker = "from helpers.startup_prefetch import start_prefetch"
    if marker in text:
        print("[IMAGE_PATCH] nonblocking prefetch already installed")
        return
    tree = ast.parse(text)
    matches = [node for node in tree.body
               if isinstance(node, ast.FunctionDef)
               and node.name == "_prefetch_all_checkpoints"]
    if len(matches) != 1:
        raise RuntimeError("vLLM prefetch function shape changed")
    node = matches[0]
    expected = ["sorted_files", "num_prefetch_threads", "block_size"]
    if [arg.arg for arg in node.args.args] != expected:
        raise RuntimeError("vLLM prefetch arguments changed")
    lines = text.splitlines(keepends=True)
    # Preserve the upstream signature and defaults; replace only the body.
    start = node.body[0].lineno - 1
    body = '''    """Reuse process-start prefetch without blocking the weight loader."""
    from helpers.startup_prefetch import start_prefetch

    start_prefetch(
        sorted_files,
        num_prefetch_threads=num_prefetch_threads,
        block_size=block_size,
    )
'''
    result = "".join(lines[:start]) + body + "".join(lines[node.end_lineno:])
    ast.parse(result)
    WEIGHT_UTILS.write_text(result, encoding="utf-8")
    print("[IMAGE_PATCH] vLLM reuses nonblocking process-start prefetch")


if __name__ == "__main__":
    patch_weight_utils()
