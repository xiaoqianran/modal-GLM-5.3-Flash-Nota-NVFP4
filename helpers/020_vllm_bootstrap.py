"""Launch prefetch before importing the vLLM CLI (or torch)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path


def main() -> None:
    started = time.perf_counter()
    os.environ.setdefault("GLM53_PROCESS_STARTED_AT", str(started))
    from helpers.startup_trace import emit, span

    emit("start", "bootstrap")
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("command")
    parser.add_argument("model")
    parser.add_argument("--safetensors-load-strategy", default="default")
    parser.add_argument("--safetensors-prefetch-num-threads", type=int, default=16)
    parser.add_argument("--safetensors-prefetch-block-size", type=int, default=16 * 1024 * 1024)
    args, _ = parser.parse_known_args()
    # A new namespace per launch: stale markers cannot skip cold page-cache reads.
    os.environ["VLLM_PREFETCH_STATE_DIR"] = tempfile.mkdtemp(prefix="glm53-prefetch-")
    if args.safetensors_load_strategy == "prefetch":
        try:
            from helpers.startup_prefetch import start_prefetch

            root = Path(args.model)
            index = json.loads((root / "model.safetensors.index.json").read_text())
            files = [str(root / name) for name in set(index["weight_map"].values())]
            print(f"[EARLY_WEIGHT_PREFETCH_REQUEST] files={len(files)} from_bootstrap_start_s={time.perf_counter() - started:.3f}", flush=True)
            start_prefetch(files, num_prefetch_threads=args.safetensors_prefetch_num_threads,
                           block_size=args.safetensors_prefetch_block_size)
        except Exception as exc:
            print(f"[EARLY_WEIGHT_PREFETCH_FAILED] error={type(exc).__name__}: {exc}", flush=True)

    with span("vllm_cli_import"):
        from vllm.entrypoints.cli.main import main as vllm_main

    sys.argv[0] = "vllm"
    vllm_main()


if __name__ == "__main__":
    main()
