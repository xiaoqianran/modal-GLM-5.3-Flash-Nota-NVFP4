from __future__ import annotations

from pathlib import Path


SITE_PACKAGES = Path("/usr/local/lib/python3.12/dist-packages")
WEIGHT_UTILS = SITE_PACKAGES / "vllm/model_executor/model_loader/weight_utils.py"
SERVE_CLI = SITE_PACKAGES / "vllm/entrypoints/cli/serve.py"


def patch_weight_utils() -> None:
    marker = "_glm53_prefetch_coordination_paths"
    text = WEIGHT_UTILS.read_text(encoding="utf-8")
    if marker in text:
        print("[IMAGE_PATCH] vLLM coordinated prefetch already installed")
        return

    anchor = "def _prefetch_all_checkpoints(\n"
    if text.count(anchor) != 1:
        raise RuntimeError(
            "vLLM weight_utils shape changed: "
            f"expected one {anchor!r}, found {text.count(anchor)}"
        )

    helpers = r'''
def _glm53_prefetch_coordination_paths(
    sorted_files: list[str],
) -> tuple[filelock.FileLock, Path]:
    """Coordinate safetensors prefetch across API and EngineCore processes."""
    state_root = Path(os.environ.get("VLLM_PREFETCH_STATE_DIR", "/tmp/vllm-prefetch"))
    state_root.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256()
    for file_path in sorted_files:
        # Do not resolve/stat 9P-backed symlinks here. The coordination key must
        # be metadata-free; otherwise "early" prefetch can spend tens of seconds
        # walking remote metadata before the first read even starts.
        identity.update(os.path.abspath(file_path).encode("utf-8"))
        identity.update(b"\0")
    digest = identity.hexdigest()[:24]
    return (
        filelock.FileLock(
            str(state_root / f"{digest}.lock"),
            thread_local=False,
        ),
        state_root / f"{digest}.done",
    )


def start_early_safetensors_prefetch(
    model_path: str,
    *,
    num_prefetch_threads: int,
    block_size: int,
) -> None:
    """Start vLLM's own safetensors prefetch as soon as CLI args are available."""
    root = Path(model_path)
    index_path = root / SAFE_WEIGHTS_INDEX_NAME
    if not index_path.is_file():
        logger.info(
            "[EARLY_WEIGHT_PREFETCH_SKIP] no safetensors index at %s",
            index_path,
        )
        return
    with index_path.open(encoding="utf-8") as file:
        index = json.load(file)
    sorted_files = sorted(
        {str(root / name) for name in index["weight_map"].values()},
        key=_natural_sort_key,
    )
    logger.info(
        "[EARLY_WEIGHT_PREFETCH_REQUEST] files=%d threads=%d block_size=%d",
        len(sorted_files),
        num_prefetch_threads,
        block_size,
    )
    _prefetch_all_checkpoints(
        sorted_files,
        num_prefetch_threads=num_prefetch_threads,
        block_size=block_size,
    )


'''
    text = text.replace(anchor, helpers + anchor, 1)

    old = '''    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = 0
        world_size = 1
'''
    new = '''    coordination_lock, done_marker = _glm53_prefetch_coordination_paths(
        sorted_files
    )
    if done_marker.is_file():
        logger.info(
            "[WEIGHT_PREFETCH_REUSE] state=done files=%d",
            len(sorted_files),
        )
        return
    try:
        coordination_lock.acquire(timeout=0)
    except filelock.Timeout:
        logger.info(
            "[WEIGHT_PREFETCH_WAIT] state=inflight files=%d",
            len(sorted_files),
        )
        coordination_lock.acquire()
        try:
            if done_marker.is_file():
                logger.info(
                    "[WEIGHT_PREFETCH_REUSE] state=done-after-wait files=%d",
                    len(sorted_files),
                )
                return
            logger.warning(
                "[WEIGHT_PREFETCH_OWNER_FAILED] "
                "files=%d action=load-without-reprefetch",
                len(sorted_files),
            )
            return
        finally:
            coordination_lock.release()

    if done_marker.is_file():
        coordination_lock.release()
        logger.info(
            "[WEIGHT_PREFETCH_REUSE] state=done-after-lock files=%d",
            len(sorted_files),
        )
        return

    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = 0
        world_size = 1
'''
    if text.count(old) != 1:
        raise RuntimeError(
            "vLLM prefetch body changed: "
            f"expected one distributed-rank block, found {text.count(old)}"
        )
    text = text.replace(old, new, 1)

    old = '''    def _run_prefetch() -> None:
        start = time.perf_counter()
        asyncio.run(_prefetch_all())
        elapsed = time.perf_counter() - start
        logger.info(
            "Prefetching checkpoint files into page cache finished in %.2fs",
            elapsed,
        )
'''
    new = '''    def _run_prefetch() -> None:
        start = time.perf_counter()
        success = False
        try:
            asyncio.run(_prefetch_all())
            success = True
            elapsed = time.perf_counter() - start
            logger.info(
                "Prefetching checkpoint files into page cache finished in %.2fs",
                elapsed,
            )
            done_marker.touch()
            logger.info(
                "[WEIGHT_PREFETCH_DONE] files=%d elapsed_s=%.2f",
                len(sorted_files),
                elapsed,
            )
        finally:
            coordination_lock.release()
            if not success:
                logger.warning("[WEIGHT_PREFETCH_ABORTED] files=%d", len(sorted_files))
'''
    if text.count(old) != 1:
        raise RuntimeError(
            "vLLM prefetch runner changed: "
            f"expected one runner block, found {text.count(old)}"
        )
    text = text.replace(old, new, 1)
    WEIGHT_UTILS.write_text(text, encoding="utf-8")
    print("[IMAGE_PATCH] vLLM prefetch is cross-process coordinated")


def patch_serve_cli() -> None:
    marker = "EARLY_WEIGHT_PREFETCH_REQUEST"
    text = SERVE_CLI.read_text(encoding="utf-8")
    if marker in text:
        print("[IMAGE_PATCH] vLLM early CLI prefetch already installed")
        return

    old = '''        if hasattr(args, "model_tag") and args.model_tag is not None:
            args.model = args.model_tag

'''
    new = '''        if hasattr(args, "model_tag") and args.model_tag is not None:
            args.model = args.model_tag

        if getattr(args, "safetensors_load_strategy", None) == "prefetch":
            from vllm.model_executor.model_loader.weight_utils import (
                start_early_safetensors_prefetch,
            )

            start_early_safetensors_prefetch(
                args.model,
                num_prefetch_threads=args.safetensors_prefetch_num_threads,
                block_size=args.safetensors_prefetch_block_size,
            )

'''
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            "vLLM serve CLI shape changed: "
            f"expected one model_tag block, found {count}"
        )
    SERVE_CLI.write_text(text.replace(old, new, 1), encoding="utf-8")
    print("[IMAGE_PATCH] vLLM starts its single prefetch from ServeSubcommand.cmd")


def main() -> None:
    patch_weight_utils()
    patch_serve_cli()


if __name__ == "__main__":
    main()
