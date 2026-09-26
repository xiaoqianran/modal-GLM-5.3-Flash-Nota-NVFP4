import time


def step_001_download_model(model: str, revision: str, volume) -> None:
    """下载并缓存固定 revision 的模型权重；已完整缓存时直接返回。"""
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(
            model,
            revision=revision,
            local_files_only=True,
        )
        print("Model already cached.", flush=True)
        return
    except Exception:
        pass

    for attempt in range(1, 9):
        try:
            snapshot_download(model, revision=revision)
            volume.commit()
            print("Model download complete.", flush=True)
            return
        except Exception as exc:
            # 每次失败后提交已下载的数据，下一轮可以继续复用。
            volume.commit()
            print(
                f"Download attempt {attempt}/8 failed: {exc}",
                flush=True,
            )

            if attempt == 8:
                raise

            time.sleep(min(5 * 2 ** (attempt - 1), 60))
