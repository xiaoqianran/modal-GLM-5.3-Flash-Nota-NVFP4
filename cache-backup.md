# Runtime cache backup

## 目标

- Modal Volume 是运行时主缓存。
- GitHub Release 是可移植备份 / fallback。
- B300 只负责生成缓存并 commit 到 Volume，不等待 GitHub 上传。
- GitHub 上传由独立 CPU App `glm53-cache-backup` 完成。

## 运行链路

```text
B300 warmup
  -> cache sync
  -> .github-backup-dirty
  -> Modal Volume commit
  -> CACHE_VOLUME_SAFE
  -> spawn glm53-cache-backup
  -> B300 可立即停止

glm53-cache-backup
  -> 读取 Modal Volume
  -> 打包 cache
  -> 安全替换 GitHub Release asset
  -> 清理 dirty marker
```

backup app 不设置定时任务，空闲时没有 CPU Function 执行。

如果 B300 在 Volume commit 后、spawn 前被强制停止，dirty marker 会继续保留在 Modal Volume。下一次部署、启动或显式 backup 调用会重新检查并补传，不需要小时级轮询。

## GitHub Release

- Repository: `xiaoqianran/modal-GLM-5.3-Flash-Nota-NVFP4`
- Tag: `cache-b300-glm53-flash-nota-v1`

当前可备份缓存：

- FlashInfer autotune
- TileLang
- Triton
- TorchInductor
- CUDA Compute
- FlashInfer JIT：只有出现有效 `.so` / `.o` / `.cubin` 后才上传

## 安全替换

已有 asset 更新时：

1. 先上传新的临时 asset；
2. 旧 asset 改临时名保留；
3. 新 asset 改成正式名；
4. 最后删除旧 asset。

避免“先删除唯一备份、随后上传失败”的风险。

## 手动操作

部署 backup worker：

```powershell
uv run modal deploy cache_backup_app.py
```

强制刷新所有已就绪缓存：

```powershell
uv run python -c "import modal; print(modal.Function.from_name('glm53-cache-backup','backup_all_force').remote())"
```

普通增量检查：

```powershell
uv run python -c "import modal; print(modal.Function.from_name('glm53-cache-backup','backup_runtime_caches').remote())"
```

`delete-modal.bat` 只停止 `glm53-flash-nota-b300`，不会停止独立 CPU backup app。
