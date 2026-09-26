# Runtime cache backup

## 目标

- Modal Volume 是运行时主缓存。
- GitHub Release 是可移植备份 / fallback。
- B300 只负责生成缓存并 commit 到 Volume，不等待 GitHub 上传。
- GitHub 上传由主 App `glm53-flash-nota-b300` 内的独立 CPU function 完成。
- CPU cache worker 与 B300 serving 属于同一个 Modal App，因此 deploy / stop 生命周期一致。
- 所有持久缓存统一收口到一个项目 Volume：`modal-GLM-5.3-Flash-Nota-NVFP4`。

## Volume 布局

```text
modal-GLM-5.3-Flash-Nota-NVFP4
├── glm53-flash-nota-hf-cache/
├── glm53-flash-nota-cuda-compute-cache/
├── glm53-flash-nota-flashinfer-autotune/
├── glm53-flash-nota-torchinductor-cache/
├── glm53-flash-nota-flashinfer-jit/
├── glm53-flash-nota-triton-cache/
└── glm53-flash-nota-tilelang-cache/
```

## 运行链路

```text
B300 warmup
  -> cache sync
  -> .github-backup-dirty
  -> Modal Volume commit
  -> CACHE_VOLUME_SAFE
  -> spawn backup_runtime_caches (CPU function, same Modal App)
  -> B300 可立即停止

glm53-flash-nota-b300 / backup_runtime_caches
  -> 读取 Modal Volume
  -> 打包 cache
  -> 安全替换 GitHub Release asset
  -> 清理 dirty marker
```

同一 App 内的 scheduled CPU backup function 每小时还会自动检查一次，因此即使 B300 在 Volume commit 后、spawn 前被强制停止，dirty marker 仍会被后续 CPU 任务发现。

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

backup worker 随 `modal deploy app.py` 一起部署，不再单独部署第二个 App。

强制刷新所有已就绪缓存：

```powershell
uv run python -c "import modal; print(modal.Function.from_name('glm53-flash-nota-b300','backup_all_force').remote())"
```

普通增量检查：

```powershell
uv run python -c "import modal; print(modal.Function.from_name('glm53-flash-nota-b300','backup_runtime_caches').remote())"
```

`delete-modal.bat` 停止 `glm53-flash-nota-b300` 时，B300 serving、定时 CPU backup worker 和手动 backup function 会一起停止。
