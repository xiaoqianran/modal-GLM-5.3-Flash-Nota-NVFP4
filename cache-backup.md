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
├── glm53-flash-nota-flashinfer-jit-workspace/
├── glm53-flash-nota-triton-cache/
├── glm53-flash-nota-tilelang-cache/
├── glm53-flash-nota-vllm-startup-plan/
├── glm53-flash-nota-vllm-modelinfos/
└── glm53-cache-release-availability/
```

## 当前缓存 / 恢复架构

```text
第一层：Modal Volume runtime caches
        ↓
FlashInfer / CUDA / Triton / HF weights...
每个新 B300 容器直接命中这些缓存后正常初始化

第二层：GitHub Release
        ↓
Volume 丢失 / 新环境时的 portable fallback
```

当前状态：

| 层 | 当前状态 | 说明 |
|---|---|---|
| Modal Volume runtime caches | ✅ 已启用且当前已有数据 | 单一项目 Volume：`modal-GLM-5.3-Flash-Nota-NVFP4`，承载 HF weights 和全部 runtime cache。 |
| GitHub Release | ✅ 已启用且已有 5 个 assets | 作为 portable fallback；`flashinfer-jit` 目前因无有效 `.so/.o/.cubin` 尚未形成 asset。 |

当前优先级：

```text
Modal Volume 恢复 HF + runtime caches
→ 完整初始化

如果 Modal Volume 某 cache 缺失
→ GitHub Release fallback
→ 恢复后 commit 回 Modal Volume
```

> 2026-09-27 起，生产运行路径彻底移除 Modal CPU/GPU Memory Snapshot、vLLM `/sleep?level=1` 和 `/wake_up`。每个新容器都直接依赖文件级持久缓存正常启动。

`VLLM_CACHE_ROOT` 保持本地。仅 `startup_plan` 子目录链接到 Volume；原生
`modelinfos` 架构检查缓存从独立 seed 复制到本地，变化后单独写回。两者在 API
Ready 后按有效内容差异提交，已有旧 plan 不会阻止本次新 plan 的提交。
目录里有缓存不等于实际命中：分别看原生 plan apply 日志及 registry lookup 返回值。

FlashInfer JIT 的可选备份可用性在 CPU 部署恢复阶段记录。普通 GPU 启动跳过
absent/unknown 的 GitHub 查询；下一次部署重新检查。灾备可使用
`GLM53_CACHE_DISASTER_RECOVERY=1` 绕过记录查询，或先运行 CPU 的
`step_009_restore_runtime_caches`。网络错误只记为 unknown。
细节和历史 82 秒拆解见 [启动分析](startup-preweight-analysis-2026-09-27.md)。

## 运行链路

FlashInfer autotune 单独持久化，不受默认关闭的 `GLM53_RUNTIME_CACHE_SYNC`
影响：检测到 `Saved ... configs` 后，后台比较配置内容指纹；变化时写 dirty
marker、commit Volume，再触发 CPU 备份。未变化时跳过 commit。只有看到
`FLASHINFER_CACHE_COMMIT_DONE` 才表示本次新配置已显式提交。

完整复用的验收标准是同一 serving 配置在下一次新容器启动时出现
`new_configs=0 previous_configs>0 actual_cache_hit=true`。Volume 里有文件或
`Loaded ... configs` 只证明读到了配置，不代表全部命中；新增 key/shape 首次仍需调优。

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

## 启动预读

`helpers/020_vllm_bootstrap.py` 在导入 vLLM/torch 前启动后台预读，读取本地
model mirror 的 index。API 与 EngineCore 共用每次启动新建的本地协调目录，
避免旧 `.done` 标记跳过新进程的冷 page cache。遇到 inflight 任务时 EngineCore
立即继续加载；MTP 的重叠 shard 也共用文件锁，只在后台预读线程等待。
读取失败不写完成标记，后续 loader 可以重试。

验证日志：`EARLY_WEIGHT_PREFETCH_REQUEST` 应紧随 `003_VLLM_PROCESS_START`；
inflight 复用应显示 `action=load-without-wait`。本地回归测试使用
`python -m unittest discover -s tests -v`；真实提前时间和 autotune 命中率需在
B300 新容器上连续两次启动验收。

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

## 部署与手动操作

backup worker 随 `modal deploy app.py` 一起部署，不再单独部署第二个 App。

`deploy-modal.bat` 的第 `[7/7]` 步调用 `backup_runtime_caches`：只同步 dirty / missing cache；GitHub 已存在且内容未标记变化时直接命中，不重新打包或上传。

`backup_all_force` 仅保留给人工修复/强制刷新，不参与常规 deploy。

强制刷新所有已就绪缓存：

```powershell
uv run python -c "import modal; print(modal.Function.from_name('glm53-flash-nota-b300','backup_all_force').remote())"
```

普通增量检查：

```powershell
uv run python -c "import modal; print(modal.Function.from_name('glm53-flash-nota-b300','backup_runtime_caches').remote())"
```

`delete-modal.bat` 停止 `glm53-flash-nota-b300` 时，B300 serving、定时 CPU backup worker 和手动 backup function 会一起停止。
