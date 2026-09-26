# GLM-5.3 Flash B300 冷启动基准 — 2026-09-26

模型：nota-ai/GLM-5.3-Flash-Nota-NVFP4  
Revision：c5fc7f5ef0447ab030559bb4b08861a36dbbe847  
GPU：Modal B300，单卡  
vLLM：0.28.1rc1.dev580+g385dce36b  
权重：191.01 GiB / 386 个 safetensors shard / Modal Volume 9P  
加载策略：prefetch-16，16 MiB block  
KV Cache：FP8  
MTP：5 speculative tokens  
初始基线 CUDA Graph：PIECEWISE，max capture size=1008

## A 组：初始 PIECEWISE / 1008 基线

| 阶段 | 实测 |
|---|---:|
| vLLM process → 权重加载开始 | 98.539 s |
| 第一段 Loading weights | 132.870 s |
| 第二段 Loading weights | 13.190 s |
| Model loading 总计 | 162.526 s |
| Model memory | 191.54 GiB |
| 第一段 weight done → model init done | 22.660 s |
| CUDA Graph capture | 158.000 s |
| CUDA Graph shapes | 82 |
| CUDA Graph capture 报告显存 | 5.21 GiB |
| CUDA Graph 实际显存 | 4.62 GiB |
| Available KV cache memory | 46.75 GiB |
| GPU KV cache | 6,411,293 tokens |
| 1,048,576-token request 最大理论并发 | 6.11x |
| JIT kernel warmup | 5.06 s |
| FlashInfer autotune | 256.847 s |
| FlashInfer autotune configs saved | 66 |
| engine profile/create KV/warmup | 571.72 s |
| vLLM process → API Ready | 829.884 s |
| API Ready → 第一发 generation 完成 | 0.329 s |
| vLLM process → 第一发 generation 完成 | 830.213 s |

## 权重加载细分

本轮 vLLM 出现两次 checkpoint prefetch / weight-load 报告：

- 第一段 page-cache prefetch：114.85 s
- 第一段 Loading weights：132.87 s
- 第二段 page-cache prefetch：9.32 s
- 第二段 Loading weights：13.19 s
- 最终 Model loading：162.526291 s

第二段出现在 GLM-5.3 MTP/speculative model 初始化链路中，因此不能把 132.87 s 简化为整个模型唯一的权重 I/O 时间。

## CUDA Graph

PIECEWISE capture：

- max capture size：1008
- capture shapes：82
- capture：158 s
- capture 报告显存：5.21 GiB
- 启动收尾时 profiler 报实际 CUDA Graph memory：4.62 GiB
- estimated：5.21 GiB
- difference：0.59 GiB / 12.8%

CUDA Graph 已经是明确的大型冷启动瓶颈。

## Kernel / JIT / Autotune

本轮没有出现旧版的统一 torch.compile total 日志；当前镜像主要表现为具体 kernel/JIT 路径：

- Inductor
- FlashInfer
- CUTLASS
- TRT-LLM
- TileLang
- Fused-MoE
- JIT kernel warmup：5.06 s
- FlashInfer autotune：约 256.847 s

FlashInfer autotune 从 11:09:11.416 到 11:13:28.263，最终保存 66 个新配置。  
其中 flashinfer::trtllm_fp4_block_scale_moe 的 22-profile 主循环约 190 s。

## A 组当时结论

在 A 组初始基线中，cold start 的主要问题已经不只是权重：

1. FlashInfer autotune：约 256.85 s
2. Model loading：162.53 s
3. CUDA Graph capture：158 s

这三项合计约 577.37 s，占 API Ready 829.884 s 的约 69.6%。

当时规划的下一轮测试：

- A：当前 PIECEWISE / 1008
- B：enforce eager
- C：PIECEWISE / 512
- D：PIECEWISE / 256

当时仍需特别验证 FlashInfer autotune cache 在下一次新容器中是否能复用；该问题后来已在 D 组验证解决。

---

## B 组：enforce eager

实验只新增 `--enforce-eager`，其余保持：

- prefetch-16 / 16 MiB block
- FP8 KV
- MTP=5
- gpu_memory_utilization=0.96
- FlashInfer autotune 开启

### A / B 对比

| 指标 | A：PIECEWISE / 1008 | B：enforce eager | 差值 |
|---|---:|---:|---:|
| process → weight load start | 98.539 s | 154.425 s | +55.886 s |
| 主权重 Loading weights | 132.870 s | 76.630 s | -56.240 s |
| Model loading 总计 | 162.526 s | 115.501 s | -47.025 s |
| CUDA Graph capture | 158.000 s | 0 s | -158.000 s |
| Available KV cache memory | 46.75 GiB | 51.96 GiB | +5.21 GiB |
| GPU KV cache | 6,411,293 tokens | 7,130,316 tokens | +719,023 |
| 1M context 理论并发 | 6.11x | 6.80x | +0.69x |
| JIT kernel warmup | 5.06 s | 6.80 s | +1.74 s |
| FlashInfer autotune | 256.847 s | 286.193 s | +29.346 s |
| engine profile/KV/warmup | 571.72 s | 533.71 s | -38.01 s |
| process → API Ready | 829.884 s | 797.083 s | -32.801 s |
| API Ready → first warmup | 0.329 s | 14.842 s | +14.513 s |
| process → first warmup done | 830.213 s | 811.925 s | -18.288 s |

API Ready 总时间只缩短约 32.80 s（约 3.95%）。

### 关键发现

1. `enforce_eager=True` 已真实生效：
   - `[006_CUDAGRAPH_DISABLED]`
   - `cudagraph_mode=NONE`
   - CUDA Graph 显存从约 4.62 GiB 降为 0 GiB。

2. 去掉 CUDA Graph 后，KV Cache 明显增加：
   - 46.75 GiB → 51.96 GiB
   - 6,411,293 → 7,130,316 tokens
   - 1M context 理论并发 6.11x → 6.80x。

3. 但 CUDA Graph 的 158 s 并没有完整转化为 cold-start 收益：
   - process 前置初始化多了约 55.9 s；
   - FlashInfer autotune 又多了约 29.3 s；
   - 最终 API Ready 只快 32.8 s。

4. FlashInfer autotune cache 没有跨新容器复用。
   - A 组 cache hash：`103a/b0bb...`
   - B 组 cache hash：`103a/92ab...`
   - B 组仍然保存 `66 new, 0 from previous config`。
   - 因为 eager 改变了运行配置，cache key 也发生变化。

5. eager 的第一发真实 generation 很慢：
   - localhost warmup：14.842 s
   - 随后的外部同类请求：约 1.531 s（包含 Modal 网关/网络，不与 localhost 0.329 s 直接等价比较）。

### B 组结论

`enforce eager` 可以完全消除 CUDA Graph capture，并释放约 4.62 GiB 显存给 KV Cache，但本轮真实 cold start 只从 829.884 s 降到 797.083 s。

在 B 组当时的未命中状态下，最大的启动瓶颈仍然是 FlashInfer autotune，而不是 CUDA Graph；该结论只描述该历史实验，不代表后续 cache-hit 配置。

---

## C 组：FlashInfer autotune cache 持久化

目标：把默认的

`/root/.cache/vllm/flashinfer_autotune_cache`

挂载到独立 Modal Volume：

`glm53-flash-nota-flashinfer-autotune`

并在 API Ready 后显式执行 `commit()`。

### C1：首次灌 cache

- vLLM process start：19:36:44
- 主权重 Loading weights：72.46 s
- Model loading：112.401 s
- FlashInfer autotune start：19:41:52
- FlashInfer autotune end：19:46:31
- autotune 总时长：约 279.4 s
- 保存：66 configs
- previous config：0
- API Ready：752.019 s
- cache commit：19:49:16
- first warmup：13.955 s

日志明确：

`Saved 66 configs ... (66 new, 0 from previous config)`

`[FLASHINFER_AUTOTUNE_CACHE_COMMIT]`

Volume 中已经确认存在：

`0.6.18/103a/92abe8178cfd81374e9ac2c11b7dd931d918e418d0296c3303a037b5c33df826/autotune_configs.json`

文件大小：15.9 KiB。

因此 FlashInfer autotune cache 的持久化写入已经验证成功。

### C2：新容器 cache-hit 验证

第一次 C2 在权重加载结束时被 Dashboard 外部 stop：

`Stopping app - user stopped from dashboard.`

第二次改用独立 benchmark app：

`glm53-flash-nota-b300-cache-bench`

但本地触发请求会话先因 AgentDock 默认约 8 分钟超时而中断，导致 web-server startup 请求结束。

随后已修正为 1800 秒请求会话，但在重新部署时 Modal workspace 达到 spend limit：

`Workspace ... has exceeded its spend limit`

该阶段当时因为 spend limit 暂时无法完成 C2；后续 D 组已经完成最终验证：

- cache 文件持久化：✅
- cache 文件跨容器 Volume 可见：✅
- 新容器读取 previous configs：✅
- 原约 279~286 s autotune 在相同配置 key 下可降到约 2 s 级：✅

因此 C2 的验证项已经关闭。不同 serving 参数可能产生新的 FlashInfer cache key；首次生成后仍需重新持久化对应 key。


---

## D 组：cache-hit 后的真实冷启动验证

### D1：21:46 部署实例

App：`ap-RCGXZcAX9hi32IgcC0IbwV`

关键日志：

- 第一段主权重 Loading weights：38.29 s
- 第二段 MTP 权重 Loading weights：7.32 s
- Model loading 总计：50.883 s
- model init done：155.609 s（from process start）
- KV cache ready：173.988 s
- KV cache：51.96 GiB / 7,130,316 tokens / 1M context 6.80x
- JIT kernel warmup：0.10 s
- FlashInfer autotune：命中 66/66，`0 new, 66 from previous config`
- FlashInfer autotune done：177.631 s（from process start）
- engine profile/create KV/warmup：31.69 s
- API Ready：191.389 s
- 1-token warmup：0.317 s

结论：FlashInfer autotune cache 已经完成跨容器复用，原先约 279~286 s 的 autotune 已经降到约 2 s 级别。

### D2：22:09 部署实例

App：`ap-3vhMrH10hFWT4c33srBXQv`

关键日志：

- page-cache prefetch：2.94 s
- 第二段 MTP 权重 Loading weights：6.04 s
- Model loading 总计：41.542 s
- model init done：123.492 s（from process start）
- KV cache ready：136.344 s
- KV cache：51.96 GiB / 7,130,316 tokens / 1M context 6.80x
- JIT kernel warmup：0.06 s
- FlashInfer autotune：66/66 cache hit
- FlashInfer autotune done：139.186 s（from process start）
- engine profile/create KV/warmup：22.65 s
- API Ready：149.504 s
- 1-token warmup：0.241 s
- Triton archive sync：2216 files / 85.1 MB，unchanged
- TorchInductor archive sync：26 files / 1.66 MB，unchanged
- CUDA compute cache sync：94 files / 761 MB，0.572 s，unchanged

截至目前，这是已观察到的最快完整 cold-start：**vLLM process → API Ready = 149.504 s**。

与最早 A 组 829.884 s 相比，缩短约 **680.38 s / 82.0%**。

> 注意：这里仍然使用 `--enforce-eager`，因此 CUDA Graph capture 为 0；该结果不能代表开启 CUDA Graph 后的启动时间。

---

## Modal App 运行历史（2026-09-26）

下面记录 Modal 返回的 app 生命周期，便于后续把日志和具体实验对应起来。  
**生命周期不等于模型 cold-start 时间**：其中很多 8~36 秒的 app 是 deploy 流程里的 CPU helper（secret/cache/compact/publish 等）任务。

| App ID | 创建 | 停止 | 生命周期 |
|---|---|---|---:|
| ap-3vhMrH10hFWT4c33srBXQv | 22:09:42 | 22:15:06 | 324 s |
| ap-NXPTHXVaQmeETFDKx0pjJq | 22:04:39 | 22:04:52 | 13 s |
| ap-Pd6DKguo2cwBc8Ugoe8NR6 | 22:03:51 | 22:04:07 | 16 s |
| ap-EbKMBbSwYoxPhG2N9Zn2m5 | 22:03:51 | 22:04:05 | 14 s |
| ap-oxsSlQS6MICawp3VTg3aiT | 22:03:26 | 22:04:02 | 36 s |
| ap-isW5qFyfcZoyklq0d5LGwP | 22:01:58 | 22:03:49 | 111 s |
| ap-YqSihyINNbyMCyLDNqeLgI | 22:00:57 | 22:03:48 | 171 s |
| ap-RCGXZcAX9hi32IgcC0IbwV | 21:46:42 | 21:53:16 | 394 s |
| ap-iAb1SqaqEY8C6Ts4yrAncw | 21:45:58 | 21:46:33 | 35 s |
| ap-BvZ8vHhv2CUT3xKlL94zXH | 21:34:12 | 21:38:49 | 277 s |
| ap-tw3IVYzAmo7fECf06cHEi1 | 21:33:13 | 21:36:03 | 170 s |
| ap-HwlvZijCdlEEzUV06HYhT5 | 21:24:08 | 21:33:24 | 556 s |
| ap-s4OrsYX3qDvUaXISil6tpk | 21:24:38 | 21:24:46 | 8 s |
| ap-fHPI31hggwMQFIGd19PDgs | 21:24:27 | 21:24:36 | 9 s |
| ap-Twy28vdsbRcXUXTdUeUA0s | 20:54:45 | 21:18:30 | 1425 s |
| ap-fnfKtJJBrXJ7bBmza6uVz6 | 21:14:47 | 21:14:59 | 12 s |
| ap-hzQd3a9izG8YljQfHsmlIf | 21:14:36 | 21:14:45 | 9 s |
| ap-KQSiWsJT8fLkQpjFSwuW8i | 21:14:18 | 21:14:33 | 15 s |
| ap-TbD3K11KwEjm1regtmgpd2 | 20:54:34 | 20:54:43 | 9 s |
| ap-9GUaWcJ6cEUG0ncVrebnp9 | 20:54:21 | 20:54:31 | 10 s |
| ap-kgINww42TBriM1A0u5d7PX | 20:54:08 | 20:54:18 | 10 s |
| ap-q0PxhEG4S2naqD0ER2wTr3 | 20:36:54 | 20:52:34 | 940 s |
| ap-ieWcuuAfcL01gW0tUHJjQO | 20:36:44 | 20:36:52 | 8 s |
| ap-TX8kYJRYcLkdja9Pnr90HQ | 20:36:33 | 20:36:41 | 8 s |
| ap-neXSTmzmY8FLqF1Umridx6 | 20:36:12 | 20:36:31 | 19 s |

已确认的长 GPU 启动/服务实例至少包括：

- `ap-RCGXZcAX9hi32IgcC0IbwV`：API Ready 191.389 s
- `ap-3vhMrH10hFWT4c33srBXQv`：API Ready 149.504 s
- `ap-Twy28vdsbRcXUXTdUeUA0s`：日志中观察到主权重 Loading weights 70.84 s，但该实例未提取到完整 API Ready 关键行

---

## 017 独立长输出推理吞吐基准（手动）

生产 `008_warmup.py` 只执行 `16 tokens × 3` 的短 warmup。下面这两道
`max_tokens=8092` 长输出题已经从 startup 拆出，只有显式调用
`step_017_bench_generation` 时才会运行，不会再与首个真实请求争抢 B300。

1. 12 枚硬币 / 3 次天平称量的完整决策策略；
2. 全球多区域超大模型推理平台的架构、调度、缓存、SLO 与成本权衡。

每题参数：

- `max_tokens=8092`
- `temperature=0`
- timeout：900 s
- 日志输出：prompt tokens / completion tokens / request seconds / `approx_completion_tps` / finish reason

其中 `approx_completion_tps = completion_tokens / request_s`，它包含 prefill/TTFT，因此是保守的端到端近似值；如果要得到严格 decode tok/s，后续再增加 streaming 首 token 时间点即可。

---

## 正式高吞吐 serving 配置

目标改为：**稳态 decode token/s 优先，冷启动时间退居其次。**

当前正式配置：

- 单 B300 / TP=1
- NVFP4 权重
- FP8 KV Cache
- MTP speculative tokens = 5
- `max_num_seqs = 16`
- `max_num_batched_tokens = 8192`
- CUDA Graph：`FULL_DECODE_ONLY`
- MTP5 uniform decode query length = 6
- CUDA Graph capture sizes：`6,12,18,...,96`
- 最大 CUDA Graph capture size：96
- 已删除 `--enforce-eager`

capture sizes 与 1~16 并发一一对应：

`6 * concurrent_requests`

因此分别覆盖 1 到 16 个并发请求的 MTP5 uniform decode batch，避免重新回到 PIECEWISE/1008 时大量无关 shape 的 capture。


---

## E 组：FULL_DECODE_ONLY 正式部署实测

部署 App：`ap-kJAdxaxdHeyeMxQL0IMwiy`

配置：

- `FULL_DECODE_ONLY`
- capture sizes：`6,12,...,96`
- `max_num_seqs=16`
- `max_num_batched_tokens=1024`
- MTP5
- FP8 KV
- 无 `--enforce-eager`

启动关键数据：

- vLLM process start：23:14:05
- 主权重 Loading weights：39.30 s
- MTP 权重 Loading weights：6.32 s
- Model loading：50.675 s
- model init done：146.368 s
- CUDA Graph capture：**9 s**
- CUDA Graph memory：**1.39 GiB**
- KV cache：53.63 GiB / 7,355,011 tokens / 1M context 7.01x
- JIT kernel warmup：0.15 s
- FlashInfer autotune：首次新 key，36 new / 0 previous
- FlashInfer autotune：约 169.7 s
- engine init：219.84 s
- API Ready：**370.030 s**

说明：本轮 API Ready 较慢主要是新的 FlashInfer autotune cache key 首次生成；该 cache 已在本轮结束后 commit 并发布到 GitHub Release。

长输出 benchmark：

| Case | Prompt | Completion | 时间 | 端到端 completion tok/s |
|---|---:|---:|---:|---:|
| logic | 107 | 8092 | 43.048 s | **187.976 tok/s** |
| systems | 148 | 8092 | 52.044 s | **155.485 tok/s** |

合计：

- completion tokens：16,184
- 总 generation request 时间：95.092 s
- 加权端到端吞吐：约 **170.20 tok/s**

vLLM 10 秒窗口观察到的 generation throughput：

- Case 1：167.0 / 178.4 / 171.1 / **189.2 tok/s**
- Case 2：164.3 / 146.6 / 148.5 / 163.1 / 180.1 tok/s

MTP5 acceptance：

- Case 1 观察区间约 32.9%~47.9%
- Case 2 观察区间约 24.0%~39.6%

本轮同时生成/更新的可持久化缓存：

- Triton：2288 files / 90.5 MB，changed=true
- CUDA Compute：310 files / 922.5 MB，changed=true
- FlashInfer autotune：已 commit，并重新发布 GitHub Release
- TileLang：已重新发布 GitHub Release

额外观察：

上一轮 `max_num_batched_tokens=1024` 时 vLLM 明确给出性能警告：在 MTP5 下 scheduled token budget 偏小，可能限制 speculative decoding 的最佳吞吐。正式配置已提高到 `8192`；该值也与当前 vLLM GLM-5.3 recipe 中 `max_num_seqs=16` 的已验证配置一致。

---

## 三层启动/缓存架构：GPU Snapshot → Modal Volume → GitHub Release

当前生产设计不是单一缓存，而是三层互补结构：

```text
第一层：GPU Memory Snapshot
        ↓
最快
恢复已经初始化好的 vLLM / 权重 / KV 运行状态

第二层：Modal Volume runtime caches
        ↓
HF weights / FlashInfer / TileLang / Triton / TorchInductor / CUDA Compute
snapshot 失效或不可用时帮助快速重新构建

第三层：GitHub Release
        ↓
Modal Volume 缺失、新环境或灾备恢复时的 portable fallback
```

### 第一层：Modal CPU + GPU Memory Snapshot

生产 `VllmServer` 已配置：

```python
enable_memory_snapshot=True
experimental_options={"enable_gpu_snapshot": True}
```

生命周期：

```text
@modal.enter(snap=True)
完整初始化 vLLM
→ API Ready
→ 16-token × 3 轻量 warmup
→ /sleep?level=1
→ 同步 runtime caches 到 Modal Volume
→ Modal 保存 CPU/GPU Memory Snapshot

@modal.enter(snap=False)
后续新容器恢复 snapshot
→ /wake_up
→ API Ready
```

注意：

- Snapshot **不是**保存在项目 Modal Volume 中。
- Snapshot 由 Modal 平台内部管理，不会出现在 `modal volume list`。
- 项目 Volume 只负责文件级持久缓存；Snapshot 是已初始化进程/显存状态的更高层恢复机制。
- 当前代码已经启用 snapshot，且历史上已经真实跑通过 snapshot restore。
- 每个新的 deployment / function revision 是否已有可用 snapshot，需要该 revision 至少成功完成一次 `snap=True` build；仅看到配置开启不能证明当前 revision 已经生成新 snapshot。

历史实测：

| Snapshot 指标 | 实测 |
|---|---:|
| 首次完整 snapshot build / prepare | 约 **476.782 s** |
| vLLM `wake_up` 恢复 weights + KV | 约 **5.457 s** |
| Snapshot restore → API Ready | 约 **5.500 s** |

因此，在 snapshot 已存在且有效时，历史实测启动从完整 cold-build 的分钟级下降到约 **5.5 s**。

### 第二层：Modal Volume runtime caches

当前统一项目 Volume：

```text
modal-GLM-5.3-Flash-Nota-NVFP4/
├── glm53-flash-nota-hf-cache/
├── glm53-flash-nota-cuda-compute-cache/
├── glm53-flash-nota-flashinfer-autotune/
├── glm53-flash-nota-torchinductor-cache/
├── glm53-flash-nota-flashinfer-jit/
├── glm53-flash-nota-triton-cache/
└── glm53-flash-nota-tilelang-cache/
```

运行产生的新 cache 会先安全持久化到 Modal Volume：

```text
B300 runtime
→ runtime cache 产生/变化
→ project_volume.commit()
→ Modal Volume 成为 authoritative runtime cache
→ 标记 .github-backup-dirty
→ CPU backup worker 异步处理 GitHub
```

两类路径：

1. 直接写项目 Volume：
   - FlashInfer autotune
   - FlashInfer JIT
   - TileLang

2. 先写容器本地 `/tmp/glm53-runtime-cache`，再归档同步到 Volume：
   - Triton
   - TorchInductor
   - CUDA Compute

因此即使 GitHub backup 失败，已经 `commit()` 的 cache 仍保存在 Modal Volume 中，不依赖 B300 容器继续存活。

### 第三层：GitHub Release portable backup

Repository：

`xiaoqianran/modal-GLM-5.3-Flash-Nota-NVFP4`

Release tag：

`cache-b300-glm53-flash-nota-v1`

当前已存在的 portable cache assets：

| Asset | 当前大小 |
|---|---:|
| `flashinfer-autotune-0.6.18-b300.tar.gz` | 9,592 B |
| `tilelang-b300.tar.gz` | 2,672,718 B |
| `triton-b300.tar.gz` | 19,890,082 B |
| `torchinductor-b300.tar.gz` | 479,783 B |
| `cuda-compute-b300.tar.gz` | 194,429,984 B |

`flashinfer-jit-0.6.18-b300.tar.gz` 当前尚不存在，因为项目 Volume 中还没有满足 `*.so / *.o / *.cubin` 条件的有效 JIT binary；一旦出现并发生 fingerprint 变化，会被标记 dirty 并交给 CPU backup worker 发布。

常规 deploy 的第 `[7/7]` 步已经改为增量：

```text
backup_runtime_caches
→ dirty / missing 才重新打包上传
→ GitHub 已存在且无变化：RELEASE_HIT
→ 不再每次 force 全量重传
```

2026-09-27 实测一次无变化的增量检查：

- 总耗时：**7.189 s**
- 重新打包：0
- 重新上传：0
- FlashInfer autotune：RELEASE_HIT
- TileLang：RELEASE_HIT
- Triton：RELEASE_HIT
- TorchInductor：RELEASE_HIT
- CUDA Compute：RELEASE_HIT
- FlashInfer JIT：SKIP / cache_not_ready

### 故障恢复优先级

```text
1. 有有效 GPU Snapshot
   → restore + wake_up
   → 历史实测约 5.5 s API Ready

2. Snapshot 不存在 / 失效
   → 从 Modal Volume 读取 HF + runtime caches
   → 完整初始化
   → 生成新的 snapshot

3. Modal Volume 中某 cache 缺失
   → GitHub Release fallback 恢复
   → commit 回 Modal Volume
   → 再进入正常启动
```

三层职责：

- **GPU Snapshot**：启动速度层。
- **Modal Volume**：运行时持久化主存储 / authoritative cache。
- **GitHub Release**：可移植备份与灾备 fallback。

三层不是重复存储，而是分别解决“极速恢复 / 稳定持久化 / 跨环境灾备”三个问题。
