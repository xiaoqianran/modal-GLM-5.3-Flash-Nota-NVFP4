# GLM-5.3 Flash B300 冷启动基准 — 2026-09-26

模型：nota-ai/GLM-5.3-Flash-Nota-NVFP4  
Revision：c5fc7f5ef0447ab030559bb4b08861a36dbbe847  
GPU：Modal B300，单卡  
vLLM：0.28.1rc1.dev580+g385dce36b  
权重：191.01 GiB / 386 个 safetensors shard / Modal Volume 9P  
加载策略：prefetch-16，16 MiB block  
KV Cache：FP8  
MTP：5 speculative tokens  
CUDA Graph：PIECEWISE，max capture size=1008

## 本轮结果

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

## 结论

当前真实 cold start 的主要问题已经不只是权重：

1. FlashInfer autotune：约 256.85 s
2. Model loading：162.53 s
3. CUDA Graph capture：158 s

这三项合计约 577.37 s，占 API Ready 829.884 s 的约 69.6%。

下一轮最值得测试：

- A：当前 PIECEWISE / 1008
- B：enforce eager
- C：PIECEWISE / 512
- D：PIECEWISE / 256

另外需要特别验证 FlashInfer autotune cache 在下一次新容器中是否能复用；若 cache 没有跨容器持久化，256 s 级 autotune 会持续成为最大的冷启动成本。

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

当前真正最大的启动瓶颈仍然是 FlashInfer autotune，而不是 CUDA Graph。
