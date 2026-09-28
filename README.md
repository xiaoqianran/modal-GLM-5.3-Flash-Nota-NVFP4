# modal-GLM-5.3-Flash-Nota-NVFP4

[English](#) | 简体中文

在 **Modal 单张 NVIDIA B300** 上部署 `nota-ai/GLM-5.3-Flash-Nota-NVFP4` 的高吞吐推理项目。

项目基于 vLLM，提供 OpenAI Compatible API，并针对 GLM-5.3-Flash NVFP4、超大权重加载、FlashInfer、CUDA Graph、MTP speculative decoding 和 Modal 冷启动进行了专项优化。

## 当前实测性能

> 测试环境：Modal / 1× NVIDIA B300 / TP=1 / NVFP4 / FP8 KV Cache / MTP=5。  
> 以下均为项目实际运行记录，不代表所有输入、上下文长度和并发场景下都能达到相同结果。

| 指标 | 实测 |
|---|---:|
| 最新完整冷启动：vLLM process → API Ready | **122.302 s** |
| Runtime Ready | **126.606 s** |
| 历史早期完整冷启动 | 829.884 s |
| 冷启动总体缩短 | 约 **85%** |
| 长输出端到端吞吐 | 约 **155–188 tok/s** |
| 两组长输出加权吞吐 | **约 170.2 tok/s** |
| vLLM 10 秒窗口最高观测 generation throughput | **约 189.2 tok/s** |

其中两组 `max_tokens=8092` 长输出实测：

| Case | Completion Tokens | 时间 | 端到端 Completion 吞吐 |
|---|---:|---:|---:|
| Logic | 8092 | 43.048 s | **187.976 tok/s** |
| Systems | 8092 | 52.044 s | **155.485 tok/s** |

这里的端到端吞吐按：

```
completion_tokens / request_seconds
```

计算，因此包含 prefill / TTFT，属于偏保守的实际请求吞吐，而不是纯 decode benchmark。

## 冷启动优化结果

最早基线：

```
vLLM process
→ 权重加载
→ FlashInfer autotune
→ CUDA Graph
→ KV / warmup
→ API Ready

829.884 s
```

经过持续优化后，最新新容器完整启动基线：

```
vLLM process
→ early weight prefetch
→ modelinfo cache hit
→ persisted startup plan hit
→ 权重加载
→ KV / multimodal warmup / CUDA Graph
→ API Ready

122.302 s
```

主要优化包括：

- 191 GiB / 386 个 safetensors shard 使用单一 `prefetch-16`。
- 权重预读提前到 vLLM 进程启动初期，与 Python / registry / EngineCore 初始化并行。
- 删除重复权重 reader，避免同一批 shard 双重读取。
- FlashInfer autotune cache 持久化，同配置下从约 256–286 s 降到约 2 s 级。
- 持久化 vLLM `modelinfos`，避免新容器重复执行昂贵的模型架构检查子进程。
- 持久化并复用 startup plan，同时保留显存安全检查。
- 对被 patch 的 vLLM Python 源文件在镜像构建阶段预编译。
- 生产 warmup 只保留轻量 warmup，长输出 benchmark 与服务启动解耦。
- Modal Volume 作为 runtime cache 主存储，GitHub Release 仅作为 portable fallback。
- 当前生产路径不依赖 GPU Memory Snapshot / sleep / wake。

## 当前 Serving 配置

核心配置：

```text
GPU                    1 × NVIDIA B300
Tensor Parallel        1
Model                   nota-ai/GLM-5.3-Flash-Nota-NVFP4
Weights                 NVFP4
KV Cache                FP8
MTP speculative tokens  5
max_num_seqs            16
max_num_batched_tokens  8192
CUDA Graph              FULL_DECODE_ONLY
```

项目保留完整多模态能力，没有通过 `--language-model-only` 裁掉 vision / video 路径来换取冷启动速度。

## 部署

### 1. 安装环境

项目使用 `uv`。

先安装 Modal CLI 并完成登录：

```bash
uv sync
uv run modal setup
```

确认当前 Modal workspace：

```bash
uv run modal profile current
```

### 2. 配置环境变量

复制：

```
.env.example
```

为：

```
.env
```

填写：

```env
HF_TOKEN=hf_xxx
GITHUB_TOKEN=github_xxx
```

`HF_TOKEN` 用于模型访问。

`GITHUB_TOKEN` 用于 runtime cache 的 GitHub Release portable backup。

### 3. 一键部署

Windows：

```bat
deploy-modal.bat
```

部署脚本会依次完成：

1. 同步 Hugging Face secret。
2. 同步 GitHub secret。
3. 下载并确认模型权重缓存。
4. 恢复 runtime caches。
5. 整理 Triton / TorchInductor / CUDA Compute cache。
6. 部署 B300 inference service。
7. 增量备份有变化的 runtime cache。

## 启动与停止 B300

启动已部署服务：

```bat
start.bat
```

或：

```bat
start-b300.bat
```

停止 B300 容器：

```bat
stop-b300.bat
```

当前设计是停止容器而不是删除整个部署，便于之后重新启动。

## OpenAI Compatible API

部署完成后脚本会打印：

```text
Base URL
OpenAI Base URL
/v1/models
/v1/chat/completions
```

URL 形式：

```
https://<modal-workspace>--glm53-flash-nota-b300-vllmserver-serve.modal.run/v1
```

可以直接按 OpenAI Compatible API 使用。

Python 示例：

```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="https://<workspace>--glm53-flash-nota-b300-vllmserver-serve.modal.run/v1",
)

response = client.chat.completions.create(
    model="nota-ai/GLM-5.3-Flash-Nota-NVFP4",
    messages=[{"role": "user", "content": "你好，请介绍一下你自己。"}],
)

print(response.choices[0].message.content)
```

## Cache 架构

当前生产缓存层级：

```text
B300 Runtime
    ↓
Modal Volume
    ├─ Hugging Face weights
    ├─ FlashInfer autotune
    ├─ FlashInfer JIT
    ├─ TileLang
    ├─ Triton
    ├─ TorchInductor
    ├─ CUDA Compute
    ├─ modelinfos
    └─ startup plan
    ↓
GitHub Release
portable fallback / disaster recovery
```

Modal Volume 是运行时 authoritative cache。

GitHub Release 不位于 GPU 启动关键路径，主要用于跨环境恢复和灾备。

## Benchmark 与技术记录

仓库中保留了完整优化过程：

- `startup-benchmark-2026-09-26.md`：早期冷启动、FlashInfer cache、CUDA Graph、吞吐基准。
- `startup-preweight-analysis-2026-09-27.md`：early prefetch、modelinfo、startup plan、122 秒级最新冷启动优化。
- `weight-load-benchmark-2026-09-26.md`：191 GiB 权重读取策略对比。
- `cache-backup.md`：runtime cache / GitHub Release 备份设计。

## 性能说明

当前约 **170 tok/s** 可以作为单 B300 长输出的实际参考值，但吞吐会受到以下因素影响：

- prompt 长度；
- 输出长度；
- reasoning / tool calling；
- 多模态输入；
- 并发数；
- speculative decoding acceptance；
- KV Cache 使用情况；
- CUDA Graph shape 命中；
- Modal 容器和存储后端状态。

因此 README 中的数字应理解为当前项目配置下的代表性实测，而不是固定 SLA。

## License

模型许可请以 `nota-ai/GLM-5.3-Flash-Nota-NVFP4` 上游模型仓库为准。

项目代码许可如需公开分发，建议在仓库中补充独立 `LICENSE` 文件。
