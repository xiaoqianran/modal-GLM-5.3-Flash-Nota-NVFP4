# 权重加载前的启动耗时与缓存检查

## 历史 82.044 秒的边界

数据来自 `logs/startup-current.log`，同一个 task
`ta-01M3FK5WWJTKDPFMZWNW359JHR`，2026-09-27 03:31:00 +08:00 启动。

| 边界 | 本地时间 | 距进程启动 | 距上一边界 |
|---|---|---:|---:|
| vLLM process start | 03:31:00 | 0 s | — |
| API banner | 03:31:24 | 约 24 s | 约 24 s |
| 主模型架构解析完成 | 03:31:42 | 约 42 s | 约 18 s |
| MTP 架构解析完成 | 03:32:03 | 约 63 s | 约 21 s |
| EngineCore 初始化日志 | 03:32:17 | 76.893 s | 约 14 s |
| 权重加载开始 | 03:32:22 | 82.044 s | 5.151 s |

前几个时间是秒级日志边界，并不是函数测量。banner 前包含解释器启动、导入、CLI
初始化；两段架构解析之间也包含配置处理。MTP → EngineCore 区间里还有两次
multimodal warmup（日志分别报告 2.034 s、1.491 s），不能把整段算成子进程启动。
这轮历史配置仍启用了 sleep mode，与当前配置不同，不能直接当作新代码的性能基线。

这些日志不足以判断 CPU 配额是否是瓶颈，也不能给 Python import、注册表子进程
各自分配精确耗时。未增加 CPU 配额。

## 实际镜像核对

已运行 CPU-only `inspect_startup_runtime`，没有启动 B300 或部署生产服务。
[检查记录](https://modal.com/apps/xiaoyueliangqqq/main/ap-sbS3MiLxmUZHsXd9P9wbFh)。
随后完成 [CPU platform 下的 CLI --help 冒烟](https://modal.com/apps/xiaoyueliangqqq/main/ap-s4KoRN2dCpt3yoXNBot6pT)：
bootstrap 与已打补丁的 CLI 可以正常运行。CPU platform 只在诊断子进程内指定，
不会改变 serving。直接在无 GPU 容器运行 CUDA 镜像的完整 CLI 会在 device type
推断处失败，因此这个冒烟不是 GPU 初始化测试。

| 项目 | 镜像实际结果 |
|---|---|
| vLLM | `0.28.1rc1.dev580+g385dce36b` |
| Torch | `2.13.0+cu130` |
| Transformers | `5.16.1` |
| FlashInfer | `0.6.18` |
| vLLM console script 的解释器 | `/usr/bin/python3` |
| 模型架构检查缓存 | `VLLM_CACHE_ROOT/modelinfos` |
| startup plan schema | `1` |

Modal 添加的 `/usr/local/bin/python3` 使用另一套 `site-packages`，不能导入镜像
vLLM。上一轮 early-prefetch 启动入口里的裸 `python3` 已改为实际 console script
使用的 `/usr/bin/python3`。镜像构建会校验该解释器、注册表源码 hash 检查、plan
schema、实际 apply/save 日志及空闲显存检查；不匹配时停止构建。

原生注册表会读取 `modelinfos/*.json`，检查模型源码 hash，再重建 `_ModelInfo`。
未命中时才通过 `_run_in_subprocess` 检查模型。当前改动只把这一小目录从独立
Volume seed 复制到本地 cache，API Ready 后把变化的 JSON 原子写回并 commit。
seed 按库版本和 `_ModelInfo` schema 隔离，原生模型源码 hash 校验仍然执行。
整个 `VLLM_CACHE_ROOT` 保持本地目录。

核对的源码是日志对应的 commit，而不是用 0.30.0 的实现替换旧版本：

- [385dce36b 模型注册表](https://github.com/vllm-project/vllm/blob/385dce36b/vllm/model_executor/models/registry.py)
- [385dce36b startup plan](https://github.com/vllm-project/vllm/blob/385dce36b/vllm/v1/worker/startup_plan.py)
- [用户提供的 0.30.0 文档](https://docs.vllm.ai/en/v0.30.0/api/vllm/v1/worker/startup_plan/)作为对照参考。

## 下一次真实启动的测量

默认开启轻量 `[STARTUP_TRACE]`，包含 PID、统一的进程启动时间基准、wall time
和当前线程 CPU time。新增阶段：

| phase | 测量内容 |
|---|---|
| `vllm_cli_import` | 导入 vLLM CLI 入口 |
| `cli_command_imports` | CLI main 内导入各 command 的代码块 |
| `registry_inspect` | 主模型/MTP 各自的注册表检查，含 cache 检查和必要的子进程 |
| `registry_cache_lookup` | 原生缓存读取与校验；`actual_hit` 来自返回值 |
| `registry_subprocess` | 注册表子进程整段 round trip，含序列化、启动、导入、执行和返回 |
| `registry_child` / `model_class_import` | 检查子进程进入任务后、模型类导入的耗时 |
| `engine_process_start` | 父进程调用 `Process.start()` 的耗时与 child PID |
| `engine_entry` | 子进程进入 target；`spawn_to_entry_s` 含启动、解释器导入和反序列化 |
| `worker_init_device` | worker 设备/分布式环境初始化 |

`engine_init` 的整个 span 包含后续加载和 KV 初始化，不应把它的完整时长计入
“加载权重前”。`003_PREWEIGHT_MILESTONE` 单独标记 banner、两次架构解析、
EngineCore、模型加载、权重加载的边界。

注册表原本捕获子进程 stdout/stderr，现在仅转发其中的 trace 和可选 importtime
记录，便于按 PID 对齐。多个 span 有嵌套，不能直接相加；thread CPU time 也不
包含子进程或后台线程的 CPU 时间，不能据此单独判定 CPU 配额不足。

需要逐模块导入树时，在下一次部署前设置 `GLM53_PROFILE_IMPORTS=1`。
它只给 vLLM 进程及子进程设置 `PYTHONPROFILEIMPORTTIME=1`；默认关闭，避免日志量
和 profiling 开销影响普通启动。注册表子进程的导入记录标为 `REGISTRY_IMPORT_TIME`。

## startup plan 与 JIT 恢复

2026-09-27 12:24 的首轮 B300 实测发现：已有 startup plan 因 `current_free_memory < baseline`
的严格字节级比较被拒绝，但日志四舍五入后两者均为 `267.08 GiB`。因此镜像补丁保留原有
free-memory 安全门，同时允许默认 **256 MiB** 的启动抖动（可通过
`GLM53_STARTUP_PLAN_FREE_MEMORY_TOLERANCE_MIB` 调整，代码硬上限 1024 MiB）。只有 deficit
超过容差才回退 full profiling；容差内仍使用原 plan，并记录实际 deficit。这样没有把
co-tenant / 显存泄漏 / 大幅可用显存下降的保护删除。

plan 提交判断已从“启动前目录有没有文件”改为“有效 plan 的 fingerprint/内容是否
新增或改变”。仅 mtime/JSON 排版变化不会提交；损坏、临时文件、filename 与
fingerprint 不一致、无效 KV 大小不会被当作有效 plan。新 plan 或 modelinfos
变化任一发生时，API Ready 后后台提交 Volume，不受整体 cache sync 开关影响。

`018_STARTUP_PLAN_APPLIED` 只由原生实际 apply 日志触发；目录有文件和 `Saved`
均不算命中。没有收到 apply/reject 证据时报告 `actual_hit=unknown`，避免日志读取
滞后时误报。`018_STARTUP_METADATA_COMMIT_DONE` 才表示本次后台提交成功。

部署阶段的 CPU cache restore 记录 FlashInfer JIT 的 `available / absent / local /
unknown` 状态并显式 commit，身份包含 repo、release tag、镜像引用及 asset 名。
网络错误记录 unknown，不伪装成确认不存在。普通 GPU 启动优先使用 Volume；
optional JIT 的记录是 absent/unknown 时不发 GitHub 请求。

部署阶段的 `step_009_restore_runtime_caches` 总会刷新可用性。灾备时也可以在部署
前设置 `GLM53_CACHE_DISASTER_RECOVERY=1`，允许 GPU 路径绕过该记录查询；恢复后应
清除此开关。其他必要 cache 的 fallback 行为保持原有策略。

## 验证边界

本地回归覆盖跨进程非阻塞预读、失败重试、旧 plan 下新增配置、同一 plan 内容
变化、实际 apply/reject、modelinfos 本地暂存与增量同步、JIT 负缓存与灾备刷新。
25 项本地测试通过；已构建实际镜像并运行 CPU 能力检查及 CLI 冒烟。后续真实 B300
结果见下方“完整优化过程与实测结果”；旧的“仍待验证”结论已经关闭。

---

## 完整启动优化过程与实测结果

这一节按时间记录从最初分钟级 cold start 到当前 122 秒级基线的全过程。更早的 A/B/C/D
原始 benchmark 细节仍保留在 `startup-benchmark-2026-09-26.md`；这里记录的是最终决策、
为什么改、改完实际发生了什么。

### 1. 最早基线：启动不是单一“权重慢”

2026-09-26 A 组完整 cold start：

| 指标 | 实测 |
|---|---:|
| 191.01 GiB / 386 shard 首段 prefetch | 114.85 s |
| 第一段 Loading weights | 132.87 s |
| Model loading 总计 | 162.53 s |
| JIT kernel warmup | 5.06 s |
| FlashInfer autotune | 256.85 s |
| CUDA Graph capture | 158 s |
| engine profile/create KV/warmup | 571.72 s |
| vLLM process → API Ready | **829.884 s** |

结论从一开始就不是“只优化 191 GiB 权重”，而是同时存在 I/O、autotune、CUDA Graph、
模型初始化和 API 前置初始化等多个串行阶段。

### 2. 权重读取：确定 prefetch-16，而不是盲目加线程

独立权重实验显示：

| 策略 | 场景 | 代表结果 |
|---|---|---:|
| default | 原始偏冷启动 | 405.65 s |
| prefetch-16 | 完整 vLLM | 47.80 s |
| prefetch-16 | 独立 deployed 读取 | 43.208 s |
| prefetch-32 | 后端较热的新容器 | 12.259 s |
| eager | 后端较热的新容器 | 76.992 s |

32 线程没有稳定优于 16；计划内同条件测试甚至出现 16 线程 7.695 s、32 线程 13.221 s。
因此生产固定 `prefetch-16 + 16 MiB block`，不继续把线程数当作主要优化变量。

### 3. FlashInfer autotune：从 256~286 秒降到约 2 秒级

最初 FlashInfer autotune 每个新容器重新运行，典型为 256.847~286.193 s。随后把
autotune config 放到独立持久目录，并在生成后显式 commit 到 Modal Volume；GitHub Release
只做 portable fallback。C2/D 组新容器确认 `previous configs` 命中后，同一 cache key 的
autotune 降到约 2 秒级。

这一步把冷启动从 829.884 s 量级压到 D2 的 **149.504 s**，也是迄今收益最大的单项缓存
优化之一。

### 4. 生产 warmup 与 benchmark 解耦

旧生产路径曾在 API Ready 后自动跑长输出 benchmark（两道题、`max_tokens=8092`），会让
真实首请求与 benchmark 抢 B300。现在生产启动只保留 `16 tokens × 3` 的短 warmup；8092-token
题移动到独立 `step_017_bench_generation`，只有显式 benchmark 才运行。

这项不直接改变 `API Ready` 时间，但消除了“API 已 ready、GPU 却仍被内部 benchmark 占用”
的生产假就绪。

### 5. runtime cache：Modal Volume 主存储，GitHub Release 只做异步 fallback

当前稳定层级：

```text
HF weights / FlashInfer / TileLang / Triton / TorchInductor / CUDA Compute
        ↓
Modal Volume（生产 authoritative cache）
        ↓
GitHub Release（CPU worker 异步 portable fallback）
```

GPU 启动不等待 GitHub backup。CPU backup worker 使用 single-use container；每小时 dirty-check
全部 clean 时立即退出，不保持 idle 容器。FlashInfer JIT 尚无有效 `.so/.o/.cubin` 时记录
absent/unknown，普通 GPU 启动不会反复远程查询。

### 6. GPU Memory Snapshot / sleep mode：历史验证有效，但当前生产路径主动移除

历史版本曾实测 snapshot restore → API Ready 约 **5.5 s**，但 snapshot build 与 autotune
耦合、失败重试及 B300 生命周期管理带来了高复杂度和重复占卡风险。2026-09-27 起生产路径
明确移除：

```text
enable_memory_snapshot
enable_gpu_snapshot
/sleep?level=1
/wake_up
```

并确保 vLLM `enable_sleep_mode=False`。当前优化目标是在**普通新容器初始化路径**本身做到稳定、
可复现、可分析，而不是依赖 snapshot 命中。Modal 官方现在仍支持 GPU memory snapshot，且
vLLM 官方示例路线通常要求 sleep mode；这是未来独立实验，不混回当前生产基线。

### 7. 82 秒 pre-weight 黑盒：先测量，再优化

历史新容器里从 vLLM process start 到 weight load start 约 **82.044 s**。新增统一
`[STARTUP_TRACE]` 后拆出：CLI import、registry inspect/cache/subprocess、EngineCore spawn、
worker init 等边界，不再把 82 秒当作一个整体猜 CPU 或 I/O。

同时确认实际 console script 使用 `/usr/bin/python3`；Modal 添加的 `/usr/local/bin/python3`
site-packages 不同，不能作为 vLLM runtime interpreter。

### 8. early weight prefetch：只保留一个 prefetch，并提前到进程 0 秒

曾出现高风险结构：自定义 16-thread reader 与 vLLM 自己的 `prefetch-16` 同时读同一批
386 shards，可能形成双重 I/O。最终结构改成：

```text
/usr/bin/python3
  → 020_vllm_bootstrap
      ├─ 唯一 early prefetch 立即启动
      └─ Python / vLLM / registry / EngineCore 初始化并行继续

vLLM loader 到达原生 prefetch 点
  → 复用同一个 state
  → 不再启动第二个 reader
```

2026-09-27 12:21 首轮真实 B300：

```text
EARLY_WEIGHT_PREFETCH_REQUEST  0.248 s
WEIGHT_PREFETCH_DONE          12.156 s
weight_load_start             87.502 s
WEIGHT_PREFETCH_REUSE         state=done
```

即 191 GiB page-cache prefetch 完全隐藏在前置初始化里，没有制造新的串行 barrier。

### 9. 第一轮新架构实测：151.779 秒，暴露 registry 与 startup plan 两个问题

容器 `ta-01M3GHHN3D3KRCKMRNE0QVR88R`：

| 阶段 | 实测 |
|---|---:|
| API banner | 31.389 s |
| main architecture | 49.652 s |
| MTP architecture | 69.272 s |
| Engine init milestone | 83.098 s |
| weight load start | 87.502 s |
| main Loading weights | 24.71 s |
| MTP/secondary Loading weights | 4.21 s |
| model init total | 33.28 s |
| API Ready | **151.779 s** |

这一轮两个 registry cache 都 miss：

```text
main registry_subprocess = 17.994 s
MTP  registry_subprocess = 14.793 s
合计                    ≈ 32.787 s
```

API Ready 后生成并持久化 2 个 `modelinfos`。同时已有 startup plan 因严格字节级
`current_free_memory < baseline` 被拒绝，日志四舍五入后两边却都显示 `267.08 GiB`。

### 10. modelinfos 持久化 + startup-plan 有界显存容差

`modelinfos` 只暂存 vLLM 原生模型架构检查缓存，按 vLLM/Torch/Transformers/FlashInfer、
`_ModelInfo` schema 和 registry contract namespace 隔离；原生 model source hash 仍是最后
有效性检查，不把整个 `VLLM_CACHE_ROOT` 绑到 Volume。

startup plan 保留 free-memory OOM 安全门，但把“1 byte 少了就拒绝”改成默认 256 MiB 有界
抖动容差，代码硬上限 1024 MiB。超过容差仍 full profile；没有删除安全检查。

对应提交：

```text
0b1c9f1 optimize vllm weight prefetch coordination
a5eb1df optimize startup prefetch and cache persistence
1b02762 fix startup plan free memory tolerance
```

### 11. 第二轮新容器：两个 cache 与 startup plan 全部真实命中

容器 `ta-01M3GJGWGXBCY0RXEM6B6YGSMR`：

```text
[018_MODELINFO_STAGE] files=2
Glm5NextForConditionalGeneration registry_cache_lookup actual_hit=true
Glm5NextMTP                      registry_cache_lookup actual_hit=true
[018_STARTUP_PLAN_APPLIED] actual_hit=true fingerprint=da9a883c7383e0c8
Applying persisted startup plan ... Memory profiling will be skipped.
```

关键结果：

| 指标 | 第一轮 | 第二轮 | 改善 |
|---|---:|---:|---:|
| main architecture | 18.263 s 区间 | 0.102 s 区间 | 大幅下降 |
| MTP architecture | 19.620 s 区间 | 8.044 s 区间 | 明显下降 |
| weight_load_start | 87.502 s | **58.361 s** | -29.141 s |
| API Ready | 151.779 s | **122.302 s** | **-29.477 s / -19.4%** |
| Runtime Ready | 157.573 s | **126.606 s** | -30.967 s |

early prefetch 本轮 11.704 s 完成，仍在 weight loader 到达前约 46 秒完成；两次 loader 均
`WEIGHT_PREFETCH_REUSE state=done`。因此 prefetch 已不再是当前需要继续改的部分。

### 12. 122.302 秒基线剩余瓶颈

第二轮新容器已把瓶颈重新暴露为：

```text
vllm CLI / Python imports                  ≈ 30.64 s
main+MTP weights / model finalize          ≈ 34.63 s
engine startup / KV / warmup / graph       ≈ 26.69 s
其它 API finalization                      ≈ 数秒
```

其中 `init engine (profile, create kv cache, warmup model)` 仍为 **25.55 s**，但 startup plan
已经命中；说明这 25.55 秒不能再叫“memory profiling”。实际日志进一步拆出：

```text
12:40:18 startup plan apply
12:40:20 encoder cache: 32242-token budget，profile 1 个最大 video item
12:40:21 CUTLASS/FlashAttention 首次路径
12:40:29 Fused-MoE 初始化
12:40:31 KV cache ready
12:40:31 JIT kernel warmup（0.07 s）
12:40:32~33 FlashInfer autotune cache-hit（63/63，约 1.5 s）
12:40:39~43 CUDA Graph capture（日志报告 6 s）
12:40:44 engine init done（25.55 s）
```

因此当前 warmup 最大异常点不是 JIT warmup，而是**文本服务仍在初始化并 profile multimodal
encoder/video 路径**，其次才是 CUDA Graph capture。

## 当前进行中的下一阶段优化（尚待新 B300 验证）

### A. import：serve-only 路线已实测否决

曾尝试跳过 generic CLI 中的 benchmark / collect_env / launch / run_batch 等模块，直接构建
`ServeSubcommand`。热容器里 parser 路径一度看到 `11.961 s → 9.909 s`，但这不是 cold start。

随后在两个全新 Modal CPU 容器、相同镜像和 production argv 下重新 A/B：

```text
serve-only import + parser: 31.572 s + 0.301 s
generic CLI:                0.004 s + 22.587 s
```

因此 serve-only 在真正的新容器里**反而更慢约 9 秒**。该代码已撤回，生产继续使用 vLLM
原生 generic CLI。这个实验也说明不能用热容器 import 数字替代冷启动结论。

### B. import：镜像构建时预编译被 patch 的 vLLM 源文件

`021_patch_startup_observability.py` 会在 image build 修改 registry、startup plan、engine、
worker、model runner、CLI 源文件。修改源码会使原 `.pyc` 失效。现在 patch 完成后立即对这些
**确切文件**执行 `py_compile`，把 source→bytecode 编译成本从新容器首次 import 移到一次性
image build。它不改变运行语义，只减少可避免的首次解释器工作。

同一个 CPU smoke harness 下已有初步 A/B：

```text
预编译前 cli_command_imports = 28.037 s
预编译后 cli_command_imports = 22.587 s
初步减少                    =  5.450 s
```

这是比 serve-only 更可信的方向，因为没有改 CLI 语义；不过 CPU 容器与 B300 的文件/CPU 环境
并不完全相同，最终收益仍以新 B300 的 `vllm_cli_import + cli_command_imports` 为准。

### C. warmup：必须保留多模态，不能用 `--language-model-only`

当前模型 config 明确包含 `vision_config`、image/video token，并且本部署需要保留多模态能力。
因此曾提出的 `--language-model-only` 优化已经**撤回，不进入生产**。

这意味着日志中的 encoder cache / 最大 video item profile 不能简单删除。后续 warmup 优化必须
在**完整保留图片/视频能力**的前提下进行，例如优化缓存、编译产物、profile 复用或初始化顺序，
不能通过裁掉 multimodal tower 来换冷启动时间。

### D. 新增 warmup 细粒度 trace

下一轮会额外记录：

```text
determine_available_memory
initialize_from_config
compile_or_warm_up_model
model_profile_run
model_capture
```

这样可以直接回答 25.55 秒里完整多模态路径分别花在多少 KV 初始化、encoder profile、
kernel warmup 和 CUDA Graph，
再决定是否值得调整 capture sizes。当前没有为了省 4~6 秒而减少 CUDA Graph shapes，因为这
可能损失稳态 decode 性能，与“稳态 token/s 优先”的目标冲突。

### E. 暂不默认改 multiprocessing

EngineCore `spawn_to_entry_s` 本轮约 11.176 s。vLLM 支持 `fork/forkserver`，理论上可能利用
父进程已导入的模块减少第二次解释器启动；但 CLI 默认 `spawn` 是为了 CUDA/线程兼容安全。
当前不把这类高风险变化设成生产默认。先验证 `.pyc` 与新增 warmup trace，再根据下一轮数据
继续优化完整多模态路径，并决定是否单独做 forkserver 实验。
