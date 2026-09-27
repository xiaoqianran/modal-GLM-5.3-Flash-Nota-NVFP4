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
24 项本地测试通过；已构建实际镜像并运行 CPU 能力检查及 CLI 冒烟。真实 B300 的各 span 耗时、modelinfos 二次
启动命中和整体提速仍需新容器验证，未将源码推断当作性能实测。
