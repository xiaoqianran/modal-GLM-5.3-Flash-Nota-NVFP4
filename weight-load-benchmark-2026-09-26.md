# GLM-5.3 Flash 权重加载基准测试 — 2026-09-26

模型：nota-ai/GLM-5.3-Flash-Nota-NVFP4  
Revision：c5fc7f5ef0447ab030559bb4b08861a36dbbe847  
Checkpoint：191.01 GiB / 386 个 safetensors 分片  
文件系统：Modal Volume，以 9P 挂载  
测试模式：deployed app，`max_containers=1`；每组计划测试都通过 `stop -> deploy -> remote()` 强制使用新容器。  
读取块大小：16 MiB

## 计划测试对比

| 策略 | 线程数 | 耗时（秒） | 吞吐量（GiB/s） | 说明 |
|---|---:|---:|---:|---|
| prefetch | 16 | 7.695 | 24.822 | 新容器；受此前实验影响，后端缓存已经较热 |
| prefetch | 32 | 13.221 | 14.447 | 新容器；后端缓存已经较热 |
| eager | 1 | 222.174 | 0.860 | 新容器；包含完整分片读取和 `safetensors.load` 解析 |

## 之前的参考测试数据

| 策略 | 测试场景 | 权重加载耗时（秒） | 吞吐量 / 说明 |
|---|---|---:|---|
| default | 原始 vLLM 偏冷启动 | 405.65 | ~0.47 GiB/s |
| prefetch-16 | 完整 vLLM 测试 | 47.80 | 第一遍权重加载 |
| prefetch-16 | 独立 deployed 权重读取测试 | 43.208 | 4.421 GiB/s |
| prefetch-16 | 同一容器，page cache 已热 | 5.463 | 34.966 GiB/s |
| prefetch-32 | 新容器，但后端缓存已热 | 12.259 | 15.581 GiB/s |
| eager | 新容器，但后端缓存已热 | 76.992 | 2.481 GiB/s |
| eager | 同一容器，page cache 已热 | 36.343 | 5.256 GiB/s |
| prefetch-32 | 同一容器，page cache 已热 | 2.575 | 74.190 GiB/s |

## 结论

1. 对当前 386 个 safetensors 分片、Modal Volume 9P 文件系统这一场景，`prefetch` 明显优于 `eager`。
2. `prefetch-16` 已经在完整 vLLM 测试和独立 I/O 测试中重复得到较快结果，稳定性较好。
3. `prefetch-32` 并没有稳定快于 16 线程；在计划内统一测试中，32 线程反而更慢：13.221 秒，对比 16 线程的 7.695 秒。
4. Modal 后端缓存状态会显著影响测试结果。使用新容器并不代表底层存储后端一定处于冷缓存状态。
5. 同一容器内的 Linux page cache 会让重复读取速度大幅提升，因此必须把“同容器热缓存”结果和“偏冷启动”结果分开记录，不能混为一谈。
