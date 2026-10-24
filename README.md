# Multi-Head Causal Self-Attention — CUDA 算子融合与性能优化

从朴素三 Kernel、手写 Flash Attention，到 cuBLAS Tensor Core、FP16/BF16 混合精度与 cuDNN 融合 SDPA，记录 GPT-2 多头因果自注意力前向传播的 **11 个版本迭代**。

本项目重点不是单纯调用高层框架接口，而是沿着 Attention 的真实计算链路，逐步分析：

- 朴素实现为什么慢，瓶颈究竟在计算还是显存；
- Kernel Fusion、Warp 协作、Shared Memory、在线 Softmax 如何减少全局内存往返；
- Flash Attention 为什么能把注意力矩阵的显存复杂度从 `O(T²)` 降到 `O(T)`；
- 为什么手写矩阵乘最终远不如 cuBLAS / cuDNN 的 Tensor Core 实现；
- FP16/BF16 混合精度、Stride 零拷贝和官方 SDPA 融合还能继续带来多少收益。

> **默认测试配置**：GPT-2 Small，`B=8`，`T=1024`，`C=768`，`NH=12`，`HS=C/NH=64`，TF32 开启，Profile Block Size = 128。
>
> **最终结果**：FP32 朴素三 Kernel 总耗时 **421.15 ms**；全低精度 cuDNN SDPA 总耗时 **0.97 ms**，端到端相对加速约 **434.18×**。

---

## 目录

- [算子简介](#算子简介)
- [张量布局与显存开销](#张量布局与显存开销)
- [优化主线](#优化主线)
- [性能总览](#性能总览)
- [编译、运行与 Profiling](#编译运行与-profiling)
- [版本演进详解](#版本演进详解)
- [关键机制拆解](#关键机制拆解)
- [路径总结与规律](#路径总结与规律)
- [已知注意事项](#已知注意事项)
- [后续优化方向](#后续优化方向)
- [参考资料](#参考资料)

---

## 算子简介

多头因果自注意力（Multi-Head Causal Self-Attention）位于 GPT 类模型的每个 Transformer Block 中。

输入 `inp` 已经在最后一维拼接了 Q、K、V：

```text
inp = [Q | K | V]
```

每个 Attention Head 独立执行：

```text
S = Q @ Kᵀ / √HS
S = causal_mask(S)
P = softmax(S)
O = P @ V
```

其中：

- `Q`：Query，表示当前 token 正在“查询”什么信息；
- `K`：Key，表示每个历史 token 可以提供什么信息；
- `V`：Value，表示真正被聚合的内容；
- `HS`：单头维度，本项目中 `HS = 768 / 12 = 64`；
- `1/√HS`：缩放因子，防止点积值随维度增大而过大，导致 Softmax 饱和；
- `causal_mask`：第 `t` 个 token 只能关注 `0..t`，不能关注未来 token。

完整计算链路包含两次矩阵乘和一次行级 Softmax：

```text
第一次 GEMM：Q @ Kᵀ
逐行归一化：Softmax
第二次 GEMM：P @ V
```

这也是 Attention 优化的核心矛盾：

1. `Q @ Kᵀ` 和 `P @ V` 是规则矩阵乘，适合 Tensor Core；
2. Softmax 带行最大值、指数、求和和因果掩码，不完全等同于 GEMM；
3. 中间注意力矩阵形状为 `[T, T]`，序列变长时显存和 HBM 流量快速膨胀；
4. Flash Attention 必须依赖在线 Softmax，才能把完整 `[T,T]` 矩阵拆成 Tile 并在片上完成计算。

---

## 张量布局与显存开销

### 张量形状

| 张量 | 形状 | 说明 |
|---|---|---|
| `inp` | `[B, T, 3C]` | Q/K/V 拼接输入，逻辑布局为 `[B,T,3,NH,HS]` |
| `out` | `[B, T, C]` | Attention 输出，逻辑布局为 `[B,T,NH,HS]` |
| `Q/K/V` | `[B, NH, T, HS]` | permute 后头优先布局，便于批量 GEMM |
| `preatt` | `[B, NH, T, T]` | Softmax 前的注意力分数 |
| `att` | `[B, NH, T, T]` | Softmax 后的注意力权重 |
| `L` | `[B, NH, T]` | Flash Attention 写出的 logsumexp，主要供反向传播使用 |

### 为什么注意力矩阵很贵

在本项目配置下：

```text
B * NH * T * T
= 8 * 12 * 1024 * 1024
= 100,663,296 个元素
```

FP32 下，单个注意力矩阵约为：

```text
100,663,296 * 4 bytes ≈ 384 MiB
```

朴素实现和 cuBLAS 分离式实现通常同时需要 `preatt` 和 `att` 两个矩阵：

```text
384 MiB + 384 MiB = 768 MiB
```

这还不包括 Q、K、V、输出和 cuBLAS Workspace。

因此，Attention 的主要优化目标不只是提高 FLOPS，更是：

- 避免完整 `[T,T]` 中间矩阵落回 HBM；
- 减少 Q/K/V 和中间矩阵的重复读取；
- 让 GEMM、Softmax、Mask、Scale 在片上流水起来；
- 使用低精度 IO 减少带宽压力；
- 避免不必要的物理 permute。

---

## 优化主线

本项目的 11 个版本可以分成三条技术路线。

### 第一阶段：手写 CUDA Kernel，理解硬件映射

- **v1**：QK、Softmax、PV 三个独立 Kernel；
- **v2**：一个 Warp 处理一行，融合 QK / Softmax / PV；
- **v3**：列优先朴素 Flash Attention，重点演示在线 Softmax；
- **v4**：FlashAttention-2 行优先调度，Q Tile 在外、KV Tile 在内；
- **v5**：Warp 级 split-Q，每个 Warp 负责连续多行 Q。

### 第二阶段：使用成熟 GEMM 库 + 自定义融合 Softmax

- **v6**：cuBLAS TF32 Strided Batched GEMM + 独立 Scale/Mask + Softmax；
- **v7**：cuBLAS GEMM + 融合 Scale/Mask/Online Softmax；
- **v8**：FP16/BF16 IO，FP32 内部累加；
- **v9**：跳过 permute/unpermute，单独测量低精度核心算子下界。

### 第三阶段：官方融合 SDPA

- **v10**：cuDNN Flash Attention，包含 FP32↔低精度转换；
- **v11**：全低精度 cuDNN SDPA，假设输入输出已经是 FP16/BF16。

---

## 性能总览

> 所有性能数据均采集自 Nsight Compute，命令口径为 `ncu --set full --cache-control all`。
>
> NCU 采集包含硬件计数器采样开销，绝对耗时适合做同口径横向比较；正式 benchmark 还应结合 CUDA Event 多次重复运行。
>
> 图中的 Compute 和 Memory 是两个独立维度的硬件吞吐指标，二者不是相加为 100% 的关系。

![GPT-2 Attention NCU 性能总览](images/Attention.png)

### 版本总耗时

| 版本 | 核心实现 | Kernel 数 | 总耗时 [ms] | 相对 v1 加速 | 关键结论 |
|:---:|---|:---:|---:|---:|---|
| v1 | 朴素三 Kernel：QK / Softmax / PV | 3 | 421.15 | 1.00× | FP32 基线，HBM 往返严重 |
| v2 | Warp 级融合 Kernel | 1 | 37.61 | 11.20× | Warp 协作和 Kernel Fusion 收益明显 |
| v3 | 列优先朴素 Flash Attention | 3 | 644.68 | 0.65× | 教学版实现，比 v1 慢约 1.53× |
| v4 | FlashAttention-2 行优先调度 | 3 | 413.17 | 1.02× | 调度正确，但手写矩阵乘仍慢 |
| v5 | FA2 Warp 级 split-Q | 3 | 198.80 | 2.12× | Warp 并行有效，但没有 Tensor Core |
| v6 | cuBLAS TF32 + 独立 Softmax | 6 | 9.14 | 46.08× | 成熟 GEMM 带来决定性跳变 |
| v7 | cuBLAS + 融合在线 Softmax | 5 | 5.06 | 83.23× | 消掉 Scale Kernel，减少访存遍数 |
| v8 | FP16/BF16 IO + FP32 累加 | 5 | 2.89 | 145.73× | 低精度 IO 和 Tensor Core 同时生效 |
| v9 | 跳过 permute 的低精度核心 | 3 | 2.27 | 185.53× | 理想核心下界，不是完整部署路径 |
| v10 | cuDNN SDPA + 精度转换 | 3 | 1.73 | 243.44× | 官方融合 Flash Attention |
| v11 | 全低精度 cuDNN SDPA | 1 | 0.97 | 434.18× | 本轮最快，代表全低精度网络链路 |

### Kernel 级耗时分解

| 版本 | Kernel 耗时分解 [ms] | 小计 [ms] |
|:---:|---|---:|
| v1 | QK `84.47` + Softmax `12.23` + PV `324.45` | **421.15** |
| v2 | Fused Attention `37.61` | **37.61** |
| v3 | permute `0.46` + 列优先 Flash `643.99` + unpermute `0.23` | **644.68** |
| v4 | permute `0.46` + 行优先 FA2 `412.48` + unpermute `0.23` | **413.17** |
| v5 | permute `0.46` + Warp FA2 `198.11` + unpermute `0.23` | **198.80** |
| v6 | permute `0.46` + QK `1.83` + scale `2.11` + softmax `3.19` + PV `1.32` + unpermute `0.23` | **9.14** |
| v7 | permute `0.46` + QK `1.83` + fused softmax `1.22` + PV `1.32` + unpermute `0.23` | **5.06** |
| v8 | lowp permute `0.39` + QK `0.93` + lowp softmax `0.67` + PV `0.67` + lowp unpermute `0.23` | **2.89** |
| v9 | QK `0.93` + lowp softmax `0.67` + PV `0.67` | **2.27** |
| v10 | FP32→lowp `0.57` + cuDNN SDPA `0.97` + lowp→FP32 `0.19` | **1.73** |
| v11 | cuDNN SDPA `0.97` | **0.97** |

> 图中 11 个版本小计直接相加为 1737.47 ms，NCU 总表显示 1737.46 ms，0.01 ms 差异来自四舍五入。

---

## 编译、运行与 Profiling

### 基础版本：v1–v9

基础版本只依赖 CUDA Runtime 和 cuBLAS：

```bash
nvcc -O3 --use_fast_math -lcublas -lcublasLt attention_forward.cu -o attention_forward
```

运行指定版本：

```bash
./attention_forward 1   # 朴素三 Kernel
./attention_forward 6   # cuBLAS + 自定义 Softmax
./attention_forward 9   # 低精度核心下界
```

### 启用 cuDNN：v10–v11

v10、v11 使用 cuDNN Frontend 的计算图 API，需要安装 `cudnn-frontend`：

```bash
nvcc -I/path/to/cudnn-frontend/include \
     -DENABLE_CUDNN \
     -O3 --use_fast_math \
     -lcublas -lcublasLt -lcudnn -lnvrtc \
     attention_forward.cu -o attention_forward
```

### 正确性验证与普通 Benchmark

不带 `profile` 参数时，程序会：

1. 先运行 CPU 参考实现；
2. 遍历 block size：`32、64、128、256、512`；
3. v1–v7 使用 `1e-3` 误差阈值；
4. v8 及以后使用 `1e-2` 误差阈值；
5. 每个 block size 重复运行 20 次统计耗时。

### Nsight Compute 采集

基础版本 v1–v9：

- 每轮共 32 个 Kernel；
- 预热 2 轮，共跳过 64 次 Kernel Launch；
- 正式采集第 3 轮的 32 个 Kernel。

```bash
nvcc -O3 -lineinfo --use_fast_math -lcublas -lcublasLt attention_forward.cu -o attention_forward

ncu --set full --cache-control all \
    --launch-skip 64 --launch-count 32 \
    -o attention_profile_128 \
    ./attention_forward 0 profile
```

启用 cuDNN 后，v1–v11 每轮共 36 个 Kernel，预热 2 轮后跳过 72 次 Launch：

```bash
ncu --set full --cache-control all \
    --launch-skip 72 --launch-count 36 \
    -o attention_profile_cudnn_128 \
    ./attention_forward 0 profile
```

也可以使用 Nsight Systems 查看系统级调度：

```bash
nsys profile --stats=true -o attention_nsys ./attention_forward 0 profile
```

---

## 版本演进详解

## v1 — 朴素三 Kernel：最直观的 CPU 移植

v1 将注意力拆成三个独立 Kernel：

```text
attention_query_key_kernel1：计算 Q @ Kᵀ / √HS
attention_softmax_kernel1  ：逐行 Softmax
attention_value_kernel1    ：计算 P @ V
```

### 并行映射

| Kernel | 线程映射 | 主要工作 |
|---|---|---|
| QK Kernel | 一个线程计算一个注意力分数 | 反解 `(b,h,t,t2)`，计算长度为 HS 的点积 |
| Softmax Kernel | 一个线程处理一整行 | 串行求 max、exp sum、归一化 |
| Value Kernel | 一个线程处理一个输出向量 | 遍历所有历史 token，对 V 加权求和 |

### 性能数据

| Kernel | Duration [ms] | Compute [%] | Memory [%] | Registers |
|---|---:|---:|---:|---:|
| QK | 84.47 | 12.49 | 97.54 | 38 |
| Softmax | 12.23 | 6.43 | 74.14 | 40 |
| PV | 324.45 | 7.02 | 73.71 | 36 |
| **合计** | **421.15** | — | — | — |

### 瓶颈

1. **QK 中每个点积都重新读取 Q/K**，线程之间缺少向量级复用；
2. **Softmax 一个线程串行处理一行**，并行度不足；
3. **PV Kernel 最慢**：每个线程对长度为 HS 的输出向量串行遍历所有历史 token；
4. `preatt` 和 `att` 都完整写回 HBM，再被下一个 Kernel 读回来；
5. 大部分 Kernel 的 Memory 指标显著高于 Compute，属于典型访存受限。

其中 PV Kernel 单独耗时 `324.45 ms`，占 v1 总耗时约 77%，是 v1 最大的性能瓶颈。

---

## v2 — Warp 级融合 Kernel：一个 Warp 处理一行

v2 将 QK、Softmax、PV 融合到一个 Kernel 中：

- 一个 Warp 固定 32 个线程；
- 一个 Warp 处理一个 `(b,h,t)` 行；
- Warp 内线程协作计算 Q/K 点积；
- 使用 Cooperative Groups 的 `cg::reduce` 做 Warp 规约；
- Grid 为 `(ceil(T / warps_per_block), NH, B)`。

### 性能数据

| Kernel | Duration [ms] | Compute [%] | Memory [%] | Registers | Block Size |
|---|---:|---:|---:|---:|---:|
| attention_forward_kernel2 | 37.61 | 93.25 | 93.25 | 40 | 128 |

相比 v1 加速约 **11.20×**。

### 为什么快

- 三个 Kernel 合并为一个，减少 Kernel Launch 开销；
- 点积由 Warp 内 32 个线程共同完成；
- Softmax 的指数和可以在 Warp 内并行规约；
- 输出向量的不同维度由不同 Lane 负责。

### 仍然存在的问题

v2 仍然会把 `preatt` 和 `att` 写入全局内存：

```text
QK 分数写 preatt → 读 preatt 算 Softmax → 写 att → 读 att 算 PV
```

因此它只是做了 Kernel Fusion，还没有像 Flash Attention 那样让注意力分数完全在片上流动。

---

## v3 — 朴素 Flash Attention：列优先，重点演示在线 Softmax

v3 开始引入 Tile 分块：

- `Bc = Br = 32`；
- Q、K、V、S Tile 放入 Shared Memory；
- 外层循环遍历 KV Block，内层循环遍历 Q Block；
- 每行维护 `m` 和 `l`，即当前最大值和指数和；
- 每处理一个 KV Tile，就按新的最大值修正历史结果。

伪代码如下：

```text
for j in KV blocks:
    load Kj, Vj to shared memory
    for i in Q blocks:
        load Qi
        Sij = Qi @ Kjᵀ / √HS
        Pij = exp(Sij - rowmax)
        update m, l
        Oi = merge(Oi, Pij @ Vj)
```

### 性能数据

| Kernel | Duration [ms] | Compute [%] | Memory [%] | Registers |
|---|---:|---:|---:|---:|
| permute | 0.46 | 40.12 | 85.70 | 18 |
| 列优先 Flash Kernel | 643.99 | 6.44 | 55.03 | 40 |
| unpermute | 0.23 | 83.20 | 47.40 | 26 |
| **合计** | **644.68** | — | — | — |

### 为什么实现了 Flash 思想却最慢

v3 是教学演示版本，性能慢主要有四个原因：

1. **列优先调度**：固定 KV Tile，遍历所有 Q Tile，导致每个 Q 行的 `m/l/O` 每处理一个 KV Block 都要写回 HBM；
2. **序列维度没有充分并行**：Grid 只有 `(B, NH)`，没有对 Q Tile 开 Block；
3. **矩阵乘仍是手写标量循环**，没有使用 Tensor Core；
4. Block 只有 32 个线程，硬件并行度和延迟隐藏能力不足。

这个版本的价值在于清楚展示 Flash Attention 的数学推导，尤其是在线 Softmax 如何合并历史 Tile；它不是生产级实现。

---

## v4 — FlashAttention-2：行优先 + 序列维度并行

v4 将循环顺序改为：

```text
for each Q block:
    load Qi once
    init row_m, row_l, O_acc
    for each KV block:
        load Kj, Vj
        Sij = Qi @ Kjᵀ
        update m, l
        O_acc = merge(O_acc, Pij @ Vj)
    O = O_acc / l
```

### 相对 v3 的关键变化

| 对比项 | v3 列优先 | v4 行优先 |
|---|---|---|
| 外层循环 | KV Block | Q Block |
| 内层循环 | Q Block | KV Block |
| Q 数据读取 | 随 KV 外层反复加载 | 每个 Q Tile 只加载一次 |
| 行状态 | 反复读写 HBM | 驻留 Shared Memory / 寄存器 |
| Grid | `(B, NH)` | `(Tr, NH, B)` |
| 归一化 | 每个 Tile 后都做除法 | 遍历完所有 KV 后统一归一化 |

### 性能数据

| Kernel | Duration [ms] | Compute [%] | Memory [%] | Registers |
|---|---:|---:|---:|---:|
| permute | 0.46 | 40.12 | 85.41 | 18 |
| 行优先 FA2 Kernel | 412.48 | 10.20 | 83.68 | 40 |
| unpermute | 0.23 | 83.13 | 47.23 | 26 |
| **合计** | **413.17** | — | — | — |

v4 比 v3 快 **1.56×**，但总耗时仍然和 v1 基本持平。

原因是调度方向虽然正确了，但最耗时的矩阵乘仍然是每个线程串行遍历 HS 和 KV Tile，没有 Tensor Core，也没有成熟 GEMM Kernel 中的软件流水线和寄存器分块。

### 延迟归一化

v4 在内层循环中只维护未归一化分子，不提前做除法：

```text
O_acc = O_acc * exp(m_old - m_new) + P_block @ V_block * exp(m_block - m_new)
```

所有 KV Tile 处理完后才执行：

```text
O = O_acc / l
```

这样每行只做一次最终除法，减少了高频除法开销。

---

## v5 — FlashAttention-2 Warp 优化：每个 Warp 负责 8 行 Q

v5 进一步改变任务划分：

```text
WARPS_PER_BLOCK = 4
ROWS_PER_WARP   = 8
Br              = 4 * 8 = 32
Bc              = 32
```

- 一个 Block 有 4 个 Warp；
- 每个 Warp 独立负责连续 8 行 Q；
- Lane 0–31 分别对应 KV Tile 中的 32 个列；
- 每个 Warp 内部完成 Max、Sum 规约；
- Warp 之间不通信、不争用同一组行状态；
- Grid 仍为 `(Tr, NH, B)`。

### 性能数据

| Kernel | Duration [ms] | Compute [%] | Memory [%] | Registers |
|---|---:|---:|---:|---:|
| permute | 0.46 | 40.18 | 85.49 | 18 |
| Warp FA2 Kernel | 198.11 | 22.21 | 88.81 | 110 |
| unpermute | 0.23 | 83.15 | 47.15 | 26 |
| **合计** | **198.80** | — | — | — |

v5 比 v4 快 **2.08×**。

### 寄存器压力

v5 的寄存器数从 v4 的 40 上升到 **110**。每个 Warp 都要维护多行 Q 的：

- 当前最大值；
- 当前指数和；
- Score Tile；
- 输出累加值。

寄存器越多，单个线程私有状态越多，但也可能降低 Occupancy。v5 的 Memory 指标达到 88.81%，说明它仍然主要受片外访存和手写 GEMM 效率限制。

### v5 的定位

v5 是手写 Flash Attention 的较好教学版本，但和 v6 相比仍然慢：

```text
v5：198.80 ms
v6：  9.14 ms
v6 比 v5 快约 21.75×
```

这说明在 Attention 这种包含大矩阵乘的算子中，手写 FMA 循环很难替代 cuBLAS、CUTLASS 或 cuDNN 中的 Tensor Core Kernel。

---

## v6 — cuBLAS TF32 + 自定义 Softmax：工业框架式基线

v6 将两次计算密集的矩阵乘交给 cuBLAS：

```text
permute
  → cuBLAS: Q @ Kᵀ
  → scale + causal mask
  → block-wise softmax
  → cuBLAS: P @ V
  → unpermute
```

两次 GEMM 使用：

```cpp
cublasSgemmStridedBatched
```

它一次提交 `B * NH` 个小矩阵乘，比在 Host 端循环调用单矩阵 GEMM 开销更低。

在 SM 8.0 及以上设备中，代码开启：

```cpp
CUBLAS_TF32_TENSOR_OP_MATH
```

因此 FP32 接口下的 cuBLAS GEMM 可以使用 TF32 Tensor Core。

### 性能数据

| Kernel | Duration [ms] | Compute [%] | Memory [%] | Registers |
|---|---:|---:|---:|---:|
| permute | 0.46 | 40.22 | 85.61 | 18 |
| cuBLAS QK | 1.83 | 71.19 | 62.79 | 255 |
| scale + mask | 2.11 | 59.30 | 73.28 | 16 |
| softmax kernel5 | 3.19 | 49.98 | 64.94 | 23 |
| cuBLAS PV | 1.32 | 96.45 | 89.25 | 90 |
| unpermute | 0.23 | 83.11 | 47.11 | 26 |
| **合计** | **9.14** | — | — | — |

相比 v5，v6 快约 **21.75×**；相比 v1，v6 快约 **46.08×**。

### 瓶颈转移

v6 中两个 GEMM 合计：

```text
1.83 + 1.32 = 3.15 ms
```

Scale 和 Softmax 合计：

```text
2.11 + 3.19 = 5.30 ms
```

Scale + Softmax 占 v6 总耗时约 **58%**，成为新的瓶颈。

这符合典型优化规律：当最重的矩阵乘被库优化到足够快后，原来“顺便执行”的逐元素 Scale、Mask、Softmax 和布局转换会变成主要成本。

### v6 的限制

v6 仍然显式分配并写回完整 `[T,T]` 注意力矩阵，显存复杂度仍是 `O(T²)`，长序列下会受限。

---

## v7 — cuBLAS + 融合在线 Softmax

v7 保持两次 cuBLAS GEMM 不变，但将：

- Scale；
- Causal Mask；
- Online Softmax；
- float4 向量化读取；
- `__ldcs` / `__stcs` 流式访存；

融合进 `softmax_forward_kernel6`。

### 在线 Softmax 的核心递推

处理一个新 Tile 时：

```text
old_m：旧最大值
old_l：旧指数和
cur_m：当前 Tile 最大值
cur_l：当前 Tile 指数和

new_m = max(old_m, cur_m)
new_l = old_l * exp(old_m - new_m)
      + cur_l * exp(cur_m - new_m)
```

旧的累加结果不需要重新计算指数，只要乘一个修正系数即可折算到新的最大值基准。

### 性能数据

| Kernel | Duration [ms] | Compute [%] | Memory [%] | Registers |
|---|---:|---:|---:|---:|
| permute | 0.46 | 40.30 | 85.46 | 18 |
| cuBLAS QK | 1.83 | 71.14 | 62.66 | 255 |
| fused online softmax | 1.22 | 30.81 | 84.10 | 36 |
| cuBLAS PV | 1.32 | 96.45 | 89.25 | 90 |
| unpermute | 0.23 | 83.17 | 47.16 | 26 |
| **合计** | **5.06** | — | — | — |

v7 比 v6 快约 **1.81×**。

v6 的 Scale + Softmax 是 `5.30 ms`，v7 融合后只有 `1.22 ms`。收益来自：

1. 省掉独立 Scale Kernel；
2. 不再需要单独对完整注意力矩阵做一遍缩放读写；
3. 在线 Softmax 减少行遍历次数；
4. float4 一次处理 4 个 FP32，提高访存效率；
5. Streaming Load/Store 减少缓存污染。

### 仍然存在的限制

v7 仍然要把 Softmax 后的 `att` 交给第二个 cuBLAS GEMM，因此完整注意力矩阵仍然物化在 HBM 中。它优化了 Softmax，但还不是完全 Flash Attention 式的端到端融合。

---

## v8 — FP16/BF16 混合精度：IO 用低精度，累加用 FP32

v8 使用 `cublasGemmStridedBatchedEx`，显式指定：

- A/B/C 矩阵的数据类型；
- GEMM 内部计算精度；
- 支持 FP16/BF16 输入输出；
- 内部可使用 FP32 累加。

代码通过 `floatX` 宏支持 FP16/BF16；本次 NCU 中的 `bf16`、`tst_mma` 内核名显示实际走的是 BF16 Tensor Core 路径。

### 低精度 Softmax 的原则

Softmax 对指数和最大值很敏感，不能直接用低精度做完整累加。v8 的做法是：

```text
加载 FP16/BF16
  → 转成 FP32
  → 求 max / exp / sum
  → 归一化
  → 转回 FP16/BF16
```

类型转换主要发生在寄存器中，成本通常远低于 HBM 访问。

### 性能数据

| Kernel | Duration [ms] | Compute [%] | Memory [%] | Registers |
|---|---:|---:|---:|---:|
| lowp permute | 0.39 | 66.48 | 75.39 | 26 |
| 低精度 cuBLAS QK | 0.93 | 69.86 | 65.92 | 255 |
| lowp online softmax | 0.67 | 79.08 | 75.43 | 26 |
| 低精度 cuBLAS PV | 0.67 | 95.65 | 88.13 | 96 |
| lowp unpermute | 0.23 | 82.85 | 30.01 | 26 |
| **合计** | **2.89** | — | — | — |

v8 比 v7 快约 **1.75×**，相对 v1 加速约 **145.73×**。

低精度同时带来两类收益：

1. **显存带宽减半**：FP16/BF16 每个元素 2 字节，FP32 为 4 字节；
2. **Tensor Core 吞吐更高**：低精度矩阵乘更容易跑满专用硬件。

### 布局转换成本

v8 中 permute 和 unpermute 合计：

```text
0.39 + 0.23 = 0.62 ms
```

占 v8 总耗时约 **21.5%**。当核心 GEMM 和 Softmax 已经足够快后，布局转换成为不可忽略的成本。

---

## v9 — 跳过 permute/unpermute：低精度核心下界

v9 直接调用 v8，并传入 `skip_permute=true`：

```text
QK GEMM → fused softmax → PV GEMM
```

### 性能数据

| Kernel | Duration [ms] | Compute [%] | Memory [%] | Registers |
|---|---:|---:|---:|---:|
| 低精度 cuBLAS QK | 0.93 | 69.84 | 66.02 | 255 |
| lowp online softmax | 0.67 | 79.01 | 75.82 | 26 |
| 低精度 cuBLAS PV | 0.67 | 94.93 | 95.15 | 96 |
| **合计** | **2.27** | — | — | — |

v9 比 v8 快约 **1.27×**。

### 为什么 v9 不是完整部署版本

v9 假设：

- Q/K/V 已经是低精度；
- Q/K/V 已经是 `[B,NH,T,HS]` 头优先布局；
- 输出不需要立即转回 `[B,T,NH,HS]`。

这些假设在真实模型中不一定成立。它的价值是给出“矩阵乘 + Softmax”核心部分的性能下界，用来量化布局转换和精度转换的成本。

因此，v9 不能直接与包含完整输入输出处理的版本做简单优劣对比。

---

## v10 — cuDNN 融合 SDPA：Stride 零拷贝 + FP32 转换

v10 使用 cuDNN Frontend 构建 SDPA 计算图。cuDNN 在内部融合：

```text
Q @ Kᵀ
scale
causal mask
softmax
P @ V
```

中间注意力矩阵不需要作为完整 `[T,T]` 张量写回 HBM。

### 用 Stride 表示打包 QKV

输入物理布局为：

```text
[B, T, 3, H, HS]
```

但可以通过 Stride 让 cuDNN 直接看到三个逻辑张量：

```text
Q/K/V logical shape : [B, H, T, HS]
Q/K/V stride        : (3*H*HS*T, HS, 3*H*HS, 1)
```

这意味着 Q、K、V 不需要物理 permute，也不需要额外复制成三份连续数组。

输出直接按 Token 优先布局写出：

```text
O logical shape : [B, H, T, HS]
O physical shape: [B, T, H, HS]
O stride        : (H*T*HS, HS, H*HS, 1)
```

### 性能数据

| Kernel | Duration [ms] | Compute [%] | Memory [%] | Registers |
|---|---:|---:|---:|---:|
| FP32 → FP16/BF16 | 0.57 | 17.14 | 51.20 | 16 |
| cuDNN SDPA | 0.97 | 83.05 | 13.36 | 255 |
| FP16/BF16 → FP32 | 0.19 | 26.16 | 37.66 | 16 |
| **合计** | **1.73** | — | — | — |

v10 相对 v1 加速约 **243.44×**。

核心 cuDNN SDPA 的 Memory 指标只有 **13.36%**，Compute 指标为 **83.05%**，说明融合后大量中间访存被消除，Kernel 不再像朴素版本那样严重依赖 HBM 搬运。

### Graph 缓存

cuDNN Graph 的构建、校验和 Kernel 编译成本较高，但相同形状和参数可以复用。代码中根据 graph key 缓存已编译图：

- 首次运行：构建并编译；
- 后续运行：直接绑定显存地址并执行；
- Profile 时先进行 2 轮预热，避免把图编译成本计入正式 Kernel 数据。

---

## v11 — 全低精度 cuDNN SDPA：本轮最快版本

v11 在 v10 的基础上跳过 FP32↔低精度转换，直接假设输入输出已经是 FP16/BF16：

```text
FP16/BF16 QKV → cuDNN fused SDPA → FP16/BF16 output
```

### 性能数据

| Kernel | Duration [ms] | Compute [%] | Memory [%] | Registers |
|---|---:|---:|---:|---:|
| cuDNN SDPA | 0.97 | 83.02 | 13.36 | 255 |
| **合计** | **0.97** | — | — | — |

v11 相对 v1 加速约 **434.18×**，相对 v10 快约 **1.78×**。

v10 中精度转换合计：

```text
0.57 + 0.19 = 0.76 ms
```

占 v10 总耗时约 **43.9%**。v11 去掉这部分后，性能进一步提高。

### 适用场景

v11 对应真实的全低精度推理网络：

- 上游输入已经是 FP16/BF16；
- 后续层也能接受低精度输出；
- 不需要在 Attention 算子边界转回 FP32；
- 模型精度已经通过校准或混合精度策略验证。

如果模型入口仍是 FP32，或者系统要求输出 FP32，那么 v10 更接近完整链路。

---

## 关键机制拆解

## 1. 在线 Softmax：Flash Attention 的数学前提

标准 Softmax 需要先看到完整行，才能求最大值和分母。Flash Attention 将 K/V 分块后，不可能一次看到完整行，因此需要在线递推。

### 标准 Softmax

```text
m = max(x)
l = Σ exp(x - m)
y = exp(x - m) / l
```

### 分块递推

设已经处理过的状态为 `(m_old, l_old, O_old)`，当前 Tile 的局部状态为 `(m_block, l_block, P_block @ V_block)`：

```text
m_new = max(m_old, m_block)

l_new = l_old * exp(m_old - m_new)
      + l_block * exp(m_block - m_new)

O_new = O_old * exp(m_old - m_new)
      + (P_block @ V_block) * exp(m_block - m_new)
```

所有 Tile 处理完后：

```text
O = O_new / l_new
```

为什么这样是对的？

Softmax 的分子和分母同时依赖同一个最大值基准。当最大值变化时，旧分子和旧分母都乘以同一个修正系数：

```text
exp(m_old - m_new)
```

因此二者的比值不变，可以在不读取旧 Tile 原始 Score 的情况下持续合并。

训练或反向传播中通常还会保存：

```text
L = m + log(l)
```

---

## 2. 为什么 v3 列优先慢，v4 行优先快

| 对比项 | v3：列优先 | v4：行优先 |
|---|---|---|
| 外层循环 | KV Tile | Q Tile |
| 内层循环 | Q Tile | KV Tile |
| KV 复用方式 | 一个 KV Tile 被多个 Q Tile 共享 | 一个 Q Tile 流式遍历 KV |
| Q 行状态 | 每个 KV Tile 后写回 HBM | 驻留片上直到结束 |
| 输出 O | 多次 HBM 往返 | 最后统一写回 |
| Grid 并行 | `(B,H)` | `(Tr,H,B)` |
| L2 命中 | 调度不友好 | 连续 Q Block 更容易共享 KV |

Flash Attention 的关键不只是“用了 Shared Memory”，还包括：

- 谁在循环外层；
- 哪些状态可以常驻；
- Block 是否在序列维度并行；
- 是否避免重复读写 O、m、l；
- 是否有足够的 Grid Size 和 Occupancy。

---

## 3. 为什么手写 GEMM 远不如 cuBLAS

v5 是手写 Flash Attention 的优化版本，但 v6 一引入 cuBLAS，总耗时直接从：

```text
198.80 ms → 9.14 ms
```

加速约 **21.75×**。

原因不是 cuBLAS“编译器更神奇”，而是高性能 GEMM 本身需要大量工程：

- 多级 Tile 分块；
- Shared Memory 排布和 Bank Conflict 处理；
- 寄存器分块；
- 软件流水线；
- 异步拷贝；
- Tensor Core MMA 指令；
- 针对不同架构的 Instruction Scheduling；
- 尾块处理和多种 Shape 特化。

手写教学 Kernel 通常每个线程执行简单 FMA 循环，无法自动等价为上述 Tensor Core Kernel。

工程上应遵循：

- **学习阶段**：手写 v1–v5，理解访存、规约、分块和在线 Softmax；
- **生产阶段**：GEMM 优先使用 cuBLAS、CUTLASS、cuDNN、FlashAttention 等经过验证的实现；
- 除非有明确的定制融合需求，否则不要在工业项目中手写通用 GEMM。

---

## 4. 混合精度：低精度 IO，高精度累加

v8 和 v10 的共同原则是：

```text
张量存储 / HBM 传输：FP16 或 BF16
Softmax / GEMM 累加：FP32
```

这样可以同时获得：

- 更小的显存占用；
- 更低的 HBM 带宽压力；
- 更高的 Tensor Core 吞吐；
- 相对稳定的数值计算。

Softmax 中最敏感的是：

- 最大值；
- 指数差；
- 指数和；
- 归一化除法。

因此 `softmax_forward_kernel6_lowp` 会把输入元素先转成 FP32，再完成递推。

---

## 5. cuDNN 为什么连 permute 都能省掉

v6–v8 中，先物理重排：

```text
[B,T,3,H,HS] → Q,K,V [B,H,T,HS]
```

这会产生额外 Kernel 和显存读写。

cuDNN Frontend 允许直接描述逻辑张量的 Stride。对于打包 QKV，Q、K、V 只是同一物理缓冲区中的三个视图，不需要真的复制：

```text
Q pointer = inp
K pointer = inp + C
V pointer = inp + 2*C
```

再通过 Stride 告诉 cuDNN 如何解释这些地址。

这体现了推理优化中的另一个原则：

> 能用 Layout / Stride 表达的逻辑视图，不要轻易做物理重排。

---

## 路径总结与规律

```text
v1 (421.15 ms)
  朴素三 Kernel，QK / Softmax / PV 分离，完整注意力矩阵反复经过 HBM
  ↓ Warp 级融合
v2 (37.61 ms)
  一个 Warp 处理一行，点积和规约并行化，加速 11.20×
  ↓ 引入在线 Softmax 与 Tile，但采用列优先教学实现
v3 (644.68 ms)
  m/l/O 随 KV Tile 反复写回 HBM，手写矩阵乘效率低，反而最慢
  ↓ 改为行优先 FlashAttention-2
v4 (413.17 ms)
  Q Tile 在外，KV Tile 在内，状态常驻片上，序列维度并行
  ↓ Warp 级 split-Q
v5 (198.80 ms)
  每个 Warp 负责 8 行 Q，减少 Warp 间通信，但仍未使用 Tensor Core
  ↓ GEMM 交给 cuBLAS
v6 (9.14 ms)
  cuBLAS TF32 两次批量 GEMM，性能跳变 21.75×
  ↓ 融合 Scale/Mask/Online Softmax
v7 (5.06 ms)
  消掉独立 Scale Kernel，float4 + 流式访存，减少带宽流量
  ↓ FP16/BF16 IO + FP32 累加
v8 (2.89 ms)
  低精度带宽和 Tensor Core 收益明显
  ↓ 去除布局转换，得到核心下界
v9 (2.27 ms)
  仅代表已经完成布局和精度准备后的核心算子
  ↓ cuDNN 官方融合 SDPA
v10 (1.73 ms)
  Stride 零拷贝 + 片上融合，包含 FP32 与低精度转换
  ↓ 全低精度链路
v11 (0.97 ms)
  cuDNN 纯低精度 SDPA，本轮最快，相对 v1 加速 434.18×
```

### 核心规律

1. **先减少 HBM 往返，再谈指令级优化**  
   v1 的主要问题是中间矩阵写了又读；Flash Attention 的核心价值是减少片外访存。

2. **融合不是目的，正确的并行映射才是前提**  
   错误的线程划分、过低的 Occupancy 或不匹配的循环顺序，会让“理论上更优”的算法更慢。v3 就是典型例子。

3. **在线 Softmax 是 Flash Attention 的数学基础**  
   没有最大值和指数和的在线递推，就无法在不保存完整 `[T,T]` 矩阵的情况下逐 Tile 合并结果。

4. **行优先调度比列优先更适合前向推理**  
   固定 Q Tile、流式遍历 KV Tile，可以让 `m/l/O` 常驻片上，并天然支持序列维度并行。

5. **不要手写生产级 GEMM**  
   v5 到 v6 的 21.75× 跳变说明，规则矩阵乘应交给 cuBLAS、CUTLASS、cuDNN 或成熟 FlashAttention 实现。

6. **库 Kernel 优化完后，瓶颈会转移到周边算子**  
   v6 中 GEMM 已经很快，Scale、Softmax、permute、精度转换反而成为主要耗时。

7. **低精度和零拷贝布局是推理优化的重要来源**  
   v8 到 v11 的收益主要来自低精度 IO、Tensor Core、消除物理重排和消除格式转换。

8. **性能比较必须看完整链路**  
   v9 跳过布局转换，v11 假设输入输出已是低精度，它们是有价值的基准，但不能不加说明地等同于完整业务路径。

---

## 已知注意事项

本项目主要用于学习和性能分析，部分 Kernel 保留了教学化写法。若要改造成生产级实现，建议注意以下问题。

### 1. v2 建议补充 Warp 同步

v2 中 Lane 0 会写入 `preatt`，随后其他 Lane 会读取；归一化阶段各 Lane 写入 `att` 后，PV 阶段也会跨 Lane 读取。

从严格的 CUDA 内存模型看，建议在以下位置补充 Warp 级同步：

- QK 循环结束、读取 `preatt` 之前；
- 写入 `att` 完成、PV 读取之前。

可使用 Cooperative Groups 的 `warp.sync()` 或等价的 `__syncwarp()`。

### 2. v7/v8 需要保证未来位置为 0

融合 Softmax Kernel 只写出 `t2 <= t` 的下三角区域，而上三角未来位置没有显式写 0。但后续 cuBLAS `P @ V` 是完整矩阵乘，会读取整个注意力矩阵。

如果 `att` 缓冲区没有被预先清零，上三角可能保留脏数据。教学 Profile 中 v6 先执行，可能恰好把上三角写成 0，从而掩盖该问题。

生产级写法有三种选择：

1. 显式写出上三角 0；
2. 每轮确保 Softmax 输出缓冲区清零；
3. 更优地，直接使用完全融合的 Flash Attention / cuDNN SDPA，不物化完整 `att`。

### 3. 不能用 `cudaMemset` 初始化任意 float

v3 中使用了类似如下写法：

```cpp
cudaMemset(m, -10000.0f, B * NH * T * sizeof(float));
```

`cudaMemset` 按字节填充，不能把一组 float 初始化为任意浮点值。它只能可靠完成类似全 0 字节填充。

更合适的方式包括：

- 写一个简单的 Fill Kernel；
- 使用 `thrust::fill`；
- 或在 Flash Kernel 内部直接初始化 `m = -INFINITY`、`l = 0`、`O = 0`。

### 4. v9 和 v11 是有前提的性能基准

- v9 假设输入已经完成低精度转换和头优先布局；
- v11 假设上下游已经全部使用 FP16/BF16。

它们适合作为核心算子下界或全低精度网络场景，但不能代表包含格式转换的 FP32 输入链路。

### 5. 教学 Kernel 不等于生产 Kernel

v3–v5 的价值是解释 Flash Attention 数学和硬件映射。生产环境中更推荐：

- cuDNN SDPA；
- 官方 Flash Attention；
- 基于 CUTLASS 的 Tensor Core 实现；
- 或厂商提供的融合 Attention 库。

---

## 后续优化方向

1. **基于 CUTLASS 实现 Tensor Core 版 Flash Attention**  
   将手写 FMA 矩阵乘替换为 CUTLASS MMA / Tensor Core 路径，对比 cuBLAS 分离式实现和 cuDNN 融合实现。

2. **实现反向传播 Kernel**  
   使用前向保存的 logsumexp `L = m + log(l)`，完成 Flash Attention Backward，并统计前向保存布局对反向 HBM 流量的影响。

3. **支持更多 Attention 形态**  
   包括 GQA、MQA、MHA KV Cache、Variable-Length Batching、PagedAttention 和 Sliding Window Attention。

4. **Autotune Tile 和 Block 参数**  
   根据 Head Dim、序列长度、Shared Memory 容量和寄存器限制，自动搜索 `Br`、`Bc`、Warps per Block、Rows per Warp。

5. **增加 Roofline 分析**  
   分别估算 GEMM、Softmax、permute、转换 Kernel 的理论字节数和 FLOPs，与 NCU 实测带宽、Tensor Core 利用率对比。

6. **扩展低精度路径**  
   在硬件支持且精度验证通过的前提下，继续测试 BF16、FP8 等低精度格式，并区分训练与推理的可接受误差。

7. **消除布局转换**  
   与上游 QKV Projection 和下游 Output Projection 联合设计 Layout，使 Attention 边界尽量不做物理 permute。

---

## 参考资料

- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness, Dao et al., 2022
- FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning, Tri Dao, 2023
- NVIDIA cuDNN Frontend Attention 文档：  
  <https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/operations/Attention.md>
- flash-attention-minimal：  
  <https://github.com/tspeterkim/flash-attention-minimal>
- CUDA C++ Programming Guide：Cooperative Groups、Shared Memory、Warp Shuffle、Tensor Core
- cuBLAS Documentation：`cublasSgemmStridedBatched`、`cublasGemmStridedBatchedEx`

---

## 总结

这个项目完整展示了 Attention 优化中的典型路径：

```text
朴素多 Kernel
  → Warp 融合
  → 在线 Softmax
  → Flash Attention Tile 化
  → 行优先调度与 Warp 并行
  → cuBLAS Tensor Core
  → 融合 Softmax
  → FP16/BF16 混合精度
  → Stride 零拷贝
  → cuDNN 官方融合 SDPA
```

最重要的结论不是“某一个 Kernel 写法最快”，而是：

- Attention 的本质是 **两次 GEMM + 行级 Softmax + 大量中间张量搬运**；
- 手写 Flash Attention 有助于理解算法，但生产级 GEMM 应依赖成熟库；
- 优化顺序应从 HBM 流量、并行映射、Tensor Core 利用率，逐步推进到布局和精度转换；
- 当融合足够彻底时，`421.15 ms` 的朴素实现可以压缩到 `0.97 ms`，相对加速约 **434.18×**。
