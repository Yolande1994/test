# Softmax Forward — CUDA 算子优化

从朴素的三趟遍历，到标准块级规约，再到在线 Softmax（Online Softmax），记录 Softmax 在 GPU 上的 9 个版本迭代过程。

---

## 算子简介

对输入矩阵的每一行独立做 Softmax 归一化，将一行任意实数转换为总和为 1、取值在 0~1 之间的概率分布。

输入 `inp` 维度为 `(N, C)`，其中 `N = B*T`（总 Token 数），`C` 为特征维度。

标准做法需要三趟遍历一行：

```
maxval = max(row)                     # 第一趟：求行内最大值（避免 exp 上溢）
sum    = Σ exp(row - maxval)          # 第二趟：计算指数并求和
out    = exp(row - maxval) / sum      # 第三趟：归一化写回
```

>**数值稳定性**：`exp(x)` 在 `x > 88` 时会上溢为 inf，因此先减去行内最大值再算指数，数学上等价但数值稳定。

**访存特征**：当 `C = 50257`（GPT-2 词表维度）时，单行数据量约 200 KB，远超 L2 缓存。整个算子需要把输入张量读一遍、输出张量写一遍，是典型的**访存受限型算子（Memory-Bound）**，优化核心是减少全局内存往返、提升访存合并率与带宽利用率。

---

## 性能总览
> 本仓库测试基准全环境统一，所有性能数据均采集自 Nsight Compute（`ncu --set full --cache-control all`）。NCU 单次采集含硬件计数器采样开销，绝对耗时略高于干净执行，跨版本结论基于同口径下的相对比较与硬件利用率指标。（硬件等指标详见根目录 README）

> 默认测试配置：FP32，`N=8192`（`B=8 × T=1024`），`C=50257`（模拟 GPT-2 输出层），主要分析配置 `block_size=512`。

> v3 固定 `block_size=32`（单 Warp），其余版本在 32~1024 间扫参。

![全 block size 耗时矩阵](images/softmax.png)

### Block Size = 512 下的硬件指标

![NCU 512 尺寸实测](images/softmax1.png)

| 版本 | 核心实现 | 耗时 (ms) | Compute (%) | Memory (%) | 寄存器 | Grid Size |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| v1 | 单线程一行，三趟朴素 | 105.91 | 9.58 | 45.84 | 40 | 16 |
| v2 | 1 Block 一行，共享内存二分规约 | 11.52 | 36.39 | 78.16 | 28 | 8192 |
| v3 | 1 Warp 一行，纯 Shuffle 规约 | 44.93 | 8.49 | 57.09 | 16 | 8192 |
| v4 | 两级规约（Warp Shuffle + 跨 Warp 共享内存） | 14.35 | 27.28 | 60.66 | 22 | 8192 |
| v5 | 两级规约 + 循环展开 + 计算合并 + 流式访存 | **10.56** | 36.26 | **85.06** | 40 | 8192 |
| v6 | 在线 Softmax，单线程一行 | 141.65 | 53.74 | 20.82 | 33 | 16 |
| v7 | 在线 Softmax + 协作组，1 Warp 一行 | 15.20 | 16.62 | 84.59 | 22 | 512 |
| v8 | 在线 Softmax + 无分支 Shuffle | 15.22 | 12.56 | 84.43 | 21 | 512 |
| v9 | 在线 Softmax + 块级两级规约 | 10.86 | 26.44 | 80.09 | 18 | 8192 |

> **从 v1 到 v5，耗时从 105.91ms 降至 10.56ms，提升约 10 倍。**
> 全局最优为 v5@512（10.56ms）；在线算法 v9@512（10.86ms）以更少访存量持平标准三趟法。

---

## v1 — 朴素移植：单线程处理一行

最直接的 CPU 移植：每个线程独立处理一行 `C` 个元素，串行完成三趟遍历。

```cuda
__global__ void softmax_forward_kernel1(float* out, const float* inp, int N, int C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const float* inp_row = inp + i * C;
        float* out_row = out + i * C;
        // 三趟串行循环：最大值 → 指数求和 → 归一化
        ...
    }
}
```

### 问题
1. **并行度严重不足**：只有 `N=8192` 个线程，block=512 时 Grid 仅 16，26 个 SM 一半以上空闲（NCU 实测 Memory Throughput 仅 45.84%）。
2. **访存不合并**：相邻线程处理的行地址间隔 `C=50257` 个元素，Warp 内 32 个线程访问完全不连续的地址，每个 32 字节 Cache Line 只用到 4 字节，带宽利用率极低。
3. **长串行依赖**：单线程串行跑 5 万次循环，延迟无法被其他 Warp 隐藏。

> v1@1024 时 Grid 进一步缩小到 8，耗时暴涨至 247.26ms——这不是算法退化，而是 SM 完全欠载。

<br>

---

## v2 — 1 Block 一行：共享内存二分规约

v1 的问题是并行度和访存合并。v2 改为**一个 Block 处理一行**，块内所有线程协作分摊 `C` 个元素：

- 相邻线程访问连续地址（thread 0 读 `x[0]`、thread 1 读 `x[1]`……），**恢复合并访存**。
- 块内规约用经典的共享内存二分规约（每轮 stride 折半），需要多次 `__syncthreads()`。

```cuda
__global__ void softmax_forward_kernel2(...) {
    extern __shared__ float shared[];
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    const float* x = inp + bid * C;
    // 线程粗化遍历求局部最大值
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x)
        maxval = fmaxf(maxval, x[i]);
    // 共享内存二分规约
    shared[tid] = maxval;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        if (tid < stride) shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        __syncthreads();
    }
    ...
}
```

### 效果
- 耗时从 v1 的 105.91ms 降到 **11.52ms**——9 倍提升，合并访存恢复后带宽利用率从 45.84% 升到 78.16%。
- Memory Throughput 78.16%——还有提升空间。

### 现存问题
1. **共享内存规约开销大**：每个线程都写共享内存（block_size 个 float），多次块级同步。
2. **第三趟重新读 exp 结果**：求 sum 时要把刚写回全局内存的 `out` 再读一遍，多一次全局往返。

<br>

---

## v3 — 单 Warp 一行：纯 Shuffle 规约

v2 的共享内存二分规约有同步开销。v3 把任务划分缩小到**一个 Warp（32 线程）处理一行**，全程用 `__shfl_down_sync` 完成 Warp 内规约，不碰共享内存。

```cuda
__global__ void softmax_forward_kernel3(float* out, const float* inp, int N, int C) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;       // 固定 blockDim.x = 32
    const float* x = inp + bid * C;
    // Warp 内跨步遍历，shuffle 规约求最大值
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += 32)
        maxval = fmaxf(maxval, x[i]);
    maxval = warpReduceMax(maxval);
    float offset = __shfl_sync(0xFFFFFFFF, maxval, 0);  // 广播给所有线程
    ...
}
```

### 为什么 v3 在大 C 下反而慢？
v3 固定 block=32，一个 Warp 处理一行 `C=50257`，每个线程要串行循环约 1570 次。虽然无共享内存、无块级同步，但：
- **并行度不足**：一个 Block 只有 32 线程，SM 上每个 Block 占用的 Warp 数太少，无法填满 SM 的 warp slot，延迟隐藏能力弱。
- NCU 实测 Memory Throughput 仅 57.09%，Compute 8.49%——大量时间花在长循环的串行依赖上。

> v3 适合小 C（如 C=768，32 个线程循环 24 次就结束），延迟优势明显；但在 C=50257 的词表维度下，并行度成为瓶颈。

<br>

---

## v4 — 两级规约：Warp Shuffle + 跨 Warp 共享内存

v4 结合 v2 和 v3 的优点，采用**两级规约架构**：

- **第一级（Warp 内）**：32 线程用 Shuffle 指令在寄存器内完成规约，延迟极低。
- **第二级（跨 Warp）**：每个 Warp 的 lane 0 把结果写入共享内存，由 Block 主线程串行合并。

```
Block 内 N 个 Warp
  ├── Warp 0：32 线程 shuffle 规约 → 结果写 shared[0]
  ├── Warp 1：32 线程 shuffle 规约 → 结果写 shared[1]
  ├── ...
  └── Warp k：32 线程 shuffle 规约 → 结果写 shared[k]
         ↓
  Block 主线程：串行合并 shared[0..k] → 广播全局结果
```

### 共享内存用量对比
| | v2（共享内存二分） | v4（两级规约） |
|---|---|---|
| 共享内存 | block_size 个 float | warpsPerBlock 个 float |
| block=1024 时 | 1024 × 4B = 4 KB | 32 × 4B = 128 B |

共享内存占用大幅降低，SM 可驻留更多 Block。

### 效果与问题
- 耗时 14.35ms，比 v2 的 11.52ms 略慢。原因是 v4 多了跨 Warp 串行合并的 `__syncthreads()` 开销，且仍保留"重新读 exp 结果求 sum"的额外全局内存往返。
- Memory Throughput 60.66%——v5 会针对这两点优化。

<br>

---

## v5 — 高性能版：循环展开 + 计算合并 + 流式访存

v5 在 v4 的两级规约基础上做了 6 项关键优化：

1. **8 倍循环展开**：`UNROLL_FACTOR=8`，减少循环控制指令，批量发射访存请求，用计算时间隐藏访存延迟。
2. **计算合并（省一趟全局读）**：计算 exp 写回的同时，在寄存器内直接累加局部 sum，不再需要把 exp 结果写回全局内存后再读回来求 sum。
3. **共享内存分区**：`maxvals` / `sumvals` 独立两段，逻辑清晰。
4. **流式访存**：`__ldcs` 加载输入，输入和输出只读一次，不污染 L2 缓存。
5. **边界处理**：读操作用 `min(C-1, idx)` 钳位（重复读不影响 max/sum 结果），写操作严格 `if (idx < C)` 判断（越界写会破坏其他行）。
6. **读写分离**：先批量加载到寄存器、再集中计算写回，让编译器有充足的指令调度空间。

```cuda
// 第二趟：计算 exp + 写回 + 寄存器内局部求和（三合一）
float sumval = 0.0f;
for (int i = tid; i < C; i += blockDim.x * 8) {
    float reg_array[8];
    // 阶段 1：批量加载到寄存器
    #pragma unroll
    for (int u = 0; u < 8; u++)
        reg_array[u] = __ldcs(&x[min(C-1, i + u * blockDim.x)]);
    // 阶段 2：批量计算、写回、寄存器内累加
    #pragma unroll
    for (int u = 0; u < 8; u++) {
        if (i + u * blockDim.x < C) {
            float output = expf(reg_array[u] - offset);
            y[i + u * blockDim.x] = output;
            sumval += output;   // 省掉一趟全局内存读
        }
    }
}
```

### 效果
- 耗时 **10.56ms**，全局最优。
- Memory Throughput **85.06%**——接近 DRAM 带宽上限，算子已被带宽吃满。
- Compute Throughput 36.26%——剩余算力空转是访存受限算子的正常表现。

<br>

---

## v6 — 在线 Softmax 朴素移植

前面 v1~v5 都是标准三趟法，需要把 exp 中间结果写回全局内存再读回来求 sum。v6 引入**在线 Softmax**（Online Softmax，基于论文《Online normalizer calculation for softmax》），将"求最大值"和"求指数和"合并到一次遍历中：

```
遍历每个元素 x：
  if x > 当前 maxval:
      maxval = x
      sum = sum * exp(旧max - 新max) + exp(x - 新max)   # 旧 sum 折算到新基准
  else:
      sum = sum + exp(x - maxval)
```

数学等价性：全程只有加法和乘法，没有大数相消误差。相比三趟法，访存从 3 次降为 2 次（1 读输入 + 1 写输出）。

### 为什么 v6 这么慢？
v6 仍是"单线程处理一行"的并行模式，问题和 v1 完全一样：
- block=512 时 Grid 仅 16，SM 严重欠载。
- Memory Throughput 仅 20.82%——不是在线算法慢，是并行度不足导致数据喂不进去。
- Compute Throughput 反而有 53.74%（单线程串行算 exp，计算单元在忙但带宽没吃满）。

> v6 的存在价值是作为在线算法的基线，证明"在线算法本身不是银弹，必须配合正确的并行策略"。

<br>

---

## v7 — 在线 Softmax + 协作组：1 Warp 一行

v6 的问题是单线程并行度不足。v7 改为**一个 Warp 处理一行**，用 `cooperative_groups` 做 Warp 级规约。

核心数据结构 `SumMax` 把"局部最大值"和"基于该最大值的指数和"打包成 8 字节对齐结构体，通过 shuffle 一次性交换：

```cuda
struct __align__(8) SumMax { float maxval; float sum; };

__device__ SumMax reduce_sum_max_op(SumMax a, SumMax b) {
    // 选更大的 max 作为基准，较小基准下的 sum 折算过来
    float bigger = fmaxf(a.maxval, b.maxval);
    float sum = (a.maxval == bigger ? a.sum : a.sum * expf(a.maxval - bigger))
              + (b.maxval == bigger ? b.sum : b.sum * expf(b.maxval - bigger));
    return {bigger, sum};
}
```

```cuda
__global__ void online_softmax_forward_kernel7(...) {
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int row = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    const float* x = inp + row * C;
    SumMax sm_partial = {-INFINITY, 0.0f};
    // Warp 内跨步遍历，逐元素合并到局部 SumMax
    for (int i = warp.thread_rank(); i < C; i += 32)
        sm_partial = reduce_sum_max_op(sm_partial, {x[i], 1.0f});
    // Warp 内规约
    SumMax sm_total = cg::reduce(warp, sm_partial, reduce_sum_max_op);
    // 归一化写回
    for (int i = warp.thread_rank(); i < C; i += 32)
        out[row * C + i] = expf(x[i] - sm_total.maxval) / sm_total.sum;
}
```

### 效果
- 耗时 15.20ms，Memory Throughput 84.59%——带宽利用率很高。
- **但比 v5（10.56ms）慢了 44%**，尽管 Memory Throughput 几乎相同。

### 为什么 Memory 高却更慢？
关键在 **Grid Size**：
- v5：Grid = 8192（一个 Block 处理一行，512 线程并行）
- v7：Grid = 512（一个 Block 内 16 个 Warp 各处理一行）

v7 的 Block 总数只有 v5 的 1/16，SM 上驻留的 Block 数不足，虽然"有数据传输时带宽利用率高"，但部分 SM 在空转，总数据搬运时间更长。**Memory Throughput 百分比不是唯一指标，Grid Size 是否填满 SM 同样关键。**

<br>

---

## v8 — 在线 Softmax + 无分支 Shuffle

v8 用 `fmaxf` + 统一公式抹掉 v6/v7 中的 if-else 分支：

```cuda
float maxval = -INFINITY, sumval = 0.0f, bigger;
for (int i = laneId; i < C; i += 32) {
    bigger = fmaxf(maxval, x[i]);
    sumval = sumval * expf(maxval - bigger) + expf(x[i] - bigger);
    maxval = bigger;
}
```

所有线程执行完全相同的指令，无分支发散。

### 实测：v8 没有比 v7 更快
v8 耗时 15.22ms，与 v7 的 15.20ms 差异在测量噪声内，**性能基本一致**。原因：
1. 本算子是访存受限，ALU 上多算一次 exp 被访存延迟完全隐藏。
2. 在线 Softmax 的 max 刷新天然稀疏（一行 5 万元素里真正刷新最大值的次数极少），if-else 分支高度可预测，发散开销本就很小。

> v8 的价值不是性能提升，而是验证了"无分支优化的适用边界"——它在计算密集 + 分支随机的场景才值钱，在访存密集 + 分支可预测的场景里就是被隐藏的多余运算。

<br>

---

## v9 — 在线 Softmax + 块级两级规约

v9 把在线合并算子 `reduce_sum_max_op` 嫁接到 v4 的块级两级规约架构上：一个 Block（512 线程）处理一行，Warp 内 shuffle 规约 SumMax，跨 Warp 通过共享内存合并。

```
每线程遍历自己负责的元素 → 累积局部 SumMax
  ↓ Warp 内 shuffle 规约（合并 32 个 SumMax）
各 Warp lane 0 写 shared_sm[warpId]
  ↓ Block 主线程串行合并所有 Warp 的 SumMax
广播全局 maxval / sum → 各线程归一化写回
```

共享内存仅需 `warpsPerBlock × sizeof(SumMax)` = 16 × 8B = 128 B，远小于 v2 的 2 KB。

### 效果
- 耗时 **10.86ms**，Memory Throughput 80.09%。
- **在线算法以 2 趟访存（1 读 1 写）追平了 v5 三趟法的 10.56ms**，差距仅 2.8%。

这验证了在线 Softmax 的工程价值：在大 C（词表维度）场景下，减少一次全局内存往返的收益，被块级并行充分发挥，最终性能逼近甚至超过标准三趟法。

<br>

---

## 优化路径总结

```
v1 (105.91ms)  单线程一行三趟，并行度不足 + 访存不合并
  ↓ 1 Block 一行，恢复合并访存
v2 (11.52ms)   共享内存二分规约 → 带宽利用率升至 78%
  ↓ 引入两级规约
v4 (14.35ms)   Warp Shuffle + 跨 Warp 共享内存，减少共享内存占用
  ↓ 循环展开 + 计算合并 + 流式访存
v5 (10.56ms)   Memory Throughput 85%，全局最优
  ↓ 在线 Softmax 减少访存趟数
v6 (141.65ms)  在线朴素版，单线程并行度不足 → 反而最慢
  ↓ 1 Warp 一行
v7 (15.20ms)   在线 + 协作组，带宽高但 Grid 不足
  ↓ 无分支优化
v8 (15.22ms)   无分支写法，实测无收益（访存受限 + 分支可预测）
  ↓ 块级两级规约
v9 (10.86ms)   在线 + 块级并行，以 2 趟访存追平三趟法
```

---

## 规律总结

1. **访存合并是第一优先级**：v1 到 v2 的 9 倍提升，本质是把"线程间地址隔 C 个元素"改成了"相邻线程访问连续地址"。地址连续性比任何算法优化都重要。
2. **Grid Size 决定 SM 是否被填满**：v1/v6 在 block=512 时 Grid 仅 16，一半 SM 空闲，Memory Throughput 不足 46%；v5/v9 Grid=8192，带宽吃满 80%+。
3. **Memory Throughput 高 ≠ 一定快**：v7/v8 的 Memory Throughput（84%+）和 v5（85%）几乎相同，但 Grid 只有 512，总耗时多了 44%。带宽利用率百分比要结合 Grid Size / Occupancy 一起看。
4. **在线 Softmax 的价值依赖并行架构**：v6（单线程）最慢，v9（块级并行）追平最优——减少访存趟数的收益，必须有足够的并行度才能兑现。
5. **无分支优化有适用边界**：v8 消除了 if-else 分支但没有收益，因为访存受限算子的 ALU 空闲时间足够掩盖多余运算。优化手段要匹配瓶颈类型。

---

## 后续优化方向

1. **Float4 向量化访存**
   当前 v5/v9 仍是标量加载（4 字节/次），改为 `float4`（16 字节/次）可将访存指令数减至 1/4，进一步压榨带宽利用率。需处理 C 非 4 对齐的边界。

2. **GEMM 尾融合：Linear + Softmax**
   在语言模型中，Softmax 紧跟在输出层 GEMM 之后。基于 CUTLASS / cuBLAS Lt 自定义 Epilogue，将 Softmax 挂载到 GEMM 输出端，省去 logits 张量的一次 DRAM 往返，是推理框架的标准优化。

3. **在线 + 块级两级规约 + 向量化**
   v9 已经追平 v5，若再叠加 float4 向量化和循环展开，有望在词表维度下成为真正的最优版本（访存趟数更少 + 带宽利用率更高）。
