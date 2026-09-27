# Fused Residual + LayerNorm Forward — CUDA 算子融合优化

从两个独立 Kernel 的朴素移植，到自适应融合算子，记录 Residual + LayerNorm 在 GPU 上的 6 个版本迭代过程。

>**关于 Residual 算子和 LayerNorm 算子各自的优化历程，可阅读其 README**

---

## 算子简介

Transformer 编码器中，残差连接和 LayerNorm 是两个连续操作：

```
residual = inp1 + inp2                           # 残差连接
mean     = mean(residual)                        # 均值
rstd     = 1 / sqrt(var(residual) + eps)         # 倒标准差
normed   = (residual - mean) * rstd * weight + bias  # 归一化 + 缩放平移
```

**融合的价值**：如果不融合，中间结果 `residual`（B×T×C 大小）需要写回显存再被 LayerNorm 读回来，额外增加 2 次全局内存往返。

融合后，`residual` 留在寄存器/共享内存中直接传递，大幅减少显存流量。

这是一个**访存受限型算子（Memory-Bound）**，优化核心是减少 DRAM 访问次数和提升访存合并率。

---

## 性能总览

![全尺寸性能对比](images/fused.png)

| 版本 | 核心实现 | 耗时 (ms) | Compute (%) | Memory (%) | 寄存器 |
|:---:|---|:---:|:---:|:---:|:---:|
| v1 | 两个独立 Kernel（非融合） | 1.44 | — | — | — |
| v2 | 朴素融合，1 线程处理 1 Token | 2.56 | 4.54 | 58.54 | 28 |
| v3 | Warp 级融合，恢复合并访存 | 0.17 | 50.22 | 64.62 | 35 |
| v4 | 128bit 向量化 + 单遍统计 + 流式访存 + 锯齿循环 | 0.10 | 58.23 | **85.95** | 38 |
| v5 | 全共享内存缓存（权重/偏置/残差） | 0.10 | 64.12 | 84.45 | 39 |
| v6 | 三维 Block + Grid-Stride 生产级自适应 | 0.12 | 64.92 | 85.98 | 40 |

> v1 耗时为两个 Kernel 之和：residual 0.14 + layernorm 1.30 = 1.44 ms。
> **从 v1 到 v5，性能提升约 14 倍。**

---

## v1 — 未融合朴素版：两个独立 Kernel

最直观的写法：residual 和 LayerNorm 各自作为独立 Kernel 顺序执行。

```cuda
// Kernel 1：残差相加，每个线程处理 1 个元素
__global__ void residual_forward_kernel1(floatX* out, const floatX* inp1, const floatX* inp2, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) out[idx] = inp1[idx] + inp2[idx];
}

// Kernel 2：LayerNorm，1 个线程处理 1 个 Token
__global__ void layernorm_forward_kernel1(...) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        const floatX* x = inp + idx * C;
        // 三趟串行循环：均值 → 方差 → 归一化
        ...
    }
}
```

### 问题

1. **中间结果 `residual` 写回 DRAM 再读回**：额外产生 2×B×T×C 个元素的全局内存往返。
2. **两个 Kernel 串行启动**，存在启动开销。
3. LayerNorm 部分 1 个线程串行跑 C=768 次循环，并行度太低，Compute Throughput 只有 4.49%。

<br>

---

## v2 — 朴素融合：反而更慢？

v1 的问题是 residual 中间张量的全局内存读写。

v2 把两个算子融合到一个 Kernel 里——**每个线程负责一个完整 Token**。

```cuda
__global__ void fused_forward_kernel2(...) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    // 每个线程独立处理自己 Token 的 C 个元素
    for (int c = 0; c < C; ++c) {
        float out = inp1[c] + inp2[c];
        m += out;
        residual[c] = out;
    }
    // ... 方差、归一化同样串行遍历
}
```

### 疑问：2.56ms 比 v1 的 1.44ms 还慢！

融合消除了中间写回，为什么反而变慢了？

**分析：融合将输入张量的读取从合并访存的 Kernel 搬进了不合并访存的循环。**

#### v1 为什么不慢

v1 拆成两个 Kernel，每个 Kernel 采用了与自身数据匹配的访问模式：

| Kernel | 线程划分 | 读取的数据 | 访存模式 | 耗时 |
|---|---|---|---|---|
| residual_kernel | N×C 线程，每线程 1 元素 | inp1、inp2（原始输入） | **合并访存**，带宽利用率高 | 0.14ms |
| layernorm_kernel | N 线程，每线程 1 Token | residual（刚写入，驻留 L2） | 不合并，但数据在 L2 中，开销可控 | 1.30ms |

v1 的 layernorm_kernel 同样是 1 线程处理 1 个 Token，访存不合并，

但它读取的 residual 是 residual_kernel 刚写入的结果，仍驻留在 L2 缓存中，未访问 DRAM。

不合并访存的开销发生在 L2 层面，对整体性能影响有限。
>数据驻留 L2 这一现象的 NCU 报告分析可见 LayerNorm 算子 README 末尾

#### v2 的问题

v2 融合后，每个线程串行遍历 C=768 个元素，同时读取 inp1 和 inp2 两个输入张量。

由于线程间地址间隔 C 个元素，这两个输入张量的读取也变为不合并访存。

GPU 内存事务的最小粒度为 32 字节（一个 Cache Line）：

- **合并访存**：Warp 内 32 个线程访问连续地址，4 个事务即可完成，带宽利用率 100%。
- **不合并访存**：Warp 内 32 个线程地址隔着 768 个元素，每个线程单独触发一次 32 字节事务，但仅使用其中 4 字节，**带宽利用率仅 12.5%**。

v1 通过 residual_kernel 以合并访存方式仅用 0.14ms 就读完了 inp1 和 inp2。

v2 将相同的读取放入不合并循环，带宽利用率降至原来的 1/8——多出 1.12ms 耗时。

> **教训**：融合须建立在合并访存的基础上，否则性能可能更差。
>- v1 的两个 Kernel 虽然串行，但各自的访问模式与数据特性匹配：合并访存处理输入张量，不合并访存读取 L2 缓存。盲目融合会破坏这种匹配关系。
>- v3 通过调整任务划分（1 Warp 处理 1 Token）恢复合并访存，耗时即从 2.56ms 降至 0.17ms。

<br>

---

## v3 — Warp 级融合：恢复合并访存

v2 的问题是"1 个线程处理 1 个 Token"。

v3 改为**1 个 Warp（32 线程）处理 1 个 Token**：

- Warp 内 32 个线程跨步循环 `for (c = threadIdx.x; c < C; c += 32)` 共同分摊一行。
- 线程 0 访问 c=0, 32, 64...；线程 1 访问 c=1, 33, 65...——**地址连续，完美合并**。
- Warp 内规约用 `warpReduceSum`（基于 `__shfl_down_sync`），纯寄存器通信，无需共享内存。

```cuda
__global__ void fused_forward_kernel3(...) {
    int idx = blockIdx.x * blockDim.y + threadIdx.y;  // 二维 Block: dim3(32, block_y)
    if (idx >= N) return;

    // 第一趟：残差相加 + 局部求和
    float m = 0.0f;
    for (int c = threadIdx.x; c < C; c += 32) {
        float out = inp1[c] + inp2[c];
        m += out;
        residual[c] = out;
    }
    m = warpReduceSum(m) / C;  // shuffle 规约

    // 第二趟：从 residual 读回算方差
    float v = 0.0f;
    for (int c = threadIdx.x; c < C; c += 32) {
        float d = residual[c] - m;
        v += d * d;
    }
    v = warpReduceSum(v) / C;
    float s = rsqrtf(v + eps);

    // 第三趟：归一化 + 缩放平移
    for (int c = threadIdx.x; c < C; c += 32) {
        normed[c] = s * (residual[c] - m) * weight[c] + bias[c];
    }
}
```

### 效果

- 耗时从 v2 的 2.56ms 降到 **0.17ms**——15 倍提升，恢复合并访存的同时增加了并行度的功劳。
- Compute Throughput 从 4.54% 升到 50.22%——终于在干活了。
- Memory Throughput 64.62%——还有提升空间。

### 现存问题

1. **标量访存**：单线程每次加载 1 个 float（4 字节），访存指令数多。
2. **residual 写了又读**：第一趟写 residual 到全局内存，第二趟又读回来。
3. **weight/bias 每个 Token 都重新读**，没有缓存复用。

<br>

---

## v4 — 向量化 + 单遍统计 + 流式访存 + 锯齿循环

v3 已经很快了，v4 在四个维度同时优化：

### 一. 128bit 向量化访存

用 `x128`（Packed128）一次加载 4 个 float，访存指令数减少到 1/4：

```cuda
const x128 in1 = load128cs(inp1 + c);  // 一次读 4 个 float
const x128 in2 = load128cs(inp2 + c);
```

### 二. 单遍统计：省掉第二趟读 residual

用方差公式 `Var(x) = E[x²] - E[x]²`，读一趟同时累加 `sum` 和 `sum_sq`：

```cuda
for (int c = threadIdx.x * 4; c < C; c += 32 * 4) {
    const x128 in1 = load128cs(inp1 + c);
    const x128 in2 = load128cs(inp2 + c);
    x128 out;
    for (int k = 0; k < 4; ++k) {
        out[k] = in1[k] + in2[k];
        sum += out[k];              // 元素总和 → 用于算均值
        sum_sq += out[k] * out[k];  // 平方和   → 用于算方差
    }
    store128(residual + c, out);
}
float m = sum / C;             // 均值
float v = sum_sq / C - m * m;  // 方差（单趟求出）
```

### 三. 流式访存（`__ldcs` / `__stcs`）

- `load128cs`：input 和 residual 只读一次，用流式（Streaming）不驻留缓存。
- `load128`（不带 cs）：weight/bias 是全局共享的，保留在缓存中供所有 Token 复用。
- `store128cs`：normed 写完就不需要了，流式存储不污染缓存。

### 四. 锯齿形（Zigzag）遍历

第一趟从前往后写 residual，第二趟**从后往前读**：

```cuda
c -= 32 * 4;  // 回退到最后一个有效块
for (; c >= 0; c -= 32 * 4) {
    const x128 r = load128cs(residual + c);  // 倒序读
    ...
}
```

第一趟最后写入的 residual 尾部数据还在 L2 缓存里，第二趟一上来就读它们，有助于提升缓存命中率。

### 效果

- 耗时从 0.17ms 降到 **0.10ms**。
- Memory Throughput 从 64.62% 升到 **85.95%**。

> **数值稳定性提醒**：单遍公式 `var(x)=E[x²]-E[x]²` 在输入数值大但方差小时会丢精度（大数相减）。推理场景可接受，训练场景还是用两趟法。

<br>

---

## v5 — 全共享内存缓存

v4 还有几个待提升问题：
1. weight/bias 每个 Token 都要从全局内存读一次，重复读取。
2. residual 虽然用了锯齿循环，但本质上还是一次全局内存读取。
3. 单遍方差公式 var(x)=E[x²]-E[x]² 存在大数相减时的精度损失。

v5 **把 weight、bias、residual 全部缓存到共享内存**。

### 共享内存布局

| 区域 | 大小（C=768, block_y=4） |
|---|---|
| s_weight | 768 × 4B = 3 KB |
| s_bias | 768 × 4B = 3 KB |
| s_res（4 个 Warp 各一份） | 4 × 768 × 4B = 12 KB |
| **合计** | **18 KB** |

```cuda
extern __shared__ char params[];
x128* s_weight = ...;   // 全 Block 共享 weight
x128* s_bias   = ...;   // 全 Block 共享 bias
x128* s_res    = ...;   // 每个 Warp 一份私有的残差缓存

// 全 Block 合力加载 weight/bias（只读一次 DRAM）
for (int i = sidx; i < C; i += blockDim.y * 32 * 4) {
    s_weight[i/4] = load128(weight + i);
    s_bias[i/4]   = load128(bias + i);
}
__syncthreads();

// 第一趟：残差相加 + 写全局内存（供反向传播）+ 写共享内存（供后续复用）
for (int c = threadIdx.x * 4; c < C; c += 32 * 4) {
    const x128 in1 = load128cs(inp1 + c);
    const x128 in2 = load128cs(inp2 + c);
    x128 out;
    for (int k = 0; k < 4; ++k) { out[k] = in1[k] + in2[k]; sum += out[k]; }
    store128cs(residual + c, out);  // 写 DRAM（反向传播需要）
    s_res[c/4] = out;               // 写共享内存（后续三趟都从这里读）
}

// 第二趟、第三趟：全部从共享内存读数据，零全局访存
```

### 效果

- 耗时 **0.10ms**，和 v4 持平。

### 为什么实测 v5 没有比 v4 更快？

**首先**，v4 的锯齿循环已经把 residual 的 L2 命中率做得很好了，共享内存缓存的收益被 L2 吃掉了

**其次**，v5 的设计目标不是性能最优，而是**数值稳定前提下的性能最优**。

它通过共享内存缓存 weight/bias 和 residual，将标准两趟法的性能拉到了接近 v4 单趟法的水平，但受限于三方面原因，最终未能超越：

1. **优化对象占比极低**：weight/bias 数据量仅占总访存量的千分之一，共享带来的 DRAM 节省微乎其微；
2. **引入轻微额外开销**：共享内存写入、__syncthreads()同步、两趟循环带来轻微指令开销；
3. **Occupancy 下降**：共享内存占用减少了 SM 可驻留的 Block 数，削弱了延迟隐藏能力，这是最主要的原因。

**尺寸 block=256 时，各版本在 C=768/2048/4096 时的性能对比**

![256尺寸性能对比](images/fused1.png)

**尺寸 block=256 时，v5 在 C=768/2048/4096 时的 Occupancy**

![v5占用率](images/fused2.png)

>受限于共享内存限制，v5 的 Occupancy 随着 C 的变大而下降；此时 v4 的 Occupancy 一致保持着 100%（未截图）

**无论如何，v5 用 和 v4 相当的时间换来了比 v4 更好的精度，这是其重要的价值**

<br>

---

## v6 — 三维 Block + 网格步长循环：生产级自适应设计

v3~v5 都是"1 个 Warp 处理 1 个 Token"。v6 引入了两个生产级设计：

### ① 三维 Block：多 Warp 协作处理一个 Token

```
dim3(32, block_y, block_z)
  blockDim.x = 32        → 一个 Warp
  blockDim.y = warps_per_token  → 一个 Token 用几个 Warp
  blockDim.z = tokens_per_block  → 一个 Block 同时处理几个 Token
```

- C=768 时：`warps_per_token=1`，`tokens_per_block=4`（和 v5 一样）。
- C=4096 时：`warps_per_token=4`，`tokens_per_block=1`——更多 Warp 协作分摊长行。

这就是之前 LayerNorm 中讨论的"没有万能最优版本"的工程答案：**运行时根据 C 自动调整并行度**。

### ② Grid-Stride Loop：固定块数 + 循环消化

传统做法按数据量开块：`grid_size = N / tokens_per_block`。N 小时 SM 闲置，N 大时调度开销暴涨。

v6 按硬件算力开固定块数：

```cuda
const int num_blocks = cuda_num_SMs * cuda_threads_per_SM / block_size;
```

不管 N 是 1000 还是 10000，都只开刚好让 GPU 满载的块数，剩余 Token 靠 `for` 循环分批处理。

```cuda
for (int tidx = blockIdx.x * blockDim.z + threadIdx.z; tidx < N;
     tidx += gridDim.x * blockDim.z) {
    // 处理 Token tidx
}
```

### 效果

- 耗时 0.12ms，比 v4/v5 略慢 0.02ms——这是自适应设计的代价。
- 但它的优势不在单次性能，而在**生产环境的稳定性**：
  - 块数固定，SM 占用率恒定，性能波动小。
  - 自动适配任意 C 和任意 N。
  - 共享内存申请失败时自动回退到 v4。

---

## 优化路径总结

```
v1 (1.44ms)  两个独立 Kernel，residual 写回 DRAM 再读回
  ↓ 融合消除中间写回
v2 (2.56ms)  朴素融合，1 线程 1 Token —— 访存不合并，反而更慢！
  ↓ 改为 1 Warp 1 Token，恢复合并访存
v3 (0.17ms)  Warp 级 shuffle 规约，合并访存恢复
  ↓ 向量化 + 单遍统计 + 流式访存 + 锯齿循环
v4 (0.10ms)  128bit 访存，打满 85.95% 显存带宽
  ↓ 全共享内存缓存，weight/bias/residual 零全局读
v5 (0.10ms)  共享内存复用，两趟法数值稳定
  ↓ 三维 Block + Grid-Stride 自适应
v6 (0.12ms)  生产级设计，自动适配 C 和 N，性能略降但稳定性最优
```

### 核心规律

1. **访存合并是第一优先级**：v2 融合了却变慢，v3 只改了任务划分就快了 15 倍——Warp 内地址连续比任何算法优化都重要。
2. **融合的收益取决于中间张量大小**：residual 是 B×T×C 的大张量，融合省掉的 DRAM 往返非常可观。
3. **向量化是带宽受限算子的必经之路**：128bit 访存把指令数降到 1/4，直接把 Memory Throughput 从 64% 推到 86%。
4. **共享内存缓存的收益取决于 L2 命中率**：v5 和 v4 打平，说明 L2 已经接住了大部分访问；共享内存在更大 C 或更多 Token 时优势才会显现。
5. **生产代码和极致性能是两个目标**：v6 牺牲了 20% 的峰值性能，换来自适应能力和部署稳定性——这是工程上的正确取舍。
