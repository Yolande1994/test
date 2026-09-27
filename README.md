# Fused Residual + LayerNorm Forward — CUDA 算子融合优化实战

从两个独立 Kernel 到生产级自适应融合算子，完整记录 Residual Add + LayerNorm 在 GPU 上的 6 个版本迭代过程。

---

## 算子简介

Transformer 编码器中，残差连接和 LayerNorm 是两个连续操作：

```
residual = inp1 + inp2                          # 残差连接
mean     = mean(residual)                        # 均值
rstd     = 1 / sqrt(var(residual) + eps)         # 倒标准差
normed   = (residual - mean) * rstd * weight + bias  # 归一化 + 缩放平移
```

**融合的价值**：如果不融合，中间结果 `residual`（B×T×C 大小）需要写回显存再被 LayerNorm 读回来，额外增加 2 次全局内存往返。融合后，`residual` 留在寄存器/共享内存中直接传递，大幅减少显存流量。

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
> **从 v1 到 v4/v5，性能提升约 14 倍。**

---

## v1 — 非融合基线：两个独立 Kernel

最直观的写法：残差连接和 LayerNorm 各自作为独立 Kernel 顺序执行。

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
2. **两个 Kernel 串行启动**，有 launch overhead。
3. LayerNorm 部分和之前 layernorm_forward_kernel1 一样：1 个线程串行跑 C=768 次循环，Compute Throughput 只有 4.49%。

---

## v2 — 朴素融合：融合了反而更慢？

v1 的问题是 residual 中间张量的全局内存读写。v2 把两个算子融合到一个 Kernel 里——**每个线程负责一个完整 Token**。

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

### 反直觉：2.56ms 比 v1 的 1.44ms 还慢！

融合消除了中间写回，为什么反而变慢了？

**根因：访存不合并（Memory Coalescing 失效）。**

GPU 访存的基本单位是 32 个连续线程组成的 Warp。当 Warp 内 32 个线程访问**连续地址**时，硬件只需 1~2 次事务就能完成。但在 v2 中：

- 线程 0 处理 Token 0，访问地址 `inp1[0..767]`
- 线程 1 处理 Token 1，访问地址 `inp1[768..1535]`
- 线程 2 处理 Token 2，访问地址 `inp1[1536..2303]`

当 32 个线程同时执行 `inp1[c]`（比如 c=0）时，它们访问的地址间隔了 768 个元素——完全不连续。GPU 不得不为每个线程单独发起一次内存事务，访存效率暴跌 32 倍。

> **教训**：融合不等于快。如果融合破坏了访存合并，收益会被访存效率的下降完全吃掉。这也是为什么 v2 的 Memory Throughput 只有 58.54%——大量内存事务浪费在了地址间隙上。

---

## v3 — Warp 级融合：恢复合并访存

v2 的问题是"1 个线程处理 1 个 Token"。v3 改为**1 个 Warp（32 线程）处理 1 个 Token**：

- Warp 内 32 个线程用跨步循环 `for (c = threadIdx.x; c < C; c += 32)` 分摊同一行。
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

- 耗时从 v2 的 2.56ms 降到 **0.17ms**——15 倍提升，全靠恢复合并访存。
- Compute Throughput 从 4.54% 升到 50.22%——GPU 终于在干活了。
- Memory Throughput 64.62%——还有提升空间。

### 现存问题

1. **标量访存**：每次加载 4 字节（1 个 float），访存指令数多。
2. **residual 写了又读**：第一趟写 residual 到全局内存，第二趟又读回来。
3. **weight/bias 每个 Token 都重新读**，没有缓存复用。

---

## v4 — 向量化 + 单遍统计 + 流式访存 + 锯齿循环

v3 已经很快了，v4 在四个维度同时优化：

### ① 128bit 向量化访存

用 `x128`（Packed128）一次加载 4 个 float，访存指令数减少到 1/4：

```cuda
const x128 in1 = load128cs(inp1 + c);  // 一次读 4 个 float
const x128 in2 = load128cs(inp2 + c);
```

### ② 单遍统计：省掉第二趟读 residual

用方差公式 `Var(x) = E[x²] - E[x]²`，第一趟同时累加 `sum` 和 `sum_sq`：

```cuda
for (int c = threadIdx.x * 4; c < C; c += 32 * 4) {
    const x128 in1 = load128cs(inp1 + c);
    const x128 in2 = load128cs(inp2 + c);
    x128 out;
    for (int k = 0; k < 4; ++k) {
        out[k] = in1[k] + in2[k];
        sum += out[k];
        sum_sq += out[k] * out[k];  // 平方和
    }
    store128(residual + c, out);
}
float m = sum / C;
float v = sum_sq / C - m * m;  // 单遍方差
```

### ③ 流式访存（`__ldcs` / `__stcs`）

- `load128cs`：input 和 residual 只读一次，不驻留缓存（Streaming）。
- `load128`（不带 cs）：weight/bias 是全局共享的，保留在缓存中供所有 Token 复用。
- `store128cs`：normed 写完就不需要了，流式存储不污染缓存。

### ④ 锯齿形（Zigzag）遍历

第一趟从前往后写 residual，第二趟**从后往前读**：

```cuda
c -= 32 * 4;  // 回退到最后一个有效块
for (; c >= 0; c -= 32 * 4) {
    const x128 r = load128cs(residual + c);  // 倒序读
    ...
}
```

第一趟最后写入的 residual 尾部数据还在 L2 缓存里，第二趟一上来就读到它们——利用 LRU 缓存特性提升命中率。

### 效果

- 耗时从 0.17ms 降到 **0.10ms**。
- Memory Throughput 从 64.62% 升到 **85.95%**——接近显存带宽物理上限。
- Compute Throughput 58.23%——已经被带宽卡住。

> **数值稳定性提醒**：单遍公式 `E[x²]-E[x]²` 在输入数值大但方差小时会丢精度（大数相减）。推理场景可接受，训练场景建议用两趟法。

---

## v5 — 全共享内存缓存

v4 还有两个浪费：
1. weight/bias 每个 Token 都从全局内存读一次。
2. residual 虽然用了锯齿循环，但本质上还是走了一次全局内存。

v5 的解法：**把 weight、bias、residual 全部缓存到共享内存**。

```cuda
extern __shared__ char params[];
x128* s_weight = ...;   // 全 Block 共享
x128* s_bias   = ...;   // 全 Block 共享
x128* s_res    = ...;   // 每个 Warp 私有

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
    s_res[c/4] = out;              // 写共享内存（后续三趟都从这里读）
}

// 第二趟、第三趟：全部从共享内存读，零全局访存
```

### 共享内存布局

| 区域 | 大小（C=768, block_y=4） |
|---|---|
| s_weight | 768 × 4B = 3 KB |
| s_bias | 768 × 4B = 3 KB |
| s_res（4 个 Warp 各一份） | 4 × 768 × 4B = 12 KB |
| **合计** | **18 KB** |

### 效果

- 耗时 **0.10ms**，和 v4 持平。
- 为什么没有更快？因为 v4 的锯齿循环已经把 residual 的 L2 命中率做得很好了，共享内存缓存的收益被 L2 吃掉了。
- v5 的真正价值：**用回了数值稳定的两趟法**（从共享内存读 residual 算方差，不依赖 `E[x²]-E[x]²`），同时不需要锯齿循环。

---

## v6 — 三维 Block + Grid-Stride：生产级自适应设计

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
