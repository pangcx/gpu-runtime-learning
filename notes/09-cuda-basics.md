# CUDA 并行编程基础

## 核心结论

1. **GPU 并行思维的本质**：写一份代码，让成千上万个线程同时执行，每个线程通过 `threadIdx` / `blockIdx` 区分"我负责哪段数据"
2. **Warp 是硬件执行的最小单位**：32 个线程永远同步执行同一条指令，是所有性能优化的基础
3. **Block 是编程模型的协作单位**：Block 内线程可共享 Shared Memory、可同步（`__syncthreads`），Block 间完全独立
4. **Shared Memory 是性能优化的关键**：速度比 Global Memory 快约 100 倍，矩阵乘法等算法的核心优化手段
5. **内存合并访问**：相邻线程访问相邻地址，硬件才能合并成一次事务，否则性能暴跌

---

## 一、GPU 线程层次

```
一次 kernel 调用
└── Grid（所有线程的集合）
      └── Block（线程的逻辑分组，可协作）
            └── Warp（硬件执行单位，32个线程）
                  └── Thread（最小执行单元）
```

### 关键变量

| 变量 | 含义 |
|---|---|
| `threadIdx.x` | 当前线程在 Block 内的编号 |
| `blockIdx.x` | 当前 Block 在 Grid 内的编号 |
| `blockDim.x` | 每个 Block 的线程数 |
| `gridDim.x` | Grid 内的 Block 数 |

**全局线程编号**（最常用的公式）：
```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

---

## 二、Kernel 调用

**Kernel** 是用 `__global__` 修饰的 GPU 函数，用 `<<<grid, block>>>` 语法在 CPU 端启动。

```cuda
// 定义：__global__ 表示在 GPU 上执行，由 CPU 调用
__global__ void vector_add(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

// 调用：启动 4 个 Block，每个 Block 256 个线程 = 1024 个线程并行执行
vector_add<<<4, 256>>>(d_A, d_B, d_C, N);

// 等待所有 GPU 线程完成
cudaDeviceSynchronize();
```

与 CPU 函数调用的区别：

| | CPU 函数调用 | Kernel 调用 |
|---|---|---|
| 执行次数 | 1次，单线程 | 启动 N 个线程并行执行 |
| 调用语法 | `func(args)` | `func<<<grid, block>>>(args)` |
| 返回时机 | 执行完才返回 | 立即返回（异步），需要 `cudaDeviceSynchronize` 等待 |

---

## 三、Block vs Warp 的区别

| | Block | Warp |
|---|---|---|
| **是什么** | 编程模型中的逻辑分组 | 硬件调度的物理单位 |
| **大小** | 你决定（1~1024 线程）| 固定 **32 个线程**，不可改变 |
| **在哪定义** | 你写 `<<<grid, block>>>` 时 | 硬件自动从 Block 切出来 |
| **同步** | 需要显式 `__syncthreads()` | 天然同步（同一时刻执行同一指令）|
| **共享内存** | Block 内线程共享 Shared Memory | Warp 内无需额外共享机制 |

---

## 四、Warp 深入理解

### SIMT 执行模型

Warp 内 32 个线程永远在同一时刻执行同一条指令，只是操作不同的数据（Single Instruction, Multiple Threads）。

```cuda
// 看起来每个线程"各自"执行：
int i = threadIdx.x;
out[i] = in[i] * 2.0f;

// 硬件实际：Warp 内 32 个线程同时执行 "乘以 2" 这条指令
// 每个线程的 i 不同，但指令相同
```

### Warp Divergence（最重要的性能坑）

Warp 内线程走不同分支时，硬件必须串行执行每个分支：

```cuda
// ❌ 产生 Warp Divergence：偶数/奇数线程走不同路径
if (threadIdx.x % 2 == 0) {
    out[i] = in[i] * 2.0f;   // 第1步执行，奇数线程等待
} else {
    out[i] = in[i] + 1.0f;   // 第2步执行，偶数线程等待
}
// 原本 1 步完成，变成 2 步，性能减半
```

**避免方法**：让同一 Warp 内的线程尽量走相同路径；不同逻辑拆成不同 kernel。

### Warp 大小为什么是 32？

由 SM 的物理硬件结构决定，是在"延迟隐藏能力 / 硬件面积 / 分支代价 / 内存合并粒度"等多个维度上的权衡结果。

**所有 NVIDIA GPU 的 Warp 大小都是 32**，从 2006 年 Tesla 架构到最新 Blackwell，从未改变。

其他厂商对比：

| 厂商 | 名称 | 大小 |
|---|---|---|
| NVIDIA | Warp | 32 |
| AMD | Wavefront | 64（RDNA 降为 32）|
| Intel | SubGroup | 8/16/32（可变）|

### Block Size 为什么要选 32 的倍数？

硬件按 Warp（32 线程）调度。如果 block_size=48：
- Warp 0：线程 0~31（满，正常）
- Warp 1：线程 32~47（只有 16 个有效，另外 16 个空转浪费）

**实践**：block_size 通常选 128、256、512。

---

## 五、同一 Grid 内不同 Block 的关系

**代码完全相同，处理的数据不同。**

```cuda
// 所有 Block 执行同一份代码
// 靠 blockIdx 区分"我负责哪段数据"
__global__ void kernel(float* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    data[i] = data[i] * 2.0f;  // 所有 Block 执行同一行
}
// Block 0 处理 data[0~255]
// Block 1 处理 data[256~511]  ← 代码一样，数据不同
// Block 2 处理 data[512~767]
```

如果不同数据需要不同逻辑，正确做法是**拆成两个 kernel**：

```cuda
// ✅ 推荐：两个简单 kernel
process_type_a<<<grid1, block>>>(data_a, N);
process_type_b<<<grid2, block>>>(data_b, N);

// ❌ 不推荐：一个 kernel 内用 blockIdx 做条件分支
```

---

## 六、GPU 内存层次

从快到慢（RTX 4090 参考）：

| 内存类型 | 带宽 | 范围 | 生命周期 |
|---|---|---|---|
| Registers（寄存器）| ~19 TB/s | 每个线程私有 | 线程 |
| Shared Memory | ~19 TB/s | Block 内共享 | Block |
| L2 Cache | ~7 TB/s | 全 GPU | - |
| Global Memory（显存）| ~1 TB/s | 全 GPU | 应用程序 |
| PCIe（CPU↔GPU）| ~32 GB/s | - | - |

**核心优化原则**：尽量让数据待在 Shared Memory 里，减少 Global Memory 访问。

### Shared Memory 用法（以矩阵乘法为例）

```cuda
__global__ void matmul(const float* A, const float* B, float* C, int N) {
    __shared__ float tileA[16][16];  // 声明 Shared Memory
    __shared__ float tileB[16][16];

    // 协作加载数据到 Shared Memory
    tileA[threadIdx.y][threadIdx.x] = A[...];
    tileB[threadIdx.y][threadIdx.x] = B[...];

    __syncthreads();  // 等待所有线程加载完毕，再开始计算

    // 在 Shared Memory 里计算（快 100 倍）
    for (int k = 0; k < 16; ++k)
        sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

    __syncthreads();  // 计算完再进入下一轮
}
```

### 内存合并访问

```cuda
// ✅ 合并访问：线程 0 读 in[0]，线程 1 读 in[1]，...（连续地址）
// 硬件合并成一次大事务，效率高
out[i] = in[i] * 2.0f;

// ❌ 跨步访问：线程 0 读 in[0]，线程 1 读 in[32]，...（地址不连续）
// 每个线程独立触发内存事务，效率低
out[i] = in[i * 32] * 2.0f;
```

---

## 七、标准 CUDA 程序模板

所有 CUDA 程序都遵循这个流程：

```cuda
// 1. CPU 分配并初始化数据
float* h_data = (float*)malloc(bytes);

// 2. GPU 分配显存
float* d_data;
cudaMalloc(&d_data, bytes);

// 3. H2D：CPU 内存 → GPU 显存
cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

// 4. 启动 kernel
int block = 256, grid = (N + block - 1) / block;
my_kernel<<<grid, block>>>(d_data, N);

// 5. D2H：GPU 显存 → CPU 内存
cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);

// 6. 释放资源
cudaFree(d_data);
free(h_data);
```

---

## ❓ Q&A

**Q：Warp 和 Block 的区别是什么？**  
A：Block 是你写代码时定义的逻辑分组，Warp 是硬件实际执行的物理单位（固定 32 线程）。Block 按 Warp 大小自动切割后交给硬件调度。

**Q：所有 NVIDIA GPU 的 Warp 都是 32 吗？**  
A：是的，从 2006 年到现在所有 NVIDIA GPU 均为 32，由 SM 的物理硬件结构决定，不会改变（改了会破坏大量已有 CUDA 代码）。

**Q：一次 kernel 调用是什么意思？**  
A：用 `<<<>>>` 启动一个 `__global__` 函数一次。这次调用会启动一个 Grid，里面所有 Block 的所有线程并行执行同一份代码。多次调用同一个 kernel 是独立的多次启动。

**Q：同一个 Grid 下不同 Block，执行逻辑一样吗？**  
A：代码完全一样，但通过 `blockIdx` 计算出不同的 `i`，所以处理不同的数据。如果需要不同逻辑，应该拆成两个 kernel 而不是在一个 kernel 里用 if-else 区分，后者会导致性能差的 Warp Divergence。

**Q：为什么 block_size 要选 32 的倍数？**  
A：硬件按 Warp（32 线程）调度。不是 32 倍数时，最后一个 Warp 会有空转线程浪费计算资源。常用值：128、256、512。
