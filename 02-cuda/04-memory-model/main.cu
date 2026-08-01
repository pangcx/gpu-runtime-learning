/**
 * 项目04：GPU 内存模型
 *
 * 学习目标：
 *   - 理解 GPU 的内存层次：Registers → L1/Shared → L2 → Global
 *   - 掌握 Unified Memory（cudaMallocManaged）简化编程
 *   - 理解内存合并访问（Coalesced Access）vs 随机访问的性能差距
 *   - 理解 Pinned Memory 对 H2D 传输速度的提升
 *
 * GPU 内存速度对比（RTX 4090）：
 *   Registers    : ~19 TB/s  （每个 SM 私有，最快）
 *   Shared Memory: ~19 TB/s  （每个 Block 共享，极快）
 *   L2 Cache     : ~7 TB/s   （全 GPU 共享）
 *   Global Memory: ~1 TB/s   （GDDR6X，大但相对慢）
 *   PCIe（H2D）  : ~32 GB/s  （CPU↔GPU 传输，瓶颈所在）
 */

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { cudaError_t err = (call); \
         if (err != cudaSuccess) { \
             fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
             exit(EXIT_FAILURE); } } while(0)

// ─────────────────────────────────────────────
// Part 1: 合并访问 vs 随机访问
// ─────────────────────────────────────────────

// ✅ 合并访问（Coalesced）：相邻线程访问相邻内存地址
// 硬件可以把多个线程的访问合并成一次大的内存事务（高效）
__global__ void coalesced_read(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = in[i] * 2.0f;  // 线程 0 读 in[0]，线程 1 读 in[1]，... 连续
}

// ❌ 跨步访问（Strided）：相邻线程访问间隔很远的地址
// 硬件无法合并，每个线程各自触发独立内存事务（低效）
__global__ void strided_read(const float* in, float* out, int N, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i * stride < N) out[i] = in[i * stride] * 2.0f;  // 跳跃访问
}

// ─────────────────────────────────────────────
// Part 2: Unified Memory
// ─────────────────────────────────────────────
// cudaMallocManaged：CPU 和 GPU 都可以直接访问同一块内存
// 自动在需要时在 CPU/GPU 之间迁移数据（牺牲部分性能换简洁性）

__global__ void add_one(float* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) data[i] += 1.0f;
}

void demo_unified_memory() {
    printf("\n=== Unified Memory ===\n");
    const int N = 1 << 20;

    // 一次分配，CPU 和 GPU 都能用
    float* data;
    CUDA_CHECK(cudaMallocManaged(&data, N * sizeof(float)));

    // CPU 直接初始化（无需 cudaMemcpy）
    for (int i = 0; i < N; ++i) data[i] = (float)i;

    // GPU 计算
    add_one<<<(N + 255) / 256, 256>>>(data, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // CPU 直接读取结果（无需 cudaMemcpy）
    printf("data[0]=%.0f (期望 1.0)\n", data[0]);
    printf("data[100]=%.0f (期望 101.0)\n", data[100]);

    CUDA_CHECK(cudaFree(data));
}

// ─────────────────────────────────────────────
// Part 3: Pinned Memory vs 普通 malloc
// ─────────────────────────────────────────────
// Pinned（页锁定）内存：不会被 OS 换出到磁盘，可以直接 DMA 传输
// H2D/D2H 速度可提升 2~4x

void demo_pinned_vs_pageable() {
    printf("\n=== Pinned vs Pageable 内存传输速度 ===\n");
    const int N = 1 << 26;  // 64MB
    size_t bytes = N * sizeof(float);

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // 普通 malloc（Pageable Memory）
    float* h_pageable = (float*)malloc(bytes);
    for (int i = 0; i < N; ++i) h_pageable[i] = 1.0f;

    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(d_data, h_pageable, bytes, cudaMemcpyHostToDevice));
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float t_pageable; cudaEventElapsedTime(&t_pageable, start, stop);
    printf("Pageable H2D: %.1f ms  带宽: %.1f GB/s\n",
           t_pageable, bytes / t_pageable / 1e6);

    // cudaMallocHost（Pinned Memory）
    float* h_pinned;
    CUDA_CHECK(cudaMallocHost(&h_pinned, bytes));  // 分配 Pinned Memory
    for (int i = 0; i < N; ++i) h_pinned[i] = 1.0f;

    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(d_data, h_pinned, bytes, cudaMemcpyHostToDevice));
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float t_pinned; cudaEventElapsedTime(&t_pinned, start, stop);
    printf("Pinned   H2D: %.1f ms  带宽: %.1f GB/s\n",
           t_pinned, bytes / t_pinned / 1e6);

    printf("Pinned 提速: %.1fx\n", t_pageable / t_pinned);

    free(h_pageable);
    CUDA_CHECK(cudaFreeHost(h_pinned));
    CUDA_CHECK(cudaFree(d_data));
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

// ─────────────────────────────────────────────
// Part 4: 合并 vs 跨步访问对比
// ─────────────────────────────────────────────
void demo_coalesced_vs_strided() {
    printf("\n=== 合并访问 vs 跨步访问 ===\n");
    const int N = 1 << 24;
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    cudaMemset(d_in, 0, N * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    int block = 256, grid = (N + block - 1) / block;

    // 合并访问
    cudaEventRecord(start);
    for (int i = 0; i < 10; ++i)
        coalesced_read<<<grid, block>>>(d_in, d_out, N);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float t_coal; cudaEventElapsedTime(&t_coal, start, stop);
    printf("合并访问: %.2f ms\n", t_coal / 10);

    // 跨步 stride=32 访问（N 需要除以 stride）
    int stride = 32;
    cudaEventRecord(start);
    for (int i = 0; i < 10; ++i)
        strided_read<<<(N/stride + block - 1) / block, block>>>(d_in, d_out, N, stride);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float t_stride; cudaEventElapsedTime(&t_stride, start, stop);
    printf("跨步访问(stride=%d): %.2f ms\n", stride, t_stride / 10);
    printf("性能差距: %.1fx\n", t_stride / t_coal);

    cudaFree(d_in); cudaFree(d_out);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main() {
    printf("=== GPU 内存模型实验 ===\n");
    demo_unified_memory();
    demo_pinned_vs_pageable();
    demo_coalesced_vs_strided();
    printf("\n完成！\n");
    return 0;
}

/*
 * ══════════════════════════════════════════
 * 📝 练习题
 * ══════════════════════════════════════════
 *
 * 1. 用 Nsight Compute（ncu 命令）分析 coalesced_read 和 strided_read：
 *    ncu --set full ./memory_demo
 *    查看 "Global Memory" 部分，观察 "Sectors/Request" 指标的差异
 *
 * 2. 实现矩阵转置的两个版本：
 *    - 版本1：朴素转置（读合并但写不合并，或反之）
 *    - 版本2：用 Shared Memory 做中间缓冲，实现读写都合并
 *    - 对比性能
 *
 * 3. 实验 Constant Memory：
 *    - 声明一个 __constant__ float weights[256]
 *    - 用 cudaMemcpyToSymbol 赋值
 *    - 在 kernel 里读取，对比 Global Memory 的速度
 *    - 适合场景：所有线程都读同一个值（如卷积的 filter weights）
 */
