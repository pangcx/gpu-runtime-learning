/**
 * 项目02：向量加法
 *
 * 学习目标：
 *   - 掌握 CPU ↔ GPU 数据传输（cudaMalloc / cudaMemcpy / cudaFree）
 *   - 写第一个实际做计算的 kernel
 *   - 理解如何把数组下标映射到线程编号
 *   - 对比 CPU 和 GPU 的执行时间
 *
 * 核心模式（所有 GPU 计算程序都遵循这个流程）：
 *   1. CPU 分配并初始化数据
 *   2. cudaMalloc：在 GPU 显存上分配空间
 *   3. cudaMemcpy H2D：把数据从 CPU 内存拷贝到 GPU 显存
 *   4. 启动 kernel：GPU 并行计算
 *   5. cudaMemcpy D2H：把结果从 GPU 显存拷回 CPU 内存
 *   6. cudaFree：释放 GPU 显存
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { cudaError_t err = (call); \
         if (err != cudaSuccess) { \
             fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
             exit(EXIT_FAILURE); } } while(0)

// ─────────────────────────────────────────────
// GPU Kernel：并行向量加法
// ─────────────────────────────────────────────
// 每个线程负责计算一个元素：C[i] = A[i] + B[i]
__global__ void vector_add_gpu(const float* A, const float* B, float* C, int N) {
    // 计算当前线程负责的元素下标
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查：N 不一定是 blockDim 的整数倍
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// ─────────────────────────────────────────────
// CPU 版本（用于对比结果和耗时）
// ─────────────────────────────────────────────
void vector_add_cpu(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}

// ─────────────────────────────────────────────
// 验证结果是否正确
// ─────────────────────────────────────────────
bool verify(const float* cpu_result, const float* gpu_result, int N) {
    for (int i = 0; i < N; ++i) {
        if (fabs(cpu_result[i] - gpu_result[i]) > 1e-5f) {
            printf("结果不匹配！index=%d, cpu=%.6f, gpu=%.6f\n",
                   i, cpu_result[i], gpu_result[i]);
            return false;
        }
    }
    return true;
}

// ─────────────────────────────────────────────
// 计时工具
// ─────────────────────────────────────────────
struct Timer {
    cudaEvent_t start, stop;
    Timer()  { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~Timer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void begin() { cudaEventRecord(start); }
    float end()  { cudaEventRecord(stop); cudaEventSynchronize(stop);
                   float ms; cudaEventElapsedTime(&ms, start, stop); return ms; }
};

// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────
int main() {
    const int N = 1 << 24;  // 1600 万个元素
    const size_t bytes = N * sizeof(float);

    printf("向量大小: %d (%.1f MB)\n\n", N, bytes / 1e6);

    // ── 1. CPU 端：分配并初始化数据 ──────────────
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C_cpu = (float*)malloc(bytes);
    float* h_C_gpu = (float*)malloc(bytes);

    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    // ── 2. GPU 端：分配显存 ──────────────────────
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));  // 在 GPU 显存分配空间
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // ── 3. H2D：CPU 内存 → GPU 显存 ─────────────
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // ── 4. 启动 kernel ────────────────────────────
    // 每个 block 256 个线程，需要 ceil(N/256) 个 block
    int block_size = 256;
    int grid_size  = (N + block_size - 1) / block_size;

    printf("Grid 大小: %d blocks\n", grid_size);
    printf("Block 大小: %d threads\n", block_size);
    printf("总线程数: %d\n\n", grid_size * block_size);

    Timer timer;

    // GPU 计时
    timer.begin();
    vector_add_gpu<<<grid_size, block_size>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());    // 检查 kernel 启动是否有错
    float gpu_ms = timer.end();       // 隐含了 cudaDeviceSynchronize

    // ── 5. D2H：GPU 显存 → CPU 内存 ─────────────
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost));

    // CPU 计时
    auto t0 = std::chrono::high_resolution_clock::now();
    vector_add_cpu(h_A, h_B, h_C_cpu, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    float cpu_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    // ── 6. 验证 & 报告 ────────────────────────────
    bool ok = verify(h_C_cpu, h_C_gpu, N);
    printf("结果验证: %s\n\n", ok ? "✓ 正确" : "✗ 错误");
    printf("CPU 耗时: %.2f ms\n", cpu_ms);
    printf("GPU 耗时: %.2f ms\n", gpu_ms);
    printf("加速比:   %.1fx\n\n", cpu_ms / gpu_ms);

    float bandwidth = 3.0f * bytes / gpu_ms / 1e6;  // GB/s (读A, 读B, 写C)
    printf("GPU 内存带宽利用: %.1f GB/s\n", bandwidth);

    // ── 7. 释放资源 ───────────────────────────────
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);

    return 0;
}

/*
 * ══════════════════════════════════════════
 * 📝 练习题
 * ══════════════════════════════════════════
 *
 * 1. 调整 block_size，测试不同值的性能差异：
 *    - 分别测试 32 / 64 / 128 / 256 / 512 / 1024
 *    - 记录每次的 GPU 耗时，找出最优值
 *    - 思考：为什么 block_size 要是 32 的倍数？
 *
 * 2. 修改为向量点积（dot product）：
 *    - result = sum(A[i] * B[i])
 *    - 提示：每个线程算一个乘积，然后需要把所有线程的结果加起来
 *    - 这叫 "规约（Reduction）"，是 CUDA 最重要的模式之一
 *
 * 3. 测量 H2D 和 D2H 的数据传输时间：
 *    - 把 cudaMemcpy 也放进计时范围
 *    - 对比：数据传输耗时 vs 实际计算耗时
 *    - 这揭示了 GPU 加速的一个关键瓶颈：PCIe 带宽
 */
