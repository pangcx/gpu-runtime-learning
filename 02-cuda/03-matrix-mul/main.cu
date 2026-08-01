/**
 * 项目03：矩阵乘法（从朴素版到 Shared Memory 优化版）
 *
 * 学习目标：
 *   - 理解 2D 线程布局（threadIdx.x/y, blockIdx.x/y）
 *   - 理解 Global Memory 访问的性能瓶颈
 *   - 掌握 Shared Memory：block 内线程的高速缓存
 *   - 理解 Tiling（分块）优化策略
 *   - 这是 PyTorch Linear 层底层的核心算法
 *
 * 矩阵乘法：C = A × B，其中 A[M×K]，B[K×N]，C[M×N]
 *   C[row][col] = sum(A[row][k] * B[k][col]) for k in 0..K
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

const int TILE_SIZE = 16;  // 分块大小（16×16 = 256 个线程/block）

// ─────────────────────────────────────────────
// 版本1：朴素矩阵乘法（每次都访问 Global Memory）
// ─────────────────────────────────────────────
// 问题：Global Memory 延迟高（~500 cycles），A 和 B 的元素被重复读取
__global__ void matmul_naive(const float* A, const float* B, float* C,
                              int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // C 的行
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // C 的列

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];  // 每次都访问 Global Memory
        }
        C[row * N + col] = sum;
    }
}

// ─────────────────────────────────────────────
// 版本2：Shared Memory 优化版（Tiling）
// ─────────────────────────────────────────────
// 思路：把大矩阵分成 TILE_SIZE×TILE_SIZE 的小块
//       先把小块加载到 Shared Memory（速度快 100 倍）
//       然后在 Shared Memory 里做计算
__global__ void matmul_shared(const float* A, const float* B, float* C,
                               int M, int K, int N) {
    // __shared__：声明 Shared Memory，整个 block 共享，速度极快
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // 分块迭代：每次处理 K 维度上的一个 tile
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // 协作加载：每个线程加载一个元素到 Shared Memory
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        // __syncthreads()：等待 block 内所有线程都加载完毕
        // 必须！否则有的线程还没加载完，其他线程就开始计算了
        __syncthreads();

        // 在 Shared Memory 里计算这个 tile 的贡献
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        // 等待 block 内所有线程完成计算，再进入下一个 tile
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ─────────────────────────────────────────────
// CPU 版本（验证用）
// ─────────────────────────────────────────────
void matmul_cpu(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0;
            for (int k = 0; k < K; ++k)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

bool verify(const float* ref, const float* res, int size) {
    for (int i = 0; i < size; ++i)
        if (fabs(ref[i] - res[i]) / (fabs(ref[i]) + 1e-6f) > 1e-3f) {
            printf("Mismatch at %d: ref=%.4f res=%.4f\n", i, ref[i], res[i]);
            return false;
        }
    return true;
}

int main() {
    const int M = 1024, K = 1024, N = 1024;
    printf("矩阵乘法: A[%d×%d] × B[%d×%d] = C[%d×%d]\n\n", M, K, K, N, M, N);

    size_t bytesA = M * K * sizeof(float);
    size_t bytesB = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);

    // 分配 & 初始化 CPU 内存
    float *h_A = (float*)malloc(bytesA);
    float *h_B = (float*)malloc(bytesB);
    float *h_C_ref = (float*)malloc(bytesC);
    float *h_C     = (float*)malloc(bytesC);

    srand(42);
    for (int i = 0; i < M * K; ++i) h_A[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = rand() / (float)RAND_MAX;

    // CPU 基准
    auto t0 = std::chrono::high_resolution_clock::now();
    matmul_cpu(h_A, h_B, h_C_ref, M, K, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    float cpu_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    printf("CPU 耗时: %.1f ms\n", cpu_ms);

    // 分配 GPU 显存
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // 朴素版
    cudaEventRecord(start);
    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float naive_ms; cudaEventElapsedTime(&naive_ms, start, stop);
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));
    printf("GPU 朴素版: %.2f ms  %s  加速: %.0fx\n",
           naive_ms, verify(h_C_ref, h_C, M*N) ? "✓" : "✗", cpu_ms / naive_ms);

    // Shared Memory 优化版
    cudaEventRecord(start);
    matmul_shared<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float shared_ms; cudaEventElapsedTime(&shared_ms, start, stop);
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));
    printf("GPU Shared Memory版: %.2f ms  %s  加速: %.0fx  vs朴素: %.1fx\n",
           shared_ms, verify(h_C_ref, h_C, M*N) ? "✓" : "✗",
           cpu_ms / shared_ms, naive_ms / shared_ms);

    float flops = 2.0f * M * K * N;
    printf("\nTFLOPS (Shared版): %.2f\n", flops / shared_ms / 1e9);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_ref); free(h_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}

/*
 * ══════════════════════════════════════════
 * 📝 练习题
 * ══════════════════════════════════════════
 *
 * 1. 调整 TILE_SIZE（8 / 16 / 32），观察对性能的影响
 *    - 为什么有个最优值？Shared Memory 大小有限制
 *
 * 2. 理解 __syncthreads() 的必要性：
 *    - 尝试去掉第一个 __syncthreads()，观察结果是否正确
 *    - （在小矩阵上测试，结果会变成随机错误）
 *
 * 3. 用 cuBLAS 库实现同样的矩阵乘法：
 *    - 链接 -lcublas
 *    - 调用 cublasSgemm
 *    - 对比我们手写版本和 cuBLAS 的性能差距
 *    - 这就是为什么 PyTorch 底层用 cuBLAS 而不是自己写
 */
