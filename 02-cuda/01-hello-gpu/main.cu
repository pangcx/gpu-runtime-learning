/**
 * 项目01：Hello GPU
 *
 * 学习目标：
 *   - 理解 GPU 的线程层次：Grid → Block → Thread
 *   - 写第一个 __global__ kernel 函数
 *   - 理解 CPU（Host）和 GPU（Device）的分工
 *   - 掌握 CUDA 错误检查的习惯
 *
 * 编译运行：
 *   cd build && cmake .. && make && ./hello_gpu
 *
 * CUDA 文件扩展名是 .cu（不是 .cpp）
 */

#include <stdio.h>
#include <cuda_runtime.h>

// ─────────────────────────────────────────────
// CUDA 错误检查宏（好习惯：每个 CUDA 调用都要检查）
// ─────────────────────────────────────────────
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ─────────────────────────────────────────────
// Part 1: 最简单的 kernel
// ─────────────────────────────────────────────

// __global__ 表示这个函数在 GPU 上执行，由 CPU 调用
// 每个线程执行同一份代码，但 threadIdx/blockIdx 不同
__global__ void hello_kernel() {
    // threadIdx.x：当前线程在 block 内的编号
    // blockIdx.x ：当前 block 在 grid 内的编号
    // blockDim.x ：每个 block 有多少个线程
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;  // 全局线程编号

    printf("Hello from GPU! block=%d, thread=%d, global_id=%d\n",
           blockIdx.x, threadIdx.x, global_id);
}

// ─────────────────────────────────────────────
// Part 2: 查询 GPU 信息
// ─────────────────────────────────────────────
void print_gpu_info() {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    printf("\n=== GPU 信息 ===\n");
    printf("GPU 数量: %d\n", device_count);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));  // 查询第 0 块 GPU

    printf("型号: %s\n", prop.name);
    printf("显存: %.1f GB\n", prop.totalGlobalMem / 1e9);
    printf("SM 数量: %d\n", prop.multiProcessorCount);         // Streaming Multiprocessor
    printf("每个 SM 最多线程: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("每个 Block 最多线程: %d\n", prop.maxThreadsPerBlock);
    printf("Warp 大小: %d\n", prop.warpSize);                  // 32，CUDA 并行的最小单位
    printf("计算能力: %d.%d\n", prop.major, prop.minor);       // 4090 = 8.9
    printf("\n");
}

// ─────────────────────────────────────────────
// Part 3: 线程层次可视化
// ─────────────────────────────────────────────
__global__ void show_hierarchy() {
    // 计算各种 ID，帮助理解 Grid-Block-Thread 结构
    int tid   = threadIdx.x;                         // block 内线程号
    int bid   = blockIdx.x;                          // grid 内 block 号
    int bdim  = blockDim.x;                          // block 的大小
    int gid   = bid * bdim + tid;                    // 全局线程号

    if (gid < 8) {  // 只打印前 8 个，避免刷屏
        printf("Grid[%d] Block[%d] Thread[%d] → GlobalID=%d\n",
               gridDim.x, bid, tid, gid);
    }
}

// ─────────────────────────────────────────────
// Main（运行在 CPU 上）
// ─────────────────────────────────────────────
int main() {
    print_gpu_info();

    printf("=== Part 1: Hello Kernel ===\n");
    // <<<grid大小, block大小>>>：启动 2 个 block，每个 block 4 个线程 = 8 个线程
    hello_kernel<<<2, 4>>>();
    CUDA_CHECK(cudaDeviceSynchronize());  // 等待所有 GPU 线程完成

    printf("\n=== Part 2: 线程层次 ===\n");
    // 4 个 block，每个 block 8 个线程 = 32 个线程总计
    show_hierarchy<<<4, 8>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("\n=== 关键概念总结 ===\n");
    printf("Thread  : GPU 最基本的执行单元\n");
    printf("Warp    : 32 个线程为一组，同时执行同一条指令（SIMT）\n");
    printf("Block   : 一组线程，共享 Shared Memory，可以同步\n");
    printf("Grid    : 所有 Block 的集合，一次 kernel 调用的全部线程\n");
    printf("SM      : 硬件上的流式多处理器，负责调度执行 Warp\n");

    return 0;
}

/*
 * ══════════════════════════════════════════
 * 📝 练习题
 * ══════════════════════════════════════════
 *
 * 1. 修改 hello_kernel 的启动参数：
 *    - 改成 <<<4, 32>>>（4个block，每个32个线程）
 *    - 观察输出，理解线程编号的计算方式
 *    - 思考：为什么 blockDim 通常选 32 的倍数？
 *
 * 2. 写一个 kernel，让每个线程打印自己的 global_id 的平方：
 *    - 启动 1 个 block，256 个线程
 *    - 只打印 global_id 为偶数的线程的结果
 *
 * 3. 查阅 RTX 4090 的规格：
 *    - 有多少个 SM？
 *    - 每个 SM 最多跑多少个线程？
 *    - 理论最大并行线程数是多少？
 *    - 和 CPU 的并发线程数对比，差距有多大？
 */
