# 02 - CUDA 并行编程实践

通过 4 个递进项目，掌握 CUDA 核心编程模型，为 AI 框架底层开发打基础。

## 环境要求

- NVIDIA GPU（RTX 4090 最佳）
- CUDA 12.x / 13.x
- Ubuntu 22.04
- cmake >= 3.18

## 项目列表

| 目录 | 主题 | 核心知识点 |
|------|------|-----------|
| `01-hello-gpu/` | GPU 线程模型 | Grid/Block/Thread 层次、threadIdx/blockIdx、GPU 属性查询 |
| `02-vector-add/` | 第一个计算 kernel | cudaMalloc/cudaMemcpy/cudaFree、H2D/D2H 数据传输、性能计时 |
| `03-matrix-mul/` | Shared Memory 优化 | 朴素 vs Tiling、__shared__、__syncthreads__、TFLOPS 计算 |
| `04-memory-model/` | 内存层次与优化 | 合并访问、Unified Memory、Pinned Memory、跨步访问代价 |

## 编译运行（在 AutoDL 上）

```bash
cd 02-cuda/01-hello-gpu/build
cmake .. && make && ./hello_gpu
```

> CMakeLists.txt 已设置 `CMAKE_CUDA_ARCHITECTURES 89`（RTX 4090 的 Ada Lovelace 架构）

## 学习顺序

```
01-hello-gpu    理解 GPU 的并行思维，是一切的起点
    ↓
02-vector-add   掌握 CPU↔GPU 数据流，所有 kernel 的标准模板
    ↓
03-matrix-mul   Shared Memory 优化，AI 框架最核心的算法
    ↓
04-memory-model 内存层次，性能优化的根本所在
```

## 关键概念速查

```
__global__          在 GPU 上执行，CPU 调用
__device__          在 GPU 上执行，GPU 调用
__shared__          Shared Memory，Block 内共享，速度极快
__syncthreads()     Block 内线程同步屏障

threadIdx.x/y/z     当前线程在 Block 内的编号
blockIdx.x/y/z      当前 Block 在 Grid 内的编号
blockDim.x/y/z      Block 的尺寸
gridDim.x/y/z       Grid 的尺寸

全局线程号 = blockIdx.x * blockDim.x + threadIdx.x
```
