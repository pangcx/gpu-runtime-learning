# GPU Runtime Learning

> 从零系统学习 AI 框架底层原理与 GPU 并行编程，面向 **AI 框架适配工程师 / AI 驱动工程师** 岗位方向。

---

## 学习初衷

国产 GPU 赛道正在快速发展，PyTorch、vLLM 等 AI 框架在自研 GPU 上的适配与优化成为核心竞争力。
本仓库记录了从 C++ 基础、深度学习原理，到 CUDA 并行编程的完整学习路径，目标是具备以下能力：

- 读懂并修改 PyTorch/vLLM 源码
- 编写和优化 CUDA/HIP kernel
- 将 AI 框架适配到自研 GPU 软件栈

---

## 目标岗位

### AI 框架适配工程师

| 要求 | 对应学习内容 |
|---|---|
| 精通 C++ 和 Python | `01-modern-cpp/`：内存管理、模板、并发、STL |
| 熟悉 PyTorch/vLLM 源码结构 | `notes/02~05`：PyTorch 基础与神经网络原理 |
| 熟悉异构计算和并行编程（CUDA/ROCm）| `02-cuda/`：4 个递进式 CUDA 实战项目 |
| 熟悉常用大语言模型 | `notes/08`：LLM vs 小模型原理对比 |

### AI 驱动工程师

| 要求 | 对应学习内容 |
|---|---|
| 掌握 CUDA/HIP 编程 | `02-cuda/`：向量加法、矩阵乘法、内存优化 |
| 了解 CUDA/ROCm Runtime API | `notes/09`：CUDA 编程模型与内存层次 |
| 熟悉 GPU 架构 | `notes/09`：SM、Warp、线程层次 |
| 熟悉主流 AI 框架 | `mnist-demo/`：PyTorch 完整训练实战 |

---

## 仓库结构

```
gpu-runtime-learning/
│
├── 01-modern-cpp/          # C++ 现代语法实践项目
│   ├── 01-memory/          # 智能指针、RAII、内存安全
│   ├── 02-templates/       # 函数模板、类模板、constexpr
│   ├── 03-concurrency/     # thread、mutex、atomic、future
│   └── 04-stl/             # vector、map、lambda、C++17
│
├── 02-cuda/                # CUDA 并行编程实践项目
│   ├── 01-hello-gpu/       # GPU 线程模型、Grid/Block/Thread
│   ├── 02-vector-add/      # 第一个计算 kernel，H2D/D2H 传输
│   ├── 03-matrix-mul/      # Shared Memory 优化，Tiling 策略
│   └── 04-memory-model/    # 内存层次，合并访问，Pinned Memory
│
├── hello-libtorch/         # C++ LibTorch Hello World
│   ├── CMakeLists.txt
│   └── main.cpp
│
├── mnist-demo/             # PyTorch MNIST 完整实战
│   ├── train.py            # MLP 版本（~97% 准确率）
│   ├── train_cnn.py        # CNN 版本（~99% 准确率）
│   └── export_badcases.py  # 错误样本导出与分析
│
└── notes/                  # 系统学习笔记（11 篇）
    ├── README.md
    ├── 01-environment-setup.md
    ├── 02-pytorch-basics.md
    ├── 03-neural-network-concepts.md
    ├── 04-training-process.md
    ├── 05-model-design-principles.md
    ├── 06-badcase-analysis.md
    ├── 07-datasets.md
    ├── 08-llm-vs-small-model.md
    ├── 09-cuda-basics.md
    ├── 10-career-ai-gpu-engineer.md
    └── 11-hardware-setup.md
```

---

## 学习路径

### 阶段一：C++ 现代语法（2~4 周）

GPU 框架开发的基础，PyTorch 源码全部用现代 C++ 编写。

```
01-memory      → 智能指针 / RAII / 内存安全
02-templates   → 泛型编程，读懂 PyTorch 多数据类型实现的基础
03-concurrency → 多线程并发，CUDA 并行思维的 CPU 侧对照
04-stl         → 容器与算法，日常开发工具箱
```

每个项目底部有 3 道练习题，先跑通示例再动手完成练习。

### 阶段二：CUDA 并行编程（4~8 周）

核心竞争力，AI 框架底层的实际执行层。

```
01-hello-gpu   → GPU 线程层次，理解 Grid/Block/Thread/Warp
02-vector-add  → 标准 CUDA 程序模板，H2D/D2H 数据流
03-matrix-mul  → Shared Memory 优化（AI 框架最核心的算法）
04-memory-model→ 内存层次与合并访问（性能优化的根本）
```

> **环境**：需要 NVIDIA GPU。推荐使用 [AutoDL](https://www.autodl.com) 按需租用，RTX 4090 约 ¥2/小时。

### 阶段三：PyTorch 源码与框架适配（规划中）

- PyTorch Autograd 实现原理
- C++ Extension 开发（自定义 CUDA 算子注册到 PyTorch）
- torch.compile / FX Graph 计算图编译
- vLLM PagedAttention 适配

---

## 快速开始

### C++ 项目（本地 macOS/Linux）

```bash
cd 01-modern-cpp/01-memory/build
cmake .. -G Ninja && ninja && ./memory_demo
```

### CUDA 项目（需要 NVIDIA GPU）

```bash
# 在 AutoDL 或有 NVIDIA GPU 的机器上
git clone https://github.com/pangcx/gpu-runtime-learning.git
cd gpu-runtime-learning/02-cuda/01-hello-gpu/build
cmake .. && make && ./hello_gpu
```

### Python 深度学习实验

```bash
pip install torch torchvision
cd mnist-demo

# 训练 MLP
python train.py

# 训练 CNN（更高精度）
python train_cnn.py

# 导出错误样本分析
python export_badcases.py
```

---

## 配套学习笔记

所有笔记都遵循"**核心结论 → 展开说明 → Q&A**"的结构，适合边实践边阅读。

| 笔记 | 内容摘要 |
|------|---------|
| [01 环境配置](notes/01-environment-setup.md) | cmake/ninja/LibTorch 安装避坑 |
| [02 PyTorch 基础](notes/02-pytorch-basics.md) | 张量、常用函数、autograd |
| [03 神经网络概念](notes/03-neural-network-concepts.md) | 全连接、卷积、池化、感受野 |
| [04 训练流程](notes/04-training-process.md) | 损失函数、优化器、反向传播 |
| [05 模型设计](notes/05-model-design-principles.md) | 超参数、层数、滤波器数量 |
| [06 Badcase 分析](notes/06-badcase-analysis.md) | 错误样本的分类与优化策略 |
| [07 数据集](notes/07-datasets.md) | 公开数据集汇总与选择建议 |
| [08 LLM vs 小模型](notes/08-llm-vs-small-model.md) | 大语言模型的训练阶段与区别 |
| [09 CUDA 基础](notes/09-cuda-basics.md) | 线程层次、Warp、内存模型 |
| [10 岗位分析](notes/10-career-ai-gpu-engineer.md) | AI框架适配 vs AI驱动工程师对比 |
| [11 硬件配置](notes/11-hardware-setup.md) | CUDA 学习主机选购建议 |

---

## 技术栈

- **语言**：C++17、Python 3.x、CUDA C++
- **框架**：PyTorch、LibTorch
- **工具**：cmake、ninja、nvcc、git
- **环境**：macOS（C++ 开发）、Ubuntu + NVIDIA GPU（CUDA 开发）
