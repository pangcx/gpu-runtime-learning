# 01 - Modern C++ 实践练习

通过 4 个小项目，覆盖成为 AI 框架工程师所需的 C++ 核心技能。

## 项目列表

| 目录 | 主题 | 核心知识点 | 与 AI 框架的关联 |
|------|------|-----------|----------------|
| `01-memory/` | 内存管理与智能指针 | unique_ptr / shared_ptr / RAII | PyTorch Tensor 的内存生命周期管理 |
| `02-templates/` | 模板与泛型编程 | 函数模板 / 类模板 / 特化 / constexpr | PyTorch 支持 float/half/int 等多种数据类型的底层实现 |
| `03-concurrency/` | 多线程与并发 | thread / mutex / atomic / future | CUDA 并行编程的 CPU 侧基础；算子调度 |
| `04-stl/` | STL 容器与算法 | vector / map / lambda / C++17 | PyTorch 源码大量使用；算子注册表用 unordered_map |

## 每个项目的使用方式

```bash
cd 01-memory/build   # 进入对应项目的 build 目录
cmake .. -G Ninja    # 第一次配置（只需一次）
ninja                # 编译
./memory_demo        # 运行
```

## 学习顺序建议

```
01-memory      先搞清楚内存模型，是理解一切的基础
    ↓
02-templates   泛型编程，读懂 PyTorch C++ 源码的关键
    ↓
03-concurrency 并发基础，CUDA 并行思维的 CPU 侧对照
    ↓
04-stl         现代 C++ 日常开发工具箱
```

## 每个项目的结构

```
xx-topic/
├── CMakeLists.txt   # 构建配置
├── main.cpp         # 含注释的示例代码 + 底部练习题
└── build/           # 编译产物（git 忽略）
```

**学习方式**：
1. 先运行 `main.cpp`，看懂输出
2. 修改代码，观察行为变化
3. 完成底部 `练习题`，自己实现后对比
