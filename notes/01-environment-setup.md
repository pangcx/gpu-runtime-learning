# 01 - C++ 开发环境配置

## 核心结论

- cmake 是「翻译官」，把 CMakeLists.txt 翻译成 ninja 能读的施工图；ninja 是「施工队」，调用 clang 实际编译
- 日常只需要 `ninja`，只有第一次或修改 CMakeLists.txt 后才需要重跑 `cmake`
- LibTorch 内嵌在 PyTorch Python 包里，不需要单独下载，通过 `torch.utils.cmake_prefix_path` 指向它
- Homebrew 在旧版 macOS 上会从源码编译 cmake，耗时极长；改用 `pip install cmake` 下载预编译包更快
- 修改代码只需改 `main.cpp`，然后 `ninja && ./hello_libtorch` 一键编译运行

---

## ❓ 如何修改代码、如何编译、如何运行？

### 项目结构

```
hello-libtorch/
├── main.cpp          ← 你写代码的地方
├── CMakeLists.txt    ← 告诉编译器"怎么编译"
└── build/            ← 编译产物（自动生成）
    └── hello_libtorch  ← 编译出来的可执行文件
```

- 改代码：直接编辑 `main.cpp`，保存即可
- `CMakeLists.txt` 只有**新增 .cpp 文件**或**新增依赖库**时才需要改

### 编译

```bash
cd hello-libtorch/build
ninja                  # 只重新编译有改动的文件，很快
```

第一次（或删除 build 目录后）需要先跑 cmake：

```bash
cd hello-libtorch
mkdir -p build && cd build
cmake .. -G Ninja
ninja
```

### 运行

```bash
./hello_libtorch
```

### 改完代码后的一键编译运行

```bash
ninja && ./hello_libtorch
```

---

## ❓ `cmake .. -G Ninja` 是什么意思？ninja 是做什么的？

两个工具分工合作：

| 角色 | 类比 |
|---|---|
| `CMakeLists.txt` | 建筑设计图（你写的）|
| `cmake` | 把设计图转成施工规范（build.ninja）|
| `ninja` | 按施工规范实际盖房子（调用 clang 编译）|
| `clang` | 具体的工人（ninja 调用它）|

- `cmake .. -G Ninja`：读取上一层目录的 CMakeLists.txt，翻译成 Ninja 格式
- `ninja`：读取 build.ninja，执行实际编译，只重新编译有变动的文件

平时**只需要跑 `ninja`**，cmake 只在第一次或修改了 CMakeLists.txt 后才需要重跑。

---

## 系统环境

- macOS 12 Monterey，Intel x86_64
- Apple Clang 13（随 Xcode Command Line Tools 安装）
- 无 NVIDIA GPU → PyTorch 使用 CPU 模式

## 安装的工具

| 工具 | 版本 | 安装方式 |
|------|------|---------|
| cmake | 4.4.0 | `pip3 install cmake` |
| ninja | 1.13.2 | `brew install ninja` |
| pkg-config | 3.0.4 | `brew install pkg-config` |
| PyTorch | 2.2.2 | `pip3 install torch torchvision torchaudio` |

> **踩坑记录**：Homebrew 在旧版 macOS 上没有 cmake 的预编译包，会从源码编译耗时 15~30 分钟甚至失败。改用 `pip install cmake` 更快。

## 环境变量配置

```bash
# 写入 ~/.zshrc，让 cmake 命令全局可用
export PATH="$HOME/Library/Python/3.9/bin:$PATH"
```

## LibTorch 路径

LibTorch 内嵌在 PyTorch Python 包里，不需要单独下载：

```bash
python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
# /usr/local/lib/python3.9/site-packages/torch/share/cmake
```

在 CMakeLists.txt 里引用：

```cmake
execute_process(
    COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
find_package(Torch REQUIRED)
```
