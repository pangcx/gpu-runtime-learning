# CUDA 学习主机配置建议

## 核心原则

**GPU 是绝对重点**，其他配件够用即可，预算向 GPU 倾斜。

---

## GPU 推荐

| 预算 | 型号 | 显存 | 适合场景 |
|---|---|---|---|
| ~3000元 | RTX 4060 | 8GB | CUDA 入门、小模型训练、学习足够 |
| ~5000元 | **RTX 4070** | 12GB | 能跑 7B 级别模型，推荐 ✓ |
| ~8000元 | **RTX 4070 Ti Super** | 16GB | 能跑 13B 模型，调试 vLLM 更宽裕 |
| ~12000元 | RTX 4080 Super | 16GB | 接近专业级，性价比开始下降 |
| ~25000元 | RTX 4090 | 24GB | 最强消费级，24GB 显存跑大模型 |

**学习 CUDA + 框架适配的最优选：RTX 4070 Ti Super（16GB）**

- 16GB 显存是关键门槛：能加载 13B 模型做推理优化，够用 2~3 年
- 8GB 的 4060 显存太小，学 vLLM/内存管理时很快撞墙
- 4090 性价比急剧下降，学习阶段用不到 24GB

---

## 整机配置建议（约 1.5 万元）

| 配件 | 推荐型号 | 预算 | 说明 |
|---|---|---|---|
| **GPU** | RTX 4070 Ti Super | ~7500 | 核心，向这里倾斜 |
| **CPU** | AMD Ryzen 7 7700X | ~1800 | 8核16线程，数据预处理够用；不需要顶配 |
| **内存** | DDR5 64GB（32×2）| ~1000 | AI 开发内存要大，32GB 容易不够 |
| **主板** | B650 系列 | ~800 | 支持 PCIe 4.0，不需要 X670 |
| **SSD** | 2TB NVMe | ~500 | 数据集大，1TB 很快不够 |
| **电源** | 850W 金牌 | ~500 | 4070 Ti Super TDP 285W，留足余量 |
| **散热** | 240/360 水冷 | ~400 | 编译 PyTorch 时 CPU 也会很热 |
| **机箱** | 支持 ATX + 3槽 GPU 即可 | ~300 | |

**合计约 12800~15000 元**

---

## 操作系统

**装 Ubuntu 22.04 LTS**，不要装 Windows。

原因：
- CUDA toolkit、NVIDIA 驱动在 Linux 上安装更简单稳定
- PyTorch / vLLM 源码编译在 Linux 环境更成熟
- 几乎所有 AI 框架的 CI/CD 都在 Linux 上跑
- Windows WSL2 有额外开销，会给 GPU 内存管理带来不必要的复杂性

---

## 如果暂时不想装机：云 GPU 方案

在买机器之前，可以先用云 GPU 环境练习 CUDA 基础：

| 平台 | 价格 | 适合 |
|---|---|---|
| **AutoDL** | ¥0.5~2/小时 | 国内首选，RTX 3090/4090 按小时计费 |
| **Google Colab Pro** | ¥70/月 | T4/A100，写 notebook 方便 |
| **Vast.ai** | $0.2~0.5/小时 | 国外，GPU 型号多 |

CUDA 基础语法学习（前 4 周）在 Colab 免费版或 AutoDL 上就够，不用急着装机。

### AutoDL 使用要点

AutoDL 上的实例通常已预装：
- CUDA Toolkit（当前版本如 13.2）
- Python 环境（Anaconda）
- 常用深度学习库（PyTorch 等）

首次使用建议：
```bash
# 确认 GPU 和 CUDA 版本
nvidia-smi && nvcc --version

# 安装开发工具
apt-get update && apt-get install -y git cmake ninja-build build-essential
```

---

## 学习路径和硬件对应关系

| 学习阶段 | 硬件需求 | 建议 |
|---|---|---|
| C++ 基础（01-modern-cpp）| 无需 GPU | 本地 Mac 即可 |
| CUDA 基础（02-cuda）| 需要 NVIDIA GPU | AutoDL 按需租用 |
| PyTorch 源码学习 | 需要 NVIDIA GPU | AutoDL 或装机 |
| vLLM 适配 | 16GB+ 显存 | 建议装机（RTX 4070 Ti Super）|
