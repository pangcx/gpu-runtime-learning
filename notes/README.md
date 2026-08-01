# AI / 深度学习学习笔记

个人学习笔记，记录从 C++ 环境配置到深度学习核心概念的系统性学习过程。

## 目录

| 文件 | 内容 |
|------|------|
| [01-environment-setup.md](./01-environment-setup.md) | C++ 开发环境配置（cmake / ninja / LibTorch）|
| [02-pytorch-basics.md](./02-pytorch-basics.md) | PyTorch 基础：张量、常用函数 |
| [03-neural-network-concepts.md](./03-neural-network-concepts.md) | 神经网络核心概念：全连接、卷积、池化、激活函数 |
| [04-training-process.md](./04-training-process.md) | 训练流程：损失函数、优化器、反向传播 |
| [05-model-design-principles.md](./05-model-design-principles.md) | 模型设计原则：层数、滤波器数量、超参数 |
| [06-badcase-analysis.md](./06-badcase-analysis.md) | Badcase 分析与模型优化方法论 |
| [07-datasets.md](./07-datasets.md) | 公开数据集汇总 |
| [08-llm-vs-small-model.md](./08-llm-vs-small-model.md) | 大语言模型 vs 垂类小模型 |

## 配套代码

| 目录 | 说明 |
|------|------|
| `hello-libtorch/` | C++ LibTorch Hello World |
| `mnist-demo/train.py` | MNIST 全连接网络（MLP） |
| `mnist-demo/train_cnn.py` | MNIST 卷积神经网络（CNN） |
| `mnist-demo/export_badcases.py` | Badcase 导出与分析 |
