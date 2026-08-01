# 02 - PyTorch 基础：张量与常用函数

## 核心结论

- **张量是多维数组**，标量/向量/矩阵都是张量的特殊情况；图片 = 3维，一批图片 = 4维
- `torch::` 是命名空间前缀，表示函数属于 torch 库；Python 里写 `torch.`，C++ 里写 `torch::`
- 创建张量的常用函数：`zeros` / `ones` / `rand` / `randn` / `arange`
- 计算常用函数：`+` `-` `*` / `torch.mm` / `torch.sum` / `torch.mean`
- 形状操作：`reshape` / `transpose` / `cat` / `squeeze` / `unsqueeze`
- 自动求导：用 `requires_grad=True` 标记，调用 `.backward()` 后用 `.grad` 取得梯度

---

## ❓ 注释里写“创建一个随机张量”，什么是张量？

张量是**多维数组**，是 PyTorch 存储和运算数据的基本单位。

| 维度 | 名称 | 例子 |
|------|------|------|
| 0维 | 标量 | `3.14` |
| 1维 | 向量 | `[1, 2, 3]` |
| 2维 | 矩阵 | 表格（行×列）|
| 3维+ | 张量 | 多个矩阵叠加 |

标量/向量/矩阵都是张量的特殊情况，张量是这些概念的统一称呼。

实际 AI 数据天然就是多维的：
- 一张图片 = `[通道, 高, 宽]` = 3维张量
- 一批图片 = `[批次, 通道, 高, 宽]` = 4维张量

PyTorch 的核心就是对张量做各种数学运算，然后用这些运算训练神经网络。

---

## ❓ `torch::ones` 是什么意思？

`torch::ones({2, 3})` 创建一个**全部填满 1 的张量**，形状是 2 行 3 列：

```
1  1  1
1  1  1
```

`torch::` 是**命名空间**，表示"这个函数来自 torch 这个库"。C++ 里用 `::` 表示归属关系，Python 里写 `torch.`，作用相同。

---

## ❓ `torch` 命名空间里有多少函数，最常用的有哪些？

有数百个，日常用到的集中在以下几类：

### 创建张量

```python
torch.tensor([1, 2, 3])          # 从数据直接创建
torch.zeros([3, 4])              # 全 0
torch.ones([3, 4])               # 全 1
torch.rand([3, 4])               # 0~1 均匀随机
torch.randn([3, 4])              # 正态分布随机（均值0，标准差1）
torch.eye(3)                     # 单位矩阵
torch.arange(0, 10, 2)           # [0, 2, 4, 6, 8]
```

### 数学运算

```python
a + b                            # 逐元素加（等价 torch.add）
a * b                            # 逐元素乘（等价 torch.mul）
torch.mm(a, b)                   # 矩阵乘法（2D）
torch.matmul(a, b)               # 通用矩阵乘法（多维）
torch.sum(t)                     # 求和
torch.mean(t)                    # 均值
torch.max(t)                     # 最大值
torch.sqrt(t)                    # 开方
```

### 形状操作

```python
torch.reshape(t, [2, 6])         # 改变形状（总元素数不变）
torch.transpose(t, 0, 1)         # 转置
torch.cat([a, b], dim=0)         # 沿已有维度拼接
torch.stack([a, b], dim=0)       # 新增维度后拼接
torch.squeeze(t)                 # 去掉大小为 1 的维度
torch.unsqueeze(t, 0)            # 在指定位置插入维度
```

### 自动求导

```python
x = torch.tensor([2.0], requires_grad=True)  # 标记需要求梯度
y = 3 * x ** 2                               # y = 3x²
y.backward()                                  # 反向传播
print(x.grad)                                 # dy/dx = 6x = 12
```

### 神经网络相关（torch.nn 子命名空间）

| 函数 | 含义 |
|---|---|
| `nn.Linear` | 全连接层 |
| `nn.Conv2d` | 二维卷积层 |
| `nn.ReLU` | 激活函数 |
| `nn.Softmax` | 概率归一化 |
| `nn.CrossEntropyLoss` | 交叉熵损失函数 |
| `nn.MSELoss` | 均方误差损失函数 |

> 刚开始只需要熟悉**创建 + 数学运算**，神经网络部分等概念建立后再深入。
