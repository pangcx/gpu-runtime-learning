import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ── 1. 数据准备 ──────────────────────────────────────────────
# 把图片转成张量，并做标准化（均值0.1307，标准差0.3081 是 MNIST 的统计值）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 自动下载 MNIST 数据集（第一次运行会下载，之后复用缓存）
train_data = datasets.MNIST('./data', train=True,  download=True, transform=transform)
test_data  = datasets.MNIST('./data', train=False, download=True, transform=transform)

# DataLoader：把数据分成一批一批（batch）喂给网络
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)

print(f"训练集大小: {len(train_data)} 张图")
print(f"测试集大小: {len(test_data)} 张图")
print(f"每张图尺寸: {train_data[0][0].shape}")  # [1, 28, 28] = 1通道, 28x28像素


# ── 2. 定义网络结构 ───────────────────────────────────────────
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 28x28 = 784 个像素输入 → 128个神经元 → 10个输出（对应0~9）
        self.net = nn.Sequential(
            nn.Flatten(),            # 把 [1,28,28] 展平成 [784]
            nn.Linear(784, 128),     # 全连接层
            nn.ReLU(),               # 激活函数
            nn.Linear(128, 64),      # 全连接层
            nn.ReLU(),               # 激活函数
            nn.Linear(64, 10),       # 输出层，10个类别
        )

    def forward(self, x):
        return self.net(x)

model = SimpleNet()
print(f"\n网络结构:\n{model}")


# ── 3. 训练配置 ───────────────────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # 优化器（负责更新参数）
loss_fn   = nn.CrossEntropyLoss()                          # 损失函数（分类任务用这个）


# ── 4. 训练循环 ───────────────────────────────────────────────
def train(epoch):
    model.train()
    total_loss = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()          # 清空上一步的梯度
        outputs = model(images)        # 前向传播：输入图片，得到预测
        loss = loss_fn(outputs, labels) # 计算损失
        loss.backward()                # 反向传播：计算梯度
        optimizer.step()               # 更新参数
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}  训练损失: {avg_loss:.4f}")


# ── 5. 测试（评估准确率）─────────────────────────────────────
def test():
    model.eval()
    correct = 0
    with torch.no_grad():   # 测试时不需要计算梯度，节省内存
        for images, labels in test_loader:
            outputs = model(images)
            pred = outputs.argmax(dim=1)   # 取概率最大的类别作为预测结果
            correct += (pred == labels).sum().item()

    accuracy = 100.0 * correct / len(test_data)
    print(f"         测试准确率: {accuracy:.2f}%\n")


# ── 6. 开始训练 ───────────────────────────────────────────────
print("\n开始训练...")
for epoch in range(1, 6):   # 训练 5 轮
    train(epoch)
    test()

print("训练完成！")
