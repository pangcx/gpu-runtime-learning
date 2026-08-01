import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

# ── 1. 数据准备（和之前一样）────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST('./data', train=True,  download=True, transform=transform)
test_data  = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)


# ── 2. 定义 CNN 网络结构 ──────────────────────────────────────
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 卷积块 1
        # Conv2d(输入通道数, 输出通道数, 卷积核大小)
        # 输入: [batch, 1, 28, 28]  → 输出: [batch, 32, 26, 26]
        # 解释: 用 32 个 3×3 的滤波器扫描图片，每个滤波器学习一种特征（边缘/笔画）
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),   # 1通道输入, 32个滤波器, 3×3卷积核
            nn.ReLU(),
            nn.MaxPool2d(2),                   # 2×2 最大池化，尺寸减半: [batch, 32, 13, 13]
        )

        # 卷积块 2
        # 输入: [batch, 32, 13, 13] → 输出: [batch, 64, 11, 11] → 池化后: [batch, 64, 5, 5]
        # 解释: 在第一层特征的基础上，学习更抽象的组合特征
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),  # 32通道输入, 64个滤波器
            nn.ReLU(),
            nn.MaxPool2d(2),                   # 尺寸再减半: [batch, 64, 5, 5]
        )

        # 全连接块（在卷积提取特征后做最终分类）
        # 64 × 5 × 5 = 1600 个特征 → 128 → 10
        self.fc = nn.Sequential(
            nn.Flatten(),                      # [batch, 64, 5, 5] → [batch, 1600]
            nn.Linear(1600, 128),
            nn.ReLU(),
            nn.Dropout(0.5),                   # 随机丢弃 50% 神经元，防止过拟合
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return x

model = CNN()

# 打印每层的参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"网络结构:\n{model}")
print(f"\n总参数量: {total_params:,}")  # 对比 MLP 的参数量


# ── 3. 训练配置 ───────────────────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn   = nn.CrossEntropyLoss()


# ── 4. 训练 + 测试 ────────────────────────────────────────────
def train(epoch):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def test():
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            pred = model(images).argmax(dim=1)
            correct += (pred == labels).sum().item()
    return 100.0 * correct / len(test_data)


print("\n开始训练 CNN...")
start = time.time()

for epoch in range(1, 6):
    loss = train(epoch)
    acc  = test()
    print(f"Epoch {epoch}  损失: {loss:.4f}  测试准确率: {acc:.2f}%")

elapsed = time.time() - start
print(f"\n训练完成！总耗时: {elapsed:.1f} 秒")

# 保存模型权重，供 export_badcases.py 使用
torch.save(model.state_dict(), 'cnn_model.pth')
print("模型已保存至 cnn_model.pth")
