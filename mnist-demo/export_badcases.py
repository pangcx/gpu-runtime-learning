"""
运行方式：先跑 train_cnn.py 训练并保存模型，再运行本脚本导出 badcase。
也可以直接运行本脚本，会自动训练并导出。
"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import os, csv

# ── 网络结构（和 train_cnn.py 保持一致）────────────────────────
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 3), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2))
        self.fc    = nn.Sequential(nn.Flatten(), nn.Linear(1600, 128), nn.ReLU(),
                                   nn.Dropout(0.5), nn.Linear(128, 10))
    def forward(self, x):
        return self.fc(self.conv2(self.conv1(x)))


# ── 数据准备 ──────────────────────────────────────────────────
transform = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))])
test_data   = datasets.MNIST('./data', train=False, download=True, transform=transform)
# batch_size=1：一次处理一张图，方便记录每张图的信息
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)


# ── 加载或训练模型 ─────────────────────────────────────────────
model = CNN()
MODEL_PATH = 'cnn_model.pth'

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    print(f"已加载模型: {MODEL_PATH}")
else:
    print("未找到已保存的模型，开始快速训练...")
    train_data   = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    optimizer    = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn      = nn.CrossEntropyLoss()
    for epoch in range(5):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss_fn(model(images), labels).backward()
            optimizer.step()
        print(f"  Epoch {epoch+1}/5 完成")
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"模型已保存至 {MODEL_PATH}")


# ── 导出 badcase ───────────────────────────────────────────────
OUTPUT_DIR = 'badcases'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 按"真实标签/预测标签"分子目录，方便批量查看
# 例：badcases/true_7/pred_1/ 存放所有"真是7却被认成1"的图片

model.eval()
badcases = []  # 记录所有错误样本的信息

with torch.no_grad():
    for idx, (image, label) in enumerate(test_loader):
        output  = model(image)
        pred    = output.argmax(dim=1).item()
        true    = label.item()

        if pred != true:
            # 建立子目录
            sub_dir = os.path.join(OUTPUT_DIR, f'true_{true}', f'pred_{pred}')
            os.makedirs(sub_dir, exist_ok=True)

            # 把张量转回图片并保存
            # image shape: [1, 1, 28, 28]，需要 squeeze 掉多余维度
            img_array = image.squeeze().numpy()          # [28, 28]
            img_array = (img_array * 0.3081 + 0.1307)   # 反标准化，还原到 0~1
            img_array = (img_array * 255).clip(0, 255).astype('uint8')  # 转成 0~255
            img = Image.fromarray(img_array, mode='L')  # L = 灰度图

            filename = f'idx{idx:05d}.png'
            img.save(os.path.join(sub_dir, filename))

            # 记录置信度（网络对预测结果有多"自信"）
            probs      = torch.softmax(output, dim=1).squeeze()
            confidence = probs[pred].item()

            badcases.append({
                'index':      idx,
                'true_label': true,
                'pred_label': pred,
                'confidence': f'{confidence:.4f}',
                'image_path': os.path.join(sub_dir, filename),
            })


# ── 输出统计报告 ───────────────────────────────────────────────
print(f"\n共找到 {len(badcases)} 个 badcase（总测试集 10000 张）")
print(f"准确率: {(1 - len(badcases)/10000)*100:.2f}%")

# 按"哪对组合出错最多"排序
from collections import Counter
pair_counter = Counter((b['true_label'], b['pred_label']) for b in badcases)
print("\n出错最多的 10 种混淆组合:")
print(f"  {'真实':>4}  {'预测':>4}  {'数量':>6}")
for (true, pred), count in pair_counter.most_common(10):
    print(f"  {true:>4} → {pred:>4}  {count:>5} 次")

# 保存 CSV 方便后续分析
csv_path = os.path.join(OUTPUT_DIR, 'badcases.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['index','true_label','pred_label','confidence','image_path'])
    writer.writeheader()
    writer.writerows(badcases)

print(f"\n图片已保存至: {OUTPUT_DIR}/")
print(f"CSV 报告已保存至: {csv_path}")
print(f"\n目录结构示例:")
print(f"  badcases/")
print(f"  ├── true_4/pred_9/   ← 真是4，被认成9的图片")
print(f"  ├── true_7/pred_1/   ← 真是7，被认成1的图片")
print(f"  └── badcases.csv     ← 完整记录")
