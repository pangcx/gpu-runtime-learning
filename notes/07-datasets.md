# 07 - 公开数据集汇总

## 核心结论

- **数据集是 AI 项目核心资产**，采集/清洗/标注通常占整个项目工作量的 60~80%
- 图像入门路径：MNIST → CIFAR-10 → ImageNet（预训练微调）
- NLP 入门路径：IMDb → SQuAD → 微调 BERT/GPT
- **下一步推荐 CIFAR-10**：格式与 MNIST 类似但彩色+10类自然物体，难度明显上升
- 找数据集：Kaggle（竞赛+排行榜）/ Hugging Face（NLP）/ torchvision（图像内置）/ Papers with Code（SOTA 基准）

---

## ❓ 除了 MNIST 之外，还有哪些好用的公开数据集？

> 数据集是 AI 项目的核心资产，往往比代码更重要。
> 数据采集、清洗、标注通常占整个项目工作量的 60~80%。

## 图像分类

| 数据集 | 规模 | 难度 | 说明 |
|---|---|---|---|
| **MNIST** | 7万张，10类 | ★☆☆ | 手写数字，入门经典 |
| **Fashion-MNIST** | 7万张，10类 | ★☆☆ | 服装品类，格式与 MNIST 完全相同，可直接替换 |
| **CIFAR-10/100** | 6万张，10/100类 | ★★☆ | 飞机/汽车/猫狗，32×32 彩色，进阶首选 |
| **STL-10** | 13万张，10类 | ★★☆ | 96×96，适合练习半监督学习 |
| **ImageNet** | 120万张，1000类 | ★★★ | 工业标准基准，主流网络（ResNet/VGG）都在此评测 |

## 目标检测 / 分割

| 数据集 | 说明 |
|---|---|
| **COCO** | 33万张，80类，带边界框+分割掩码，检测任务主流基准 |
| **Pascal VOC** | 20类，比 COCO 小，适合入门检测任务 |
| **Open Images** | Google 出品，900万张，标注极丰富 |

## 自然语言处理

| 数据集 | 任务 | 说明 |
|---|---|---|
| **IMDb** | 情感分类 | 电影评论正负面，NLP 入门经典 |
| **SQuAD** | 阅读理解 | 给段落+问题，模型找答案，BERT 评测标准 |
| **Wikipedia Dump** | 预训练语料 | 语言模型基础训练集 |
| **Common Crawl** | 通用语料 | GPT 系列主要数据来源 |

## 语音

| 数据集 | 说明 |
|---|---|
| **LibriSpeech** | 1000小时英语，语音识别标准基准 |
| **Common Voice** | Mozilla 出品，多语言，含中文 |

## 结构化数据

| 数据集 | 任务 | 说明 |
|---|---|---|
| **Titanic**（Kaggle）| 分类 | 预测生还率，机器学习入门经典 |
| **House Prices**（Kaggle）| 回归 | 房价预测 |
| **UCI ML Repository** | 多种 | 几百个不同领域小数据集 |

## 数据集获取平台

| 平台 | 地址 | 特点 |
|---|---|---|
| **Kaggle** | kaggle.com | 最大竞赛平台，数据+代码+排行榜 |
| **Hugging Face** | huggingface.co/datasets | NLP 数据集最全，一行代码加载 |
| **torchvision.datasets** | PyTorch 内置 | MNIST/CIFAR/ImageNet 直接下载 |
| **Papers with Code** | paperswithcode.com | 论文+数据集+SOTA 排行榜 |

## 学习路径建议

```
图像:  MNIST → CIFAR-10 → ImageNet（用预训练模型微调）
NLP:   IMDb情感分类 → SQuAD → 微调 BERT/GPT
表格:  Titanic → Kaggle 竞赛
```

下一步推荐：**CIFAR-10**
- 格式和 MNIST 类似，代码改动很小
- 彩色图（3通道）+ 10个自然物体类别
- 难度明显上升，能有效检验 CNN 设计能力
