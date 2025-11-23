# Vision Transformer (ViT) 实现

这是一个基于PyTorch的Vision Transformer（ViT）完整实现，支持自定义数据集和CIFAR-10数据集的训练与推理。

## 📋 项目简介

Vision Transformer (ViT) 是Google在2020年提出的纯Transformer架构用于图像分类的模型。本项目实现了标准的ViT架构，包含：

- **Patch Embedding**: 将图像切分为patches并进行线性映射
- **Multi-Head Self-Attention**: 多头自注意力机制
- **MLP**: 前馈神经网络
- **Position Encoding**: 位置编码
- **Classification Token**: 分类token


## 环境依赖

```bash
pip install torch torchvision tqdm matplotlib pillow numpy
```

或使用conda：
```bash
conda install pytorch torchvision tqdm matplotlib pillow numpy -c pytorch
```

## 🚀 使用方法

### 1. 自定义数据集训练

#### 数据集格式
```
data/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── class2/
│       ├── img3.jpg
│       └── img4.jpg
└── val/
    ├── class1/
    │   └── img5.jpg
    └── class2/
        └── img6.jpg
```

#### 训练
```bash
python3 train.py
```

#### 预测
```bash
python3 predict.py <图片路径>
```

### 2. CIFAR-10数据集训练

#### 下载CIFAR-10数据集
从[官网](https://www.cs.toronto.edu/~kriz/cifar.html)下载CIFAR-10 Python版本，解压到指定目录。

#### 训练
```bash
python3 train_cifar10.py
```

#### 预测
```bash
# 单个预测
python3 predict_cifar10.py <图片路径>

# Top-K预测
python3 predict_cifar10.py <图片路径> --top-k 3
```

## ⚙️ 模型配置

### 默认参数（ViT-Base）
- **图像尺寸**: 224×224
- **Patch尺寸**: 16×16
- **嵌入维度**: 768
- **编码器层数**: 12
- **注意力头数**: 12
- **MLP比例**: 4.0
- **Dropout率**: 0.1

### 自定义参数
可在训练脚本中修改以下参数：
```python
train_cifar10(
    img_size=224,        # 图像尺寸
    patch_size=16,       # Patch尺寸
    num_features=768,    # 嵌入维度
    depth=12,            # 编码器层数
    num_heads=12,        # 注意力头数
    mlp_ratio=4.0,       # MLP比例
    epochs=50,           # 训练轮数
    batch_size=32,       # 批次大小
    lr=3e-4             # 学习率
)
```

## 🏗️ 模型架构

### 核心组件

1. **VisionPatchEmbedding**: 
   - 使用卷积层将图像切分为patches
   - 线性映射到嵌入空间
   - LayerNorm归一化

2. **SelfAttention**:
   - 多头自注意力机制
   - 支持QKV偏置
   - Dropout正则化

3. **MLP**:
   - 两层全连接网络
   - GELU激活函数
   - Dropout正则化

4. **Block**:
   - Pre-Norm结构
   - 残差连接
   - DropPath随机深度

5. **VisonTransformer**:
   - 位置编码插值
   - 分类token
   - 多层Transformer编码器

### 参数量
- **ViT-Base**: ~86M参数
- **ViT-Small**: ~22M参数
- **ViT-Large**: ~307M参数
## 📝 训练日志

训练过程会自动保存：
- 损失曲线图 (`loss_curve.png` / `cifar10_training_curves.png`)
- 准确率曲线图 (`acc_curve.png`)
- 最优模型权重 (`best_vit.pth` / `best_vit_cifar10.pth`)



## 📚 参考文献

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)


**参考**: SkyXZ 
