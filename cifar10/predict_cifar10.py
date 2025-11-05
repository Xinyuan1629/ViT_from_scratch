import torch
from model.transformer_net import VisonTransformer
import torchvision.transforms as transforms
from PIL import Image
import sys
import os

# 配置参数（需与训练时一致）
img_size = 224
patch_size = 16
in_channels = 3
num_features = 768
depth = 12
num_heads = 12
mlp_ratio = 4.0
qkv_bias = True
drop_rate = 0.1
attn_drop_rate = 0.1
drop_path_rate = 0.1

# CIFAR-10类别名称
classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
num_classes = len(classes)
input_shape = (in_channels, img_size, img_size)

def load_model(device):
    model = VisonTransformer(
        input_shape=input_shape,
        patch_size=patch_size,
        in_channels=in_channels,
        num_classes=num_classes,
        num_features=num_features,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        norm_layer=torch.nn.LayerNorm,
        act_layer=torch.nn.GELU
    ).to(device)
    
    # 加载CIFAR-10训练的权重
    model.load_state_dict(torch.load("best_vit_cifar10.pth", map_location=device))
    model.eval()
    return model

def predict(img_path, model, device):
    # 与训练时相同的预处理
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img)
        probabilities = torch.softmax(output, dim=1)
        pred = output.argmax(dim=1).item()
        confidence = probabilities[0][pred].item()
    
    return classes[pred], confidence, probabilities[0]

def predict_top_k(img_path, model, device, k=3):
    """返回Top-K预测结果"""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img)
        probabilities = torch.softmax(output, dim=1)
        top_k_probs, top_k_indices = torch.topk(probabilities, k, dim=1)
    
    results = []
    for i in range(k):
        class_name = classes[top_k_indices[0][i].item()]
        prob = top_k_probs[0][i].item()
        results.append((class_name, prob))
    
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python3 predict_cifar10.py <图片路径> [--top-k K]")
        print("示例: python3 predict_cifar10.py test.jpg")
        print("示例: python3 predict_cifar10.py test.jpg --top-k 3")
        sys.exit(1)
    
    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"图片不存在: {img_path}")
        sys.exit(1)
    
    # 检查是否要显示Top-K结果
    show_top_k = False
    k = 3
    if len(sys.argv) >= 4 and sys.argv[2] == "--top-k":
        show_top_k = True
        k = int(sys.argv[3])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    model = load_model(device)
    
    if show_top_k:
        results = predict_top_k(img_path, model, device, k)
        print(f"\n图片 {img_path} 的Top-{k}预测结果:")
        for i, (class_name, prob) in enumerate(results, 1):
            print(f"{i}. {class_name}: {prob:.4f} ({prob*100:.2f}%)")
    else:
        pred_class, confidence, _ = predict(img_path, model, device)
        print(f"图片 {img_path} 的预测类别为: {pred_class}")
        print(f"置信度: {confidence:.4f} ({confidence*100:.2f}%)") 