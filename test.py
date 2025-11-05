import torch
from model.transformer_net import VisionTransformer
import torchvision.transforms as transforms
from PIL import Image
import os
import sys

img_size=224
patch_size=16
channels=3
num_features=768
depth=12
num_heads=12
mlp_ratio=4.0
qkv_bias=True
drop_rate=0.1
attn_drop_rate=0.1
drop_path_rate=0.1

classes=[]
num_classes=len(classes)
input_shape=(channels,img_size,img_size)

def load_model(device):
    model=VisionTransformer(
        input_shape=input_shape,
        patch_size=patch_size,
        channels=channels,
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
    model.load_state_dict(torch.load('best_vit.pth',map_location=device))
    model.eval()
    return model

def predict(img_path,model,device):
    transform=transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor()
    ])
    image=Image.open(img_path).convert('RGB')
    image=transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs=model(image)
        _,predicted=torch.max(outputs,1)
    
    return predicted.item()

if __name__=="__main__":
    img_path=sys.argv[1]
    if not os.path.exists(img_path):
        print(f"图像路径 {img_path} 不存在。")
        sys.exit(1)
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=load_model(device)
    predicted_class=predict(img_path,model,device)
    print(f"图片{img_path}的预测的类别为: {predicted_class}")
                