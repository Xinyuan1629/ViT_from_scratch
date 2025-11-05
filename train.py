from model.transformer_net import VisionTransformer
from data_load import ViTDataLoad
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt



def train(
    root="",
    img_size=224,
    patch_size=16,
    channels=3,
    num_features=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4.0,
    qkv_bias=True,
    drop_rate=0.1,
    attn_drop_rate=0.1,
    drop_path_rate=0.1,
    epochs=50,
    batch_size=32,
    num_workers=4,
    lr=1e-4,
    device=None 
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader,val_loader=ViTDataLoad(root, batch_size,num_workers,img_size)
    num_classes = len(train_loader.dataset.classes)
    input_shape=(channels, img_size, img_size)
    
    
    model = VisionTransformer(
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
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    best_acc=0
    train_loss_list,train_acc_list=[],[]
    val_loss_list,val_acc_list=[],[]
    
    for epoch in range(epochs):
        model.train()
        total_loss,correct,total=0,0,0
        pbar=tqdm(train_loader,desc=f"Epoch {epoch+1}/{epochs}")
        for images,labels in pbar:
            images,labels=images.to(device),labels.to(device)
            optimizer.zero_grad()
            outputs=model(images)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            
            total_loss+=loss.item()*images.size(0)
            _,preds=torch.max(outputs,1)
            correct+=(preds==labels).sum().item()
            total+=labels.size(0)

        train_loss=total_loss/total
        train_acc=correct/total
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        pbar=tqdm(val_loader,desc=f"Epoch {epoch+1}/{epochs}")
        model.eval()
        with torch.no_grad():
            total_loss,correct,total=0,0,0
            for images,labels in pbar:
                images,labels=images.to(device),labels.to(device)
                outputs=model(images)
                loss=criterion(outputs,labels)
                total_loss+=loss.item()*images.size(0)
                #当前loss损失*当前batch的样本数量
                _,preds=torch.max(outputs,1)#最大值，索引
                correct+=(preds==labels).sum().item()#统计正确率
                total+=labels.size(0)#总样本数

        train_loss=total_loss/total
        train_acc=correct/total
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        
        #验证
        model.eval()
        with torch.no_grad():
            total_loss,correct,total=0,0,0
            for images,labels in pbar:
                images,labels=images.to(device),labels.to(device)
                outputs=model(images)
                loss=criterion(outputs,labels)
                val_total_loss += loss.item()*images.size(0)
                _,preds=outputs.max(1)
                val_correct+=(preds==labels).sum().item()
                val_total+=labels.size(0)#batch_size
                
            val_loss=val_total_loss/val_total
            val_acc=val_correct/val_total
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)
            
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
            # 保存最优模型
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), "best_vit.pth")

    #可视化
    plt.figure()
    plt.plot(train_loss_list,label='Train Loss')
    plt.plot(val_loss_list,label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig("loss_curve.png")
    plt.figure()
    plt.plot(train_acc_list,label='Train Acc')
    plt.plot(val_acc_list,label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title("Accuracy Curve")
    plt.savefig("accuracy_curve.png")
    print("训练完成 最优验证准确率： {:.4f}".format(best_acc))

if __name__=="__main__":
    train()#只有直接运行这个文件时才执行，要不然耗时