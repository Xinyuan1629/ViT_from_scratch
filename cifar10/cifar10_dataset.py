import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class CIFAR10Dataset(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        super().__init__()
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        
        # CIFAR-10类别名称
        self.classes = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.data = []
        self.targets = []
        
        if self.train:
            for i in range(1, 6):
                batch_file = os.path.join(root, f'data_batch_{i}')
                with open(batch_file, 'rb') as f:
                    batch_data = pickle.load(f, encoding='bytes')
                    self.data.append(batch_data[b'data'])
                    self.targets.extend(batch_data[b'labels'])
            self.data = np.vstack(self.data)
        else:
            test_file = os.path.join(root, 'test_batch')
            with open(test_file, 'rb') as f:
                test_data = pickle.load(f, encoding='bytes')
                self.data = test_data[b'data']
                self.targets = test_data[b'labels']
        
        # 将数据reshape为图像格式 (N, 32, 32, 3)
        self.data = self.data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        
        print(f"加载了 {len(self.data)} 张CIFAR-10图像用于{'训练' if train else '测试'}，共{len(self.classes)}个类别")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img = self.data[index]
        target = self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target

def CIFAR10DataLoad(root, batch_size, num_workers=4, img_size=224):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.RandomRotation(10), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = CIFAR10Dataset(
        root=root,
        train=True,
        transform=train_transform
    )
    
    test_dataset = CIFAR10Dataset(
        root=root,
        train=False,
        transform=test_transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    # 测试数据加载器
    root = "./cifar10"
    train_loader, test_loader = CIFAR10DataLoad(root, batch_size=32)
    # 测试一个batch
    for images, labels in train_loader:
        print(f"图像batch形状: {images.shape}")
        print(f"标签batch形状: {labels.shape}")
        print(f"标签范围: {labels.min()} - {labels.max()}")
        break 