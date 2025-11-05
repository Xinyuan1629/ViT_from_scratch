import torch
import torch.nn as nn
from torch.nn import functional as F

class VisionPatchEmbedding(nn.Module):
    def __init__(self,image_size,patch_size,channels,embed_dim,flatten=True):
        super().__init__()
        self.image_size=image_size
        self.patch_size=patch_size
        self.embed_dim=embed_dim
        #展平成向量后，再通过一个 Linear 层映射到模型的 embedding 空间
        # （手动设置，ViT-Base为 768 维，ViT-Large为1024，ViT-Huge为1280，通常使用768）
        self.flatten=flatten
        self.channels=channels
        
        #卷积实现patch划分，channel->embed_dim被整合了
        self.proj=nn.Conv2d(self.channels,self.embed_dim,self.patch_size,self.patch_size)
        #取norm准备进入MHA
        self.norm=nn.LayerNorm(self.embed_dim)
        
    def forward(self,x):
        x=self.proj(x)
        if self.flatten:
            x=x.flatten(2).transpose(1,2)#BCHW -> BCN -> BNC N:Patch**2将平面二维乘积“展平”为一维
        x=self.norm(x)
        return x
        

#cls_token + Position Embedding直接写在ViT主类（最常见）
# class PositionalEncoding(nn.Module):
#     def __init__(self, num_patches, embed_dim):
#         super().__init__()
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
#     def forward(self, x):
#         B = x.shape[0]
#         cls_tokens = self.cls_token.expand(B, -1, -1)#将单个CLS token复制扩展成batch中每个样本都有一个CLS token
#         x = torch.cat([cls_tokens, x], dim=1)
#         x = x + self.pos_embed
#         return x

# #最简单的单头注意力
# class SelfAttention(nn.Module):
#     def __init__(self,n_embd,head_size):#dim输入，dim输出
#         super().__init__()
#         self.key=nn.Linear(n_embd,head_size,bias=False)
#         self.query=nn.Linear(n_embd,head_size,bias=False)
#         self.value=nn.Linear(n_embd,head_size,bias=False)
#         self.head_size=head_size
#         self.scale=head_size**-0.5
        
#     def forward(self,x):
#         #三个可学习的参数矩阵
#         k=self.key(x)
#         q=self.query(x)
#         v=self.value(x)
#         wei=q@k.transpose(-2,-1)*self.scale
#         wei=wei.softmax(dim=-1)
#         x=wei@v
#         return x
# #多头注意力
# class MultiHeadAttention(nn.Module):
#     def __init__(self,n_embd,num_heads):
#         super().__init__()
#         self.head_size=n_embd//num_heads
#         self.num_heads=num_heads
#         self.attentions=nn.ModuleList([SelfAttention(n_embd,self.head_size) for _ in range(num_heads)])
        
#     def forward(self,x):
#         out=torch.cat([attn(x) for attn in self.attentions],dim=-1)
#         return out

#更标准的做法是用大矩阵做分割，避免重复计算attention
#还没有添加dropout和projection
class Attention(nn.Module):
    def __init__(self,n_embd,num_heads,qkv_bias=False, attn_drop_rate=0.0, proj_drop_rate=0.0):
        super().__init__()
        self.num_heads=num_heads
        self.head_size=n_embd//num_heads
        self.scale=self.head_size**-0.5

        self.qkv=nn.Linear(n_embd,n_embd*3,bias=qkv_bias)
        self.attn_drop=nn.Dropout(attn_drop_rate)
        self.proj=nn.Linear(n_embd,n_embd)
        self.proj_drop=nn.Dropout(proj_drop_rate)
        
    def forward(self,x):
        B,N,n_embd=x.shape# B=batch size,N=sequence length,C=embedding dim
        qkv=self.qkv(x).reshape(B,N,3,self.num_heads,self.head_size).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]
        attn=q@k.transpose(-2,-1)*self.scale
        attn=attn.softmax(dim=-1)
        
        attn=self.attn_drop(attn)
        
        x=torch.matmul(attn,v).transpose(1,2).reshape(B,N,n_embd)
        
        x=self.proj(x)
        x=self.proj_drop(x)
        
        return x

class MLP(nn.Module):
    def __init__(self,in_features,hidden_features,out_features,act_layer,drop_rate):
        super().__init__()
        #如果只给一样，就默认in和out一样
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        #可能会有不同的drop率，这样写便于后面修改
        drop_probs = (drop_rate, drop_rate)
        
        #两个全连接层，中间有激活函数和drop
        self.fc1=nn.Linear(in_features,hidden_features)
        self.act=act_layer()#GELU/ReLU
        self.drop1=nn.Dropout(drop_probs[0])
        self.fc2=nn.Linear(hidden_features,out_features)
        self.drop2=nn.Dropout(drop_probs[1])

    def forward(self,x):
        x=self.fc1(x)
        x=self.act(x)
        x=self.drop1(x)
        x=self.fc2(x)
        x=self.drop2(x)
        return x

#新面孔！丢层版的Dropout
class DropPath(nn.Module):
    def __init__(self,drop_prob=None):
        super(DropPath,self).__init__()
        self.drop_prob=drop_prob
    
    def drop_path(self,x,drop_prob,training):
        if drop_prob==0. or not training:
            return x
        keep_prob=1-drop_prob
        shape=(x.shape[0],)+(1,)*(x.ndim-1)#第0维（batch_size维度）保持原样；其他所有维度都变成1
        #具体表现为在每个batch里矩阵乘积另部分sample为0
        
        #mask
        #0.2概率小于1 取floor为0
        random_tensor=keep_prob+torch.rand(shape,dtype=x.dtype,device=x.device)
        random_tensor.floor_()
        
        output=x.div(keep_prob)*random_tensor
        return output
        
    def forward(self,x):
        return self.drop_path(x,self.drop_prob,self.training)
    #在训练时才丢弃，推理时不丢
    
class Block(nn.Module):
    def __init__(self,n_embd,num_heads,mlp_ratio,qkv_bias,proj_drop,attn_drop,drop_path,act_layer,norm_layer):
        '''
        Transformer的基本模块:norm+注意力+MLP+残差连接
        mlp_ratio: MLP中隐藏层维度与输入输出维度的比率
        qkv_bias: 是否在qkv线性映射中加入偏置
        proj_drop: dropout概率
        attn_drop: 注意力权重的dropout概率
        drop_path: 丢弃路径的概率
        act_layer: MLP中使用的激活函数类型
        norm_layer: 归一化层类型
        '''
        super().__init__()
        self.norm1=norm_layer(n_embd)
        self.attn=Attention(n_embd, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop_rate=attn_drop, proj_drop_rate=proj_drop)
        self.norm2=norm_layer(n_embd)
        self.mlp=MLP(in_features=n_embd, hidden_features=int(n_embd*mlp_ratio), out_features=None, act_layer=act_layer, drop_rate=drop_path)
        self.drop_path=DropPath(drop_path) if drop_path>0.0 else nn.Identity()
        
    def forward(self,x):
        x=x+self.drop_path(self.attn(self.norm1(x)))
        x=x+self.drop_path(self.mlp(self.norm2(x)))
        return x

#z最终组合！
class VisionTransformer(nn.Module):
    def __init__(self,input_shape,patch_size,channels,num_classes,num_features,depth,num_heads,mlp_ratio,qkv_bias,drop_rate,attn_drop_rate,drop_path_rate,norm_layer,act_layer):
        super().__init__()
        
        #在class中定义对象的成员变量
        self.input_shape=input_shape
        self.patch_size=patch_size
        self.channels=channels
        self.num_classes=num_classes
        self.num_features=num_features
        self.depth=depth#transformer encoder层数/block数
        self.num_heads=num_heads
        self.mlp_ratio=mlp_ratio
        self.qkv_bias=qkv_bias
        self.drop_rate=drop_rate
        self.attn_drop_rate=attn_drop_rate
        self.drop_path_rate=drop_path_rate
        self.norm_layer=norm_layer
        self.act_layer=act_layer
        
        self.features_shape=(input_shape[0]//patch_size,input_shape[1]//patch_size)#特征图尺寸
        self.num_patches=self.features_shape[0]*self.features_shape[1]
        self.patch_embed=VisionPatchEmbedding(image_size=self.input_shape,patch_size=self.patch_size,channels=self.channels,embed_dim=self.num_features,flatten=True)
        
        #引入位置编码和cls token
        self.pretrained_features_shape=[224//patch_size,224//patch_size]#预训练的特征图尺寸
        self.cls_token=nn.Parameter(torch.zeros(1,1,num_features))
        self.pos_embd=nn.Parameter(torch.zeros(1,self.num_pathces+1,num_features))
        
        self.pos_drop=nn.Dropout(drop_rate)
        self.norm=norm_layer(self.num_features)
        
        self.dpr=[x.item() for x in torch.linspace(0,drop_path_rate,depth)]#生成一个从 0 到 drop_path_rate 的线性增长序列，长度等于 Transformer 的层数（depth）
        
        self.blocks=nn.Sequential(
            *[
                Block(
                    n_embd=self.num_features,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=self.qkv_bias,
                    proj_drop=self.drop_rate,
                    attn_drop=self.attn_drop_rate,
                    drop_path=self.dpr[i],
                    act_layer=self.act_layer,
                    norm_layer=self.norm_layer
                )for i in range(self.depth)
            ]
        )
        self.head=nn.Linear(self.num_features,num_classes) if num_classes>0 else nn.Identity()
        
    def forward_features(self,x):
        x=self.patch_embed(x)#BNC
        cls_token=self.cls_token.expand(x.shape[0],-1,-1)
        x=torch.cat((cls_token,x),dim=1)
        cls_token_pos_embd=self.pos_embd[:,0,:]
        img_token_pos_embd=self.pos_embd[:,1:,:]
        
        #之前没有提到的细节处理：插值
        #1d paatch -> reshape 2d grid
        #把线性 patch 序列恢复成二维空间分布的特征图，方便做插值
        #把原来 14×14 的位置编码“放大”或“缩小”到新的分辨率（比如 24×24）
        
        # 变成[1, H, W, C]
        img_token_pos_embed = img_token_pos_embed.view(1, self.features_shape[0], self.features_shape[1], -1).permute(0, 3, 1, 2)  # [1, C, H, W]
        # 插值
        img_token_pos_embed = F.interpolate(
            img_token_pos_embed,
            size=self.features_shape,  # [H, W]
            mode='bicubic',
            align_corners=False
        )
        # 变回[1, num_patches, C]
        img_token_pos_embed = img_token_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, img_token_pos_embed.shape[1])

        pos_embd=torch.cat((cls_token_pos_embd,img_token_pos_embd),dim=1)
        x=self.pos_drop(x+pos_embd)
        x=self.blocks(x)
        x=x.self.norm(x)
           
        return x[:,0]#只取cls token对应的输出作为分类依据
    
    def forward(self,x):
        x=self.forward_features(x)
        x=self.head(x)
        return x
    
    
    