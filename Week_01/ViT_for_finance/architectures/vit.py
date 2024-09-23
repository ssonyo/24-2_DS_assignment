import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from timm.models.layers import trunc_normal_

class EmbeddingLayer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super(EmbeddingLayer, self).__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))  # +1 for class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        return x

    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(dim, num_heads)
        
    def forward(self, x):
        x = x.transpose(0, 1)  # (B, num_patches, dim) -> (num_patches, B, dim) for nn.MultiheadAttention
        x, _ = self.attention(x, x, x)
        return x.transpose(0, 1)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.act(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        return x

    
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., dropout=0.):
        super(Block, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))  # Attention + Residual
        x = x + self.mlp(self.norm2(x))   # MLP + Residual
        return x


#여기까지 Encoder 구현 끝!!


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., dropout=0.):
        super(VisionTransformer, self).__init__()
        self.embed_layer = EmbeddingLayer(img_size, patch_size, in_chans, embed_dim)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embed_layer(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x[:, 0])  # Only use class token output for classification

    
