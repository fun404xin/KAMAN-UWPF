import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseMultiheadAttention(nn.Module):
    def __init__(self, embed_size, num_heads=4, reduction_ratio=2):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        # 确保维度能被num_heads整除
        assert self.head_dim * num_heads == embed_size, "embed_size must be divisible by num_heads"
        
        # 共享投影
        self.qkv_proj = nn.Linear(embed_size, 3 * embed_size // reduction_ratio)
        self.out_proj = nn.Linear(embed_size // reduction_ratio, embed_size)

    def forward(self, x):
        # x: [batch, seq_len, embed_size]
        batch_size, seq_len, _ = x.shape
        
        # 投影到QKV
        qkv = self.qkv_proj(x)  # [batch, seq_len, 3*reduced_dim]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, -1)
        q, k, v = qkv.unbind(2)  # 每个形状[batch, seq_len, num_heads, head_dim]
        
        # 注意力计算
        attn = torch.softmax(
            torch.einsum('bqhd,bkhd->bhqk', q, k) / (self.head_dim ** 0.5),
            dim=-1
        )
        out = torch.einsum('bhqk,bkhd->bqhd', attn, v)
        out = out.reshape(batch_size, seq_len, -1)
        
        return self.out_proj(out)

class CompactAttentionKAN(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=4, hidden_dim=64):
        super().__init__()
        # 确保hidden_dim能被num_heads整除
        hidden_dim = (hidden_dim // num_heads) * num_heads
        
        # 输入适配层
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 注意力层
        self.attention = SparseMultiheadAttention(hidden_dim, num_heads)
        
        # KAN层
        self.kan_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 输出层
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [batch, features]
        x = self.input_proj(x)  # [batch, hidden_dim]
        
        # 添加序列维度 [batch, 1, hidden_dim]
        x = x.unsqueeze(1)
        x = self.attention(x)
        x = x.squeeze(1)  # [batch, hidden_dim]
        
        x = self.kan_layer(x)
        return self.output_proj(x)