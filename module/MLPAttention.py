import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPWithAttention(nn.Module):
    def __init__(self, dims):
        """
        dims: [input_dim, hidden1, hidden2, output_dim]
        """
        super().__init__()
        self.input_dim = dims[0]
        self.hidden_dim = dims[-2]  # 最后一个隐藏层维度

        # MLP 主干
        self.fc1 = nn.Linear(dims[0], dims[1])
        self.fc2 = nn.Linear(dims[1], dims[2])
        self.relu = nn.ReLU()

        # Single-head Self Attention
        self.query = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key   = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value = nn.Linear(self.hidden_dim, self.hidden_dim)

        # 输出层
        self.fc_out = nn.Linear(self.hidden_dim, dims[-1])

    def forward(self, x):
        # --- MLP feature extraction ---
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))  # 得到隐藏特征 [B, hidden_dim]

        # --- Self Attention (KAN风格) ---
        Q = self.query(x)  # [B, H]
        K = self.key(x)    # [B, H]
        V = self.value(x)  # [B, H]

        # 单向注意力分数（对特征维度的 pairwise 相关性）
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.hidden_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        x = torch.matmul(attn_weights, V)  # [B, H]

        # 残差增强稳定性
        x = x + V  

        # --- 输出层 ---
        out = self.fc_out(x)
        return out
