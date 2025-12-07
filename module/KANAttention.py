import torch
import torch.nn as nn
import torch.nn.functional as F


# 自定义的注意力模块
from module.KAN import KANLinear


class AttentionLayer(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(AttentionLayer, self).__init__()
        # 多头自注意力机制
        self.attention = nn.MultiheadAttention(embed_size, num_heads)

    def forward(self, x):
        # x 形状: (batch_size, seq_len, embed_size)
        # 需要转换为 (seq_len, batch_size, embed_size) 以便使用 MultiheadAttention
        x = x.transpose(0, 1)  # (batch_size, seq_len, embed_size) -> (seq_len, batch_size, embed_size)
        attn_output, _ = self.attention(x, x, x)  # 使用输入本身进行自注意力
        return attn_output.transpose(0, 1)  # 转换回 (batch_size, seq_len, embed_size)


# 修改 KAN 类以添加注意力机制
class KANWithAttention(torch.nn.Module):
    def __init__(
            self,
            layers_hidden,
            embed_size=64,  # 输入特征的嵌入维度，用于注意力机制
            num_heads=4,  # 注意力头的数量
            grid_size=2,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=0.5,
            base_activation=torch.nn.Tanh,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        """
        初始化带注意力机制的 KAN 模型。
        """
        super(KANWithAttention, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # 创建 KAN 网络的每一层
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

        # 添加注意力层
        self.attention_layer = AttentionLayer(embed_size=embed_size, num_heads=num_heads)

    def forward(self, x: torch.Tensor, update_grid=False):
        """
        前向传播函数，增加了注意力机制。
        """
        # 先经过 KAN 层
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        # 添加注意力层
        # 注意力层要求输入的形状是 (batch_size, seq_len, embed_size)
        # 这里假设 embed_size 是 x 的最后一维大小
        x_attention = self.attention_layer(x)

        return x_attention

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        计算正则化损失，保持与原 KAN 模型一致。
        """
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )

