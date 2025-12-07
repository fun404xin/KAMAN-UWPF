import torch.nn as nn
import torch
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer
# 定义 TransformerModel 模型
class TransformerModel(nn.Module):
    def __init__(self, batch_size, input_dim, unsampling_dim, hidden_dim, num_layers, num_heads, output_dim,
                 output_size, dropout_rate=0.5):
        """
        预测任务  params:
        batch_size       : 批次量大小
        input_dim        : 输入数据的维度
        unsampling_dim   : 上采样维度
        hidden_dim       : Transformer隐层维度
        num_layers       : 堆叠 编码器 数量
        num_heads        : 多头注意力 头数
        output_dim       : 输出维度数
        output_size      : 输出序列长度，对应多步预测步长
        dropout_rate     : 随机丢弃神经元的概率
        """
        super().__init__()
        # 批次量大小
        self.batch_size = batch_size

        # 上采样操作
        self.unsampling = nn.Conv1d(input_dim, unsampling_dim, 1)

        # Transformer layers
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(unsampling_dim, num_heads, hidden_dim, dropout=dropout_rate, batch_first=True),
            num_layers
        )
        # 序列平均池化操作
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # 定义线性层
        self.linear = nn.Linear(unsampling_dim, output_dim * output_size)

    def forward(self, input_seq):
        # Transformer 处理
        # Transformer 处理
        # 预处理  先进行上采样
        unsampling = self.unsampling(input_seq.permute(0, 2, 1))
        # 在PyTorch中，transformer模型的性能与batch_first参数的设置相关。
        # 当batch_first为True时，输入的形状应为(batch, sequence, dim)，这种设置在某些情况下可以提高推理性能。
        # 送入 transformer层
        transformer_output = self.transformer(unsampling.permute(0, 2, 1))  # torch.Size([64, 12, 32])
        # 平均池化
        x = self.avgpool(transformer_output.permute(0, 2, 1))  # ttorch.Size([64, 32, 1])
        # 平铺
        flat_tensor = x.view(self.batch_size, -1)  # torch.Size([64, 32])
        predict = self.linear(flat_tensor)  # torch.Size([64, 3])
        return predict