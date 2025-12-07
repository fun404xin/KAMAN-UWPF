
import torch
import torch.nn as nn

from module.KAN import KANLinear


class LSTMKANModel(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes, output_size):
        super().__init__()
        """
        预测任务  params:
        input_dim        : 输入数据的维度
        hidden_layer_size: lstm 隐层的数目和维度
        output_size      : 输出的步数
        """
        # lstm层数
        self.num_layers = len(hidden_layer_sizes)
        self.lstm_layers = nn.ModuleList()  # 用于保存LSTM层的列表

        # 定义第一层LSTM
        self.lstm_layers.append(nn.LSTM(input_dim, hidden_layer_sizes[0], batch_first=True))

        # 定义后续的LSTM层
        for i in range(1, self.num_layers):
            self.lstm_layers.append(nn.LSTM(hidden_layer_sizes[i - 1], hidden_layer_sizes[i], batch_first=True))

        # kan 相关参数
        grid_size = 5
        spline_order = 3
        scale_noise = 0.1
        scale_base = 1.0
        scale_spline = 1.0
        base_activation = torch.nn.ReLU
        grid_eps = 0.02
        grid_range = [-1, 1]
        self.kan_layer = KANLinear(
            hidden_layer_sizes[-1],
            output_size,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range,
        )

    def forward(self, input_seq):
        batch_size = input_seq.size(0)
        # 数据预处理
        # 输入形状，适应网络输入[batch, seq_length, dim]
        input_seq = input_seq.view(batch_size, -1, 1)
        # 使用 permute 方法进行维度变换， 实现了维度的变换，而不改变数据的顺序
        lstm_out = input_seq
        for lstm in self.lstm_layers:
            lstm_out, _ = lstm(lstm_out)  ## 进行一次LSTM层的前向传播
        # print(lstm_out.size())  # torch.Size([64, 12, 128])
        output = self.kan_layer(lstm_out[:, -1, :])  # 仅使用最后一个时间步的输出
        return output

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):  # 计算正则化损失的方法，用于约束模型的参数，防止过拟合。
        """
        计算正则化损失。

        参数:
            regularize_activation (float): 正则化激活项的权重，默认为 1.0。
            regularize_entropy (float): 正则化熵项的权重，默认为 1.0。

        返回:
            torch.Tensor: 正则化损失。
        """
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )