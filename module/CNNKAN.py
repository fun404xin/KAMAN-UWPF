import torch.nn as nn
import torch
from module.KAN import KANLinear


class CNN1DKANModel(nn.Module):
    def __init__(self, input_channels, conv_archs, output_size):
        super(CNN1DKANModel, self).__init__()
        """
        预测任务  params:
        input_channels      : 输入维度数， 特征数量
        conv_archs          : 1dvgg 卷积池化网络结构
        output_size         : 输出的步数
        """
        # self.batch_size = batch_size
        # CNN参数
        self.conv_archs = conv_archs  # 网络结构
        self.input_channels = input_channels  # 输入通道数
        self.features = self.make_layers()
        # 自适应平局池化              在这里做了改进，摒弃了 大参数量的三层全连接层，改为自适应平均池化来替代
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # 定义全连接层  替换为 KAN层
        # self.classifier = nn.Linear(conv_archs[-1][-1], output_dim)

        # kan 相关参数
        grid_size = 5
        spline_order = 3
        scale_noise = 0.1
        scale_base = 1.0
        scale_spline = 1.0
        base_activation = torch.nn.SiLU
        grid_eps = 0.02
        grid_range = [-1, 1]
        self.kan_layer = KANLinear(
            conv_archs[-1][-1],
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

    # CNN1d卷积池化结构
    def make_layers(self):
        layers = []
        for (num_convs, out_channels) in self.conv_archs:
            for _ in range(num_convs):
                layers.append(nn.Conv1d(self.input_channels, out_channels, kernel_size=3, padding=1))
                layers.append(nn.ReLU(inplace=True))
                self.input_channels = out_channels
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, input_seq):  # torch.Size([64, 24])
        batch_size = input_seq.size(0)
        # 改变输入形状，适应网络输入[batch,dim, seq_length]
        input_seq = input_seq.view(batch_size, 2, 18)
        # 送入CNN网络模型
        features = self.features(input_seq)  # torch.Size([64, 64, 6])
        # 自适应平均池化
        x = self.avgpool(features)  # ttorch.Size([64, 64, 1])
        # 平铺
        flat_tensor = x.view(batch_size, -1)  # torch.Size([64, 64])
        output = self.kan_layer(flat_tensor)  # torch.Size([64, 1])
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

