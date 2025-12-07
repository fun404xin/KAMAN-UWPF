import torch.nn as nn
import torch
# 定义 CNN1DModel 模型
class CNN1DModel(nn.Module):
    def __init__(self, batch_size, input_dim, conv_archs, output_dim, output_size):
        """
        预测任务  params:
        batch_size       : 批次量大小
        input_dim        : 输入数据的维度
        conv_archs       : CNN 网络结构，层数和每层通道数
        output_dim       : 输出维度
        output_size      : 输出序列长度，对应多步预测步长
        """
        super().__init__()
        # 批次量大小
        self.batch_size = batch_size
        # CNN参数
        self.conv_arch = conv_archs  # 网络结构
        self.input_channels = input_dim  # 输入通道数
        self.features = self.make_layers()

        # 平局池化
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # 定义线性层
        self.linear = nn.Linear(conv_archs[-1][-1], output_dim * output_size)

    # CNN卷积池化结构
    def make_layers(self):
        layers = []
        for (num_convs, out_channels) in self.conv_arch:
            for _ in range(num_convs):
                layers.append(nn.Conv1d(self.input_channels, out_channels, kernel_size=3, padding=1))
                layers.append(nn.ReLU(inplace=True))
                self.input_channels = out_channels
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, input_seq):
        # 改变输入形状，适应网络输入[batch , dim, seq_length]
        input_seq = input_seq.permute(0, 2, 1)
        # 送入 CNN 网络
        cnn_features = self.features(input_seq)  # torch.Size([64, 64, 6])
        # 平均池化
        x = self.avgpool(cnn_features)  # ttorch.Size([64, 64, 1])
        # 平铺
        flat_tensor = x.view(self.batch_size, -1)  # torch.Size([64, 64])
        predict = self.linear(flat_tensor)  # torch.Size([64, 3])
        return predict