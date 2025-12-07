import torch
import torch.nn as nn
class TransformerBiLSTM(nn.Module):
    def __init__(self, input_dim, unsampling_dim, hidden_layer_sizes,hidden_dim, num_layers, num_heads, output_dim, output_size, dropout_rate=0.5):
        """
        params:
        input_dim          : 输入数据的维度
        unsampling_dim     : 上采样维度数, 把特征映射到高维， 以便被多头注意力头数 整除
        hidden_layer_sizes : bilstm 隐藏层的数目和维度
        hidden_dim          : 注意力维度
        num_layers          : Transformer编码器层数
        num_heads           : 多头注意力头数
        output_dim         : 输出维度
        output_size      : 输出序列长度，对应多步预测步长
        dropout_rate        : 随机丢弃神经元的概率
        """
        super().__init__()
        # 参数
        # 上采样操作
        self.unsampling = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=1)
        # self.unsampling = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=1)
        # self.unsampling = nn.Conv1d(input_dim, unsampling_dim, 1)

        # Transformer编码器  Transformer layers
        self.hidden_dim = hidden_dim
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(unsampling_dim, num_heads, hidden_dim, dropout=dropout_rate, batch_first=True),
            num_layers
        )

        # BiLSTM参数
        self.num_layers = len(hidden_layer_sizes)  # BiLSTM层数
        self.bilstm_layers = nn.ModuleList()  # 用于保存BiLSTM层的列表
        # 定义第一层BiLSTM
        self.bilstm_layers.append(nn.LSTM(unsampling_dim, hidden_layer_sizes[0], batch_first=True, bidirectional=True))
        # 定义后续的BiLSTM
        for i in range(1, self.num_layers):
                self.bilstm_layers.append(nn.LSTM(hidden_layer_sizes[i-1]* 2, hidden_layer_sizes[i], batch_first=True, bidirectional=True))

        # 定义线性层
        self.linear  = nn.Linear(hidden_layer_sizes[-1] * 2 , output_dim * output_size)


    def forward(self, input_seq):
        # 预处理  先进行上采样
        unsampling = self.unsampling(input_seq.permute(0,2,1))
        # Transformer 处理
        # 在PyTorch中，transformer模型的性能与batch_first参数的设置相关。
        # 当batch_first为True时，输入的形状应为(batch, sequence, feature)，这种设置在某些情况下可以提高推理性能。

        transformer_output = self.transformer(unsampling.permute(0,2,1))  #  torch.Size([64, 12, 32])

        # 送入 BiLSTM 层
        bilstm_out = transformer_output
        for bilstm in self.bilstm_layers:
            bilstm_out, _= bilstm(bilstm_out)  ## 进行一次 BiLSTM 层的前向传播  # torch.Size([64, 12, 128])
        bigru_features = (bilstm_out[:, -1, :]) # torch.Size([64, 128]  # 仅使用最后一个时间步的输出

        predict = self.linear(bigru_features)
        return predict