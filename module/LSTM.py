import torch
import torch.nn as nn
# 定义 LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_sizes):
        """
        params:
        input_dim          : 输入数据的维度
        output_dim         : 输出维度
        hidden_layer_sizes : lstm 隐层的数目和维度
        """
        super().__init__()
        # 参数
        self.output_dim = output_dim

        # LSTM参数
        self.num_layers = len(hidden_layer_sizes)  # lstm层数
        self.lstm_layers = nn.ModuleList()  # 用于保存LSTM层的列表
        # 定义第一层LSTM
        self.lstm_layers.append(nn.LSTM(input_dim, hidden_layer_sizes[0], batch_first=True))
        # 定义后续的LSTM层
        for i in range(1, self.num_layers):
                self.lstm_layers.append(nn.LSTM(hidden_layer_sizes[i-1], hidden_layer_sizes[i], batch_first=True))

        # 定义线性层
        self.linear  = nn.Linear(hidden_layer_sizes[-1], output_dim)

    def forward(self, input_seq):
        # 送入 LSTM 层
        #改变输入形状，lstm 适应网络输入[batch, seq_length, H_in]
        lstm_out = input_seq
        for lstm in self.lstm_layers:
            lstm_out, _ = lstm(lstm_out)  ## 进行一次LSTM层的前向传播

        predict = self.linear(lstm_out[:, -1, :]) # torch.Size([256, 1]  # 仅使用最后一个时间步的输出
        return predict