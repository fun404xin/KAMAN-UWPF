from joblib import dump, load
import torch.utils.data as Data
import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib
import matplotlib.pyplot as plt
# 加载数据集
from module.CNN1DModel import CNN1DModel
# from module.CNNKAN import CNN1DKANModel
# from module.ChebyKAN import ChebyKAN
from module.DLinear import DLinear
# from module.GRUKAN import GRUKANModel
from module.KAN import KAN
from module.KANAttention import KANWithAttention
from module.LSTM import LSTMModel
# from module.LSTMKAN import LSTMKANModel
from module.MLP import MLP
# from module.TCNKAN import TCNKANModel
# from module.Transformer import TransformerModel
# from module.TransformerBiLSTM import TransformerBiLSTM
# from module.CompactAttentionKAN import CompactAttentionKAN

def dataloader(batch_size, workers=2):
    # 训练集
    train_set = load('train_xdata')
    train_label = load('train_ylabel')
    # 测试集
    val_set = load('val_xdata')
    val_label = load('val_ylabel')
    test_set = load('test_xdata')
    test_label = load('test_ylabel')
    # 加载数据
    train_loader = Data.DataLoader(dataset=Data.TensorDataset(train_set, train_label),
                                   batch_size=batch_size, num_workers=workers, drop_last=True)
    val_loader = Data.DataLoader(dataset=Data.TensorDataset(val_set, val_label),
                                  batch_size=batch_size, num_workers=workers, drop_last=True)
    test_loader = Data.DataLoader(dataset=Data.TensorDataset(test_set, test_label),
                                  batch_size=batch_size, num_workers=workers, drop_last=True)
    return train_loader, val_loader, test_loader
def count_parameters(model):
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    for item in params:
        print(f'{item:>6}')
    print(f'______\n{sum(params):>6}')
# -------- 分位数损失（Pinball Loss） --------
class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles  # e.g. [0.1, 0.5, 0.9]

    def forward(self, preds, target):
        """
        preds: [B, H, Q] 或 [B, H*Q] 或 [B, Q]
        target: [B, H] 或 [B, H, 1] 或 [B]
        """
        B = preds.size(0)
        Q = len(self.quantiles)

        # 如果 preds 是 [B, H*Q]，reshape 成 [B, H, Q]
        if preds.dim() == 2 and preds.size(1) % Q == 0:
            H = preds.size(1) // Q
            preds = preds.view(B, H, Q)

        # 如果 preds 是 [B, Q]，变成 [B, 1, Q]
        elif preds.dim() == 2:
            preds = preds.unsqueeze(1)  # [B, 1, Q]

        # target reshape
        if target.dim() == 1:
            target = target.unsqueeze(1)  # [B, 1]
        if target.dim() == 3 and target.size(-1) == 1:
            target = target.squeeze(-1)  # [B, H]
        # 保证 target = [B, H]
        if target.dim() == 2:
            target = target.unsqueeze(-1)  # [B, H, 1]

        # 🔑 这里现在一定是 preds: [B, H, Q], target: [B, H, 1]
        assert preds.shape[0] == target.shape[0] and preds.shape[1] == target.shape[1], \
            f"Shape mismatch: preds {preds.shape}, target {target.shape}"

        # pinball loss
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[..., i:i+1]  # [B, H, 1]
            loss_q = torch.max(q * errors, (q - 1) * errors)
            losses.append(loss_q.mean())

        return torch.stack(losses).mean()

def model_train(epochs, model, optimizer, loss_function, train_loader, val_loader, device):
    model = model.to(device)
    # 最低MSE
    minimum_mse = 1000.
    # 最佳模型
    best_model = model

    train_mse = []     # 记录在训练集上每个epoch的 MSE 指标的变化情况   平均值
    val_mse = []      # 记录在测试集上每个epoch的 MSE 指标的变化情况   平均值

     # 计算模型运行时间
    start_time = time.time()
    for epoch in range(epochs):
         # 训练
        model.train()
        quantiles = [0.25, 0.5, 0.75]
        loss_function = QuantileLoss(quantiles)
        train_mse_loss = []    #保存当前epoch的MSE loss和
        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.to(device)
            # 每次更新参数前都梯度归零和初始化
            optimizer.zero_grad()
            seq = seq.view(seq.size(0), -1)
            # 前向传播
            y_pred = model(seq)  #   torch.Size([16, 10])
            labels = labels.squeeze(-1)
            # print(y_pred.size())
            # print(labels.size())

            # 损失计算
            
            loss = loss_function(y_pred, labels)
            train_mse_loss.append(loss.item()) # 计算 MSE 损失
            # 反向传播和参数更新
            loss.backward()
            optimizer.step()
            #     break
        # break
        # 计算总损失
        train_av_mseloss = np.average(train_mse_loss) # 平均
        train_mse.append(train_av_mseloss)

        print(f'Epoch: {epoch+1:2} train_MSE-Loss: {train_av_mseloss:10.4f}')
        # 每一个epoch结束后，在验证集上验证实验结果。
        with torch.no_grad():
            # 将模型设置为评估模式
            model.eval()
            val_mse_loss = []    #保存当前epoch的MSE loss和
            for data, label in val_loader:
                data, label = data.to(device), label.to(device)
                data = data.view(data.size(0),-1)
                pre = model(data)
                # 计算损失
                label = label.squeeze(-1)
                val_loss = loss_function(pre, label)
                val_mse_loss.append(val_loss.item())

            # 计算总损失
            val_av_mseloss = np.average(val_mse_loss) # 平均
            val_mse.append(val_av_mseloss)
            print(f'Epoch: {epoch+1:2} val_MSE_Loss:{val_av_mseloss:10.4f}')
            # 早停机制
            if val_av_mseloss < minimum_mse:
                minimum_mse = val_av_mseloss
                patience_counter = 0
                torch.save(best_model, 'best_model_kan.pt')
            else:
                patience_counter += 1
                if patience_counter >= 5:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
    # 可视化
    # plt.plot(range(epochs), train_mse, color = 'b',label = 'train_MSE-loss')
    # plt.plot(range(epochs), val_mse, color = 'y',label = 'val_MSE-loss')
    # plt.legend()
    # plt.show()   #显示 lable
    print(f'min_MSE: {minimum_mse}')
if __name__ =="__main__":
    # 参数与配置
    matplotlib.rc("font", family='Microsoft YaHei')
    torch.manual_seed(100)  # 设置随机种子，以使实验结果具有可重复性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    # 加载数据
    train_loader, val_loader, test_loader = dataloader(batch_size)
    dump(test_loader,"test_loader")
    print(len(train_loader))
    print(len(test_loader))
     # 定义模型参数
    input_size = 18*6
    # 输入为 12 步
    # 定义 一个三层的KAN 网络
    hidden_dim1 = 128  # 第一层隐藏层 神经元 64个
    hidden_dim2 = 64   # 第二层隐藏层 神经元 32个
    hidden_dim3 = 32
    output_size = 6# 多步预测输出
    # Define model
    model = KANWithAttention([input_size, 32, 64, output_size*len([0.25, 0.5, 0.75])]) # 输入特征为12，输出层有1个神经元，用于单特征预测
    loss_function = nn.MSELoss()  # loss
    learn_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), learn_rate)  # 优化器
    count_parameters(model)
    #  模型训练
    epochs = 50
    model_train(epochs, model, optimizer, loss_function, train_loader, val_loader, device)