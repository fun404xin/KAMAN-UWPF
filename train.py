from joblib import dump, load
import torch.utils.data as Data
import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib
import matplotlib.pyplot as plt
import optuna
# 加载数据集
from module.KAN import KAN
from module.KANAttention import KANWithAttention
from module.MLPAttention import MLPWithAttention
from module.HybridKAN import HybridKAN
from module.MLP import MLP
from module.PatchTST import PatchTST
from module.TSMixer import TSMixer
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
def time_freq_loss(y_pred, labels, loss_function,alpha):
    time_loss = loss_function(y_pred, labels)
    pred_fft = torch.fft.rfft(y_pred, dim=1)
    true_fft = torch.fft.rfft(labels, dim=1)
    freq_loss = (pred_fft - true_fft).abs().mean()
    loss = alpha * time_loss + (1-alpha) * freq_loss
    return loss
def model_train(epochs, model, optimizer, loss_function, train_loader, val_loader, device):
    model = model.to(device)
    # 最低MSE
    minimum_mse = 1000.
    # 最佳模型
    best_model = model
    # train_mse = []     # 记录在训练集上每个epoch的 MSE 指标的变化情况   平均值
    # val_mse = []      # 记录在测试集上每个epoch的 MSE 指标的变化情况   平均值

     # 计算模型运行时间
    start_time = time.time()
    for epoch in range(epochs):
         # 训练
        model.train()
        total_train_loss = []    #保存当前epoch的MSE loss和
        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.to(device)
            # 每次更新参数前都梯度归零和初始化
            optimizer.zero_grad()
            seq = seq.view(seq.size(0), -1)
            # 前向传播
            y_pred = model(seq) 
            labels = labels.squeeze(-1)
            # print(y_pred.size())
            # print(labels.size())
            # 损失计算
            # 2. 正则化损失（L1 + 熵）
            # loss_reg = model.regularization_loss(
            # regularize_activation=1,  # 你可以调节这个系数
            # regularize_entropy=1)
            # 3. 总损失 = 主损失 + 正则化项
            # train_loss = time_freq_loss(y_pred, labels, loss_function, 0.7)
            train_loss = loss_function(y_pred, labels)
            total_train_loss.append(train_loss.item()) # 计算 MSE 损失
            # 反向传播和参数更新
            train_loss.backward()
            optimizer.step()
            #     break
        # break
        # 计算总损失
        train_av_loss = np.average(total_train_loss) # 平均
        # train_mse.append(train_av_mseloss)

        print(f'Epoch: {epoch+1:2} train_Loss: {train_av_loss:10.4f}')
        # 每一个epoch结束后，在验证集上验证实验结果。
        with torch.no_grad():
            # 将模型设置为评估模式
            model.eval()
            total_val_loss = []    #保存当前epoch的MSE loss和
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                data = data.view(data.size(0),-1)
                pre = model(data)
                # 计算损失
                labels = labels.squeeze(-1)
                val_loss = loss_function(pre, labels)
                total_val_loss.append(val_loss.item())

            # 计算总损失
            val_av_loss = np.average(total_val_loss) # 平均
            # val_mse.append(val_av_mseloss)
            print(f'Epoch: {epoch+1:2} val_Loss:{val_av_loss:10.4f}')
            # 早停机制
            if val_av_loss < minimum_mse:
                minimum_mse = val_av_loss
                best_model = model
                patience_counter = 0
                torch.save(best_model, 'best_model_kan.pt')
            else:
                patience_counter += 1
                if patience_counter >= 5:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
            # 如果当前模型的 MSE 低于于之前的最佳准确率，则更新最佳模型
            #保存当前最优模型参数

    # 保存最后的参数
    # torch.save(model, 'final_model_kan.pt')
    # 保存最好的参数
    # torch.save(best_model, 'best_model_kan.pt')
    # print(f'\nDuration: {time.time() - start_time:.0f} seconds')

    # 可视化
    plt.plot(range(len(total_train_loss)), total_train_loss, color = 'b',label = 'train_MSE-loss')
    plt.plot(range(len(total_val_loss)), total_val_loss, color = 'y',label = 'val_MSE-loss')
    plt.legend()
    plt.show()   #显示 lable
    print(f'min_MSE: {minimum_mse}')
    return minimum_mse
if __name__ =="__main__":
    # 参数与配置
    matplotlib.rc("font", family='Microsoft YaHei')
    torch.manual_seed(100)  # 设置随机种子，以使实验结果具有可重复性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
   
    train_loader, val_loader, test_loader = dataloader(batch_size)
    dump(test_loader,"test_loader")
    print(len(train_loader))
    print(len(val_loader))
     # 定义模型参数
    input_size = 18*6
    # 输入为 12 步
    # 定义 一个三层的KAN 网络
    hidden_dim1 = 128  
    hidden_dim2 = 64   
    hidden_dim3 = 32
    output_size = 6# 多步预测输出
    # Define model
    model = KAN([input_size, 32, 64, output_size]) 

 
    # 定义损失函数和优化函数
    loss_function = nn.MSELoss()
    # loss_function = nn.SmoothL1Loss(beta=1.0)  # loss
    learn_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), learn_rate)  # 优化器
    count_parameters(model)
    #  模型训练
    epochs = 50
    model_train(epochs, model, optimizer, loss_function, train_loader, val_loader, device)