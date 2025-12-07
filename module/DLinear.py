import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

# DLinear模型定义
class DLinear(nn.Module):
    def __init__(self, window_size, forecast_step, num_features):
        super(DLinear, self).__init__()
        self.window_size = window_size
        self.forecast_step = forecast_step
        self.num_features = num_features
        
        # 趋势分量线性层
        self.linear_trend = nn.Linear(window_size * num_features, forecast_step)
        
        # 季节性分量线性层
        self.linear_seasonal = nn.Linear(window_size * num_features, forecast_step)
        
        # 原始序列线性层
        self.linear_original = nn.Linear(window_size * num_features, forecast_step)
        
    def forward(self, x):
        # x shape: (batch_size, window_size, num_features)
        batch_size = x.shape[0]
        
        # 提取趋势分量（假设输入的最后两个特征是FFT分解的趋势和季节分量）
        trend = x[:, :, -2:-1]  # 倒数第二个特征是趋势分量
        seasonal = x[:, :, -1:]  # 最后一个特征是季节分量
        original = x[:, :, :-2]  # 其余是原始特征
        
        # 展平处理
        trend_flat = trend.reshape(batch_size, -1)
        seasonal_flat = seasonal.reshape(batch_size, -1)
        original_flat = original.reshape(batch_size, -1)
        
        # 分别通过线性层
        trend_out = self.linear_trend(trend_flat)
        seasonal_out = self.linear_seasonal(seasonal_flat)
        original_out = self.linear_original(original_flat)
        
        # 组合结果
        output = trend_out + seasonal_out + original_out
        
        return output.unsqueeze(-1)  # 保持输出形状为 (batch_size, forecast_step, 1)

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, patience=5):
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # 验证阶段
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_dlinear_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_dlinear_model.pth'))
    return model

# 评估函数
def evaluate_model(model, test_loader, scaler):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            predictions.append(outputs.numpy())
            actuals.append(targets.numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    
    # 反归一化
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(predictions.shape)
    actuals = scaler.inverse_transform(actuals.reshape(-1, 1)).reshape(actuals.shape)
    
    # 计算指标
    r2 = r2_score(actuals[:, :, 0], predictions[:, :, 0])
    rmse = np.sqrt(mean_squared_error(actuals[:, :, 0], predictions[:, :, 0]))
    mae = mean_absolute_error(actuals[:, :, 0], predictions[:, :, 0])
    
    print(f'R2 Score: {r2:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'MAE: {mae:.4f}')
    
    # 绘制预测结果
    plt.figure(figsize=(15, 5))
    plt.plot(actuals[0, :, 0], label='Actual')
    plt.plot(predictions[0, :, 0], label='Predicted')
    plt.xlabel('Time Step')
    plt.ylabel('Wind Power (kW)')
    plt.legend()
    plt.title('Wind Power Prediction')
    plt.show()
    
    return predictions, actuals, r2, rmse, mae

# 主程序
def main():
    # 加载数据
    train_xdata = load('train_xdata')
    train_ylabel = load('train_ylabel')
    test_xdata = load('test_xdata')
    test_ylabel = load('test_ylabel')
    scaler = load('scaler')
    
    # 打印数据形状
    print("Train data shape:", train_xdata.shape)
    print("Train label shape:", train_ylabel.shape)
    print("Test data shape:", test_xdata.shape)
    print("Test label shape:", test_ylabel.shape)
    
    # 划分训练集和验证集 (80%训练, 20%验证)
    val_size = int(0.2 * len(train_xdata))
    train_x, val_x = train_xdata[:-val_size], train_xdata[-val_size:]
    train_y, val_y = train_ylabel[:-val_size], train_ylabel[-val_size:]
    
    # 创建数据集和数据加载器
    train_dataset = TensorDataset(train_x, train_y)
    val_dataset = TensorDataset(val_x, val_y)
    test_dataset = TensorDataset(test_xdata, test_ylabel)
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 模型参数
    window_size = train_xdata.shape[1]
    forecast_step = train_ylabel.shape[1]
    num_features = train_xdata.shape[2]
    
    # 初始化模型
    model = DLinear(window_size, forecast_step, num_features)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, patience=10)
    
    # 评估模型
    predictions, actuals, r2, rmse, mae = evaluate_model(model, test_loader, scaler)
    
    # 保存模型
    torch.save(model.state_dict(), 'dlinear_wind_power.pth')

if __name__ == '__main__':
    main()