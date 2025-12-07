import matplotlib
import torch
from joblib import load
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from matplotlib import font_manager
font_path = "/root/autodl-tmp/fonts/times.ttf"
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Times New Roman'  # 全局字体
plt.rcParams['font.size'] = 16   
forecast_step = 6
def model_test(model,test_loader):
    # 预测数据
    original_data = []
    pre_data = []
    with torch.no_grad():
        for data, label in test_loader:
            origin_lable = label.tolist()
            original_data += origin_lable
            model.eval()  # 将模型设置为评估模式
            data, label = data.to(device), label.to(device)
            # 预测
            data = data.view(data.size(0), -1)
            test_pred = model(data)  # 对测试集进行预测
            test_pred = test_pred.tolist()
            pre_data += test_pred
    original_data = np.array(original_data).reshape(-1,forecast_step)
    pre_data = np.array(pre_data).reshape(-1,forecast_step)
    # scaler = load('scaler')
    # original_data = scaler.inverse_transform(original_data)
    # pre_data = scaler.inverse_transform(pre_data)
    return original_data,pre_data
def model_evaluate(original_data, pre_data):
    # 模型分数
    print(len(original_data))
    print(len(pre_data))
    score = r2_score(original_data, pre_data)
    print('*' * 50)
    print(f'模型分数--R^2: {score:.4f}')

    print('*' * 50)
    # 测试集上的预测误差
    # 计算准确率
    test_mse = mean_squared_error(original_data, pre_data)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(original_data, pre_data)
    # print(f'测试数据集上的均方误差--MSE: {test_mse:.4f}')
    print(f'测试数据集上的均方根误差--RMSE: {test_rmse:.4f}')
    print(f'测试数据集上的平均绝对误差--MAE: {test_mae:.4f}')
def visualization(original_data, pre_data):
    original_data = np.array(original_data)
    pre_data = np.array(pre_data)
    print('数据 形状：')
    print(original_data.shape, pre_data.shape)

    # 反归一化处理
    # 使用相同的均值和标准差对预测结果进行反归一化处理
    # 反标准化
    scaler = load('scaler')
    # target_index = load('target_col_index')
    # y_mean = scaler.mean_[target_index]
    # y_std = scaler.scale_[target_index]
    # print(y_mean,y_std)
    original_data = scaler.inverse_transform(original_data)
    pre_data = scaler.inverse_transform(pre_data)
    # np.save("temp/Shanghai/original18.npy",original_data);
    # np.save("temp/Shanghai/pre18.npy",pre_data);
    # 多步预测 步数 根据自己的预测步数进行调整

    labels = []  # 用于存储标签的列表
    for i in range(forecast_step):
        label = f"T + {i + 1} 步预测值"
        labels.append(label)

    #第一步
    step = 5
    # 可视化结果
    plt.figure(figsize=(15, 5), dpi=300)
    plt.plot(original_data[:3000, step], label='Actual', color='c')  # 真实值
    plt.plot(pre_data[:3000, step], label=f'Predicted:T+ {step + 1} ', color='hotpink')  # 预测值
    plt.legend()
    plt.savefig(f"utils/img/T+{step+1} ahead Yalova.jpg",dpi=300)

    # 创建线性回归模型
    model = LinearRegression()

    # 将实际值作为输入，预测值作为输出进行拟合
    model.fit(np.array(original_data[:,step]).reshape(-1,1), pre_data[:,step])

    # 获取拟合的直线的预测值
    y_pred_line = model.predict(np.array(original_data[:,step]).reshape(-1,1))
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=original_data[:3000,step], y=pre_data[:3000,step], color='blue', label='prediction vs actual',s=30)
    plt.plot(original_data[:,step], y_pred_line, color='red', label='LR')
    plt.xlabel('actual',fontsize=16)
    plt.ylabel('prediction',fontsize=16)
    plt.legend()

    # 显示图形
    plt.show()

if __name__ =="__main__":
    # 模型预测
    # 模型 测试集 验证
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = load("test_loader")
    # 模型加载
    model = torch.load('best_model_kan.pt',weights_only=False)
    model = model.to(device)
    original_data,pre_data=model_test(model,test_loader)
    model_evaluate(original_data, pre_data)
    visualization(original_data, pre_data)
    