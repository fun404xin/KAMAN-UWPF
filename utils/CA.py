import pandas as pd
import numpy as np
import os

# 定义CA计算函数 (与之前相同)
def safe_float(x):
    return float(str(x))

def calculate_ca(row):
    """
    计算 Combined Accuracy (CA)
    CA = 0.33 * (RMSE + MAE + (1 - R^2))
    """
    R2 = safe_float(row['R^2'])
    RMSE = safe_float(row['RMSE'])
    MAE = safe_float(row['MAE'])
    R2_term = 1 - R2 if R2 <= 1 else 0.0
    return 0.33 * (RMSE + MAE + R2_term)

# --- 1. 表格数据输入 (与之前相同) ---

# 表格 1 数据 (对应 image_18635c.png)
data_table_2 = {
    'Models': ['LSTM', 'CNN', 'PatchTST', 'MLP', 'DLinear', 'TSMixer', 'LSTM-KAN', 'C-KAN', 'Transformer-BiLSTM', 'KAMAN'],
    'R^2_T1': [0.9673, 0.9569, 0.9636, 0.9606, 0.9712, 0.9710, 0.9682, 0.9671, 0.9690, 0.9758],
    'RMSE_T1': [0.1836, 0.2169, 0.1854, 0.2075, 0.1734, 0.1741, 0.1807, 0.1856, 0.1788, 0.1591],
    'MAE_T1': [0.1173, 0.1540, 0.1363, 0.1468, 0.1023, 0.1068, 0.1111, 0.1238, 0.1101, 0.1143],
    'R^2_T6': [0.9059, 0.9003, 0.9181, 0.9013, 0.9135, 0.9141, 0.9050, 0.9053, 0.9031, 0.9362],
    'RMSE_T6': [0.3112, 0.3301, 0.2889, 0.3283, 0.3007, 0.2997, 0.3128, 0.3148, 0.3159, 0.2585],
    'MAE_T6': [0.1995, 0.2296, 0.2123, 0.2365, 0.1872, 0.1874, 0.2011, 0.2154, 0.2044, 0.1829]
}
df2 = pd.DataFrame(data_table_2)

# 表格 2 数据 (对应 image_1863bc.png)
data_table_1 = {
    'Models': ['LSTM', 'CNN', 'PatchTST', 'MLP', 'DLinear', 'TSMixer', 'LSTM-KAN', 'C-KAN', 'Transformer-BiLSTM', 'KAMAN'],
    'R^2_T1': [0.9705, 0.9698, 0.9600, 0.9140, 0.9735, 0.9731, 0.9725, 0.9734, 0.9694, 0.9823],
    'RMSE_T1': [0.1561, 0.1580, 0.1623, 0.2668, 0.1482, 0.1493, 0.1512, 0.1489, 0.1593, 0.1212],
    'MAE_T1': [0.0906, 0.0907, 0.1104, 0.1551, 0.0808, 0.0822, 0.0898, 0.0796, 0.0915, 0.0744],
    'R^2_T6': [0.9089, 0.9076, 0.9091, 0.8521, 0.9146, 0.9107, 0.9164, 0.9172, 0.9106, 0.9370],
    'RMSE_T6': [0.2744, 0.2764, 0.2653, 0.3498, 0.2659, 0.2720, 0.2637, 0.2624, 0.2719, 0.2294],
    'MAE_T6': [0.1675, 0.1696, 0.1846, 0.2110, 0.1660, 0.1707, 0.1526, 0.1529, 0.1580, 0.1541]
}
df1 = pd.DataFrame(data_table_1)

# --- 2. CA 值计算和保存 ---

def calculate_and_save_ca(df, file_name):
    # T+1-ahead 计算
    df['CA_T1'] = df.apply(
        lambda row: calculate_ca({'R^2': row['R^2_T1'], 'RMSE': row['RMSE_T1'], 'MAE': row['MAE_T1']}), axis=1).round(4)
    
    # T+6-ahead 计算
    df['CA_T6'] = df.apply(
        lambda row: calculate_ca({'R^2': row['R^2_T6'], 'RMSE': row['RMSE_T6'], 'MAE': row['MAE_T6']}), axis=1).round(4)
    
    # 提取结果列
    results = df[['Models', 'CA_T1', 'CA_T6']].copy()
    
    # 保存为 CSV
    results.to_csv(file_name, index=False)
    print(f"CA 结果已成功保存到文件: {file_name}")
    return results

# 执行计算和保存
results1 = calculate_and_save_ca(df1, 'ca_results_table1.csv')
results2 = calculate_and_save_ca(df2, 'ca_results_table2.csv')