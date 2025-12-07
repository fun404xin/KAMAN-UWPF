import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import math

# 设置随机种子
np.random.seed(45409)

def calculate_rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

if __name__ == '__main__':
    # 1. 读取数据
    try:
        original_data = pd.read_csv('dataset/Yalova.csv')
    except FileNotFoundError:
        print("未找到文件，生成模拟数据...")
        x = np.linspace(0, 100, 1000)
        # 模拟数据
        original_data = pd.DataFrame({'LV ActivePower (kW)': np.sin(x)*50 + x*2 + 200})

    target_col = 'LV ActivePower (kW)'
    df = original_data[[target_col]].copy()
    df = df[45409:45509]
    df = df.reset_index(drop=True)
    # ==========================================
    # [新增步骤] Z-score 归一化
    # ==========================================
    
    scaler = StandardScaler()
    # fit_transform 返回的是 numpy array，需要重新赋值给 DataFrame
    df[target_col] = scaler.fit_transform(df[[target_col]])
    

    # 保存一份完整的归一化后的数据作为 Ground Truth (真实值)
    df_full_normalized = df.copy()

    # ==========================================
    # 2. 人工制造缺失 (在归一化后的数据上操作)
    # ==========================================
    n_samples = len(df)
    n_missing = int(n_samples * 0.05)
    missing_indices = np.random.choice(n_samples, n_missing, replace=False)
    
    df_missing = df_full_normalized.copy()
    df_missing.loc[missing_indices, target_col] = np.nan

    # ==========================================
    # 3. 填补
    # ==========================================
    
    # Linear
    df_linear = df_missing.copy()
    df_linear[target_col] = df_linear[target_col].interpolate(method='linear').ffill().bfill()

    # Quadratic
    df_quad = df_missing.copy()
    df_quad[target_col] = df_quad[target_col].interpolate(method='quadratic').ffill().bfill()

    # KNN (因为数据已经归一化，KNN 计算的距离会更准确)
    df_knn = df_missing.copy()
    knn_imputer = KNNImputer(n_neighbors=5)
    df_knn[target_col] = knn_imputer.fit_transform(df_knn[[target_col]])

    # ==========================================
    # 4. 对比误差 (RMSE)
    # ==========================================
    
    # 获取真实值 (归一化后的)
    true_values = df_full_normalized.loc[missing_indices, target_col]
    
    # 计算各方法的 RMSE (基于归一化数据的误差)
    rmse_linear = calculate_rmse(true_values, df_linear.loc[missing_indices, target_col])
    rmse_quad = calculate_rmse(true_values, df_quad.loc[missing_indices, target_col])
    rmse_knn = calculate_rmse(true_values, df_knn.loc[missing_indices, target_col])

    print("-" * 30)
    print(f"Imputation RMSE (Normalized Scale):")
    print(f"1. Linear RMSE    : {rmse_linear:.4f}")
    print(f"2. Quadratic RMSE : {rmse_quad:.4f}")
    print(f"3. KNN RMSE       : {rmse_knn:.4f}")
    
    # 补充：如果你想看还原回原始单位（kW）的误差，可以这样做：
    # 1. 还原真实值和预测值
    # true_original = scaler.inverse_transform(true_values.values.reshape(-1, 1))
    # pred_linear_original = scaler.inverse