import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols

# ==========================================
# 1. 在这里填入你的数据 (Mean, Std)
# ==========================================
# 假设每组跑了 N 次 (例如 5 次)
N_RUNS = 5 

# 格式: '模型名': (均值 RMSE, 标准差 Std)
# 注意：一定要按照这个顺序填，对应 2^3 的 8 种组合
data_summary = {
    # FFT=0, KAN=0, MSM=0
    'Baseline':      (0.4751, 0.0090), 
    
    # FFT=1, KAN=0, MSM=0
    'FFT_only':      (0.3418, 0.0105),
    
    # FFT=0, KAN=1, MSM=0
    'KAN_only':      (0.3421, 0.0047),
    
    # FFT=0, KAN=0, MSM=1
    'MSM_only':      (0.3496, 0.0110),
    
    # FFT=1, KAN=1, MSM=0
    'FFT_KAN':       (0.3154, 0.0093),
    
    # FFT=1, KAN=0, MSM=1
    'FFT_MSM':       (0.2703, 0.0811),
    
    # FFT=0, KAN=1, MSM=1
    'KAN_MSM':       (0.2960, 0.00886),
    
    # FFT=1, KAN=1, MSM=1 (Full)
    'Proposed':      (0.2457, 0.0737)
}

# ==========================================
# 2. 数据重构与 ANOVA 计算 (无需修改)
# ==========================================

def generate_exact_data(mean, std, n):
    """
    生成一组数据，使其均值和标准差严格等于输入的 mean 和 std。
    这保证了 ANOVA 结果与使用原始数据完全一致。
    """
    # 1. 生成 N 个随机正态分布数
    np.random.seed(42) # 固定种子保证可重复
    raw = np.random.randn(n)
    
    # 2. 标准化 (变成 均值0, 标准差1)
    current_mean = np.mean(raw)
    current_std = np.std(raw, ddof=1) # ddof=1 for sample std
    standardized = (raw - current_mean) / current_std
    
    # 3. 缩放和平移到目标均值和标准差
    final_data = standardized * std + mean
    return final_data

# 准备 DataFrame
records = []

# 定义 8 种组合的因子状态 (FFT, KAN, MSM)
# 顺序必须与上面 data_summary 的顺序对应
configs = [
    (0, 0, 0), # Baseline
    (1, 0, 0), # FFT
    (0, 1, 0), # KAN
    (0, 0, 1), # MSM
    (1, 1, 0), # FFT+KAN
    (1, 0, 1), # FFT+MSM
    (0, 1, 1), # KAN+MSM
    (1, 1, 1)  # All
]

model_names = list(data_summary.keys())

for i, (name, (mu, sigma)) in enumerate(data_summary.items()):
    # 生成 N 个样本
    samples = generate_exact_data(mu, sigma, N_RUNS)
    
    fft_val, kan_val, msm_val = configs[i]
    
    for val in samples:
        records.append({
            'RMSE': val,
            'FFT': fft_val,
            'KAN': kan_val,
            'MSM': msm_val,
            'Group': name
        })

df = pd.DataFrame(records)

# 强制将因子转换为分类变量 (Categorical)
df['FFT'] = df['FFT'].astype('category')
df['KAN'] = df['KAN'].astype('category')
df['MSM'] = df['MSM'].astype('category')

print(f"生成的总样本数: {len(df)}")

# 3. 执行 3-Way ANOVA
# 公式包含主效应 (*) 和所有交互效应
model = ols('RMSE ~ C(FFT) * C(KAN) * C(MSM)', data=df).fit()
anova_results = sm.stats.anova_lm(model, typ=2)

# 4. 格式化输出表格
print("\n" + "="*60)
print("ANOVA 结果表 (可以直接填入论文 Table 2)")
print("="*60)
# 格式化一下显示，保留4位小数
print(anova_results.round(5))

print("\n" + "="*60)
print("关键解释:")
print("PR(>F) 即 p-value:")
print("  < 0.001 (***) : 极度显著")
print("  < 0.05  (*)   : 显著")
print("  > 0.05        : 不显著 (No significant effect)")
print("="*60)