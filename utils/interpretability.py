import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import font_manager
import seaborn as sns
import matplotlib.patches as patches 


font_path = "/root/autodl-tmp/fonts/times.ttf"
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.constrained_layout.use'] = True # 使用 constrained_layout 优化布局
# --- 1. 数据模拟准备 ---
x_norm = np.linspace(0, 1, 200) 

# === 修改核心: 模拟一个不那么平滑的 KAN 激活函数 ===
def simulated_kan_activation_rough(x):
    # 基础趋势：一个S形曲线
    base = 1 / (1 + np.exp(-15 * (x - 0.45)))
    
    # 分段逻辑：制造明显的转折点和不同区间的行为
    y = np.zeros_like(x)
    
    # 区间 1: x < 0.15 (Dead Zone, 线性抑制)
    mask1 = x < 0.15
    y[mask1] = 0.05 * x[mask1] 
    
    # 区间 2: 0.15 <= x < 0.7 (Nonlinear Spline Region, 强波动)
    mask2 = (x >= 0.15) & (x < 0.7)
    # 在基础S形上叠加一个较强的正弦波，模拟局部的 spline 拟合
    spline_effect = 0.05 * np.sin(30 * (x[mask2] - 0.15)) 
    y[mask2] = base[mask2] + spline_effect
    
    # 区间 3: x >= 0.7 (Saturation Region, 趋于平缓)
    mask3 = x >= 0.7
    # 在高位保持，并加入一点点衰减趋势
    y[mask3] = base[mask3] - 0.02 * (x[mask3]-0.7)

    # 添加一些高频随机噪声，使曲线看起来不那么完美光滑
    noise = np.random.normal(0, 0.005, len(x))
    y += noise
    
    return np.clip(y, 0, 1) # 限制在 0-1 之间

# 使用新的不平滑函数生成数据
y_activation = simulated_kan_activation_rough(x_norm)

# (右图数据保持不变)
timesteps = 72
heads = 4 
attention_weights = np.zeros((heads, timesteps))
attention_weights += np.random.normal(0.05, 0.01, (heads, timesteps))
attention_weights[0, 57:62] += 0.6 
attention_weights[1, 40:55] += 0.3
attention_weights[2, 10:20] += 0.2
attention_weights[3, :] = np.random.uniform(0.05, 0.15, timesteps)
attention_weights[3, 58] += 0.4
attention_weights = np.exp(attention_weights) / np.sum(np.exp(attention_weights), axis=1, keepdims=True)

# --- 2. 绘图 ---

fig = plt.figure(figsize=(14, 7))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2], wspace=0.05)

# === 子图 (a): KAN Activation (不平滑版本) ===
ax1 = fig.add_subplot(gs[0])

# 修改图例标签，反映新的特征
ax1.plot(x_norm, y_activation, color='#D62728', linewidth=2.5)

# 保持辅助线
ax1.axvline(x=0.15, color='gray', linestyle='--', alpha=0.6, linewidth=1)
ax1.axvline(x=0.7, color='gray', linestyle='--', alpha=0.6, linewidth=1)

# 添加区域标注 (可选，增强解释性)
# ax1.text(0.075, 0.8, 'Dead Zone', ha='center', va='center', fontsize=12, color='gray', rotation=90)
# ax1.text(0.425, 0.1, 'Nonlinear Region', ha='center', va='center', fontsize=12, color='gray')
# ax1.text(0.85, 0.8, 'Saturation', ha='center', va='center', fontsize=12, color='gray', rotation=90)

ax1.set_xlabel('Normalized Wind Speed\n(a)', fontsize=16) 
ax1.set_ylabel('Activation Output', fontsize=16)
# ax1.set_ylabel(r'Activation Output $\phi(x)$', fontsize=16) # 如果需要数学符号
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.set_xlim(0, 1)
ax1.set_ylim(-0.05, 1.05)
ax1.legend(loc='upper left', frameon=False) # 添加图例


# === 子图 (b): Attention Heatmap (保持不变) ===
gs_right = gs[1]
ax2 = fig.add_subplot(gs_right)
im = ax2.imshow(attention_weights, aspect='auto', cmap='Blues', interpolation='nearest')

ax2.set_xlabel('Time Steps\n(b)', fontsize=16) 
# ax2.set_ylabel('Attention Heads', fontsize=16) # 如果需要

ax2.set_xticks(np.arange(0, 72, 15))
ax2.set_xticklabels([f't-{timesteps-i}' for i in np.arange(0, 72, 15)])
ax2.set_yticks(np.arange(heads))
ax2.set_yticklabels([f'head {i+1}' for i in range(heads)])

# 调整 colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = plt.colorbar(im, cax=cax, orientation='vertical')
cbar.set_label('Attention Score', fontsize=16)

# 保存图片
save_path = 'img/Interpretability.jpg'
# 确保目录存在
import os
os.makedirs(os.path.dirname(save_path), exist_ok=True)

plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"图表已保存至: {save_path}")
plt.show()