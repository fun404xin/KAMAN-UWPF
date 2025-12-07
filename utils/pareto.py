import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib import font_manager

# --- 字体设置 ---
try:
    font_path = "/root/autodl-tmp/fonts/times.ttf"
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = 'Times New Roman'
    else:
        plt.rcParams['font.family'] = 'Times New Roman'
except:
    plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['font.size'] = 16

# --- 1. 数据准备 ---
data = {
    'Model': ['MLP', 'KAN', 'C-KAN', 'LSTM-KAN', 'KAMAN'],
    'CA': [0.2190, 0.2269, 0.2062, 0.2009, 0.1667],
    'Latency_ms': [157.06, 242.47, 224.01, 280.92, 254.46],
}
df = pd.DataFrame(data)

# --- 定义 Pareto Frontier (修正为匹配 df 的数据) ---
# 逻辑：连接那些在给定延迟下 CA 最低的点
# 路径：MLP (最快) -> C-KAN (中等) -> KAMAN (最准)
pareto_frontier = pd.DataFrame({
    'Model': ['MLP', 'C-KAN', 'KAMAN'], 
    'Latency_ms': [157.06, 224.01, 254.46],
    'CA': [0.2190, 0.2062, 0.1667] 
}).sort_values('Latency_ms')

# --- 2. 绘图 ---
fig, ax = plt.subplots(figsize=(10, 7))

# 2.1 绘制所有模型点
# 修改点：hue='Model', style='Model' 实现不同颜色和形状
sns.scatterplot(
    x='Latency_ms', 
    y='CA', 
    data=df, 
    hue='Model',   # 按模型区分颜色
    style='Model', # 按模型区分形状
    s=200,         # 稍微加大点的大小
    alpha=0.9,
    palette='bright', # 使用鲜艳一点的颜色
    ax=ax,
    legend=False   # 如果不需要图例（因为点上方有名字），可以关闭。若需要保留图例改为 True
)

# 2.2 绘制 Pareto Frontier
ax.plot(
    pareto_frontier['Latency_ms'], 
    pareto_frontier['CA'], 
    color='red', 
    linestyle='--', 
    linewidth=2, 
    marker='o',
    markersize=0, 
    label='Pareto Frontier',
    zorder=1 # 确保线在点下方
)

# 2.3 标注每个点 (修改为在点上方)
y_offset = 0.003 # Y轴偏移量，根据CA的数据范围微调
for i, row in df.iterrows():
    # 判断是否加粗（在Pareto线上则加粗）
    is_bold = 'bold' if row['Model'] in pareto_frontier['Model'].values else 'normal'
    # 颜色：如果是 KAMAN，可以用红色突出，或者统一用黑色
    text_color = 'red' if row['Model'] == 'KAMAN' else 'black'
    
    ax.text(
        row['Latency_ms'],        # X轴：居中
        row['CA'] + y_offset,     # Y轴：向上偏移
        row['Model'], 
        fontsize=16, 
        ha='center',              # 水平居中
        va='bottom',              # 垂直底部对齐（即文字在坐标点上方）
        fontweight=is_bold,
        color=text_color
    )

# 2.4 理想区域标注
# 获取当前坐标轴范围来定位
x_lims = ax.get_xlim()
y_lims = ax.get_ylim()

# 定位在左下角附近
x_target = x_lims[0] + (x_lims[1] - x_lims[0]) * 0.05
y_target = y_lims[0] + (y_lims[1] - y_lims[0]) * 0.05

ax.annotate('Ideal Region',
            xy=(150, 0.16), # 指向的大致坐标 (根据数据范围预估)
            xytext=(170, 0.175), # 文字位置
            fontsize=16,
            color='green',
            fontweight='bold',
            ha='center',
            arrowprops=dict(facecolor='green', shrink=0.05, width=1, headwidth=8))

# 2.5 调整和美化
ax.set_xlabel('Inference Time (ms)', fontsize=16)
ax.set_ylabel('CA', fontsize=16) # 修正Y轴标签更明确
# ax.set_title('Accuracy-Latency Pareto Chart', fontsize=16, fontweight='bold')
ax.grid(True, linestyle=':', alpha=0.6)

# 适当扩展Y轴上限，给上面的文字留出空间
ax.set_ylim(df['CA'].min() - 0.01, df['CA'].max() + 0.015)
ax.set_xlim(140, 300) # 根据数据手动微调X轴范围，保证美观

plt.tight_layout()
plt.legend()
# --- 3. 保存图表 ---
save_path = 'img/pareto.jpg'
os.makedirs(os.path.dirname(save_path), exist_ok=True) 
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"图表已保存至: {save_path}")