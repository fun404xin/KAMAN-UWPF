import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
labels = ['T+1', 'T+6', 'T+12', 'T+18']
Shanghai_rmse = [0.1780, 0.2294, 0.3223, 0.3954]
Turkey_rmse = [0.1591, 0.2585, 0.4531, 0.4992]
Shanghai_mae = [0.1318, 0.1541, 0.1992, 0.2143]
Turkey_mae = [0.1143, 0.1872, 0.3004, 0.3203]


# 设置字体
font_path = "/root/autodl-tmp/fonts/times.ttf"
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
colors = ['#1f77b4', '#ff7f0e']
plt.rcParams['axes.unicode_minus'] = False
bar_width = 0.4
x = np.arange(len(labels))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
plt.subplots_adjust(wspace=0.25)

# 手动设置背景色（浅灰）
for ax in [ax1, ax2]:
    ax.set_facecolor('#EBF0F7')  # seaborn浅灰背景类似颜色
    ax.grid(True, axis='x', linestyle='-', color='white', alpha=1)
    ax.grid(True, axis='y', linestyle='--', color='white', alpha=0.8)
    ax.set_axisbelow(True)  # 网格在数据之下（推荐）

# RMSE子图
bars1 = ax1.bar(x - bar_width/2, Shanghai_rmse, bar_width, label='Shanghai', color=colors[0])
bars2 = ax1.bar(x + bar_width/2, Turkey_rmse, bar_width, label='Yalova', color=colors[1])
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_ylim(0.1, 0.6)
ax1.set_yticks(np.arange(0.1, 0.6+ 0.0001, 0.1))
ax1.set_ylabel('RMSE',  labelpad=10)
ax1.tick_params(axis='y')
ax1.grid(True)
ax1.set_xlabel('Time Steps\n(a)',fontsize=16)
# MAE子图
bars3 = ax2.bar(x - bar_width/2, Shanghai_mae, bar_width, label='Shanghai', color=colors[0])
bars4 = ax2.bar(x + bar_width/2, Turkey_mae, bar_width, label='Yalova', color=colors[1])
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.set_ylim(0.1, 0.35)
ax2.set_ylabel('MAE',  labelpad=10)
ax2.tick_params(axis='y')
ax2.set_xlabel('Time Steps\n(b)',fontsize=16)
# def add_value_labels(bars, ax, offset=0.01):
#     for bar in bars:
#         height = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width()/2., height + offset,
#                 f'{height:.4f}', ha='center', va='bottom',
#                 fontsize=font_small, rotation=0)
#
# add_value_labels(bars1, ax1)
# add_value_labels(bars2, ax1)
# add_value_labels(bars3, ax2)
# add_value_labels(bars4, ax2)

for ax in [ax1, ax2]:
    # ax.set_xlabel('Time Horizons',  labelpad=10)
    ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig('img/performance_comparison.jpg', dpi=300, bbox_inches='tight')
print(plt.rcParams['font.family'])
