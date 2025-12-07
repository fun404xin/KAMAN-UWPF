import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from matplotlib import font_manager
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# 设置字体
font_path = "/root/autodl-tmp/fonts/times.ttf"
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Times New Roman'  # 全局字体
plt.rcParams['font.size'] = 16

# 1. 读取数据 (假设路径正确)
print("正在读取数据文件...")
# 假设文件路径不变
df_csv = pd.read_csv('../dataset/Yalova.csv')  # 亚洛瓦数据集
df_xlsx = pd.read_excel('../dataset/Shanghai.xlsx')  # 上海数据集

# 2. 为数据集创建正确的时间索引
# 亚洛瓦数据集
start_date_y = datetime(2018, 2, 1)
date_range_y = [start_date_y + timedelta(minutes=10*i) for i in range(len(df_csv))]
df_csv['Date'] = date_range_y
df_csv.set_index('Date', inplace=True)
yalova_data_col = df_csv.iloc[:, 1] # 第二列数据

# 上海数据集
start_date_s = datetime(2021, 4, 1)
date_range_s = [start_date_s + timedelta(minutes=10*i) for i in range(len(df_xlsx))]
df_xlsx['Date'] = date_range_s
df_xlsx.set_index('Date', inplace=True)
shanghai_data_col = df_xlsx.iloc[:, -1] # 最后一列数据

# 3. 定义局部放大的一天
# 上海：2021年4月15日
local_start_s = datetime(2021, 4, 17, 0, 0)
local_end_s = local_start_s + timedelta(days=1)
shanghai_daily_data = shanghai_data_col.loc[local_start_s:local_end_s]

# 亚洛瓦：2018年3月15日
local_start_y = datetime(2018, 4, 17, 0, 0)
local_end_y = local_start_y + timedelta(days=1)
yalova_daily_data = yalova_data_col.loc[local_start_y:local_end_y]

# 4. 创建画布和主子图 (2行1列)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# --- 绘制上海数据 (主图 ax1) ---
ax1.plot(df_xlsx.index, shanghai_data_col,
         color='#FF7E79',
         linewidth=1,
         alpha=0.8)
ax1.set_xlabel('Shanghai (Full view)\n(a)', fontsize=16)
ax1.set_ylabel('Wind power (kW)', fontsize=16)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)

# --- 添加上海局部放大内嵌图 ---
axins1 = inset_axes(ax1, width="40%", height="40%", loc='upper right', borderpad=1) 
axins1.plot(shanghai_daily_data.index, shanghai_daily_data,
            color='#FF7E79', linewidth=1.5)
axins1.text(0.5, 0.9, 
            '2021-03-17', 
            transform=axins1.transAxes, 
            ha='center', 
            va='top', 
            fontsize=16, 
            color='k', 
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))
axins1.set_xlim(local_start_s, local_end_s)

# **关键修改：移除内嵌图的 X 轴和 Y 轴刻度/标签**
axins1.set_xticks([]) # 移除 X 轴刻度
# axins1.set_yticks([]) # 移除 Y 轴刻度
# 如果您想保留 X 轴但移除 Y 轴，则只保留 set_yticks([])

# 标记主图中的矩形区域并连接到内嵌图
mark_inset(ax1, axins1, loc1=2, loc2=4, fc="none", ec="k", ls='--', lw=1.5, zorder=5)

# --- 绘制亚洛瓦数据 (主图 ax2) ---
ax2.plot(df_csv.index, yalova_data_col,
         color='#2E86AB',
         linewidth=1,
         alpha=0.8)
ax2.set_xlabel('Yalova (Full view)\n(b)', fontsize=16)
ax2.set_ylabel('Wind power (kW)', fontsize=16)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0)

# --- 添加亚洛瓦局部放大内嵌图 ---
axins2 = inset_axes(ax2, width="40%", height="40%", loc='center right', borderpad=1)
axins2.plot(yalova_daily_data.index, yalova_daily_data,
            color='#2E86AB', linewidth=1.5)
axins2.text(0.5, 0.9, 
            '2018-03-17', 
            transform=axins2.transAxes, 
            ha='center', 
            va='top', 
            fontsize=16, 
            color='k', 
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))
axins2.set_xlim(local_start_y, local_end_y)

# **关键修改：移除内嵌图的 X 轴和 Y 轴刻度/标签**
axins2.set_xticks([]) # 移除 X 轴刻度
# axins2.set_yticks([]) # 移除 Y 轴刻度

# 标记主图中的矩形区域并连接到内嵌图
mark_inset(ax2, axins2, loc1=1, loc2=3, fc="none", ec="k", ls='--', lw=1.5, zorder=5)

# 5. 优化布局和保存
plt.tight_layout(pad=1.0) 

# 8. 保存图表
plt.savefig('img/timeseries_with_insets_no_ticks.jpg', dpi=300, bbox_inches='tight')