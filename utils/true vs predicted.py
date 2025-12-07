import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib import font_manager
# 设置字体
font_path = "/root/autodl-tmp/fonts/times.ttf"
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
# 设置预测步长
forecast_steps = [1, 6]
data_dir1 = "temp/Shanghai"  # 替换为你的.npy文件所在目录
data_dir2 = "temp/Turkey"  # 替换为你的.npy文件所在目录

plot_len = 3000  # 每张图展示前50个样本

# 子图布局（可调整）
rows, cols = 2, 2
fig, axes = plt.subplots(rows, cols, figsize=(14, 8))
# fig.text(0.5, 0.52, 'Shanghai Wind Farm', ha='center')
# fig.text(0.5, 0.04, 'Yalova Wind Farm', ha='center')
for i, step in enumerate(forecast_steps):
    # 加载数据
    true_path = os.path.join(data_dir1, f"original{step}.npy")
    pred_path = os.path.join(data_dir1, f"pre{step}.npy")
    true_data = np.load(true_path)  # shape: (samples, step)
    pred_data = np.load(pred_path)

    # 获取第 step-1 个时间点（如 step=3，取 t+3 的预测，即索引为 2）
    true_seq = true_data[:plot_len, step - 1]
    pred_seq = pred_data[:plot_len, step - 1]
    pred_seq[pred_seq<-20] = 0
    row = i // cols
    col = i % cols
    ax = axes[row, i]
    ax.set_ylim(0, 2500)
    ax.plot(true_seq, label='True', color='black')
    ax.plot(pred_seq, label='Pred', color='orange')
    if (i == 0):
        ax.set_xlabel(f"T+{step}-ahead\n(a)")
    else:
        ax.set_xlabel(f"T+{step}-ahead\n(b)")
    ax.set_ylabel("Wind power (kW)")
    ax.legend(loc='upper left')
    # 添加局部细节图
    # 选择要放大的区域（例如中间部分）
    zoom_start = 10 * len(true_seq) // 20
    zoom_end = 11 * len(true_seq) // 20
    zoom_range = slice(zoom_start, zoom_end)

    # 创建内嵌图
    ax_inset = inset_axes(ax, width="40%", height="30%", loc='upper right')

    # 绘制局部细节
    ax_inset.plot(true_seq[zoom_range], color='black', linewidth=0.8)
    ax_inset.plot(pred_seq[zoom_range], color='orange', linewidth=0.8)

    # 设置内嵌图的样式
    ax_inset.set_xticks([])  # 隐藏x轴刻度
    ax_inset.set_yticks([])  # 隐藏y轴刻度
    ax_inset.spines['top'].set_visible(True)
    ax_inset.spines['right'].set_visible(True)
    ax_inset.spines['bottom'].set_visible(True)
    ax_inset.spines['left'].set_visible(True)

    # 在内嵌图中添加矩形框指示放大区域
    rect = plt.Rectangle((zoom_start, min(true_seq.min(), pred_seq.min())),
                         zoom_end - zoom_start,
                         max(true_seq.max(), pred_seq.max()) - min(true_seq.min(), pred_seq.min()),
                         fill=False, color='red', linestyle='--', alpha=0.7, linewidth=0.8)
    # 添加单个指示箭头
    ax.add_patch(rect)
    con = ConnectionPatch(
        xyA=(0, 0), coordsA=ax_inset.transAxes,   # 内嵌图左下角 (0,0)
        xyB=(zoom_end, max(true_seq.max(), pred_seq.max())), coordsB=ax.transData,  # 主图矩形右上角
        axesA=ax_inset, axesB=ax,
        color="red", linewidth=0.8, linestyle="-", arrowstyle="<-"
    )
    ax_inset.add_artist(con)

for i, step in enumerate(forecast_steps):
    # 加载数据
    true_path = os.path.join(data_dir2, f"original{step}.npy")
    pred_path = os.path.join(data_dir2, f"pre{step}.npy")
    true_data = np.load(true_path)  # shape: (samples, step)
    pred_data = np.load(pred_path)

    # 获取第 step-1 个时间点（如 step=3，取 t+3 的预测，即索引为 2）
    true_seq = true_data[:plot_len, step - 1]
    pred_seq = pred_data[:plot_len, step - 1]
    pred_seq[pred_seq<-20] = 0
    row = i // cols +1
    col = i % cols
    ax = axes[row, i]
    ax.set_ylim(0, 5700)
    ax.set_yticks(np.arange(0, 5700+ 1, 1000))
    ax.plot(true_seq, label='True', color='black')
    ax.plot(pred_seq, label='Pred', color='orange')
    if(i ==0):
        ax.set_xlabel(f"T+{step}-ahead\n(c)")
    else:
        ax.set_xlabel(f"(d)T+{step}-ahead\n(d)")
    ax.set_ylabel("Wind power (kW)")
    ax.legend(loc='upper left')
    # 添加局部细节图
    # 选择要放大的区域（例如中间部分）
    zoom_start = 10 * len(true_seq) // 20
    zoom_end = 11 * len(true_seq) // 20
    zoom_range = slice(zoom_start, zoom_end)

    # 创建内嵌图
    ax_inset = inset_axes(ax, width="40%", height="30%", loc='upper right')

    # 绘制局部细节
    ax_inset.plot(true_seq[zoom_range], color='black', linewidth=0.8)
    ax_inset.plot(pred_seq[zoom_range], color='orange', linewidth=0.8)

    # 设置内嵌图的样式
    ax_inset.set_xticks([])  # 隐藏x轴刻度
    ax_inset.set_yticks([])  # 隐藏y轴刻度
    ax_inset.spines['top'].set_visible(True)
    ax_inset.spines['right'].set_visible(True)
    ax_inset.spines['bottom'].set_visible(True)
    ax_inset.spines['left'].set_visible(True)

    # 在内嵌图中添加矩形框指示放大区域
    rect = plt.Rectangle((zoom_start, min(true_seq.min(), pred_seq.min())),
                         zoom_end - zoom_start,
                         max(true_seq.max(), pred_seq.max()) - min(true_seq.min(), pred_seq.min()),
                         fill=False, color='red', linestyle='--', alpha=0.7, linewidth=0.8)
    ax.add_patch(rect)
    con = ConnectionPatch(
        xyA=(0, 0), coordsA=ax_inset.transAxes,   # 内嵌图左下角 (0,0)
        xyB=(zoom_end, max(true_seq.max(), pred_seq.max())), coordsB=ax.transData,  # 主图矩形右上角
        axesA=ax_inset, axesB=ax,
        color="red", linewidth=0.8, linestyle="-", arrowstyle="<-"
    )
    ax_inset.add_artist(con)

    # ax.grid(True)
    # ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)  # 添加y=0基准线
    # ax.set_ylim(bottom=0)  # 纵坐标从0开始

# 删除多余子图
# for j in range(len(forecast_steps), rows * cols):
#     fig.delaxes(axes.flatten()[j])

plt.tight_layout()
plt.savefig("utils/img/multi_step_target_plot.jpg", dpi=300)
