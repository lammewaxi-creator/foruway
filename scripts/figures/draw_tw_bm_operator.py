import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# 全局字体和样式设置
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

def draw_background_grid(ax, title):
    """绘制统一的黑色货架网格背景和坐标轴"""
    ax.set_title(title, fontsize=14, pad=10)
    
    # 货架参数 (8列 4行)
    x_centers = [4, 9, 14, 19, 24, 29, 34, 39]
    y_centers = [6, 15, 24, 33]
    width, height = 3.5, 7.5
    
    # 绘制黑色货架块
    for xc in x_centers:
        for yc in y_centers:
            rect = FancyBboxPatch((xc - width/2, yc - height/2), 
                                  width, height, boxstyle="round,pad=0.02,rounding_size=0.5",
                                  facecolor='black', edgecolor='none')
            ax.add_patch(rect)
            
    # 设置坐标轴范围
    ax.set_xlim(0, 45)
    ax.set_ylim(0, 40)
    
    # 隐藏上方和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # X轴刻度设置
    ax.set_xlabel('Grid X', fontsize=12, labelpad=0)
    ax.set_xticks([10, 20, 30, 40])
    ax.set_yticks([]) # 隐藏Y轴
    
    # 绘制 Packing Station (橙色圆点)
    station_x, station_y = 29, 1
    station = plt.Circle((station_x, station_y), 1.2, color='#fca311', zorder=5)
    ax.add_patch(station)
    ax.text(station_x, station_y + 1.5, 'Packing Station', ha='center', fontsize=8, zorder=6)

def add_path(ax, x_coords, y_coords, color, style='-'):
    """绘制带有节点标记的路径"""
    ax.plot(x_coords, y_coords, color=color, linewidth=2.5, linestyle=style, zorder=3)
    ax.scatter(x_coords, y_coords, color=color, s=40, marker='v', zorder=4)

# ================= 创建 2x2 画布 =================
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
plt.subplots_adjust(wspace=0.25, hspace=0.35)

# ----------------- 图 (a) Before TW -----------------
ax_a = axs[0, 0]
draw_background_grid(ax_a, "(a) Before TW")

# 路径 (a)
path_a_x = [29, 20, 11,  6,  6, 11, 11, 21, 26, 26, 31, 36, 41, 41, 36, 31, 29]
path_a_y = [ 1,  5,  5, 13, 20, 22, 28, 28, 22, 29, 30, 29, 29, 13, 11, 13,  1]
add_path(ax_a, path_a_x, path_a_y, color='#50c5d9')

# Idle Gap 区域
idle_gap = patches.Rectangle((17, 12), 4, 16, facecolor='#e0e0e0', edgecolor='gray', hatch='///', zorder=2)
ax_a.add_patch(idle_gap)
ax_a.text(19, 20, 'Idle\nGap', ha='center', va='center', fontsize=10, 
          bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

# ----------------- 图 (b) After TW -----------------
ax_b = axs[0, 1]
draw_background_grid(ax_b, "(b) After TW")

# 路径 (b) (穿过Gap)
path_b_x = [29, 20, 11,  6,  6, 11, 15, 17, 17, 21, 26, 26, 31, 36, 41, 41, 36, 31, 29]
path_b_y = [ 1,  5,  5, 13, 20, 22, 22, 16, 28, 28, 22, 29, 30, 29, 29, 13, 11, 13,  1]
add_path(ax_b, path_b_x, path_b_y, color='#50c5d9')

# 强制插入的任务块 (购物车图标用带颜色的方块代替模拟)
cart1 = patches.Rectangle((17, 23), 4, 5, facecolor='#e5e5e5', edgecolor='gray', zorder=4)
cart2 = patches.Rectangle((17, 14), 4, 4, facecolor='#135c24', edgecolor='none', zorder=4)
ax_b.add_patch(cart1)
ax_b.add_patch(cart2)
ax_b.text(19, 25.5, '🛒', color='#135c24', ha='center', va='center', fontsize=14, zorder=5) # 浅色框深色车
ax_b.text(19, 16, '🛒', color='white', ha='center', va='center', fontsize=12, zorder=5)    # 深色框白色车

# 跨图注释: (a) -> (b)
ax_b.annotate('Compress\nFinish Time', xy=(15, 27), xytext=(-12, 27),
              arrowprops=dict(facecolor='#50c5d9', edgecolor='#50c5d9', width=2, headwidth=8),
              ha='center', va='center', fontsize=10, clip_on=False)

ax_b.annotate('Force-insert\nCoupled\nTask', xy=(15, 15), xytext=(-12, 15),
              arrowprops=dict(facecolor='#135c24', edgecolor='#135c24', width=1.5, headwidth=7),
              ha='center', va='center', fontsize=10, clip_on=False)


# ----------------- 图 (c) Before BM -----------------
ax_c = axs[1, 0]
draw_background_grid(ax_c, "(c) Before BM (Local Optimum)")

# 路径 (c)
path_c_x = [29, 20, 11,  6,  6, 11, 11, 15, 26, 26, 31, 36, 31, 29]
path_c_y = [ 1,  5,  5, 13, 22, 26, 28, 29, 29, 22, 12, 11,  8,  1]
add_path(ax_c, path_c_x, path_c_y, color='#bc5050')

# Key Block & Bottleneck
bbox_style = dict(boxstyle="round,pad=0.3", fc="#fbd6d6", ec="#bc5050", lw=1.5)
ax_c.text(21, 27, "Key Block", ha="center", va="center", size=10, bbox=bbox_style, zorder=5)
ax_c.text(21, 17, "Bottleneck", ha="center", va="center", size=10, bbox=bbox_style, zorder=5)

# 绘制锯齿状的拥堵连接线
zigzag_x = [18, 19, 20, 21, 22, 23, 24]
zigzag_y = [23, 21, 23, 20, 23, 21, 22]
ax_c.plot(zigzag_x, zigzag_y, color='red', lw=2)

# 右侧指向注释
ax_c.annotate('Identify\nKey Path', xy=(28, 27), xytext=(48, 27),
              arrowprops=dict(facecolor='#bc5050', edgecolor='#bc5050', width=1.5, headwidth=7),
              ha='left', va='center', fontsize=11, clip_on=False)


# ----------------- 图 (d) After BM -----------------
ax_d = axs[1, 1]
draw_background_grid(ax_d, "(d) After BM (Breaking Barrier)")

# 路径 (d) (紫色，绕开中心区)
path_d_x = [29, 22, 16, 16, 11,  6,  6, 16, 16, 26, 26, 34, 34, 42, 42, 34, 34, 29]
path_d_y = [ 1,  2,  5, 15, 15, 20, 26, 28, 30, 30, 28, 28, 35, 35, 13, 11,  6,  1]
add_path(ax_d, path_d_x, path_d_y, color='#7e57c2')

# 左侧指向注释
ax_d.annotate('Directed\nMutation', xy=(11, 15), xytext=(-10, 10),
              arrowprops=dict(facecolor='#7e57c2', edgecolor='#7e57c2', width=1.5, headwidth=7),
              ha='right', va='center', fontsize=11, clip_on=False)

# 保存与显示
plt.savefig("routing_algorithm_visualization.svg", format="svg", bbox_inches="tight")
plt.show()
