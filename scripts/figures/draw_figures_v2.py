#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw all figures for the paper - v2 (fixes)."""
import json
import csv
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Fix minus sign issue
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = 'results/journal/figures_new'
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================================
# Fig4: Pymoo Comparison Bar Chart (all R2 tasks) - UPDATED
# ============================================================================
print("Drawing Fig4: Pymoo Comparison (with new data)...")

# 使用新的数据文件（全部R2数据集实验结果）
with open('paper_revision/P1_modern_algos/p1_modern_algo_results.json', 'r', encoding='utf-8') as f:
    pymoo_data = json.load(f)

from collections import defaultdict
algo_results = defaultdict(list)
for r in pymoo_data['records']:
    algo = r['algo']
    algo_results[algo].append(r['W_best'])

# 算法顺序：pymoo算法在前，基线算法在后
ordered_algos = ['NSGA3', 'MOEAD', 'NSGA2_BM', 'RL_NSGA2_BM_9dim']
labels = ['NSGA-III\n(pymoo)', 'MOEA/D\n(pymoo)', 'NSGA-II+BM\n(heuristic)', 'RL-NSGA-II\n(ours)']
colors = ['#e74c3c', '#e67e22', '#3498db', '#2ecc71']

w_vals = []
s_vals = []
for algo in ordered_algos:
    vals = algo_results.get(algo, [])
    if vals:
        w_vals.append(np.mean(vals))
        s_vals.append(np.std(vals, ddof=1) if len(vals) > 1 else 0)
    else:
        w_vals.append(0)
        s_vals.append(0)

fig, ax = plt.subplots(figsize=(9, 6))

# 双Y轴：线性轴显示基线算法，对数轴显示pymoo算法
# 主轴用于显示基线算法（更好的可视化）
bars = ax.bar(range(len(labels)), w_vals, color=colors, edgecolor='black', linewidth=0.5)

# Draw error bars manually with positive clamping
for i, (bar, val, std) in enumerate(zip(bars, w_vals, s_vals)):
    height = bar.get_height()
    # Clamp lower error to avoid negative values
    lower = min(std, val - 0.001) if val > 0.001 else 0
    upper = std
    ax.errorbar(bar.get_x() + bar.get_width()/2, val,
                yerr=[[lower], [upper]], fmt='none', ecolor='black',
                capsize=4, capthick=1, elinewidth=1)

# Add value labels
for i, (bar, val, std) in enumerate(zip(bars, w_vals, s_vals)):
    height = bar.get_height()
    label = f'{val:.2f}'
    ax.annotate(label, xy=(bar.get_x() + bar.get_width()/2, height + std + max(w_vals) * 0.02),
                xytext=(0, 5), textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel(r'$W_{total}$ (s)', fontsize=11)
# 从配置中获取实际任务数量
n_jobs = pymoo_data['config'].get('n_jobs', 'all')
n_jobs_str = str(n_jobs) if n_jobs else 'all'
ax.set_title(f'Comparison with Modern MOEAs (R2 / {n_jobs_str} tasks / pop=30 / gen=30 / 3 seeds)',
             fontsize=12, fontweight='bold')

# 分段Y轴：pymoo算法用对数，基线算法用线性
ax.set_yscale('log')
ax.set_ylim(1, 20000)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add annotation about the performance gap
improvement = w_vals[0] / w_vals[-1]  # NSGA3 / RL_NSGA2
ax.annotate(f'~{int(improvement)}x worse\n(pymoo algorithms)', xy=(0.5, 0.75), xycoords='axes fraction',
            fontsize=9, ha='center', style='italic', color='gray',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/Fig4_Pymoo_Comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved Fig4_Pymoo_Comparison.png")

# ============================================================================
# Fig7: Lift Count Sensitivity Line Chart
# ============================================================================
print("Drawing Fig7: Lift Count Sensitivity...")

lift_data = []
with open('results/rho_regression_v2/rho_summary.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        lift_data.append({
            'lift_count': int(row['lift_count']),
            'algorithm': row['algorithm'],
            'rho_mean': float(row['rho_mean']),
            'w_mean': float(row['w_mean']),
            'w_std': float(row['w_std']),
        })

colors_map = {'FIFO': '#e74c3c', 'NSGA-II': '#3498db', 'IGA': '#9b59b6', 'RL-NSGA-II': '#2ecc71'}
markers = {'FIFO': 's', 'NSGA-II': 'o', 'IGA': '^', 'RL-NSGA-II': 'D'}

fig, ax1 = plt.subplots(figsize=(9, 5))

for algo in ['FIFO', 'NSGA-II', 'IGA', 'RL-NSGA-II']:
    pts = [d for d in lift_data if d['algorithm'] == algo]
    pts = sorted(pts, key=lambda x: x['lift_count'])
    x = [p['lift_count'] for p in pts]
    y = [p['w_mean'] for p in pts]
    e = [p['w_std'] for p in pts]
    ax1.plot(x, y, marker=markers[algo], color=colors_map[algo], label=algo,
             linewidth=1.5, markersize=7)
    ax1.errorbar(x, y, yerr=e, fmt='none', color=colors_map[algo], capsize=3, alpha=0.5)

ax1.set_xlabel('Number of Lifts', fontsize=11)
ax1.set_ylabel(r'$W_{total}$ (s)', fontsize=11, color='black')
ax1.set_xticks([1, 2, 3, 4, 6])
ax1.set_title('Lift Count Sensitivity Analysis (R2 / 1120 tasks / decoupled / 3 seeds)',
              fontsize=12, fontweight='bold')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(alpha=0.3, linestyle='--')
ax1.spines['top'].set_visible(False)

# Secondary x-axis for rho
ax2 = ax1.twiny()
fifo_pts = sorted([d for d in lift_data if d['algorithm'] == 'FIFO'], key=lambda x: x['lift_count'])
rho_ticks = [p['rho_mean'] for p in fifo_pts]
lift_ticks = [p['lift_count'] for p in fifo_pts]
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(lift_ticks)
ax2.set_xticklabels([f'{r:.3f}' for r in rho_ticks], fontsize=9)
ax2.set_xlabel(r'$\rho$ (congestion degree)', fontsize=10)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/Fig7_Lift_Sensitivity.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved Fig7_Lift_Sensitivity.png")

# ============================================================================
# Fig5: Ablation Study Waterfall Chart - FIXED arrows
# ============================================================================
print("Drawing Fig5: Ablation Waterfall...")

ablation_data = [
    {'variant': 'M0_FIFO', 'w_mean': 195.52, 'w_std': 0},
    {'variant': 'M1_BaseNSGA2', 'w_mean': 102.32, 'w_std': 17.83},
    {'variant': 'M2_TimewindowNSGA2', 'w_mean': 106.45, 'w_std': 7.06},
    {'variant': 'M3_CriticalBlockNSGA2', 'w_mean': 107.07, 'w_std': 13.65},
    {'variant': 'M4_RL_MathOnly', 'w_mean': 103.58, 'w_std': 33.26},
    {'variant': 'M5_RL_PhysicalMath', 'w_mean': 93.61, 'w_std': 25.88},
]

labels5 = ['M0\nFIFO', 'M1\nBase NSGA-II', 'M2\n+TW', 'M3\n+TW+BM', 'M4\nRL-Math', 'M5\nFull Model']
colors5 = ['#e74c3c', '#3498db', '#e67e22', '#2ecc71', '#9b59b6', '#1abc9c']

fig, ax = plt.subplots(figsize=(10, 5.5))

x = np.arange(len(labels5))
w_means5 = [d['w_mean'] for d in ablation_data]
w_stds5 = [d['w_std'] for d in ablation_data]

bars = ax.bar(x, w_means5, yerr=w_stds5, capsize=5, color=colors5, edgecolor='black',
              linewidth=0.5, zorder=3)

# Add value labels
for i, (bar, val, std) in enumerate(zip(bars, w_means5, w_stds5)):
    height = bar.get_height()
    if i == 0:
        label = f'{val:.2f}s\n(baseline)'
    else:
        pct = (1 - val / w_means5[0]) * 100
        label = f'{val:.2f}s\n({pct:.1f}%)'
    ax.annotate(label, xy=(bar.get_x() + bar.get_width()/2, height + std),
                xytext=(0, 5), textcoords="offset points",
                ha='center', va='bottom', fontsize=8)

# Add improvement arrows between consecutive bars (M0→M1 only, others show change)
for i in range(1, len(bars)):
    y_prev = w_means5[i-1]
    y_curr = w_means5[i]
    x_prev = bars[i-1].get_x() + bars[i-1].get_width()
    x_curr = bars[i].get_x()
    xmid = (x_prev + x_curr) / 2

    pct_change = (y_prev - y_curr) / y_prev * 100
    color = 'green' if y_curr < y_prev else 'red'
    symbol = '↓' if y_curr < y_prev else '↑'

    # Arrow from higher to lower
    y_high = max(y_prev, y_curr) + max(w_stds5[i-1], w_stds5[i]) + 3
    y_low = min(y_prev, y_curr) - 3

    ax.annotate('', xy=(xmid, y_low), xytext=(xmid, y_high),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
    ax.text(xmid, (y_high + y_low) / 2, f'{symbol}{abs(pct_change):.1f}%',
            ha='center', va='center', fontsize=7, color=color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color, alpha=0.8))

ax.set_xticks(x)
ax.set_xticklabels(labels5, fontsize=9)
ax.set_ylabel(r'$W_{total}$ (s)', fontsize=11)
ax.set_title('Ablation Study: Module Contribution Analysis (R2 / 1120 tasks / decoupled / 5 seeds)',
             fontsize=12, fontweight='bold')
ax.set_ylim(0, 220)
ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/Fig5_Ablation_Waterfall.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved Fig5_Ablation_Waterfall.png")

# ============================================================================
# Fig6: Hyperparameter Robustness
# ============================================================================
print("Drawing Fig6: Hyperparameter Robustness...")

with open('results/hyperparam/hyperparam_r2_results.json', 'r', encoding='utf-8') as f:
    hp_data = json.load(f)

sweeps = {'POP': [], 'GEN': [], 'EPS': [], 'GAM': []}
for r in hp_data['records']:
    sweep = r['sweep']
    if sweep in sweeps:
        sweeps[sweep].append(r)

for k in sweeps:
    sweeps[k] = sorted(sweeps[k], key=lambda x: x.get({'POP':'pop_size','GEN':'n_gen','EPS':'epsilon','GAM':'gamma'}[k]))

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
fig.suptitle('Hyperparameter Robustness Analysis (R2 / 1120 tasks / sync / 3 seeds)',
             fontsize=13, fontweight='bold', y=1.02)

plot_configs = [
    ('POP', 'pop_size', axes[0, 0], '(a) Population Size', 'POP'),
    ('GEN', 'n_gen', axes[0, 1], '(b) Generation Count', 'GEN'),
    ('EPS', 'epsilon', axes[1, 0], '(c) Exploration Rate', r'$\epsilon$'),
    ('GAM', 'gamma', axes[1, 1], '(d) Discount Factor', r'$\gamma$'),
]

for sweep_key, param_key, ax, title, xlabel in plot_configs:
    records = sweeps[sweep_key]
    x = [r[param_key] for r in records]
    y = [r['W_mean'] for r in records]
    err = [r['W_std'] for r in records]

    ax.errorbar(x, y, yerr=err, fmt='o-', color='#2c7fb8', capsize=5,
                markersize=8, linewidth=1.5, ecolor='#e74c3c', capthick=1.5)

    # Mark default value
    default_idx = 1
    ax.plot(x[default_idx], y[default_idx], 'o', color='#2c7fb8', markersize=12,
            markeredgecolor='gold', markeredgewidth=2, zorder=5)
    ax.annotate('★ Default', xy=(x[default_idx], y[default_idx]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=8, color='#d35400', fontweight='bold')

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(r'$W_{total}$ (s)', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/Fig6_Hyperparam_Robustness.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved Fig6_Hyperparam_Robustness.png")

print("\nDone! Figures saved to:", OUT_DIR)
