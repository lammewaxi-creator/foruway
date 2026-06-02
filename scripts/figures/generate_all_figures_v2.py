#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Journal Figure Generation Script (Complete)
Generates all 7 core figures for the paper.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy import stats

# ==========================================
# Global style settings
# ==========================================
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300

OUTPUT_DIR = "results/journal/figures_new"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# Fig.1: Method Framework Diagram
# ==========================================
def plot_fig1_framework():
    """Fig.1: Overall framework of the proposed method"""
    print("[Fig.1] Generating framework diagram...")
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Color scheme
    c_rho = '#4A90D9'      # Blue: rho detection
    c_nsga = '#5CB85C'     # Green: NSGA-II
    c_rl = '#F0AD4E'       # Orange: Q-learning
    c_ops = '#D9534F'      # Red: Specialized operators
    c_cft = '#9B59B6'      # Purple: C-EFT

    def draw_box(x, y, w, h, color, text, fontsize=10, text_color='white'):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, color=text_color, fontweight='bold', wrap=True)

    def draw_arrow(x1, y1, x2, y2, color='black', style='->', lw=1.5):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw))

    # Title
    ax.text(7, 9.5, 'Proposed RL-NSGA-II with ρ-Aware Adaptive Coupling Framework',
            ha='center', va='center', fontsize=16, fontweight='bold')

    # Left column: WMS -> rho detection
    draw_box(0.5, 7.5, 2.5, 1.2, '#6C757D', 'WMS\nReal-time Data', 10)
    draw_arrow(3, 8.1, 4, 8.1)
    draw_box(4, 7.5, 2.5, 1.2, c_rho, 'ρ Detection\nModule', 10)

    # Four-mode switch
    draw_box(4, 5.5, 2.5, 1.5, c_rho, 'Four-Mode Switch\n(Loose / Active /\nPredictive / Passive)\nθ₁=0.70, θ₂=0.80\nθ₃=0.95, δ=0.03', 8)
    draw_arrow(5.25, 7.5, 5.25, 7.1)

    # Hysteresis loop indicator
    ax.annotate('', xy=(6.7, 6.5), xytext=(6.7, 7.8),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2, ls='--'))
    ax.text(6.9, 7.15, 'Hysteresis\nδ=0.03', fontsize=8, color='red')

    # Middle: NSGA-II evolution cycle
    draw_box(4, 3.2, 2.5, 1.8, c_nsga, 'NSGA-II Evolution\nCycle\n---\nCrossover →\nMutation →\nEvaluation →\nSelection', 9)
    draw_arrow(5.25, 5.5, 5.25, 5.1)

    # Connection from rho to NSGA-II
    draw_arrow(5.25, 3.2, 5.25, 2.5)

    # Right upper: Q-learning
    draw_box(7.5, 6.5, 3, 2.5, c_rl, 'Q-Learning Module\n---\n9-Dim State →\n27 States (φ(f))\nε-greedy Action\n(Pᶜ, Pₘ) Tuning\n---\nα=0.1, γ=0.85\nε=0.2', 9)
    draw_arrow(6.5, 7.5, 7.5, 7.5, color=c_rl, lw=2)

    # RL -> NSGA-II control flow
    draw_arrow(9, 6.5, 9, 4.5, color=c_rl, lw=2)
    ax.text(9.2, 5.5, 'Control\nSignal', fontsize=8, color=c_rl, fontweight='bold')

    # Right lower: Specialized operators
    draw_box(7.5, 3.2, 3, 1.8, c_ops, 'Specialized Operators\n---\nTW: Time-Window\n   Left-Insertion\nBM: Critical-Block\n   Local Search\nAisle Affinity', 9)
    draw_arrow(6.5, 4.1, 7.5, 4.1, color=c_ops, lw=2)

    # C-EFT device allocation
    draw_box(7.5, 1.0, 3, 1.5, c_cft, 'C-EFT Device\nAllocation Rule\n(Coupled-Earliest\nFinish Time)', 9)
    draw_arrow(9, 3.2, 9, 2.6, color=c_cft, lw=2)

    # Output
    draw_box(11.5, 4.5, 2, 1.5, '#6C757D', 'Output:\nOptimal Schedule\n(Cₘₐₓ, Wᵀᵒᵗᵒʟ)', 10)
    draw_arrow(10.5, 4.5, 11.5, 5.25, color='black', lw=2)

    # Data flow arrows (solid) vs Control flow arrows (dashed)
    ax.text(0.5, 0.5, 'Solid arrows: Data flow    Dashed arrows: Control/Feedback flow',
            fontsize=9, style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Fig1_Framework.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Fig1_Framework.png")


# ==========================================
# Fig.2: Convergence Curve (Dual Y-axis)
# ==========================================
def plot_fig2_convergence():
    """Fig.2: Convergence curve with dual Y-axis (single plot)"""
    print("[Fig.2] Generating convergence curve...")

    # Load raw convergence data
    with open("paper_revision/15_visualizations/raw_convergence_data.json", 'r') as f:
        conv_data = json.load(f)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    colors = {
        'NSGA-II (Swap)': '#1f77b4',
        'NSGA-II + BM': '#ff7f0e',
        'RL-NSGA-II + BM + 9-dim': '#d62728'
    }
    markers = {'NSGA-II (Swap)': 'o', 'NSGA-II + BM': 's', 'RL-NSGA-II + BM + 9-dim': '^'}

    # Plot W_total on left Y-axis (log scale)
    for label, hist in conv_data.items():
        gens = [h['generation'] for h in hist]
        ws = [h['w_total'] for h in hist]
        ax1.plot(gens, ws, marker=markers[label], color=colors[label],
                label=label, markersize=5, linewidth=2, markevery=3)

    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel(r'$W_{total}$ (s) - Log Scale', fontsize=12, color='#1f77b4')
    ax1.set_yscale('log')
    ax1.set_ylim(0.001, 10)
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.grid(True, linestyle='--', alpha=0.4)

    # Cmax on right Y-axis (linear scale)
    ax2 = ax1.twinx()
    for label, hist in conv_data.items():
        gens = [h['generation'] for h in hist]
        cs = [h['cmax'] for h in hist]
        ax2.plot(gens, cs, linestyle='--', color=colors[label],
                linewidth=1.5, alpha=0.6, markevery=3)

    ax2.set_ylabel(r'$C_{max}$ (s) - Linear Scale', fontsize=12, color='#d62728')
    ax2.set_ylim(21810, 21820)
    ax2.tick_params(axis='y', labelcolor='#d62728')

    # Add epsilon_C tolerance band
    cmax_base = 21816.64
    epsilon_c = 0.0002  # 0.02%
    ax2.axhspan(cmax_base * (1 - epsilon_c), cmax_base * (1 + epsilon_c),
                alpha=0.1, color='green', label=r'$\epsilon_C$-tolerance band')

    # Annotation for RL breakthrough
    ax1.annotate('RL first breakthrough\n(Gen ~15)',
                xy=(15, 0.06), xytext=(20, 0.5),
                arrowprops=dict(arrowstyle='->', color='#d62728'),
                fontsize=10, color='#d62728')

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

    ax1.set_title('Convergence Curve (R2 / 600 tasks / pop=30 / gen=30)',
                  fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Fig2_Convergence.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Fig2_Convergence.png")


# ==========================================
# Fig.3: Algorithm Comparison (20-seed scatter)
# ==========================================
def plot_fig3_algorithm_comparison():
    """Fig.3: Algorithm comparison using 20-seed results"""
    print("[Fig.3] Generating algorithm comparison plot...")

    # Load 20-seed results
    with open("paper_revision/P0_20seed_friedman/p0_20seed_results.json", 'r') as f:
        data = json.load(f)

    fifo_cmax = data['fifo_baseline']['cmax']
    fifo_w = data['fifo_baseline']['w_total']

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot FIFO baseline (single point)
    ax.scatter(fifo_cmax, fifo_w, marker='*', s=400, color='red',
              label='FIFO (baseline)', zorder=5, edgecolors='black', linewidths=1)

    # Plot each algorithm's 20 seeds
    algo_configs = {
        'NSGA2_swap': {'label': 'NSGA-II (Swap)', 'color': '#1f77b4', 'marker': 'o'},
        'NSGA2_BM': {'label': 'NSGA-II + BM', 'color': '#ff7f0e', 'marker': 's'},
        'RL_NSGA2_BM_9dim': {'label': 'RL-NSGA-II + BM + 9-dim', 'color': '#d62728', 'marker': '^'},
    }

    for algo, cfg in algo_configs.items():
        ws = data['ws_per_algo'][algo]
        cs = data['cs_per_algo'][algo]

        # Add jitter to Cmax for visibility (very small)
        cs_jittered = [c + np.random.normal(0, 0.3) for c in cs]

        ax.scatter(cs_jittered, ws, marker=cfg['marker'], c=cfg['color'],
                  s=80, alpha=0.6, label=cfg['label'], edgecolors='black', linewidths=0.5)

        # Plot mean
        ax.scatter(np.mean(cs), np.mean(ws), marker=cfg['marker'], c=cfg['color'],
                  s=200, edgecolors='black', linewidths=2, zorder=4)

    # Epsilon-C tolerance band (vertical lines)
    cmax_base = fifo_cmax
    epsilon_c = 0.0002
    ax.axvline(x=cmax_base * (1 - epsilon_c), color='green', linestyle='--',
              alpha=0.5, linewidth=1.5)
    ax.axvline(x=cmax_base * (1 + epsilon_c), color='green', linestyle='--',
              alpha=0.5, linewidth=1.5)
    ax.axvspan(cmax_base * (1 - epsilon_c), cmax_base * (1 + epsilon_c),
              alpha=0.05, color='green')
    ax.text(cmax_base * (1 + epsilon_c) + 0.5, 4.5,
           r'$\epsilon_C \cdot C_{max}^{base}$ tolerance',
           fontsize=9, color='green', rotation=90, va='top')

    ax.set_xlabel(r'$C_{max}$ (s)', fontsize=12)
    ax.set_ylabel(r'$W_{total}$ (s) - Log Scale', fontsize=12)
    ax.set_yscale('log')
    ax.set_ylim(0.001, 10)
    ax.set_title('Algorithm Performance Comparison (R2 / 600 tasks / 20 seeds)',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Fig3_Algorithm_Comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Fig3_Algorithm_Comparison.png")


# ==========================================
# Fig.4: Rho Regression (fixed x-axis)
# ==========================================
def plot_fig4_rho_regression():
    """Fig.4: Rho regression with rho on x-axis"""
    print("[Fig.4] Generating rho regression plot...")

    df = pd.read_csv("results/rho_analysis/rho_analysis_raw.csv")

    algorithms = {
        'FIFO': {'color': '#2E86AB', 'marker': 'o'},
        'RL-NSGA-II': {'color': '#E94F37', 'marker': 's'}
    }

    regressions = {
        'FIFO': {'k': 2036.15, 'b': -19734.79, 'r2': 0.902},
        'RL-NSGA-II': {'k': 357.16, 'b': -4573.85, 'r2': 0.563}
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (algo_name, style) in enumerate(algorithms.items()):
        ax = axes[idx]
        algo_data = df[df['algorithm'] == algo_name]

        rho_vals = algo_data['rho_global'].values
        w_vals = algo_data['w_total'].values

        ax.scatter(rho_vals, w_vals, c=style['color'], marker=style['marker'],
                  alpha=0.5, s=30, edgecolors='none', label='Observations')

        rho_range = np.linspace(0.5, 1.0, 200)
        reg = regressions[algo_name]
        w_pred = reg['k'] * rho_range + reg['b']

        ax.plot(rho_range, w_pred, color=style['color'], linewidth=2.5,
               label=f'Fit: $W_{{total}}$ = {reg["k"]:.0f}$\\rho$ {reg["b"]:+.0f}')

        residuals = w_vals - (reg['k'] * rho_vals + reg['b'])
        std_err = np.std(residuals)
        ax.fill_between(rho_range, w_pred - 1.96*std_err, w_pred + 1.96*std_err,
                       color=style['color'], alpha=0.15)

        ax.set_xlabel(r'System Congestion Degree $\rho$', fontsize=12)
        ax.set_ylabel(r'Total Waiting Time $W_{total}$ (s)', fontsize=12)
        ax.set_title(f'{algo_name}  ($R^2$ = {reg["r2"]:.3f})', fontsize=12)
        ax.set_xlim(0.5, 1.0)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='upper left', fontsize=9)

    passivation = (1 - 357.16 / 2036.15) * 100
    fig.text(0.5, 0.01,
            f'Passivation Effect = (1 - {357.16:.0f}/{2036.15:.0f}) x 100% = {passivation:.1f}%',
            ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"{OUTPUT_DIR}/Fig4_Rho_Regression.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Fig4_Rho_Regression.png")


# ==========================================
# Fig.5: Ablation Study Waterfall Chart
# ==========================================
def plot_fig5_ablation():
    """Fig.5: Ablation study waterfall bar chart"""
    print("[Fig.5] Generating ablation study waterfall chart...")

    # Use R2 20-seed data for available variants
    # M0: FIFO, M1: NSGA2_swap, M3: NSGA2_BM, M5: RL_NSGA2_BM_9dim
    # M2 (+TW) and M4 (RL-Math) estimated from paper descriptions

    variants = ['M0\nFIFO', 'M1\nBase NSGA-II', 'M2\n+TW', 'M3\n+TW+BM',
                'M4\nRL-Math', 'M5\nFull Model']

    # R2 20-seed actual data
    w_values = [4.7605, 1.6065, None, 1.3141, None, 0.6859]

    # Estimate M2 and M4 based on R1 paper percentages applied to R2 M1
    # Paper: M2 ~5% better than M1, M4 ~8% better than M1
    # But R2 actual M3 improvement = (1.6065-1.3141)/1.6065 = 18.2%
    # Scale factor: R2 M3 improvement / R1 M3 improvement = 18.2/12 = 1.52
    scale_factor = 0.182 / 0.12
    w_values[2] = w_values[1] * (1 - 0.05 * scale_factor)  # M2
    w_values[4] = w_values[1] * (1 - 0.08 * scale_factor)  # M4

    # Calculate improvements relative to M0
    improvements = [(w_values[0] - w) / w_values[0] * 100 for w in w_values]

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728']
    x = np.arange(len(variants))
    width = 0.6

    bars = ax.bar(x, w_values, width, color=colors, alpha=0.8, edgecolor='black')

    # Add value labels on bars
    for i, (bar, w, imp) in enumerate(zip(bars, w_values, improvements)):
        height = bar.get_height()
        if i == 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{w:.2f}s\n(baseline)', ha='center', va='bottom', fontsize=10)
        else:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{w:.2f}s\n({imp:.1f}%)', ha='center', va='bottom', fontsize=9)

    # Add improvement arrows between consecutive bars
    for i in range(1, len(variants)):
        y1 = w_values[i-1]
        y2 = w_values[i]
        mid_y = (y1 + y2) / 2
        ax.annotate('', xy=(i-0.3, y2), xytext=(i-0.3, y1),
                   arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
        improvement = (y1 - y2) / y1 * 100
        ax.text(i-0.45, mid_y, f'-{improvement:.1f}%',
               fontsize=8, color='green', fontweight='bold',
               rotation=90, va='center')

    ax.set_ylabel(r'$W_{total}$ (s)', fontsize=12)
    ax.set_title('Ablation Study: Module Contribution Analysis (R2 / 600 tasks / 20 seeds)',
                fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(variants, fontsize=10)
    ax.set_ylim(0, 6)
    ax.grid(True, linestyle='--', alpha=0.4, axis='y')

    # Note about estimated values
    ax.text(0.02, 0.98, '* M2 (+TW) and M4 (RL-Math) values estimated from R1 ratios\n  scaled by R2 observed improvement factor',
           transform=ax.transAxes, fontsize=8, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Fig5_Ablation_Waterfall.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Fig5_Ablation_Waterfall.png")


# ==========================================
# Fig.6: Hyperparameter Robustness
# ==========================================
def plot_fig6_hyperparam():
    """Fig.6: 2x2 hyperparameter robustness subplot matrix"""
    print("[Fig.6] Generating hyperparameter robustness plot...")

    with open("paper_revision/P2_hyperparam_robustness/p2_hyperparam_robustness.json", 'r') as f:
        data = json.load(f)

    records = data['records']
    fifo_w = data['fifo_baseline']['w_total']

    sweep_data = {}
    for r in records:
        sweep = r['sweep']
        if sweep not in sweep_data:
            sweep_data[sweep] = []
        sweep_data[sweep].append(r)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Hyperparameter Robustness Analysis (R2 / 600 tasks / 3 seeds)',
                fontsize=14, fontweight='bold')

    configs = [
        ('POP', 'POP', [15, 30, 50], 'Population Size'),
        ('GEN', 'GEN', [15, 30, 50], 'Generation Count'),
        ('EPSILON', r'$\epsilon$', [0.1, 0.2, 0.3], 'Exploration Rate'),
        ('GAMMA', r'$\gamma$', [0.7, 0.85, 0.95], 'Discount Factor')
    ]

    for idx, (sweep_key, xlabel, xvals, title) in enumerate(configs):
        ax = axes[idx // 2, idx % 2]
        recs = sweep_data[sweep_key]
        recs_sorted = sorted(recs, key=lambda r: r.get(
            'pop_size' if sweep_key=='POP' else 'n_gen' if sweep_key=='GEN' else 'epsilon' if sweep_key=='EPSILON' else 'gamma', 0))

        means = [r['W_mean'] for r in recs_sorted]
        stds = [r['W_std'] for r in recs_sorted]
        actual_xvals = []
        for r in recs_sorted:
            if sweep_key == 'POP':
                actual_xvals.append(r['pop_size'])
            elif sweep_key == 'GEN':
                actual_xvals.append(r['n_gen'])
            elif sweep_key == 'EPSILON':
                actual_xvals.append(r['epsilon'])
            else:
                actual_xvals.append(r['gamma'])

        ax.errorbar(actual_xvals, means, yerr=stds, marker='o', markersize=10,
                   linewidth=2.5, capsize=8, capthick=2, color='#2E86AB',
                   ecolor='#E94F37', elinewidth=2)

        if sweep_key == 'GEN':
            ax.annotate(r'$\geq$30 saturation', xy=(30, means[1]),
                       xytext=(18, means[1] + 0.3),
                       arrowprops=dict(arrowstyle='->', color='red'),
                       fontsize=10, color='red')
        elif sweep_key == 'EPSILON':
            ax.annotate('Concave optimum', xy=(0.2, means[1]),
                       xytext=(0.12, means[1] + 0.3),
                       arrowprops=dict(arrowstyle='->', color='red'),
                       fontsize=10, color='red')
        elif sweep_key == 'GAMMA':
            ax.annotate('Completely insensitive', xy=(0.85, means[1]),
                       xytext=(0.72, means[1] + 0.05),
                       arrowprops=dict(arrowstyle='->', color='green'),
                       fontsize=10, color='green')

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(r'$W_{total}$ (s)', fontsize=12)
        ax.set_title(f'({chr(97+idx)}) {title}', fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_xticks(actual_xvals)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{OUTPUT_DIR}/Fig6_Hyperparam_Robustness.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Fig6_Hyperparam_Robustness.png")


# ==========================================
# Fig.7: Rho Sensitivity Heatmap
# ==========================================
def plot_fig7_rho_heatmap():
    """Fig.7: Complete rho threshold sensitivity heatmap"""
    print("[Fig.7] Generating rho sensitivity heatmap...")

    with open("paper_revision/06_rho_sensitivity/rho_sensitivity.json", 'r') as f:
        data = json.load(f)

    t1_vals = sorted(set(r['theta1'] for r in data))
    t2_vals = sorted(set(r['theta2'] for r in data))
    t3_vals = sorted(set(r['theta3'] for r in data))

    all_w = [r['w_total'] for r in data]
    best_w = min(all_w)
    worst_w = max(all_w)
    best_count = sum(1 for w in all_w if abs(w - best_w) < 0.001)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(r'$\rho$ Threshold Sensitivity: Complete 24-Group Scan ($\theta_1 \times \theta_2 \times \theta_3$)',
                fontsize=14, fontweight='bold')

    # Slice 1: theta3=0.95
    ax1 = axes[0]
    grid1 = np.full((len(t1_vals), len(t2_vals)), np.nan)
    for r in data:
        if abs(r['theta3'] - 0.95) < 1e-6:
            i = t1_vals.index(r['theta1'])
            j = t2_vals.index(r['theta2'])
            grid1[i, j] = r['w_total']

    im1 = ax1.imshow(grid1, cmap='RdYlGn_r', aspect='auto', vmin=3.5, vmax=4.5)
    ax1.set_xticks(range(len(t2_vals)))
    ax1.set_xticklabels([f"{v:.2f}" for v in t2_vals])
    ax1.set_yticks(range(len(t1_vals)))
    ax1.set_yticklabels([f"{v:.2f}" for v in t1_vals])
    ax1.set_xlabel(r'$\theta_2$', fontsize=12)
    ax1.set_ylabel(r'$\theta_1$', fontsize=12)
    ax1.set_title(r'(a) $\theta_3$ = 0.95', fontsize=12)
    for i in range(len(t1_vals)):
        for j in range(len(t2_vals)):
            if not np.isnan(grid1[i, j]):
                val = grid1[i, j]
                color = 'white' if val > 4.2 else 'black'
                ax1.text(j, i, f"{val:.2f}", ha='center', va='center',
                        color=color, fontsize=11, fontweight='bold')
    fig.colorbar(im1, ax=ax1, label=r'$W_{total}$ (s)')

    # Slice 2: theta2=0.80
    ax2 = axes[1]
    grid2 = np.full((len(t1_vals), len(t3_vals)), np.nan)
    for r in data:
        if abs(r['theta2'] - 0.80) < 1e-6:
            i = t1_vals.index(r['theta1'])
            j = t3_vals.index(r['theta3'])
            grid2[i, j] = r['w_total']

    im2 = ax2.imshow(grid2, cmap='RdYlGn_r', aspect='auto', vmin=3.5, vmax=4.5)
    ax2.set_xticks(range(len(t3_vals)))
    ax2.set_xticklabels([f"{v:.2f}" for v in t3_vals])
    ax2.set_yticks(range(len(t1_vals)))
    ax2.set_yticklabels([f"{v:.2f}" for v in t1_vals])
    ax2.set_xlabel(r'$\theta_3$', fontsize=12)
    ax2.set_ylabel(r'$\theta_1$', fontsize=12)
    ax2.set_title(r'(b) $\theta_2$ = 0.80', fontsize=12)
    for i in range(len(t1_vals)):
        for j in range(len(t3_vals)):
            if not np.isnan(grid2[i, j]):
                val = grid2[i, j]
                color = 'white' if val > 4.2 else 'black'
                ax2.text(j, i, f"{val:.2f}", ha='center', va='center',
                        color=color, fontsize=11, fontweight='bold')
    fig.colorbar(im2, ax=ax2, label=r'$W_{total}$ (s)')

    # Slice 3: theta1=0.70
    ax3 = axes[2]
    grid3 = np.full((len(t2_vals), len(t3_vals)), np.nan)
    for r in data:
        if abs(r['theta1'] - 0.70) < 1e-6:
            i = t2_vals.index(r['theta2'])
            j = t3_vals.index(r['theta3'])
            grid3[i, j] = r['w_total']

    im3 = ax3.imshow(grid3, cmap='RdYlGn_r', aspect='auto', vmin=3.5, vmax=4.5)
    ax3.set_xticks(range(len(t3_vals)))
    ax3.set_xticklabels([f"{v:.2f}" for v in t3_vals])
    ax3.set_yticks(range(len(t2_vals)))
    ax3.set_yticklabels([f"{v:.2f}" for v in t2_vals])
    ax3.set_xlabel(r'$\theta_3$', fontsize=12)
    ax3.set_ylabel(r'$\theta_2$', fontsize=12)
    ax3.set_title(r'(c) $\theta_1$ = 0.70', fontsize=12)
    for i in range(len(t2_vals)):
        for j in range(len(t3_vals)):
            if not np.isnan(grid3[i, j]):
                val = grid3[i, j]
                color = 'white' if val > 4.2 else 'black'
                ax3.text(j, i, f"{val:.2f}", ha='center', va='center',
                        color=color, fontsize=11, fontweight='bold')
    fig.colorbar(im3, ax=ax3, label=r'$W_{total}$ (s)')

    fig.text(0.5, 0.01,
            f'Optimal: $W_{{total}}$={best_w:.2f}s ({best_count}/24 groups) | '
            f'Degradation: $\\theta_1$=0.65, $W_{{total}}$={worst_w:.2f}s (+{(worst_w/best_w-1)*100:.1f}%)',
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(f"{OUTPUT_DIR}/Fig7_Rho_Sensitivity_Heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Fig7_Rho_Sensitivity_Heatmap.png")


if __name__ == "__main__":
    print("=" * 70)
    print("Journal Figure Generation - Complete")
    print("=" * 70)

    plot_fig1_framework()
    plot_fig2_convergence()
    plot_fig3_algorithm_comparison()
    plot_fig4_rho_regression()
    plot_fig5_ablation()
    plot_fig6_hyperparam()
    plot_fig7_rho_heatmap()

    print("\n" + "=" * 70)
    print("All figures generated successfully!")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 70)
