#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Draw convergence curve v3: Three-level comparison
Swap (baseline) -> +BM (accelerated) -> +RL (dynamic tuning)
Shows progressive improvement with each component
"""
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

with open('../../paper_revision/15_visualizations/raw_convergence_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

swap = data['NSGA-II (Swap)']
bm = data['NSGA-II + BM']
rl = data['RL-NSGA-II + BM + 9-dim']

gens = [d['generation'] for d in swap]
swap_w = [d['w_total'] for d in swap]
bm_w = [d['w_total'] for d in bm]
rl_w = [d['w_total'] for d in rl]

fig, ax = plt.subplots(figsize=(11, 6.5))

# Plot with different line styles
ax.plot(gens, swap_w, 'o--', color='#95A5A6', linewidth=2, markersize=4, label='NSGA-II (Swap only)', alpha=0.7)
ax.plot(gens, bm_w, 's-', color='#E67E22', linewidth=2.5, markersize=5, label='NSGA-II + BM', alpha=0.85)
ax.plot(gens, rl_w, '^-', color='#2980B9', linewidth=2.5, markersize=5, label='RL-NSGA-II + BM + 9-dim', alpha=0.85)

# Highlight key milestones
ax.annotate('BM achieves\nW=0.06 at Gen 16', xy=(16, bm_w[15]), xytext=(10, 2.5),
            arrowprops=dict(arrowstyle='->', color='#E67E22', lw=1.5),
            fontsize=10, color='#E67E22', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#E67E22', alpha=0.8))

ax.annotate('RL finds W=0.79\nat Gen 14 (ahead of BM)', xy=(14, rl_w[13]), xytext=(20, 2.0),
            arrowprops=dict(arrowstyle='->', color='#2980B9', lw=1.5),
            fontsize=10, color='#2980B9', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#2980B9', alpha=0.8))

# Add convergence threshold line
ax.axhline(y=0.06, color='green', linestyle=':', linewidth=1.5, alpha=0.6, label='Convergence threshold (W=0.06)')

# Formatting
ax.set_xlabel('Generation', fontsize=14, fontweight='bold')
ax.set_ylabel('W_total (s)', fontsize=14, fontweight='bold')
ax.set_title('Convergence Analysis: Component-wise Progressive Improvement\n(R2 / 600 tasks / sync mode / pop=30 / gen=30)',
             fontsize=15, fontweight='bold')
ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
ax.grid(True, alpha=0.25, linestyle='--')
ax.set_xlim(1, 30)
ax.set_ylim(0, 5.5)

# Text box with key takeaways
textstr = 'Key Takeaways:\n'
textstr += f'• Baseline (Swap): W={swap_w[-1]:.2f}s at Gen 30\n'
textstr += f'• +BM: Reaches W=0.06 at Gen 16\n'
textstr += f'• +RL: Finds W=0.79 at Gen 14 (earlier)\n'
textstr += f'• Both +BM and +RL converge to W=0.06'
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.85, edgecolor='gray')
ax.text(0.02, 0.55, textstr, transform=ax.transAxes, fontsize=10, bbox=props)

plt.tight_layout()
plt.savefig('../../results/journal/figures_new/FigX_Convergence_Analysis.png', dpi=300, bbox_inches='tight')
plt.savefig('../../results/journal/figures_new/FigX_Convergence_Analysis.pdf', bbox_inches='tight')
print('Saved: FigX_Convergence_Analysis.png/.pdf')
plt.savefig('../../FigX_Convergence_Analysis.png', dpi=300, bbox_inches='tight')
print('Saved: FigX_Convergence_Analysis.png (root)')
plt.close()

print('\nConvergence milestones:')
print(f'  Swap:     Gen 7 W={swap_w[6]:.3f}, Gen 30 W={swap_w[-1]:.3f}')
print(f'  +BM:      Gen 10 W={bm_w[9]:.3f}, Gen 16 W={bm_w[15]:.3f}, Gen 30 W={bm_w[-1]:.3f}')
print(f'  +RL:      Gen 10 W={rl_w[9]:.3f}, Gen 14 W={rl_w[13]:.3f}, Gen 30 W={rl_w[-1]:.3f}')
