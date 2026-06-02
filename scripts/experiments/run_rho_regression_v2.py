#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ρ 回归实验 v2 — 使用当前仿真配置

通过改变 lift_count 来调节 ρ 范围，固定 RGV=15。
测试算法：FIFO、NSGA-II、IGA、RL-NSGA-II
"""
import os, sys, time, json, csv, random, statistics
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.simulation import Simulator, Resource
from src.strategies import heuristic_dispatch
from src.optimization import NSGA2Optimizer, IGAOptimizer, RL_NSGA2Optimizer
from src.utils import load_jobs_from_csv
from src.simulation_rho_extension import calculate_rho_metrics
from src.config import LIFT_LOCATIONS, RGV_COUNT


def build_resources(lift_count, rgv_count):
    """根据 lift_count 和 rgv_count 构建资源池"""
    resources = {}
    for i in range(1, lift_count + 1):
        rid = f'Lift{i:02d}'
        if rid in LIFT_LOCATIONS:
            row, col = LIFT_LOCATIONS[rid]
            initial_node = f"{col:02d}-{row:03d}-01"
        else:
            initial_node = '01-001-01'
        resources[rid] = Resource(rid, 'Lift', initial_node)

    rgv_per_lift = rgv_count // lift_count
    remainder = rgv_count % lift_count
    rgv_idx = 1
    for lift_idx in range(1, lift_count + 1):
        rid = f'Lift{lift_idx:02d}'
        if rid in LIFT_LOCATIONS:
            lift_row, lift_col = LIFT_LOCATIONS[rid]
            count = rgv_per_lift + (1 if lift_idx <= remainder else 0)
            for i in range(count):
                rgv_id = f'FRGV{rgv_idx:02d}'
                rgv_row = lift_row + (i + 1)
                rgv_col = lift_col
                initial_node = f"{rgv_col:02d}-{rgv_row:03d}-01"
                resources[rgv_id] = Resource(rgv_id, 'FRGV', initial_node)
                rgv_idx += 1
    return resources


def select_balanced_solution(solutions, cmax_tolerance=0.005):
    """在Cmax容忍范围内选择W_total最小的解"""
    if not solutions:
        return None
    min_cmax = min(s['fitness'][1] for s in solutions)
    threshold = min_cmax * (1 + cmax_tolerance)
    candidates = [s for s in solutions if s['fitness'][1] <= threshold]
    return min(candidates, key=lambda x: x['fitness'][0])


def run_algorithm(sim, algorithm, pop_size=30, n_gen=30):
    """运行指定算法并返回结果"""
    num_jobs = len(sim.jobs)

    if algorithm == 'FIFO':
        heuristic_dispatch(sim, strategy='FIFO')
    elif algorithm == 'NSGA-II':
        opt = NSGA2Optimizer(sim, num_jobs, pop_size=pop_size, n_gen=n_gen)
        data = opt.optimize()
        best = select_balanced_solution(data['solutions'])
        sim.reset()
        sim.run([sim.jobs[i].id for i in best['genome']])
    elif algorithm == 'IGA':
        opt = IGAOptimizer(sim, num_jobs, pop_size=pop_size, n_gen=n_gen)
        data = opt.optimize()
        best = select_balanced_solution(data['solutions'])
        sim.reset()
        sim.run([sim.jobs[i].id for i in best['genome']])
    elif algorithm == 'RL-NSGA-II':
        opt = RL_NSGA2Optimizer(sim, num_jobs, pop_size=pop_size, n_gen=n_gen)
        data = opt.optimize()
        best = select_balanced_solution(data['solutions'])
        sim.reset()
        sim.run([sim.jobs[i].id for i in best['genome']])

    # 计算指标
    cmax = max(j.finish_times[-1] for j in sim.jobs if j.finish_times)

    total_wait = 0.0
    for job in sim.jobs:
        for s_idx, stage in enumerate(job.stages):
            if stage['type'] == 'Lift' and s_idx > 0:
                if s_idx - 1 < len(job.finish_times) and s_idx < len(job.start_times):
                    rgv_finish = job.finish_times[s_idx - 1]
                    lift_start = job.start_times[s_idx]
                    total_wait += abs(lift_start - rgv_finish)

    rho_info = calculate_rho_metrics(sim, [j.id for j in sim.jobs], cmax=cmax)

    return {
        'cmax': cmax,
        'w_total': total_wait,
        'rho_global': rho_info['rho_global'],
        'rho_cv': rho_info['rho_cv'],
        'lift_count': rho_info['lift_count'],
        'total_handoffs': rho_info['total_handoffs'],
        'avg_service_time': rho_info['avg_service_time'],
    }


def main():
    JOBS_FILE = "R2_highest_arrival_rate.csv"
    jobs = load_jobs_from_csv(JOBS_FILE)
    print(f"Dataset: {JOBS_FILE}, {len(jobs)} jobs")

    # 参数配置
    LIFT_COUNTS = [1, 2, 3, 4, 6]
    ALGORITHMS = ['FIFO', 'NSGA-II', 'IGA', 'RL-NSGA-II']
    N_SEEDS = 3
    POP_SIZE = 30
    N_GEN = 30
    RGV_COUNT = 15

    os.makedirs('results/rho_regression_v2', exist_ok=True)

    all_results = []
    csv_path = 'results/rho_regression_v2/rho_regression_v2.csv'

    # 写入CSV头
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'lift_count', 'algorithm', 'seed', 'cmax', 'w_total',
            'rho_global', 'rho_cv', 'total_handoffs', 'avg_service_time'
        ])

    total_runs = len(LIFT_COUNTS) * len(ALGORITHMS) * N_SEEDS
    run_idx = 0

    for lift_count in LIFT_COUNTS:
        for algorithm in ALGORITHMS:
            for seed in range(N_SEEDS):
                run_idx += 1
                print(f"\n[{run_idx}/{total_runs}] lift={lift_count}, algo={algorithm}, seed={seed}")

                # 混沌洗牌（不同种子不同顺序）
                random.seed(seed)
                shuffled_jobs = list(jobs)
                original_release = sorted([j.release_time for j in shuffled_jobs])
                random.shuffle(shuffled_jobs)
                for i, job in enumerate(shuffled_jobs):
                    job.release_time = original_release[i]

                resources = build_resources(lift_count, RGV_COUNT)
                sim = Simulator(shuffled_jobs, resource_map=resources, mode='decoupled')

                t0 = time.time()
                result = run_algorithm(sim, algorithm, pop_size=POP_SIZE, n_gen=N_GEN)
                elapsed = time.time() - t0

                row = {
                    'lift_count': lift_count,
                    'algorithm': algorithm,
                    'seed': seed,
                    'cmax': result['cmax'],
                    'w_total': result['w_total'],
                    'rho_global': result['rho_global'],
                    'rho_cv': result['rho_cv'],
                    'total_handoffs': result['total_handoffs'],
                    'avg_service_time': result['avg_service_time'],
                    'solve_time': elapsed,
                }
                all_results.append(row)

                # 实时写入CSV
                with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        row['lift_count'], row['algorithm'], row['seed'],
                        f"{row['cmax']:.2f}", f"{row['w_total']:.2f}",
                        f"{row['rho_global']:.4f}", f"{row['rho_cv']:.4f}",
                        row['total_handoffs'], f"{row['avg_service_time']:.4f}"
                    ])

                print(f"  -> Cmax={result['cmax']:.2f}, W={result['w_total']:.2f}, "
                      f"rho={result['rho_global']:.4f}, time={elapsed:.1f}s")

    # 生成汇总统计
    summary = []
    for lift_count in LIFT_COUNTS:
        for algorithm in ALGORITHMS:
            subset = [r for r in all_results
                      if r['lift_count'] == lift_count and r['algorithm'] == algorithm]
            if subset:
                summary.append({
                    'lift_count': lift_count,
                    'algorithm': algorithm,
                    'rho_mean': statistics.mean(r['rho_global'] for r in subset),
                    'w_mean': statistics.mean(r['w_total'] for r in subset),
                    'w_std': statistics.stdev(r['w_total'] for r in subset) if len(subset) > 1 else 0,
                    'cmax_mean': statistics.mean(r['cmax'] for r in subset),
                    'n': len(subset),
                })

    df_summary = pd.DataFrame(summary)
    df_summary.to_csv('results/rho_regression_v2/rho_summary.csv', index=False)
    print("\n=== Summary ===")
    print(df_summary.to_string(index=False))

    # 生成回归图
    generate_regression_plot(all_results)
    print("\nDone. Results saved to results/rho_regression_v2/")


def generate_regression_plot(all_results):
    """生成 ρ-W_total 回归图"""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {'FIFO': '#d62728', 'NSGA-II': '#2ca02c',
              'IGA': '#ff7f0e', 'RL-NSGA-II': '#1f77b4'}
    markers = {'FIFO': 'o', 'NSGA-II': 's', 'IGA': '^', 'RL-NSGA-II': 'D'}

    # 按算法分组聚合
    from collections import defaultdict
    algo_data = defaultdict(lambda: {'rho': [], 'w': []})

    for r in all_results:
        algo_data[r['algorithm']]['rho'].append(r['rho_global'])
        algo_data[r['algorithm']]['w'].append(r['w_total'])

    # 绘制散点 + 回归线
    for algo in ['FIFO', 'NSGA-II', 'IGA', 'RL-NSGA-II']:
        data = algo_data[algo]
        rho_arr = np.array(data['rho'])
        w_arr = np.array(data['w'])

        # 散点
        ax.scatter(rho_arr, w_arr, c=colors[algo], marker=markers[algo],
                   s=60, alpha=0.6, label=algo, edgecolors='white', linewidth=0.5)

        # 线性回归
        if len(rho_arr) > 1:
            z = np.polyfit(rho_arr, w_arr, 1)
            p = np.poly1d(z)
            rho_sorted = np.sort(rho_arr)
            ax.plot(rho_sorted, p(rho_sorted), '--', c=colors[algo], linewidth=1.5, alpha=0.8)

            # 计算R²
            w_pred = p(rho_arr)
            ss_res = np.sum((w_arr - w_pred) ** 2)
            ss_tot = np.sum((w_arr - np.mean(w_arr)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            print(f"  {algo}: slope={z[0]:.1f}, R²={r2:.3f}")

    ax.set_xlabel(r'System Congestion Factor $\rho$', fontsize=12)
    ax.set_ylabel(r'Total Waiting Time $W_{\mathrm{total}}$ (s)', fontsize=12)
    ax.set_title(r'$W_{\mathrm{total}}$ vs. $\rho$ Regression (R2 Dataset, Decoupled Mode)',
                 fontsize=13)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig('results/rho_regression_v2/rho_regression_v2.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 同时生成对数坐标版本
    fig, ax = plt.subplots(figsize=(8, 6))
    for algo in ['FIFO', 'NSGA-II', 'IGA', 'RL-NSGA-II']:
        data = algo_data[algo]
        rho_arr = np.array(data['rho'])
        w_arr = np.array(data['w'])
        ax.scatter(rho_arr, w_arr, c=colors[algo], marker=markers[algo],
                   s=60, alpha=0.6, label=algo, edgecolors='white', linewidth=0.5)
        if len(rho_arr) > 1:
            z = np.polyfit(rho_arr, w_arr, 1)
            p = np.poly1d(z)
            rho_sorted = np.sort(rho_arr)
            ax.plot(rho_sorted, p(rho_sorted), '--', c=colors[algo], linewidth=1.5, alpha=0.8)

    ax.set_xlabel(r'System Congestion Factor $\rho$', fontsize=12)
    ax.set_ylabel(r'Total Waiting Time $W_{\mathrm{total}}$ (s)', fontsize=12)
    ax.set_title(r'$W_{\mathrm{total}}$ vs. $\rho$ (Log Scale)', fontsize=13)
    ax.set_yscale('log')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig('results/rho_regression_v2/rho_regression_v2_log.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
