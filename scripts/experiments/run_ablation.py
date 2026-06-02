"""
run_ablation.py - 消融实验主脚本

执行六个算法变体（M0-M5）的对比实验：
- M0: FIFO 先到先服务 (最简Baseline)
- M1: 标准 NSGA-II (Baseline)
- M2: NSGA-II + 时间窗左插入解码 (TW)
- M3: NSGA-II + TW + 关键块邻域搜索变异 (BM)
- M4: RL-NSGA-II 纯数学状态版
- M5: RL-NSGA-II 数学+物理双驱状态版 (Proposed)

实验设计：
1. 对每个变体运行多次（默认10次），取统计平均值
2. 记录 Cmax, W_total, P95, 运行时间等指标
3. 生成消融实验结果图表

【重要修改】
本实验在 adaptive_rho 模式下进行，以评估各算子在 ρ 感知框架内的边际贡献。
"""

import os
import sys
import json
import random
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.simulation import Simulator
from src.utils import load_jobs_from_csv, build_resources_from_csv
from src.ablation_variants import (
    M0_FIFO,
    M1_BaseNSGA2,
    M1b_BaseWithBM,
    M2_TimewindowNSGA2,
    M3_CriticalBlockNSGA2,
    M4_RL_MathOnly,
    M5_RL_PhysicalMath
)
# HV calculation disabled (module not available)
def calculate_hv_for_solutions(solutions):
    return 0.0


def select_balanced_solution(solutions, cmax_tolerance=0.005):
    """
    选择Pareto前沿上更"平衡"的解。
    策略：在Cmax不超过最小值(1+tolerance)的解中，选择W_total最小的。
    """
    if not solutions:
        return None
    min_cmax = min(s['fitness'][1] for s in solutions)
    threshold = min_cmax * (1 + cmax_tolerance)
    candidates = [s for s in solutions if s['fitness'][1] <= threshold]
    return min(candidates, key=lambda x: x['fitness'][0])


def run_single_variant(variant_class, variant_name, simulator, num_jobs,
                       pop_size=30, n_gen=30, data_name=""):
    """
    运行单个变体的一次实验
    
    参数：
        variant_class: 变体类
        variant_name: 变体名称
        simulator: 仿真器
        num_jobs: 任务数量
        pop_size: 种群大小
        n_gen: 迭代代数
        data_name: 数据集名称
    
    返回：
        result_dict: 包含实验结果的字典
    """
    print(f"\n{'='*60}")
    print(f"运行 {variant_name} - {data_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # 创建优化器实例
    optimizer = variant_class(simulator, num_jobs, pop_size=pop_size, n_gen=n_gen)
    
    # 运行优化（FIFO特殊处理）
    if variant_name == "M0_FIFO":
        best_sol, (final_cmax, final_w_total, p95) = optimizer.optimize()
        best_solutions = [best_sol]  # FIFO只有一个解
        best_cmax = final_cmax
        best_w_total = final_w_total
    else:
        evolve_result = optimizer.evolve()

        # 处理两种返回格式：
        # - NSGA2Optimizer.evolve() 返回 dict: {'solutions': [...], ...}
        # - ablation_variants 的 evolve() 返回 list: [individual, ...]
        if isinstance(evolve_result, dict) and 'solutions' in evolve_result:
            best_solutions = evolve_result['solutions']
        elif isinstance(evolve_result, list):
            best_solutions = evolve_result
        else:
            best_solutions = []

        # 获取最优解的指标（使用平衡选择，避免Cmax极小但W_total极大的极端解）
        if best_solutions:
            best_sol = select_balanced_solution(best_solutions, cmax_tolerance=0.005)
            best_cmax = best_sol['fitness'][1]
            best_w_total = best_sol['fitness'][0]

            # 运行最终仿真获取完整指标
            simulator.reset()
            final_cmax, final_w_total, p95 = simulator.run(best_sol['genome'],
                                                           use_time_window_insertion=True)
        else:
            best_cmax = float('inf')
            best_w_total = float('inf')
            final_cmax = float('inf')
            final_w_total = float('inf')
            p95 = float('inf')
    
    solve_time = time.time() - start_time
    
    # 计算帕累托前沿的超体积 (HV)
    hv_value = 0.0
    if best_solutions:
        hv_value = calculate_hv_for_solutions(best_solutions)
    
    result = {
        'variant': variant_name,
        'data': data_name,
        'cmax': final_cmax,
        'w_total': final_w_total,
        'p95': p95,
        'hv': hv_value,
        'solve_time': solve_time,
        'generations': n_gen
    }
    
    print(f"\n{variant_name} 结果:")
    print(f"  Cmax: {final_cmax:.2f}s")
    print(f"  W_total: {final_w_total:.2f}s")
    print(f"  P95: {p95:.2f}s")
    print(f"  HV (超体积): {hv_value:.2f}")
    print(f"  求解时间: {solve_time:.2f}s")
    
    return result


def run_ablation_experiment(data_file, data_name, n_runs=5, pop_size=30, n_gen=30):
    """
    在一个数据集上运行完整的消融实验
    
    参数：
        data_file: 数据文件路径
        data_name: 数据集名称
        n_runs: 每个变体运行次数
        pop_size: 种群大小
        n_gen: 迭代代数
    
    返回：
        results_list: 所有实验结果列表
    """
    print(f"\n{'#'*70}")
    print(f"# 消融实验 - {data_name}")
    print(f"# 数据文件: {data_file}")
    print(f"# 运行次数: {n_runs}, 种群大小: {pop_size}, 迭代代数: {n_gen}")
    print(f"{'#'*70}\n")
    
    # 加载数据
    jobs = load_jobs_from_csv(data_file)
    resources = build_resources_from_csv(data_file)
    num_jobs = len(jobs)

    # 混沌洗牌（与主实验一致，消除初始序列偏置）
    original_release_times = sorted([job.release_time for job in jobs])
    random.seed(42)
    random.shuffle(jobs)
    for i, job in enumerate(jobs):
        job.release_time = original_release_times[i]
    print(f"[数据预处理] 已对 {data_file} 进行混沌洗牌")

    print(f"加载了 {num_jobs} 个任务，{len(resources)} 个资源")
    
    # 定义变体 (M0为FIFO，只运行一次)
    variants = [
        (M0_FIFO, "M0_FIFO"),
        (M1_BaseNSGA2, "M1_BaseNSGA2"),
        (M1b_BaseWithBM, "M1b_BaseWithBM"),
        (M2_TimewindowNSGA2, "M2_TimewindowNSGA2"),
        (M3_CriticalBlockNSGA2, "M3_CriticalBlockNSGA2"),
        (M4_RL_MathOnly, "M4_RL_MathOnly"),
        (M5_RL_PhysicalMath, "M5_RL_PhysicalMath")
    ]
    
    results = []
    
    # 资源分配模式：使用 adaptive_rho 模式进行消融实验
    # 这样可以验证各算子在 ρ 感知框架内的边际贡献
    resource_mode = 'adaptive_rho'
    
    print(f"\n【资源分配模式】: {resource_mode}")
    print(f"【实验目的】: 在 ρ 感知框架内评估各进化算子的边际贡献")
    print(f"{'='*70}")
    
    # 首先运行FIFO (M0) 一次（确定性算法）
    print(f"\n{'='*70}")
    print(f"运行 M0_FIFO (确定性算法，只运行1次)")
    print(f"{'='*70}")
    simulator = Simulator(jobs, resources, mode=resource_mode)
    result = run_single_variant(
        M0_FIFO, "M0_FIFO", simulator, num_jobs,
        pop_size=pop_size, n_gen=n_gen, data_name=data_name
    )
    results.append(result)
    
    # 然后运行其他变体多次
    for run in range(n_runs):
        print(f"\n{'='*70}")
        print(f"第 {run + 1}/{n_runs} 次运行 (M1-M5)")
        print(f"{'='*70}")

        for variant_class, variant_name in variants[1:]:  # 跳过M0
            # 【重要】每个变体使用独立的随机种子，用变体名称哈希确保种子稳定，
            # 添加/删除变体不会影响其他变体的结果
            variant_seed = hash(f"ablation_{run}_{variant_name}") % (2**31)
            random.seed(variant_seed)
            np.random.seed(variant_seed)

            # 为每次运行创建新的仿真器副本（使用 adaptive_rho 模式）
            simulator = Simulator(jobs, resources, mode=resource_mode)

            result = run_single_variant(
                variant_class, variant_name, simulator, num_jobs,
                pop_size=pop_size, n_gen=n_gen, data_name=data_name
            )
            results.append(result)
    
    return results


def analyze_results(results, output_dir="results/ablation"):
    """
    分析消融实验结果并生成报告
    
    参数：
        results: 实验结果列表
        output_dir: 输出目录
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 转换为 DataFrame
    df = pd.DataFrame(results)
    
    # 按变体分组统计
    summary = df.groupby('variant').agg({
        'cmax': ['mean', 'std', 'min'],
        'w_total': ['mean', 'std', 'min'],
        'p95': ['mean', 'std'],
        'hv': ['mean', 'std', 'max'],
        'solve_time': ['mean', 'std']
    }).round(2)
    
    # 保存详细结果
    df.to_csv(f"{output_dir}/ablation_raw_results.csv", index=False)
    summary.to_csv(f"{output_dir}/ablation_summary.csv")
    
    # 生成消融实验报告
    report = []
    report.append("="*80)
    report.append("消融实验结果报告 - ρ感知框架内消融")
    report.append("="*80)
    report.append(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"资源分配模式: adaptive_rho (ρ感知自适应)")
    report.append("")
    
    report.append("【消融实验设计（在 ρ 感知框架内）】")
    report.append("M0 (FIFO): 先到先服务 + ρ感知调度")
    report.append("M1 (Base): 标准 NSGA-II + ρ感知")
    report.append("M1b (+BM): 标准 NSGA-II + BM 关键块变异")
    report.append("M2 (+TW):  + 时间窗左插入解码")
    report.append("M3 (+BM+TW): + 关键块变异 + 时间窗插入")
    report.append("M4 (Math): + RL 动态调参（纯数学状态）")
    report.append("M5 (Full): + RL 动态调参（数学+物理双驱）")
    report.append("")
    report.append("【实验目的】")
    report.append("在 ρ 感知自适应框架内，评估各进化算子（TW、BM、RL）的边际贡献")
    report.append("")
    
    report.append("【性能对比（均值±标准差）】")
    report.append("-"*80)
    report.append(f"{'变体':<18} {'Cmax (s)':<18} {'W_total (s)':<18} {'HV':<15} {'时间 (s)':<12}")
    report.append("-"*80)
    
    # 首先添加M0
    if 'M0_FIFO' in summary.index:
        row = summary.loc['M0_FIFO']
        cmax_str = f"{row['cmax']['mean']:.2f} (地板)"
        w_str = f"{row['w_total']['mean']:.2f} (地板)"
        hv_str = "N/A"
        time_str = f"{row['solve_time']['mean']:.2f}"
        report.append(f"{'M0_FIFO':<18} {cmax_str:<18} {w_str:<18} {hv_str:<15} {time_str:<12}")
    
    for variant in ['M1_BaseNSGA2', 'M1b_BaseWithBM', 'M2_TimewindowNSGA2', 'M3_CriticalBlockNSGA2',
                    'M4_RL_MathOnly', 'M5_RL_PhysicalMath']:
        if variant in summary.index:
            row = summary.loc[variant]
            cmax_str = f"{row['cmax']['mean']:.2f}±{row['cmax']['std']:.2f}"
            w_str = f"{row['w_total']['mean']:.2f}±{row['w_total']['std']:.2f}"
            hv_str = f"{row['hv']['mean']:.2e}±{row['hv']['std']:.2e}"
            time_str = f"{row['solve_time']['mean']:.2f}±{row['solve_time']['std']:.2f}"
            report.append(f"{variant:<15} {cmax_str:<18} {w_str:<18} {hv_str:<15} {time_str:<12}")
    
    report.append("-"*80)
    report.append("")
    
    # 计算改进幅度
    report.append("【各模块贡献分析】")
    report.append("-"*80)
    
    base_cmax = summary.loc['M1_BaseNSGA2', ('cmax', 'mean')]
    base_w = summary.loc['M1_BaseNSGA2', ('w_total', 'mean')]
    
    # 首先显示M0相对于M1的表现（展示遗传算法的价值）
    if 'M0_FIFO' in summary.index:
        fifo_cmax = summary.loc['M0_FIFO', ('cmax', 'mean')]
        fifo_w = summary.loc['M0_FIFO', ('w_total', 'mean')]
        fifo_cmax_gap = (fifo_cmax - base_cmax) / fifo_cmax * 100
        fifo_w_gap = (fifo_w - base_w) / fifo_w * 100
        report.append(f"M0_FIFO vs M1_BaseNSGA2 (遗传算法搜索价值):")
        report.append(f"  Cmax: M1比M0优 {fifo_cmax_gap:.2f}%")
        report.append(f"  W_total: M1比M0优 {fifo_w_gap:.2f}%")
        report.append("")
    
    report.append("各改进模块相对于M1的贡献:")
    report.append("-"*40)
    for variant in ['M1b_BaseWithBM', 'M2_TimewindowNSGA2', 'M3_CriticalBlockNSGA2',
                    'M4_RL_MathOnly', 'M5_RL_PhysicalMath']:
        if variant in summary.index:
            cmax = summary.loc[variant, ('cmax', 'mean')]
            w = summary.loc[variant, ('w_total', 'mean')]
            
            cmax_improve = (base_cmax - cmax) / base_cmax * 100
            w_improve = (base_w - w) / base_w * 100
            
            report.append(f"{variant}:")
            report.append(f"  Cmax 改进: {cmax_improve:+.2f}%")
            report.append(f"  W_total 改进: {w_improve:+.2f}%")
            report.append("")
    
    report.append("="*80)
    
    report_text = "\n".join(report)
    print("\n" + report_text)
    
    # 保存报告
    with open(f"{output_dir}/ablation_report.txt", 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    # 生成可视化图表
    generate_plots(df, output_dir)
    
    return summary


def generate_plots(df, output_dir):
    """生成消融实验可视化图表"""
    
    # 包含M0_FIFO作为地板线
    variants = ['M0_FIFO', 'M1_BaseNSGA2', 'M1b_BaseWithBM', 'M2_TimewindowNSGA2', 'M3_CriticalBlockNSGA2',
                'M4_RL_MathOnly', 'M5_RL_PhysicalMath']

    variant_labels = ['M0\n(FIFO)', 'M1\n(Base)', 'M1b\n(+BM)', 'M2\n(+TW)', 'M3\n(+BM+TW)',
                      'M4\n(Math)', 'M5\n(Proposed)']
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Ablation Study Results', fontsize=16, fontweight='bold')
    
    # Cmax 对比
    ax1 = axes[0, 0]
    cmax_data = [df[df['variant'] == v]['cmax'].values for v in variants if v in df['variant'].values]
    cmax_labels = [variant_labels[i] for i, v in enumerate(variants) if v in df['variant'].values]
    bp1 = ax1.boxplot(cmax_data, labels=cmax_labels, patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('lightblue')
    ax1.set_ylabel('Cmax (s)')
    ax1.set_title('Maximum Completion Time')
    ax1.grid(True, alpha=0.3)
    
    # W_total 对比
    ax2 = axes[0, 1]
    w_data = [df[df['variant'] == v]['w_total'].values for v in variants if v in df['variant'].values]
    bp2 = ax2.boxplot(w_data, labels=cmax_labels, patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('lightgreen')
    ax2.set_ylabel('Total Wait Time (s)')
    ax2.set_title('Total Waiting Time')
    ax2.grid(True, alpha=0.3)
    
    # P95 对比
    ax3 = axes[1, 0]
    p95_data = [df[df['variant'] == v]['p95'].values for v in variants if v in df['variant'].values]
    bp3 = ax3.boxplot(p95_data, labels=cmax_labels, patch_artist=True)
    for patch in bp3['boxes']:
        patch.set_facecolor('lightcoral')
    ax3.set_ylabel('P95 Wait Time (s)')
    ax3.set_title('95th Percentile Waiting Time')
    ax3.grid(True, alpha=0.3)
    
    # 求解时间对比
    ax4 = axes[1, 1]
    time_data = [df[df['variant'] == v]['solve_time'].values for v in variants if v in df['variant'].values]
    bp4 = ax4.boxplot(time_data, labels=cmax_labels, patch_artist=True)
    for patch in bp4['boxes']:
        patch.set_facecolor('lightyellow')
    ax4.set_ylabel('Solve Time (s)')
    ax4.set_title('Computational Time')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ablation_boxplots.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 生成超体积对比图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    hv_data = [df[df['variant'] == v]['hv'].values for v in variants if v in df['variant'].values]
    hv_labels = [variant_labels[i] for i, v in enumerate(variants) if v in df['variant'].values]
    
    bp_hv = ax.boxplot(hv_data, labels=hv_labels, patch_artist=True)
    for patch in bp_hv['boxes']:
        patch.set_facecolor('lightcyan')
    
    ax.set_ylabel('Hypervolume (HV)')
    ax.set_title('Pareto Front Quality (Higher is Better)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ablation_hypervolume.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"超体积对比图已保存到 {output_dir}/ablation_hypervolume.png")
    
    # 生成改进幅度柱状图（包含M0作为地板线）
    fig, ax = plt.subplots(figsize=(12, 6))
    
    summary = df.groupby('variant')[['cmax', 'w_total', 'p95', 'solve_time']].mean()
    base_cmax = summary.loc['M1_BaseNSGA2', 'cmax']
    base_w = summary.loc['M1_BaseNSGA2', 'w_total']
    
    improvements_cmax = []
    improvements_w = []
    labels = []
    colors_cmax = []
    colors_w = []
    
    for i, v in enumerate(variants):
        if v in summary.index:
            imp_cmax = (base_cmax - summary.loc[v, 'cmax']) / base_cmax * 100
            imp_w = (base_w - summary.loc[v, 'w_total']) / base_w * 100
            improvements_cmax.append(imp_cmax)
            improvements_w.append(imp_w)
            labels.append(variant_labels[i])
            # M0用红色表示（负改进），其他根据正负用不同颜色
            if v == 'M0_FIFO':
                colors_cmax.append('red')
                colors_w.append('red')
            else:
                colors_cmax.append('skyblue')
                colors_w.append('lightgreen')
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, improvements_cmax, width, label='Cmax Improvement',
                   color=colors_cmax, alpha=0.8)
    bars2 = ax.bar(x + width/2, improvements_w, width, label='W_total Improvement',
                   color=colors_w, alpha=0.8)
    
    ax.set_ylabel('Improvement (%)')
    ax.set_title('Performance Improvement over M1 (Base NSGA-II)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加说明文本
    ax.text(0.02, 0.98, 'Red bars: M0_FIFO (performance floor)\n'
                        'Blue/Green bars: M1-M5 (improvement over M1)',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ablation_improvements.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"图表已保存到 {output_dir}/")


def main():
    """主函数"""
    import sys
    
    # 参数设置（与主实验一致：30/30）
    n_runs = 5
    pop_size = 30
    n_gen = 30
    
    # 解析命令行参数
    if len(sys.argv) > 1:
        n_runs = int(sys.argv[1])
    if len(sys.argv) > 2:
        pop_size = int(sys.argv[2])
    if len(sys.argv) > 3:
        n_gen = int(sys.argv[3])
    
    print("="*80)
    print("消融实验脚本")
    print("="*80)
    print(f"运行参数: 次数={n_runs}, 种群={pop_size}, 代数={n_gen}")
    print("")
    
    all_results = []

    # 在 R2 数据集上运行（与论文消融实验一致）
    print("\n【实验1: R2 高负载】")
    results_r2 = run_ablation_experiment(
        'R2_highest_arrival_rate.csv', 'R2_High',
        n_runs=n_runs, pop_size=pop_size, n_gen=n_gen
    )
    all_results.extend(results_r2)

    # 保存完整结果
    with open('results/ablation/all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # 分析结果
    summary = analyze_results(all_results)
    
    print("\n" + "="*80)
    print("消融实验完成！")
    print("="*80)
    print("结果保存位置:")
    print("  - results/ablation/ablation_raw_results.csv")
    print("  - results/ablation/ablation_summary.csv")
    print("  - results/ablation/ablation_report.txt")
    print("  - results/ablation/ablation_boxplots.png")
    print("  - results/ablation/ablation_improvements.png")
    print("="*80)


if __name__ == "__main__":
    main()
