#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主实验脚本：优化JSON文件生成
只保存必要的日志文件用于可视化
"""

import os
import sys
import json
import time
import random
import statistics
import argparse
import numpy as np
import pandas as pd

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# 自定义 JSON 编码器处理 numpy 类型
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

from src.simulation import Simulator
from src.strategies import heuristic_dispatch
from src.optimization import NSGA2Optimizer, RL_NSGA2Optimizer, IGAOptimizer
from src.utils import load_jobs_from_csv


def select_balanced_solution(solutions, cmax_tolerance=0.005):
    """
    选择Pareto前沿上更"平衡"的解。
    策略：在Cmax不超过最小值(1+tolerance)的解中，选择W_total最小的。
    这避免了单纯追求Cmax最小而牺牲W_total的极端解。
    """
    if not solutions:
        return None
    min_cmax = min(s['fitness'][1] for s in solutions)
    threshold = min_cmax * (1 + cmax_tolerance)
    candidates = [s for s in solutions if s['fitness'][1] <= threshold]
    return min(candidates, key=lambda x: x['fitness'][0])


def analyze_solution(simulator, algorithm_name, mode_name):
    """分析调度结果，提取关键指标（使用与Simulator.run()一致的双向同步偏差）"""
    # 提取 Cmax
    job_finish_times = [job.finish_times[-1] for job in simulator.jobs if job.finish_times]
    cmax = max(job_finish_times) if job_finish_times else 0

    # 提取等待时间 — 使用双向同步偏差（与Simulator.run()一致）
    total_wait = 0.0
    for job in simulator.jobs:
        for s_idx, stage in enumerate(job.stages):
            if stage['type'] == 'Lift' and s_idx > 0:
                if s_idx - 1 < len(job.finish_times) and s_idx < len(job.start_times):
                    rgv_finish = job.finish_times[s_idx - 1]
                    lift_start = job.start_times[s_idx]
                    total_wait += abs(lift_start - rgv_finish)
    
    # 提取提升机利用率并计算方差
    lift_utils = []
    device_stats = {}
    
    for r_id, resource in simulator.resources.items():
        utilization = (resource.total_busy_time / cmax) * 100 if cmax > 0 else 0
        device_stats[r_id] = {
            'utilization': utilization,
            'processed_count': resource.processed_count
        }
        if resource.type == 'Lift':
            lift_utils.append(utilization)
    
    lift_util_variance = statistics.variance(lift_utils) if len(lift_utils) > 1 else 0
    lift_util_range = max(lift_utils) - min(lift_utils) if lift_utils else 0
    
    return {
        'cmax': cmax,
        'total_wait': total_wait,
        'lift_util_variance': lift_util_variance,
        'lift_util_range': lift_util_range,
        'device_stats': device_stats,
        'execution_log': simulator.execution_log
    }


def run_experiment(jobs_file, algorithm, coupling_mode, pop_size=50, n_gen=50,
                   shuffle_jobs=True, save_gantt=False, data_name="", save_optdata=False):
    """
    运行单组实验
    :param save_gantt: 是否保存甘特图数据（仅用于PPT可视化的关键实验）
    :param save_optdata: 是否保存优化过程数据（仅用于Fig1/2的收敛曲线）
    """
    jobs = load_jobs_from_csv(jobs_file)
    
    # 混沌洗牌（消除初始序列偏置）
    if shuffle_jobs:
        original_release_times = sorted([job.release_time for job in jobs])
        random.seed(42)
        random.shuffle(jobs)
        for i, job in enumerate(jobs):
            job.release_time = original_release_times[i]
        print(f"  [数据预处理] 已对 {jobs_file} 进行混沌洗牌")
    
    # 将耦合模式映射到Simulator支持的模式
    # static_decoupled -> decoupled (静态解耦)
    # adaptive_rho -> adaptive_rho (自适应ρ耦合)
    simulator_mode = 'decoupled' if coupling_mode == 'static_decoupled' else 'adaptive_rho'
    simulator = Simulator(jobs, mode=simulator_mode)
    num_jobs = len(jobs)
    start_time = time.time()
    
    print(f"  >>> 正在运行: {algorithm} + {coupling_mode}")
    
    optimization_data = None
    
    if algorithm == 'FIFO':
        heuristic_dispatch(simulator, strategy='FIFO')
    elif algorithm == 'NSGA-II':
        # NSGA-II 基础配置（不含BM算子，作为对比基线）
        optimizer = NSGA2Optimizer(simulator, num_jobs, pop_size=pop_size, n_gen=n_gen,
                                   use_bm=False)
        optimization_data = optimizer.optimize()
        best_sol = select_balanced_solution(optimization_data['solutions'])
        simulator.reset()
        simulator.run([simulator.jobs[i].id for i in best_sol['genome']])
    elif algorithm == 'IGA':
        # IGA 基础配置（不含BM算子，作为对比基线）
        optimizer = IGAOptimizer(simulator, num_jobs, pop_size=pop_size, n_gen=n_gen)
        optimization_data = optimizer.optimize()
        best_sol = select_balanced_solution(optimization_data['solutions'])
        simulator.reset()
        simulator.run([simulator.jobs[i].id for i in best_sol['genome']])
    elif algorithm == 'RL-NSGA-II':
        # RL-NSGA-II 完整配置：BM算子 + 9维状态空间
        optimizer = RL_NSGA2Optimizer(simulator, num_jobs, pop_size=pop_size, n_gen=n_gen,
                                      use_bm=True, bm_probability=0.3, use_9dim_state=True)
        optimization_data = optimizer.optimize()
        best_sol = select_balanced_solution(optimization_data['solutions'])
        simulator.reset()
        simulator.run([simulator.jobs[i].id for i in best_sol['genome']])
    
    elapsed = time.time() - start_time
    results = analyze_solution(simulator, algorithm, coupling_mode)
    results['solve_time'] = elapsed
    
    # 仅在需要时保存甘特图数据
    if save_gantt and data_name:
        gantt_data = {
            'execution_log': results.get('execution_log', []),
            'device_stats': results.get('device_stats', {})
        }
        # 只保存RL-NSGA-II的结果（代表性算法），避免重复文件
        gantt_filename = f"results/journal/gantt_{data_name}_RL-NSGA-II_{coupling_mode}.json"
        os.makedirs("results/journal", exist_ok=True)
        with open(gantt_filename, 'w', encoding='utf-8') as f:
            json.dump(gantt_data, f, indent=2, ensure_ascii=False)
        print(f"  -> 甘特图数据已保存至: {gantt_filename}")
    
    # 仅在需要时保存优化数据
    if save_optdata and optimization_data and data_name:
        optdata_filename = f"results/journal/optdata_{data_name}_{algorithm}_{coupling_mode}.json"
        os.makedirs("results/journal", exist_ok=True)
        with open(optdata_filename, 'w', encoding='utf-8') as f:
            json.dump(optimization_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        print(f"  -> 优化数据已保存至: {optdata_filename}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="四向车系统动态耦合调度实验")
    parser.add_argument('--dataset', type=str, default='all', 
                       choices=['R1', 'R2', 'all'], 
                       help='指定运行的数据集 (R1, R2 或 all)')
    parser.add_argument('--algo', type=str, default='all', 
                       help='指定运行的算法 (例如 RL-NSGA-II, 或 all)')
    parser.add_argument('--pop', type=int, default=30, help='种群大小')
    parser.add_argument('--gen', type=int, default=30, help='迭代次数')
    parser.add_argument('--save-gantt', action='store_true', 
                       help='保存甘特图数据（用于Fig3/4可视化）')
    parser.add_argument('--save-optdata', action='store_true',
                       help='保存优化过程数据（用于Fig1/2收敛曲线）')
    args = parser.parse_args()

    print("="*60)
    print("基于 ρ 状态感知的四向车动态耦合调度实验")
    print(f"参数设置: 数据集={args.dataset}, 种群={args.pop}, 迭代={args.gen}")
    print(f"保存甘特图: {args.save_gantt}, 保存优化数据: {args.save_optdata}")
    print("="*60)
    
    os.makedirs("results/journal", exist_ok=True)
    
    # 数据集配置
    ALL_DATASETS = {
        'R1_Medium_Load': 'R1_medium_arrival_rate.csv',
        'R2_High_Load': 'R2_highest_arrival_rate.csv'
    }
    
    # 筛选数据集
    if args.dataset == 'R1':
        TARGET_DATASETS = {'R1_Medium_Load': ALL_DATASETS['R1_Medium_Load']}
    elif args.dataset == 'R2':
        TARGET_DATASETS = {'R2_High_Load': ALL_DATASETS['R2_High_Load']}
    else:
        TARGET_DATASETS = ALL_DATASETS
    
    # 筛选算法
    ALL_ALGORITHMS = ['FIFO', 'NSGA-II', 'IGA', 'RL-NSGA-II']
    if args.algo != 'all' and args.algo in ALL_ALGORITHMS:
        TARGET_ALGORITHMS = [args.algo]
    else:
        TARGET_ALGORITHMS = ALL_ALGORITHMS

    MODES = ['static_decoupled', 'adaptive_rho']
    all_results = []
    
    # 运行实验
    for data_name, file_path in TARGET_DATASETS.items():
        print(f"\n{'#'*60}\n开始评估数据集: {data_name}\n{'#'*60}")
        for algo in TARGET_ALGORITHMS:
            for mode in MODES:
                try:
                    res = run_experiment(
                        file_path, algo, mode, 
                        pop_size=args.pop, n_gen=args.gen,
                        data_name=data_name,
                        save_gantt=args.save_gantt,
                        save_optdata=args.save_optdata
                    )
                    all_results.append({
                        'Dataset': data_name,
                        'Algorithm': algo,
                        'Coupling_Mode': mode,
                        'Cmax': res['cmax'],
                        'Total_Wait': res['total_wait'],
                        'Lift_Var': res['lift_util_variance'],
                        'Time': res['solve_time']
                    })
                except Exception as e:
                    print(f"运行失败 {algo}-{mode}: {e}")
                    import traceback
                    traceback.print_exc()
    
    # 输出结果
    df = pd.DataFrame(all_results)
    csv_path = f"results/journal/results_{args.dataset}_{args.algo}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n实验完成！结果已保存至 {csv_path}")
    print(df.to_string(index=False, float_format="%.2f"))


if __name__ == "__main__":
    main()
