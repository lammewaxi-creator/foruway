"""
调度策略模块
功能：FIFO、SPT、LPT、MOR 等基于规则的调度算法
"""
import numpy as np
import random

def baseline_fifo(jobs):
    """
    B1: FIFO（先进先出）
    按release_time排序，返回 Job.id
    """
    sorted_jobs = sorted(enumerate(jobs), key=lambda x: x[1].release_time)
    # 修改：返回 job.id 而不是 index
    return [job.id for _, job in sorted_jobs]

def baseline_greedy_eft(jobs, simulator):
    """
    B2: Greedy EFT（贪心最早完成时间）
    每次选择能最早完成的订单
    """
    scheduled_indices = [] # 存储 index 用于逻辑判断
    scheduled_ids = []     # 存储 id 用于返回
    remaining = set(range(len(jobs))) # 仍然操作 index
    
    while remaining:
        best_job_idx = None
        best_cmax = float('inf')
        
        for job_idx in remaining:
            # 构建临时的 index 序列
            temp_seq_indices = scheduled_indices + [job_idx]
            
            # 将 index 转换为 id 传给 simulator
            temp_seq_ids = [jobs[i].id for i in temp_seq_indices]
            
            cmax, _, _ = simulator.run(temp_seq_ids)
            
            if cmax < best_cmax:
                best_cmax = cmax
                best_job_idx = job_idx
        
        scheduled_indices.append(best_job_idx)
        scheduled_ids.append(jobs[best_job_idx].id) # 存储 ID
        remaining.remove(best_job_idx)
    
    return scheduled_ids # 返回 ID 列表


def spt_dispatch(jobs, simulator):
    """
    SPT (Shortest Processing Time) - 最短加工时间优先
    选择总加工时间最短的任务优先
    """
    # 计算每个任务的总加工时间
    job_processing_times = []
    for job in jobs:
        total_time = sum(stage['duration'] for stage in job.stages)
        job_processing_times.append((job.id, total_time))
    
    # 按加工时间升序排序
    sorted_jobs = sorted(job_processing_times, key=lambda x: x[1])
    return [job_id for job_id, _ in sorted_jobs]


def lpt_dispatch(jobs, simulator):
    """
    LPT (Longest Processing Time) - 最长加工时间优先
    选择总加工时间最长的任务优先
    """
    # 计算每个任务的总加工时间
    job_processing_times = []
    for job in jobs:
        total_time = sum(stage['duration'] for stage in job.stages)
        job_processing_times.append((job.id, total_time))
    
    # 按加工时间降序排序
    sorted_jobs = sorted(job_processing_times, key=lambda x: x[1], reverse=True)
    return [job_id for job_id, _ in sorted_jobs]


def mor_dispatch(jobs, simulator):
    """
    MOR (Most Operations Remaining) - 最多工序数优先
    选择工序数最多的任务优先
    """
    # 计算每个任务的工序数
    job_op_counts = []
    for job in jobs:
        op_count = len(job.stages)
        job_op_counts.append((job.id, op_count))
    
    # 按工序数降序排序
    sorted_jobs = sorted(job_op_counts, key=lambda x: x[1], reverse=True)
    return [job_id for job_id, _ in sorted_jobs]


def heuristic_dispatch(simulator, strategy='FIFO'):
    """
    基于规则的调度算法统一接口
    
    参数:
        simulator: 仿真器对象
        strategy: 调度策略 ('FIFO', 'SPT', 'LPT', 'MOR')
    
    返回:
        调度结果
    """
    jobs = simulator.jobs
    
    if strategy == 'FIFO':
        sequence = baseline_fifo(jobs)
    elif strategy == 'SPT':
        sequence = spt_dispatch(jobs, simulator)
    elif strategy == 'LPT':
        sequence = lpt_dispatch(jobs, simulator)
    elif strategy == 'MOR':
        sequence = mor_dispatch(jobs, simulator)
    else:
        raise ValueError(f"未知的调度策略：{strategy}")
    
    # 运行仿真
    result = simulator.run(sequence)
    return result


# 注意：优化器类 (RL_NSGA2Optimizer, IGAOptimizer, MOEADOptimizer) 已移至 src/optimization.py
# 以避免循环导入问题
