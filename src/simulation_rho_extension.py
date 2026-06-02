"""
simulation_rho_extension.py - ρ 值计算扩展模块

职责：为仿真器提供计算阻塞因子（ρ）的功能

关键概念：
- ρ_global: 全局阻塞因子 = λ / (M × μ)
- ρ_local: 局部阻塞因子（每台提升机）
- ρ_cv: 局部阻塞因子的变异系数
"""

import numpy as np


def calculate_rho_metrics(simulator, sequence_ids, cmax=None):
    """
    计算阻塞因子（ρ）相关指标
    
    参数：
        simulator: Simulator 实例
        sequence_ids: 任务执行序列
        cmax: 最大完成时间（如果为None，会先运行仿真获取）
    
    返回：
        rho_dict: 包含以下指标的字典
            - rho_global: 全局阻塞因子
            - rho_local: 每台提升机的局部阻塞因子字典
            - rho_cv: 局部阻塞因子变异系数
            - lambda_arrival: 任务到达率
            - mu_service: 平均服务率
            - rho_per_lift: 各提升机的ρ值列表
    """
    # 先运行仿真获取 Cmax
    if cmax is None:
        cmax, _, _ = simulator.run(sequence_ids, use_time_window_insertion=False)
    
    # 获取任务数量
    total_jobs = len(simulator.jobs)
    
    # 获取提升机数量
    lift_count = len(simulator.lift_resources)
    
    # 计算总交接次数（提升机服务次数）
    total_handoffs = sum(r.processed_count for r in simulator.lift_resources)
    
    # 计算总服务时间（所有提升机）
    total_service_time = sum(r.total_busy_time for r in simulator.lift_resources)
    
    # 计算平均服务率 μ = 1 / 平均服务时间
    if total_handoffs > 0 and total_service_time > 0:
        mu_service = total_handoffs / total_service_time  # 服务率 (任务/秒)
        avg_service_time = total_service_time / total_handoffs  # 平均服务时间
    else:
        mu_service = 1.0
        avg_service_time = 1.0
    
    # 计算到达率 λ = 总交接次数 / Cmax
    if cmax > 0:
        lambda_arrival = total_handoffs / cmax  # 到达率 (任务/秒)
    else:
        lambda_arrival = 0.0
    
    # 计算全局阻塞因子 ρ_global = λ / (M × μ)
    # 其中 M = 提升机数量
    if lift_count > 0 and mu_service > 0:
        rho_global = lambda_arrival / (lift_count * mu_service)
    else:
        rho_global = 0.0
    
    # 计算每台提升机的局部阻塞因子
    rho_local = {}
    rho_per_lift = []
    
    for lift in simulator.lift_resources:
        # 局部到达率 = 该提升机处理的任务数 / Cmax
        if cmax > 0:
            lambda_i = lift.processed_count / cmax
        else:
            lambda_i = 0.0
        
        # 局部服务率 = 该提升机的处理能力
        if lift.total_busy_time > 0 and lift.processed_count > 0:
            mu_i = lift.processed_count / lift.total_busy_time
        else:
            mu_i = mu_service if mu_service > 0 else 1.0
        
        # 局部阻塞因子 ρ_i = λ_i / μ_i
        if mu_i > 0:
            rho_i = lambda_i / mu_i
        else:
            rho_i = 0.0
        
        rho_local[lift.id] = rho_i
        rho_per_lift.append(rho_i)
    
    # 计算局部阻塞因子的变异系数 (CV)
    if rho_per_lift and len(rho_per_lift) > 1:
        rho_cv = np.std(rho_per_lift) / np.mean(rho_per_lift) if np.mean(rho_per_lift) > 0 else 0.0
    else:
        rho_cv = 0.0
    
    return {
        'rho_global': rho_global,
        'rho_local': rho_local,
        'rho_cv': rho_cv,
        'lambda_arrival': lambda_arrival,
        'mu_service': mu_service,
        'avg_service_time': avg_service_time,
        'rho_per_lift': rho_per_lift,
        'total_handoffs': total_handoffs,
        'lift_count': lift_count,
        'cmax': cmax
    }


def get_rho_category(rho_global):
    """
    根据全局阻塞因子判断系统负载类别
    
    参数：
        rho_global: 全局阻塞因子
    
    返回：
        category: 负载类别字符串
    """
    if rho_global >= 1.0:
        return "超载 (Overloaded)"
    elif rho_global >= 0.9:
        return "高负载 (High Load)"
    elif rho_global >= 0.7:
        return "中高负载 (Medium-High Load)"
    elif rho_global >= 0.5:
        return "中等负载 (Medium Load)"
    elif rho_global >= 0.3:
        return "中低负载 (Medium-Low Load)"
    else:
        return "低负载 (Low Load)"


def get_rho_recommendation(rho_global):
    """
    根据 ρ 值给出优化策略建议
    
    参数：
        rho_global: 全局阻塞因子
    
    返回：
        recommendation: 优化建议字典
    """
    if rho_global >= 0.95:
        return {
            'priority': '等待时间优化',
            'strategy': '减少提升机等待时间',
            'actions': [
                '优先安排高优先级任务',
                '增加任务到达间隔',
                '启用时间窗插入机制',
                '缩短穿梭车准备时间'
            ]
        }
    elif rho_global >= 0.8:
        return {
            'priority': '均衡负载',
            'strategy': '平衡各提升机负载',
            'actions': [
                '动态调整任务分配',
                '优化提升机选择策略',
                '平衡局部阻塞因子',
                '监控瓶颈设备'
            ]
        }
    else:
        return {
            'priority': '整体效率',
            'strategy': '优化整体调度效率',
            'actions': [
                '优化任务顺序',
                '减少空驶时间',
                '提高设备利用率',
                '关注Cmax最小化'
            ]
        }
