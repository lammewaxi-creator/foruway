"""
physics_utils.py - 物理引擎
职责：纯粹的数学计算器
核心逻辑：实现梯形/三角形速度规划
输入：距离 (S)、设备类型
输出：时间 (t)
细节：不关心任务是谁，只关心"跑这段路要多久"
必须区分空载（快）和负载（慢）
"""
import math
import re
from functools import lru_cache
from src.config import *

def calculate_time(distance, v_max, acc):
    """
    计算梯形/三角形速度规划下的运动时间
    
    物理原理：
    - 加速阶段：从0加速到v_max，时间 t_acc = v_max / acc
    - 匀速阶段：如果距离足够长，会有一段匀速运动
    - 减速阶段：从v_max减速到0，时间 t_dec = v_max / acc
    
    参数：
        distance: 移动距离 (m)
        v_max: 最大速度 (m/s)
        acc: 加速度 (m/s^2)
    
    返回：
        时间 (s)
    """
    if distance <= 0:
        return 0.0
        
    # 1. 计算加速到 v_max 所需的距离和时间
    t_acc = v_max / acc
    d_acc = 0.5 * acc * (t_acc ** 2)
    
    # 2. 判断是否能达到最大速度
    if distance >= 2 * d_acc:
        # 梯形速度曲线 (加速 -> 匀速 -> 减速)
        d_const = distance - 2 * d_acc
        t_const = d_const / v_max
        t = 2 * t_acc + t_const
    else:
        # 三角形速度曲线 (加速 -> 减速，未达 v_max)
        # distance = 2 * (0.5 * acc * t_half^2)
        # t_total = 2 * t_half
        t = 2 * math.sqrt(distance / acc)
    
    return t

def get_stage_duration(dist_x, dist_y, dist_z, device_type, is_loaded=False):
    """
    统一获取工序持续时间
    
    这是物理引擎的核心接口，被 utils.py 在数据加载时调用
    目的是预计算所有工序的耗时，避免在仿真循环中重复计算
    
    参数：
        dist_x: X轴距离 (m) - Row方向
        dist_y: Y轴距离 (m) - Col方向
        dist_z: Z轴距离 (m) - 垂直方向
        device_type: 'FRGV' (四向车) 或 'Lift' (提升机)
        is_loaded: 是否负载 (影响提升机速度)
    
    返回：
        持续时间 (s)
    """
    total_time = 0.0
    
    if device_type == 'Lift':
        # 提升机：主要看垂直距离
        # 根据负载情况选择速度
        v_lift = LIFT_V_LOADED if is_loaded else LIFT_V_EMPTY
        total_time = calculate_time(dist_z, v_lift, LIFT_ACC)
        
    elif device_type == 'FRGV':
        # 四向车：曼哈顿距离 + L型转向惩罚
        # 四向车需要停车换向，故X和Y方向的时间叠加计算
        t_x = calculate_time(dist_x, RGV_V_MAX, RGV_ACC)
        t_y = calculate_time(dist_y, RGV_V_MAX, RGV_ACC)
        total_time = t_x + t_y

        # L型行驶换向惩罚：当X和Y方向都有位移时，添加3秒转向惩罚
        if dist_x > 0 and dist_y > 0:
            total_time += 3.0
    
    # 返回时间，最小给0.1s防止除零错误
    return max(total_time, 0.1)

def parse_node_str(node_str):
    """
    解析节点字符串，返回内部标准的 (layer, row, col) 坐标
    
    CSV格式: 04-032-008 -> Col-Row-Layer
    - 第一部分 (04): Col (列/Y)
    - 第二部分 (032): Row (行/X)
    - 第三部分 (008): Layer (层/Z)
    
    内部标准: (Layer, Row, Col)
    
    参数：
        node_str: 节点字符串
    
    返回：
        (Layer, Row, Col) 元组
    """
    return _parse_node_str_cached(node_str)


@lru_cache(maxsize=2048)
def _parse_node_str_cached(node_str):
    """
    解析节点字符串 (LRU 缓存优化版)
    
    使用 LRU 缓存避免重复解析相同的节点字符串，
    因为仓库节点数量有限，重复解析是常见操作。
    """
    if not isinstance(node_str, str):
        return (1, 1, 1)  # 默认值
    
    # 使用正则表达式匹配
    match = re.match(r'(\d+)-(\d+)-(\d+)', node_str)
    if match:
        # 根据CSV格式: Col, Row, Layer
        c_str, r_str, l_str = match.groups()
        
        c = int(c_str)
        r = int(r_str)
        l = int(l_str)
        
        return (l, r, c)  # 返回内部标准顺序 (Layer, Row, Col)
    
    return (1, 1, 1)

def calculate_physics(from_node, to_node, device_type):
    """
    计算两个节点间的移动时间（空载）
    
    这是仿真时动态调用的函数，用于计算设备空驶时间
    
    参数：
        from_node: 起始节点字符串
        to_node: 目标节点字符串
        device_type: 设备类型 ('FRGV' 或 'Lift')
    
    返回：
        移动时间 (s)
    """
    # 解析坐标
    c1 = parse_node_str(from_node)
    c2 = parse_node_str(to_node)
    
    # 转换为物理距离
    dist_z = abs(c1[0] - c2[0]) * LAYER_HEIGHT
    dist_x = abs(c1[1] - c2[1]) * CELL_LENGTH
    dist_y = abs(c1[2] - c2[2]) * CELL_WIDTH
    
    # 空载计算（is_loaded=False）
    return get_stage_duration(dist_x, dist_y, dist_z, device_type, is_loaded=False)
