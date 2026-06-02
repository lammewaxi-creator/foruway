"""
utils.py - 数据工厂（修正版）
职责：ETL（提取、转换、加载）
将原始的 CSV 脏数据转化为仿真器能理解的 Job 对象

核心逻辑：
1. 坐标解析：把 04-032-008 翻译成 (Layer, Row, Col)
2. 任务类型判断：区分出库、入库、倒库任务
3. 工序定义：正确定义每个任务的工序流程
4. 物理预计算：计算每个工序的耗时

任务流程修正
- 出库任务（货架层 > 1 → 地面层 = 1）：
  工序1：穿梭车从起点取货 → 运到提升机门口（目标层）
  工序2：提升机从目标层 → 降到1层 → 穿梭车卸货

- 入库任务（地面层 = 1 → 货架层 > 1）：
  工序1：穿梭车在1层提升机门口取货 → 坐提升机上升到目标层
  工序2：穿梭车从提升机门口 → 运到目标货架卸货

- 倒库任务（同层或跨层货架间搬运）：
  类似出库/入库，根据具体层判断
"""
import pandas as pd
import numpy as np
import re
from src.simulation import Job
from src.config import *
from src.physics_utils import get_stage_duration
from src.config import parse_lift_device_code, is_convery_device

def normalize_device_id(raw_id, device_type):
    """标准化设备ID"""
    if pd.isna(raw_id):
        return None
    
    raw_str = str(raw_id).strip()
    nums = re.findall(r'\d+', raw_str)
    if not nums:
        return f"{device_type}01"
    
    idx = int(nums[0])
    
    if device_type == 'Lift':
        valid_idx = (idx - 1) % LIFT_COUNT + 1
        return f"Lift{valid_idx:02d}"
    elif device_type == 'FRGV':
        valid_idx = (idx - 1) % RGV_COUNT + 1
        return f"FRGV{valid_idx:02d}"
        
    return raw_str

def parse_node_str(node_str):
    """
    解析节点字符串，返回 (Layer, Row, Col)
    CSV格式: 04-032-008 -> Col-Row-Layer
    """
    if not isinstance(node_str, str):
        return (1, 1, 1)
        
    match = re.match(r'(\d+)-(\d+)-(\d+)', node_str)
    if match:
        c_str, r_str, l_str = match.groups()
        c = int(c_str)
        r = int(r_str)
        l = int(l_str)
        return (l, r, c)  # (Layer, Row, Col)
    
    return (1, 1, 1)

def classify_task(from_node, to_node):
    """
    判断任务类型：出库、入库、倒库
    
    返回：
        'outbound': 出库（货架层 > 1 → 地面层 = 1）
        'inbound': 入库（地面层 = 1 → 货架层 > 1）
        'relocate': 倒库（货架层之间搬运）
    """
    from_layer = parse_node_str(from_node)[0]
    to_layer = parse_node_str(to_node)[0]
    
    if from_layer > 1 and to_layer == 1:
        return 'outbound'  # 出库：从货架到地面
    elif from_layer == 1 and to_layer > 1:
        return 'inbound'   # 入库：从地面到货架
    else:
        return 'relocate'  # 倒库：货架之间搬运

def find_nearest_lift(row, col):
    """
    找到距离指定位置最近的提升机
    
    参数：
        row: 目标行号
        col: 目标列号
    
    返回：
        (lift_id, lift_node): 提升机ID和对应的节点坐标
    """
    min_distance = float('inf')
    nearest_lift_id = 'Lift01'
    
    for lift_id, (lift_row, lift_col) in LIFT_LOCATIONS.items():
        distance = abs(lift_row - row) + abs(lift_col - col)
        if distance < min_distance:
            min_distance = distance
            nearest_lift_id = lift_id
    
    # 获取提升机位置对应的节点字符串（第1层）
    lift_row, lift_col = LIFT_LOCATIONS[nearest_lift_id]
    lift_node = f"{lift_col:02d}-{lift_row:03d}-01"
    
    return nearest_lift_id, lift_node

def create_outbound_stages(from_node, to_node):
    """
    创建出库任务的工序流程（重构为3个独立stage，明确设备状态）
    
    流程：
    1. RGV空车去货架（空载，速度快）
    2. RGV取货动作（固定时间T_LOAD_UNLOAD）
    3. RGV满载去提升机（满载，速度慢）
    4. 提升机垂直下降
    
    注意：to_node 是提升机在1层的入口位置
    """
    from_coords = parse_node_str(from_node)
    to_coords = parse_node_str(to_node)
    
    from_layer, from_row, from_col = from_coords
    to_layer, to_row, to_col = to_coords
    
    # 找到最近的提升机
    lift_id, lift_node = find_nearest_lift(from_row, from_col)
    lift_coords = parse_node_str(lift_node)
    lift_layer, lift_row, lift_col = lift_coords
    
    stages = []
    
    # 工序1：RGV空车去货架（空驶阶段）
    # 注意：此时RGV位置未知，距离设为0，实际空驶时间在simulation中动态计算
    stage1 = {
        'type': 'FRGV',
        'duration': 0,  # 空驶时间动态计算
        'device_id': None,
        'from_node': None,  # 当前位置（动态）
        'to_node': from_node,  # 货架位置
        'is_loaded': False,  # 空载
        'is_empty_travel': True,  # 标记为空驶阶段
        'description': f'RGV: 空车去货架 {from_node}'
    }
    stages.append(stage1)
    
    # 工序2：RGV取货动作
    stage2 = {
        'type': 'Load',  # 新类型：取货动作
        'duration': T_LOAD_UNLOAD,  # 固定取货时间（config中配置）
        'device_id': None,
        'from_node': from_node,
        'to_node': from_node,  # 位置不变
        'is_loaded': False,  # 取货过程中仍为未满载
        'description': f'RGV: 在{from_node}取货'
    }
    stages.append(stage2)
    
    # 工序3：RGV满载去提升机
    dist_x_3 = abs(from_row - lift_row) * CELL_LENGTH
    dist_y_3 = abs(from_col - lift_col) * CELL_WIDTH
    rgv_time_3 = get_stage_duration(dist_x_3, dist_y_3, 0, 'FRGV', is_loaded=True)
    
    stage3 = {
        'type': 'FRGV',
        'duration': rgv_time_3,  # 仅移动时间，不含取货
        'device_id': None,
        'from_node': from_node,
        'to_node': lift_node,
        'is_loaded': True,  # 满载
        'description': f'RGV: 从{from_node}运货到{lift_node}'
    }
    stages.append(stage3)
    
    # 工序4：提升机垂直下降
    dist_z = abs(from_layer - 1) * LAYER_HEIGHT
    lift_time = get_stage_duration(0, 0, dist_z, 'Lift', is_loaded=True)
    
    stage4 = {
        'type': 'Lift',
        'duration': lift_time + T_LOAD_UNLOAD,  # 垂直时间 + 卸货时间
        'device_id': lift_id,
        'from_node': lift_node,
        'to_node': to_node,
        'is_loaded': True,
        'affinity_layer': from_layer,
        'description': f'Lift: 从{from_layer}层降到1层'
    }
    stages.append(stage4)
    
    return stages

def create_inbound_stages(from_node, to_node):
    """
    创建入库任务的工序流程（重构为独立stages，明确设备状态）
    
    流程：
    1. RGV空车去1层提升机门口（空载，速度快）
    2. RGV在1层提升机取货（固定时间T_LOAD_UNLOAD）
    3. 提升机上升到目标层（RGV在电梯内）
    4. RGV满载去目标货架（满载，速度慢）
    5. RGV卸货动作（固定时间T_LOAD_UNLOAD）
    
    注意：from_node 是提升机在1层的入口位置
    """
    from_coords = parse_node_str(from_node)
    to_coords = parse_node_str(to_node)
    
    from_layer, from_row, from_col = from_coords
    to_layer, to_row, to_col = to_coords
    
    # 找到最近的提升机
    lift_id, lift_node = find_nearest_lift(to_row, to_col)
    lift_coords = parse_node_str(lift_node)
    lift_layer, lift_row, lift_col = lift_coords
    
    # 提升机在第1层和目标层的节点
    lift_node_1 = f"{lift_col:02d}-{lift_row:03d}-01"
    lift_node_target = f"{lift_col:02d}-{lift_row:03d}-{to_layer:02d}"
    
    stages = []
    
    # 工序1：RGV空车去1层提升机门口（空驶阶段）
    stage1 = {
        'type': 'FRGV',
        'duration': 0,  # 空驶时间动态计算
        'device_id': None,
        'from_node': None,  # 当前位置（动态）
        'to_node': lift_node_1,
        'is_loaded': False,  # 空载
        'is_empty_travel': True,  # 标记为空驶阶段
        'description': f'RGV: 空车去1层提升机 {lift_node_1}'
    }
    stages.append(stage1)
    
    # 工序2：RGV在1层提升机取货
    stage2 = {
        'type': 'Load',
        'duration': T_LOAD_UNLOAD,
        'device_id': None,
        'from_node': lift_node_1,
        'to_node': lift_node_1,
        'is_loaded': False,
        'description': f'RGV: 在1层提升机取货'
    }
    stages.append(stage2)
    
    # 工序3：提升机上升（RGV在电梯内）
    dist_z = abs(to_layer - 1) * LAYER_HEIGHT
    lift_time = get_stage_duration(0, 0, dist_z, 'Lift', is_loaded=True)
    
    stage3 = {
        'type': 'Lift',
        'duration': lift_time + T_LOAD_UNLOAD,  # 垂直时间 + 出电梯时间
        'device_id': lift_id,
        'from_node': lift_node_1,
        'to_node': lift_node_target,
        'is_loaded': True,
        'affinity_layer': to_layer,
        'description': f'Lift: 从1层升到{to_layer}层'
    }
    stages.append(stage3)
    
    # 工序4：RGV满载去目标货架
    dist_x_4 = abs(lift_row - to_row) * CELL_LENGTH
    dist_y_4 = abs(lift_col - to_col) * CELL_WIDTH
    rgv_time_4 = get_stage_duration(dist_x_4, dist_y_4, 0, 'FRGV', is_loaded=True)
    
    stage4 = {
        'type': 'FRGV',
        'duration': rgv_time_4,
        'device_id': None,
        'from_node': lift_node_target,
        'to_node': to_node,
        'is_loaded': True,  # 满载
        'description': f'RGV: 从提升机运货到{to_node}'
    }
    stages.append(stage4)
    
    # 工序5：RGV卸货
    stage5 = {
        'type': 'Unload',  # 新类型：卸货动作
        'duration': T_LOAD_UNLOAD,
        'device_id': None,
        'from_node': to_node,
        'to_node': to_node,
        'is_loaded': True,  # 卸货前仍为满载
        'description': f'RGV: 在{to_node}卸货'
    }
    stages.append(stage5)
    
    return stages

def create_relocate_stages(from_node, to_node):
    """
    创建倒库任务的工序流程（重构为独立stages）
    
    倒库可能是：
    - 同层：拆分为空驶、取货、运送3个阶段
    - 跨层：类似出库+入库的组合
    """
    from_layer = parse_node_str(from_node)[0]
    to_layer = parse_node_str(to_node)[0]
    
    if from_layer == to_layer:
        # 同层倒库：拆分为3个独立阶段
        from_row, from_col = parse_node_str(from_node)[1:3]
        to_row, to_col = parse_node_str(to_node)[1:3]
        
        stages = []
        
        # 阶段1：RGV空车去源货架（空驶）
        stage1 = {
            'type': 'FRGV',
            'duration': 0,
            'device_id': None,
            'from_node': None,
            'to_node': from_node,
            'is_loaded': False,
            'is_empty_travel': True,
            'description': f'RGV: 空车去源货架 {from_node}'
        }
        stages.append(stage1)
        
        # 阶段2：取货动作
        stage2 = {
            'type': 'Load',
            'duration': T_LOAD_UNLOAD,
            'device_id': None,
            'from_node': from_node,
            'to_node': from_node,
            'is_loaded': False,
            'description': f'RGV: 在{from_node}取货'
        }
        stages.append(stage2)
        
        # 阶段3：RGV满载去目标货架
        dist_x = abs(from_row - to_row) * CELL_LENGTH
        dist_y = abs(from_col - to_col) * CELL_WIDTH
        rgv_time = get_stage_duration(dist_x, dist_y, 0, 'FRGV', is_loaded=True)
        
        stage3 = {
            'type': 'FRGV',
            'duration': rgv_time,
            'device_id': None,
            'from_node': from_node,
            'to_node': to_node,
            'is_loaded': True,
            'description': f'RGV: 从{from_node}运货到{to_node}'
        }
        stages.append(stage3)
        
        # 阶段4：卸货动作
        stage4 = {
            'type': 'Unload',
            'duration': T_LOAD_UNLOAD,
            'device_id': None,
            'from_node': to_node,
            'to_node': to_node,
            'is_loaded': True,
            'description': f'RGV: 在{to_node}卸货'
        }
        stages.append(stage4)
        
        return stages
    else:
        # 跨层倒库：拆分为出库+入库
        # 找到起点层附近的提升机作为中间点
        from_row, from_col = parse_node_str(from_node)[1:3]
        lift_id, lift_node_from = find_nearest_lift(from_row, from_col)
        lift_coords = parse_node_str(lift_node_from)
        lift_row, lift_col = lift_coords[1], lift_coords[2]
        
        # 中间点：1层的提升机入口
        intermediate_node = f"{lift_col:02d}-{lift_row:03d}-01"
        
        # 先执行出库到1层
        outbound_stages = create_outbound_stages(from_node, intermediate_node)
        
        # 再执行入库到目标层
        # 需要找到目标层对应的提升机位置
        lift_node_target = f"{lift_col:02d}-{lift_row:03d}-{to_layer:02d}"
        inbound_stages = create_inbound_stages(intermediate_node, to_node)
        
        return outbound_stages + inbound_stages

def load_jobs_from_csv(file_path):
    """
    从CSV加载任务，根据任务类型创建正确的工序流程
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return []

    # 预处理列名
    df.columns = [c.strip() for c in df.columns]

    # 基于真实发送时间恢复任务到达节奏，避免把动态到达压成静态全释放
    base_send_time = None
    if 'SendTime' in df.columns:
        df['_parsed_send_time'] = pd.to_datetime(df['SendTime'], errors='coerce')
        valid_send_times = df['_parsed_send_time'].dropna()
        if not valid_send_times.empty:
            base_send_time = valid_send_times.min()
    
    jobs = []
    
    # 按任务号分组
    if 'WmsTaskNo' not in df.columns:
        print("Error: 'WmsTaskNo' column not found in CSV.")
        return []
    
    grouped = df.groupby('WmsTaskNo')
    
    for wms_no, group in grouped:
        group = group.sort_values('SendTime')
        
        release_time = 0.0
        if base_send_time is not None and '_parsed_send_time' in group.columns:
            group_send_times = group['_parsed_send_time'].dropna()
            if not group_send_times.empty:
                release_time = max(0.0, (group_send_times.min() - base_send_time).total_seconds())
        
        stages = []
        
        # 处理每个子任务
        for idx, row in group.iterrows():
            from_node = str(row.get('FromNode', ''))
            to_node = str(row.get('ToNode', ''))
            
            # 过滤掉传送带设备（3xxx开头）
            if is_convery_device(from_node) or is_convery_device(to_node):
                continue
            
            # 转换2xxx开头的提升机设备编码为实际坐标
            from_node = parse_lift_device_code(from_node) or from_node
            to_node = parse_lift_device_code(to_node) or to_node
            
            # 判断任务类型并创建对应的工序
            task_type = classify_task(from_node, to_node)
            
            if task_type == 'outbound':
                task_stages = create_outbound_stages(from_node, to_node)
            elif task_type == 'inbound':
                task_stages = create_inbound_stages(from_node, to_node)
            else:  # relocate
                task_stages = create_relocate_stages(from_node, to_node)
            
            stages.extend(task_stages)
        
        # 创建 Job 对象
        if stages:
            job = Job(str(wms_no), release_time, stages, str(wms_no))
            jobs.append(job)
    
    print(f"[DataLoader] Loaded {len(jobs)} jobs ({len([j for j in jobs if any('出库' in str(s.get('description','')) for s in j.stages)])} outbound, "
          f"{len([j for j in jobs if any('入库' in str(s.get('description','')) for s in j.stages)])} inbound, "
          f"{len([j for j in jobs if any('倒库' in str(s.get('description','')) for s in j.stages)])} relocate).")
    return jobs

def build_resources_from_csv(file_path):
    """
    从CSV构建资源池
    
    根据配置创建默认的资源池（提升机和穿梭车）
    """
    from src.simulation import Resource
    
    resources = {}
    
    # 创建提升机（在各自的IO端口，都在第一层）
    for i in range(1, LIFT_COUNT + 1):
        rid = f'Lift{i:02d}'
        if rid in LIFT_LOCATIONS:
            row, col = LIFT_LOCATIONS[rid]
            initial_node = f"{col:02d}-{row:03d}-01"
        else:
            initial_node = '01-001-01'
        resources[rid] = Resource(rid, 'Lift', initial_node)
    
    # 创建穿梭车（都在第一层，平均分配在提升机附近）
    rgv_per_lift = RGV_COUNT // LIFT_COUNT
    remainder = RGV_COUNT % LIFT_COUNT
    
    rgv_idx = 1
    for lift_idx in range(1, LIFT_COUNT + 1):
        rid = f'Lift{lift_idx:02d}'
        if rid in LIFT_LOCATIONS:
            lift_row, lift_col = LIFT_LOCATIONS[rid]
            count = rgv_per_lift + (1 if lift_idx <= remainder else 0)
            
            for i in range(count):
                rgv_id = f'FRGV{rgv_idx:02d}'
                # 穿梭车在提升机附近（同一列，相邻行）
                rgv_row = lift_row + (i + 1)
                rgv_col = lift_col
                initial_node = f"{rgv_col:02d}-{rgv_row:03d}-01"
                
                resources[rgv_id] = Resource(rgv_id, 'FRGV', initial_node)
                rgv_idx += 1
    
    print(f"[ResourceBuilder] Created {len(resources)} resources.")
    return resources
