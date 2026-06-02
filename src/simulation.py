"""
simulation.py - 仿真解码器（修正版）
职责：四向穿梭车仓储系统的仿真器

系统配置：
- 11层货架
- 15台穿梭车
- 6台提升机
- 任务类型：出库、入库、倒库

核心逻辑修正：
1. 出库任务（货架层 > 1 → 地面层 = 1）：
   - 工序1：RGV从货架取货 → 运到to_node（提升机1层入口）
   - 工序2：Lift从起点层降到1层
   
2. 入库任务（地面层 = 1 → 货架层 > 1）：
   - 工序1：Lift从1层升到目标层
   - 工序2：RGV从提升机门口 → 运到目标货架卸货

3. 每个工序独立计算：
   - 设备空驶时间
   - 任务准备时间
   - 等待时间
   - 实际作业时间
"""
import numpy as np
from src.config import T_EXIT, LIFT_COUNT, RGV_COUNT, T_LOAD_UNLOAD, LIFT_LOCATIONS
from src.physics_utils import parse_node_str, calculate_physics

class Job:
    """任务对象"""
    def __init__(self, job_id, release_time, stages, original_task_no=None):
        self.id = job_id
        self.original_task_no = original_task_no
        self.release_time = release_time
        self.stages = stages
        # 记录仿真结果
        self.start_times = []
        self.finish_times = []
        self.waits = []
        self.lift_waits = 0  # 穿梭车等待提升机的总时间
        self.actual_devices = []  # 实际使用的设备

class Resource:
    """
    资源对象（设备）
    
    状态：(x, y, z, t, o)
    - x, y, z: 设备位置（节点坐标）
    - t: 设备可用时间
    - o: 设备是否可用（通过available_time判断）
    """
    def __init__(self, r_id, r_type, initial_node=None):
        self.id = r_id
        self.type = r_type
        self.available_time = 0  # t: 设备可用时间（t=0时刻可用）
        self.initial_node = initial_node if initial_node else '01-001-001'
        self.current_node = self.initial_node
        # 解析当前层
        if self.current_node:
            coords = parse_node_str(self.current_node)
            self.current_layer = coords[0]  # z: Layer
        else:
            self.current_layer = 1
            
        # 真实工作量统计
        self.total_busy_time = 0.0  # 累计真实工作时间
        self.processed_count = 0    # 实际处理的工序数量
        self.total_idle_time = 0.0  # 累计空闲等待时间（设备等待任务）
        self.last_busy_end = 0.0    # 上次忙碌结束时间

class Simulator:
    """
    四向穿梭车仓储系统仿真器（修正版）
    """
    
    def __init__(self, jobs, resource_map=None, mode='decoupled',
                 rho_thresholds=(0.70, 0.80, 0.95),
                 rho_hysteresis=0.03,
                 enable_hysteresis=True):
        """
        参数：
            rho_thresholds: 三档阈值 (θ1, θ2, θ3)，论文 §3.1 默认 (0.70, 0.80, 0.95)
            rho_hysteresis: 滞回间隙 δ，默认 0.03（避免在阈值附近模式抖振）
            enable_hysteresis: 是否启用滞回；False 时退化为原版即时切换
        """
        self.mode = mode  # 'decoupled' 或 'adaptive_rho'（自适应ρ耦合）
        self.jobs_map = {j.id: j for j in jobs}
        self.jobs = jobs
        # ρ 感知调度参数（§3.1）
        self.rho_thresholds = tuple(rho_thresholds)
        self.rho_hysteresis = float(rho_hysteresis)
        self.enable_hysteresis = bool(enable_hysteresis)
        # 当前所处档位（0=松耦合 1=主动协同 2=预测同步 3=被动同步）
        self.current_rho_band = 0
        # 模式切换计数（供诊断与论文 §3.1 抖振分析使用）
        self.mode_switch_count = 0
        # 初始化资源池
        if resource_map is None:
            self.resources = self._init_default_resources()
        else:
            self.resources = resource_map

        # 性能优化：预分类资源
        self._build_resource_cache()
    
    def _init_default_resources(self):
        """初始化默认资源池"""
        res = {}
        
        # 初始化提升机（在各自的IO端口，都在第一层）
        for i in range(1, LIFT_COUNT + 1):
            rid = f'Lift{i:02d}'
            if rid in LIFT_LOCATIONS:
                row, col = LIFT_LOCATIONS[rid]
                initial_node = f"{col:02d}-{row:03d}-01"
            else:
                initial_node = '01-001-001'
            res[rid] = Resource(rid, 'Lift', initial_node)
        
        # 初始化穿梭车（都在第一层，平均分配在提升机附近）
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
                    rgv_row = lift_row + (i + 1)
                    rgv_col = lift_col
                    initial_node = f"{rgv_col:02d}-{rgv_row:03d}-01"
                    
                    res[rgv_id] = Resource(rgv_id, 'FRGV', initial_node)
                    rgv_idx += 1
        
        return res
    
    def _build_resource_cache(self):
        """构建资源缓存，预分类穿梭车和提升机"""
        self.rgv_resources = []
        self.lift_resources = []
        self.rgv_ids = []
        self.lift_ids = []
        
        for r_id, resource in self.resources.items():
            if resource.type == 'FRGV':
                self.rgv_resources.append(resource)
                self.rgv_ids.append(r_id)
            elif resource.type == 'Lift':
                self.lift_resources.append(resource)
                self.lift_ids.append(r_id)
        
        # 预计算提升机位置缓存
        self.lift_positions = {}
        for resource in self.lift_resources:
            coords = parse_node_str(resource.current_node)
            self.lift_positions[resource.id] = coords[1]  # 记录行号
    
    def reset(self):
        """重置仿真状态"""
        for j in self.jobs:
            j.start_times = []
            j.finish_times = []
            j.waits = []
            j.lift_waits = 0
            j.actual_devices = []
        
        for r in self.rgv_resources:
            r.available_time = 0
            r.total_busy_time = 0.0
            r.total_idle_time = 0.0
            r.processed_count = 0
            r.current_node = r.initial_node
            if r.current_node:
                coords = parse_node_str(r.current_node)
                r.current_layer = coords[0]

        for r in self.lift_resources:
            r.available_time = 0
            r.total_busy_time = 0.0
            r.total_idle_time = 0.0
            r.processed_count = 0
            r.current_node = r.initial_node
            if r.current_node:
                coords = parse_node_str(r.current_node)
                r.current_layer = coords[0]
        
        # 用于绘制甘特图的全局执行日志
        self.execution_log = []

        # W_total 双向分量缓存（在 run() 末尾填充）
        self.last_W_rgv_wait = 0.0
        self.last_W_lift_idle = 0.0
        self.last_W_sync_total = 0.0

        # 重置 ρ 档位状态（用于滞回切换计数）
        self.current_rho_band = 0
        self.mode_switch_count = 0
    
    def _find_available_rgv(self, target_layer, current_time):
        """
        查找可用的穿梭车
        
        优先使用目标层的穿梭车，如果没有，则使用其他层的穿梭车
        """
        best_rgv = None
        min_available_time = float('inf')
        
        # 1. 优先查找目标层且可用的穿梭车
        for resource in self.rgv_resources:
            if resource.available_time <= current_time and resource.current_layer == target_layer:
                return resource  # 找到目标层的可用穿梭车，立即返回
        
        # 2. 查找其他层可用的穿梭车
        for resource in self.rgv_resources:
            if resource.available_time <= current_time:
                return resource  # 找到可用穿梭车，立即返回
        
        # 3. 如果没有可用的，返回最早可用的
        for resource in self.rgv_resources:
            if resource.available_time < min_available_time:
                min_available_time = resource.available_time
                best_rgv = resource
        
        return best_rgv
    
    def _find_available_lift(self, lift_id_preferred, current_time):
        """
        查找可用的提升机（FIFO规则）
        
        优先使用指定的提升机（从工序的device_id），如果不可用再找其他
        """
        # 1. 首先尝试使用指定的提升机
        if lift_id_preferred and lift_id_preferred in self.resources:
            lift = self.resources[lift_id_preferred]
            if lift.available_time <= current_time:
                return lift
        
        # 2. 找任意可用的提升机
        for resource in self.lift_resources:
            if resource.available_time <= current_time:
                return resource
        
        # 3. 如果没有可用的，返回最早可用的
        best_lift = min(self.lift_resources, key=lambda x: x.available_time)
        return best_lift
    
    def _find_available_lift_c_eft(self, lift_id_preferred, rgv_arrival_time,
                                   lift_duration, exit_time=5.0, lift_pref_idx=None,
                                   target_layer=None):
        """
        协同最早完成时间（C-EFT）规则选择提升机（增强版）
        预测同步模式 (0.93 < ρ ≤ 0.98): 增强版 C-EFT，加入楼层亲和力
        
        选择使该工序物理完工时间最早的提升机
        支持基因中的lift偏好指导选择
        支持楼层亲和力奖励（鼓励 Lift 连续接载同一楼层任务）
        
        参数：
            lift_id_preferred: 指定的提升机ID
            rgv_arrival_time: 穿梭车到达接驳口的时刻 D_{j,k-1}
            lift_duration: 垂直运行时长 p_{j,k}^{lift}
            exit_time: 穿梭车驶出轿厢的固定时间 t_{exit}
            lift_pref_idx: 基因中指定的lift偏好索引（0-5）
            target_layer: 目标楼层，用于楼层亲和力计算
            
        返回：
            best_lift: 最优提升机对象
        """
        best_lift = None
        min_completion_time = float('inf')
        
        # 遍历所有提升机
        candidates = []
        for lift in self.lift_resources:
            # FreeTime(m): 提升机下一可用时刻
            free_time = lift.available_time
            
            # max(FreeTime(m), D_{j,k-1}): 协同开始时间
            start_time = max(free_time, rgv_arrival_time)
            
            # + p_{j,k}^{lift} + t_{exit}: 工序完成时间
            completion_time = start_time + lift_duration + exit_time
            
            # 楼层亲和力奖励 (Layer Affinity Bonus)
            # 如果这台 Lift 上一个任务结束所在的楼层，刚好是当前任务所在的楼层 target_layer
            if target_layer is not None and getattr(lift, 'current_layer', 1) == target_layer:
                # 给予 15% 的完成时间"时间窗打折奖励"，促使算法优先选择免去垂直空驶的方案
                completion_time = completion_time * 0.85
            
            candidates.append((lift, completion_time))
        
        # 如果有lift偏好，给偏好的lift一个奖励（更早的完成时间）
        if lift_pref_idx is not None and 0 <= lift_pref_idx < len(self.lift_resources):
            for i, (lift, comp_time) in enumerate(candidates):
                if i == lift_pref_idx:
                    # 给偏好的lift一个小的奖励（减少10%的完成时间）
                    candidates[i] = (lift, comp_time * 0.9)
        
        # 选择完成最早的提升机
        for lift, completion_time in candidates:
            if completion_time < min_completion_time:
                min_completion_time = completion_time
                best_lift = lift
        
        return best_lift
    
    def _find_best_lift_active_sync(self, rgv_node, rgv_arrival_time):
        """主动协同模式 (0.85 < ρ ≤ 0.93): 带有距离惩罚的局部负载均衡"""
        best_lift = None
        min_total_cost = float('inf')
        
        # RGV 横向移动的时间惩罚系数
        # 进一步简化物理模型，让优化算法更灵活
        RGV_TRAVEL_PENALTY = 1.0  # 保持1.0表示无惩罚
        
        for lift in self.lift_resources:
            # 1. RGV横向跑去接驳口的时间
            rgv_travel_to_lift = calculate_physics(rgv_node, lift.current_node, 'FRGV')
            actual_rgv_arrival = rgv_arrival_time + rgv_travel_to_lift
            
            # 2. Lift的排队等待时间
            lift_wait_time = max(0, lift.available_time - actual_rgv_arrival)
            
            # 放大 RGV 空跑成本，迫使其在局部寻找相对空闲的 Lift
            total_cost = (rgv_travel_to_lift * RGV_TRAVEL_PENALTY) + lift_wait_time
            
            if total_cost < min_total_cost:
                min_total_cost = total_cost
                best_lift = lift
                
        return best_lift
    
    def _calculate_local_rho(self, current_time):
        """
        实时计算当前系统的拥挤度 ρ (平滑抗抖版)
        """
        busy_lifts = 0
        total_queue_time = 0
        total_lifts = len(self.lift_resources)

        for lift in self.lift_resources:
            if lift.available_time > current_time:
                busy_lifts += 1
                total_queue_time += (lift.available_time - current_time)

        # 基础 ρ: 正在干活的电梯比例
        base_rho = busy_lifts / total_lifts if total_lifts > 0 else 0

        # 惩罚项: 平均每台电梯的积压时长 (假设提升机平均一趟需要 15 秒)
        avg_queue_time = total_queue_time / total_lifts if total_lifts > 0 else 0
        queue_penalty = avg_queue_time / 15.0  # 积压越长，ρ 越大

        # 最终 ρ 值
        rho = base_rho + queue_penalty * 0.2

        return min(rho, 1.15) # 封顶

    def _select_rho_band(self, current_rho):
        """
        基于阈值与滞回机制选择当前调度档位。

        档位定义：
          0 = 松耦合      (ρ ≤ θ1)
          1 = 主动协同    (θ1 < ρ ≤ θ2)
          2 = 预测同步    (θ2 < ρ ≤ θ3)
          3 = 被动同步    (ρ > θ3)

        滞回逻辑（默认开启）：
          - 升档（band ↑）：ρ 超过 上阈值 才触发
          - 降档（band ↓）：ρ 需回落至 (下阈值 - δ) 以下才触发
        这样可避免 ρ 在阈值附近抖振导致的高频模式切换。
        """
        t1, t2, t3 = self.rho_thresholds
        new_band = self.current_rho_band
        if not self.enable_hysteresis:
            # 即时切换（原版逻辑）
            if current_rho <= t1:
                new_band = 0
            elif current_rho <= t2:
                new_band = 1
            elif current_rho <= t3:
                new_band = 2
            else:
                new_band = 3
        else:
            δ = self.rho_hysteresis
            cur = self.current_rho_band
            # 升档：使用原始阈值
            if cur == 0 and current_rho > t1:
                new_band = 1 if current_rho <= t2 else (2 if current_rho <= t3 else 3)
            elif cur == 1 and current_rho > t2:
                new_band = 2 if current_rho <= t3 else 3
            elif cur == 2 and current_rho > t3:
                new_band = 3
            # 降档：阈值整体往下推 δ，必须明显回落才切换
            elif cur == 3 and current_rho <= (t3 - δ):
                new_band = 2 if current_rho > (t2 - δ) else (1 if current_rho > (t1 - δ) else 0)
            elif cur == 2 and current_rho <= (t2 - δ):
                new_band = 1 if current_rho > (t1 - δ) else 0
            elif cur == 1 and current_rho <= (t1 - δ):
                new_band = 0
            # 否则维持当前档（滞回缓冲区内）
        if new_band != self.current_rho_band:
            self.mode_switch_count += 1
            self.current_rho_band = new_band
        return new_band
    
    def run(self, sequence_ids, use_time_window_insertion=False, use_c_eft=False, lift_preferences=None):
        """
        执行仿真解码（修正版）
        
        正确的工序执行逻辑：
        - 每个工序独立处理
        - FRGV工序：分配穿梭车，计算空驶、装载、运输
        - Lift工序：分配提升机，计算垂直运行
        - 等待时间只计算在Lift工序（RGV等Lift）
        
        参数：
            sequence_ids: 任务 ID 序列
            use_time_window_insertion: 是否启用时间窗左插入
            use_c_eft: 是否启用C-EFT协同最早完成时间规则
            lift_preferences: 每个任务偏好的lift索引列表（用于联合优化）
        
        返回：
            (Cmax, W_total, P95_Wait)
        """
        self.reset()
        
        global_cmax = 0
        
        # 转换lift_preferences为字典格式，方便查询
        lift_pref_dict = {}
        if lift_preferences:
            for i, job_id in enumerate(sequence_ids):
                if isinstance(job_id, int):
                    lift_pref_dict[job_id] = lift_preferences[i] if i < len(lift_preferences) else None
                else:
                    lift_pref_dict[job_id] = lift_preferences[i] if i < len(lift_preferences) else None
        
        for job_id in sequence_ids:
            # 获取任务对象
            if isinstance(job_id, int):
                if job_id < len(self.jobs):
                    job = self.jobs[job_id]
                else:
                    continue
            else:
                job = self.jobs_map.get(job_id)
                if job is None:
                    continue
            
            task_ready_time = job.release_time
            
            # 记录上一工序完成时的设备（用于计算空驶）
            prev_rgv_finish_time = 0
            prev_lift_finish_time = 0
            current_rgv_id = None
            
            for s_idx, stage in enumerate(job.stages):
                s_type = stage['type']
                duration = stage['duration']
                from_node = stage.get('from_node', '01-001-001')
                to_node = stage.get('to_node', '01-001-001')
                preferred_device = stage.get('device_id')  # 指定的设备（如Lift01）
                
                # 解析坐标
                from_coords = parse_node_str(from_node)
                to_coords = parse_node_str(to_node)
                target_layer = from_coords[0]  # 普通工序默认使用起始层
                if s_type == 'Lift':
                    target_layer = stage.get('affinity_layer', to_coords[0] if to_coords[0] > 1 else from_coords[0])
                lift_pref_idx = lift_pref_dict.get(job.id)
                
                # ==========================================
                # 分配设备 (动态耦合调度修改区)
                # ==========================================
                if s_type == 'FRGV' or s_type == 'Load' or s_type == 'Unload':  # Load/Unload类型使用RGV
                    resource = self._find_available_rgv(target_layer, task_ready_time)
                    current_rgv_id = resource.id if resource else None
                elif s_type == 'Lift':
                    # 根据 mode 选择调度策略
                    if self.mode == 'decoupled':
                        # static_decoupled 模式：始终使用松耦合 (FIFO/C-EFT)
                        # 禁用动态耦合，不计算ρ，始终使用模式1
                        resource = self._find_available_lift_c_eft(
                            preferred_device,
                            prev_rgv_finish_time,
                            duration,
                            lift_pref_idx=lift_pref_idx,
                            target_layer=target_layer
                        )
                    else:
                        # adaptive_rho 模式：启用动态耦合，根据ρ值切换模式
                        # 1. 实时感知系统状态
                        current_rho = self._calculate_local_rho(task_ready_time)

                        # 2. 滞回档位选择（避免阈值抖振，见 §3.1 修订）
                        band = self._select_rho_band(current_rho)

                        if band == 0:
                            # 模式1: 松耦合
                            resource = self._find_available_lift(preferred_device, task_ready_time)

                        elif band == 1:
                            # 模式2: 主动协同 (负载均衡)
                            resource = self._find_best_lift_active_sync(from_node, prev_rgv_finish_time)

                        elif band == 2:
                            # 模式3: 预测同步 (时间窗聚合)
                            resource = self._find_available_lift_c_eft(
                                preferred_device,
                                prev_rgv_finish_time,
                                duration,
                                lift_pref_idx=lift_pref_idx,
                                target_layer=target_layer
                            )

                        else:
                            # 模式4: 被动同步 (严重拥堵)
                            resource = self._find_available_lift(preferred_device, task_ready_time)
                            # 发生被动同步时，RGV必须在原地死等Lift接走货物
                        # 此时Lift接货的真实开始时间：
                        lift_start = max(task_ready_time, resource.available_time)
                        actual_rgv_release_time = lift_start + T_LOAD_UNLOAD  # T_LOAD_UNLOAD 交接时间
                        
                        # 追溯上一个刚刚完成的RGV，延迟它的可用时间
                        if current_rgv_id and current_rgv_id in self.resources:
                            prev_rgv = self.resources[current_rgv_id]
                            if actual_rgv_release_time > prev_rgv.available_time:
                                # 记录被动等待的惩罚时间
                                penalty = actual_rgv_release_time - prev_rgv.available_time
                                
                                # 记录由于极度拥堵造成的被动死等色块
                                self.execution_log.append({
                                    'device': prev_rgv.id,
                                    'start': prev_rgv.available_time,
                                    'duration': penalty,
                                    'type': 'Wait_Passive_Sync',
                                    'job_id': job.id
                                })
                                
                                prev_rgv.total_busy_time += penalty
                                prev_rgv.available_time = actual_rgv_release_time
                                job.lift_waits += penalty  # 算入总等待时间
                else:
                    continue
                
                if not resource:
                    continue
                
                # ==========================================
                # 计算空驶时间（设备从当前位置到接货点）
                # ==========================================
                # 对于标记为空驶的stage，动态计算空驶时间
                if stage.get('is_empty_travel', False):
                    empty_travel_time = calculate_physics(
                        resource.current_node, from_node, s_type
                    )
                    # 空驶阶段的真实duration就是空驶时间
                    duration = empty_travel_time
                else:
                    empty_travel_time = calculate_physics(
                        resource.current_node, from_node, s_type
                    )
                
                # ==========================================
                # 计算设备到达接货点的时间
                # ==========================================
                arrival_at_pickup = resource.available_time + empty_travel_time
                
                # ==========================================
                # 计算开始时间和等待时间
                # ==========================================
                start_time = max(task_ready_time, arrival_at_pickup)
                wait_time = max(0, start_time - task_ready_time)
                
                # 对于Lift工序，记录RGV等待时间
                if s_type == 'Lift' and s_idx > 0:
                    # 上一工序是RGV，记录RGV到达时间
                    rgv_arrival_time = prev_rgv_finish_time
                    lift_arrival_time = arrival_at_pickup
                    lift_wait = max(0, lift_arrival_time - rgv_arrival_time)
                    job.lift_waits += lift_wait
                    job.waits.append(lift_wait)
                else:
                    job.waits.append(0)
                
                # ==========================================
                # 计算完成时间
                # ==========================================
                finish_time = start_time + duration

                # 记录正常的作业色块
                # 处理 Load/Unload 类型的显示
                log_type = s_type
                if s_type == 'Load':
                    log_type = 'FRGV_Load'  # 统一显示为RGV相关
                elif s_type == 'Unload':
                    log_type = 'FRGV_Unload'
                self.execution_log.append({
                    'device': resource.id,
                    'start': start_time,
                    'duration': duration,
                    'type': log_type,  # 'FRGV', 'Lift', 'FRGV_Load', 'FRGV_Unload'
                    'job_id': job.id
                })

                # 对于Lift工序，如果RGV在接驳口等了，记录等待色块
                if s_type == 'Lift' and s_idx > 0:
                    rgv_arrival_time = prev_rgv_finish_time
                    lift_arrival_time = arrival_at_pickup
                    lift_wait = max(0, lift_arrival_time - rgv_arrival_time)
                    if lift_wait > 0:
                        # 上一个RGV处于等待状态
                        if current_rgv_id:
                            self.execution_log.append({
                                'device': current_rgv_id,
                                'start': rgv_arrival_time,
                                'duration': lift_wait,
                                'type': 'Wait_For_Lift',
                                'job_id': job.id
                            })
                
                # 记录时间
                job.start_times.append(start_time)
                job.finish_times.append(finish_time)
                job.actual_devices.append(resource.id)
                
                # ==========================================
                # 更新设备状态
                # ==========================================
                # 真实工作时长 = 空驶时间 + 工序持续时间
                actual_busy_duration = empty_travel_time + duration
                resource.total_busy_time += actual_busy_duration
                resource.processed_count += 1
                resource.last_busy_end = finish_time
                
                # 更新位置和可用时间
                resource.current_node = to_node
                resource.available_time = finish_time
                
                if s_type == 'FRGV' or s_type == 'Load' or s_type == 'Unload':
                    coords = parse_node_str(to_node)
                    resource.current_layer = coords[0]
                    prev_rgv_finish_time = finish_time
                elif s_type == 'Lift':
                    # Lift 也更新当前楼层，用于楼层亲和力计算
                    coords = parse_node_str(to_node)
                    resource.current_layer = coords[0]
                    prev_lift_finish_time = finish_time
                
                # 更新任务准备好时间（下一工序可以开始）
                task_ready_time = finish_time
            
            # 更新全局最大完成时间
            if job.finish_times:
                job_finish_time = job.finish_times[-1]
                global_cmax = max(global_cmax, job_finish_time)
        
        # ==========================================
        # 统计结果 — 双向同步偏差及其分量
        # ==========================================
        # 论文 §2.2.3 修订后的定义：
        #   W_total = Σ_j Σ_k |t_lift_start(j,k) − t_rgv_finish(j,k-1)|  (双向 L1 同步偏差)
        # 同时分别累计两个方向，便于诊断与论文 §4 解析：
        #   W_RGV_wait  = Σ max(0, t_lift_start − t_rgv_finish)  (RGV 等 Lift, 单向 ReLU)
        #   W_Lift_idle = Σ max(0, t_rgv_finish − t_lift_start)  (Lift 已就位、等 RGV)
        total_sync_deviation = 0.0
        total_rgv_wait = 0.0     # 单向：RGV 在接驳口阻塞
        total_lift_idle = 0.0    # 单向：Lift 空闲等 RGV
        all_sync_deviations = []

        for job in self.jobs:
            # 对于每个任务的每个工序对（RGV -> Lift）
            for s_idx, stage in enumerate(job.stages):
                if stage['type'] == 'Lift' and s_idx > 0:
                    # 找到对应的 RGV 工序完成时间
                    if s_idx - 1 < len(job.finish_times):
                        rgv_finish = job.finish_times[s_idx - 1]
                        lift_start = job.start_times[s_idx]
                        delta = lift_start - rgv_finish
                        sync_deviation = abs(delta)
                        total_sync_deviation += sync_deviation
                        if delta >= 0:
                            total_rgv_wait += delta
                        else:
                            total_lift_idle += -delta
                        all_sync_deviations.append(sync_deviation)

        # 如果没有同步偏差数据，回退到原来的等待时间计算
        if total_sync_deviation == 0:
            total_sync_deviation = sum(j.lift_waits for j in self.jobs)

        # 暴露分量供上层（统计实验、消融、回归）使用
        self.last_W_rgv_wait = total_rgv_wait
        self.last_W_lift_idle = total_lift_idle
        self.last_W_sync_total = total_sync_deviation
        
        # P95 同步偏差
        p95 = 0
        if all_sync_deviations:
            all_sync_deviations.sort()
            idx = int(len(all_sync_deviations) * 0.95)
            p95 = all_sync_deviations[min(idx, len(all_sync_deviations)-1)]
        
        return global_cmax, total_sync_deviation, p95
    
    def get_critical_block(self, sequence_ids, use_time_window_insertion=False):
        """
        提取关键块
        
        分析仿真结果，找出"排队等待最长的那台提升机"上的任务
        
        参数:
            sequence_ids: 任务ID序列
            use_time_window_insertion: 是否使用时间窗左插入模式识别瓶颈
                                      应与父级评估时的TW模式保持一致
        """
        # 先运行仿真（瓶颈识别模式应与评估模式一致）
        self.run(sequence_ids, use_time_window_insertion=use_time_window_insertion)
        
        # 统计每台提升机的总等待时间
        lift_wait_times = {}
        for job in self.jobs:
            if hasattr(job, 'actual_devices') and job.actual_devices:
                for i, device_id in enumerate(job.actual_devices):
                    if 'Lift' in device_id:
                        wait_time = job.waits[i] if i < len(job.waits) else 0
                        if device_id not in lift_wait_times:
                            lift_wait_times[device_id] = 0
                        lift_wait_times[device_id] += wait_time
        
        if not lift_wait_times:
            return [], None, 0
        
        # 找出等待时间最长的提升机（瓶颈设备）
        bottleneck_lift = max(lift_wait_times, key=lift_wait_times.get)
        max_wait_time = lift_wait_times[bottleneck_lift]
        
        # 找出使用该提升机的所有任务
        critical_jobs = []
        for idx, job_id in enumerate(sequence_ids):
            job = self.jobs_map.get(job_id) if isinstance(job_id, str) else self.jobs[job_id]
            if job and hasattr(job, 'actual_devices'):
                if bottleneck_lift in job.actual_devices:
                    critical_jobs.append(idx)
        
        return critical_jobs, bottleneck_lift, max_wait_time
    
    def get_physical_state(self, sequence_ids):
        """
        获取物理系统状态
        
        返回数字孪生底层的真实物理状态
        """
        # 先运行仿真
        cmax, total_wait, p95 = self.run(sequence_ids, use_time_window_insertion=False)
        
        # 计算各提升机的利用率（拥堵指数）
        lift_utilizations = {}
        lift_schedules = {}
        for resource in self.lift_resources:
            total_busy = resource.total_busy_time
            utilization = total_busy / cmax if cmax > 0 else 0
            lift_utilizations[resource.id] = utilization
            lift_schedules[resource.id] = resource.processed_count
        
        # 计算穿梭车利用率
        rgv_utilizations = {}
        for resource in self.rgv_resources:
            total_busy = resource.total_busy_time
            utilization = total_busy / cmax if cmax > 0 else 0
            rgv_utilizations[resource.id] = utilization
        
        # 拥堵指数：最大提升机利用率 / 平均提升机利用率
        avg_lift_util = np.mean(list(lift_utilizations.values())) if lift_utilizations else 0
        max_lift_util = max(lift_utilizations.values()) if lift_utilizations else 0
        congestion_index = max_lift_util / avg_lift_util if avg_lift_util > 0 else 0
        
        state_dict = {
            'cmax': cmax,
            'total_wait': total_wait,
            'p95_wait': p95,
            'lift_utilizations': lift_utilizations,
            'rgv_utilizations': rgv_utilizations,
            'avg_lift_utilization': avg_lift_util,
            'max_lift_utilization': max_lift_util,
            'congestion_index': congestion_index,
            'lift_schedules': lift_schedules
        }
        
        return state_dict
    
    def calculate_rho_metrics(self, sequence_ids, cmax=None):
        """
        计算阻塞因子（ρ）相关指标
        
        阻塞因子是排队论中的重要概念，表示系统负载程度：
        - ρ_global = λ / (M × μ)
        - 其中 λ 是任务到达率，μ 是服务率，M 是提升机数量
        
        参数：
            sequence_ids: 任务执行序列
            cmax: 最大完成时间（如果为None，会先运行仿真获取）
        
        返回：
            rho_dict: 包含以下指标的字典
                - rho_global: 全局阻塞因子
                - rho_local: 每台提升机的局部阻塞因子字典
                - rho_cv: 局部阻塞因子的变异系数
                - lambda_arrival: 任务到达率
                - mu_service: 平均服务率
                - rho_per_lift: 各提升机的ρ值列表
        """
        # 导入扩展模块中的函数
        from src.simulation_rho_extension import calculate_rho_metrics as calc_rho
        return calc_rho(self, sequence_ids, cmax)
