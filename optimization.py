"""
optimization.py - 优化大脑
职责：NSGA-II 算法实现
它不关心仓库怎么运作，只关心数字

核心逻辑：
- 进化：交叉（OX）、变异（Swap）
- 选择：非支配排序（帕累托前沿）、拥挤度距离
- 调用：不断生成新的序列（Genome），扔给 simulation.py 跑分，优胜劣汰

将主优化目标设定为 W_total（消除等待时间），严格优先级：W_total > Cmax
"""
import random
import copy
import numpy as np
from src.config import POP_SIZE, N_GEN, P_CROSSOVER, P_MUTATION

class NSGA2Optimizer:
    """
    NSGA-II 多目标优化算法
    
    目标：同时优化 W_total 和 Cmax
    优先级：W_total > Cmax（严格优先）
    方法：进化算法 + 帕累托前沿
    """
    
    def __init__(self, simulator, num_jobs, pop_size=100, n_gen=100, p_crossover=0.8, p_mutation=0.1, use_c_eft=False,
                 use_bm=False, bm_probability=0.3, use_heuristic_seeds=True,
                 use_time_window_insertion=False):
        self.sim = simulator
        self.num_jobs = num_jobs
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.p_cross = p_crossover
        self.p_mut = p_mutation
        self.use_c_eft = use_c_eft  # 是否使用C-EFT规则解码
        # TW 时间窗左插入解码模式（用于消融实验对比）
        self.use_time_window_insertion = bool(use_time_window_insertion)
        # BM 关键块变异（论文 §3.3）
        self.use_bm = bool(use_bm)
        self.bm_probability = float(bm_probability)
        # 公平对比开关：False 时初始种群全随机，与 pymoo 等通用框架对齐
        self.use_heuristic_seeds = bool(use_heuristic_seeds)
        
        # 构建 ID 到 Index 的映射表，用于后续转换
        if hasattr(self.sim, 'jobs'):
            self.jobs_ref = self.sim.jobs
        elif hasattr(self.sim, 'sim') and hasattr(self.sim.sim, 'jobs'):
             self.jobs_ref = self.sim.sim.jobs
        else:
             raise ValueError("Optimizer cannot access jobs list from simulator")

        self.job_id_to_idx = {job.id: i for i, job in enumerate(self.jobs_ref)}
        self.fitness_cache = {}
        self.population = []

    def initialize(self):
        """
        初始化种群 - 可选启发式种子

        策略：
        1. 若 use_heuristic_seeds=True：FIFO、SPT、LPT、MOR 各 1 个 + 随机解填充
        2. 若 use_heuristic_seeds=False：全随机解（用于与通用框架做公平基线对比）
        """
        population = []
        seeds_added = 0

        if self.use_heuristic_seeds:
            print("  [Init] Generating initial population with heuristic seeds...")

            # 延迟导入避免循环依赖
            from src.strategies import baseline_fifo, spt_dispatch, lpt_dispatch, mor_dispatch

            heuristic_functions = [
                ('FIFO', baseline_fifo),
                ('SPT', spt_dispatch),
                ('LPT', lpt_dispatch),
                ('MOR', mor_dispatch)
            ]

            for heuristic_name, heuristic_func in heuristic_functions:
                try:
                    if heuristic_name == 'FIFO':
                        heuristic_ids = heuristic_func(self.jobs_ref)
                    else:
                        heuristic_ids = heuristic_func(self.jobs_ref, self.sim)

                    heuristic_genome = [self.job_id_to_idx[jid] for jid in heuristic_ids]
                    cmax, w_total, p95 = self.sim.run(heuristic_genome)
                    population.append({
                        'genome': heuristic_genome,
                        'fitness': (w_total, cmax),
                        'rank': 0,
                        'crowding_distance': 0
                    })
                    print(f"    [OK] {heuristic_name} seed added: Cmax={cmax:.2f}, W_total={w_total:.2f}")
                    seeds_added += 1
                except Exception as e:
                    print(f"    [FAIL] {heuristic_name} seed failed: {e}")
                    continue

            if seeds_added == 0:
                print("    [WARN] No heuristic seeds available, using random initialization")
        else:
            print("  [Init] Generating initial population with RANDOM seeds only (heuristic seeds disabled for fair baseline)")

        # 2. 随机解：生成剩余个体
        base_indices = list(range(self.num_jobs))
        remaining_slots = self.pop_size - seeds_added
        
        for i in range(remaining_slots):
            random_genome = base_indices[:]
            random.shuffle(random_genome)
            
            cmax, w_total, p95 = self.sim.run(random_genome)
            
            population.append({
                'genome': random_genome,
                'fitness': (w_total, cmax),
                'rank': 0,
                'crowding_distance': 0
            })
            
        self.population = population
        print(f"  [Init] Population initialized: {seeds_added} heuristic seeds + {remaining_slots} random individuals")
        return population

    def evaluate(self, population):
        """
        适应度评估 (优化版)
        
        对种群中的每个个体进行仿真评估
        使用缓存避免重复计算，并添加 LRU 缓存大小控制
        
        优化点：
        1. 控制缓存大小，避免内存无限增长
        2. 批量处理缓存命中/未命中的个体
        """
        # 限制缓存大小，防止内存无限增长 (LRU 策略)
        MAX_CACHE_SIZE = 50000
        if len(self.fitness_cache) > MAX_CACHE_SIZE:
            # 保留最近 80% 的缓存
            items = list(self.fitness_cache.items())
            self.fitness_cache = dict(items[int(MAX_CACHE_SIZE * 0.2):])
        
        # 分类处理：缓存命中和未命中的个体
        to_evaluate = []
        for ind in population:
            if ind['fitness'] is None:
                genome_key = tuple(ind['genome'])
                if genome_key in self.fitness_cache:
                    # 缓存命中
                    cmax, w_total, p95 = self.fitness_cache[genome_key]
                    ind['fitness'] = (w_total, cmax)
                else:
                    # 缓存未命中
                    to_evaluate.append((ind, genome_key))
        
        # 批量评估未命中的个体
        for ind, genome_key in to_evaluate:
            # 使用C-EFT规则和时间窗左插入解码（如果启用）
            cmax, w_total, p95 = self.sim.run(
                ind['genome'],
                use_c_eft=self.use_c_eft,
                use_time_window_insertion=self.use_time_window_insertion
            )
            ind['fitness'] = (w_total, cmax)
            self.fitness_cache[genome_key] = (cmax, w_total, p95)

    def fast_non_dominated_sort(self, pop):
        """
        快速非支配排序 (NumPy 向量化优化版)
        
        将种群分成多个前沿面（Fronts）
        Front 0: 帕累托前沿（不被任何其他解支配）
        
        优化：使用 NumPy 矩阵运算替代嵌套循环，时间复杂度从 O(N²) 常数优化
        """
        n = len(pop)
        if n == 0:
            return [[]]
        
        # 提取所有个体的适应度值到 NumPy 数组
        # fitness[0] = W_total, fitness[1] = Cmax
        fitness = np.array([ind['fitness'] for ind in pop])
        
        # 向量化支配关系计算
        # 使用标准帕累托支配：W_total_i <= W_total_j 且 Cmax_i < Cmax_j
        #                                   或 W_total_i < W_total_j 且 Cmax_i <= Cmax_j
        w_total = fitness[:, 0]
        cmax = fitness[:, 1]
        
        # 计算支配矩阵: (n, n) 的布尔矩阵
        w_less = w_total[:, np.newaxis] < w_total[np.newaxis, :]  # W_total_i < W_total_j
        w_leq = w_total[:, np.newaxis] <= w_total[np.newaxis, :]  # W_total_i <= W_total_j
        c_less = cmax[:, np.newaxis] < cmax[np.newaxis, :]  # Cmax_i < Cmax_j
        c_leq = cmax[:, np.newaxis] <= cmax[np.newaxis, :]  # Cmax_i <= Cmax_j
        
        # 标准帕累托支配关系
        dominates_matrix = (w_leq & c_less) | (w_less & c_leq)
        
        # 计算支配计数 (被多少个体支配)
        dominated_counts = np.sum(dominates_matrix, axis=0)
        
        # 构建前沿面
        fronts = [[]]
        remaining = set(range(n))
        
        # Front 0: 支配计数为 0 的个体
        for i in range(n):
            pop[i]['dominated_count'] = int(dominated_counts[i])
            pop[i]['dominates'] = []  # 清空之前的支配列表
            if dominated_counts[i] == 0:
                pop[i]['rank'] = 0
                fronts[0].append(pop[i])
                remaining.discard(i)
        
        # 构建支配列表 (哪些个体被 i 支配)
        for i in range(n):
            for j in np.where(dominates_matrix[i])[0]:
                pop[i]['dominates'].append(pop[j])
        
        # 构建后续前沿面
        current_rank = 0
        while remaining:
            next_front = []
            next_remaining = remaining.copy()
            
            for ind in fronts[current_rank]:
                for dominated_ind in ind['dominates']:
                    dominated_ind['dominated_count'] -= 1
                    if dominated_ind['dominated_count'] == 0:
                        dominated_ind['rank'] = current_rank + 1
                        next_front.append(dominated_ind)
                        next_remaining.discard(pop.index(dominated_ind))
            
            if not next_front:
                break
            
            fronts.append(next_front)
            remaining = next_remaining
            current_rank += 1
        
        return fronts
    
    def _dominates(self, ind1, ind2):
        """
        判断 ind1 是否支配 ind2
        
        使用标准帕累托支配关系，平衡 W_total 和 Cmax
        ind1 支配 ind2 当且仅当：
        - W_total1 <= W_total2 且 Cmax1 < Cmax2，或
        - W_total1 < W_total2 且 Cmax1 <= Cmax2
        - 且至少在一个目标上严格更优
        """
        f1 = ind1['fitness']
        f2 = ind2['fitness']
        # f1[0] = W_total, f1[1] = Cmax
        w1, c1 = f1[0], f1[1]
        w2, c2 = f2[0], f2[1]
        
        # 标准帕累托支配：在至少一个目标上更优，且不劣于任何目标
        if (w1 <= w2 and c1 < c2) or (w1 < w2 and c1 <= c2):
            return True
        return False
    
    def crowding_distance_assignment(self, front):
        """
        拥挤度距离分配 (NumPy 向量化优化版)
        
        在同一前沿面内，计算每个个体的拥挤度
        优先按 W_total 计算拥挤度
        
        优化：使用 NumPy 矩阵运算替代循环
        """
        n = len(front)
        if n <= 2:
            for ind in front:
                ind['crowding_distance'] = float('inf')
            return
        
        # 提取适应度值到 NumPy 数组
        fitness = np.array([ind['fitness'] for ind in front])
        w_total = fitness[:, 0]
        cmax = fitness[:, 1]
        
        # 初始化拥挤度数组
        crowding = np.zeros(n)
        
        # 按 W_total 计算拥挤度（主目标）
        w_order = np.argsort(w_total)
        crowding[w_order[0]] = float('inf')
        crowding[w_order[-1]] = float('inf')
        
        w_range = w_total[w_order[-1]] - w_total[w_order[0]]
        if w_range > 1e-10:
            w_sorted = w_total[w_order]
            # 向量化计算中间点的拥挤度
            w_diff = np.zeros(n)
            w_diff[w_order[1:-1]] = (w_sorted[2:] - w_sorted[:-2]) / w_range
            crowding += w_diff
        
        # 按 Cmax 计算拥挤度（次目标）
        c_order = np.argsort(cmax)
        crowding[c_order[0]] = float('inf')
        crowding[c_order[-1]] = float('inf')
        
        c_range = cmax[c_order[-1]] - cmax[c_order[0]]
        if c_range > 1e-10:
            c_sorted = cmax[c_order]
            # 向量化计算中间点的拥挤度
            c_diff = np.zeros(n)
            c_diff[c_order[1:-1]] = (c_sorted[2:] - c_sorted[:-2]) / c_range
            crowding += c_diff
        
        # 将结果写回个体
        for i, ind in enumerate(front):
            ind['crowding_distance'] = float(crowding[i])

    def crossover_ox(self, p1, p2):
        """
        顺序交叉 (Order Crossover, OX) - 优化版
        
        保持序列中元素的相对顺序，适用于调度问题
        
        优化：使用集合替代列表的 'in' 操作，从 O(n) 降为 O(1)
        """
        size = len(p1)
        a, b = sorted(random.sample(range(size), 2))
        
        # 使用集合存储中间段元素，O(1) 查找
        middle_set = set(p1[a:b+1])
        child = [None] * size
        child[a:b+1] = p1[a:b+1]
        
        # 从 p2 中填充剩余位置
        ptr = 0
        for gene in p2:
            if gene not in middle_set:  # O(1) 查找
                # 跳过已填充的位置
                while ptr < size and child[ptr] is not None:
                    ptr += 1
                if ptr < size:
                    child[ptr] = gene
        
        return child

    def mutation_swap(self, ind):
        """
        交换变异

        随机交换序列中的两个位置
        """
        if random.random() < self.p_mut:
            idx1, idx2 = random.sample(range(len(ind)), 2)
            ind[idx1], ind[idx2] = ind[idx2], ind[idx1]
        return ind

    def mutation_bottleneck_block(self, genome, intensity=0.5):
        """
        BM 算子：关键块邻域搜索变异（Bottleneck Mutation）

        论文 §3.3 描述的算子。通过最长路径法识别制约 W_total 的瓶颈提升机
        及其上的任务"关键块"，然后对关键块内的基因片段做局部邻域操作
        （逆序 / 邻接交换 / 位移），而不是无差别 Swap。

        实现路径：
          1. 调用 Simulator.get_critical_block 获取瓶颈任务集合（按基因位置返回索引）
          2. 在关键块中按 intensity 概率执行下列三种邻域操作之一：
             (a) reverse  — 子序列逆序
             (b) shift    — 子序列位移到序列其他位置
             (c) swap_pair— 块内两个位置互换
          3. 若关键块为空或长度 < 2，回退到普通 swap

        参数：
          genome    : 基因（int 索引序列）
          intensity : [0,1]，控制邻域操作触发概率
        返回：变异后的基因（新对象）
        """
        n = len(genome)
        if n < 4:
            return genome

        # 关键块识别（基于当前序列的仿真结果）
        # 瓶颈识别应与父级评估模式一致（use_time_window_insertion）
        try:
            # get_critical_block 接收 ID 序列
            id_seq = [self.jobs_ref[i].id for i in genome]
            critical_indices, bottleneck_lift, _ = self.sim.get_critical_block(
                id_seq,
                use_time_window_insertion=self.use_time_window_insertion
            )
        except Exception:
            return self.mutation_swap(genome[:])

        # 关键块过短：回退到普通 swap
        if not critical_indices or len(critical_indices) < 2:
            return self.mutation_swap(genome[:])

        # 限制关键块大小，避免对全基因做剧烈扰动
        max_block_size = max(3, n // 10)
        if len(critical_indices) > max_block_size:
            # 随机选取连续的子集
            start_idx = random.randint(0, len(critical_indices) - max_block_size)
            critical_indices = critical_indices[start_idx:start_idx + max_block_size]

        new_genome = genome[:]

        if random.random() < intensity:
            op = random.choice(['reverse', 'shift', 'swap_pair'])

            if op == 'reverse':
                # 对关键块对应基因位置的子序列做逆序
                positions = sorted(critical_indices)
                segment = [new_genome[p] for p in positions]
                segment.reverse()
                for p, val in zip(positions, segment):
                    new_genome[p] = val

            elif op == 'shift':
                # 把关键块整体抽出，插到序列的另一位置
                positions = sorted(critical_indices)
                block = [new_genome[p] for p in positions]
                # 移除关键块
                remaining = [g for idx, g in enumerate(new_genome) if idx not in set(positions)]
                # 在剩余序列随机位置插入
                insert_pos = random.randint(0, len(remaining))
                new_genome = remaining[:insert_pos] + block + remaining[insert_pos:]

            else:  # swap_pair
                # 块内随机两位置交换
                p1, p2 = random.sample(critical_indices, 2)
                new_genome[p1], new_genome[p2] = new_genome[p2], new_genome[p1]

        return new_genome

    def aisle_affinity_local_search(self, genome):
        """
        巷道亲和性局部搜索 (Memetic Operator)
        强行将同一巷道的任务在基因序列中聚类
        """
        if len(genome) < 2:
            return genome
            
        # 1. 随机选一个锚点任务
        idx = random.randint(0, len(genome) - 1)
        anchor_job_idx = genome[idx]
        
        try:
            anchor_job = self.jobs_ref[anchor_job_idx]
            
            # 2. 找到这个任务所在的物理巷道 (Col)
            target_col = None
            for stage in anchor_job.stages:
                if stage['type'] in ['FRGV', 'Load', 'Unload']:
                    node_str = stage.get('from_node', '')
                    # 你的 CSV 节点格式通常是 Col-Row-Layer，如 "04-032-01"
                    if '-' in node_str:
                        target_col = node_str.split('-')[0]
                        break
            
            if not target_col:
                return genome
                
            # 3. 扫描整个基因序列，把所有同巷道的任务揪出来
            same_aisle_genes = []
            other_genes = []
            
            for gene in genome:
                job = self.jobs_ref[gene]
                is_same = False
                for stage in job.stages:
                    if stage['type'] in ['FRGV', 'Load', 'Unload']:
                        node_str = stage.get('from_node', '')
                        if '-' in node_str and node_str.split('-')[0] == target_col:
                            is_same = True
                            break
                if is_same:
                    same_aisle_genes.append(gene)
                else:
                    other_genes.append(gene)
                    
            # 4. 将同巷道的任务"打包连号"，插回序列中
            insert_pos = max(0, min(idx, len(other_genes)))
            new_genome = other_genes[:insert_pos] + same_aisle_genes + other_genes[insert_pos:]
            
            return new_genome
            
        except Exception as e:
            # 容错处理，防止坐标解析异常
            return genome

    def evolve(self):
        """
        主进化循环
        
        流程：
        1. 初始化种群
        2. 进化迭代
        3. 返回帕累托前沿和收敛历史
        """
        # 记录收敛历史
        convergence_history = []
        
        # 确保已初始化
        if not self.population:
            self.initialize()
        
        for gen in range(self.n_gen):
            offspring = []
            while len(offspring) < self.pop_size:
                # 锦标赛选择
                p1, p2 = random.sample(self.population, 2)
                parent1 = p1 if p1['rank'] < p2['rank'] else p2
                p3, p4 = random.sample(self.population, 2)
                parent2 = p3 if p3['rank'] < p4['rank'] else p4
                
                # 交叉
                if random.random() < self.p_cross:
                    child_genome = self.crossover_ox(parent1['genome'], parent2['genome'])
                else:
                    child_genome = parent1['genome'][:]

                # 变异：BM 关键块变异 优先于普通 Swap（互斥）
                if self.use_bm and random.random() < self.bm_probability:
                    child_genome = self.mutation_bottleneck_block(child_genome)
                else:
                    child_genome = self.mutation_swap(child_genome)

                # 以 20% 的概率触发巷道亲和性定向变异
                if random.random() < 0.2:
                    child_genome = self.aisle_affinity_local_search(child_genome)

                offspring.append({'genome': child_genome, 'fitness': None})

            # 评估子代
            self.evaluate(offspring)
            
            # 合并种群
            self.population.extend(offspring)
            
            # 快速非支配排序
            fronts = self.fast_non_dominated_sort(self.population)
            
            # 计算拥挤度
            for front in fronts:
                self.crowding_distance_assignment(front)
            
            # 精英保留策略
            new_population = []
            i = 0
            while len(new_population) + len(fronts[i]) <= self.pop_size:
                new_population.extend(fronts[i])
                i += 1
                if i >= len(fronts): break
            
            if len(new_population) < self.pop_size and i < len(fronts):
                remaining = self.pop_size - len(new_population)
                fronts[i].sort(key=lambda x: x['crowding_distance'], reverse=True)
                new_population.extend(fronts[i][:remaining])
            
            self.population = new_population
            
            # --- 打印每代最优状态 ---
            current_fronts = self.fast_non_dominated_sort(self.population)
            if current_fronts:
                current_pareto = current_fronts[0]
                
                # 严格优先 Cmax，如果 Cmax 相同，选 W_total 小的
                # 先按 Cmax 排序，再按 W_total 排序
                current_pareto_sorted = sorted(current_pareto, key=lambda x: (x['fitness'][0], x['fitness'][1]))
                
                best_sol = current_pareto_sorted[0]
                # fitness = (w_total, cmax)
                best_w_total = best_sol['fitness'][0]
                best_cmax = best_sol['fitness'][1]
                
                # 获取 P95
                p95 = 0
                genome_key = tuple(best_sol['genome'])
                if genome_key in self.fitness_cache:
                    _, _, p95 = self.fitness_cache[genome_key]

                print(f"[Gen {gen+1}/{self.n_gen}] Best Cmax: {best_cmax:.2f}s (W_total: {best_w_total:.2f}s, P95: {p95:.2f}s)")
                
                # 记录收敛历史
                convergence_history.append({
                    'generation': gen + 1,
                    'cmax': best_cmax,
                    'w_total': best_w_total,
                    'p95': p95
                })
            
        # 返回最终的帕累托前沿和收敛历史
        final_fronts = self.fast_non_dominated_sort(self.population)
        best_solutions = final_fronts[0]
        
        # 构建帕累托前沿点集 (Cmax, W_total)
        pareto_front = [(sol['fitness'][1], sol['fitness'][0]) for sol in best_solutions]
        
        return {
            'solutions': best_solutions,
            'pareto_front': pareto_front,
            'convergence_history': convergence_history
        }

    def optimize(self):
        """
        优化方法的别名，兼容不同调用方式
        """
        return self.evolve()


class NSGA2SyncRelease(NSGA2Optimizer):
    """
    Baseline B3: NSGA-II 但不使用解耦释放（提升机释放=工序完成）
    """
    def __init__(self, simulator, num_jobs):
        self.sync_sim = copy.deepcopy(simulator)
        self.sync_sim.mode = 'adaptive_rho'
        super().__init__(self.sync_sim, num_jobs)


# ==========================================
# 改进的优化器类
# ==========================================

class RL_NSGA2Optimizer(NSGA2Optimizer):
    """
    基于 Q-learning 动态调参的改进型 NSGA-II

    论文 §3.2：RL 状态空间应包含 9 维特征，反映多维度种群状态。
    本实现：9 维特征 -> 3 维粗粒度状态（共 27 个状态），保证 Q 表可学习。
    """
    def __init__(self, simulator, num_jobs, pop_size=100, n_gen=100, use_bm=False, bm_probability=0.3,
                 use_9dim_state=True, use_heuristic_seeds=True):
        # 初始参数随便给，因为会被 RL 动态覆盖
        super().__init__(simulator, num_jobs, pop_size, n_gen, p_crossover=0.8, p_mutation=0.1,
                         use_bm=use_bm, bm_probability=bm_probability,
                         use_heuristic_seeds=use_heuristic_seeds)

        # Q-learning 参数设定
        self.alpha = 0.1  # 学习率
        self.gamma = 0.9  # 折扣因子
        self.epsilon = 0.2 # 探索率

        # 动作空间 (Action): 3 种 [P_cross, P_mut] 组合
        self.actions = [
            (0.9, 0.05), # 动作 0：高交叉，低变异（偏重局部开发/收敛）
            (0.8, 0.1),  # 动作 1：常规比例
            (0.6, 0.3)   # 动作 2：低交叉，高变异（偏重全局探索/跳出局部最优）
        ]

        # 状态空间设计：
        # use_9dim_state=False（旧版）: 2 状态（进化良好 / 陷入停滞）
        # use_9dim_state=True（新版）:  9 特征 -> 3 维聚合 -> 27 状态
        self.use_9dim_state = bool(use_9dim_state)
        if self.use_9dim_state:
            # 27 状态 x 3 动作
            self.q_table = np.zeros((27, 3))
            # 用于差分计算的历史缓冲
            self._feat_hist = []  # 每一代 9 维特征向量
        else:
            # 兼容旧版：2 状态 x 3 动作
            self.q_table = np.zeros((2, 3))

    def _compute_9dim_features(self, gen, best_cmax_hist):
        """
        计算 9 维 RL 状态特征向量（论文 §3.2）

        特征定义：
        f1: gen_progress         当前代数进度 (gen/n_gen)
        f2: best_cmax_norm       最优 Cmax 归一化（相对初代）
        f3: best_w_norm          最优 W_total 归一化
        f4: pareto_size_norm     非劣前沿规模 / pop_size
        f5: improve_rate_cmax    最近 3 代 Cmax 改进率
        f6: improve_rate_w       最近 3 代 W_total 改进率
        f7: stagnation_steps     连续无改善代数 / n_gen
        f8: diversity            种群多样性（fitness 标准差/均值）
        f9: front_spread         前沿在目标空间的分散度

        返回：(features, state_idx)
            features: list of 9 floats（用于诊断和日志）
            state_idx: int in [0, 27)（用于 Q 表索引）
        """
        pop = self.population
        if not pop or not all(p.get('fitness') is not None for p in pop):
            # 兜底
            return ([0.0] * 9, 0)
        ws = np.array([ind['fitness'][0] for ind in pop])
        cs = np.array([ind['fitness'][1] for ind in pop])

        # 提取前沿（rank=0）
        front0 = [ind for ind in pop if ind.get('rank', 0) == 0]
        if not front0:
            front0 = pop[:1]
        front_ws = np.array([ind['fitness'][0] for ind in front0])
        front_cs = np.array([ind['fitness'][1] for ind in front0])

        f1 = gen / max(self.n_gen, 1)
        c0 = best_cmax_hist[0] if best_cmax_hist else cs.min()
        c0_safe = c0 if c0 > 1e-9 else 1.0
        f2 = (cs.min() / c0_safe) if c0_safe > 0 else 1.0
        w0 = max(ws.max(), 1.0)
        f3 = ws.min() / w0
        f4 = len(front0) / max(len(pop), 1)

        # 最近 3 代改进率
        if len(best_cmax_hist) >= 3 and best_cmax_hist[-3] > 1e-9:
            f5 = (best_cmax_hist[-3] - best_cmax_hist[-1]) / best_cmax_hist[-3]
        else:
            f5 = 0.0
        f5 = max(min(f5, 1.0), -1.0)

        # 用前沿中最佳 W 的近似改进
        if len(self._feat_hist) >= 3 and self._feat_hist[-3][2] > 1e-9:
            w_now = ws.min() / w0
            f6 = (self._feat_hist[-3][2] - w_now) / max(self._feat_hist[-3][2], 1e-9)
        else:
            f6 = 0.0
        f6 = max(min(f6, 1.0), -1.0)

        # 停滞代数（连续无 Cmax 改善的代数）
        stagnation = 0
        if len(best_cmax_hist) >= 2:
            for k in range(len(best_cmax_hist) - 1, 0, -1):
                if best_cmax_hist[k] >= best_cmax_hist[k - 1] - 1e-9:
                    stagnation += 1
                else:
                    break
        f7 = stagnation / max(self.n_gen, 1)

        # 多样性：CV of population fitness
        cs_mean = cs.mean() if cs.mean() > 0 else 1.0
        f8 = float(cs.std() / cs_mean)
        f8 = min(f8, 1.0)

        # 前沿散布：(cmax_max - cmax_min) / cmax_mean
        if len(front0) > 1 and front_cs.mean() > 0:
            f9 = float((front_cs.max() - front_cs.min()) / front_cs.mean())
        else:
            f9 = 0.0
        f9 = min(f9, 1.0)

        features = [f1, f2, f3, f4, f5, f6, f7, f8, f9]

        # 9 维特征聚合为 3 个"维度信号"，每个三档（低/中/高），共 27 个状态
        # 维度 A：进化阶段（基于 f1, f7）—— 早期/中期/末期或停滞
        # 维度 B：解质量梯度（基于 f5, f6）—— 强改进/弱改进/无改进
        # 维度 C：多样性（基于 f4, f8, f9）—— 多样/中等/收敛

        progress_score = 0.5 * f1 + 0.5 * (1.0 - f7)  # 综合进度
        if progress_score < 0.35:
            A = 0  # 早期
        elif progress_score < 0.70:
            A = 1  # 中期
        else:
            A = 2  # 末期/停滞

        improve_score = 0.5 * f5 + 0.5 * f6
        if improve_score > 0.05:
            B = 0  # 强改进
        elif improve_score > -0.02:
            B = 1  # 弱改进/平稳
        else:
            B = 2  # 退步/停滞

        diversity_score = 0.4 * f4 + 0.4 * f8 + 0.2 * f9
        if diversity_score > 0.5:
            C = 0  # 多样
        elif diversity_score > 0.2:
            C = 1  # 中等
        else:
            C = 2  # 收敛/单一

        state_idx = A * 9 + B * 3 + C  # 0..26
        return (features, state_idx)
        
    def evolve(self):
        self.initialize()
        best_fitness_history = []
        convergence_history = []  # 记录收敛历史

        for gen in range(self.n_gen):
            # 1. 确定当前状态
            if self.use_9dim_state:
                features, current_state = self._compute_9dim_features(gen, best_fitness_history)
                self._feat_hist.append(features)
            else:
                current_state = 0
                if gen > 5:
                    if best_fitness_history[-1] >= best_fitness_history[-5]:
                        current_state = 1

            # 2. 动作选择 (epsilon-greedy)
            if random.random() < self.epsilon:
                action_idx = random.randint(0, 2)
            else:
                action_idx = int(np.argmax(self.q_table[current_state]))

            # 应用动作
            self.p_cross, self.p_mut = self.actions[action_idx]

            # 3. 执行标准的 NSGA-II 交叉变异与选择逻辑
            offspring = []
            while len(offspring) < self.pop_size:
                p1, p2 = random.sample(self.population, 2)
                parent1 = p1 if p1['rank'] < p2['rank'] else p2
                p3, p4 = random.sample(self.population, 2)
                parent2 = p3 if p3['rank'] < p4['rank'] else p4

                if random.random() < self.p_cross:
                    child_genome = self.crossover_ox(parent1['genome'], parent2['genome'])
                else:
                    child_genome = parent1['genome'][:]

                # BM 关键块变异 vs. 普通 Swap（互斥）
                if self.use_bm and random.random() < self.bm_probability:
                    child_genome = self.mutation_bottleneck_block(child_genome)
                else:
                    child_genome = self.mutation_swap(child_genome)

                # 以 20% 的概率触发巷道亲和性定向变异
                if random.random() < 0.2:
                    child_genome = self.aisle_affinity_local_search(child_genome)

                offspring.append({'genome': child_genome, 'fitness': None})

            self.evaluate(offspring)
            self.population.extend(offspring)
            fronts = self.fast_non_dominated_sort(self.population)
            for front in fronts:
                self.crowding_distance_assignment(front)

            # 精英保留
            new_population = []
            i = 0
            while len(new_population) + len(fronts[i]) <= self.pop_size:
                new_population.extend(fronts[i])
                i += 1
                if i >= len(fronts): break
            if len(new_population) < self.pop_size and i < len(fronts):
                remaining = self.pop_size - len(new_population)
                fronts[i].sort(key=lambda x: x['crowding_distance'], reverse=True)
                new_population.extend(fronts[i][:remaining])
            self.population = new_population

            # 4. 计算奖励 (Reward) 并更新 Q 表
            current_fronts = self.fast_non_dominated_sort(self.population)
            current_best_cmax = min([ind['fitness'][1] for ind in current_fronts[0]])
            current_best_wtotal = min([ind['fitness'][0] for ind in current_fronts[0]])
            best_fitness_history.append(current_best_cmax)

            # 记录收敛历史
            convergence_history.append({
                'generation': gen + 1,
                'cmax': current_best_cmax,
                'w_total': current_best_wtotal,
                'state': current_state,
                'action': action_idx
            })

            # 奖励机制：双目标（W_total 优先 > Cmax）综合奖励
            if gen > 0:
                prev_cmax = best_fitness_history[-2]
                cmax_reward = 0.0
                if current_best_cmax < prev_cmax - 1e-6:
                    cmax_reward = 0.5
                # W_total 维度的奖励（论文将其设为主要目标）
                w_reward = 0.0
                if len(convergence_history) >= 2:
                    prev_w = convergence_history[-2]['w_total']
                    if current_best_wtotal < prev_w - 1e-6:
                        w_reward = 1.0
                    elif current_best_wtotal > prev_w + 1e-6:
                        w_reward = -0.3
                reward = cmax_reward + w_reward
                if reward == 0.0:
                    reward = -0.2  # 平稳但无改进
            else:
                reward = 0.0

            # 更新下一个状态
            if self.use_9dim_state:
                _, next_state = self._compute_9dim_features(gen + 1, best_fitness_history)
            else:
                next_state = 0
                if gen > 4 and best_fitness_history[-1] >= best_fitness_history[-5]:
                    next_state = 1

            # Q-learning 公式更新
            self.q_table[current_state, action_idx] = self.q_table[current_state, action_idx] + \
                self.alpha * (reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[current_state, action_idx])

            if self.use_9dim_state:
                print(f"[RL-NSGA-II Gen {gen+1}] State27={current_state}, Action(Pc,Pm): {self.actions[action_idx]}, Best Cmax: {current_best_cmax:.2f}, W_total: {current_best_wtotal:.2f}")
            else:
                print(f"[RL-NSGA-II Gen {gen+1}] State: {current_state}, Action(Pc,Pm): {self.actions[action_idx]}, Best Cmax: {current_best_cmax:.2f}")

        final_fronts = self.fast_non_dominated_sort(self.population)
        best_solutions = final_fronts[0]
        
        # 构建帕累托前沿点集和收敛历史
        pareto_front = [(sol['fitness'][1], sol['fitness'][0]) for sol in best_solutions]
        
        return {
            'solutions': best_solutions,
            'pareto_front': pareto_front,
            'convergence_history': convergence_history
        }


class IGAOptimizer(NSGA2Optimizer):
    """
    免疫遗传算法 (Immune Genetic Algorithm)
    """
    def extract_vaccine(self):
        """提取疫苗：从当前代最优的解中，提取一段优秀的工序序列"""
        best_ind = min(self.population, key=lambda x: x['fitness'][1]) # 取 Cmax 最小的
        genome = best_ind['genome']
        # 随机抽取长度为 3 的优秀基因片段作为疫苗
        start = random.randint(0, len(genome) - 4)
        return genome[start:start+3]

    def vaccinate(self, genome, vaccine):
        """接种疫苗：将优秀片段强制植入后代"""
        if random.random() < 0.2: # 20% 接种率
            new_genome = [g for g in genome if g not in vaccine]
            insert_pos = random.randint(0, len(new_genome))
            new_genome = new_genome[:insert_pos] + vaccine + new_genome[insert_pos:]
            return new_genome
        return genome

    def evolve(self):
        self.initialize()
        convergence_history = []  # 记录收敛历史
        
        for gen in range(self.n_gen):
            vaccine = self.extract_vaccine()
            offspring = []
            while len(offspring) < self.pop_size:
                p1, p2 = random.sample(self.population, 2)
                parent1 = p1 if p1['rank'] < p2['rank'] else p2
                p3, p4 = random.sample(self.population, 2)
                parent2 = p3 if p3['rank'] < p4['rank'] else p4
                
                if random.random() < self.p_cross:
                    child_genome = self.crossover_ox(parent1['genome'], parent2['genome'])
                else:
                    child_genome = parent1['genome'][:]
                
                child_genome = self.mutation_swap(child_genome)
                
                # 以 20% 的概率触发巷道亲和性定向变异
                if random.random() < 0.2:
                    child_genome = self.aisle_affinity_local_search(child_genome)
                
                # IGA 特有：接种疫苗
                child_genome = self.vaccinate(child_genome, vaccine)
                offspring.append({'genome': child_genome, 'fitness': None})
            
            self.evaluate(offspring)
            self.population.extend(offspring)
            
            # IGA 免疫选择（保留不退化的个体），此处简化复用 NSGA-II 的精英策略
            fronts = self.fast_non_dominated_sort(self.population)
            for front in fronts:
                self.crowding_distance_assignment(front)
            
            new_population = []
            i = 0
            while len(new_population) + len(fronts[i]) <= self.pop_size:
                new_population.extend(fronts[i])
                i += 1
                if i >= len(fronts): break
            if len(new_population) < self.pop_size and i < len(fronts):
                remaining = self.pop_size - len(new_population)
                fronts[i].sort(key=lambda x: x['crowding_distance'], reverse=True)
                new_population.extend(fronts[i][:remaining])
            self.population = new_population
            
            # 记录收敛历史
            current_fronts = self.fast_non_dominated_sort(self.population)
            if current_fronts:
                current_pareto = current_fronts[0]
                best_cmax = min([ind['fitness'][1] for ind in current_pareto])
                best_wtotal = min([ind['fitness'][0] for ind in current_pareto])
                convergence_history.append({
                    'generation': gen + 1,
                    'cmax': best_cmax,
                    'w_total': best_wtotal
                })

        final_fronts = self.fast_non_dominated_sort(self.population)
        best_solutions = final_fronts[0]
        
        # 构建帕累托前沿点集和收敛历史
        pareto_front = [(sol['fitness'][1], sol['fitness'][0]) for sol in best_solutions]
        
        return {
            'solutions': best_solutions,
            'pareto_front': pareto_front,
            'convergence_history': convergence_history
        }


class MOEADOptimizer(NSGA2Optimizer):
    """
    基于分解的多目标算法适配版 (MOEA/D)
    """
    def __init__(self, simulator, num_jobs, pop_size=100, n_gen=100):
        super().__init__(simulator, num_jobs, pop_size, n_gen)
        # 生成权重向量 (2 维目标)
        self.weights = [(i/(pop_size-1), 1 - i/(pop_size-1)) for i in range(pop_size)]
        self.T = max(2, int(0.1 * pop_size)) # 邻居数量
        self.ideal_point = [float('inf'), float('inf')] # 理想点 (Z*)
        
    def evolve(self):
        self.initialize()
        # 初始化理想点
        for ind in self.population:
            w_total, cmax = ind['fitness']
            self.ideal_point[0] = min(self.ideal_point[0], w_total)
            self.ideal_point[1] = min(self.ideal_point[1], cmax)
            
        for gen in range(self.n_gen):
            for i in range(self.pop_size):
                # 随机选择两个邻居进行交叉变异
                neighbors = [random.randint(max(0, i-self.T), min(self.pop_size-1, i+self.T)) for _ in range(2)]
                p1, p2 = self.population[neighbors[0]], self.population[neighbors[1]]
                
                if random.random() < self.p_cross:
                    y_genome = self.crossover_ox(p1['genome'], p2['genome'])
                else:
                    y_genome = p1['genome'][:]
                y_genome = self.mutation_swap(y_genome)
                
                # 评估新解
                y_cmax, y_w_total, _ = self.sim.run(y_genome)
                y_fitness = (y_w_total, y_cmax)
                
                # 更新理想点
                self.ideal_point[0] = min(self.ideal_point[0], y_w_total)
                self.ideal_point[1] = min(self.ideal_point[1], y_cmax)
                
                # Tchebycheff 方法更新邻居
                for j in range(max(0, i-self.T), min(self.pop_size, i+self.T)):
                    weight = self.weights[j]
                    # 计算原解的切比雪夫距离
                    f_old = max(weight[0]*abs(self.population[j]['fitness'][0] - self.ideal_point[0]),
                                weight[1]*abs(self.population[j]['fitness'][1] - self.ideal_point[1]))
                    # 计算新解的切比雪夫距离
                    f_new = max(weight[0]*abs(y_fitness[0] - self.ideal_point[0]),
                                weight[1]*abs(y_fitness[1] - self.ideal_point[1]))
                    
                    # 如果新解在这个方向上更好，替换邻居
                    if f_new < f_old:
                        self.population[j] = {'genome': y_genome, 'fitness': y_fitness, 'rank': 0, 'crowding_distance': 0}

        # 最后返回前沿
        final_fronts = self.fast_non_dominated_sort(self.population)
        return final_fronts[0]
