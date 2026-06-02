"""
ablation_variants.py - 消融实验变体模块

包含六个算法变体（M0-M5）：
- M0: FIFO 先到先服务 (最简Baseline)
- M1: 标准 NSGA-II (Baseline)
- M2: NSGA-II + 时间窗左插入解码 (TW)
- M3: NSGA-II + TW + 关键块邻域搜索变异 (BM)
- M4: RL-NSGA-II 纯数学状态版
- M5: RL-NSGA-II 数学+物理双驱状态版 (Proposed)

职责：为消融实验提供标准化的算法实现
"""

import random
import copy
import numpy as np
from src.optimization import NSGA2Optimizer


class M0_FIFO:
    """
    M0: FIFO (First-In-First-Out) 先到先服务
    
    配置：
    - 无搜索，仅按任务到达顺序执行
    - 用于展示问题本身的难度和搜索空间的下界
    
    目的：作为性能地板，证明遗传算法的搜索价值
    """
    
    def __init__(self, simulator, num_jobs, pop_size=100, n_gen=100):
        self.sim = simulator
        self.num_jobs = num_jobs
        self.pop_size = pop_size
        self.n_gen = n_gen
    
    def optimize(self):
        """
        FIFO优化：按任务ID顺序执行
        
        返回：
            best_individual, best_fitness
        """
        # FIFO顺序：按任务编号排序
        sequence = list(range(self.num_jobs))
        
        # 评估
        cmax, w_total, p95 = self.sim.run(sequence, use_time_window_insertion=False)
        
        # 构造与NSGA-II兼容的返回格式
        best_ind = {
            'genome': sequence,
            'fitness': (w_total, cmax)
        }
        
        return best_ind, (cmax, w_total, p95)


class M1_BaseNSGA2(NSGA2Optimizer):
    """
    M1: 标准 NSGA-II (Baseline)
    
    配置：
    - 固定交叉变异率 (Pc=0.8, Pm=0.1)
    - 半主动解码（工序只能排在最后，无缝隙插入）
    - 随机位置交换变异
    
    目的：作为性能的地板，垫底用
    """
    
    def __init__(self, simulator, num_jobs, pop_size=100, n_gen=100):
        super().__init__(simulator, num_jobs, pop_size, n_gen, 
                        p_crossover=0.8, p_mutation=0.1)
    
    def evaluate(self, population):
        """评估：不使用时间窗插入"""
        for ind in population:
            if ind['fitness'] is None:
                genome_key = tuple(ind['genome'])
                if genome_key in self.fitness_cache:
                    cmax, w_total, p95 = self.fitness_cache[genome_key]
                    ind['fitness'] = (w_total, cmax)
                else:
                    # M1: 不使用时间窗插入
                    cmax, w_total, p95 = self.sim.run(ind['genome'], 
                                                      use_time_window_insertion=False)
                    ind['fitness'] = (w_total, cmax)
                    self.fitness_cache[genome_key] = (cmax, w_total, p95)


class M1b_BaseWithBM(NSGA2Optimizer):
    """
    M1b: Base NSGA-II + BM（无TW）

    配置：
    - 与 M1 一致，但变异算子使用关键块邻域搜索（BM）替代随机交换
    - 不使用时间窗插入解码（use_time_window_insertion=False）

    目的：与 M1 对比可评估 BM 算子在无 TW 下的独立贡献；
         与 M2 对比可评估 BM 与 TW 的交互效应
    """

    def __init__(self, simulator, num_jobs, pop_size=100, n_gen=100, bm_probability=0.3):
        super().__init__(simulator, num_jobs, pop_size, n_gen,
                        p_crossover=0.8, p_mutation=0.1,
                        use_bm=True, bm_probability=bm_probability,
                        use_time_window_insertion=False)

    def evaluate(self, population):
        """评估：不使用时间窗插入"""
        for ind in population:
            if ind['fitness'] is None:
                genome_key = tuple(ind['genome'])
                if genome_key in self.fitness_cache:
                    cmax, w_total, p95 = self.fitness_cache[genome_key]
                    ind['fitness'] = (w_total, cmax)
                else:
                    cmax, w_total, p95 = self.sim.run(ind['genome'],
                                                      use_time_window_insertion=False)
                    ind['fitness'] = (w_total, cmax)
                    self.fitness_cache[genome_key] = (cmax, w_total, p95)


class M2_TimewindowNSGA2(NSGA2Optimizer):
    """
    M2: NSGA-II + 时间窗左插入解码 (TW)

    配置：
    - 在 M1 的基础上，仿真器启用时间窗左插入机制（见缝插针）
    - 随机位置交换变异

    目的：证明针对三维跨层特性的物理约束解码策略，
         能有效填补时间缝隙，初步降低 Cmax 和 W_total
    """
    
    def __init__(self, simulator, num_jobs, pop_size=100, n_gen=100):
        super().__init__(simulator, num_jobs, pop_size, n_gen,
                        p_crossover=0.8, p_mutation=0.1)
    
    def evaluate(self, population):
        """评估：启用时间窗左插入"""
        for ind in population:
            if ind['fitness'] is None:
                genome_key = tuple(ind['genome'])
                if genome_key in self.fitness_cache:
                    cmax, w_total, p95 = self.fitness_cache[genome_key]
                    ind['fitness'] = (w_total, cmax)
                else:
                    # M2: 启用时间窗插入
                    cmax, w_total, p95 = self.sim.run(ind['genome'],
                                                      use_time_window_insertion=True)
                    ind['fitness'] = (w_total, cmax)
                    self.fitness_cache[genome_key] = (cmax, w_total, p95)


class M3_CriticalBlockNSGA2(NSGA2Optimizer):
    """
    M3: NSGA-II + TW + 关键块邻域搜索变异 (BM)
    
    配置：
    - 在 M2 基础上，变异算子不再是随机交换
    - 提取仿真中"排队等待最长的那台提升机"上的任务（关键块）
    - 对关键块内的任务进行 reverse/shift/swap_pair 三种邻域操作
    
    目的：证明面向物理瓶颈的靶向变异，比盲目的随机变异
         具有更强的局部穿透力，能加速算法收敛
    
    注意：此实现与 optimization.py 中的 NSGA2Optimizer.evolve() 
         BM 模块保持一致（reverse/shift/swap_pair 三种操作）
    """
    
    def __init__(self, simulator, num_jobs, pop_size=100, n_gen=100, bm_probability=0.3):
        super().__init__(simulator, num_jobs, pop_size, n_gen,
                        p_crossover=0.8, p_mutation=0.1)
        # BM 关键块变异概率（与 optimization.py 一致，默认 0.3）
        self.bm_probability = float(bm_probability)
    
    def evaluate(self, population):
        """评估：启用时间窗插入"""
        for ind in population:
            if ind['fitness'] is None:
                genome_key = tuple(ind['genome'])
                if genome_key in self.fitness_cache:
                    cmax, w_total, p95 = self.fitness_cache[genome_key]
                    ind['fitness'] = (w_total, cmax)
                else:
                    cmax, w_total, p95 = self.sim.run(ind['genome'],
                                                      use_time_window_insertion=True)
                    ind['fitness'] = (w_total, cmax)
                    self.fitness_cache[genome_key] = (cmax, w_total, p95)
    
    def mutation_bottleneck_block(self, genome, intensity=0.5):
        """
        BM 算子：关键块邻域搜索变异（Bottleneck Mutation）
        
        与 optimization.py 中的 NSGA2Optimizer.mutation_bottleneck_block() 一致。
        
        通过最长路径法识别制约 W_total 的瓶颈提升机及其上的任务"关键块"，
        然后对关键块内的基因片段做局部邻域操作（reverse/shift/swap_pair），
        而不是无差别 Swap。
        
        参数：
          genome    : 基因（int 索引序列）
          intensity : [0,1]，控制邻域操作触发概率
        返回：变异后的基因（新对象）
        """
        n = len(genome)
        if n < 4:
            return genome

        # 关键块识别：需要将 genome 索引转换为 job IDs
        # M3 启用了 TW，瓶颈识别应与其一致
        try:
            # genome 中的值是 0~num_jobs-1 的索引，需要转换为 job.id
            id_seq = [self.jobs_ref[i].id for i in genome]
            critical_indices, bottleneck_lift, _ = self.sim.get_critical_block(
                id_seq, use_time_window_insertion=True)
        except Exception:
            return self.mutation_swap_on_genome(genome[:])

        # 关键块过短：回退到普通 swap
        if not critical_indices or len(critical_indices) < 2:
            return self.mutation_swap_on_genome(genome[:])

        # 限制关键块大小，避免对全基因做剧烈扰动
        max_block_size = max(3, n // 10)
        if len(critical_indices) > max_block_size:
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
    
    def mutation_swap_on_genome(self, genome):
        """
        随机交换变异（操作基因组列表）
        """
        if random.random() > self.p_mut:
            return genome
        
        genome = genome[:]
        idx1, idx2 = random.sample(range(len(genome)), 2)
        genome[idx1], genome[idx2] = genome[idx2], genome[idx1]
        return genome
    
    def aisle_affinity_local_search(self, genome):
        """
        巷道亲和性局部搜索 (Memetic Operator)
        与 optimization.py 中的 NSGA2Optimizer.aisle_affinity_local_search() 一致
        """
        if len(genome) < 2:
            return genome
            
        idx = random.randint(0, len(genome) - 1)
        anchor_job_idx = genome[idx]
        
        try:
            anchor_job = self.jobs_ref[anchor_job_idx]
            
            target_col = None
            for stage in anchor_job.stages:
                if stage['type'] in ['FRGV', 'Load', 'Unload']:
                    node_str = stage.get('from_node', '')
                    if '-' in node_str:
                        target_col = node_str.split('-')[0]
                        break
            
            if not target_col:
                return genome
                
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
                    
            insert_pos = max(0, min(idx, len(other_genes)))
            new_genome = other_genes[:insert_pos] + same_aisle_genes + other_genes[insert_pos:]
            
            return new_genome
            
        except Exception:
            return genome
    
    def evolve(self):
        """重写进化循环，使用关键块变异（与 optimization.py 一致）"""
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
                
                # M3: 关键块变异（BM）- 使用与 optimization.py 一致的实现
                if random.random() < self.bm_probability:
                    child_genome = self.mutation_bottleneck_block(child_genome)
                else:
                    child_genome = self.mutation_swap_on_genome(child_genome)
                
                # 以 20% 的概率触发巷道亲和性定向变异
                if random.random() < 0.2:
                    child_genome = self.aisle_affinity_local_search(child_genome)
                
                offspring.append({'genome': child_genome, 'fitness': None})
            
            # 评估子代
            self.evaluate(offspring)
            
            # 合并种群
            self.population.extend(offspring)
            
            # 非支配排序和精英保留
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
            
            # 打印进度
            current_fronts = self.fast_non_dominated_sort(self.population)
            if current_fronts:
                current_pareto = current_fronts[0]
                current_pareto_sorted = sorted(current_pareto, 
                                               key=lambda x: (x['fitness'][0], x['fitness'][1]))
                best_sol = current_pareto_sorted[0]
                best_w_total = best_sol['fitness'][0]
                best_cmax = best_sol['fitness'][1]
                
                p95 = 0
                genome_key = tuple(best_sol['genome'])
                if genome_key in self.fitness_cache:
                    _, _, p95 = self.fitness_cache[genome_key]
                
                print(f"[M3 Gen {gen+1}/{self.n_gen}] Best Cmax: {best_cmax:.2f}s "
                      f"(W_total: {best_w_total:.2f}s, P95: {p95:.2f}s)")
        
        final_fronts = self.fast_non_dominated_sort(self.population)
        return final_fronts[0]


class M4_RL_MathOnly(NSGA2Optimizer):
    """
    M4: RL-NSGA-II（纯数学状态版）
    
    配置：
    - 在 M3 基础上加入 Q-learning 动态调参
    - 智能体的状态仅包含种群适应度停滞和多样性指标
      （这也是目前绝大多数文献的做法）
    
    目的：制造一个"传统前沿算法"作为靶子，
         证明单纯利用数学特征进行调参存在局限性
    
    注意：BM 算子与 optimization.py 一致（reverse/shift/swap_pair）
    """
    
    def __init__(self, simulator, num_jobs, pop_size=100, n_gen=100, bm_probability=0.3):
        super().__init__(simulator, num_jobs, pop_size, n_gen)
        
        # Q-learning 参数
        self.alpha = 0.1  # 学习率
        self.gamma = 0.9  # 折扣因子
        self.epsilon = 0.2  # 探索率
        
        # 动作空间：3种 (Pc, Pm) 组合
        self.actions = [
            (0.9, 0.05),  # 动作 0：高交叉，低变异（偏重局部开发）
            (0.8, 0.1),   # 动作 1：常规比例
            (0.6, 0.3)    # 动作 2：低交叉，高变异（偏重全局探索）
        ]
        
        # M4: 状态空间仅基于数学特征
        # 状态 0-3：基于收敛停滞和种群多样性
        self.q_table = np.zeros((4, 3))
        # BM 概率（与 optimization.py 一致，默认 0.3）
        self.bm_probability = float(bm_probability)
    
    def evaluate(self, population):
        """评估：启用时间窗插入"""
        for ind in population:
            if ind['fitness'] is None:
                genome_key = tuple(ind['genome'])
                if genome_key in self.fitness_cache:
                    cmax, w_total, p95 = self.fitness_cache[genome_key]
                    ind['fitness'] = (w_total, cmax)
                else:
                    cmax, w_total, p95 = self.sim.run(ind['genome'],
                                                      use_time_window_insertion=True)
                    ind['fitness'] = (w_total, cmax)
                    self.fitness_cache[genome_key] = (cmax, w_total, p95)
    
    def calculate_population_diversity(self, population):
        """
        计算种群多样性（数学指标）
        
        基于适应度值的标准差计算
        """
        if len(population) < 2:
            return 0.0
        
        fitness_values = np.array([ind['fitness'] for ind in population if ind['fitness']])
        if len(fitness_values) == 0:
            return 0.0
        
        # 计算归一化的标准差
        std_w = np.std(fitness_values[:, 0])
        std_c = np.std(fitness_values[:, 1])
        
        # 多样性阈值判断
        diversity_score = (std_w + std_c) / 2.0
        return diversity_score
    
    def mutation_bottleneck_block(self, genome, intensity=0.5):
        """
        BM 算子：关键块邻域搜索变异（与 optimization.py 一致）
        """
        n = len(genome)
        if n < 4:
            return genome

        # M4 启用了 TW，瓶颈识别应与其一致
        try:
            id_seq = [self.jobs_ref[i].id for i in genome]
            critical_indices, bottleneck_lift, _ = self.sim.get_critical_block(
                id_seq, use_time_window_insertion=True)
        except Exception:
            return self.mutation_swap_on_genome(genome[:])

        if not critical_indices or len(critical_indices) < 2:
            return self.mutation_swap_on_genome(genome[:])

        max_block_size = max(3, n // 10)
        if len(critical_indices) > max_block_size:
            start_idx = random.randint(0, len(critical_indices) - max_block_size)
            critical_indices = critical_indices[start_idx:start_idx + max_block_size]

        new_genome = genome[:]

        if random.random() < intensity:
            op = random.choice(['reverse', 'shift', 'swap_pair'])

            if op == 'reverse':
                positions = sorted(critical_indices)
                segment = [new_genome[p] for p in positions]
                segment.reverse()
                for p, val in zip(positions, segment):
                    new_genome[p] = val

            elif op == 'shift':
                positions = sorted(critical_indices)
                block = [new_genome[p] for p in positions]
                remaining = [g for idx, g in enumerate(new_genome) if idx not in set(positions)]
                insert_pos = random.randint(0, len(remaining))
                new_genome = remaining[:insert_pos] + block + remaining[insert_pos:]

            else:  # swap_pair
                p1, p2 = random.sample(critical_indices, 2)
                new_genome[p1], new_genome[p2] = new_genome[p2], new_genome[p1]

        return new_genome
    
    def mutation_swap_on_genome(self, genome):
        """随机交换变异（操作基因组列表）"""
        if random.random() > self.p_mut:
            return genome
        
        genome = genome[:]
        idx1, idx2 = random.sample(range(len(genome)), 2)
        genome[idx1], genome[idx2] = genome[idx2], genome[idx1]
        return genome
    
    def evolve(self):
        """M4: 基于纯数学状态的 RL-NSGA-II"""
        self.initialize()
        
        best_fitness_history = []
        diversity_history = []
        
        for gen in range(self.n_gen):
            # M4: 计算数学状态
            diversity = self.calculate_population_diversity(self.population)
            diversity_history.append(diversity)
            
            # 判断是否停滞
            stagnation = 0
            if gen > 5:
                if best_fitness_history[-1] >= best_fitness_history[-5]:
                    stagnation = 1
            
            # 判断是否多样性低
            diversity_low = 1 if diversity < 50.0 else 0  # 阈值可调
            
            # 状态编码
            current_state = stagnation * 2 + diversity_low  # 0-3
            
            # 动作选择
            if random.random() < self.epsilon:
                action_idx = random.randint(0, 2)
            else:
                action_idx = np.argmax(self.q_table[current_state])
            
            # 应用动作
            self.p_cross, self.p_mut = self.actions[action_idx]
            
            # 生成子代
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
                
                # M4: 关键块变异（BM）
                if random.random() < self.bm_probability:
                    child_genome = self.mutation_bottleneck_block(child_genome)
                else:
                    child_genome = self.mutation_swap_on_genome(child_genome)
                
                # 以 20% 的概率触发巷道亲和性定向变异
                if random.random() < 0.2:
                    child_genome = self.aisle_affinity_local_search(child_genome)
                
                offspring.append({'genome': child_genome, 'fitness': None})
            
            self.evaluate(offspring)
            self.population.extend(offspring)
            
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
            
            # 计算奖励并更新 Q 表
            current_best_cmax = min([ind['fitness'][1] for ind in fronts[0]])
            best_fitness_history.append(current_best_cmax)
            
            reward = 0.0
            if gen > 0 and current_best_cmax < best_fitness_history[-2]:
                reward = 1.0
            else:
                reward = -0.5
            
            # 下一状态
            next_stagnation = 0
            if gen > 4 and best_fitness_history[-1] >= best_fitness_history[-5]:
                next_stagnation = 1
            next_diversity_low = 1 if diversity < 50.0 else 0
            next_state = next_stagnation * 2 + next_diversity_low
            
            # Q-learning 更新
            self.q_table[current_state, action_idx] = self.q_table[current_state, action_idx] + \
                self.alpha * (reward + self.gamma * np.max(self.q_table[next_state]) - 
                             self.q_table[current_state, action_idx])
            
            print(f"[M4 RL-Math Gen {gen+1}] State: {current_state}, Action: {action_idx}, "
                  f"Best Cmax: {current_best_cmax:.2f}, Diversity: {diversity:.2f}")
        
        final_fronts = self.fast_non_dominated_sort(self.population)
        return final_fronts[0]


class M5_RL_PhysicalMath(NSGA2Optimizer):
    """
    M5: RL-NSGA-II（数学+物理双驱状态版 - Proposed）
    
    配置：
    - 智能体的状态同时包含"种群多样性（数学）"
      与"提升机拥堵指数（物理仿真）"
    
    目的：绝杀！证明引入数字孪生底层的真实物理状态后，
         强化学习能做出更精准的调参决策，达成全局最优
    
    注意：BM 算子与 optimization.py 一致（reverse/shift/swap_pair）
    """
    
    def __init__(self, simulator, num_jobs, pop_size=100, n_gen=100, bm_probability=0.3):
        super().__init__(simulator, num_jobs, pop_size, n_gen)
        
        # Q-learning 参数
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.2
        
        # 动作空间
        self.actions = [
            (0.9, 0.05),
            (0.8, 0.1),
            (0.6, 0.3)
        ]
        
        # M5: 扩展状态空间
        self.q_table = np.zeros((8, 3))
        # BM 概率（与 optimization.py 一致，默认 0.3）
        self.bm_probability = float(bm_probability)
    
    def evaluate(self, population):
        """评估：启用时间窗插入"""
        for ind in population:
            if ind['fitness'] is None:
                genome_key = tuple(ind['genome'])
                if genome_key in self.fitness_cache:
                    cmax, w_total, p95 = self.fitness_cache[genome_key]
                    ind['fitness'] = (w_total, cmax)
                else:
                    cmax, w_total, p95 = self.sim.run(ind['genome'],
                                                      use_time_window_insertion=True)
                    ind['fitness'] = (w_total, cmax)
                    self.fitness_cache[genome_key] = (cmax, w_total, p95)
    
    def calculate_population_diversity(self, population):
        """计算种群多样性"""
        if len(population) < 2:
            return 0.0
        
        fitness_values = np.array([ind['fitness'] for ind in population if ind['fitness']])
        if len(fitness_values) == 0:
            return 0.0
        
        std_w = np.std(fitness_values[:, 0])
        std_c = np.std(fitness_values[:, 1])
        return (std_w + std_c) / 2.0
    
    def get_congestion_state(self, best_individual):
        """
        获取拥堵指数状态（物理状态）
        
        通过仿真器获取真实的物理系统状态
        """
        genome = best_individual['genome']
        physical_state = self.sim.get_physical_state(genome)
        
        # 拥堵指数：最大提升机利用率 / 平均提升机利用率
        congestion_index = physical_state.get('congestion_index', 0)
        
        # 判断是否高拥堵
        return congestion_index > 1.5  # 阈值：最大利用率是平均的1.5倍以上
    
    def mutation_bottleneck_block(self, genome, intensity=0.5):
        """
        BM 算子：关键块邻域搜索变异（与 optimization.py 一致）
        """
        n = len(genome)
        if n < 4:
            return genome

        # M5 启用了 TW，瓶颈识别应与其一致
        try:
            id_seq = [self.jobs_ref[i].id for i in genome]
            critical_indices, bottleneck_lift, _ = self.sim.get_critical_block(
                id_seq, use_time_window_insertion=True)
        except Exception:
            return self.mutation_swap_on_genome(genome[:])

        if not critical_indices or len(critical_indices) < 2:
            return self.mutation_swap_on_genome(genome[:])

        max_block_size = max(3, n // 10)
        if len(critical_indices) > max_block_size:
            start_idx = random.randint(0, len(critical_indices) - max_block_size)
            critical_indices = critical_indices[start_idx:start_idx + max_block_size]

        new_genome = genome[:]

        if random.random() < intensity:
            op = random.choice(['reverse', 'shift', 'swap_pair'])

            if op == 'reverse':
                positions = sorted(critical_indices)
                segment = [new_genome[p] for p in positions]
                segment.reverse()
                for p, val in zip(positions, segment):
                    new_genome[p] = val

            elif op == 'shift':
                positions = sorted(critical_indices)
                block = [new_genome[p] for p in positions]
                remaining = [g for idx, g in enumerate(new_genome) if idx not in set(positions)]
                insert_pos = random.randint(0, len(remaining))
                new_genome = remaining[:insert_pos] + block + remaining[insert_pos:]

            else:  # swap_pair
                p1, p2 = random.sample(critical_indices, 2)
                new_genome[p1], new_genome[p2] = new_genome[p2], new_genome[p1]

        return new_genome
    
    def mutation_swap_on_genome(self, genome):
        """随机交换变异（操作基因组列表）"""
        if random.random() > self.p_mut:
            return genome
        
        genome = genome[:]
        idx1, idx2 = random.sample(range(len(genome)), 2)
        genome[idx1], genome[idx2] = genome[idx2], genome[idx1]
        return genome
    
    def evolve(self):
        """M5: 基于数学+物理双驱状态的 RL-NSGA-II"""
        self.initialize()
        
        best_fitness_history = []
        diversity_history = []
        
        for gen in range(self.n_gen):
            # M5: 计算数学状态
            diversity = self.calculate_population_diversity(self.population)
            diversity_history.append(diversity)
            
            # 判断是否停滞
            stagnation = 0
            if gen > 5:
                if best_fitness_history[-1] >= best_fitness_history[-5]:
                    stagnation = 1
            
            # 判断是否多样性低
            diversity_low = 1 if diversity < 50.0 else 0
            
            # M5: 获取物理状态（拥堵指数）
            # 使用当前最优个体进行物理仿真
            current_fronts = self.fast_non_dominated_sort(self.population)
            if current_fronts and len(current_fronts[0]) > 0:
                best_ind = min(current_fronts[0], key=lambda x: x['fitness'][1])
                congestion_high = 1 if self.get_congestion_state(best_ind) else 0
            else:
                congestion_high = 0
            
            # M5: 扩展状态编码（3位）
            current_state = stagnation * 4 + diversity_low * 2 + congestion_high  # 0-7
            
            # 动作选择
            if random.random() < self.epsilon:
                action_idx = random.randint(0, 2)
            else:
                action_idx = np.argmax(self.q_table[current_state])
            
            # 应用动作
            self.p_cross, self.p_mut = self.actions[action_idx]
            
            # 生成子代
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
                
                # M5: 关键块变异（BM）
                if random.random() < self.bm_probability:
                    child_genome = self.mutation_bottleneck_block(child_genome)
                else:
                    child_genome = self.mutation_swap_on_genome(child_genome)
                
                # 以 20% 的概率触发巷道亲和性定向变异
                if random.random() < 0.2:
                    child_genome = self.aisle_affinity_local_search(child_genome)
                
                offspring.append({'genome': child_genome, 'fitness': None})
            
            self.evaluate(offspring)
            self.population.extend(offspring)
            
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
            
            # 计算奖励
            current_best_cmax = min([ind['fitness'][1] for ind in fronts[0]])
            best_fitness_history.append(current_best_cmax)
            
            reward = 0.0
            if gen > 0 and current_best_cmax < best_fitness_history[-2]:
                reward = 1.0
            else:
                reward = -0.5
            
            # 下一状态
            next_stagnation = 0
            if gen > 4 and best_fitness_history[-1] >= best_fitness_history[-5]:
                next_stagnation = 1
            next_diversity_low = 1 if diversity < 50.0 else 0
            
            if fronts[0]:
                best_ind = min(fronts[0], key=lambda x: x['fitness'][1])
                next_congestion = 1 if self.get_congestion_state(best_ind) else 0
            else:
                next_congestion = 0
            
            next_state = next_stagnation * 4 + next_diversity_low * 2 + next_congestion
            
            # Q-learning 更新
            self.q_table[current_state, action_idx] = self.q_table[current_state, action_idx] + \
                self.alpha * (reward + self.gamma * np.max(self.q_table[next_state]) - 
                             self.q_table[current_state, action_idx])
            
            print(f"[M5 RL-Phys-Math Gen {gen+1}] State: {current_state}, Action: {action_idx}, "
                  f"Best Cmax: {current_best_cmax:.2f}, Diversity: {diversity:.2f}")
        
        final_fronts = self.fast_non_dominated_sort(self.population)
        return final_fronts[0]
