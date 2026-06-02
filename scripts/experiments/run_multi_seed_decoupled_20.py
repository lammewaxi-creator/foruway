#!/usr/bin/env python3
"""
20-seed statistical experiment: decoupled mode, pop=30/gen=30
For Friedman + Wilcoxon tests in §4.4
"""
import os
import sys
import json
import time
import random
import statistics
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from src.simulation import Simulator
from src.strategies import heuristic_dispatch
from src.optimization import NSGA2Optimizer, RL_NSGA2Optimizer, IGAOptimizer
from src.utils import load_jobs_from_csv

JOBS_FILE = "R2_highest_arrival_rate.csv"
POP = 30
GEN = 30
# 20 seeds for statistical power
SEEDS = [42, 123, 456, 789, 1000, 2023, 2024, 2025, 2026, 3141,
         2718, 1618, 7777, 8888, 9999, 11111, 22222, 33333, 44444, 55555]

print("=" * 80)
print(f"20-seed experiment / decoupled mode / pop={POP} / gen={GEN} / {len(SEEDS)} seeds")
print("=" * 80)

jobs_all = load_jobs_from_csv(JOBS_FILE)
print(f"[Data loaded] {JOBS_FILE}: {len(jobs_all)} jobs")

# FIFO baseline (deterministic, run once)
sim_fifo = Simulator(jobs_all, mode='decoupled')
heuristic_dispatch(sim_fifo, strategy='FIFO')
fifo_cmax = max(j.finish_times[-1] for j in sim_fifo.jobs if j.finish_times)
total_wait = 0.0
for job in sim_fifo.jobs:
    for s_idx, stage in enumerate(job.stages):
        if stage['type'] == 'Lift' and s_idx > 0:
            if s_idx - 1 < len(job.finish_times) and s_idx < len(job.start_times):
                rgv_finish = job.finish_times[s_idx - 1]
                lift_start = job.start_times[s_idx]
                total_wait += abs(lift_start - rgv_finish)
fifo_w = total_wait
print(f"\n[FIFO baseline] Cmax={fifo_cmax:.2f}  W_total={fifo_w:.2f}")

# Evolutionary algorithms
algos = [
    ("NSGA-II", lambda sim: NSGA2Optimizer(sim, len(jobs_all), pop_size=POP, n_gen=GEN)),
    ("IGA",     lambda sim: IGAOptimizer(sim, len(jobs_all), pop_size=POP, n_gen=GEN)),
    ("RL-NSGA-II", lambda sim: RL_NSGA2Optimizer(sim, len(jobs_all), pop_size=POP, n_gen=GEN)),
]

records = []
t_start = time.time()

for seed in SEEDS:
    print(f"\n--- seed = {seed} (elapsed={time.time()-t_start:.0f}s) ---")
    for algo_name, factory in algos:
        random.seed(seed)
        np.random.seed(seed)
        # Chaos shuffle
        original_release_times = sorted([job.release_time for job in jobs_all])
        random.shuffle(jobs_all)
        for i, job in enumerate(jobs_all):
            job.release_time = original_release_times[i]

        sim = Simulator(jobs_all, mode='decoupled')
        opt = factory(sim)
        t0 = time.time()
        if algo_name == "RL-NSGA-II":
            res = opt.evolve()
        else:
            res = opt.optimize()
        elapsed = time.time() - t0

        # Balanced selection
        min_cmax = min(s['fitness'][1] for s in res['solutions'])
        threshold = min_cmax * 1.005
        candidates = [s for s in res['solutions'] if s['fitness'][1] <= threshold]
        best = min(candidates, key=lambda x: x['fitness'][0])

        w_best, c_best = best['fitness']
        records.append({
            "algo": algo_name, "seed": seed,
            "W_total": float(w_best), "Cmax": float(c_best),
            "time_s": float(elapsed)
        })
        print(f"  {algo_name:15s}  W_total={w_best:8.2f}  Cmax={c_best:8.2f}  time={elapsed:5.1f}s")

print(f"\nTotal time = {time.time()-t_start:.0f}s")

# Descriptive statistics
print("\n" + "=" * 80)
print("Descriptive Statistics")
print("=" * 80)
algo_names = [a for a, _ in algos]
ws_per_algo = {a: [r['W_total'] for r in records if r['algo'] == a] for a in algo_names}
cs_per_algo = {a: [r['Cmax']    for r in records if r['algo'] == a] for a in algo_names}

print(f"{'Algorithm':18s} {'W_mean':>10s} {'W_std':>9s} {'Cmax_mean':>10s} {'Cmax_std':>9s}")
for a in algo_names:
    ws = ws_per_algo[a]
    cs = cs_per_algo[a]
    print(f"{a:18s} {np.mean(ws):>10.2f} {np.std(ws,ddof=1):>9.2f} {np.mean(cs):>10.2f} {np.std(cs,ddof=1):>9.2f}")

# Relative FIFO improvement
print("\n" + "=" * 80)
print("W_total improvement vs FIFO baseline")
print("=" * 80)
for a in algo_names:
    m = np.mean(ws_per_algo[a])
    imp = (fifo_w - m) / fifo_w * 100
    print(f"  {a:18s}  mean={m:.2f}  vs FIFO={fifo_w:.2f}  improvement {imp:+.1f}%")

# Friedman + Wilcoxon
print("\n" + "=" * 80)
print("Statistical Tests")
print("=" * 80)
try:
    from scipy.stats import friedmanchisquare, wilcoxon
    matrix = np.array([ws_per_algo[a] for a in algo_names])
    stat, p_fried = friedmanchisquare(*matrix)
    sig_text = "significant" if p_fried < 0.05 else "not significant"
    print(f"Friedman chi2 = {stat:.4f}  p = {p_fried:.6f}  {sig_text} (alpha=0.05)")

    print("\nWilcoxon paired test:")
    k = len(algo_names)
    for i in range(k):
        for j in range(i+1, k):
            a, b = algo_names[i], algo_names[j]
            stat_w, p_w = wilcoxon(ws_per_algo[a], ws_per_algo[b])
            sig = "significant" if p_w < 0.05 else "not significant"
            print(f"  {a} vs {b}: p={p_w:.5f} {sig}")
except ImportError:
    print("[skip] scipy not installed")

# Save
os.makedirs("results/multi_seed_decoupled", exist_ok=True)
out = "results/multi_seed_decoupled/multi_seed_decoupled_20_results.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump({
        "config": {"jobs_file": JOBS_FILE, "pop_size": POP, "n_gen": GEN, "n_seeds": len(SEEDS), "seeds": SEEDS, "mode": "decoupled"},
        "fifo_baseline": {"cmax": fifo_cmax, "w_total": fifo_w},
        "records": records,
        "summary": {
            a: {"W_mean": float(np.mean(ws_per_algo[a])), "W_std": float(np.std(ws_per_algo[a], ddof=1)),
                "Cmax_mean": float(np.mean(cs_per_algo[a])), "Cmax_std": float(np.std(cs_per_algo[a], ddof=1))}
            for a in algo_names
        }
    }, f, indent=2, ensure_ascii=False)
print(f"\n[OK] Data saved: {out}")
