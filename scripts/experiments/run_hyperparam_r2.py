#!/usr/bin/env python3
"""
Hyperparameter robustness scan on R2 full dataset
"""
import os, sys, json, time, random
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from src.simulation import Simulator
from src.optimization import RL_NSGA2Optimizer
from src.utils import load_jobs_from_csv
from src.strategies import heuristic_dispatch

JOBS_FILE = "R2_highest_arrival_rate.csv"
SEEDS = [42, 123, 2026]
DEFAULT = dict(pop_size=30, n_gen=30, epsilon=0.2, gamma=0.85)

print("=" * 80)
print(f"Hyperparam scan / RL-NSGA-II / R2 full dataset / {len(SEEDS)} seeds")
print("=" * 80)

jobs_all = load_jobs_from_csv(JOBS_FILE)
print(f"[Data loaded] {JOBS_FILE}: {len(jobs_all)} jobs")

# FIFO baseline
sim_fifo = Simulator(jobs_all, mode='adaptive_rho')
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
print(f"[FIFO baseline] Cmax={fifo_cmax:.2f}  W_total={fifo_w:.2f}\n")

def run_one(pop_size, n_gen, epsilon, gamma, seed):
    random.seed(seed); np.random.seed(seed)
    original_release_times = sorted([job.release_time for job in jobs_all])
    random.shuffle(jobs_all)
    for i, job in enumerate(jobs_all):
        job.release_time = original_release_times[i]
    sim = Simulator(jobs_all, mode='adaptive_rho')
    opt = RL_NSGA2Optimizer(sim, len(jobs_all), pop_size=pop_size, n_gen=n_gen)
    if hasattr(opt, 'epsilon'):
        opt.epsilon = epsilon
    if hasattr(opt, 'gamma'):
        opt.gamma = gamma
    t0 = time.time()
    res = opt.evolve()
    elapsed = time.time() - t0
    sols = res['solutions']
    min_cmax = min(s['fitness'][1] for s in sols)
    threshold = min_cmax * 1.005
    candidates = [s for s in sols if s['fitness'][1] <= threshold]
    best = min(candidates, key=lambda x: x['fitness'][0])
    return best['fitness'][0], best['fitness'][1], elapsed

records = []
t_start = time.time()

# A: POP scan
print("--- A: POP scan ---")
for pop in [15, 30, 50]:
    cfg = dict(DEFAULT, pop_size=pop)
    ws = []
    for seed in SEEDS:
        w, c, t = run_one(**cfg, seed=seed)
        ws.append(w)
        print(f"  pop={pop:>3d}  seed={seed:>5d}  W={w:8.2f}  C={c:9.2f}  t={t:5.1f}s")
    records.append({"sweep": "POP", **cfg, "W_list": ws,
                    "W_mean": float(np.mean(ws)), "W_std": float(np.std(ws, ddof=1))})

# B: GEN scan
print("\n--- B: GEN scan ---")
for gen in [15, 30, 50]:
    cfg = dict(DEFAULT, n_gen=gen)
    ws = []
    for seed in SEEDS:
        w, c, t = run_one(**cfg, seed=seed)
        ws.append(w)
        print(f"  gen={gen:>3d}  seed={seed:>5d}  W={w:8.2f}  C={c:9.2f}  t={t:5.1f}s")
    records.append({"sweep": "GEN", **cfg, "W_list": ws,
                    "W_mean": float(np.mean(ws)), "W_std": float(np.std(ws, ddof=1))})

# C: epsilon scan
print("\n--- C: epsilon scan ---")
for eps in [0.1, 0.2, 0.3]:
    cfg = dict(DEFAULT, epsilon=eps)
    ws = []
    for seed in SEEDS:
        w, c, t = run_one(**cfg, seed=seed)
        ws.append(w)
        print(f"  eps={eps:>3.1f}  seed={seed:>5d}  W={w:8.2f}  C={c:9.2f}  t={t:5.1f}s")
    records.append({"sweep": "EPS", **cfg, "W_list": ws,
                    "W_mean": float(np.mean(ws)), "W_std": float(np.std(ws, ddof=1))})

# D: gamma scan
print("\n--- D: gamma scan ---")
for gam in [0.7, 0.85, 0.95]:
    cfg = dict(DEFAULT, gamma=gam)
    ws = []
    for seed in SEEDS:
        w, c, t = run_one(**cfg, seed=seed)
        ws.append(w)
        print(f"  gam={gam:>4.2f}  seed={seed:>5d}  W={w:8.2f}  C={c:9.2f}  t={t:5.1f}s")
    records.append({"sweep": "GAM", **cfg, "W_list": ws,
                    "W_mean": float(np.mean(ws)), "W_std": float(np.std(ws, ddof=1))})

print(f"\nTotal time = {time.time()-t_start:.0f}s")

# Summary table
print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print(f"{'Sweep':8s} {'Value':>8s} {'W_mean':>10s} {'W_std':>9s} {'vs_default':>10s}")
for r in records:
    sweep = r['sweep']
    if sweep == 'POP':
        val = r['pop_size']
    elif sweep == 'GEN':
        val = r['n_gen']
    elif sweep == 'EPS':
        val = f"{r['epsilon']:.1f}"
    else:
        val = f"{r['gamma']:.2f}"
    val_str = str(val)
    marker = " *" if val_str in ('30', '0.2', '0.85') else ""
    print(f"{sweep:8s} {val_str:>8s} {r['W_mean']:>10.2f} {r['W_std']:>9.2f}{marker}")

# Save
os.makedirs("results/hyperparam", exist_ok=True)
out = "results/hyperparam/hyperparam_r2_results.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump({
        "config": {"jobs_file": JOBS_FILE, "default": DEFAULT, "seeds": SEEDS},
        "fifo_baseline": {"cmax": fifo_cmax, "w_total": fifo_w},
        "records": records
    }, f, indent=2, ensure_ascii=False)
print(f"\n[OK] Data saved: {out}")
