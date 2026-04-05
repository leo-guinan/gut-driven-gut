#!/usr/bin/env python3
"""
Analytical Investigation 1: The ~47:1 Hierarchy Ratio

Question: Is the ratio between Level 0 and Level 1 closure rates
deterministic (a property of S³) or stochastic (depends on parameters)?

Approach:
1. Run the recursive sim at many seeds — does the ratio hold?
2. Vary perturbation magnitude — does the ratio change?
3. Vary cell count — does the ratio change?
4. Attempt to derive from random walk on S³

If the ratio is stable across parameters, it's geometric.
If it varies, it's parametric.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))

import numpy as np
import json
from exp8_recursive import run_recursive_sim


def seed_stability():
    """Run at many seeds. Is the ratio consistent?"""
    ratios = []
    l0_counts = []
    l1_counts = []
    
    for seed in range(20):
        levels = run_recursive_sim(n_cells=60, n_steps=800, 
                                   perturbation=0.4, n_levels=2, seed=seed)
        c0 = levels[0]["closure_count"]
        c1 = levels[1]["closure_count"]
        
        if c1 > 0:
            r = c0 / c1
            ratios.append(r)
        l0_counts.append(c0)
        l1_counts.append(c1)
    
    return {
        "ratios": ratios,
        "l0_counts": l0_counts,
        "l1_counts": l1_counts,
        "mean_ratio": float(np.mean(ratios)) if ratios else 0,
        "std_ratio": float(np.std(ratios)) if ratios else 0,
        "cv_ratio": float(np.std(ratios) / np.mean(ratios)) if ratios and np.mean(ratios) > 0 else float('inf'),
        "n_with_l1": sum(1 for c in l1_counts if c > 0),
    }


def perturbation_scan():
    """Vary perturbation. Does the ratio change?"""
    results = []
    for pert in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        ratios_at_pert = []
        for seed in range(10):
            levels = run_recursive_sim(n_cells=60, n_steps=800,
                                       perturbation=pert, n_levels=2, seed=seed)
            c0 = levels[0]["closure_count"]
            c1 = levels[1]["closure_count"]
            if c1 > 0:
                ratios_at_pert.append(c0 / c1)
        
        results.append({
            "perturbation": pert,
            "mean_ratio": float(np.mean(ratios_at_pert)) if ratios_at_pert else 0,
            "std_ratio": float(np.std(ratios_at_pert)) if ratios_at_pert else 0,
            "n_valid": len(ratios_at_pert),
        })
    return results


def cell_count_scan():
    """Vary cell count. Does the ratio change?"""
    results = []
    for n in [20, 40, 60, 80, 100]:
        ratios = []
        for seed in range(10):
            levels = run_recursive_sim(n_cells=n, n_steps=800,
                                       perturbation=0.4, n_levels=2, seed=seed)
            c0 = levels[0]["closure_count"]
            c1 = levels[1]["closure_count"]
            if c1 > 0:
                ratios.append(c0 / c1)
        
        results.append({
            "n_cells": n,
            "mean_ratio": float(np.mean(ratios)) if ratios else 0,
            "std_ratio": float(np.std(ratios)) if ratios else 0,
            "n_valid": len(ratios),
        })
    return results


def random_walk_analysis():
    """
    Analytical approach: what's the expected return time for a 
    random walk on S³?
    
    A random walk on S³ with step size θ has a characteristic 
    return time (time to return near identity) that depends on 
    the step size and the geometry of the sphere.
    
    For a random walk on S^n, the probability of being within 
    distance ε of the starting point after k steps of size θ is
    related to the eigenvalues of the Laplacian on S^n.
    
    On S³ specifically, the heat kernel is known analytically.
    The return probability at time t is:
    P(return) ~ Σ (2l+1)² exp(-l(l+2)t/2) for l = 0, 1, 2, ...
    where t is proportional to k*θ².
    """
    # Numerical estimate: run many independent random walks on S³
    # and measure the return time distribution
    
    rng = np.random.default_rng(42)
    
    def random_walk_return_time(step_mag, epsilon=0.3, max_steps=5000):
        """Time for random walk on S³ to return near identity after excursion."""
        q = np.array([1., 0., 0., 0.])
        has_excursion = False
        peak_sigma = 0.0
        
        for step in range(1, max_steps + 1):
            # Random rotation
            angle = rng.exponential(step_mag)
            axis = rng.normal(0, 1, 3)
            axis = axis / (np.linalg.norm(axis) + 1e-10)
            half = angle / 2
            kick = np.array([np.cos(half), *(axis * np.sin(half))])
            kick = kick / np.linalg.norm(kick)
            
            # Compose
            w = q[0]*kick[0] - q[1]*kick[1] - q[2]*kick[2] - q[3]*kick[3]
            x = q[0]*kick[1] + q[1]*kick[0] + q[2]*kick[3] - q[3]*kick[2]
            y = q[0]*kick[2] - q[1]*kick[3] + q[2]*kick[0] + q[3]*kick[1]
            z = q[0]*kick[3] + q[1]*kick[2] - q[2]*kick[1] + q[3]*kick[0]
            q = np.array([w, x, y, z])
            q = q / np.linalg.norm(q)
            
            s = np.arccos(np.clip(np.abs(q[0]), 0, 1))
            if s > 0.5:
                has_excursion = True
            if s > peak_sigma:
                peak_sigma = s
            
            if has_excursion and s < epsilon and peak_sigma > 0.4 and s < peak_sigma * 0.5:
                return step, peak_sigma
        
        return None, peak_sigma  # didn't return
    
    # Measure return times at different step sizes
    step_sizes = [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
    results = {}
    
    for mag in step_sizes:
        times = []
        peaks = []
        for _ in range(200):
            t, p = random_walk_return_time(mag)
            if t is not None:
                times.append(t)
                peaks.append(p)
        
        results[f"step_{mag}"] = {
            "step_size": mag,
            "mean_return_time": float(np.mean(times)) if times else float('inf'),
            "std_return_time": float(np.std(times)) if times else 0,
            "median_return_time": float(np.median(times)) if times else float('inf'),
            "n_returned": len(times),
            "mean_peak": float(np.mean(peaks)) if peaks else 0,
        }
    
    return results


def main():
    print("=" * 60)
    print("ANALYTICAL INVESTIGATION: The Hierarchy Ratio")
    print("=" * 60)
    
    # Test 1: Seed stability
    print("\n  TEST 1: Seed stability (20 seeds)")
    seed_data = seed_stability()
    print(f"    Seeds with Level 1 closures: {seed_data['n_with_l1']}/20")
    print(f"    Mean ratio: {seed_data['mean_ratio']:.1f}")
    print(f"    Std ratio:  {seed_data['std_ratio']:.1f}")
    print(f"    CV:         {seed_data['cv_ratio']:.3f}")
    
    if seed_data['cv_ratio'] < 0.3:
        print(f"    ★ RATIO IS STABLE across seeds (CV < 0.3)")
    elif seed_data['cv_ratio'] < 0.5:
        print(f"    ✓ Ratio moderately stable (CV < 0.5)")
    else:
        print(f"    ✗ Ratio varies significantly across seeds")
    
    # Test 2: Perturbation scan
    print(f"\n  TEST 2: Perturbation scan")
    pert_data = perturbation_scan()
    print(f"    {'Perturbation':>12} {'Ratio':>8} {'±':>8} {'N':>4}")
    for d in pert_data:
        if d['n_valid'] > 0:
            print(f"    {d['perturbation']:>12.1f} {d['mean_ratio']:>8.1f} {d['std_ratio']:>8.1f} {d['n_valid']:>4}")
    
    ratios = [d['mean_ratio'] for d in pert_data if d['n_valid'] > 0 and d['mean_ratio'] > 0]
    if ratios:
        ratio_cv = np.std(ratios) / np.mean(ratios) if np.mean(ratios) > 0 else float('inf')
        print(f"    Ratio CV across perturbations: {ratio_cv:.3f}")
        if ratio_cv < 0.3:
            print(f"    ★ RATIO INDEPENDENT OF ENERGY SCALE")
        elif ratio_cv < 0.5:
            print(f"    ✓ Ratio weakly dependent on energy")
        else:
            print(f"    ✗ Ratio changes with perturbation ({ratio_cv:.2f})")
    
    # Test 3: Cell count scan
    print(f"\n  TEST 3: Cell count scan")
    cell_data = cell_count_scan()
    print(f"    {'N_cells':>8} {'Ratio':>8} {'±':>8} {'N':>4}")
    for d in cell_data:
        if d['n_valid'] > 0:
            print(f"    {d['n_cells']:>8} {d['mean_ratio']:>8.1f} {d['std_ratio']:>8.1f} {d['n_valid']:>4}")
    
    # Test 4: Random walk return times
    print(f"\n  TEST 4: Random walk return times on S³")
    rw_data = random_walk_analysis()
    print(f"    {'Step size':>10} {'Mean return':>12} {'Median':>10} {'N':>5}")
    for key, d in sorted(rw_data.items(), key=lambda x: x[1]['step_size']):
        print(f"    {d['step_size']:>10.1f} {d['mean_return_time']:>12.1f} "
              f"{d['median_return_time']:>10.1f} {d['n_returned']:>5}")
    
    # Check if return time ratio gives the hierarchy ratio
    rw_sorted = sorted(rw_data.values(), key=lambda x: x['step_size'])
    if len(rw_sorted) >= 2:
        slow = rw_sorted[0]
        fast = rw_sorted[-1]
        if fast['mean_return_time'] > 0 and slow['mean_return_time'] > 0:
            rw_ratio = slow['mean_return_time'] / fast['mean_return_time']
            print(f"\n    Return time ratio (slowest/fastest): {rw_ratio:.1f}x")
            print(f"    Hierarchy ratio from simulation:      {seed_data['mean_ratio']:.1f}x")
    
    # Save
    output = {
        "seed_stability": seed_data,
        "perturbation_scan": pert_data,
        "cell_count_scan": cell_data,
        "random_walk": {k: v for k, v in rw_data.items()},
    }
    output_path = os.path.join(os.path.dirname(__file__), "..", "results", "analysis_ratio.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to {output_path}")


if __name__ == "__main__":
    main()
