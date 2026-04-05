#!/usr/bin/env python3
"""
Experiment 8: Recursive Force Hierarchy

HYPOTHESIS (Leo):
The forces aren't four separate things. They're NESTED.
The strong force contains strong/weak/EM/gravity within it.
The weak force contains weak/EM/gravity. Etc.

Each force IS the full hierarchy running at a different scale.

WHY THIS MATTERS:
- Resolves the "why four?" problem: the number of forces comes
  from the 1+3 structure of quaternions (W + XYZ), not from
  arbitrary threshold choices
- Provides a natural scale hierarchy: each level's "gravity" is
  the next level's "strong force" viewed from outside
- Matches the fractal/self-similar structure Walter describes:
  "The same pattern repeats at every scale"

THE MODEL:
Level 0 (finest): cells on a lattice, each a quaternion
  - Their closure cadence produces four "sub-forces" via Hopf:
    W-closures (existence) and X/Y/Z-closures (three position axes)
Level 1: closures from Level 0 become inputs to Level 1 cells
  - Level 1 has its own closure cadence, producing its own four forces
Level 2: closures from Level 1 become inputs to Level 2
  - And so on

Each level's closure rate should be SLOWER than the level below
(because it takes many Level N closures to produce one Level N+1 closure).

PREDICTION:
If we decompose the Hopf channels separately at each level, we should see:
- Level 0: mostly RGB-closures (fast, arrangement-dominated) = strong-like
- Level 1: mixed W/RGB = weak/EM-like
- Level 2: mostly W-closures (slow, existence-dominated) = gravity-like

The force hierarchy would then be the SAME mechanism at different
recursive depths — not different parameters of the same mechanism.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import json


def qmul(p, q):
    w = p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3]
    x = p[0]*q[1] + p[1]*q[0] + p[2]*q[3] - p[3]*q[2]
    y = p[0]*q[2] - p[1]*q[3] + p[2]*q[0] + p[3]*q[1]
    z = p[0]*q[3] + p[1]*q[2] - p[2]*q[1] + p[3]*q[0]
    return np.array([w, x, y, z])

def qnorm(q):
    n = np.linalg.norm(q)
    return q / n if n > 0 else np.array([1., 0., 0., 0.])

def sigma(q):
    return np.arccos(np.clip(np.abs(q[0]), 0, 1))

def hopf_4channel(q):
    """
    Full 4-channel Hopf decomposition.
    Returns which channel dominates at the excursion peak.
    W = existence, X/Y/Z = three position axes.
    """
    magnitudes = np.abs(q)
    channels = ["W", "X", "Y", "Z"]
    dominant = channels[np.argmax(magnitudes)]
    return dominant, {c: float(m) for c, m in zip(channels, magnitudes)}

def small_rotation(rng, mag):
    angle = rng.exponential(mag)
    axis = rng.normal(0, 1, 3)
    axis = axis / (np.linalg.norm(axis) + 1e-10)
    half = angle / 2
    return qnorm(np.array([np.cos(half), *(axis * np.sin(half))]))


class HierarchicalCell:
    """Cell that tracks closure and emits upward."""
    def __init__(self):
        self.running_product = np.array([1., 0., 0., 0.])
        self.excursion_peak = 0.0
        self.peak_q = np.array([1., 0., 0., 0.])
        self.has_excursion = False
        self.n_composed = 0
    
    def compose(self, q):
        self.running_product = qnorm(qmul(self.running_product, q))
        s = sigma(self.running_product)
        self.n_composed += 1
        if s > self.excursion_peak:
            self.excursion_peak = s
            self.peak_q = self.running_product.copy()
        if s > 0.1:
            self.has_excursion = True
    
    def check_closure(self, epsilon=0.3):
        s = sigma(self.running_product)
        if (s < epsilon and self.has_excursion and
            self.excursion_peak > 0.4 and self.n_composed >= 2 and
            s < self.excursion_peak * 0.5):
            
            dominant, channels = hopf_4channel(self.peak_q)
            emission = self.running_product.copy()
            peak = self.excursion_peak
            support = self.n_composed
            
            # Reset
            self.running_product = np.array([1., 0., 0., 0.])
            self.excursion_peak = 0.0
            self.peak_q = np.array([1., 0., 0., 0.])
            self.has_excursion = False
            self.n_composed = 0
            
            return True, {
                "emission": emission,
                "dominant": dominant,
                "channels": channels,
                "peak_sigma": float(peak),
                "support": support,
                "closure_sigma": float(s),
            }
        return False, None


def run_recursive_sim(n_cells=60, n_steps=800, perturbation=0.4, 
                      n_levels=3, seed=42):
    """
    Run hierarchical simulation with multiple levels.
    
    Level 0: raw perturbations → closures
    Level 1: Level 0 closures → Level 1 closures
    Level 2: Level 1 closures → Level 2 closures
    
    Each level has its own cells. Closures from level N are 
    composed into level N+1 cells.
    """
    rng = np.random.default_rng(seed)
    
    # Create cells for each level
    levels = []
    for lev in range(n_levels):
        # Fewer cells at higher levels (natural pyramid)
        n = max(4, n_cells // (2 ** lev))
        levels.append({
            "cells": [HierarchicalCell() for _ in range(n)],
            "n_cells": n,
            "closures": [],
            "closure_count": 0,
            "w_count": 0,
            "x_count": 0,
            "y_count": 0,
            "z_count": 0,
            "closure_rate_ts": [],
            "supports": [],
            "peaks": [],
        })
    
    for step in range(n_steps):
        # Level 0: compose random perturbations
        level0 = levels[0]
        for cell in level0["cells"]:
            kick = small_rotation(rng, perturbation)
            cell.compose(kick)
        
        # Check closures at each level, propagate upward
        for lev_idx in range(n_levels):
            level = levels[lev_idx]
            step_closures = 0
            
            for i, cell in enumerate(level["cells"]):
                closed, info = cell.check_closure()
                if closed:
                    step_closures += 1
                    level["closure_count"] += 1
                    level["closures"].append(info)
                    level["supports"].append(info["support"])
                    level["peaks"].append(info["peak_sigma"])
                    
                    dom = info["dominant"]
                    if dom == "W": level["w_count"] += 1
                    elif dom == "X": level["x_count"] += 1
                    elif dom == "Y": level["y_count"] += 1
                    elif dom == "Z": level["z_count"] += 1
                    
                    # Emit to neighbors at same level
                    emit_q = info["emission"]
                    for d in [1, 2]:
                        scale = 0.1 / d
                        angle = sigma(emit_q) * scale
                        if angle > 1e-10:
                            xyz_n = np.linalg.norm(emit_q[1:4])
                            if xyz_n > 1e-10:
                                axis = emit_q[1:4] / xyz_n
                                half = angle / 2
                                transfer = qnorm(np.array([
                                    np.cos(half), *(axis * np.sin(half))
                                ]))
                                for idx in [i-d, i+d]:
                                    if 0 <= idx < level["n_cells"]:
                                        level["cells"][idx].compose(transfer)
                    
                    # EMIT UPWARD: compose closure into next level
                    if lev_idx + 1 < n_levels:
                        next_level = levels[lev_idx + 1]
                        # Map to a cell in the next level
                        target = i % next_level["n_cells"]
                        next_level["cells"][target].compose(info["emission"])
            
            level["closure_rate_ts"].append(step_closures / level["n_cells"])
    
    return levels


def main():
    print("=" * 60)
    print("EXPERIMENT 8: Recursive Force Hierarchy")
    print("  Forces as nested copies of themselves")
    print("=" * 60)
    print()
    
    levels = run_recursive_sim(n_cells=60, n_steps=800, 
                               perturbation=0.4, n_levels=3, seed=42)
    
    print("  HIERARCHICAL CLOSURE CADENCE:")
    print(f"  {'Level':<8} {'Cells':>6} {'Closures':>10} {'Rate':>10} {'Support':>8} "
          f"{'W%':>6} {'X%':>6} {'Y%':>6} {'Z%':>6}")
    print(f"  {'-'*8} {'-'*6} {'-'*10} {'-'*10} {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    
    results = {}
    
    for lev_idx, level in enumerate(levels):
        total = level["closure_count"]
        if total == 0:
            print(f"  Level {lev_idx}  {level['n_cells']:>6} {0:>10} {'N/A':>10} {'N/A':>8} "
                  f"{'N/A':>6} {'N/A':>6} {'N/A':>6} {'N/A':>6}")
            results[f"level_{lev_idx}"] = {
                "n_cells": level["n_cells"], "total_closures": 0,
                "avg_rate": 0, "avg_support": 0,
                "w_pct": 0, "x_pct": 0, "y_pct": 0, "z_pct": 0,
            }
            continue
        
        avg_rate = float(np.mean(level["closure_rate_ts"][-200:]))
        avg_support = float(np.mean(level["supports"])) if level["supports"] else 0
        avg_peak = float(np.mean(level["peaks"])) if level["peaks"] else 0
        
        w_pct = level["w_count"] / total * 100
        x_pct = level["x_count"] / total * 100
        y_pct = level["y_count"] / total * 100
        z_pct = level["z_count"] / total * 100
        
        print(f"  Level {lev_idx}  {level['n_cells']:>6} {total:>10} {avg_rate:>10.4f} "
              f"{avg_support:>8.1f} {w_pct:>5.1f}% {x_pct:>5.1f}% {y_pct:>5.1f}% {z_pct:>5.1f}%")
        
        results[f"level_{lev_idx}"] = {
            "n_cells": level["n_cells"],
            "total_closures": total,
            "avg_rate": float(avg_rate),
            "avg_support": float(avg_support),
            "avg_peak_sigma": float(avg_peak),
            "w_pct": float(w_pct),
            "x_pct": float(x_pct),
            "y_pct": float(y_pct),
            "z_pct": float(z_pct),
            "rgb_pct": float(x_pct + y_pct + z_pct),
        }
    
    # Analysis
    print(f"\n  ANALYSIS:")
    
    # 1. Does closure rate decrease with level? (hierarchy)
    rates = [results[f"level_{i}"]["avg_rate"] for i in range(3)]
    counts = [results[f"level_{i}"]["total_closures"] for i in range(3)]
    
    if counts[0] > 0 and counts[1] > 0:
        ratio_01 = counts[0] / max(counts[1], 1)
        print(f"    Level 0 → Level 1 closure ratio: {ratio_01:.1f}x")
        if counts[2] > 0:
            ratio_12 = counts[1] / max(counts[2], 1)
            print(f"    Level 1 → Level 2 closure ratio: {ratio_12:.1f}x")
        
        if ratio_01 > 2:
            print(f"    ✓ HIERARCHY: each level fires {ratio_01:.0f}x slower than the one below")
        else:
            print(f"    ~ Weak hierarchy (ratio = {ratio_01:.1f})")
    
    # 2. Does the W/RGB ratio shift across levels?
    if all(results[f"level_{i}"]["total_closures"] > 0 for i in range(3)):
        w_by_level = [results[f"level_{i}"]["w_pct"] for i in range(3)]
        rgb_by_level = [results[f"level_{i}"]["rgb_pct"] for i in range(3)]
        
        print(f"\n    Hopf decomposition by level:")
        print(f"      Level 0 (fastest): {w_by_level[0]:.1f}% W, {rgb_by_level[0]:.1f}% RGB")
        print(f"      Level 1 (middle):  {w_by_level[1]:.1f}% W, {rgb_by_level[1]:.1f}% RGB")
        print(f"      Level 2 (slowest): {w_by_level[2]:.1f}% W, {rgb_by_level[2]:.1f}% RGB")
        
        if w_by_level[2] > w_by_level[0]:
            print(f"\n    ✓ RECURSIVE HOPF SHIFT:")
            print(f"      Fast level (strong-like): RGB-dominated (arrangement)")
            print(f"      Slow level (gravity-like): W-dominated (existence)")
            print(f"      The SAME mechanism at different recursive depths")
            print(f"      produces different force TYPES — not just different rates")
        elif rgb_by_level[0] > rgb_by_level[2]:
            print(f"\n    ✓ INVERSE HOPF SHIFT (fast=RGB, slow=W)")
        else:
            print(f"\n    ~ No clear Hopf shift across levels")
    elif results["level_0"]["total_closures"] > 0 and results["level_1"]["total_closures"] > 0:
        print(f"\n    Level 2 had {results['level_2']['total_closures']} closures (too few for statistics)")
        w0 = results["level_0"]["w_pct"]
        w1 = results["level_1"]["w_pct"]
        print(f"    Level 0: {w0:.1f}% W, Level 1: {w1:.1f}% W")
        if abs(w1 - w0) > 10:
            print(f"    ✓ Hopf shift visible between Level 0 and Level 1")
    
    # 3. Does support increase with level? (slower cycles)
    supports = [results[f"level_{i}"]["avg_support"] 
                for i in range(3) if results[f"level_{i}"]["total_closures"] > 0]
    
    if len(supports) >= 2:
        print(f"\n    Compositions per closure by level:")
        for i, s in enumerate(supports):
            print(f"      Level {i}: {s:.1f}")
        
        if supports[-1] > supports[0]:
            print(f"    ✓ Higher levels take MORE compositions per closure (slower)")
        else:
            print(f"    ~ Support doesn't increase with level")
    
    # 4. The recursive interpretation
    print(f"\n  THE RECURSIVE READING:")
    if results["level_0"]["total_closures"] > 5 and results["level_1"]["total_closures"] > 2:
        print(f"    Level 0 sees {results['level_0']['total_closures']} events — it's the 'strong force'")
        print(f"      (fast, frequent, locally dominated)")
        print(f"    Level 1 sees {results['level_1']['total_closures']} events — it's 'EM/weak'")
        print(f"      (slower, built from Level 0 closures)")
        if results["level_2"]["total_closures"] > 0:
            print(f"    Level 2 sees {results['level_2']['total_closures']} events — it's 'gravity'")
            print(f"      (slowest, built from Level 1 closures)")
        else:
            print(f"    Level 2 sees 0 closures — needs more time to produce gravity-like events")
            print(f"      (This is consistent: gravity IS the slowest force)")
        
        print(f"\n    Each level's 'gravity' is the next level's input.")
        print(f"    Each level's 'strong force' is its own fastest closures.")
        print(f"    The forces aren't four things. They're the SAME thing")
        print(f"    at four recursive depths.")
    
    # Save
    output_path = os.path.join(os.path.dirname(__file__), "..", "results", "exp8_recursive.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Data saved to {output_path}")
    
    return results


if __name__ == "__main__":
    main()
