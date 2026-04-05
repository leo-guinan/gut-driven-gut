#!/usr/bin/env python3
"""
Experiment 2: Dynamic Threshold — Does Confinement Emerge?

Question: If the activation threshold depends on local energy density,
does the strong force exhibit confinement-like behavior (force gets
STRONGER as particles separate)?

Setup:
- Standard model: tau is fixed
- Dynamic model: tau increases when local energy density drops
  (i.e., it costs MORE to update in low-energy regions)
  
The physical intuition: in QCD, the gluon field between separating quarks
forms a "flux tube" that stores energy. The further apart they get, the
more energy is in the tube, until it's enough to create a new quark pair.

In our framework: if tau depends on local density, then a particle that
"fires" and depletes its local region raises the threshold around it,
making it harder for energy to propagate further — effectively confining
the influence to a local region that RESISTS separation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import json
from engine import ForceConfig, SimConfig, run_1d, update_1d


def density_dependent_tau(cells, base_tau):
    """
    Threshold increases in low-density regions.
    tau(x) = base_tau / local_density_ratio
    
    In high-density regions: tau stays low (easy to fire)
    In low-density regions: tau goes up (hard to fire)
    
    This creates a natural "confinement": energy stays where energy is.
    """
    # Local density = rolling mean of absolute energy
    window = 5
    kernel = np.ones(window) / window
    local_density = np.convolve(np.abs(cells), kernel, mode='same')
    
    # Global mean for normalization
    global_mean = max(np.mean(np.abs(cells)), 1e-6)
    density_ratio = local_density / global_mean
    
    # In low-density regions, tau goes UP (harder to fire)
    # In high-density regions, tau stays at base or goes slightly DOWN
    # Clamp to prevent tau from going to 0 or infinity
    dynamic_tau = base_tau / np.clip(density_ratio, 0.1, 10.0)
    
    return dynamic_tau


def separation_test(tau, dynamic=False, config=None):
    """
    Create two energy clusters and measure how they evolve.
    Does energy stay confined or spread out?
    """
    if config is None:
        config = SimConfig(n_cells=200, n_steps=300, energy_inject=0.02, seed=42)
    
    rng = np.random.default_rng(config.seed)
    cells = rng.uniform(0, 0.01, config.n_cells)  # low background
    
    # Create two "particles" — concentrated energy clusters
    center1, center2 = config.n_cells // 3, 2 * config.n_cells // 3
    width = 5
    cells[center1-width:center1+width] = 2.0  # high energy cluster 1
    cells[center2-width:center2+width] = 2.0  # high energy cluster 2
    
    tau_fn = density_dependent_tau if dynamic else None
    
    history = {
        "cluster1_energy": [],
        "cluster2_energy": [],
        "between_energy": [],
        "spread_metric": [],
        "snapshots": [],
    }
    
    region1 = slice(center1 - width * 3, center1 + width * 3)
    region2 = slice(center2 - width * 3, center2 + width * 3)
    between = slice(center1 + width * 3, center2 - width * 3)
    
    for step in range(config.n_steps):
        cells += rng.uniform(0, config.energy_inject, config.n_cells)
        cells *= (1.0 - config.decay_rate)
        
        update_1d(cells, tau, dynamic_tau_fn=tau_fn)
        
        # Track energy distribution
        c1_energy = float(np.sum(np.abs(cells[region1])))
        c2_energy = float(np.sum(np.abs(cells[region2])))
        btwn_energy = float(np.sum(np.abs(cells[between])))
        
        # Spread metric: what fraction of total energy is outside the clusters?
        total = float(np.sum(np.abs(cells)))
        if total > 0:
            confinement_ratio = (c1_energy + c2_energy) / total
        else:
            confinement_ratio = 0.0
        
        history["cluster1_energy"].append(c1_energy)
        history["cluster2_energy"].append(c2_energy)
        history["between_energy"].append(btwn_energy)
        history["spread_metric"].append(confinement_ratio)
        
        if step % 30 == 0:
            history["snapshots"].append(cells.copy().tolist())
    
    return history


def run_experiment():
    config = SimConfig(n_cells=200, n_steps=300, energy_inject=0.02, seed=42)
    
    results = {}
    
    for tau_name, tau_val in [("gravity", 0.01), ("EM", 0.10), ("weak", 0.50), ("strong", 0.90)]:
        print(f"  Running {tau_name} (τ={tau_val})...")
        
        # Fixed threshold
        fixed = separation_test(tau_val, dynamic=False, config=config)
        # Dynamic threshold
        dynamic = separation_test(tau_val, dynamic=True, config=config)
        
        results[tau_name] = {
            "tau": tau_val,
            "fixed": {
                "final_confinement": float(np.mean(fixed["spread_metric"][-50:])),
                "initial_confinement": float(np.mean(fixed["spread_metric"][:20])),
                "between_energy_final": float(np.mean(fixed["between_energy"][-50:])),
                "confinement_timeseries": fixed["spread_metric"],
                "between_timeseries": fixed["between_energy"],
                "snapshots": fixed["snapshots"],
            },
            "dynamic": {
                "final_confinement": float(np.mean(dynamic["spread_metric"][-50:])),
                "initial_confinement": float(np.mean(dynamic["spread_metric"][:20])),
                "between_energy_final": float(np.mean(dynamic["between_energy"][-50:])),
                "confinement_timeseries": dynamic["spread_metric"],
                "between_timeseries": dynamic["between_energy"],
                "snapshots": dynamic["snapshots"],
            },
        }
    
    return results


def main():
    print("=" * 60)
    print("EXPERIMENT 2: Dynamic Threshold — Confinement Test")
    print("=" * 60)
    print()
    
    results = run_experiment()
    
    print("\n  CONFINEMENT RATIO (fraction of energy in cluster regions)")
    print(f"  Higher = more confined. If dynamic > fixed, confinement emerges.\n")
    print(f"  {'Force':<10} {'τ':>6} {'Fixed(init)':>12} {'Fixed(final)':>12} {'Dynamic(init)':>14} {'Dynamic(final)':>14} {'Δ':>8}")
    print(f"  {'-'*10} {'-'*6} {'-'*12} {'-'*12} {'-'*14} {'-'*14} {'-'*8}")
    
    for name, data in results.items():
        f_init = data["fixed"]["initial_confinement"]
        f_final = data["fixed"]["final_confinement"]
        d_init = data["dynamic"]["initial_confinement"]
        d_final = data["dynamic"]["final_confinement"]
        delta = d_final - f_final
        
        print(f"  {name:<10} {data['tau']:>6.2f} {f_init:>12.4f} {f_final:>12.4f} "
              f"{d_init:>14.4f} {d_final:>14.4f} {delta:>+8.4f}")
    
    print(f"\n  BETWEEN-CLUSTER ENERGY (energy in the gap between clusters)")
    print(f"  Lower = more confined.\n")
    print(f"  {'Force':<10} {'Fixed':>12} {'Dynamic':>12} {'Ratio':>8}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*8}")
    
    for name, data in results.items():
        f_btwn = data["fixed"]["between_energy_final"]
        d_btwn = data["dynamic"]["between_energy_final"]
        ratio = f_btwn / max(d_btwn, 1e-10)
        print(f"  {name:<10} {f_btwn:>12.4f} {d_btwn:>12.4f} {ratio:>8.2f}x")
    
    # Check for confinement specifically in the strong force
    strong = results["strong"]
    conf_improvement = strong["dynamic"]["final_confinement"] - strong["fixed"]["final_confinement"]
    
    print(f"\n  ANALYSIS:")
    if conf_improvement > 0.05:
        print(f"    ✓ DYNAMIC THRESHOLD INCREASES CONFINEMENT for strong force (+{conf_improvement:.4f})")
        print(f"      Energy stays more concentrated in cluster regions with density-dependent τ")
    elif conf_improvement > 0:
        print(f"    ~ Weak confinement effect for strong force (+{conf_improvement:.4f})")
    else:
        print(f"    ✗ No confinement improvement from dynamic threshold ({conf_improvement:+.4f})")
    
    # Does the effect scale with tau?
    deltas = [(name, data["dynamic"]["final_confinement"] - data["fixed"]["final_confinement"]) 
              for name, data in results.items()]
    deltas.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n    Confinement improvement ranking:")
    for name, delta in deltas:
        marker = "✓" if delta > 0.01 else "~" if delta > 0 else "✗"
        print(f"      {marker} {name}: {delta:+.4f}")
    
    # Save
    # Convert any non-serializable items
    output_path = os.path.join(os.path.dirname(__file__), "..", "results", "exp2_confinement.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Data saved to {output_path}")
    
    return results


if __name__ == "__main__":
    main()
