#!/usr/bin/env python3
"""
Experiment 1: Sign (Attraction vs Repulsion)

Question: If cells carry charge (+/-), does EM-like cancellation emerge
at large scales while gravity (unsigned) remains universally attractive?

Setup:
- Run EM-like force WITH charge signs (+/-) randomly assigned
- Run gravity-like force WITHOUT signs
- Compare: does the signed force cancel out at large scales while
  the unsigned force accumulates?

This tests whether a single mechanism can produce both:
  - A force that's individually strong but cancels at scale (EM)
  - A force that's individually weak but accumulates at scale (gravity)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import json
from engine import ForceConfig, SimConfig, run_1d

def run_experiment():
    config = SimConfig(n_cells=200, n_steps=500, seed=42)
    
    # Gravity: unsigned (always attractive)
    gravity = ForceConfig("gravity_unsigned", tau=0.01, signed=False)
    
    # EM: signed (can attract or repel depending on charge)
    em_signed = ForceConfig("EM_signed", tau=0.10, signed=True)
    
    # Control: EM without sign (to see what sign changes)
    em_unsigned = ForceConfig("EM_unsigned", tau=0.10, signed=False)
    
    # Also run strong with sign — quarks carry color charge
    strong_signed = ForceConfig("strong_signed", tau=0.90, signed=True)
    
    rng = np.random.default_rng(config.seed)
    # Fixed charge distribution: half positive, half negative
    charges = rng.choice([-1.0, 1.0], config.n_cells)
    
    results = {}
    
    for force in [gravity, em_signed, em_unsigned, strong_signed]:
        print(f"  Running {force.name}...")
        hist = run_1d(force, config, charges=charges if force.signed else None)
        
        # Compute scale-dependent metrics
        ss = slice(-100, None)
        
        # Net force at different scales (averaging windows)
        final_snapshot = hist["snapshots"][-1]
        
        scale_analysis = {}
        for window in [2, 5, 10, 20, 50]:
            # Average energy over windows of this size
            n_windows = len(final_snapshot) // window
            windowed = np.array([
                np.mean(final_snapshot[i*window:(i+1)*window]) 
                for i in range(n_windows)
            ])
            # Variance of windowed averages — high variance = force doesn't cancel
            scale_analysis[f"variance_at_scale_{window}"] = float(np.var(windowed))
            scale_analysis[f"mean_at_scale_{window}"] = float(np.mean(np.abs(windowed)))
        
        results[force.name] = {
            "avg_firing_rate": float(np.mean(hist["firing_rates"][ss])),
            "avg_energy": float(np.mean(hist["energy_means"][ss])),
            "energy_std": float(np.mean(hist["energy_stds"][ss])),
            "spatial_corr": float(np.mean(hist["spatial_correlation"][ss])),
            "net_attraction": float(np.mean(hist["net_attraction"][ss])),
            "net_repulsion": float(np.mean(hist["net_repulsion"][ss])),
            "scale_analysis": scale_analysis,
            "timeseries_energy_std": [float(x) for x in hist["energy_stds"]],
        }
    
    return results


def main():
    print("=" * 60)
    print("EXPERIMENT 1: Sign (Attraction vs Repulsion)")
    print("=" * 60)
    print()
    
    results = run_experiment()
    
    print("\n  RESULTS:")
    print(f"  {'Force':<20} {'Rate':>8} {'Energy':>8} {'StdDev':>8} {'Corr':>8} {'Attract':>10} {'Repulse':>10}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")
    
    for name, data in results.items():
        print(f"  {name:<20} {data['avg_firing_rate']:>8.4f} {data['avg_energy']:>8.4f} "
              f"{data['energy_std']:>8.4f} {data['spatial_corr']:>8.4f} "
              f"{data['net_attraction']:>10.4f} {data['net_repulsion']:>10.4f}")
    
    print("\n  SCALE-DEPENDENT CANCELLATION:")
    print(f"  {'Force':<20} {'Scale=2':>10} {'Scale=5':>10} {'Scale=10':>10} {'Scale=20':>10} {'Scale=50':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    for name, data in results.items():
        sa = data["scale_analysis"]
        vals = [sa[f"variance_at_scale_{s}"] for s in [2, 5, 10, 20, 50]]
        print(f"  {name:<20} {vals[0]:>10.6f} {vals[1]:>10.6f} {vals[2]:>10.6f} "
              f"{vals[3]:>10.6f} {vals[4]:>10.6f}")
    
    # Analysis
    em_s = results["EM_signed"]["scale_analysis"]
    em_u = results["EM_unsigned"]["scale_analysis"]
    grav = results["gravity_unsigned"]["scale_analysis"]
    
    # Does EM_signed cancel more than EM_unsigned at large scales?
    signed_cancellation = em_u["variance_at_scale_50"] / max(em_s["variance_at_scale_50"], 1e-10)
    
    # Does gravity accumulate more than EM at large scales?
    grav_vs_em = grav["variance_at_scale_50"] / max(em_s["variance_at_scale_50"], 1e-10)
    
    print(f"\n  ANALYSIS:")
    print(f"    Signed EM cancellation ratio (unsigned/signed variance at scale 50): {signed_cancellation:.3f}")
    print(f"    (>1 means signed version cancels more → EM-like behavior)")
    print(f"    Gravity accumulation ratio (gravity/signed_EM variance at scale 50): {grav_vs_em:.3f}")
    print(f"    (>1 means gravity accumulates more at large scales)")
    
    if signed_cancellation > 1.5:
        print(f"\n    ✓ SIGNED FORCE CANCELS AT LARGE SCALES — EM-like behavior emerges")
    else:
        print(f"\n    ✗ Signed force does NOT cancel significantly more than unsigned")
    
    if grav_vs_em > 1.5:
        print(f"    ✓ GRAVITY ACCUMULATES RELATIVE TO SIGNED EM — hierarchy preserved")
    else:
        print(f"    ~ Gravity does not clearly dominate at large scales (ratio close to 1)")
    
    # Save
    output_path = os.path.join(os.path.dirname(__file__), "..", "results", "exp1_sign.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Data saved to {output_path}")
    
    return results


if __name__ == "__main__":
    main()
