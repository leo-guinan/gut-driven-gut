#!/usr/bin/env python3
"""
Experiment 4: Find the Conserved Quantity

The original simulation found that Range × Strength is NOT conserved.
But the framework claims all forces are one mechanism. If so, SOMETHING
should be conserved across threshold values.

Strategy: systematically test every reasonable combination of observables
to find what (if anything) is approximately constant across all four forces.

Candidates:
- R × S (range × strength) — already tested, failed
- R × S × Rate (total influence) — already tested, failed  
- R^a × S^b for various exponents
- Rate × Range (how much of the lattice is covered per step)
- Information-theoretic measures (entropy of energy distribution)
- τ × R (threshold-range product — should be ~1 by construction)
- Energy flow rate (total energy moved per step)
- Variance of energy distribution
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import json
from engine import ForceConfig, SimConfig, run_1d, FORCES


def compute_observables(force, config):
    """Run simulation and extract all measurable quantities."""
    hist = run_1d(force, config)
    ss = slice(-200, None)
    
    rate = np.mean(hist["firing_rates"][ss])
    rng = np.mean(hist["effective_ranges"][ss])
    strength = np.mean(hist["avg_transfers"][ss])
    total_xfer = np.mean(hist["total_transferred"][ss])
    energy_mean = np.mean(hist["energy_means"][ss])
    energy_std = np.mean(hist["energy_stds"][ss])
    corr = np.mean(hist["spatial_correlation"][ss])
    
    # Entropy of energy distribution from last snapshot
    snapshot = np.abs(hist["snapshots"][-1])
    probs = snapshot / max(snapshot.sum(), 1e-10)
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))
    
    return {
        "tau": force.tau,
        "rate": float(rate),
        "range": float(rng),
        "strength": float(strength),
        "total_transfer": float(total_xfer),
        "energy_mean": float(energy_mean),
        "energy_std": float(energy_std),
        "correlation": float(corr),
        "entropy": float(entropy),
    }


def test_combinations(observables_by_force):
    """
    Systematically test combinations of observables.
    For each combination, compute coefficient of variation across forces.
    Lower CV = more conserved.
    """
    force_names = list(observables_by_force.keys())
    
    # Single observables
    single_keys = ["rate", "range", "strength", "total_transfer", 
                   "energy_mean", "energy_std", "entropy"]
    
    results = []
    
    # Test single observables
    for key in single_keys:
        values = [observables_by_force[f][key] for f in force_names]
        mean_v = np.mean(values)
        cv = np.std(values) / mean_v if mean_v != 0 else float('inf')
        results.append({
            "formula": key,
            "values": {f: observables_by_force[f][key] for f in force_names},
            "cv": float(cv),
        })
    
    # Test products and ratios of pairs
    for k1 in single_keys:
        for k2 in single_keys:
            if k1 >= k2:
                continue
            # Product
            values = [observables_by_force[f][k1] * observables_by_force[f][k2] for f in force_names]
            mean_v = np.mean(values)
            cv = np.std(values) / abs(mean_v) if mean_v != 0 else float('inf')
            results.append({
                "formula": f"{k1} × {k2}",
                "values": {f: v for f, v in zip(force_names, values)},
                "cv": float(cv),
            })
            
            # Ratio
            values = [observables_by_force[f][k1] / max(observables_by_force[f][k2], 1e-10) 
                     for f in force_names]
            mean_v = np.mean(values)
            cv = np.std(values) / abs(mean_v) if mean_v != 0 else float('inf')
            results.append({
                "formula": f"{k1} / {k2}",
                "values": {f: v for f, v in zip(force_names, values)},
                "cv": float(cv),
            })
    
    # Test power law combinations: R^a × S^b for various a,b
    for a in np.arange(0.5, 3.0, 0.5):
        for b in np.arange(0.5, 3.0, 0.5):
            values = [
                observables_by_force[f]["range"]**a * observables_by_force[f]["strength"]**b 
                for f in force_names
            ]
            mean_v = np.mean(values)
            cv = np.std(values) / abs(mean_v) if mean_v != 0 else float('inf')
            results.append({
                "formula": f"range^{a:.1f} × strength^{b:.1f}",
                "values": {f: v for f, v in zip(force_names, values)},
                "cv": float(cv),
            })
    
    # Test tau-normalized quantities
    for key in single_keys:
        # Multiply by tau
        values = [observables_by_force[f][key] * observables_by_force[f]["tau"] 
                 for f in force_names]
        mean_v = np.mean(values)
        cv = np.std(values) / abs(mean_v) if mean_v != 0 else float('inf')
        results.append({
            "formula": f"τ × {key}",
            "values": {f: v for f, v in zip(force_names, values)},
            "cv": float(cv),
        })
        
        # Divide by tau
        values = [observables_by_force[f][key] / observables_by_force[f]["tau"] 
                 for f in force_names]
        mean_v = np.mean(values)
        cv = np.std(values) / abs(mean_v) if mean_v != 0 else float('inf')
        results.append({
            "formula": f"{key} / τ",
            "values": {f: v for f, v in zip(force_names, values)},
            "cv": float(cv),
        })
    
    # Triple products
    for k1 in ["rate", "range", "strength"]:
        for k2 in ["energy_mean", "energy_std", "entropy"]:
            for k3 in ["rate", "range", "strength"]:
                if k1 >= k3:
                    continue
                values = [
                    observables_by_force[f][k1] * observables_by_force[f][k2] * observables_by_force[f][k3]
                    for f in force_names
                ]
                mean_v = np.mean(values)
                cv = np.std(values) / abs(mean_v) if mean_v != 0 else float('inf')
                results.append({
                    "formula": f"{k1} × {k2} × {k3}",
                    "values": {f: v for f, v in zip(force_names, values)},
                    "cv": float(cv),
                })
    
    return results


def main():
    print("=" * 60)
    print("EXPERIMENT 4: Find the Conserved Quantity")
    print("=" * 60)
    print()
    
    config = SimConfig(n_cells=200, n_steps=500, seed=42)
    
    # Collect observables for each force
    observables = {}
    for force in FORCES:
        print(f"  Measuring {force.name} (τ={force.tau})...")
        observables[force.name] = compute_observables(force, config)
    
    # Print raw observables
    print(f"\n  RAW OBSERVABLES:")
    print(f"  {'Force':<10} {'τ':>6} {'Rate':>8} {'Range':>8} {'Strength':>10} {'Energy':>8} {'StdDev':>8} {'Entropy':>8}")
    print(f"  {'-'*10} {'-'*6} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
    
    for name, obs in observables.items():
        print(f"  {name:<10} {obs['tau']:>6.2f} {obs['rate']:>8.4f} {obs['range']:>8.1f} "
              f"{obs['strength']:>10.6f} {obs['energy_mean']:>8.4f} {obs['energy_std']:>8.4f} "
              f"{obs['entropy']:>8.4f}")
    
    # Test all combinations
    print(f"\n  Testing {200}+ combinations...")
    all_results = test_combinations(observables)
    
    # Sort by CV (most conserved first)
    all_results.sort(key=lambda x: x["cv"])
    
    # Filter to finite values
    valid = [r for r in all_results if np.isfinite(r["cv"]) and r["cv"] < 10]
    
    print(f"\n  TOP 20 MOST CONSERVED QUANTITIES (lowest CV):")
    print(f"  {'Rank':>4} {'CV':>8} {'Formula':<40} {'Values'}")
    print(f"  {'-'*4} {'-'*8} {'-'*40} {'-'*50}")
    
    for i, r in enumerate(valid[:20]):
        vals = "  ".join(f"{name}={v:.6f}" for name, v in r["values"].items())
        marker = "★" if r["cv"] < 0.1 else "✓" if r["cv"] < 0.3 else " "
        print(f"  {i+1:>4} {r['cv']:>8.4f} {marker} {r['formula']:<38} {vals}")
    
    # Analysis
    best = valid[0] if valid else None
    
    print(f"\n  ANALYSIS:")
    if best and best["cv"] < 0.1:
        print(f"    ★ STRIKING: '{best['formula']}' has CV = {best['cv']:.4f}")
        print(f"      This quantity is approximately conserved across all four forces.")
        print(f"      Values: {best['values']}")
    elif best and best["cv"] < 0.3:
        print(f"    ✓ SUGGESTIVE: '{best['formula']}' has CV = {best['cv']:.4f}")
        print(f"      Weakly conserved. Not definitive but worth investigating.")
    else:
        print(f"    ✗ No strongly conserved quantity found (best CV = {best['cv']:.4f})")
        print(f"      The forces may not share a simple conserved scalar.")
    
    # Check if any tau-dependent normalization helps
    tau_results = [r for r in valid if "τ" in r["formula"]]
    if tau_results:
        best_tau = tau_results[0]
        print(f"\n    Best τ-normalized quantity: '{best_tau['formula']}' (CV = {best_tau['cv']:.4f})")
    
    # Save
    output = {
        "observables": observables,
        "top_50_conserved": valid[:50],
    }
    output_path = os.path.join(os.path.dirname(__file__), "..", "results", "exp4_conservation.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Data saved to {output_path}")
    
    return all_results


if __name__ == "__main__":
    main()
