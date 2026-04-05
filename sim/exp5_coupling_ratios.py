#!/usr/bin/env python3
"""
Experiment 5: Coupling Constant Ratios

Question: Can a single activation-cost parameter recover the QUANTITATIVE
hierarchy of the real forces, not just the qualitative ordering?

The real force hierarchy (approximate coupling strengths at low energy):
  Strong : EM : Weak : Gravity ≈ 1 : 1/137 : 10⁻⁶ : 10⁻³⁹

That's an absurd range — 39 orders of magnitude from strong to gravity.

Strategy:
1. Define "effective coupling" in our model (rate × strength × range)
2. Vary τ continuously from 0.001 to 0.999
3. Plot effective coupling vs τ
4. Ask: is there a mapping τ → real coupling that's monotonic and smooth?
5. If so, what function τ(α) maps activation cost to coupling constant?
6. Can we find τ values that reproduce the real ratios?

This is the hardest test. If it works even approximately, the framework
has predictive power. If not, we know where the analogy breaks.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import json
from engine import ForceConfig, SimConfig, run_1d


# Real coupling constants (approximate, at low energy ~1 GeV)
REAL_COUPLINGS = {
    "strong": 1.0,          # α_s ≈ 1 (normalized to 1)
    "EM": 1.0 / 137.0,      # α_EM ≈ 1/137
    "weak": 1.05e-5,         # G_F (Fermi constant, normalized)
    "gravity": 5.9e-39,      # G_N (Newton's constant, normalized to strong)
}

# Real coupling constants at GUT scale (~10^16 GeV) — they converge
REAL_COUPLINGS_GUT = {
    "strong": 0.0358,        # α_s at GUT scale
    "EM": 0.0169,            # α_EM at GUT scale (including hypercharge)
    "weak": 0.0337,          # α_2 at GUT scale
    # gravity doesn't participate in standard GUT
}


def effective_coupling(hist, config):
    """
    Compute effective coupling from simulation history.
    Multiple definitions to see which correlates best.
    """
    ss = slice(-200, None)
    
    rate = np.mean(hist["firing_rates"][ss])
    rng = np.mean(hist["effective_ranges"][ss])
    strength = np.mean(hist["avg_transfers"][ss])
    total_xfer = np.mean(hist["total_transferred"][ss])
    
    return {
        "rate": float(rate),
        "range": float(rng),
        "strength": float(strength),
        "total_transfer": float(total_xfer),
        "coupling_v1": float(rate * strength),             # how much moves per step
        "coupling_v2": float(rate * strength * rng),        # total influence
        "coupling_v3": float(total_xfer / config.n_cells),  # transfer density
    }


def tau_scan():
    """Scan coupling vs tau across the full range."""
    config = SimConfig(n_cells=100, n_steps=300, seed=42)
    
    tau_values = np.concatenate([
        np.linspace(0.001, 0.01, 5),
        np.linspace(0.01, 0.1, 10),
        np.linspace(0.1, 0.5, 10),
        np.linspace(0.5, 0.99, 10),
    ])
    tau_values = np.unique(tau_values)
    
    results = []
    
    for tau in tau_values:
        force = ForceConfig(f"tau_{tau:.4f}", tau=float(tau))
        hist = run_1d(force, config)
        couplings = effective_coupling(hist, config)
        couplings["tau"] = float(tau)
        results.append(couplings)
    
    return results


def find_best_tau_mapping(scan_results):
    """
    Try to find tau values that reproduce the real coupling ratios.
    For each definition of coupling, find the best-fit tau mapping.
    """
    taus = np.array([r["tau"] for r in scan_results])
    
    best_mappings = {}
    
    for coupling_key in ["coupling_v1", "coupling_v2", "coupling_v3"]:
        couplings = np.array([r[coupling_key] for r in scan_results])
        
        # Normalize to max = 1
        if couplings.max() > 0:
            couplings_norm = couplings / couplings.max()
        else:
            continue
        
        # For each real force, find the tau that gives the closest match
        mapping = {}
        for force_name, real_alpha in REAL_COUPLINGS.items():
            # Normalize real coupling to strong = 1
            real_norm = real_alpha / REAL_COUPLINGS["strong"]
            
            # Find closest match in our scan
            idx = np.argmin(np.abs(couplings_norm - real_norm))
            mapping[force_name] = {
                "real_coupling_normalized": float(real_norm),
                "best_tau": float(taus[idx]),
                "sim_coupling_normalized": float(couplings_norm[idx]),
                "error": float(abs(couplings_norm[idx] - real_norm)),
            }
        
        best_mappings[coupling_key] = mapping
    
    return best_mappings


def main():
    print("=" * 60)
    print("EXPERIMENT 5: Coupling Constant Ratios")
    print("=" * 60)
    print()
    
    # Real hierarchy for reference
    print("  REAL FORCE HIERARCHY (normalized to strong = 1):")
    for name, alpha in sorted(REAL_COUPLINGS.items(), key=lambda x: -x[1]):
        log_ratio = np.log10(alpha) if alpha > 0 else -99
        print(f"    {name:<10} α = {alpha:.2e}  (10^{log_ratio:.1f})")
    
    print(f"\n  Running τ scan (35 values from 0.001 to 0.99)...")
    scan = tau_scan()
    
    # Print scan results
    print(f"\n  TAU SCAN — EFFECTIVE COUPLING vs THRESHOLD:")
    print(f"  {'τ':>8} {'Rate':>8} {'Range':>8} {'Strength':>10} {'Coupling_v1':>12} {'Coupling_v2':>12} {'Coupling_v3':>12}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*12} {'-'*12} {'-'*12}")
    
    for r in scan[::3]:  # every 3rd for readability
        print(f"  {r['tau']:>8.4f} {r['rate']:>8.4f} {r['range']:>8.1f} "
              f"{r['strength']:>10.6f} {r['coupling_v1']:>12.6f} "
              f"{r['coupling_v2']:>12.6f} {r['coupling_v3']:>12.6f}")
    
    # Check monotonicity
    for key in ["coupling_v1", "coupling_v2", "coupling_v3"]:
        values = [r[key] for r in scan]
        diffs = np.diff(values)
        monotonic = np.all(diffs >= 0) or np.all(diffs <= 0)
        direction = "decreasing" if np.mean(diffs) < 0 else "increasing"
        
        # Check smoothness
        if len(diffs) > 1:
            second_diffs = np.diff(diffs)
            smooth = np.std(second_diffs) / max(np.mean(np.abs(diffs)), 1e-10)
        else:
            smooth = 0
        
        print(f"\n  {key}: {'monotonic' if monotonic else 'NOT monotonic'} ({direction}), smoothness={smooth:.3f}")
    
    # Dynamic range test
    print(f"\n  DYNAMIC RANGE (does the model span enough orders of magnitude?):")
    for key in ["coupling_v1", "coupling_v2", "coupling_v3"]:
        values = [r[key] for r in scan if r[key] > 0]
        if values:
            ratio = max(values) / min(values)
            log_range = np.log10(ratio)
            print(f"    {key}: max/min = {ratio:.1f} ({log_range:.1f} orders of magnitude)")
            print(f"      (Need ~39 orders to match real hierarchy; ")
            print(f"       need ~1.5 orders for just strong/EM/weak)")
    
    # Try to find matching tau values
    print(f"\n  BEST TAU MAPPING TO REAL FORCES:")
    mappings = find_best_tau_mapping(scan)
    
    for coupling_key, mapping in mappings.items():
        print(f"\n  Using {coupling_key}:")
        print(f"    {'Force':<10} {'Real α':>12} {'Best τ':>8} {'Sim α':>12} {'Error':>10}")
        print(f"    {'-'*10} {'-'*12} {'-'*8} {'-'*12} {'-'*10}")
        
        for name in ["strong", "EM", "weak", "gravity"]:
            m = mapping[name]
            print(f"    {name:<10} {m['real_coupling_normalized']:>12.2e} "
                  f"{m['best_tau']:>8.4f} {m['sim_coupling_normalized']:>12.6f} "
                  f"{m['error']:>10.6f}")
    
    # Verdict
    print(f"\n  ANALYSIS:")
    
    # Check if any coupling definition spans enough range
    best_range = 0
    best_key = None
    for key in ["coupling_v1", "coupling_v2", "coupling_v3"]:
        values = [r[key] for r in scan if r[key] > 0]
        if values:
            ratio = max(values) / min(values)
            if ratio > best_range:
                best_range = ratio
                best_key = key
    
    log_range = np.log10(best_range) if best_range > 0 else 0
    
    if log_range > 5:
        print(f"    ✓ Model spans {log_range:.1f} orders of magnitude (best: {best_key})")
        print(f"      Enough to capture strong/EM/weak hierarchy")
    elif log_range > 2:
        print(f"    ~ Model spans {log_range:.1f} orders of magnitude (best: {best_key})")
        print(f"      Captures some hierarchy but not the full range")
    else:
        print(f"    ✗ Model only spans {log_range:.1f} orders of magnitude (best: {best_key})")
        print(f"      Cannot reproduce the real force hierarchy quantitatively")
        print(f"      The 39-order-of-magnitude gap between strong and gravity")
        print(f"      cannot emerge from a threshold in [0, 1]")
    
    # Save
    output = {
        "real_couplings": REAL_COUPLINGS,
        "tau_scan": scan,
        "mappings": mappings,
        "dynamic_range_log10": float(log_range),
    }
    output_path = os.path.join(os.path.dirname(__file__), "..", "results", "exp5_coupling.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Data saved to {output_path}")
    
    return scan, mappings


if __name__ == "__main__":
    main()
