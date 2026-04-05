#!/usr/bin/env python3
"""
Deep Harmonic Analysis — Four Investigations

1. Find the fundamental: is there ONE base frequency all forces are overtones of?
2. Flat space comparison: is the harmonic structure from S³ or from the update rule?
3. Closure hierarchy as octaves: are Walter's levels harmonically related?
4. Coupling constant prediction: do harmonic ratios predict real force strengths?
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))

import numpy as np
import json
from scipy.signal import find_peaks
from engine import ForceConfig, SimConfig, run_1d, FORCES


# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────

def get_spectrum(timeseries):
    """Power spectrum of a time series, normalized."""
    centered = np.array(timeseries) - np.mean(timeseries)
    if np.std(centered) < 1e-10:
        return np.zeros(len(centered) // 2 + 1), np.fft.rfftfreq(len(centered)), 0.0
    fft = np.fft.rfft(centered)
    power = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(len(centered))
    total = power.sum()
    power_norm = power / total if total > 0 else power
    pn = power_norm[power_norm > 0]
    entropy = -np.sum(pn * np.log(pn))
    return power_norm, freqs, entropy


def get_peaks(power, freqs, threshold_frac=0.05):
    """Find spectral peaks."""
    if len(power) < 5:
        return [], []
    height = np.max(power[1:]) * threshold_frac  # skip DC
    indices, _ = find_peaks(power[1:], height=height, distance=2)
    indices += 1
    return freqs[indices], power[indices]


# ─────────────────────────────────────────────
# INVESTIGATION 1: Find the fundamental
# ─────────────────────────────────────────────

def find_fundamental():
    """
    Find a single base frequency f0 such that the dominant frequencies
    of all four force regimes are approximately n × f0 for integer n.
    """
    print("=" * 60)
    print("INVESTIGATION 1: Find the Fundamental")
    print("=" * 60)
    
    config = SimConfig(n_cells=200, n_steps=2000, seed=42)
    
    dominant_freqs = {}
    all_peak_freqs = {}
    
    for force in FORCES:
        hist = run_1d(force, config)
        power, freqs, entropy = get_spectrum(hist["firing_rates"])
        peak_freqs, peak_powers = get_peaks(power, freqs)
        
        if len(peak_freqs) > 0:
            # Dominant = highest power peak
            best_idx = np.argmax(peak_powers)
            dominant_freqs[force.name] = float(peak_freqs[best_idx])
            all_peak_freqs[force.name] = [float(f) for f in peak_freqs[:10]]
        else:
            dominant_freqs[force.name] = 0.0
            all_peak_freqs[force.name] = []
    
    print(f"\n  Dominant frequencies:")
    for name, freq in dominant_freqs.items():
        print(f"    {name:<10}: {freq:.6f}")
    
    # Try to find GCD-like fundamental
    # For continuous frequencies, find f0 that minimizes the sum of 
    # |f_i / f0 - round(f_i / f0)| across all forces
    valid_freqs = [f for f in dominant_freqs.values() if f > 0]
    
    if len(valid_freqs) < 2:
        print(f"\n  Not enough valid frequencies to find fundamental")
        return {"dominant_freqs": dominant_freqs, "fundamental": None}
    
    # Scan candidate fundamentals
    best_f0 = 0
    best_error = float('inf')
    best_harmonics = {}
    
    # Try f0 from min_freq/20 to min_freq
    min_f = min(valid_freqs)
    max_f = max(valid_freqs)
    
    for f0_candidate in np.linspace(min_f / 30, min_f, 500):
        if f0_candidate < 1e-6:
            continue
        ratios = [f / f0_candidate for f in valid_freqs]
        rounded = [round(r) for r in ratios]
        errors = [abs(r - n) for r, n in zip(ratios, rounded)]
        total_error = sum(errors)
        
        if total_error < best_error and all(n > 0 for n in rounded):
            best_error = total_error
            best_f0 = f0_candidate
            best_harmonics = {name: round(dominant_freqs[name] / f0_candidate) 
                            for name in dominant_freqs if dominant_freqs[name] > 0}
    
    print(f"\n  Best fundamental: f₀ = {best_f0:.6f}")
    print(f"  Total harmonic error: {best_error:.4f}")
    print(f"\n  Force as harmonics of f₀:")
    for name, n in best_harmonics.items():
        actual = dominant_freqs[name]
        predicted = n * best_f0
        error = abs(actual - predicted) / actual * 100 if actual > 0 else 0
        print(f"    {name:<10}: harmonic #{n:<4}  actual={actual:.6f}  predicted={predicted:.6f}  error={error:.1f}%")
    
    # Also try using ALL peaks across all forces
    all_freqs_flat = []
    for peaks in all_peak_freqs.values():
        all_freqs_flat.extend(peaks)
    all_freqs_flat = sorted(set([f for f in all_freqs_flat if f > 0]))
    
    if len(all_freqs_flat) >= 3:
        # GCD of frequency differences
        diffs = np.diff(all_freqs_flat)
        diffs = diffs[diffs > 1e-4]
        if len(diffs) > 0:
            median_diff = np.median(diffs)
            print(f"\n  Median spacing between all peaks: {median_diff:.6f}")
            print(f"  (If constant → equispaced harmonics → overtone series)")
            
            spacing_cv = np.std(diffs) / np.mean(diffs) if np.mean(diffs) > 0 else float('inf')
            print(f"  Spacing CV: {spacing_cv:.3f}")
            if spacing_cv < 0.3:
                print(f"  ★ Peaks are approximately equispaced → harmonic series")
            elif spacing_cv < 0.5:
                print(f"  ✓ Peaks show some regularity")
            else:
                print(f"  ~ Peaks not clearly equispaced")
    
    result = {
        "dominant_freqs": dominant_freqs,
        "all_peak_freqs": all_peak_freqs,
        "fundamental": float(best_f0),
        "harmonic_numbers": best_harmonics,
        "total_error": float(best_error),
    }
    
    return result


# ─────────────────────────────────────────────
# INVESTIGATION 2: Flat space comparison
# ─────────────────────────────────────────────

def flat_space_comparison():
    """
    Run the SAME simulation on flat (scalar) space.
    Check if spectral entropy is conserved.
    If YES → harmonics come from the update rule.
    If NO → harmonics come from S³ geometry.
    """
    print("\n" + "=" * 60)
    print("INVESTIGATION 2: Flat Space vs S³")
    print("=" * 60)
    
    config = SimConfig(n_cells=200, n_steps=2000, seed=42)
    
    # Flat space: just use the scalar model (already what engine.run_1d does)
    flat_entropies = {}
    print(f"\n  Flat space (scalar model):")
    for force in FORCES:
        hist = run_1d(force, config)
        _, _, entropy = get_spectrum(hist["firing_rates"])
        flat_entropies[force.name] = entropy
        print(f"    {force.name:<10}: spectral entropy = {entropy:.4f}")
    
    flat_values = list(flat_entropies.values())
    flat_cv = np.std(flat_values) / np.mean(flat_values) if np.mean(flat_values) > 0 else float('inf')
    print(f"  Flat space spectral entropy CV: {flat_cv:.4f}")
    
    # S³ space: use the quaternion model
    # Import from exp6
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
    from exp6_quaternion import run_quaternion_sim
    
    s3_entropies = {}
    thresholds = {"gravity": 0.05, "EM": 0.3, "weak": 0.7, "strong": 1.2}
    
    print(f"\n  S³ space (quaternion model):")
    for name, tau in thresholds.items():
        hist = run_quaternion_sim(tau, n_cells=80, n_steps=500, seed=42)
        _, _, entropy = get_spectrum(hist["firing_rates"])
        s3_entropies[name] = entropy
        print(f"    {name:<10}: spectral entropy = {entropy:.4f}")
    
    s3_values = list(s3_entropies.values())
    s3_cv = np.std(s3_values) / np.mean(s3_values) if np.mean(s3_values) > 0 else float('inf')
    print(f"  S³ spectral entropy CV: {s3_cv:.4f}")
    
    print(f"\n  COMPARISON:")
    print(f"    Flat space CV:  {flat_cv:.4f}")
    print(f"    S³ space CV:    {s3_cv:.4f}")
    
    if flat_cv < 0.1 and s3_cv < 0.1:
        print(f"    → Conserved in BOTH → harmonic structure is from the UPDATE RULE")
        print(f"      The threshold-and-fire dynamics create self-similar spectra")
        print(f"      regardless of the underlying manifold.")
        source = "update_rule"
    elif flat_cv > 0.3 and s3_cv < 0.1:
        print(f"    → Conserved only on S³ → harmonic structure is from GEOMETRY")
        print(f"      The quaternion manifold creates the harmonic relationship.")
        source = "geometry"
    elif flat_cv < 0.1 and s3_cv > 0.3:
        print(f"    → Conserved only in flat space → S³ DISRUPTS the harmonics")
        source = "flat_only"
    else:
        print(f"    → Neither strongly conserved — unclear origin")
        source = "unclear"
    
    return {
        "flat_entropies": flat_entropies,
        "flat_cv": float(flat_cv),
        "s3_entropies": s3_entropies,
        "s3_cv": float(s3_cv),
        "source": source,
    }


# ─────────────────────────────────────────────
# INVESTIGATION 3: Closure hierarchy as octaves
# ─────────────────────────────────────────────

def closure_hierarchy_octaves():
    """
    In the recursive model (Exp 8), are the closure cadences at 
    different levels harmonically related? Specifically, is the 
    ratio between levels close to a power of 2 (octaves), or 
    some other simple ratio?
    """
    print("\n" + "=" * 60)
    print("INVESTIGATION 3: Closure Hierarchy as Octaves")
    print("=" * 60)
    
    from exp8_recursive import run_recursive_sim
    
    # Run at multiple seeds to get stable ratios
    level_rates = {0: [], 1: [], 2: []}
    
    print(f"\n  Running recursive sim at 20 seeds...")
    for seed in range(20):
        levels = run_recursive_sim(n_cells=60, n_steps=1000, 
                                   perturbation=0.4, n_levels=3, seed=seed)
        for lev_idx, level in enumerate(levels):
            total = level["closure_count"]
            n_steps = 1000
            rate = total / n_steps
            level_rates[lev_idx].append(rate)
    
    print(f"\n  Mean closure rates by level:")
    means = {}
    for lev, rates in level_rates.items():
        mean_rate = np.mean(rates)
        means[lev] = mean_rate
        nonzero = [r for r in rates if r > 0]
        print(f"    Level {lev}: {mean_rate:.6f} closures/step "
              f"(nonzero in {len(nonzero)}/20 seeds)")
    
    # Ratios between levels
    print(f"\n  Level ratios:")
    ratios = {}
    if means[0] > 0 and means[1] > 0:
        r01 = means[0] / means[1]
        ratios["L0/L1"] = r01
        print(f"    Level 0 / Level 1: {r01:.1f}x")
        
        # What musical interval is this?
        log2_ratio = np.log2(r01)
        print(f"    log₂(ratio) = {log2_ratio:.2f}")
        print(f"    Nearest octave: 2^{round(log2_ratio)} = {2**round(log2_ratio)}")
        
        # Check common musical intervals
        intervals = {
            "unison": 1, "octave": 2, "2 octaves": 4, "3 octaves": 8,
            "4 octaves": 16, "5 octaves": 32, "6 octaves": 64,
            "7 octaves": 128, "perfect fifth": 3/2, "perfect fourth": 4/3,
            "major third": 5/4, "minor third": 6/5,
        }
        
        best_interval = None
        best_error = float('inf')
        for name, ratio_val in intervals.items():
            error = abs(r01 - ratio_val) / ratio_val
            if error < best_error:
                best_error = error
                best_interval = name
        
        print(f"    Closest musical interval: {best_interval} (error: {best_error*100:.1f}%)")
        
        if means[2] > 0:
            r12 = means[1] / means[2]
            ratios["L1/L2"] = r12
            print(f"    Level 1 / Level 2: {r12:.1f}x")
            
            # Is the ratio self-similar? (same between all levels)
            ratio_ratio = r01 / r12 if r12 > 0 else float('inf')
            print(f"    L0/L1 vs L1/L2 similarity: {ratio_ratio:.2f}x")
            if 0.5 < ratio_ratio < 2.0:
                print(f"    ✓ Ratio is approximately self-similar across levels")
            else:
                print(f"    ~ Ratios differ between levels")
    
    # Spectral analysis of Level 0 firing times
    print(f"\n  Level 0 spectral analysis (one seed):")
    levels = run_recursive_sim(n_cells=60, n_steps=2000, 
                               perturbation=0.4, n_levels=2, seed=42)
    
    rates_l0 = levels[0]["closure_rate_ts"]
    power, freqs, entropy = get_spectrum(rates_l0)
    peak_freqs, peak_powers = get_peaks(power, freqs)
    
    if len(peak_freqs) >= 2:
        print(f"    Dominant peaks: {[f'{f:.4f}' for f in peak_freqs[:6]]}")
        # Check for octave relationships between peaks
        for i in range(min(5, len(peak_freqs))):
            for j in range(i+1, min(6, len(peak_freqs))):
                ratio = peak_freqs[j] / peak_freqs[i]
                log2_r = np.log2(ratio) if ratio > 0 else 0
                if abs(log2_r - round(log2_r)) < 0.1:
                    print(f"    Peak {i}→{j}: ratio {ratio:.2f} ≈ 2^{round(log2_r)} (octave!)")
    
    return {"means": {str(k): float(v) for k, v in means.items()}, "ratios": ratios}


# ─────────────────────────────────────────────
# INVESTIGATION 4: Predict coupling constants
# ─────────────────────────────────────────────

def predict_coupling_constants():
    """
    If forces are harmonics with frequency f_n = n × f0, and coupling 
    strength α ∝ f^k for some exponent k, can we find k that matches 
    real coupling constant ratios?
    """
    print("\n" + "=" * 60)
    print("INVESTIGATION 4: Harmonic Prediction of Coupling Constants")
    print("=" * 60)
    
    config = SimConfig(n_cells=200, n_steps=2000, seed=42)
    
    # Get simulation frequencies and coupling strengths
    sim_data = {}
    for force in FORCES:
        hist = run_1d(force, config)
        ss = slice(-500, None)
        
        rate = np.mean(hist["firing_rates"][ss])
        strength = np.mean(hist["avg_transfers"][ss])
        rng = np.mean(hist["effective_ranges"][ss])
        
        power, freqs, entropy = get_spectrum(hist["firing_rates"])
        peak_freqs, peak_powers = get_peaks(power, freqs)
        dominant_freq = float(peak_freqs[np.argmax(peak_powers)]) if len(peak_freqs) > 0 else 0
        
        sim_data[force.name] = {
            "tau": force.tau,
            "rate": float(rate),
            "strength": float(strength),
            "range": float(rng),
            "coupling": float(rate * strength * rng),
            "dominant_freq": dominant_freq,
        }
    
    # Real coupling constants (normalized to strong = 1)
    real = {
        "strong": 1.0,
        "EM": 1.0 / 137.0,
        "weak": 1.05e-5,
        "gravity": 5.9e-39,
    }
    
    print(f"\n  Simulation data:")
    print(f"  {'Force':<10} {'τ':>6} {'Rate':>8} {'Freq':>8} {'Coupling':>10}")
    for name, d in sim_data.items():
        print(f"  {name:<10} {d['tau']:>6.2f} {d['rate']:>8.4f} {d['freq']:>8.4f} {d['coupling']:>10.6f}"
              if 'freq' in d else
              f"  {name:<10} {d['tau']:>6.2f} {d['rate']:>8.4f} {d['dominant_freq']:>8.4f} {d['coupling']:>10.6f}")
    
    # Try to find α ∝ f^k relationship
    # Using simulation frequencies and simulation couplings first
    sim_freqs = {name: d["dominant_freq"] for name, d in sim_data.items() if d["dominant_freq"] > 0}
    sim_couplings = {name: d["coupling"] for name, d in sim_data.items() if d["coupling"] > 0}
    
    common_forces = set(sim_freqs.keys()) & set(sim_couplings.keys())
    if len(common_forces) >= 2:
        f_arr = np.array([sim_freqs[n] for n in sorted(common_forces)])
        c_arr = np.array([sim_couplings[n] for n in sorted(common_forces)])
        
        # Fit log(coupling) = k * log(freq) + const
        # i.e., coupling ∝ freq^k
        log_f = np.log(f_arr[f_arr > 0])
        log_c = np.log(c_arr[c_arr > 0])
        
        if len(log_f) >= 2 and len(log_c) >= 2:
            min_len = min(len(log_f), len(log_c))
            log_f, log_c = log_f[:min_len], log_c[:min_len]
            
            # Linear regression
            A = np.vstack([log_f, np.ones(len(log_f))]).T
            result = np.linalg.lstsq(A, log_c, rcond=None)
            k, const = result[0]
            residuals = result[1]
            
            print(f"\n  Power law fit: coupling ∝ freq^k")
            print(f"    k = {k:.3f}")
            print(f"    R² = {1 - (residuals[0] / np.var(log_c) / len(log_c)) if len(residuals) > 0 else 'N/A'}")
    
    # Now the real test: can harmonic numbers predict REAL coupling ratios?
    print(f"\n  REAL COUPLING CONSTANT PREDICTION:")
    print(f"  If forces are harmonics n=1,2,3,4 and α ∝ n^k:")
    
    # Assign harmonic numbers: gravity=1 (fundamental), strong=4 (4th harmonic)
    # Or reversed: strong=1, gravity=high harmonic
    
    for assignment_name, harmonic_assignment in [
        ("gravity=1st, strong=4th", {"gravity": 1, "EM": 2, "weak": 3, "strong": 4}),
        ("strong=1st, gravity=4th", {"strong": 1, "weak": 2, "EM": 3, "gravity": 4}),
        ("gravity=1st, strong=Nth (fit N)", None),  # will fit
    ]:
        print(f"\n    Assignment: {assignment_name}")
        
        if harmonic_assignment is None:
            # Fit harmonic numbers to match real coupling ratios
            # log(α_real) = k * log(n) + const
            # We need to find both n values and k
            # Fix gravity=1, find n for others
            
            best_k = 0
            best_ns = {}
            best_error = float('inf')
            
            for k_try in np.linspace(-50, -1, 500):
                # Given k, what n_strong would give ratio 1/5.9e-39?
                # α_strong/α_gravity = (n_strong/n_gravity)^k = (n_strong)^k
                # So n_strong = (α_strong/α_gravity)^(1/k)
                
                log_ratio_sg = np.log(real["strong"] / real["gravity"])  # ~89.5
                n_strong = np.exp(log_ratio_sg / k_try)
                
                if 1 < n_strong < 1e6:
                    # Derive others
                    log_ratio_eg = np.log(real["EM"] / real["gravity"])
                    log_ratio_wg = np.log(real["weak"] / real["gravity"])
                    
                    n_em = np.exp(log_ratio_eg / k_try)
                    n_weak = np.exp(log_ratio_wg / k_try)
                    
                    # How "nice" are these numbers? (close to integers or simple fractions)
                    ns = [1, n_em, n_weak, n_strong]
                    niceness = sum(abs(n - round(n)) for n in ns if n < 1000)
                    
                    if niceness < best_error:
                        best_error = niceness
                        best_k = k_try
                        best_ns = {
                            "gravity": 1.0,
                            "EM": float(n_em),
                            "weak": float(n_weak),
                            "strong": float(n_strong),
                        }
            
            print(f"      Best fit: k = {best_k:.2f}")
            for name, n in best_ns.items():
                predicted_ratio = (n / 1.0) ** best_k * real["gravity"]
                actual = real[name]
                error = abs(np.log10(predicted_ratio) - np.log10(actual)) if predicted_ratio > 0 and actual > 0 else float('inf')
                print(f"        {name:<10}: n={n:>10.2f}  predicted={predicted_ratio:.2e}  actual={actual:.2e}  log₁₀ error={error:.1f}")
            
            continue
        
        # Fixed assignment: check if there's a k that works
        forces_ordered = sorted(harmonic_assignment.keys(), key=lambda x: harmonic_assignment[x])
        ns = np.array([harmonic_assignment[f] for f in forces_ordered], dtype=float)
        alphas = np.array([real[f] for f in forces_ordered])
        
        # Fit: log(α) = k * log(n) + const
        log_n = np.log(ns)
        log_alpha = np.log(alphas)
        
        A = np.vstack([log_n, np.ones(len(log_n))]).T
        result = np.linalg.lstsq(A, log_alpha, rcond=None)
        k, const = result[0]
        
        predicted = np.exp(k * log_n + const)
        
        print(f"      k = {k:.2f}")
        for i, f in enumerate(forces_ordered):
            log_err = abs(np.log10(predicted[i]) - np.log10(alphas[i]))
            print(f"        {f:<10}: n={harmonic_assignment[f]}  predicted={predicted[i]:.2e}  actual={alphas[i]:.2e}  log₁₀ error={log_err:.1f}")
        
        # Overall fit quality
        log_errors = [abs(np.log10(predicted[i]) - np.log10(alphas[i])) for i in range(len(alphas))]
        mean_log_error = np.mean(log_errors)
        print(f"      Mean log₁₀ error: {mean_log_error:.1f} orders of magnitude")
        
        if mean_log_error < 1:
            print(f"      ★ GOOD FIT — within 1 order of magnitude on average")
        elif mean_log_error < 3:
            print(f"      ✓ Rough fit — within 3 orders")
        else:
            print(f"      ✗ Poor fit — {mean_log_error:.0f} orders off")
    
    return sim_data


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    results = {}
    
    # Investigation 1
    results["fundamental"] = find_fundamental()
    
    # Investigation 2
    results["flat_vs_s3"] = flat_space_comparison()
    
    # Investigation 3
    results["octaves"] = closure_hierarchy_octaves()
    
    # Investigation 4
    results["coupling_prediction"] = predict_coupling_constants()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY OF HARMONIC INVESTIGATIONS")
    print("=" * 60)
    
    print(f"""
  1. FUNDAMENTAL FREQUENCY:
     f₀ = {results['fundamental']['fundamental']:.6f}
     Harmonic numbers: {results['fundamental']['harmonic_numbers']}

  2. FLAT vs S³:
     Flat space spectral entropy CV: {results['flat_vs_s3']['flat_cv']:.4f}
     S³ space spectral entropy CV:   {results['flat_vs_s3']['s3_cv']:.4f}
     Source of harmonics: {results['flat_vs_s3']['source']}

  3. HIERARCHY OCTAVES:
     Level ratios: {results['octaves']['ratios']}

  4. COUPLING PREDICTION:
     See detailed output above.
""")
    
    # Save
    output = {k: v for k, v in results.items() if k != "coupling_prediction"}
    output_path = os.path.join(os.path.dirname(__file__), "..", "results", "analysis_harmonic_deep.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved to {output_path}")


if __name__ == "__main__":
    main()
