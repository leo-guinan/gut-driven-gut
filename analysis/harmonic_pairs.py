#!/usr/bin/env python3
"""
Harmonic Pairs: Test whether forces come in paired harmonics 
with Fibonacci-structured gaps.

Observed:  Pair 1 = (3, 5),  Pair 2 = (18, 23)
Predicted: Pair 3 = (57, 70), Pair 4 = (159, 193)

Rule: internal gap of pair N+1 = external gap after pair N
Gaps follow every-other-Fibonacci: F(3)=2, F(5)=5, F(7)=13, F(9)=34

TEST: Run simulation with enough harmonics to see if there's 
spectral activity at the predicted pair locations (57, 70).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))

import numpy as np
from scipy.signal import find_peaks
from engine import ForceConfig, SimConfig, run_1d


def extended_spectral_analysis():
    """
    Run a long simulation with many cells to get high frequency 
    resolution, then check for peaks at predicted harmonics.
    """
    # Need long time series for frequency resolution
    # And small enough f0 that harmonics up to 70+ are resolvable
    config = SimConfig(n_cells=200, n_steps=4000, seed=42)
    
    # Run all four forces and combine their spectra
    all_peak_freqs = []
    all_spectra = {}
    
    f0 = None
    
    for force in [
        ForceConfig("gravity", tau=0.01, color="#4466cc"),
        ForceConfig("EM",      tau=0.10, color="#44bb44"),
        ForceConfig("weak",    tau=0.50, color="#dd8800"),
        ForceConfig("strong",  tau=0.90, color="#cc3333"),
    ]:
        hist = run_1d(force, config)
        rates = np.array(hist["firing_rates"])
        centered = rates - rates.mean()
        
        fft = np.fft.rfft(centered)
        power = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(len(centered))
        
        # Find peaks
        if np.max(power[1:]) > 0:
            height = np.max(power[1:]) * 0.02
            indices, props = find_peaks(power[1:], height=height, distance=2)
            indices += 1
            peak_f = freqs[indices]
            peak_p = power[indices]
            
            all_peak_freqs.extend(peak_f.tolist())
            all_spectra[force.name] = {
                "peak_freqs": peak_f.tolist(),
                "peak_powers": peak_p.tolist(),
            }
    
    # Find the fundamental f0 from all peaks
    all_peak_freqs = sorted(set([f for f in all_peak_freqs if f > 0.001]))
    
    if len(all_peak_freqs) < 2:
        print("  Not enough peaks found")
        return
    
    # Find f0 by looking at GCD of peak frequencies
    # Try candidates
    min_f = min(all_peak_freqs)
    best_f0 = 0
    best_score = float('inf')
    
    for f0_try in np.linspace(min_f / 50, min_f / 2, 1000):
        if f0_try < 1e-5:
            continue
        harmonics = [f / f0_try for f in all_peak_freqs[:20]]
        errors = [abs(h - round(h)) for h in harmonics if round(h) > 0]
        score = np.mean(errors) if errors else float('inf')
        if score < best_score:
            best_score = score
            best_f0 = f0_try
    
    print(f"  Fundamental frequency f₀ = {best_f0:.6f}")
    print(f"  Mean harmonic error: {best_score:.4f}")
    
    # Map all peaks to harmonic numbers
    print(f"\n  All spectral peaks as harmonics of f₀:")
    harmonic_map = {}
    for f in all_peak_freqs[:30]:
        n = round(f / best_f0)
        error = abs(f - n * best_f0) / best_f0
        if n > 0 and error < 0.5:
            harmonic_map[n] = f
    
    occupied = sorted(harmonic_map.keys())
    print(f"  Occupied harmonics: {occupied}")
    
    # Check for pair structure
    print(f"\n  PAIR DETECTION:")
    
    # Known pairs
    known_pairs = [(3, 5), (18, 23)]
    predicted_pairs = [(57, 70), (159, 193)]
    
    for label, pairs_list in [("Known", known_pairs), ("Predicted", predicted_pairs)]:
        for a, b in pairs_list:
            a_present = a in occupied
            b_present = b in occupied
            if a_present and b_present:
                status = "✓ BOTH PRESENT"
            elif a_present or b_present:
                status = f"~ PARTIAL ({a if a_present else b} present)"
            else:
                status = "✗ neither present"
            print(f"    {label} pair ({a}, {b}): {status}")
    
    # Look for ALL pairs in the occupied harmonics
    # A pair is two occupied harmonics separated by a Fibonacci number
    fib_set = {1, 2, 3, 5, 8, 13, 21, 34, 55, 89}
    
    print(f"\n  ALL FIBONACCI-GAPPED PAIRS in occupied harmonics:")
    found_pairs = []
    for i, a in enumerate(occupied):
        for b in occupied[i+1:]:
            gap = b - a
            if gap in fib_set:
                found_pairs.append((a, b, gap))
                print(f"    ({a}, {b}) gap={gap} [F]")
    
    # How many would we expect by chance?
    n_occ = len(occupied)
    total_possible_pairs = n_occ * (n_occ - 1) // 2
    fib_gapped = len(found_pairs)
    
    # Expected by chance: for each pair, probability that gap is Fibonacci
    # depends on the range of gaps
    if len(occupied) >= 2:
        max_gap = occupied[-1] - occupied[0]
        n_fib_in_range = len([f for f in fib_set if f <= max_gap])
        expected_frac = n_fib_in_range / max_gap if max_gap > 0 else 0
        expected_count = total_possible_pairs * expected_frac
        
        print(f"\n  Total pairs: {total_possible_pairs}")
        print(f"  Fibonacci-gapped pairs: {fib_gapped}")
        print(f"  Expected by chance: {expected_count:.1f}")
        if fib_gapped > 0 and expected_count > 0:
            enrichment = fib_gapped / expected_count
            print(f"  Enrichment: {enrichment:.2f}x")
            if enrichment > 2:
                print(f"  ★ FIBONACCI PAIRS ENRICHED ({enrichment:.1f}x over chance)")
    
    return {
        "f0": best_f0,
        "occupied": occupied,
        "found_pairs": found_pairs,
        "all_spectra": {k: v["peak_freqs"][:10] for k, v in all_spectra.items()},
    }


def pair_resonance_test():
    """
    Test whether paired harmonics show correlated behavior.
    If (3,5) is a real pair, their firing patterns should be correlated.
    Run two thresholds simultaneously and measure cross-correlation.
    """
    print(f"\n{'='*60}")
    print(f"PAIR RESONANCE TEST")
    print(f"{'='*60}")
    
    config = SimConfig(n_cells=200, n_steps=4000, seed=42)
    
    # Run each force and collect firing time series
    timeseries = {}
    for force in [
        ForceConfig("gravity", tau=0.01),
        ForceConfig("EM",      tau=0.10),
        ForceConfig("weak",    tau=0.50),
        ForceConfig("strong",  tau=0.90),
    ]:
        hist = run_1d(force, config)
        timeseries[force.name] = np.array(hist["firing_rates"])
    
    # Cross-correlation between all pairs
    print(f"\n  CROSS-SPECTRAL COHERENCE:")
    print(f"  (High coherence = correlated frequency content = paired)")
    print(f"  {'Pair':<25} {'Coherence':>12} {'Pair type':>15}")
    print(f"  {'-'*25} {'-'*12} {'-'*15}")
    
    forces = ["gravity", "EM", "weak", "strong"]
    coherences = {}
    
    for i, f1 in enumerate(forces):
        for f2 in forces[i+1:]:
            # Cross-spectral coherence
            s1 = timeseries[f1] - timeseries[f1].mean()
            s2 = timeseries[f2] - timeseries[f2].mean()
            
            fft1 = np.fft.rfft(s1)
            fft2 = np.fft.rfft(s2)
            
            cross = fft1 * np.conj(fft2)
            auto1 = np.abs(fft1) ** 2
            auto2 = np.abs(fft2) ** 2
            
            # Magnitude squared coherence (averaged)
            coherence = np.mean(np.abs(cross) ** 2) / (np.mean(auto1) * np.mean(auto2) + 1e-10)
            
            # Classify
            if (f1 in ["strong", "weak"] and f2 in ["strong", "weak"]):
                pair_type = "NUCLEAR pair"
            elif (f1 in ["gravity", "EM"] and f2 in ["gravity", "EM"]):
                pair_type = "COSMIC pair"
            else:
                pair_type = "cross-pair"
            
            coherences[f"{f1}-{f2}"] = coherence
            print(f"  {f1+'-'+f2:<25} {coherence:>12.6f} {pair_type:>15}")
    
    # Check: are within-pair coherences higher than cross-pair?
    nuclear = coherences.get("strong-weak", 0) 
    cosmic = coherences.get("gravity-EM", 0)
    cross = [v for k, v in coherences.items() 
             if k not in ["strong-weak", "gravity-EM"]]
    mean_cross = np.mean(cross) if cross else 0
    
    print(f"\n  Within nuclear pair: {nuclear:.6f}")
    print(f"  Within cosmic pair:  {cosmic:.6f}")
    print(f"  Mean cross-pair:     {mean_cross:.6f}")
    
    if nuclear > mean_cross and cosmic > mean_cross:
        ratio = min(nuclear, cosmic) / (mean_cross + 1e-10)
        print(f"  ✓ WITHIN-PAIR coherence > cross-pair ({ratio:.1f}x)")
    else:
        print(f"  ✗ No clear pair structure in coherence")
    
    return coherences


def self_similar_gap_test():
    """
    The self-similar prediction: external gap at level N = internal gap at level N+1.
    
    Test by running the recursive simulation and checking if 
    closure cadences at different levels show the predicted gap structure.
    """
    print(f"\n{'='*60}")
    print(f"SELF-SIMILAR GAP TEST")
    print(f"{'='*60}")
    
    from exp8_recursive import run_recursive_sim
    
    # Run long simulation to get spectral resolution at Level 1
    levels = run_recursive_sim(n_cells=80, n_steps=3000, 
                               perturbation=0.4, n_levels=2, seed=42)
    
    # Level 0 spectrum
    rates_l0 = np.array(levels[0]["closure_rate_ts"])
    centered_l0 = rates_l0 - rates_l0.mean()
    fft_l0 = np.fft.rfft(centered_l0)
    power_l0 = np.abs(fft_l0) ** 2
    freqs_l0 = np.fft.rfftfreq(len(centered_l0))
    
    if np.max(power_l0[1:]) > 0:
        height = np.max(power_l0[1:]) * 0.05
        indices, _ = find_peaks(power_l0[1:], height=height, distance=2)
        indices += 1
        peaks_l0 = freqs_l0[indices]
        
        print(f"\n  Level 0 spectral peaks: {[f'{f:.4f}' for f in peaks_l0[:10]]}")
        
        # Check for octave/Fibonacci structure in peak spacing
        if len(peaks_l0) >= 2:
            spacings = np.diff(peaks_l0[:10])
            print(f"  Peak spacings: {[f'{s:.4f}' for s in spacings]}")
            
            if len(spacings) >= 2:
                ratios = spacings[1:] / spacings[:-1]
                print(f"  Spacing ratios: {[f'{r:.3f}' for r in ratios]}")
                
                # Check if ratios are near golden ratio or 2
                for r in ratios:
                    if abs(r - 1.618) < 0.2:
                        print(f"    → {r:.3f} ≈ φ (golden ratio)")
                    elif abs(r - 2.0) < 0.2:
                        print(f"    → {r:.3f} ≈ 2 (octave)")
                    elif abs(r - 1.0) < 0.2:
                        print(f"    → {r:.3f} ≈ 1 (uniform)")
    
    # Level 1 (if enough closures)
    rates_l1 = np.array(levels[1]["closure_rate_ts"])
    nonzero_l1 = np.sum(rates_l1 > 0)
    print(f"\n  Level 1: {nonzero_l1} nonzero steps out of {len(rates_l1)}")
    print(f"  Level 1 total closures: {levels[1]['closure_count']}")
    
    if nonzero_l1 > 20:
        centered_l1 = rates_l1 - rates_l1.mean()
        fft_l1 = np.fft.rfft(centered_l1)
        power_l1 = np.abs(fft_l1) ** 2
        freqs_l1 = np.fft.rfftfreq(len(centered_l1))
        
        if np.max(power_l1[1:]) > 0:
            height = np.max(power_l1[1:]) * 0.05
            indices, _ = find_peaks(power_l1[1:], height=height, distance=2)
            indices += 1
            peaks_l1 = freqs_l1[indices]
            
            print(f"  Level 1 spectral peaks: {[f'{f:.4f}' for f in peaks_l1[:10]]}")
            
            # Compare: is Level 1's peak spacing related to Level 0's?
            if len(peaks_l0) >= 2 and len(peaks_l1) >= 2:
                spacing_l0 = np.median(np.diff(peaks_l0[:10]))
                spacing_l1 = np.median(np.diff(peaks_l1[:10]))
                ratio = spacing_l0 / spacing_l1 if spacing_l1 > 0 else float('inf')
                print(f"\n  Median spacing L0: {spacing_l0:.6f}")
                print(f"  Median spacing L1: {spacing_l1:.6f}")
                print(f"  Ratio L0/L1: {ratio:.2f}")
                
                log2_r = np.log2(ratio) if ratio > 0 else 0
                print(f"  log₂(ratio) = {log2_r:.2f}")
    else:
        print(f"  Not enough Level 1 data for spectral analysis")


def main():
    print("=" * 60)
    print("HARMONIC PAIR ANALYSIS")
    print("=" * 60)
    
    # Test 1: Extended spectrum — find all occupied harmonics
    print(f"\n{'='*60}")
    print(f"EXTENDED SPECTRAL ANALYSIS")
    print(f"{'='*60}")
    result = extended_spectral_analysis()
    
    # Test 2: Pair resonance
    coherences = pair_resonance_test()
    
    # Test 3: Self-similar gaps
    self_similar_gap_test()
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"""
  The pair hypothesis: forces come in Fibonacci-gapped pairs.
  
  Known:     (3, 5) gap=2   Nuclear forces
             (18, 23) gap=5  Cosmic forces
  
  Predicted: (57, 70) gap=13
             (159, 193) gap=34
  
  Self-similar rule: external gap N = internal gap N+1
  All gaps are every-other-Fibonacci: F(3), F(5), F(7), F(9)...
  
  This is testable in Walter's framework: run Closure-SDK with 
  enough hierarchical levels to see if pair structure emerges 
  at harmonics 57 and 70.
""")
    
    import json
    output = {
        "occupied_harmonics": result["occupied"] if result else [],
        "found_pairs": [(a, b, g) for a, b, g in result["found_pairs"]] if result else [],
        "f0": result["f0"] if result else 0,
        "coherences": coherences,
    }
    output_path = os.path.join(os.path.dirname(__file__), "..", "results", "analysis_pairs.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved to {output_path}")


if __name__ == "__main__":
    main()
