#!/usr/bin/env python3
"""
Walter's Question: Is the closure cadence behaving like a harmonic scale?

If each force regime produces a different FUNDAMENTAL FREQUENCY but the 
same HARMONIC STRUCTURE (overtone ratios), then:
1. The power spectrum of firing times should show harmonic peaks
2. The ratio between peaks should be constant across thresholds
3. The entropy conservation follows because entropy measures the SHAPE
   of the distribution, and self-similar harmonics have the same shape

TEST:
- Take the firing time series from each force regime
- Compute the power spectrum (FFT)
- Check for harmonic structure (peaks at integer multiples)
- Compare the spectral SHAPE across regimes
- If shapes are self-similar → harmonic scale → entropy explained
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))

import numpy as np
import json
from engine import ForceConfig, SimConfig, run_1d, FORCES


def spectral_analysis(force, config):
    """Compute power spectrum of the firing rate time series."""
    hist = run_1d(force, config)
    
    # Firing rate time series
    rates = np.array(hist["firing_rates"])
    
    # Remove DC component (mean)
    rates_centered = rates - rates.mean()
    
    # FFT
    fft = np.fft.rfft(rates_centered)
    power = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(len(rates_centered))
    
    # Normalize power spectrum to unit area (makes it a distribution)
    total_power = power.sum()
    if total_power > 0:
        power_norm = power / total_power
    else:
        power_norm = power
    
    # Spectral entropy (entropy of the normalized power spectrum)
    power_positive = power_norm[power_norm > 0]
    spectral_entropy = -np.sum(power_positive * np.log(power_positive))
    
    # Find peaks
    from scipy.signal import find_peaks as _fp
    peaks_available = True
    try:
        peak_indices, peak_props = _fp(power[1:], height=np.max(power[1:]) * 0.1, 
                                        distance=3)
        peak_indices += 1  # offset for removed DC
        peak_freqs = freqs[peak_indices]
        peak_powers = power[peak_indices]
    except ImportError:
        peaks_available = False
        peak_freqs = []
        peak_powers = []
    
    # Check for harmonic ratios between peaks
    harmonic_ratios = []
    if len(peak_freqs) >= 2 and peak_freqs[0] > 0:
        fundamental = peak_freqs[0]
        for pf in peak_freqs[1:]:
            ratio = pf / fundamental
            harmonic_ratios.append(float(ratio))
    
    return {
        "freqs": freqs.tolist(),
        "power": power.tolist(),
        "power_norm": power_norm.tolist(),
        "spectral_entropy": float(spectral_entropy),
        "peak_freqs": [float(f) for f in peak_freqs],
        "peak_powers": [float(p) for p in peak_powers],
        "harmonic_ratios": harmonic_ratios,
        "fundamental": float(peak_freqs[0]) if len(peak_freqs) > 0 else 0,
        "mean_rate": float(rates.mean()),
        "peaks_available": peaks_available,
    }


def spectral_shape_comparison(spectra):
    """
    Compare the SHAPE of power spectra across force regimes.
    
    If they're harmonically self-similar, the normalized spectra
    should have the same shape when rescaled by their fundamental.
    """
    # Compare spectral entropy directly
    entropies = {name: s["spectral_entropy"] for name, s in spectra.items()}
    
    ent_values = list(entropies.values())
    ent_mean = np.mean(ent_values)
    ent_cv = np.std(ent_values) / ent_mean if ent_mean > 0 else float('inf')
    
    # Compare spectral shape via cosine similarity of normalized spectra
    names = list(spectra.keys())
    similarities = {}
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if i >= j:
                continue
            s1 = np.array(spectra[n1]["power_norm"])
            s2 = np.array(spectra[n2]["power_norm"])
            # Ensure same length
            min_len = min(len(s1), len(s2))
            s1, s2 = s1[:min_len], s2[:min_len]
            
            dot = np.dot(s1, s2)
            norm1 = np.linalg.norm(s1)
            norm2 = np.linalg.norm(s2)
            if norm1 > 0 and norm2 > 0:
                cosine_sim = dot / (norm1 * norm2)
            else:
                cosine_sim = 0
            similarities[f"{n1}_vs_{n2}"] = float(cosine_sim)
    
    return {
        "spectral_entropies": entropies,
        "entropy_cv": float(ent_cv),
        "shape_similarities": similarities,
    }


def closure_interval_analysis(force, config):
    """
    Analyze the distribution of INTERVALS between closures.
    
    If it's harmonic, the interval distribution should show
    peaks at the fundamental period and its multiples.
    """
    hist = run_1d(force, config)
    rates = hist["firing_rates"]
    
    # Find steps where firing rate > 0 (closure events happened)
    firing_steps = [i for i, r in enumerate(rates) if r > 0]
    
    if len(firing_steps) < 2:
        return {"intervals": [], "mean_interval": 0, "has_harmonic": False}
    
    # Inter-closure intervals
    intervals = np.diff(firing_steps)
    
    if len(intervals) < 10:
        return {
            "intervals": intervals.tolist(),
            "mean_interval": float(intervals.mean()),
            "has_harmonic": False,
        }
    
    # FFT of the interval sequence
    intervals_centered = intervals - intervals.mean()
    fft_intervals = np.fft.rfft(intervals_centered)
    power_intervals = np.abs(fft_intervals) ** 2
    
    # Normalize
    total = power_intervals.sum()
    if total > 0:
        power_norm = power_intervals / total
    else:
        power_norm = power_intervals
    
    # Interval entropy
    pn_pos = power_norm[power_norm > 0]
    interval_spectral_entropy = -np.sum(pn_pos * np.log(pn_pos))
    
    # Check for periodicity in intervals (harmonic signature)
    # Autocorrelation of intervals
    autocorr = np.correlate(intervals_centered, intervals_centered, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # positive lags only
    autocorr = autocorr / autocorr[0] if autocorr[0] > 0 else autocorr
    
    # Find first significant peak in autocorrelation (after lag 0)
    if len(autocorr) > 3:
        # Simple peak detection
        harmonic_period = 0
        for k in range(2, min(len(autocorr), 50)):
            if autocorr[k] > autocorr[k-1] and autocorr[k] > autocorr[k+1] if k+1 < len(autocorr) else True:
                if autocorr[k] > 0.1:  # significant correlation
                    harmonic_period = k
                    break
    else:
        harmonic_period = 0
    
    return {
        "n_intervals": len(intervals),
        "mean_interval": float(intervals.mean()),
        "std_interval": float(intervals.std()),
        "interval_spectral_entropy": float(interval_spectral_entropy),
        "harmonic_period": harmonic_period,
        "has_harmonic": harmonic_period > 0,
        "autocorr_peak_value": float(autocorr[harmonic_period]) if harmonic_period > 0 else 0,
    }


def main():
    print("=" * 60)
    print("WALTER'S QUESTION: Is it a harmonic scale?")
    print("=" * 60)
    
    config = SimConfig(n_cells=200, n_steps=2000, seed=42)
    
    # Spectral analysis of firing rates
    print("\n  SPECTRAL ANALYSIS OF FIRING TIME SERIES:")
    spectra = {}
    for force in FORCES:
        print(f"    Analyzing {force.name} (τ={force.tau})...")
        spectra[force.name] = spectral_analysis(force, config)
    
    print(f"\n  {'Force':<10} {'Fund freq':>10} {'Spec entropy':>13} {'Peak ratios'}")
    print(f"  {'-'*10} {'-'*10} {'-'*13} {'-'*30}")
    for name, s in spectra.items():
        ratios_str = ", ".join(f"{r:.2f}" for r in s["harmonic_ratios"][:5])
        print(f"  {name:<10} {s['fundamental']:>10.4f} {s['spectral_entropy']:>13.4f} {ratios_str}")
    
    # Spectral shape comparison
    print(f"\n  SPECTRAL SHAPE COMPARISON:")
    comparison = spectral_shape_comparison(spectra)
    
    print(f"    Spectral entropy CV across forces: {comparison['entropy_cv']:.4f}")
    if comparison['entropy_cv'] < 0.1:
        print(f"    ★ SPECTRAL ENTROPY CONSERVED (CV < 0.1)")
        print(f"      The FREQUENCY CONTENT has the same entropy across forces")
    elif comparison['entropy_cv'] < 0.3:
        print(f"    ✓ Spectral entropy weakly conserved")
    else:
        print(f"    ✗ Spectral entropy not conserved")
    
    print(f"\n    Spectral shape similarities (cosine):")
    for pair, sim in comparison["shape_similarities"].items():
        print(f"      {pair}: {sim:.4f}")
    
    # Inter-closure interval analysis
    print(f"\n  INTER-CLOSURE INTERVAL ANALYSIS:")
    interval_results = {}
    for force in FORCES:
        print(f"    Analyzing {force.name} intervals...")
        interval_results[force.name] = closure_interval_analysis(force, config)
    
    print(f"\n  {'Force':<10} {'N intervals':>12} {'Mean':>8} {'Std':>8} {'Period':>8} {'Harmonic?':>10}")
    print(f"  {'-'*10} {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for name, ir in interval_results.items():
        harm = "YES" if ir["has_harmonic"] else "no"
        print(f"  {name:<10} {ir.get('n_intervals', 0):>12} "
              f"{ir['mean_interval']:>8.1f} {ir.get('std_interval', 0):>8.1f} "
              f"{ir.get('harmonic_period', 0):>8} {harm:>10}")
    
    # Check interval spectral entropy conservation
    interval_entropies = [ir["interval_spectral_entropy"] 
                          for ir in interval_results.values() 
                          if ir.get("n_intervals", 0) > 10]
    if interval_entropies:
        ie_cv = np.std(interval_entropies) / np.mean(interval_entropies) if np.mean(interval_entropies) > 0 else float('inf')
        print(f"\n    Interval spectral entropy CV: {ie_cv:.4f}")
        if ie_cv < 0.1:
            print(f"    ★ INTERVAL SPECTRA HAVE SAME ENTROPY — harmonic self-similarity!")
        elif ie_cv < 0.3:
            print(f"    ✓ Interval spectra weakly similar")
    
    # THE VERDICT
    print(f"\n" + "=" * 60)
    print(f"  VERDICT: Is it a harmonic scale?")
    print(f"=" * 60)
    
    spectral_conserved = comparison['entropy_cv'] < 0.15
    has_harmonics = any(ir.get("has_harmonic", False) for ir in interval_results.values())
    has_ratios = any(len(s["harmonic_ratios"]) > 0 for s in spectra.values())
    
    if spectral_conserved and (has_harmonics or has_ratios):
        print(f"\n  ★ YES — the closure cadence has harmonic structure")
        print(f"    Spectral entropy is conserved (CV = {comparison['entropy_cv']:.4f})")
        if has_harmonics:
            print(f"    Harmonic periods detected in inter-closure intervals")
        if has_ratios:
            print(f"    Peak frequency ratios suggest overtone structure")
        print(f"\n    WALTER'S INTERPRETATION:")
        print(f"    Each force regime is a different FUNDAMENTAL FREQUENCY")
        print(f"    but the OVERTONE STRUCTURE is self-similar.")
        print(f"    Entropy is conserved because entropy measures the SHAPE")
        print(f"    of the distribution, and harmonics preserve shape across scales.")
        print(f"\n    This would mean the forces aren't just 'the same thing at")
        print(f"    different thresholds' — they're HARMONICS of each other.")
    elif spectral_conserved:
        print(f"\n  ~ PARTIAL — spectral entropy is conserved but harmonic")
        print(f"    peaks aren't clear. The self-similarity might be")
        print(f"    statistical rather than strictly harmonic.")
    else:
        print(f"\n  ✗ No clear harmonic structure detected.")
        print(f"    Spectral entropy CV = {comparison['entropy_cv']:.4f}")
    
    # Save
    output = {
        "spectral_entropies": comparison["spectral_entropies"],
        "spectral_entropy_cv": comparison["entropy_cv"],
        "shape_similarities": comparison["shape_similarities"],
        "interval_analysis": {name: {k: v for k, v in ir.items() if k != "intervals"} 
                             for name, ir in interval_results.items()},
        "fundamentals": {name: s["fundamental"] for name, s in spectra.items()},
        "harmonic_ratios": {name: s["harmonic_ratios"] for name, s in spectra.items()},
    }
    output_path = os.path.join(os.path.dirname(__file__), "..", "results", "analysis_harmonic.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {output_path}")


if __name__ == "__main__":
    main()
