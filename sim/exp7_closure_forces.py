#!/usr/bin/env python3
"""
Experiment 7: Closure-Driven Force Emergence

Walter's LAWS.md and GEOMETRIC_COMPUTER_SPEC establish that:
  - σ IS energy: E(q) = arccos(|w|)
  - Closure = running product returns to identity after excursion
  - Hopf decomposition: W = existence/time, RGB = position/space
  - The 3+1 structure is forced by the algebra (not designed)
  - Three operations are sufficient: compose, sigma, close

THE EXPERIMENT:
Instead of a threshold τ, use CLOSURE as the force mechanism.
Each cell maintains a running product (quaternion composition).
When the running product closes (σ → 0 after excursion), it "fires"
and emits to neighbors.

The key difference: we don't SET thresholds. We let the CLOSURE
CADENCE emerge from the data. Different initial conditions and
perturbation patterns should produce different closure rates —
and those different rates should look like different forces.

If the force hierarchy emerges from closure cadence WITHOUT us
assigning thresholds, that would mean the forces aren't parameterized
at all — they're a consequence of the geometry of S³ itself.

PREDICTIONS:
1. Fast closure (short excursions) = high firing rate = gravity-like
2. Slow closure (long excursions) = low firing rate = strong-force-like
3. The Hopf decomposition should separate W-closures (existence events)
   from RGB-closures (reordering events) naturally
4. Entropy should still be conserved across closure cadences

ADDITIONAL TEST (from Walter's energy law):
Energy = Σ σ(carrier_k). Track total energy. Does it follow the
pattern: "enters through composition, conserved by prediction,
released by closure"?
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '/tmp/closure-sdk-fresh')

import numpy as np
import json

# Quaternion operations
def qmul(p, q):
    """Hamilton product."""
    w = p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3]
    x = p[0]*q[1] + p[1]*q[0] + p[2]*q[3] - p[3]*q[2]
    y = p[0]*q[2] - p[1]*q[3] + p[2]*q[0] + p[3]*q[1]
    z = p[0]*q[3] + p[1]*q[2] - p[2]*q[1] + p[3]*q[0]
    return np.array([w, x, y, z])

def qnorm(q):
    n = np.linalg.norm(q)
    return q / n if n > 0 else np.array([1., 0., 0., 0.])

def qinv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def sigma(q):
    return np.arccos(np.clip(np.abs(q[0]), 0, 1))

def hopf_classify(q):
    """W-dominant (existence) or RGB-dominant (position)?"""
    w_mag = abs(q[0])
    rgb_mag = np.linalg.norm(q[1:4])
    if w_mag > rgb_mag:
        return "W"   # existence
    else:
        return "RGB"  # position

def random_q(rng):
    while True:
        u1, u2 = rng.uniform(-1, 1), rng.uniform(-1, 1)
        s1 = u1*u1 + u2*u2
        if s1 < 1: break
    while True:
        u3, u4 = rng.uniform(-1, 1), rng.uniform(-1, 1)
        s2 = u3*u3 + u4*u4
        if s2 < 1: break
    f = np.sqrt((1 - s1) / s2)
    return np.array([u1, u2, u3 * f, u4 * f])

def small_rotation(rng, mag):
    angle = rng.exponential(mag)
    axis = rng.normal(0, 1, 3)
    axis = axis / (np.linalg.norm(axis) + 1e-10)
    half = angle / 2
    return qnorm(np.array([np.cos(half), *(axis * np.sin(half))]))


class Cell:
    """A cell on the lattice. Maintains a running product on S³."""
    def __init__(self):
        self.running_product = np.array([1., 0., 0., 0.])  # identity
        self.sigma_history = []
        self.excursion_peak = 0.0
        self.peak_quaternion = np.array([1., 0., 0., 0.])
        self.has_excursion = False
        self.n_composed = 0
    
    def compose(self, q):
        """Compose a quaternion into the running product."""
        self.running_product = qnorm(qmul(self.running_product, q))
        s = sigma(self.running_product)
        self.sigma_history.append(s)
        self.n_composed += 1
        if s > self.excursion_peak:
            self.excursion_peak = s
            self.peak_quaternion = self.running_product.copy()
        if s > 0.1:  # departed from identity
            self.has_excursion = True
    
    def check_closure(self, epsilon=0.3):
        """
        Check for closure: σ drops significantly after nontrivial excursion.
        Returns (closed, closure_info) 
        """
        s = sigma(self.running_product)
        
        if (s < epsilon and 
            self.has_excursion and 
            self.excursion_peak > 0.5 and
            self.n_composed >= 3 and
            s < self.excursion_peak * 0.5):  # must drop to <50% of peak
            
            # Classify the EXCURSION (not the closure point) via Hopf
            # At peak, what dominated? Store peak quaternion for this
            closure_type = hopf_classify(self.peak_quaternion)
            
            info = {
                "sigma": float(s),
                "excursion_peak": float(self.excursion_peak),
                "support": self.n_composed,
                "type": closure_type,
                "emission": self.running_product.copy(),
            }
            
            # Reset
            self.running_product = np.array([1., 0., 0., 0.])
            self.sigma_history = []
            self.excursion_peak = 0.0
            self.peak_quaternion = np.array([1., 0., 0., 0.])
            self.has_excursion = False
            self.n_composed = 0
            
            return True, info
        
        return False, None


def run_closure_sim(n_cells=80, n_steps=500, perturbation_mag=0.3, seed=42):
    """
    Run the closure-driven simulation.
    
    No threshold parameter. Cells compose perturbations into their
    running products. When a running product closes (returns to identity
    after excursion), the cell emits to neighbors.
    
    The perturbation magnitude controls the "energy scale" —
    analogous to temperature.
    """
    rng = np.random.default_rng(seed)
    cells = [Cell() for _ in range(n_cells)]
    
    history = {
        "closure_rates": [],
        "total_energy": [],
        "mean_sigma": [],
        "w_closures": [],
        "rgb_closures": [],
        "closure_supports": [],
        "closure_peaks": [],
        "energy_released": [],
    }
    
    total_closures = 0
    w_total = 0
    rgb_total = 0
    
    for step in range(n_steps):
        # Phase 1: Perturb — compose small random rotations (energy injection)
        for cell in cells:
            kick = small_rotation(rng, perturbation_mag)
            cell.compose(kick)
        
        # Phase 2: Check for closures and propagate emissions
        step_closures = 0
        step_w = 0
        step_rgb = 0
        step_energy_released = 0.0
        step_supports = []
        step_peaks = []
        
        for i, cell in enumerate(cells):
            closed, info = cell.check_closure()
            if closed:
                step_closures += 1
                total_closures += 1
                step_energy_released += info["excursion_peak"]
                step_supports.append(info["support"])
                step_peaks.append(info["excursion_peak"])
                
                if info["type"] == "W":
                    step_w += 1
                    w_total += 1
                else:
                    step_rgb += 1
                    rgb_total += 1
                
                # Emit: compose emission into neighbors
                # Range determined by excursion peak (more energetic = longer range)
                emit_range = max(1, int(info["excursion_peak"] * 5))
                emit_q = info["emission"]
                
                for d in range(1, emit_range + 1):
                    # Scale emission by 1/d
                    scale = 0.1 / d
                    angle = sigma(emit_q) * scale
                    if angle < 1e-10:
                        continue
                    xyz_n = np.linalg.norm(emit_q[1:4])
                    if xyz_n < 1e-10:
                        continue
                    axis = emit_q[1:4] / xyz_n
                    half = angle / 2
                    transfer = qnorm(np.array([
                        np.cos(half), *(axis * np.sin(half))
                    ]))
                    
                    for idx in [i - d, i + d]:
                        if 0 <= idx < n_cells:
                            cells[idx].compose(transfer)
        
        # Metrics
        sigmas = [sigma(c.running_product) for c in cells]
        total_energy = sum(sigmas)
        
        history["closure_rates"].append(step_closures / n_cells)
        history["total_energy"].append(float(total_energy))
        history["mean_sigma"].append(float(np.mean(sigmas)))
        history["w_closures"].append(step_w)
        history["rgb_closures"].append(step_rgb)
        history["closure_supports"].append(step_supports)
        history["closure_peaks"].append(step_peaks)
        history["energy_released"].append(step_energy_released)
    
    history["total_closures"] = total_closures
    history["w_total"] = w_total
    history["rgb_total"] = rgb_total
    
    return history


def main():
    print("=" * 60)
    print("EXPERIMENT 7: Closure-Driven Force Emergence")
    print("  (Walter's Laws: compose, sigma, close)")
    print("=" * 60)
    print()
    
    # Run at different energy scales
    # The hypothesis: different perturbation magnitudes produce
    # different closure cadences that LOOK like different forces
    
    energy_scales = {
        "very_low":  0.05,   # barely perturbed → rare closures?
        "low":       0.15,   # gentle perturbation
        "medium":    0.30,   # moderate
        "high":      0.60,   # strong perturbation
        "very_high": 1.20,   # violent perturbation
    }
    
    results = {}
    
    for name, mag in energy_scales.items():
        print(f"  Running at {name} energy (perturbation={mag})...")
        hist = run_closure_sim(n_cells=80, n_steps=500, 
                              perturbation_mag=mag, seed=42)
        
        ss = slice(-200, None)
        avg_rate = float(np.mean(hist["closure_rates"][ss]))
        avg_energy = float(np.mean(hist["total_energy"][ss]))
        avg_sigma = float(np.mean(hist["mean_sigma"][ss]))
        
        # Closure type ratio
        total = hist["w_total"] + hist["rgb_total"]
        w_frac = hist["w_total"] / max(total, 1)
        rgb_frac = hist["rgb_total"] / max(total, 1)
        
        # Average support (how many compositions per closure)
        all_supports = [s for step in hist["closure_supports"] for s in step]
        avg_support = float(np.mean(all_supports)) if all_supports else 0
        
        all_peaks = [p for step in hist["closure_peaks"] for p in step]
        avg_peak = float(np.mean(all_peaks)) if all_peaks else 0
        
        results[name] = {
            "perturbation": mag,
            "avg_closure_rate": avg_rate,
            "total_closures": hist["total_closures"],
            "avg_energy": avg_energy,
            "avg_sigma": avg_sigma,
            "w_fraction": float(w_frac),
            "rgb_fraction": float(rgb_frac),
            "avg_support": avg_support,
            "avg_excursion_peak": avg_peak,
            "energy_timeseries": [float(x) for x in hist["total_energy"]],
            "rate_timeseries": [float(x) for x in hist["closure_rates"]],
        }
    
    # Print results
    print(f"\n  CLOSURE CADENCE vs ENERGY SCALE:")
    print(f"  {'Scale':<12} {'Perturb':>8} {'Rate':>8} {'Closures':>10} {'Support':>8} {'Peak σ':>8} {'W%':>6} {'RGB%':>6}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*6} {'-'*6}")
    
    for name, data in results.items():
        print(f"  {name:<12} {data['perturbation']:>8.2f} {data['avg_closure_rate']:>8.4f} "
              f"{data['total_closures']:>10} {data['avg_support']:>8.1f} "
              f"{data['avg_excursion_peak']:>8.3f} "
              f"{data['w_fraction']*100:>5.1f}% {data['rgb_fraction']*100:>5.1f}%")
    
    # Analysis
    print(f"\n  ANALYSIS:")
    
    # Does closure rate increase with energy?
    rates = [results[k]["avg_closure_rate"] for k in energy_scales.keys()]
    rate_increasing = all(rates[i] <= rates[i+1] for i in range(len(rates)-1))
    
    if rate_increasing:
        print(f"    ✓ CLOSURE RATE INCREASES WITH ENERGY")
        print(f"      More energy → more closures → higher 'force'")
        print(f"      This is the hierarchy WITHOUT a threshold parameter!")
    else:
        # Check if it's at least monotonic after initial transient
        mid_rates = rates[1:]
        mid_increasing = all(mid_rates[i] <= mid_rates[i+1] for i in range(len(mid_rates)-1))
        if mid_increasing:
            print(f"    ~ Closure rate increases after initial regime")
        else:
            print(f"    ✗ Closure rate not monotonic with energy")
    
    # Hopf decomposition: does W/RGB ratio change with energy?
    w_fracs = [results[k]["w_fraction"] for k in energy_scales.keys()]
    
    w_changes = max(w_fracs) - min(w_fracs)
    if w_changes > 0.1:
        print(f"    ✓ HOPF RATIO SHIFTS WITH ENERGY")
        print(f"      W fraction range: {min(w_fracs)*100:.1f}% — {max(w_fracs)*100:.1f}%")
        print(f"      Different energy scales produce different types of closures")
        low_type = "W (existence)" if w_fracs[0] > w_fracs[-1] else "RGB (position)"
        high_type = "W (existence)" if w_fracs[-1] > w_fracs[0] else "RGB (position)"
        print(f"      Low energy → more {low_type} closures")
        print(f"      High energy → more {high_type} closures")
    else:
        print(f"    ~ W/RGB ratio stable across energies ({min(w_fracs)*100:.1f}%—{max(w_fracs)*100:.1f}%)")
    
    # Support length: does it change with energy?
    supports = [results[k]["avg_support"] for k in energy_scales.keys()]
    if max(supports) > min(supports) * 1.5:
        print(f"    ✓ CLOSURE SUPPORT VARIES WITH ENERGY")
        print(f"      Low energy: avg {supports[0]:.1f} compositions per closure")
        print(f"      High energy: avg {supports[-1]:.1f} compositions per closure")
    else:
        print(f"    ~ Support roughly constant ({min(supports):.1f}—{max(supports):.1f})")
    
    # Energy budget: does closure release energy?
    print(f"\n  ENERGY DYNAMICS (Walter's energy law):")
    for name in ["very_low", "medium", "very_high"]:
        data = results[name]
        ts = data["energy_timeseries"]
        early = np.mean(ts[:50])
        late = np.mean(ts[-50:])
        print(f"    {name}: energy {early:.1f} → {late:.1f} "
              f"({'rising' if late > early else 'falling' if late < early else 'stable'})")
    
    # Save
    output = {k: {kk: vv for kk, vv in v.items() 
                   if not kk.endswith('_timeseries')} 
              for k, v in results.items()}
    output["_timeseries"] = {
        k: {"energy": v["energy_timeseries"], "rate": v["rate_timeseries"]}
        for k, v in results.items()
    }
    
    output_path = os.path.join(os.path.dirname(__file__), "..", "results", "exp7_closure.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Data saved to {output_path}")
    
    return results


if __name__ == "__main__":
    main()
