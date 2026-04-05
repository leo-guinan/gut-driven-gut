#!/usr/bin/env python3
"""
Experiment 6: Quaternion-Valued States on S³ (Closure-SDK Integration)

THE BIG QUESTION:
If cell states live on S³ (unit quaternions) instead of ℝ, and the
"update function" is quaternion composition instead of scalar addition,
does the four-channel structure (W, X, Y, Z) naturally separate into
behaviors resembling different forces?

STRUCTURAL FACTS:
- SU(2) ≅ S³ (the 3-sphere of unit quaternions)
- The Standard Model gauge group is SU(3) × SU(2) × U(1)
- SU(2) IS the weak force symmetry group
- Closure-SDK's σ = arccos(|w|) is geodesic distance from identity on S³
- σ is left-invariant (cost is context-free) and satisfies triangle inequality

THE EXPERIMENT:
1. Lattice of cells, each a unit quaternion (point on S³)
2. "Energy" = σ(cell) = geodesic distance from identity
3. "Firing" threshold: cell fires when σ(cell) ≥ τ
4. On firing: compose cell's state with neighbors via Hamilton product
5. Range still = 1/τ
6. Track ALL FOUR channels (W, X, Y, Z) separately

PREDICTIONS:
- The W channel (scalar part) should behave like the original scalar model
  (sanity check — our Exp 0 results should reproduce in the W channel)
- The XYZ channels (vector part) should show NEW behaviors:
  - Non-commutativity: order of composition matters → directional effects
  - Cancellation: xyz components can cancel while w accumulates
  - Channel separation: different channels might show different force signatures

THE DEEP QUESTION:
Does the geometry of S³ itself create distinct behavioral modes that map
to different forces — without us assigning different thresholds?

If YES: the force hierarchy could emerge from geometry, not parameters.
If NO: S³ adds structure but doesn't change the fundamental story.

Requires: pip install closure-sdk (from github.com/faltz009/Closure-SDK)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '/tmp/closure-sdk-fresh')

import numpy as np
import json
from engine import SimConfig

# Try to import Closure-SDK
try:
    from closure_sdk.ops import compose as q_compose, sigma as q_sigma, invert as q_invert, embed as q_embed
    from closure_sdk.state import ClosureState
    HAS_CLOSURE = True
except ImportError:
    HAS_CLOSURE = False
    print("WARNING: closure_sdk not available. Using pure numpy quaternion math.")


# ─────────────────────────────────────────────
# Pure numpy quaternion operations (fallback)
# ─────────────────────────────────────────────

def quat_multiply(p, q):
    """Hamilton product of two quaternions [w, x, y, z]."""
    w = p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3]
    x = p[0]*q[1] + p[1]*q[0] + p[2]*q[3] - p[3]*q[2]
    y = p[0]*q[2] - p[1]*q[3] + p[2]*q[0] + p[3]*q[1]
    z = p[0]*q[3] + p[1]*q[2] - p[2]*q[1] + p[3]*q[0]
    return np.array([w, x, y, z])

def quat_conjugate(q):
    """Conjugate (= inverse for unit quaternions)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_sigma(q):
    """σ = arccos(|w|) — geodesic distance from identity on S³."""
    return np.arccos(np.clip(np.abs(q[0]), -1, 1))

def quat_normalize(q):
    """Normalize to unit quaternion."""
    n = np.linalg.norm(q)
    return q / n if n > 0 else np.array([1.0, 0.0, 0.0, 0.0])

def random_quaternion(rng):
    """Random unit quaternion (uniform on S³)."""
    # Marsaglia method for uniform S³
    while True:
        u1, u2 = rng.uniform(-1, 1), rng.uniform(-1, 1)
        s1 = u1*u1 + u2*u2
        if s1 < 1:
            break
    while True:
        u3, u4 = rng.uniform(-1, 1), rng.uniform(-1, 1)
        s2 = u3*u3 + u4*u4
        if s2 < 1:
            break
    factor = np.sqrt((1 - s1) / s2)
    return np.array([u1, u2, u3 * factor, u4 * factor])

def small_rotation(rng, magnitude=0.1):
    """Small random rotation quaternion near identity."""
    # Axis-angle: small angle, random axis
    angle = rng.exponential(magnitude)
    axis = rng.normal(0, 1, 3)
    axis = axis / np.linalg.norm(axis)
    half = angle / 2
    w = np.cos(half)
    xyz = axis * np.sin(half)
    return quat_normalize(np.array([w, xyz[0], xyz[1], xyz[2]]))


# ─────────────────────────────────────────────
# Closure-SDK wrappers
# ─────────────────────────────────────────────

if HAS_CLOSURE:
    def cs_compose(q1_arr, q2_arr):
        """Compose using Closure-SDK."""
        s1 = ClosureState(group="Sphere", element=q1_arr)
        s2 = ClosureState(group="Sphere", element=q2_arr)
        result = q_compose(s1, s2)
        return result.element
    
    def cs_sigma(q_arr):
        """Sigma using Closure-SDK."""
        s = ClosureState(group="Sphere", element=q_arr)
        return q_sigma(s)
    
    def cs_invert(q_arr):
        """Invert using Closure-SDK."""
        s = ClosureState(group="Sphere", element=q_arr)
        return q_invert(s).element
    
    COMPOSE = cs_compose
    SIGMA = cs_sigma
    INVERT = cs_invert
else:
    COMPOSE = quat_multiply
    SIGMA = quat_sigma
    INVERT = quat_conjugate


# ─────────────────────────────────────────────
# Quaternion update function
# ─────────────────────────────────────────────

def update_quaternion(cells, tau):
    """
    Universal update on S³.
    
    cells: (n, 4) array of unit quaternions
    tau: activation threshold (σ value)
    
    Firing condition: σ(cell) ≥ τ
    On firing: compose cell state with neighbors via Hamilton product
    """
    n = len(cells)
    sigmas = np.array([SIGMA(cells[i]) for i in range(n)])
    
    firing_mask = sigmas >= tau
    n_fired = int(firing_mask.sum())
    
    if n_fired == 0:
        return {
            "n_fired": 0, "firing_rate": 0.0,
            "effective_range": 0.0,
            "mean_sigma": float(sigmas.mean()),
            "mean_w": float(np.abs(cells[:, 0]).mean()),
            "mean_xyz_norm": float(np.linalg.norm(cells[:, 1:4], axis=1).mean()),
            "w_std": float(cells[:, 0].std()),
            "x_std": float(cells[:, 1].std()),
            "y_std": float(cells[:, 2].std()),
            "z_std": float(cells[:, 3].std()),
            "channel_entropy": [0.0, 0.0, 0.0, 0.0],
        }
    
    effective_range = max(1, min(int(np.ceil(1.0 / tau)), n // 2))
    
    new_cells = cells.copy()
    
    for i in np.where(firing_mask)[0]:
        # Compose this cell's state with neighbors
        # The "transfer" is composition, not addition
        for d in range(1, effective_range + 1):
            # Weight by 1/d (scaled rotation)
            scale = 1.0 / (d * effective_range)
            
            # Create a scaled version of the firing cell's state
            # SLERP between identity and cell state by scale factor
            cell_q = cells[i]
            # Small rotation in the direction of the cell's state
            angle = SIGMA(cell_q) * scale
            if angle < 1e-10:
                continue
            
            # Axis of the cell's rotation
            xyz_norm = np.linalg.norm(cell_q[1:4])
            if xyz_norm < 1e-10:
                continue
            axis = cell_q[1:4] / xyz_norm
            
            # Scaled rotation quaternion
            half_angle = angle / 2
            transfer_q = np.array([
                np.cos(half_angle),
                axis[0] * np.sin(half_angle),
                axis[1] * np.sin(half_angle),
                axis[2] * np.sin(half_angle),
            ])
            
            # Compose with left and right neighbors
            for idx in [i - d, i + d]:
                if 0 <= idx < n:
                    new_cells[idx] = quat_normalize(COMPOSE(new_cells[idx], transfer_q))
        
        # Reset firing cell toward identity
        # SLERP toward identity by 0.5
        slerp_factor = 0.5
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        dot = np.clip(np.dot(cells[i], identity), -1, 1)
        if abs(dot) < 0.9999:
            omega = np.arccos(abs(dot))
            s_omega = np.sin(omega)
            s0 = np.sin((1 - slerp_factor) * omega) / s_omega
            s1 = np.sin(slerp_factor * omega) / s_omega
            if dot < 0:
                s1 = -s1
            new_cells[i] = quat_normalize(s0 * cells[i] + s1 * identity)
        else:
            new_cells[i] = identity
    
    cells[:] = new_cells
    
    # Channel-separated metrics
    sigmas_new = np.array([SIGMA(cells[i]) for i in range(n)])
    
    # Per-channel entropy
    channel_entropy = []
    for ch in range(4):
        vals = np.abs(cells[:, ch])
        vals = vals / max(vals.sum(), 1e-10)
        vals = vals[vals > 0]
        ent = -np.sum(vals * np.log(vals))
        channel_entropy.append(float(ent))
    
    return {
        "n_fired": n_fired,
        "firing_rate": float(n_fired) / n,
        "effective_range": float(effective_range),
        "mean_sigma": float(sigmas_new.mean()),
        "mean_w": float(np.abs(cells[:, 0]).mean()),
        "mean_xyz_norm": float(np.linalg.norm(cells[:, 1:4], axis=1).mean()),
        "w_std": float(cells[:, 0].std()),
        "x_std": float(cells[:, 1].std()),
        "y_std": float(cells[:, 2].std()),
        "z_std": float(cells[:, 3].std()),
        "channel_entropy": channel_entropy,
    }


# ─────────────────────────────────────────────
# Simulation runner
# ─────────────────────────────────────────────

def run_quaternion_sim(tau, n_cells=100, n_steps=300, seed=42, perturbation=0.05):
    """Run simulation with quaternion-valued cells."""
    rng = np.random.default_rng(seed)
    
    # Initialize: random quaternions (points on S³)
    cells = np.array([random_quaternion(rng) for _ in range(n_cells)])
    
    history = {
        "firing_rates": [], "mean_sigmas": [],
        "mean_w": [], "mean_xyz": [],
        "w_std": [], "x_std": [], "y_std": [], "z_std": [],
        "channel_entropy_w": [], "channel_entropy_x": [],
        "channel_entropy_y": [], "channel_entropy_z": [],
    }
    
    for step in range(n_steps):
        # Perturbation: compose each cell with a small random rotation
        # (analog of energy injection in scalar model)
        for i in range(n_cells):
            kick = small_rotation(rng, perturbation)
            cells[i] = quat_normalize(COMPOSE(cells[i], kick))
        
        metrics = update_quaternion(cells, tau)
        
        history["firing_rates"].append(metrics["firing_rate"])
        history["mean_sigmas"].append(metrics["mean_sigma"])
        history["mean_w"].append(metrics["mean_w"])
        history["mean_xyz"].append(metrics["mean_xyz_norm"])
        history["w_std"].append(metrics["w_std"])
        history["x_std"].append(metrics["x_std"])
        history["y_std"].append(metrics["y_std"])
        history["z_std"].append(metrics["z_std"])
        
        ce = metrics["channel_entropy"]
        history["channel_entropy_w"].append(ce[0])
        history["channel_entropy_x"].append(ce[1])
        history["channel_entropy_y"].append(ce[2])
        history["channel_entropy_z"].append(ce[3])
    
    return history


# ─────────────────────────────────────────────
# Experiment: channel separation test
# ─────────────────────────────────────────────

def channel_separation_test():
    """
    THE KEY TEST: Do the W and XYZ channels behave differently
    at the same threshold?
    
    If S³ geometry creates natural force separation, the W channel
    (scalar part = existence/magnitude) should behave differently
    from the XYZ channels (vector part = direction/orientation)
    without us putting that in by hand.
    """
    results = {}
    
    thresholds = {
        "gravity": 0.05,   # low σ threshold (note: σ range is [0, π/2])
        "EM": 0.3,
        "weak": 0.7,
        "strong": 1.2,
    }
    
    for name, tau in thresholds.items():
        print(f"  Running S³ {name} (τ={tau})...")
        hist = run_quaternion_sim(tau, n_cells=80, n_steps=200, seed=42)
        
        ss = slice(-80, None)
        
        results[name] = {
            "tau": tau,
            "firing_rate": float(np.mean(hist["firing_rates"][ss])),
            "mean_sigma": float(np.mean(hist["mean_sigmas"][ss])),
            "w_channel": {
                "mean": float(np.mean(hist["mean_w"][ss])),
                "std": float(np.mean(hist["w_std"][ss])),
                "entropy": float(np.mean(hist["channel_entropy_w"][ss])),
            },
            "xyz_channels": {
                "mean_norm": float(np.mean(hist["mean_xyz"][ss])),
                "x_std": float(np.mean(hist["x_std"][ss])),
                "y_std": float(np.mean(hist["y_std"][ss])),
                "z_std": float(np.mean(hist["z_std"][ss])),
                "entropy_x": float(np.mean(hist["channel_entropy_x"][ss])),
                "entropy_y": float(np.mean(hist["channel_entropy_y"][ss])),
                "entropy_z": float(np.mean(hist["channel_entropy_z"][ss])),
            },
            "timeseries": {
                "firing_rates": hist["firing_rates"],
                "mean_sigmas": hist["mean_sigmas"],
                "mean_w": hist["mean_w"],
                "mean_xyz": hist["mean_xyz"],
            },
        }
    
    return results


def single_threshold_channel_test():
    """
    DEEPER TEST: At a SINGLE threshold, do the four channels (W, X, Y, Z)
    show different behavioral signatures?
    
    If yes → the geometry of S³ itself separates behaviors
    If no → channels are symmetric and geometry doesn't add structure
    """
    tau = 0.5  # middle threshold
    print(f"  Running single-threshold channel test (τ={tau})...")
    hist = run_quaternion_sim(tau, n_cells=100, n_steps=300, seed=42)
    
    ss = slice(-100, None)
    
    return {
        "tau": tau,
        "w_entropy": float(np.mean(hist["channel_entropy_w"][ss])),
        "x_entropy": float(np.mean(hist["channel_entropy_x"][ss])),
        "y_entropy": float(np.mean(hist["channel_entropy_y"][ss])),
        "z_entropy": float(np.mean(hist["channel_entropy_z"][ss])),
        "w_std": float(np.mean(hist["w_std"][ss])),
        "x_std": float(np.mean(hist["x_std"][ss])),
        "y_std": float(np.mean(hist["y_std"][ss])),
        "z_std": float(np.mean(hist["z_std"][ss])),
        "w_timeseries": hist["channel_entropy_w"],
        "x_timeseries": hist["channel_entropy_x"],
        "y_timeseries": hist["channel_entropy_y"],
        "z_timeseries": hist["channel_entropy_z"],
    }


def convergence_test():
    """
    Does the S³ model also show high-energy convergence?
    Increase perturbation magnitude (= energy) and check.
    """
    results = {}
    
    perturbation_levels = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0]
    
    thresholds = {"gravity": 0.05, "EM": 0.3, "weak": 0.7, "strong": 1.2}
    
    for p_level in perturbation_levels:
        rates = {}
        for name, tau in thresholds.items():
            hist = run_quaternion_sim(tau, n_cells=60, n_steps=100, 
                                     seed=42, perturbation=p_level)
            ss = slice(-40, None)
            rates[name] = float(np.mean(hist["firing_rates"][ss]))
        
        rate_values = list(rates.values())
        cv = np.std(rate_values) / np.mean(rate_values) if np.mean(rate_values) > 0 else float('inf')
        
        results[f"perturbation_{p_level}"] = {
            "energy": p_level,
            "rates": rates,
            "cv": float(cv),
        }
    
    return results


def main():
    print("=" * 60)
    print("EXPERIMENT 6: Quaternion States on S³ (Closure-SDK)")
    print("=" * 60)
    print(f"  Using {'Closure-SDK' if HAS_CLOSURE else 'pure numpy'} for quaternion ops")
    print()
    
    # Test 1: Force hierarchy on S³
    print("TEST 1: Force hierarchy on S³")
    print("-" * 40)
    channel_results = channel_separation_test()
    
    print(f"\n  {'Force':<10} {'τ':>6} {'Rate':>8} {'σ_mean':>8} {'W_mean':>8} {'XYZ_norm':>8} {'W_ent':>8} {'XYZ_ent':>8}")
    print(f"  {'-'*10} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    
    for name, data in channel_results.items():
        xyz_ent = np.mean([
            data["xyz_channels"]["entropy_x"],
            data["xyz_channels"]["entropy_y"],
            data["xyz_channels"]["entropy_z"],
        ])
        print(f"  {name:<10} {data['tau']:>6.2f} {data['firing_rate']:>8.4f} "
              f"{data['mean_sigma']:>8.4f} {data['w_channel']['mean']:>8.4f} "
              f"{data['xyz_channels']['mean_norm']:>8.4f} "
              f"{data['w_channel']['entropy']:>8.4f} {xyz_ent:>8.4f}")
    
    # Check hierarchy preservation
    rates = [(name, data["firing_rate"]) for name, data in channel_results.items()]
    rates.sort(key=lambda x: -x[1])
    actual_order = [r[0] for r in rates]
    expected_order = ["gravity", "EM", "weak", "strong"]
    
    if actual_order == expected_order:
        print(f"\n  ✓ Force hierarchy preserved on S³: {' > '.join(actual_order)}")
    else:
        print(f"\n  ~ Hierarchy on S³: {' > '.join(actual_order)} (expected {' > '.join(expected_order)})")
    
    # Test 2: Channel separation at single threshold
    print(f"\n\nTEST 2: Channel separation at single threshold")
    print("-" * 40)
    single = single_threshold_channel_test()
    
    print(f"\n  At τ={single['tau']}:")
    print(f"    W channel:  entropy={single['w_entropy']:.4f}  std={single['w_std']:.4f}")
    print(f"    X channel:  entropy={single['x_entropy']:.4f}  std={single['x_std']:.4f}")
    print(f"    Y channel:  entropy={single['y_entropy']:.4f}  std={single['y_std']:.4f}")
    print(f"    Z channel:  entropy={single['z_entropy']:.4f}  std={single['z_std']:.4f}")
    
    # Are channels different?
    entropies = [single['w_entropy'], single['x_entropy'], single['y_entropy'], single['z_entropy']]
    stds = [single['w_std'], single['x_std'], single['y_std'], single['z_std']]
    
    ent_cv = np.std(entropies) / np.mean(entropies) if np.mean(entropies) > 0 else 0
    std_cv = np.std(stds) / np.mean(stds) if np.mean(stds) > 0 else 0
    
    print(f"\n    Entropy CV across channels: {ent_cv:.4f}")
    print(f"    StdDev CV across channels:  {std_cv:.4f}")
    
    if ent_cv > 0.1:
        print(f"    ✓ CHANNELS SEPARATE — W behaves differently from XYZ")
        w_vs_xyz = abs(single['w_entropy'] - np.mean([single['x_entropy'], single['y_entropy'], single['z_entropy']]))
        xyz_spread = np.std([single['x_entropy'], single['y_entropy'], single['z_entropy']])
        print(f"      W vs mean(XYZ) gap: {w_vs_xyz:.4f}")
        print(f"      XYZ internal spread: {xyz_spread:.4f}")
        
        if xyz_spread < w_vs_xyz * 0.5:
            print(f"      ★ XYZ channels similar to each other but different from W")
            print(f"        This mirrors SU(2): 1 scalar + 3 vector components")
    elif ent_cv > 0.03:
        print(f"    ~ Weak channel separation (CV = {ent_cv:.4f})")
    else:
        print(f"    ✗ Channels symmetric — geometry doesn't separate them")
    
    # Test 3: High-energy convergence on S³
    print(f"\n\nTEST 3: High-energy convergence on S³")
    print("-" * 40)
    conv = convergence_test()
    
    print(f"\n  {'Energy':>8} {'Gravity':>10} {'EM':>10} {'Weak':>10} {'Strong':>10} {'CV':>8}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    
    for key, data in sorted(conv.items(), key=lambda x: x[1]["energy"]):
        r = data["rates"]
        print(f"  {data['energy']:>8.2f} {r['gravity']:>10.4f} {r['EM']:>10.4f} "
              f"{r['weak']:>10.4f} {r['strong']:>10.4f} {data['cv']:>8.4f}")
    
    cvs = [data["cv"] for data in conv.values()]
    energies = [data["energy"] for data in conv.values()]
    
    if cvs[-1] < cvs[0] * 0.5:
        print(f"\n  ✓ CONVERGENCE: CV drops from {cvs[0]:.4f} to {cvs[-1]:.4f}")
        print(f"    Forces become more similar at high energy on S³")
    else:
        print(f"\n  ✗ No convergence: CV stays at {cvs[-1]:.4f}")
    
    # Save results
    output = {
        "using_closure_sdk": HAS_CLOSURE,
        "channel_test": {},
        "single_threshold": {k: v for k, v in single.items() 
                            if not k.endswith('_timeseries')},
        "convergence": conv,
    }
    
    for name, data in channel_results.items():
        output["channel_test"][name] = {k: v for k, v in data.items() if k != "timeseries"}
    
    output_path = os.path.join(os.path.dirname(__file__), "..", "results", "exp6_quaternion.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Data saved to {output_path}")
    
    return channel_results, single, conv


if __name__ == "__main__":
    main()
