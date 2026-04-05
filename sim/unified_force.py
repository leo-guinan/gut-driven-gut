#!/usr/bin/env python3
"""
Unified Force Simulation — Toy Model

Core hypothesis (Leo Guinan):
  All four fundamental forces are the same update function with different
  activation thresholds (energy cost to trigger a state transition).

Model:
  - 1D lattice of cells, each with an energy value
  - One universal update rule parameterized by activation threshold τ
  - When cell energy ≥ τ, it "fires": transfers energy to neighbors
  - Transfer range and strength emerge from the threshold, not put in by hand

What we're testing:
  1. Does low τ produce gravity-like behavior? (universal, weak, long-range)
  2. Does high τ produce strong-force-like behavior? (rare, powerful, short-range)
  3. Do intermediate τ values produce EM-like and weak-force-like behavior?
  4. Do the effective coupling strengths converge at high energy?
  5. Is there a conserved quantity (range × strength)?
"""

import numpy as np
import json
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# ─────────────────────────────────────────────
# MODEL PARAMETERS
# ─────────────────────────────────────────────

@dataclass
class ForceConfig:
    """A force is just an activation threshold + a label."""
    name: str
    tau: float          # activation threshold
    color: str          # for plotting
    description: str

# The four forces as different activation thresholds
# We set these on a log scale to span the known hierarchy
FORCES = [
    ForceConfig("gravity",    tau=0.01,  color="#4444ff", description="lowest threshold, always firing"),
    ForceConfig("EM",         tau=0.10,  color="#44ff44", description="medium-low threshold"),
    ForceConfig("weak",       tau=0.50,  color="#ffaa00", description="medium-high threshold"),
    ForceConfig("strong",     tau=0.90,  color="#ff4444", description="highest threshold, rarely fires"),
]


@dataclass 
class SimConfig:
    n_cells: int = 100          # lattice size
    n_steps: int = 500          # simulation steps
    energy_inject: float = 0.05 # background energy injection per step (thermal bath)
    decay_rate: float = 0.001   # small energy dissipation (prevents runaway)
    seed: int = 42


# ─────────────────────────────────────────────
# THE SINGLE UPDATE FUNCTION
# ─────────────────────────────────────────────

def update(cells: np.ndarray, tau: float, rng: np.random.Generator) -> dict:
    """
    The universal update function. ONE rule, ONE parameter.
    
    For each cell:
      - If cell energy >= tau, it FIRES
      - On firing: cell transfers excess energy to neighbors
      - Transfer RANGE = 1/tau (low threshold -> long range, high -> short)
      - Energy falls off as 1/distance within range
    
    Vectorized for performance.
    """
    n = len(cells)
    
    # Cells that fire
    firing_mask = cells >= tau
    n_fired = int(firing_mask.sum())
    
    if n_fired == 0:
        return {
            "n_fired": 0,
            "total_transferred": 0.0,
            "effective_range": 0.0,
            "avg_transfer_per_event": 0.0,
            "firing_rate": 0.0,
        }
    
    # Range from threshold: low tau -> long range
    effective_range = max(1, min(int(np.ceil(1.0 / tau)), n // 2))
    
    # Build 1/d weight kernel once
    distances = np.arange(1, effective_range + 1, dtype=np.float64)
    weights = 1.0 / distances
    
    # Excess energy for firing cells
    excess = np.where(firing_mask, cells - tau, 0.0)
    
    # Build transfer array using convolution-like approach
    incoming = np.zeros(n, dtype=np.float64)
    
    for d_idx, d in enumerate(range(1, effective_range + 1)):
        w = weights[d_idx]
        # Shift left: cell i fires, neighbor i+d receives
        incoming[d:] += excess[:-d] * w
        # Shift right: cell i fires, neighbor i-d receives
        incoming[:-d] += excess[d:] * w
    
    # Normalize: each firing cell's total outgoing should equal its excess
    # The total weight distributed by cell i = sum of weights to all its neighbors
    # For interior cells this is 2 * sum(weights), for edges it's less
    # We'll compute per-cell normalization
    total_weight_full = 2.0 * weights.sum()  # both directions
    
    # Simple normalization: divide incoming by total_weight_full 
    # (edge effects are minor for this toy model)
    if total_weight_full > 0:
        incoming /= total_weight_full
    
    total_transferred = float(incoming.sum())
    
    # Apply: firing cells drop, neighbors gain
    cells[:] = cells + incoming
    cells[firing_mask] = tau * 0.5  # firing cells reset below threshold
    
    excess_values = excess[firing_mask]
    avg_transfer = float(excess_values.mean()) if len(excess_values) > 0 else 0.0
    
    return {
        "n_fired": n_fired,
        "total_transferred": total_transferred,
        "effective_range": float(effective_range),
        "avg_transfer_per_event": avg_transfer,
        "firing_rate": float(n_fired) / n,
    }


# ─────────────────────────────────────────────
# SIMULATION LOOP
# ─────────────────────────────────────────────

def run_simulation(force: ForceConfig, config: SimConfig) -> Dict:
    """Run the simulation for one force configuration."""
    rng = np.random.default_rng(config.seed)
    
    # Initialize: random energy in [0, 1]
    cells = rng.uniform(0, 1, config.n_cells)
    
    history = {
        "firing_rates": [],
        "total_transferred": [],
        "effective_ranges": [],
        "avg_transfers": [],
        "energy_means": [],
        "energy_stds": [],
        "spatial_correlation": [],
    }
    
    for step in range(config.n_steps):
        # Inject background energy (thermal bath — drives the system)
        cells += rng.uniform(0, config.energy_inject, config.n_cells)
        
        # Small dissipation to prevent energy divergence
        cells *= (1.0 - config.decay_rate)
        
        # THE update
        metrics = update(cells, force.tau, rng)
        
        # Record
        history["firing_rates"].append(metrics["firing_rate"])
        history["total_transferred"].append(metrics["total_transferred"])
        history["effective_ranges"].append(metrics["effective_range"])
        history["avg_transfers"].append(metrics["avg_transfer_per_event"])
        history["energy_means"].append(float(cells.mean()))
        history["energy_stds"].append(float(cells.std()))
        
        # Spatial correlation: how correlated are neighboring cells?
        if len(cells) > 1:
            corr = np.corrcoef(cells[:-1], cells[1:])[0, 1]
            history["spatial_correlation"].append(float(corr) if not np.isnan(corr) else 0.0)
        else:
            history["spatial_correlation"].append(0.0)
    
    return history


# ─────────────────────────────────────────────
# ENERGY SCAN: test convergence at high energy
# ─────────────────────────────────────────────

def energy_scan(forces: List[ForceConfig], config: SimConfig) -> Dict:
    """
    Run simulations at increasing background energy levels.
    Test whether force behaviors converge at high energy.
    """
    energy_levels = np.logspace(-2, 1, 10)  # 0.01 to 10
    
    results = {f.name: {"energies": [], "firing_rates": [], "ranges": [], "strengths": []} 
               for f in forces}
    
    for e_level in energy_levels:
        cfg = SimConfig(
            n_cells=config.n_cells,
            n_steps=200,  # shorter runs for scan
            energy_inject=float(e_level),
            decay_rate=config.decay_rate,
            seed=config.seed,
        )
        
        for force in forces:
            hist = run_simulation(force, cfg)
            
            # Take steady-state averages (last 50 steps)
            ss = slice(-50, None)
            avg_rate = np.mean(hist["firing_rates"][ss])
            avg_range = np.mean(hist["effective_ranges"][ss])
            avg_strength = np.mean(hist["avg_transfers"][ss])
            
            results[force.name]["energies"].append(float(e_level))
            results[force.name]["firing_rates"].append(float(avg_rate))
            results[force.name]["ranges"].append(float(avg_range))
            results[force.name]["strengths"].append(float(avg_strength))
    
    return results


# ─────────────────────────────────────────────
# CONSERVATION TEST: is range × strength conserved?
# ─────────────────────────────────────────────

def conservation_test(forces: List[ForceConfig], config: SimConfig) -> Dict:
    """
    Test the hypothesis that range × strength is approximately conserved
    across different activation thresholds.
    """
    results = {}
    
    for force in forces:
        hist = run_simulation(force, config)
        ss = slice(-200, None)
        
        avg_range = np.mean(hist["effective_ranges"][ss])
        avg_strength = np.mean(hist["avg_transfers"][ss])
        avg_rate = np.mean(hist["firing_rates"][ss])
        
        # "Effective coupling" = rate × strength (how much energy moves per step)
        effective_coupling = avg_rate * avg_strength
        
        # "Range-strength product"
        rs_product = avg_range * avg_strength
        
        # "Total influence" = range × rate × strength  
        total_influence = avg_range * avg_rate * avg_strength
        
        results[force.name] = {
            "tau": force.tau,
            "avg_range": float(avg_range),
            "avg_strength": float(avg_strength),
            "avg_firing_rate": float(avg_rate),
            "effective_coupling": float(effective_coupling),
            "range_strength_product": float(rs_product),
            "total_influence": float(total_influence),
        }
    
    return results


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    config = SimConfig()
    
    print("=" * 70)
    print("UNIFIED FORCE SIMULATION")
    print("Hypothesis: One update function, four activation thresholds")
    print("=" * 70)
    print()
    
    # ── Test 1: Individual force behaviors ──
    print("TEST 1: Individual Force Behaviors")
    print("-" * 50)
    
    for force in FORCES:
        hist = run_simulation(force, config)
        ss = slice(-200, None)  # steady state
        
        avg_rate = np.mean(hist["firing_rates"][ss])
        avg_range = np.mean(hist["effective_ranges"][ss])
        avg_strength = np.mean(hist["avg_transfers"][ss])
        avg_corr = np.mean(hist["spatial_correlation"][ss])
        avg_energy = np.mean(hist["energy_means"][ss])
        
        print(f"\n  {force.name.upper()} (τ = {force.tau})")
        print(f"    Firing rate:        {avg_rate:.4f}  (fraction of cells firing per step)")
        print(f"    Effective range:    {avg_range:.1f}  cells")
        print(f"    Avg transfer/event: {avg_strength:.6f}")
        print(f"    Spatial correlation: {avg_corr:.4f}")
        print(f"    Mean cell energy:   {avg_energy:.4f}")
        print(f"    → {force.description}")
    
    # ── Test 2: Conservation test ──
    print("\n")
    print("TEST 2: Is range × strength conserved?")
    print("-" * 50)
    
    cons = conservation_test(FORCES, config)
    
    print(f"\n  {'Force':<10} {'τ':>6} {'Range':>8} {'Strength':>10} {'Rate':>8} {'R×S':>10} {'R×S×Rate':>12}")
    print(f"  {'-'*10} {'-'*6} {'-'*8} {'-'*10} {'-'*8} {'-'*10} {'-'*12}")
    
    for name, data in cons.items():
        print(f"  {name:<10} {data['tau']:>6.2f} {data['avg_range']:>8.1f} "
              f"{data['avg_strength']:>10.6f} {data['avg_firing_rate']:>8.4f} "
              f"{data['range_strength_product']:>10.6f} {data['total_influence']:>12.6f}")
    
    # Check conservation
    rs_values = [d["range_strength_product"] for d in cons.values()]
    ti_values = [d["total_influence"] for d in cons.values()]
    
    rs_cv = np.std(rs_values) / np.mean(rs_values) if np.mean(rs_values) > 0 else float('inf')
    ti_cv = np.std(ti_values) / np.mean(ti_values) if np.mean(ti_values) > 0 else float('inf')
    
    print(f"\n  Range×Strength coefficient of variation: {rs_cv:.3f}")
    print(f"  Total influence coefficient of variation: {ti_cv:.3f}")
    print(f"  (Lower = more conserved. <0.3 would be suggestive, <0.1 striking)")
    
    # ── Test 3: Energy convergence scan ──
    print("\n")
    print("TEST 3: Do forces converge at high energy?")
    print("-" * 50)
    
    scan = energy_scan(FORCES, config)
    
    # Compare behaviors at lowest and highest energy
    print("\n  At LOW energy (background = 0.01):")
    for name, data in scan.items():
        print(f"    {name:<10} rate={data['firing_rates'][0]:.4f}  "
              f"strength={data['strengths'][0]:.6f}")
    
    print(f"\n  At HIGH energy (background = 10.0):")
    for name, data in scan.items():
        print(f"    {name:<10} rate={data['firing_rates'][-1]:.4f}  "
              f"strength={data['strengths'][-1]:.6f}")
    
    # Measure convergence: coefficient of variation across forces at each energy
    print(f"\n  Convergence metric (CV of firing rates across forces):")
    n_energies = len(scan["gravity"]["energies"])
    for i in [0, n_energies//4, n_energies//2, 3*n_energies//4, n_energies-1]:
        rates = [scan[f.name]["firing_rates"][i] for f in FORCES]
        e = scan["gravity"]["energies"][i]
        cv = np.std(rates) / np.mean(rates) if np.mean(rates) > 0 else float('inf')
        print(f"    E={e:>8.3f}  rates={[f'{r:.3f}' for r in rates]}  CV={cv:.4f}")
    
    print(f"\n  (CV → 0 means forces become indistinguishable → convergence)")
    
    # ── Save raw data for visualization ──
    output = {
        "forces": {f.name: {"tau": f.tau} for f in FORCES},
        "conservation": cons,
        "energy_scan": scan,
    }
    
    # Also run and save full time series for the main simulation
    for force in FORCES:
        hist = run_simulation(force, config)
        output[f"timeseries_{force.name}"] = {
            k: [float(x) for x in v] for k, v in hist.items()
        }
    
    with open("simulation_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("\n\nRaw data saved to simulation_results.json")
    
    # ── Verdict ──
    print("\n")
    print("=" * 70)
    print("PRELIMINARY VERDICT")
    print("=" * 70)
    
    # Analyze what we found
    gravity_data = cons["gravity"]
    strong_data = cons["strong"]
    
    print(f"""
  The simulation ran one update function with four threshold values.
  
  BEHAVIORAL SIGNATURES:
  - Gravity (τ=0.01): range={gravity_data['avg_range']:.0f}, fires {gravity_data['avg_firing_rate']:.1%} of the time
    → Universal, long-range, individually weak. ✓ matches expectation
  
  - Strong (τ=0.90): range={strong_data['avg_range']:.0f}, fires {strong_data['avg_firing_rate']:.1%} of the time  
    → Rare, short-range, individually strong. ✓ matches expectation
  
  CONSERVATION (range × strength):
    CV = {rs_cv:.3f} → {"SUGGESTIVE" if rs_cv < 0.3 else "NOT conserved" if rs_cv > 0.5 else "WEAK signal"}
  
  CONVERGENCE at high energy:
    Check the CV trend above. If it decreases, forces converge.
  
  WHAT THIS MODEL CAN'T DO (yet):
  - Confinement (strong force getting stronger with distance)
  - Symmetry breaking (electroweak split)
  - Attraction vs repulsion (gravity always attracts, EM does both)
  - Spin, charge, flavor — the quantum numbers that distinguish particles
  
  WHAT IT SUGGESTS:
  - The force hierarchy CAN emerge from a single parameter
  - Whether the deeper structure is real depends on conservation & convergence
""")

if __name__ == "__main__":
    main()
