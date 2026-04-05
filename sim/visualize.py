#!/usr/bin/env python3
"""
Visualization for the Unified Force Simulation.

Generates four plots:
1. Force behavioral profiles (firing rate, range, strength)
2. Energy convergence — CV of firing rates vs background energy
3. Time series — firing rates over simulation steps
4. Spatial correlation over time

Reads from results/simulation_results.json (run unified_force.py first).
"""

import json
import sys
import os
import numpy as np

# Check for matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
except ImportError:
    print("matplotlib not found. Install with: pip install matplotlib")
    sys.exit(1)

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────

results_path = os.path.join(os.path.dirname(__file__), "..", "results", "simulation_results.json")
if not os.path.exists(results_path):
    print(f"Results file not found at {results_path}")
    print("Run unified_force.py first to generate data.")
    sys.exit(1)

with open(results_path) as f:
    data = json.load(f)

FORCE_COLORS = {
    "gravity": "#4466cc",
    "EM": "#44bb44", 
    "weak": "#dd8800",
    "strong": "#cc3333",
}

FORCE_ORDER = ["gravity", "EM", "weak", "strong"]
output_dir = os.path.join(os.path.dirname(__file__), "..", "results")

# ─────────────────────────────────────────────
# Plot 1: Force Behavioral Profiles
# ─────────────────────────────────────────────

def plot_force_profiles():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Force Behavioral Profiles\n(One update function, different activation thresholds)", 
                 fontsize=13, fontweight='bold')
    
    cons = data["conservation"]
    
    names = FORCE_ORDER
    taus = [cons[n]["tau"] for n in names]
    ranges = [cons[n]["avg_range"] for n in names]
    strengths = [cons[n]["avg_strength"] for n in names]
    rates = [cons[n]["avg_firing_rate"] for n in names]
    colors = [FORCE_COLORS[n] for n in names]
    
    # Range vs threshold
    ax = axes[0]
    ax.bar(names, ranges, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.set_ylabel("Effective Range (cells)")
    ax.set_title("Range")
    ax.set_xlabel("τ →")
    for i, (n, v) in enumerate(zip(names, ranges)):
        ax.text(i, v + 0.5, f"τ={taus[i]}", ha='center', fontsize=9, color='gray')
    
    # Firing rate vs threshold
    ax = axes[1]
    ax.bar(names, rates, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.set_ylabel("Firing Rate (fraction of cells)")
    ax.set_title("Firing Rate")
    ax.set_xlabel("τ →")
    
    # Strength vs threshold
    ax = axes[2]
    ax.bar(names, strengths, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.set_ylabel("Avg Energy Transfer per Event")
    ax.set_title("Per-Event Strength")
    ax.set_xlabel("τ →")
    
    plt.tight_layout()
    path = os.path.join(output_dir, "force_profiles.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────
# Plot 2: Energy Convergence
# ─────────────────────────────────────────────

def plot_convergence():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("High-Energy Convergence\n(Do forces become indistinguishable at high energy?)", 
                 fontsize=13, fontweight='bold')
    
    scan = data["energy_scan"]
    
    # Firing rates vs energy for each force
    ax = axes[0]
    for name in FORCE_ORDER:
        energies = scan[name]["energies"]
        rates = scan[name]["firing_rates"]
        ax.plot(energies, rates, 'o-', color=FORCE_COLORS[name], 
                label=f"{name} (τ={data['forces'][name]['tau']})",
                markersize=4, linewidth=1.5)
    ax.set_xscale('log')
    ax.set_xlabel("Background Energy")
    ax.set_ylabel("Firing Rate")
    ax.set_title("Firing Rates Converge")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # CV (coefficient of variation) across forces vs energy
    ax = axes[1]
    n_energies = len(scan["gravity"]["energies"])
    energies = scan["gravity"]["energies"]
    cvs = []
    for i in range(n_energies):
        rates = [scan[name]["firing_rates"][i] for name in FORCE_ORDER]
        mean_r = np.mean(rates)
        cv = np.std(rates) / mean_r if mean_r > 0 else 0
        cvs.append(cv)
    
    ax.plot(energies, cvs, 'ko-', linewidth=2, markersize=5)
    ax.set_xscale('log')
    ax.set_xlabel("Background Energy")
    ax.set_ylabel("Coefficient of Variation")
    ax.set_title("Force Divergence → 0 at High Energy")
    ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label="CV = 0.1 (near-identical)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    # Annotate
    ax.annotate("Forces\nindistinguishable", xy=(energies[-1], cvs[-1]),
                xytext=(energies[-3], cvs[0] * 0.6),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=9, color='green', fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(output_dir, "convergence.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────
# Plot 3: Time Series
# ─────────────────────────────────────────────

def plot_timeseries():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Simulation Time Series\n(Same update function, different thresholds)", 
                 fontsize=13, fontweight='bold')
    
    metrics = [
        ("firing_rates", "Firing Rate", axes[0, 0]),
        ("energy_means", "Mean Cell Energy", axes[0, 1]),
        ("avg_transfers", "Avg Transfer per Event", axes[1, 0]),
        ("spatial_correlation", "Nearest-Neighbor Correlation", axes[1, 1]),
    ]
    
    for key, title, ax in metrics:
        for name in FORCE_ORDER:
            ts_key = f"timeseries_{name}"
            if ts_key in data and key in data[ts_key]:
                values = data[ts_key][key]
                # Subsample for readability
                step = max(1, len(values) // 200)
                x = list(range(0, len(values), step))
                y = [values[i] for i in x]
                ax.plot(x, y, color=FORCE_COLORS[name], label=name, 
                        linewidth=1, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    path = os.path.join(output_dir, "timeseries.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────
# Plot 4: The Hierarchy — one summary chart
# ─────────────────────────────────────────────

def plot_hierarchy():
    """
    Range vs Firing Rate scatter, sized by per-event strength.
    Shows the force hierarchy emerging from a single parameter.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cons = data["conservation"]
    
    for name in FORCE_ORDER:
        d = cons[name]
        size = d["avg_strength"] * 8000  # scale for visibility
        ax.scatter(d["avg_firing_rate"], d["avg_range"], 
                   s=size, c=FORCE_COLORS[name], alpha=0.7,
                   edgecolors='black', linewidth=0.5, zorder=5)
        ax.annotate(f"{name}\nτ={d['tau']}", 
                    xy=(d["avg_firing_rate"], d["avg_range"]),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    color=FORCE_COLORS[name])
    
    ax.set_xlabel("Firing Rate (how often it triggers)", fontsize=11)
    ax.set_ylabel("Effective Range (cells)", fontsize=11)
    ax.set_title("The Force Hierarchy\nOne update function, one parameter (τ)\nBubble size = per-event strength",
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2)
    
    # Add annotation about convergence
    ax.text(0.95, 0.05, 
            "At high energy, all four\nconverge to the same point →\n(CV drops from 1.37 to 0.02)",
            transform=ax.transAxes, fontsize=9, color='gray',
            ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    path = os.path.join(output_dir, "hierarchy.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating visualizations...")
    plot_force_profiles()
    plot_convergence()
    plot_timeseries()
    plot_hierarchy()
    print("\nDone. All plots saved to results/")
