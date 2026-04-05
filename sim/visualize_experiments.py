#!/usr/bin/env python3
"""
Visualization for all five experiments.
"""

import json
import sys
import os
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
except ImportError:
    print("pip install matplotlib")
    sys.exit(1)

results_dir = os.path.join(os.path.dirname(__file__), "..", "results")

FORCE_COLORS = {
    "gravity": "#4466cc", "gravity_unsigned": "#4466cc",
    "EM": "#44bb44", "EM_signed": "#44bb44", "EM_unsigned": "#88cc88",
    "weak": "#dd8800",
    "strong": "#cc3333", "strong_signed": "#cc3333",
}


def plot_exp1():
    """Experiment 1: Sign — Attraction vs Repulsion"""
    with open(os.path.join(results_dir, "exp1_sign.json")) as f:
        data = json.load(f)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Experiment 1: Sign (Attraction vs Repulsion)", fontsize=13, fontweight='bold')
    
    # Panel 1: Attraction vs Repulsion balance
    ax = axes[0]
    names = ["gravity_unsigned", "EM_signed", "EM_unsigned", "strong_signed"]
    labels = ["Gravity\n(unsigned)", "EM\n(signed)", "EM\n(unsigned)", "Strong\n(signed)"]
    attract = [data[n]["net_attraction"] for n in names]
    repulse = [data[n]["net_repulsion"] for n in names]
    
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width/2, attract, width, label="Attraction", color="#4488cc", alpha=0.8)
    ax.bar(x + width/2, repulse, width, label="Repulsion", color="#cc4444", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Net Force per Step")
    ax.set_title("Attraction vs Repulsion")
    ax.legend(fontsize=8)
    
    # Panel 2: Scale-dependent variance
    ax = axes[1]
    scales = [2, 5, 10, 20, 50]
    for name in names:
        sa = data[name]["scale_analysis"]
        variances = [sa[f"variance_at_scale_{s}"] for s in scales]
        color = FORCE_COLORS.get(name, "#888888")
        ax.plot(scales, variances, 'o-', color=color, label=name, markersize=4)
    ax.set_xlabel("Averaging Scale (cells)")
    ax.set_ylabel("Variance")
    ax.set_title("Energy Variance vs Scale")
    ax.legend(fontsize=7)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.2)
    
    # Panel 3: Key finding — signed EM has balanced attract/repulse
    ax = axes[2]
    ratios = []
    for name in names:
        a = data[name]["net_attraction"]
        r = data[name]["net_repulsion"]
        total = a + r if a + r > 0 else 1
        ratios.append(a / total)
    
    colors = [FORCE_COLORS.get(n, "#888") for n in names]
    bars = ax.bar(labels, ratios, color=colors, alpha=0.8)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label="Perfect balance")
    ax.set_ylabel("Attraction Fraction")
    ax.set_title("Force Balance\n(0.5 = equal attract/repel)")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    path = os.path.join(results_dir, "exp1_sign.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


def plot_exp2():
    """Experiment 2: Confinement"""
    with open(os.path.join(results_dir, "exp2_confinement.json")) as f:
        data = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Experiment 2: Dynamic Threshold — Confinement Test", fontsize=13, fontweight='bold')
    
    forces = ["gravity", "EM", "weak", "strong"]
    
    for idx, fname in enumerate(forces):
        ax = axes[idx // 2][idx % 2]
        fdata = data[fname]
        
        fixed_conf = fdata["fixed"]["confinement_timeseries"]
        dyn_conf = fdata["dynamic"]["confinement_timeseries"]
        
        ax.plot(fixed_conf, color='gray', alpha=0.7, label="Fixed τ", linewidth=1)
        ax.plot(dyn_conf, color=FORCE_COLORS.get(fname, "#888"), 
                label="Dynamic τ", linewidth=1.5)
        ax.set_title(f"{fname} (τ={fdata['tau']})")
        ax.set_xlabel("Step")
        ax.set_ylabel("Confinement Ratio")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.set_ylim(0, 0.7)
    
    plt.tight_layout()
    path = os.path.join(results_dir, "exp2_confinement.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


def plot_exp3():
    """Experiment 3: 2D Lattice"""
    with open(os.path.join(results_dir, "exp3_2d.json")) as f:
        data = json.load(f)
    
    # Load snapshots
    snap_path = os.path.join(results_dir, "exp3_2d_full.json")
    if os.path.exists(snap_path):
        with open(snap_path) as f:
            snapshots = json.load(f)
    else:
        snapshots = None
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Experiment 3: 2D Lattice — Energy Distribution\n(Same update function in two dimensions)", 
                 fontsize=13, fontweight='bold')
    
    forces = ["gravity", "EM", "weak", "strong"]
    
    for idx, fname in enumerate(forces):
        ax = axes[idx // 2][idx % 2]
        
        if snapshots and fname in snapshots:
            grid = np.array(snapshots[fname])
            im = ax.imshow(grid, cmap='hot', aspect='equal', 
                          interpolation='nearest')
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        fdata = data[fname]
        ax.set_title(f"{fname} (τ={fdata['tau']})\n"
                    f"rate={fdata['avg_firing_rate']:.3f}, "
                    f"range={fdata['effective_range']:.0f}, "
                    f"clusters={fdata['n_clusters']}")
    
    plt.tight_layout()
    path = os.path.join(results_dir, "exp3_2d.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


def plot_exp4():
    """Experiment 4: Conservation"""
    with open(os.path.join(results_dir, "exp4_conservation.json")) as f:
        data = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Experiment 4: The Conserved Quantity Search", fontsize=13, fontweight='bold')
    
    # Panel 1: Top conserved quantities
    ax = axes[0]
    top = data["top_50_conserved"][:15]
    formulas = [r["formula"] for r in top]
    cvs = [r["cv"] for r in top]
    colors = ['gold' if cv < 0.1 else '#44bb44' if cv < 0.3 else '#888888' for cv in cvs]
    
    y_pos = np.arange(len(formulas))
    ax.barh(y_pos, cvs, color=colors, alpha=0.8, edgecolor='white')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(formulas, fontsize=8)
    ax.set_xlabel("Coefficient of Variation (lower = more conserved)")
    ax.set_title("Top 15 Most Conserved Quantities")
    ax.axvline(x=0.1, color='gold', linestyle='--', alpha=0.7, label="Striking (<0.1)")
    ax.axvline(x=0.3, color='green', linestyle='--', alpha=0.5, label="Suggestive (<0.3)")
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    
    # Panel 2: The winner — entropy across forces
    ax = axes[1]
    obs = data["observables"]
    forces = ["gravity", "EM", "weak", "strong"]
    entropies = [obs[f]["entropy"] for f in forces]
    taus = [obs[f]["tau"] for f in forces]
    colors_f = [FORCE_COLORS[f] for f in forces]
    
    ax.bar(forces, entropies, color=colors_f, alpha=0.8, edgecolor='white')
    ax.set_ylabel("Entropy of Energy Distribution")
    ax.set_title(f"ENTROPY — The Conserved Quantity\n(CV = {data['top_50_conserved'][0]['cv']:.4f})")
    
    # Show how close they are
    mean_ent = np.mean(entropies)
    ax.axhline(y=mean_ent, color='black', linestyle='--', alpha=0.5)
    ax.set_ylim(min(entropies) * 0.95, max(entropies) * 1.02)
    
    for i, (f, e) in enumerate(zip(forces, entropies)):
        ax.text(i, e + 0.01, f"τ={taus[i]}", ha='center', fontsize=9, color='gray')
    
    plt.tight_layout()
    path = os.path.join(results_dir, "exp4_conservation.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


def plot_exp5():
    """Experiment 5: Coupling Ratios"""
    with open(os.path.join(results_dir, "exp5_coupling.json")) as f:
        data = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Experiment 5: Coupling Constant Ratios", fontsize=13, fontweight='bold')
    
    scan = data["tau_scan"]
    taus = [r["tau"] for r in scan]
    
    # Panel 1: All coupling definitions vs tau
    ax = axes[0]
    for key, label, color in [
        ("coupling_v1", "Rate × Strength", "#4466cc"),
        ("coupling_v2", "Rate × Strength × Range", "#44bb44"),
        ("coupling_v3", "Transfer Density", "#dd8800"),
    ]:
        values = [r[key] for r in scan]
        # Normalize to max = 1
        max_v = max(values)
        if max_v > 0:
            normed = [v / max_v for v in values]
        else:
            normed = values
        ax.plot(taus, normed, 'o-', color=color, label=label, markersize=3, linewidth=1.5)
    
    ax.set_xlabel("Activation Threshold (τ)")
    ax.set_ylabel("Normalized Coupling Strength")
    ax.set_title("Effective Coupling vs Threshold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.set_yscale('log')
    
    # Mark real force positions (using coupling_v2 mapping)
    if "coupling_v2" in data["mappings"]:
        mapping = data["mappings"]["coupling_v2"]
        for fname in ["strong", "EM", "weak"]:
            if fname in mapping:
                m = mapping[fname]
                ax.axvline(x=m["best_tau"], color=FORCE_COLORS.get(fname, "#888"),
                          linestyle=':', alpha=0.5)
                ax.text(m["best_tau"], 0.5, fname, rotation=90, fontsize=8,
                       color=FORCE_COLORS.get(fname, "#888"), va='center')
    
    # Panel 2: Dynamic range comparison
    ax = axes[1]
    
    real = data["real_couplings"]
    real_log = {k: np.log10(v) if v > 0 else -40 for k, v in real.items()}
    
    # Our model's range
    for key, label, color in [
        ("coupling_v1", "v1: Rate×Str", "#4466cc"),
        ("coupling_v2", "v2: Rate×Str×Rng", "#44bb44"),
        ("coupling_v3", "v3: Transfer", "#dd8800"),
    ]:
        values = [r[key] for r in scan if r[key] > 0]
        if values:
            max_v = max(values)
            min_v = min(values)
            log_max = np.log10(max_v)
            log_min = np.log10(min_v)
            ax.barh(label, log_max - log_min, left=log_min, color=color, alpha=0.7, height=0.5)
            ax.text(log_max + 0.1, label, f"{log_max-log_min:.1f} OoM", fontsize=8, va='center')
    
    # Real hierarchy range
    ax.barh("Real forces", 39, left=-39, color='red', alpha=0.3, height=0.5)
    ax.text(0.5, "Real forces", "39 OoM", fontsize=8, va='center')
    
    ax.barh("Real (no gravity)", 5, left=-5, color='orange', alpha=0.3, height=0.5)
    ax.text(0.5, "Real (no gravity)", "5 OoM", fontsize=8, va='center')
    
    ax.set_xlabel("log₁₀(coupling)")
    ax.set_title("Dynamic Range Comparison\n(Orders of Magnitude)")
    ax.grid(True, alpha=0.2, axis='x')
    
    plt.tight_layout()
    path = os.path.join(results_dir, "exp5_coupling.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


def plot_summary():
    """Summary scorecard across all experiments."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    experiments = [
        ("Exp 0: Force Hierarchy", "✓", "Force hierarchy emerges from single parameter"),
        ("Exp 0: High-E Convergence", "✓", "CV drops 1.37 → 0.02 (not hand-tuned)"),
        ("Exp 1: EM Sign Cancellation", "✗", "Signed force doesn't cancel more at scale"),
        ("Exp 2: Confinement", "✗", "Dynamic threshold doesn't improve confinement"),
        ("Exp 3: 2D Hierarchy", "✓", "Force ordering preserved in 2D"),
        ("Exp 4: Conserved Quantity", "★", "ENTROPY conserved (CV=0.018!)"),
        ("Exp 4: Energy/τ ratio", "★", "energy_mean/τ ≈ 0.73 for all forces"),
        ("Exp 5: Quantitative Ratios", "~", "~3 OoM range; need 39 for full hierarchy"),
    ]
    
    y_pos = np.arange(len(experiments))
    
    for i, (name, result, desc) in enumerate(experiments):
        if result == "✓":
            color = "#44bb44"
        elif result == "★":
            color = "#ffcc00"
        elif result == "✗":
            color = "#cc3333"
        else:
            color = "#dd8800"
        
        ax.barh(i, 1, color=color, alpha=0.3, height=0.7)
        ax.text(0.02, i, f"  {result}  {name}", fontsize=10, va='center', fontweight='bold')
        ax.text(0.5, i, desc, fontsize=9, va='center', color='#444444')
    
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(experiments) - 0.5)
    ax.invert_yaxis()
    ax.set_title("A Gut-Driven Search for a GUT — Scorecard\nOne update function, one parameter (τ), five experiments",
                 fontsize=14, fontweight='bold')
    
    # Score
    hits = sum(1 for _, r, _ in experiments if r in ("✓", "★"))
    misses = sum(1 for _, r, _ in experiments if r == "✗")
    partial = sum(1 for _, r, _ in experiments if r == "~")
    ax.text(0.98, len(experiments) - 0.5, 
            f"Score: {hits} hits, {misses} misses, {partial} partial",
            fontsize=10, ha='right', va='bottom', color='gray')
    
    plt.tight_layout()
    path = os.path.join(results_dir, "summary_scorecard.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


if __name__ == "__main__":
    print("Generating experiment visualizations...")
    plot_exp1()
    plot_exp2()
    plot_exp3()
    plot_exp4()
    plot_exp5()
    plot_summary()
    print("\nDone.")
