#!/usr/bin/env python3
"""
Visualization for Experiment 6: Quaternion States on S³
"""

import json
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

results_dir = os.path.join(os.path.dirname(__file__), "..", "results")

FORCE_COLORS = {
    "gravity": "#4466cc",
    "EM": "#44bb44",
    "weak": "#dd8800",
    "strong": "#cc3333",
}

def plot_exp6():
    with open(os.path.join(results_dir, "exp6_quaternion.json")) as f:
        data = json.load(f)
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Experiment 6: Quaternion States on S³ (Closure-SDK)\nCells live on the 3-sphere. Update = Hamilton product. Threshold = σ.",
                 fontsize=13, fontweight='bold')
    
    # Panel 1: Force hierarchy on S³
    ax1 = fig.add_subplot(2, 3, 1)
    forces = ["gravity", "EM", "weak", "strong"]
    rates = [data["channel_test"][f]["firing_rate"] for f in forces]
    taus = [data["channel_test"][f]["tau"] for f in forces]
    colors = [FORCE_COLORS[f] for f in forces]
    
    ax1.bar(forces, rates, color=colors, alpha=0.8)
    ax1.set_ylabel("Firing Rate")
    ax1.set_title("Force Hierarchy on S³")
    for i, (f, t) in enumerate(zip(forces, taus)):
        ax1.text(i, rates[i] + 0.01, f"τ={t}", ha='center', fontsize=8, color='gray')
    
    # Panel 2: σ (geodesic distance) for each force  
    ax2 = fig.add_subplot(2, 3, 2)
    sigmas = [data["channel_test"][f]["mean_sigma"] for f in forces]
    ax2.bar(forces, sigmas, color=colors, alpha=0.8)
    ax2.set_ylabel("Mean σ (geodesic distance)")
    ax2.set_title("Cell Distance from Identity")
    
    # Panel 3: W vs XYZ channel separation (THE KEY RESULT)
    ax3 = fig.add_subplot(2, 3, 3)
    
    single = data["single_threshold"]
    channels = ["W", "X", "Y", "Z"]
    stds = [single["w_std"], single["x_std"], single["y_std"], single["z_std"]]
    ch_colors = ["#cc8800", "#cc3333", "#33cc33", "#3333cc"]
    
    bars = ax3.bar(channels, stds, color=ch_colors, alpha=0.8, edgecolor='white')
    ax3.set_ylabel("Standard Deviation")
    ax3.set_title(f"Channel Separation (τ={single['tau']})\nW = existence, XYZ = orientation")
    
    # Annotate the ratio
    w_std = stds[0]
    xyz_avg = np.mean(stds[1:])
    ratio = w_std / xyz_avg
    ax3.annotate(f"W is {ratio:.1f}× more\nvariable than XYZ",
                xy=(0, w_std), xytext=(1.5, w_std * 0.8),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    
    # Panel 4: Convergence on S³
    ax4 = fig.add_subplot(2, 3, 4)
    conv = data["convergence"]
    energies = [v["energy"] for v in conv.values()]
    cvs = [v["cv"] for v in conv.values()]
    
    ax4.plot(energies, cvs, 'ko-', linewidth=2, markersize=5)
    ax4.set_xlabel("Perturbation Energy")
    ax4.set_ylabel("CV of Firing Rates")
    ax4.set_title("Convergence on S³")
    ax4.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label="CV = 0.1")
    ax4.grid(True, alpha=0.2)
    ax4.legend(fontsize=8)
    
    # Panel 5: W channel entropy vs XYZ across forces
    ax5 = fig.add_subplot(2, 3, 5)
    
    w_ents = [data["channel_test"][f]["w_channel"]["entropy"] for f in forces]
    xyz_ents = [np.mean([
        data["channel_test"][f]["xyz_channels"]["entropy_x"],
        data["channel_test"][f]["xyz_channels"]["entropy_y"],
        data["channel_test"][f]["xyz_channels"]["entropy_z"],
    ]) for f in forces]
    
    x = np.arange(len(forces))
    width = 0.35
    ax5.bar(x - width/2, w_ents, width, label="W (existence)", color="#cc8800", alpha=0.8)
    ax5.bar(x + width/2, xyz_ents, width, label="mean(XYZ) (orientation)", color="#3366cc", alpha=0.8)
    ax5.set_xticks(x)
    ax5.set_xticklabels(forces)
    ax5.set_ylabel("Channel Entropy")
    ax5.set_title("W vs XYZ Entropy by Force")
    ax5.legend(fontsize=8)
    
    # Panel 6: Summary text
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = f"""KEY FINDINGS:

✓ Force hierarchy partially preserved on S³
  (gravity > EM/weak > strong, with EM/weak swap)

✓ Convergence at high energy 
  (CV: {cvs[0]:.2f} → {cvs[-1]:.2f})

★ CHANNEL SEPARATION:
  W (scalar) std = {single['w_std']:.4f}
  XYZ (vector) std ≈ {xyz_avg:.4f}
  Ratio: {ratio:.1f}×

  The geometry of S³ naturally separates
  scalar (existence) from vector (orientation)
  degrees of freedom — WITHOUT this being
  put in by hand.

  W channel ≈ "does the state update exist?"
  XYZ channels ≈ "how is the state oriented?"

Using: {'Closure-SDK' if data['using_closure_sdk'] else 'pure numpy'}"""
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    path = os.path.join(results_dir, "exp6_quaternion.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


if __name__ == "__main__":
    print("Generating Experiment 6 visualizations...")
    plot_exp6()
    print("Done.")
