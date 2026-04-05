#!/usr/bin/env python3
"""
Experiment 3: 2D Lattice — Does Geometry Change the Story?

Question: The original simulation was 1D. In 2D, forces spread over
an area rather than a line. Does the qualitative behavior survive?
Do new phenomena emerge (clustering, phase transitions)?

Key 2D predictions:
- Gravity (low τ): should create diffuse, large-scale correlations
- Strong (high τ): should create tight, localized clusters
- The force hierarchy should be preserved
- Energy should spread in 2D patterns that look different from 1D
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import json
from engine import ForceConfig, SimConfig, run_2d, FORCES


def spatial_clustering(grid, threshold=None):
    """
    Measure spatial clustering: count connected components of high-energy cells.
    More clusters = more localized. Fewer large clusters = more diffuse.
    """
    if threshold is None:
        threshold = np.mean(grid) + np.std(grid)
    
    binary = grid > threshold
    h, w = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    clusters = []
    
    def flood_fill(y, x):
        stack = [(y, x)]
        size = 0
        while stack:
            cy, cx = stack.pop()
            if cy < 0 or cy >= h or cx < 0 or cx >= w:
                continue
            if visited[cy, cx] or not binary[cy, cx]:
                continue
            visited[cy, cx] = True
            size += 1
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                stack.append((cy+dy, cx+dx))
        return size
    
    for y in range(h):
        for x in range(w):
            if binary[y, x] and not visited[y, x]:
                size = flood_fill(y, x)
                if size > 0:
                    clusters.append(size)
    
    return clusters


def run_experiment():
    config = SimConfig(n_steps=200, grid_size=40, seed=42)
    
    results = {}
    
    for force in FORCES:
        print(f"  Running 2D {force.name} (τ={force.tau})...")
        hist = run_2d(force, config)
        
        ss = slice(-50, None)
        
        # Analyze final snapshot
        final = np.array(hist["snapshots"][-1])
        clusters = spatial_clustering(final)
        
        results[force.name] = {
            "tau": force.tau,
            "avg_firing_rate": float(np.mean(hist["firing_rates"][ss])),
            "avg_energy": float(np.mean(hist["energy_means"][ss])),
            "energy_std": float(np.mean(hist["energy_stds"][ss])),
            "effective_range": float(np.mean(hist["effective_ranges"][ss])),
            "n_clusters": len(clusters),
            "avg_cluster_size": float(np.mean(clusters)) if clusters else 0.0,
            "max_cluster_size": int(max(clusters)) if clusters else 0,
            "cluster_sizes": sorted(clusters, reverse=True)[:10],  # top 10
            "final_snapshot": hist["snapshots"][-1],
            "firing_rate_ts": hist["firing_rates"],
        }
    
    return results


def main():
    print("=" * 60)
    print("EXPERIMENT 3: 2D Lattice — Geometry Test")
    print("=" * 60)
    print()
    
    results = run_experiment()
    
    print("\n  2D FORCE BEHAVIORS:")
    print(f"  {'Force':<10} {'τ':>6} {'Rate':>8} {'Range':>8} {'Energy':>8} {'Clusters':>10} {'AvgSize':>10} {'MaxSize':>10}")
    print(f"  {'-'*10} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
    
    for name, data in results.items():
        print(f"  {name:<10} {data['tau']:>6.2f} {data['avg_firing_rate']:>8.4f} "
              f"{data['effective_range']:>8.1f} {data['avg_energy']:>8.4f} "
              f"{data['n_clusters']:>10} {data['avg_cluster_size']:>10.1f} "
              f"{data['max_cluster_size']:>10}")
    
    # Compare with 1D expectations
    print(f"\n  COMPARISON WITH 1D PREDICTIONS:")
    
    gravity = results["gravity"]
    strong = results["strong"]
    
    # Gravity should have fewer, larger clusters (diffuse)
    # Strong should have more, smaller clusters (localized)
    
    if gravity["avg_cluster_size"] > strong["avg_cluster_size"]:
        print(f"    ✓ Gravity clusters larger than strong ({gravity['avg_cluster_size']:.1f} vs {strong['avg_cluster_size']:.1f})")
    else:
        print(f"    ✗ Gravity clusters NOT larger than strong")
    
    if gravity["n_clusters"] < strong["n_clusters"] or gravity["avg_cluster_size"] > strong["avg_cluster_size"] * 2:
        print(f"    ✓ Gravity more diffuse than strong (consistent with 1D)")
    else:
        print(f"    ~ Pattern unclear in 2D")
    
    # Does the hierarchy survive?
    rates = [(name, data["avg_firing_rate"]) for name, data in results.items()]
    rates.sort(key=lambda x: x[1], reverse=True)
    expected_order = ["gravity", "EM", "weak", "strong"]
    actual_order = [r[0] for r in rates]
    
    if actual_order == expected_order:
        print(f"    ✓ Force hierarchy preserved in 2D: {' > '.join(actual_order)}")
    else:
        print(f"    ~ Hierarchy in 2D: {' > '.join(actual_order)} (expected {' > '.join(expected_order)})")
    
    # Save (excluding large snapshots for JSON size)
    output = {}
    for name, data in results.items():
        output[name] = {k: v for k, v in data.items() if k != "final_snapshot"}
        # Keep one snapshot but compress it
        output[name]["has_snapshot"] = True
    
    output_path = os.path.join(os.path.dirname(__file__), "..", "results", "exp3_2d.json")
    
    # Save full data separately for visualization
    full_path = os.path.join(os.path.dirname(__file__), "..", "results", "exp3_2d_full.json")
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    # Save snapshots separately (they're large)
    snapshots = {name: data["final_snapshot"] for name, data in results.items()}
    with open(full_path, "w") as f:
        json.dump(snapshots, f)
    
    print(f"\n  Data saved to {output_path}")
    
    return results


if __name__ == "__main__":
    main()
