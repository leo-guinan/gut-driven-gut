#!/usr/bin/env python3
"""
Unified Force Engine — shared simulation core for all experiments.

The single update function lives here. Experiments import and extend it.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable


@dataclass
class ForceConfig:
    """A force is an activation threshold + optional modifiers."""
    name: str
    tau: float                          # activation threshold
    color: str = "#888888"
    signed: bool = False                # if True, cells carry charge (+/-)
    dynamic_tau: bool = False           # if True, tau depends on local density
    dynamic_tau_fn: Optional[Callable] = None  # custom tau function


@dataclass
class SimConfig:
    n_cells: int = 100
    n_steps: int = 500
    energy_inject: float = 0.05
    decay_rate: float = 0.001
    seed: int = 42
    dimensions: int = 1                 # 1 or 2
    grid_size: int = 50                 # for 2D: grid_size × grid_size


# Standard force definitions
FORCES = [
    ForceConfig("gravity", tau=0.01, color="#4466cc"),
    ForceConfig("EM",      tau=0.10, color="#44bb44"),
    ForceConfig("weak",    tau=0.50, color="#dd8800"),
    ForceConfig("strong",  tau=0.90, color="#cc3333"),
]


def update_1d(cells: np.ndarray, tau: float, 
              signed: bool = False, charges: Optional[np.ndarray] = None,
              dynamic_tau_fn: Optional[Callable] = None) -> dict:
    """
    Universal update function for 1D lattice.
    
    If signed=True, cells have charges and transfers can be attractive or repulsive.
    If dynamic_tau_fn is provided, tau varies per cell based on local environment.
    """
    n = len(cells)
    energy = np.abs(cells) if signed else cells
    
    # Dynamic threshold: tau can depend on local density
    if dynamic_tau_fn is not None:
        tau_arr = dynamic_tau_fn(cells, tau)
    else:
        tau_arr = np.full(n, tau)
    
    # Cells that fire
    firing_mask = energy >= tau_arr
    n_fired = int(firing_mask.sum())
    
    if n_fired == 0:
        return {
            "n_fired": 0, "total_transferred": 0.0,
            "effective_range": 0.0, "avg_transfer_per_event": 0.0,
            "firing_rate": 0.0, "net_attraction": 0.0, "net_repulsion": 0.0,
        }
    
    # Range from threshold (use per-cell tau for range calculation)
    mean_tau = float(tau_arr[firing_mask].mean())
    effective_range = max(1, min(int(np.ceil(1.0 / mean_tau)), n // 2))
    
    # Weight kernel
    distances = np.arange(1, effective_range + 1, dtype=np.float64)
    weights = 1.0 / distances
    
    # Excess energy
    excess = np.where(firing_mask, energy - tau_arr, 0.0)
    
    # Sign handling
    if signed and charges is not None:
        signed_excess = excess * charges  # carries the sign
    else:
        signed_excess = excess
    
    incoming = np.zeros(n, dtype=np.float64)
    
    for d_idx, d in enumerate(range(1, effective_range + 1)):
        w = weights[d_idx]
        if signed and charges is not None:
            # Like charges repel (push energy away), opposite attract (pull)
            # Transfer sign depends on source charge × distance direction
            incoming[d:] += signed_excess[:-d] * w
            incoming[:-d] += signed_excess[d:] * w
        else:
            incoming[d:] += excess[:-d] * w
            incoming[:-d] += excess[d:] * w
    
    total_weight = 2.0 * weights.sum()
    if total_weight > 0:
        incoming /= total_weight
    
    total_transferred = float(np.abs(incoming).sum())
    net_attraction = float(np.maximum(incoming, 0).sum())
    net_repulsion = float(np.maximum(-incoming, 0).sum())
    
    cells[:] = cells + incoming
    if signed:
        # Firing cells reset toward zero but keep their charge sign
        reset_vals = tau_arr * 0.5
        cells[firing_mask] = np.sign(cells[firing_mask]) * reset_vals[firing_mask]
    else:
        cells[firing_mask] = tau_arr[firing_mask] * 0.5
    
    excess_values = excess[firing_mask]
    avg_transfer = float(excess_values.mean()) if len(excess_values) > 0 else 0.0
    
    return {
        "n_fired": n_fired,
        "total_transferred": total_transferred,
        "effective_range": float(effective_range),
        "avg_transfer_per_event": avg_transfer,
        "firing_rate": float(n_fired) / n,
        "net_attraction": net_attraction,
        "net_repulsion": net_repulsion,
    }


def update_2d(grid: np.ndarray, tau: float,
              dynamic_tau_fn: Optional[Callable] = None) -> dict:
    """
    Universal update function for 2D lattice.
    Same rule, now in two dimensions.
    """
    h, w = grid.shape
    n = h * w
    
    if dynamic_tau_fn is not None:
        tau_arr = dynamic_tau_fn(grid, tau)
    else:
        tau_arr = np.full_like(grid, tau)
    
    firing_mask = grid >= tau_arr
    n_fired = int(firing_mask.sum())
    
    if n_fired == 0:
        return {
            "n_fired": 0, "total_transferred": 0.0,
            "effective_range": 0.0, "avg_transfer_per_event": 0.0,
            "firing_rate": 0.0,
        }
    
    mean_tau = float(tau_arr[firing_mask].mean())
    effective_range = max(1, min(int(np.ceil(1.0 / mean_tau)), min(h, w) // 2))
    
    excess = np.where(firing_mask, grid - tau_arr, 0.0)
    incoming = np.zeros_like(grid)
    total_weight = 0.0
    
    # 2D transfer: iterate over offsets within range
    for dx in range(-effective_range, effective_range + 1):
        for dy in range(-effective_range, effective_range + 1):
            if dx == 0 and dy == 0:
                continue
            dist = np.sqrt(dx * dx + dy * dy)
            if dist > effective_range:
                continue
            weight = 1.0 / dist
            total_weight += weight
            
            # Shifted slicing
            src_y = slice(max(0, -dy), h - max(0, dy))
            src_x = slice(max(0, -dx), w - max(0, dx))
            dst_y = slice(max(0, dy), h - max(0, -dy))
            dst_x = slice(max(0, dx), w - max(0, -dx))
            
            incoming[dst_y, dst_x] += excess[src_y, src_x] * weight
    
    if total_weight > 0:
        incoming /= total_weight
    
    total_transferred = float(incoming.sum())
    
    grid[:] = grid + incoming
    grid[firing_mask] = tau_arr[firing_mask] * 0.5
    
    excess_values = excess[firing_mask]
    avg_transfer = float(excess_values.mean()) if len(excess_values) > 0 else 0.0
    
    return {
        "n_fired": n_fired,
        "total_transferred": total_transferred,
        "effective_range": float(effective_range),
        "avg_transfer_per_event": avg_transfer,
        "firing_rate": float(n_fired) / n,
    }


def run_1d(force: ForceConfig, config: SimConfig, 
           charges: Optional[np.ndarray] = None) -> dict:
    """Run a 1D simulation."""
    rng = np.random.default_rng(config.seed)
    cells = rng.uniform(0, 1, config.n_cells)
    
    if force.signed and charges is None:
        charges = rng.choice([-1.0, 1.0], config.n_cells)
    
    history = {
        "firing_rates": [], "total_transferred": [], "effective_ranges": [],
        "avg_transfers": [], "energy_means": [], "energy_stds": [],
        "spatial_correlation": [], "net_attraction": [], "net_repulsion": [],
        "snapshots": [],
    }
    
    for step in range(config.n_steps):
        cells += rng.uniform(0, config.energy_inject, config.n_cells)
        cells *= (1.0 - config.decay_rate)
        
        metrics = update_1d(cells, force.tau, 
                           signed=force.signed, charges=charges,
                           dynamic_tau_fn=force.dynamic_tau_fn if force.dynamic_tau else None)
        
        history["firing_rates"].append(metrics["firing_rate"])
        history["total_transferred"].append(metrics["total_transferred"])
        history["effective_ranges"].append(metrics["effective_range"])
        history["avg_transfers"].append(metrics["avg_transfer_per_event"])
        history["energy_means"].append(float(np.mean(np.abs(cells))))
        history["energy_stds"].append(float(np.std(cells)))
        history["net_attraction"].append(metrics.get("net_attraction", 0.0))
        history["net_repulsion"].append(metrics.get("net_repulsion", 0.0))
        
        if len(cells) > 1:
            corr = np.corrcoef(np.abs(cells[:-1]), np.abs(cells[1:]))[0, 1]
            history["spatial_correlation"].append(float(corr) if not np.isnan(corr) else 0.0)
        
        # Save snapshots periodically
        if step % (config.n_steps // 10) == 0:
            history["snapshots"].append(cells.copy())
    
    return history


def run_2d(force: ForceConfig, config: SimConfig) -> dict:
    """Run a 2D simulation."""
    rng = np.random.default_rng(config.seed)
    gs = config.grid_size
    grid = rng.uniform(0, 1, (gs, gs))
    
    history = {
        "firing_rates": [], "total_transferred": [], "effective_ranges": [],
        "avg_transfers": [], "energy_means": [], "energy_stds": [],
        "snapshots": [],
    }
    
    for step in range(config.n_steps):
        grid += rng.uniform(0, config.energy_inject, (gs, gs))
        grid *= (1.0 - config.decay_rate)
        
        metrics = update_2d(grid, force.tau,
                           dynamic_tau_fn=force.dynamic_tau_fn if force.dynamic_tau else None)
        
        history["firing_rates"].append(metrics["firing_rate"])
        history["total_transferred"].append(metrics["total_transferred"])
        history["effective_ranges"].append(metrics["effective_range"])
        history["avg_transfers"].append(metrics["avg_transfer_per_event"])
        history["energy_means"].append(float(grid.mean()))
        history["energy_stds"].append(float(grid.std()))
        
        if step % (config.n_steps // 10) == 0:
            history["snapshots"].append(grid.copy().tolist())
    
    return history
