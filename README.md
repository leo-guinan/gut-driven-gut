# A Gut-Driven Search for a GUT

**One update function. One parameter. Four forces.**

A toy simulation exploring whether the four fundamental forces of physics
(gravity, electromagnetism, weak nuclear, strong nuclear) could emerge from
a single mechanism differentiated only by activation threshold — the energy
cost required to trigger a state transition.

## The Hypothesis

> All four forces are the same update function with different activation
> thresholds. What we call "time" at any given scale is the observable rate
> of those transitions. Our ability to observe change is directly coupled
> to our ability to create it.

This leads to a specific, testable framing:

| Force    | Threshold | Predicted Behavior                     |
|----------|-----------|----------------------------------------|
| Gravity  | Very low  | Universal, always firing, weak per event, long range |
| EM       | Low       | Common, moderate strength, long range   |
| Weak     | High      | Rare, moderate strength, short range    |
| Strong   | Very high | Very rare, powerful per event, short range |

If the forces are truly the same mechanism at different thresholds, they
should **converge to identical behavior at high energy** — which is exactly
what the real running coupling constants do near the GUT scale (~10¹⁶ GeV).

## What the Simulation Does

A 1D lattice of cells, each with an energy value. One universal update rule:

1. If a cell's energy ≥ threshold τ, it **fires**
2. On firing, it transfers excess energy (above τ) to neighbors
3. Transfer **range** = 1/τ (low threshold → long range)
4. Energy falls off as 1/distance within range
5. Background energy injection acts as a thermal bath

That's it. No force-specific rules. No hand-tuned parameters per force.
Just one threshold value that differs.

## Results

### Test 1: Behavioral Signatures ✓

The force hierarchy emerges cleanly from the single parameter:

```
GRAVITY  (τ=0.01): range=50, fires 91% of the time  → universal, diffuse, weak
EM       (τ=0.10): range=10, fires 42%               → common, moderate
WEAK     (τ=0.50): range=2,  fires 10%               → rare, localized
STRONG   (τ=0.90): range=2,  fires 5%                → very rare, powerful
```

![Force Hierarchy](results/hierarchy.png)
![Force Profiles](results/force_profiles.png)

### Test 2: High-Energy Convergence ✓

At low energy, the forces look completely different. As energy increases,
they converge:

```
E=0.01   CV=1.37  (forces maximally divergent)
E=0.05   CV=0.94
E=0.46   CV=0.29
E=2.15   CV=0.07
E=10.0   CV=0.02  (forces nearly indistinguishable)
```

This was not put in by hand. It emerges naturally from the model. The
forces become indistinguishable when everything has enough energy to fire
regardless of threshold — mirroring how the real coupling constants
converge near the GUT scale.

![Convergence](results/convergence.png)

### Test 3: Conservation ✗

Range × Strength is NOT conserved across forces (CV = 1.31). There is no
simple conserved quantity linking range and strength. This means the
threshold doesn't just repartition a fixed budget — it changes the total
amount of influence in the system.

This is an honest miss. It constrains what the model can claim.

## What This Model Can't Do (Yet)

- **Confinement**: the strong force gets stronger with distance. Our model
  has fixed range per threshold.
- **Attraction vs repulsion**: gravity only attracts, EM does both. Our
  model has no sign.
- **Symmetry breaking**: the electroweak split is a phase transition
  mediated by the Higgs, not just "different thresholds."
- **Quantum numbers**: spin, charge, flavor. The real forces are
  distinguished by symmetry groups (U(1), SU(2), SU(3)), not a scalar.

## What It Suggests

The force hierarchy **can** emerge from a single parameter. The convergence
at high energy falls out for free. Whether the deeper structure is real
depends on finding what quantity, if any, is actually conserved — and
whether confinement-like behavior can emerge from a dynamic threshold.

## Running It

```bash
pip install numpy matplotlib
python sim/unified_force.py        # run simulation, print results
python sim/visualize.py            # generate plots
```

## Next Experiments

1. **Add sign** (attraction/repulsion) — does EM-like cancellation emerge
   at large scales?
2. **Dynamic threshold** — let τ depend on local density. Does
   confinement-like behavior appear?
3. **2D/3D lattice** — geometry matters for real forces
4. **Find the conserved quantity** — if range × strength isn't it, what is?
5. **Derive coupling constant ratios** — can a single activation-cost
   function recover the known force hierarchy quantitatively?

## Philosophy

The conventional approach to unification starts from symmetry groups and
works down. This starts from a different question: what if the forces are
distinguished not by their mathematical structure but by their energetic
cost? What if time itself — the rate at which we observe change — is the
variable that makes one mechanism look like four?

This is a toy model. It proves nothing. But the convergence result fell
out for free, and that's what a good intuition looks like when it meets
a simulation.

## License

MIT

## Author

Leo Guinan — [@leo_guinan](https://twitter.com/leo_guinan)

*"The thesis means nothing without the artifact."*
