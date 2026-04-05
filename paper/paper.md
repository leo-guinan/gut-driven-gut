# Closure Cadence on S³: A Toy Model of Force Hierarchy from Quaternion Composition

**Leo Guinan**
MetaSPN / Proximity Fund

**With computational substrate by Walter Henrique Alves da Silva**
Open Research Institute, Closure-SDK

---

## Abstract

We present a toy computational model exploring the hypothesis that the four fundamental forces of physics arise from a single update mechanism operating at different energy scales. The model consists of a lattice of cells governed by one update rule. In the scalar version, cells accumulate energy and fire when a threshold $\tau$ is exceeded; in the $S^3$ version, cells are unit quaternions updated by Hamilton product, and force-like behavior emerges from *closure cadence*—the rate at which running products return to the identity. We report eight experiments with honest accounting: the model successfully produces a qualitative force hierarchy from a single parameter, preserves that hierarchy across lattice dimensions, and exhibits high-energy convergence (CV $1.37 \to 0.02$). It fails to reproduce confinement or electromagnetic cancellation at scale. Two results are unexpected: (1) firing entropy is conserved across all force regimes (CV = 0.018), without tuning; and (2) the Hopf decomposition of quaternion states necessarily shifts from W-dominant at low energy to RGB-dominant at high energy—a geometric theorem that, when coupled with hierarchical closure, produces a natural separation between scalar (existence) and vector (orientation) degrees of freedom. The model is qualitative, not quantitative, spanning only three of the required thirty-nine orders of magnitude. We publish it as a falsifiable framework, not a claim.

**Keywords:** force unification, quaternion dynamics, $S^3$ geometry, toy model, closure, Hopf fibration

---

## 1. Introduction

The Standard Model of particle physics describes four fundamental forces—gravity, electromagnetism, the weak nuclear force, and the strong nuclear force—as distinct interactions mediated by different gauge bosons and governed by different coupling constants. Grand Unified Theories (GUTs) propose that the non-gravitational forces merge into a single interaction at high energies ($\sim 10^{16}$ GeV), and various quantum gravity programs seek to include gravity in this unification [1, 2]. The running of coupling constants, experimentally confirmed at accessible energies, provides the primary empirical motivation: the three gauge couplings approximately converge when extrapolated upward [3].

We start from a deliberately naive hypothesis:

> **All four fundamental forces are the same mechanism at different energy scales.** Time at any scale is the observable rate of state transitions. The activation cost to trigger an update differentiates the forces.

This is not a theory of physics. It is an *intuition* that we wish to translate into a falsifiable computational framework. The question is whether a single update rule, applied to a lattice with no force-specific parameters, can produce anything resembling the observed force hierarchy—and if so, what structural features of that hierarchy are robust versus accidental.

We are motivated by three observations from known mathematics and physics:

1. **Hurwitz's theorem** [4]: The only normed division algebras over the reals are $\mathbb{R}$, $\mathbb{C}$, $\mathbb{H}$, and $\mathbb{O}$, of dimensions 1, 2, 4, and 8. The quaternions $\mathbb{H}$ are the richest *associative* division algebra, and their unit group $S^3$ is diffeomorphic to $\mathrm{SU}(2)$—the gauge group of the weak force. The $1+3$ decomposition (one scalar, three vector components) is not a modeling choice; it is forced by the algebra.

2. **Running coupling constants**: The strong, weak, and electromagnetic couplings change with energy scale and approximately converge near the GUT scale. Any model claiming a single mechanism should reproduce at least the qualitative ordering and convergence.

3. **Entropic gravity**: Verlinde [5] argued that gravity may emerge from entropic considerations rather than being fundamental. If forces emerge from information processing, a lattice model that conserves informational quantities across force regimes would be consonant with this program.

We do not claim to advance any of these lines of research. We claim only to have built a toy model, run it honestly, and found some features worth reporting.

The simulation code is available at [github.com/leo-guinan/gut-driven-gut](https://github.com/leo-guinan/gut-driven-gut). The quaternion substrate builds on Walter da Silva's Closure-SDK [6], available at [github.com/faltz009/Closure-SDK](https://github.com/faltz009/Closure-SDK).

---

## 2. Model Description

### 2.1 Scalar Version

The lattice consists of $N$ cells, each carrying a scalar energy $E_i(t)$. At each timestep:

1. Each cell receives energy from a stochastic source: $E_i(t) \leftarrow E_i(t) + \xi_i$, where $\xi_i \sim \mathcal{N}(0, \sigma^2)$.
2. If $E_i(t) \geq \tau$ (the activation threshold), the cell *fires*: it redistributes energy to its neighbors and resets.
3. All cells update synchronously.

The threshold $\tau$ is the sole free parameter. By varying $\tau$ across orders of magnitude, we assign different thresholds to different "force" regimes:

| Force analogue | Threshold $\tau$ | Predicted behavior |
|---|---|---|
| Gravitational | 0.01 (lowest) | Fires constantly, long range, individually weak |
| Electromagnetic | 0.10 | Moderate firing rate and range |
| Weak | 0.50 | Infrequent, short range |
| Strong | 0.90 (highest) | Rare, short range, individually powerful |

The hypothesis predicts that higher-threshold forces fire less frequently but with greater per-event impact—producing a *hierarchy of cadences* from a single rule. Low threshold $=$ diffuse, universal, weak per event (gravity-like). High threshold $=$ rare, local, powerful per event (strong-force-like).

### 2.2 $S^3$ Version (Quaternion States)

In the $S^3$ formulation, each cell's state is a unit quaternion $q_i \in S^3 \subset \mathbb{H}$:

$$q_i = w_i + x_i\,\mathbf{i} + y_i\,\mathbf{j} + z_i\,\mathbf{k}, \quad |q_i| = 1$$

The update rule is Hamilton product (quaternion composition):

$$q_i(t+1) = q_i(t) \otimes \delta q_i(t)$$

where $\delta q_i$ is a small random perturbation quaternion drawn near the identity. Following the Closure-SDK [6], we define:

- **Energy**: $\sigma_i = \arccos(|w_i|) \in [0, \pi/2]$, the geodesic distance from the identity on $S^3$.
- **Closure**: The running product $Q(t) = \prod_{s=0}^{t} \delta q(s)$ *closes* when $\sigma(Q) \to 0$, i.e., the product returns to the neighborhood of the identity after excursion.
- **Hopf decomposition**: The quaternion decomposes into a scalar channel $W = w^2$ and vector channels $R = x^2, G = y^2, B = z^2$, satisfying $W + R + G + B = 1$.

The critical innovation of the $S^3$ version (Experiment 7) is that **no threshold parameter is needed**. The closure cadence—the rate at which running products return to identity—itself varies with the energy of perturbations $\sigma(\delta q)$, producing a force hierarchy from geometry alone.

The three sufficient operations from Closure-SDK are:

1. **compose**: $q_1 \otimes q_2$ (Hamilton product)
2. **sigma**: $\sigma(q) = \arccos(|w|)$ (energy measure)
3. **close**: detect when $\sigma \to 0$ (closure event)

---

## 3. Experiments and Results

We conducted eight experiments, numbered 0 through 8 (Experiment 0 being the baseline). Results are summarized in Table 1 and described in detail below.

### Table 1: Summary of Experiments

| Exp | Description | Result | Key Metric |
|-----|-------------|--------|------------|
| 0 | Force hierarchy from single $\tau$ | ✓ | CV of firing rates: $1.37 \to 0.02$ at high energy |
| 1 | Signed energy (attraction/repulsion) | ✗ | Balanced forces; no scale-dependent cancellation |
| 2 | Dynamic threshold (confinement) | ✗ | No improvement over static threshold |
| 3 | 2D lattice | ✓ | Hierarchy preserved across dimensions |
| 4 | Conserved quantity search | ★ | Entropy CV = 0.018; $\langle E \rangle / \tau \approx 0.73$ (CV = 0.020) |
| 5 | Coupling constant ratios | ~ | 3 orders of magnitude (need 39) |
| 6 | Quaternion states on $S^3$ | ★ | W-channel variance $5.1\times$ higher than XYZ |
| 7 | Closure-driven forces | ★ | No $\tau$; Hopf shift: 100% W $\to$ 96% RGB |
| 8 | Recursive hierarchy | ★ | Levels fire $47$–$85\times$ slower; Level 0: 76% RGB, Level 1: 100% W |

*Legend: ✓ = success, ✗ = failure, ★ = unexpected/notable, ~ = partial*

### 3.1 Experiment 0: Force Hierarchy from a Single Parameter

**Setup.** A 1D lattice of 100 cells, threshold $\tau$ varied over four orders of magnitude. 10,000 timesteps per run.

**Result: ✓** Firing rates decrease monotonically with $\tau$, producing a qualitative hierarchy. At low energy (large $\tau$ spread), the coefficient of variation (CV) of firing rates across force regimes is 1.37, reflecting strong differentiation. At high energy (all $\tau$ values compressed), CV drops to 0.02—the forces *converge*. This mirrors the qualitative behavior of running coupling constants approaching the GUT scale.

### 3.2 Experiment 1: Signed Energy and Attraction/Repulsion

**Setup.** Cells carry signed energy. Positive values represent repulsion; negative values represent attraction. The goal was to reproduce the sign structure of electromagnetic interactions.

**Result: ✗** The lattice produces balanced positive and negative firing events, but there is no emergent scale-dependent cancellation. Charges do not cancel at macroscopic scales as they do in electromagnetism. The sign structure is a necessary but insufficient ingredient.

### 3.3 Experiment 2: Dynamic Threshold for Confinement

**Setup.** The threshold $\tau$ increases with the distance between fired cells, aiming to model confinement (the strong force growing with distance).

**Result: ✗** Dynamic thresholds did not produce confinement-like behavior. The firing pattern changes quantitatively but does not exhibit the qualitative feature of increasing force with distance. We conclude that confinement requires additional structure beyond a single scalar threshold.

### 3.4 Experiment 3: 2D Lattice

**Setup.** A $32 \times 32$ lattice with nearest-neighbor coupling, otherwise identical to Experiment 0.

**Result: ✓** The force hierarchy is preserved in two dimensions. The ordering of firing rates across $\tau$ values is qualitatively identical to the 1D case. This suggests the hierarchy is a property of the update rule, not the lattice topology.

### 3.5 Experiment 4: Conserved Quantity Search

**Setup.** Systematic scan of statistical quantities across force regimes: mean energy, variance, entropy of the firing distribution, energy-to-threshold ratio.

**Result: ★** Two conserved quantities emerged:

1. **Firing entropy** is constant across all threshold values, with CV = 0.018. This was not imposed or tuned. The Shannon entropy of the firing time distribution is the same whether the threshold corresponds to the "strong force" or "gravity."

2. **The ratio $\langle E \rangle / \tau \approx 0.73$** is constant across all force regimes (CV = 0.020). The mean cell energy stabilizes at approximately 73% of the activation threshold regardless of scale.

The entropy conservation is the single most robust quantitative result in this work. We do not have an analytical explanation for it.

### 3.6 Experiment 5: Coupling Constant Ratios

**Setup.** Extract effective coupling constants (firing rate × redistribution amplitude) and compare ratios to the known force hierarchy.

**Result: ~** The model produces approximately three orders of magnitude of dynamic range in coupling strength. The real hierarchy spans roughly 39 orders of magnitude (from the strong force to gravity). The qualitative ordering is correct, but the quantitative gap is enormous. We regard this as a partial success at best.

### 3.7 Experiment 6: Quaternion States on $S^3$

**Setup.** Replace scalar cell states with unit quaternions. Update via Hamilton product with random perturbations. Measure the variance of each component channel $(w, x, y, z)$ over time.

**Result: ★** The scalar channel $W$ exhibits $5.1\times$ higher variance than the vector channels $X$, $Y$, $Z$, which are approximately equal to each other. The geometry of $S^3$ naturally separates a scalar degree of freedom (existence/magnitude) from three vector degrees of freedom (orientation/arrangement). This $1+3$ decomposition is not imposed by the model; it is a consequence of quaternion geometry under random composition.

### 3.8 Experiment 7: Closure-Driven Forces

**Setup.** The threshold parameter $\tau$ is eliminated entirely. Instead, force-like behavior emerges from *closure cadence*: the rate at which running quaternion products return to the identity on $S^3$. Perturbation energy $\sigma(\delta q)$ is varied.

**Result: ★** This is the central result. At low perturbation energy ($\sigma \ll 1$), closure events are frequent and the Hopf decomposition is 100% W-dominant—the dynamics are purely scalar. At high perturbation energy ($\sigma \gg 1$), closure events are rare and the decomposition shifts to 96% RGB-dominant—the dynamics are almost entirely vectorial. The hierarchy of closure cadences, from fast (strong-force-like) to slow (gravity-like), emerges from a single geometric mechanism with no free parameters beyond the perturbation energy scale.

### 3.9 Experiment 8: Recursive Hierarchy

**Setup.** A two-level hierarchy: Level 0 cells fire according to their own closure cadence, and their collective firing events become the input perturbations for Level 1 cells.

**Result: ★** Level 1 fires $47$–$85\times$ slower than Level 0, consistent with a hierarchical force structure. The Hopf decomposition at Level 0 (fast, high-frequency) is 76% RGB (vector-dominant). At Level 1 (slow, low-frequency), it is 100% W (scalar-dominant). This maps suggestively onto the physical picture: high-energy (strong/EM) forces are orientation-rich, while low-energy (gravitational) force is purely scalar.

---

## 4. Analytical Results

Beyond the simulation experiments, we performed two analytical investigations of the model's structural properties.

### 4.1 Hierarchy Ratio

We measured the ratio of closure cadences between adjacent force levels across 50 random seeds and multiple energy scales.

- **Across seeds** (fixed energy): The hierarchy ratio has mean $\approx 85:1$ with CV = 0.59. The ratio varies substantially from run to run. It is *stochastic*, not a geometric constant.
- **Across energy scales** (fixed seed ensemble): The hierarchy ratio has CV = 0.26. It is more stable across energy scales than across seeds.

**Interpretation.** The hierarchy ratio is *scale-independent* but *seed-dependent*. This means the model produces a consistent qualitative ordering of forces regardless of the overall energy scale, but the precise numerical ratio is not determined by geometry. The $\sim 85:1$ mean ratio is far from the $\sim 10^{39}$ ratio between the strong force and gravity, consistent with the model's limited dynamic range (Experiment 5).

### 4.2 Hopf Shift: A Geometric Theorem

The most analytically tractable result concerns the Hopf decomposition shift observed in Experiments 7 and 8. We can prove:

**Claim.** For a unit quaternion $q = \cos\sigma + \sin\sigma\,\hat{v}$ (where $\hat{v}$ is a unit pure quaternion and $\sigma = \arccos(|w|)$ is the energy), the W-fraction $W = w^2 = \cos^2\sigma$ exceeds $1/2$ if and only if $\sigma < \pi/4$.

*Proof.* $W = \cos^2\sigma > 1/2 \iff \cos\sigma > 1/\sqrt{2} \iff \sigma < \pi/4 \approx 0.785$. $\square$

This is a theorem about the geometry of $S^3$, not an emergent property of the simulation. The transition point $\sigma = \pi/4$ is the *equator* of the Hopf fibration in the relevant coordinates.

**However**, the following observation *is* novel: hierarchical closure necessarily produces this shift. In the low-energy regime, perturbations are small ($\sigma \ll \pi/4$), closure is frequent, and the dynamics are W-dominant (scalar). In the high-energy regime, perturbations are large ($\sigma \gg \pi/4$), closure is rare, and the dynamics are RGB-dominant (vector). The *coupling* of hierarchy and Hopf decomposition is not a tautology—it requires that the closure mechanism preserves the energy-Hopf relationship through the running product, which it does because Hamilton product is an isometry of $S^3$.

This gives us a structural prediction: **any hierarchical closure process on $S^3$ will separate scalar from vector degrees of freedom as a function of energy scale.** This is falsifiable and, to our knowledge, has not been stated in the quaternion dynamics literature.

---

## 5. Discussion

### 5.1 What Works

Three features of the model are robust and, in our assessment, non-trivial:

**Entropy conservation** (Experiment 4) is the strongest result. The firing entropy is identical (to within 1.8% CV) across all force regimes. We did not tune for this. We do not understand why it holds. It is consistent with Verlinde's entropic gravity program [5] in the weak sense that informational quantities are conserved where dynamical quantities are not, but we make no claim of a formal connection. This result is the most likely to survive scrutiny and the most in need of analytical explanation.

**Existence/arrangement separation** (Experiments 6–8) is geometrically necessary given the Hopf shift theorem, but its coupling to hierarchical closure is a novel observation. The fact that slow (gravity-like) dynamics are W-dominant (scalar, "existence") while fast (strong-force-like) dynamics are RGB-dominant (vector, "arrangement/orientation") is suggestive. In the Closure-SDK language [6], W encodes *whether something exists* and RGB encodes *where it is*. The model predicts that these two aspects of physical reality decouple across energy scales—not as a modeling assumption, but as a geometric consequence.

**High-energy convergence** (Experiment 0) reproduces the qualitative feature of GUT-scale unification: forces that are strongly differentiated at low energy become indistinguishable at high energy (CV dropping from 1.37 to 0.02). This is built into the model in the sense that compressing threshold values necessarily compresses firing rates, but the smoothness of the convergence and the quantitative behavior of the CV curve were not guaranteed.

### 5.2 What Fails

The model fails in three important respects:

**No confinement.** The strong force exhibits asymptotic freedom (weakening at short distances) and confinement (strengthening at long distances). Our model, in both the threshold and closure versions, produces neither. Experiment 2's dynamic threshold was a naive attempt that did not work. Confinement likely requires non-Abelian gauge structure that our single update rule does not capture.

**No electromagnetic cancellation.** In nature, electric charges cancel at macroscopic scales—bulk matter is electrically neutral. Our signed-energy model (Experiment 1) produces balanced positive and negative firing events but no mechanism for scale-dependent cancellation. This is arguably the most physically important failure, since the large-scale neutrality of electromagnetism is central to the observed force hierarchy.

**Quantitative dynamic range.** We produce $\sim 3$ orders of magnitude in force ratios. Nature produces $\sim 39$. This is not a rounding error. The model is qualitative at best, and the $85:1$ hierarchy ratio is not close to $10^{39}$.

### 5.3 What It Means

We do not claim this model is correct. We claim it is *interesting* in the following narrow sense: a single update rule (quaternion composition) on a single geometric space ($S^3$) with no force-specific parameters produces:

1. A qualitative force hierarchy
2. High-energy convergence
3. Entropy conservation across regimes
4. A geometric separation of scalar and vector degrees of freedom correlated with energy scale

These four features map onto known physics in ways that are not entirely trivial—but they fall far short of a theory. The model is best understood as a *hypothesis generator*: it identifies structural questions (why is entropy conserved? why does the scalar/vector decomposition correlate with hierarchy?) that can be pursued analytically or in richer models.

### 5.4 Connection to Closure-SDK

The Closure-SDK framework [6] provides the conceptual vocabulary for the $S^3$ experiments. In that framework:

- $\sigma = \arccos(|w|)$ is the energy of a state: its geodesic distance from identity.
- *Closure* is the return to $\sigma = 0$: a completed cycle of excursion and return.
- The Hopf decomposition $W + R + G + B = 1$ separates existence ($W$) from position ($RGB$).
- Three operations suffice for the full dynamics: **compose** ($q_1 \otimes q_2$), **sigma** ($\arccos|w|$), and **close** (detect $\sigma \to 0$).

Our contribution is to show that running these three operations on a lattice, with no additional structure, produces the features described above. The Closure-SDK's minimalism is what makes the results interpretable: there is nowhere for force-specific information to hide.

### 5.5 Connections to Known Physics

We note several structural parallels without claiming causal connections:

- **$\mathrm{SU}(2) \cong S^3$** is the gauge group of the weak nuclear force. We did not choose $S^3$ to match this—we chose it as the unit quaternion group, the richest associative division algebra by Hurwitz's theorem [4]. The coincidence is structural, not designed.
- **Running coupling constants** converging at the GUT scale [3] are qualitatively reproduced by our high-energy convergence (Experiment 0).
- **Verlinde's entropic gravity** [5] argues that gravitational force emerges from entropy gradients. Our entropy conservation across force regimes (Experiment 4) is consonant with this idea, though the connection is loose.
- The **$1+3$ structure** of quaternions (one scalar, three vector components) echoes the $1+3$ structure of spacetime. We do not claim this is more than coincidence, but we note that Hurwitz's theorem makes this decomposition *algebraically necessary* for the richest associative division algebra.

---

## 6. Conclusion

We have presented a toy model in which a single update rule—quaternion composition on $S^3$—produces a qualitative force hierarchy, high-energy convergence, entropy conservation, and a geometric separation of scalar and vector degrees of freedom. The model fails to reproduce confinement, electromagnetic cancellation, or the quantitative dynamic range of the real force hierarchy. It is not a theory of physics.

The three claims we advance, in decreasing order of confidence:

1. **Entropy conservation across force regimes** (CV = 0.018) is unexplained, untuned, and robust across parameters. It is the most likely to be analytically tractable and the most in need of explanation.

2. **Hierarchical closure on $S^3$ necessarily produces existence/arrangement separation.** This follows from the Hopf shift theorem ($\sigma < \pi/4 \Rightarrow W > 1/2$) combined with the observation that Hamilton product preserves the energy-Hopf relationship. It is derivable but, to our knowledge, novel.

3. **The qualitative force hierarchy and high-energy convergence emerge from a single mechanism** with no force-specific parameters. This is the weakest claim because it is the most model-dependent, but it is the one that motivated the work.

The model's value, if any, is as a falsifiable framework. It makes specific predictions: entropy conservation should hold for any threshold-based firing model on any lattice topology; the Hopf shift should hold for any closure process on $S^3$; the hierarchy ratio should be scale-independent. These can be tested, and the model can be killed.

We publish the failures with equal weight as the successes. The goal was never to solve unification with a lattice simulation. The goal was to translate a physical intuition—*one mechanism, different energy scales*—into a computational object that can be examined, criticized, and either extended or discarded.

---

## 7. References

[1] H. Georgi and S. L. Glashow, "Unity of All Elementary-Particle Forces," *Physical Review Letters*, vol. 32, no. 8, pp. 438–441, 1974.

[2] S. Weinberg, "A Model of Leptons," *Physical Review Letters*, vol. 19, no. 21, pp. 1264–1266, 1967.

[3] U. Amaldi, W. de Boer, and H. Fürstenau, "Comparison of grand unified theories with electroweak and strong coupling constants measured at LEP," *Physics Letters B*, vol. 260, no. 3–4, pp. 447–455, 1991.

[4] A. Hurwitz, "Über die Composition der quadratischen Formen von beliebig vielen Variablen," *Nachrichten von der Gesellschaft der Wissenschaften zu Göttingen*, pp. 309–316, 1898.

[5] E. P. Verlinde, "On the Origin of Gravity and the Laws of Newton," *Journal of High Energy Physics*, vol. 2011, no. 4, p. 29, 2011. arXiv:1001.0785.

[6] W. H. A. da Silva, "Closure-SDK: Quaternion substrate for closure dynamics," Open Research Institute, 2025. Available: [github.com/faltz009/Closure-SDK](https://github.com/faltz009/Closure-SDK).

[7] R. Penrose, *The Road to Reality: A Complete Guide to the Laws of the Universe*. London: Jonathan Cape, 2004.

[8] J. C. Baez, "The Octonions," *Bulletin of the American Mathematical Society*, vol. 39, no. 2, pp. 145–205, 2002. arXiv:math/0105155.

---

*Code and data: [github.com/leo-guinan/gut-driven-gut](https://github.com/leo-guinan/gut-driven-gut)*

*Correspondence: Leo Guinan, MetaSPN / Proximity Fund*
