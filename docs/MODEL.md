# Model description

This is the model behind the paper. It is written to be read alongside `src/percolation/`, and each
section names the function that implements it.

---

## 1. Problem

A cell must conduct between two electrodes with probability at least *p*. Two fillers are available:
A is a high-aspect-ratio spherocylinder, B a sphere. A is expensive per unit volume, B cheap.
Minimise material cost subject to the reliability constraint:

```
minimise   C(v_A, v_B) = c_A v_A + c_B v_B
subject to P(v_A, v_B) >= p
```

`P` has no closed form, so it is estimated by Monte Carlo. The whole design of the code follows from
that: an engine fast enough to be sampled thousands of times, and a way to turn the sampled
probability into something an optimiser can use.

## 2. Geometry

Cube of side `L = 10 µm`, faces at `x = ±L/2` are electrodes. Filler A: spherocylinder, axis 5 µm,
radius 30 nm, so `L/D = 83`. Filler B: sphere, radius 200 nm. Centres uniform in the cell,
orientations isotropic.

Isotropy matters and is easy to get wrong. Sampling `θ ~ U(0,π)` and `φ ~ U(0,2π)` biases the
direction vector towards the poles. The correct construction uses the Archimedes projection,

```
cos θ = 1 - 2 U1
φ     = 2π U2
u     = (sin θ cos φ, sin θ sin φ, cos θ)
```

which is checked in the tests against `E|u_x| = 1/2` and `E u_x² = 1/3` over 40 000 draws.
See `geometry.py: rod_endpoints`.

## 3. Contact criterion

Two particles are connected when their **surface** separation is at most `δ = 1.8 nm`, the tunnelling
cut-off. With the capsule (spherocylinder) approximation the three cases are

```
d_AA = max(0, d_seg(axis_i, axis_j) - 2 r_A)
d_AS = max(0, d_pt(c_j, axis_i) - r_A - r_B)
d_SS = max(0, ||c_i - c_j|| - 2 r_B)
```

implemented in `geometry.py`. `d_seg` is a robust segment–segment distance: solve for the closest
points on the two infinite lines, clamp to the segments, and fall back to point–segment distance at
the ends. The capsule approximation ignores the flat end caps; with `δ/r_A = 0.06` the neglected solid
angle is small.

A rod touches an electrode when its axis comes within `r_A + δ` of that plane. The cell conducts when
a path of contacts joins the two electrodes.

## 4. Boundary treatment

Three conventions exist in practice and only two of them are defensible.

| Convention | Lateral faces | Electrode faces | Notes |
|---|---|---|---|
| `periodic` | minimum image | clipped | reference calculation |
| `walls` | reflecting | clipped | isolated patch |
| `mirror` | minimum image | **mirrored to the opposite electrode** | degenerate, do not use |

Periodic laterals are the representative-volume choice: the cell stands for one period of a film much
larger than the particle. The electrode faces are not periodic, because a filler that crosses an
electrode is inside the metal.

The `mirror` convention cuts the overhang at the electrode and translates it to the opposite face,
where it becomes a contact on the far electrode. A single rod that touches one electrode then touches
both, and *every* realisation conducts. It is kept in the code because it is worth being able to
demonstrate the degeneracy (Table II of the paper), not because it is a model of anything.

Measured effect: the 90 % rod threshold is 0.755 vol % with periodic laterals and 0.877 vol % with
insulating ones, a 16 % relative shift. The decision is unchanged. See `engine.py: _lateral_shift`.

## 5. Estimator

`P` is estimated from `K` independent realisations, `P̂ = K⁻¹ Σ 1[conducts]`. Intervals are Wilson
score intervals, not Wald: at `P̂ → 1` the Wald interval collapses and its upper bound is clipped,
which is exactly where percolation curves spend their range. See `estimate.py: wilson_interval`.

Cost of one realisation is 71 ms for 424 rods on one core. The implementation sorts particles by `x`
and only forms candidate pairs inside a sliding window, with `searchsorted` and blocked vectorised
distance evaluation (`engine.py: _pairs_by_x_window`). This is about 7× faster than the naive
double loop but is **still O(N²)** in the number of particles; the paper says so, and no linear-scaling
claim is made.

## 6. Thresholds

A two-parameter logistic is fitted on the logit scale by weighted least squares, with weights
`n p (1-p)`:

```
P(φ) = [1 + exp(-k(φ - φ_c))]⁻¹
φ*(p) = φ_c + logit(p)/k
```

Uncertainty comes from a parametric bootstrap that resamples `k_i ~ Binomial(n_i, P_fit(φ_i))` and
refits. Because the bootstrap resamples from the fitted model, it understates uncertainty when the
model is misspecified; for the sphere curve, whose transition is steeper than the logistic can follow,
the paper reports the bracket between a model-free interpolation (35.0 vol %) and the fit (35.4 vol %)
rather than the bootstrap interval alone. See `tools/fit.py`.

## 7. Surrogate constraint

The probabilistic constraint is replaced by a linear inequality normalised **at the target
probability**:

```
v_A / v_A*(0.9) + v_B / v_B*(0.9) + λ_s (v_A/v_A*(0.9))(v_B/v_B*(0.9)) >= 1
```

Normalising at `p = 0.9` rather than at the percolation threshold is the point. The classical form
divides by the 50 % threshold and multiplies by a safety factor; that works for a single filler,
where the factor is just the inverse of the transition width, `η_A = φ_A*(0.9)/φ_A*(0.5) = 1.24`.
It fails when transferred to another filler, because the transition width is a property of the filler
and not of the mixture: reusing `η_A` for the spheres would demand 41.6 vol % against a measured
35.4 vol %, an 18 % overstatement in the direction that penalises the cheap filler.

`λ_s` is zero under linear superposition. It is **not** fitted here: the two-dimensional sweep resolves
its sign but not its value, and the implied magnitudes differ between compositions, which says the
cross term saturates. The design rule uses `λ_s = 0` and treats it as conservative.

## 8. Cost criterion

With unit prices `c_A, c_B` the optimum of the linear programme is a corner, and which corner is
decided by one dimensionless number:

```
Λ = c_A v_A*(p) / (c_B v_B*(p))
```

`Λ < 1` selects pure A, `Λ > 1` pure B. Equivalent and more usable as a specification:

```
v_B*(p) < (c_A/c_B) v_A*(p)
```

Note that the unit prices in the design brief are relative cost units; every conclusion depends on
their ratio, not their absolute value.

## 9. Known limitations

1. **Connectivity, not conductivity.** Contact resistance and tunnelling nonlinearity are outside the
   model. The code predicts whether a network exists, not how well it conducts.
2. **Soft cores.** Overlap is allowed, which understates thresholds at high loading. The sphere
   threshold of 35 vol % is not a low loading. The error direction is favourable to the paper's
   conclusion, but it is not quantified.
3. **Capsule approximation.** End caps are neglected; the neglected solid angle is small at
   `δ/r_A = 0.06`.
4. **Finite thickness.** Rod finite-size effects are measured (the 90 % level moves 2.3 % between 1.2
   and 2.0 rod lengths) but not extrapolated to the bulk limit.
5. **One size ratio, one price ratio.** The design map is a slice, not a picture.

## 10. Code map

| Concept | File | Function |
|---|---|---|
| Distance kernels | `geometry.py` | `segment_segment_distance`, `point_segment_distance` |
| Rod endpoints from an isotropic direction | `geometry.py` | `rod_endpoints` |
| Model parameters, boundary conventions | `engine.py` | `Model` |
| Lateral minimum image | `engine.py` | `_lateral_shift` |
| Electrode contacts | `engine.py` | `_rod_face_contacts`, `_sphere_face_contacts` |
| Candidate pairs | `engine.py` | `_pairs_by_x_window`, `_blocked_pairs` |
| Connectivity | `engine.py` | `_DSU` |
| Single realisation | `engine.py` | `simulate` |
| Probability estimate and interval | `estimate.py` | `estimate`, `wilson_interval` |
| Logistic fit, level set, bootstrap | `tools/fit.py` | `fit_logistic`, `bootstrap_levels` |
| Aggregated numbers | `tools/summarize.py` | writes `results/numbers.json` |
| Figures | `tools/make_figures.py` | `fig1` … `fig5` |
