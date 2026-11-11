# Reproducing the paper

Two levels of reproduction are supported.

**Level 1 — from the stored data (seconds).** Every number and figure in the paper is regenerated from
`results/` and `data/supplement/`. No simulation is run. This is enough to check the manuscript.

**Level 2 — from scratch (about 1.5 CPU-hours).** Delete `results/` and re-run the drivers below.

---

## Level 1: regenerate numbers and figures

```bash
python tools/summarize.py                 # writes results/numbers.json and results/numbers.md
python tools/make_figures.py              # writes figures/fig1..fig5 (pdf, svg, png)
python tools/make_supplement_figures.py   # writes figures/figB1..figB3
python tools/check_numbers.py             # cross-checks the manuscript text against numbers.json
```

`results/numbers.md` is the authoritative list of numbers quoted in the paper. If a number in the
manuscript is not in that file, it should not be in the manuscript.

## Which script produces which claim

| Paper item | Script | Input | Runtime (level 1) |
|---|---|---|---|
| Table I (parameters) | — | constants in `src/percolation/engine.py` | — |
| Table II (boundary sensitivity) | `tools/effect_of_boundary.py` | `results/e2.jsonl` | 1 s |
| Table III (contour compositions) | `tools/summarize.py` | `results/e5.jsonl` | 1 s |
| Fig. 2 (thickness dependence) | `tools/make_figures.py fig2` | `results/e1a.jsonl` + `data/supplement/B1_rod_thickness/` | 5 s |
| Fig. 3 (percolation curves) | `tools/make_figures.py fig3` | `results/e1b_rod.jsonl`, `e1b_sphere.jsonl` | 2 s |
| Fig. 4 (composition sweep) | `tools/make_figures.py fig4` | `results/e4.jsonl` | 2 s |
| Fig. 5 (design map) | `tools/make_figures.py fig5` | `results/numbers.json` | 1 s |
| Rod 90 % threshold 0.753 vol % | `tools/summarize.py` | `results/e1b_rod.jsonl` | 1 s |
| Sphere 90 % threshold 35.4 vol % | `tools/summarize.py` | `results/e1b_sphere.jsonl` | 1 s |
| Boundary-invariance of the decision | `tools/summarize.py` | `results/e2.jsonl` | 1 s |
| Rod thickness (2.3 % vs 22 %) | `tools/fit_b1.py` | `data/supplement/B1_rod_thickness/` | 2 s |
| Tunnelling cut-off (2.1 %) | `tools/fit_b23.py` | `data/supplement/B2_delta_sensitivity/` | 2 s |
| Sphere boundary (4.4 %) | `tools/fit_b23.py` | `data/supplement/B3_sphere_boundary/` | 2 s |

## Level 2: rerun the simulations

Each driver writes append-only JSONL and accepts `--resume`, so an interrupted run continues where it
stopped. Seeds are fixed per driver and recorded in every result row.

```bash
# E1b: the two pure-filler curves (the core measurement)
python tools/run_experiments.py e1b-rod    --runs 1500
python tools/run_experiments.py e1b-sphere --runs 200

# E2: boundary-condition comparison, rods
python tools/run_experiments.py e2 --runs 400

# E1a: sphere threshold vs cell thickness
python tools/run_experiments.py e1a --runs 150

# E3, E4: pure-A curve for the design map, and the 5x5 composition sweep
python tools/run_experiments.py e3
python tools/run_experiments.py e4 --runs 120

# E5: three compositions placed on the rule's contour, high precision
python tools/run_experiments.py e5 --runs 2000 --abs 0.00753 0.0 0.00650 0.0566 0.00373 0.2070

# B1-B3: the reviewer-requested supplements
python tools/run_supplement.py b1 --runs 400     # rod thickness       (~3 min)
python tools/run_supplement.py b2 --runs 400     # tunnelling cut-off  (~7 min)
python tools/run_supplement.py b3 --runs 200     # sphere boundary     (~20 min)
```

Total wall time on one CPU core: **about 1.5 hours**, dominated by `e1b-rod` (26 min) and `b3` (20 min).

## Cost model

One realisation costs 71 ms at `N = 424` rods. The estimator's cost is therefore roughly
`K × 71 ms` per composition, which is why the paper works with 1 500 realisations on the rod curve
(about two minutes per point) and only 150–200 on the sphere curve, where each realisation is far
more expensive because there are 10 000 particles.

## Determinism

* `Model` contains only geometry and boundary conventions; no hidden state.
* `estimate(..., seed=n)` seeds a `numpy.random.default_rng` internally, so results are reproducible
  independent of call order.
* The logistic fit and the bootstrap are deterministic given the data and a seed.

## Environment used for the paper

```
Python 3.9
numpy 2.0.2
scipy 1.9.1
matplotlib 3.9.4
```

Single core, no GPU, no CUDA. The code contains no GPU path at all.
