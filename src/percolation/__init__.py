# -*- coding: utf-8 -*-
"""percolation -- a small, verified Monte Carlo engine for through-thickness
percolation in a thin micro-cell.

Quick start
-----------
>>> from percolation import Model, simulate, estimate
>>> m = Model()                                  # 10 um cell, 5 um rods, 400 nm spheres
>>> res = simulate(m, n_rod=608)                 # one realisation
>>> res.conducting
True
>>> est = estimate(m, phi_rod=0.0086, n_runs=200, seed=0)
>>> round(est.p_hat, 3)
0.9...

Command line
------------
    python -m percolation.run --config cfg/howto.json --out results/demo.json
"""

from .engine import Model, SimulationResult, n_for_fraction, simulate
from .estimate import PointEstimate, cost_of, estimate, wilson_interval

__version__ = "0.1.0"

__all__ = [
    "Model",
    "SimulationResult",
    "simulate",
    "n_for_fraction",
    "estimate",
    "wilson_interval",
    "cost_of",
    "PointEstimate",
    "__version__",
]
