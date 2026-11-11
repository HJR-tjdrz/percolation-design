# -*- coding: utf-8 -*-
"""Statistical estimation helpers: probability, confidence intervals, cost."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .engine import Model, n_for_fraction, simulate

__all__ = ["wilson_interval", "PointEstimate", "estimate", "cost_of", "NM3_PER_UM3"]

NM3_PER_UM3 = 1e9

C_A = 1.05  # CNY / um^3, filler A
C_B = 0.05  # CNY / um^3, filler B


def wilson_interval(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion.

    Preferred over the Wald interval because it does not degenerate when the
    estimate approaches 0 or 1 -- which happens systematically in percolation
    curves at high filler content.
    """
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class PointEstimate:
    phi_rod: float
    phi_sphere: float
    n_rod: int
    n_sphere: int
    k: int
    n_runs: int
    p_hat: float
    ci_low: float
    ci_high: float
    wall_time: float

    @property
    def cost(self) -> float:
        return cost_of(self.n_rod, self.n_sphere)

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["cost"] = self.cost
        return d


def cost_of(n_rod: int, n_sphere: int = 0, model: Optional[Model] = None,
            c_a: float = C_A, c_b: float = C_B) -> float:
    """Total filler cost in CNY for the given particle counts."""
    m = model or Model()
    v_a = n_rod * m.volume_rod / NM3_PER_UM3
    v_b = n_sphere * m.volume_sphere / NM3_PER_UM3
    return v_a * c_a + v_b * c_b


def estimate(model: Model, phi_rod: float = 0.0, phi_sphere: float = 0.0,
             n_runs: int = 500, seed: Optional[int] = None,
             progress: bool = False) -> PointEstimate:
    """Estimate the conduction probability at a given composition."""
    n_rod = n_for_fraction(phi_rod, model, "rod") if phi_rod > 0 else 0
    n_sphere = n_for_fraction(phi_sphere, model, "sphere") if phi_sphere > 0 else 0
    rng = np.random.default_rng(seed)

    t0 = time.perf_counter()
    k = 0
    for i in range(n_runs):
        if simulate(model, n_rod, n_sphere, rng).conducting:
            k += 1
        if progress and (i + 1) % max(1, n_runs // 10) == 0:
            print(f"    {i + 1}/{n_runs}  p={k / (i + 1):.4f}", flush=True)
    wall = time.perf_counter() - t0

    p = k / n_runs
    lo, hi = wilson_interval(k, n_runs)
    return PointEstimate(phi_rod, phi_sphere, n_rod, n_sphere, k, n_runs, p, lo, hi, wall)
