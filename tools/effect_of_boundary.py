# -*- coding: utf-8 -*-
"""Decompose the boundary-treatment effects on the measured P(phi).

Three conventions are compared at the same random seeds:

1. ``legacy``   -- the competition code: rods are cut at every face and the
   outside part is translated to the opposite face, where it becomes a *separate*
   node.  Two consequences: material exiting an electrode reappears on the
   opposite electrode, and material exiting a lateral face is disconnected from
   its own rod.
2. ``walls``    -- insulating lateral faces, electrode clipping.
3. ``periodic`` -- periodic lateral faces (minimum image), electrode clipping.

(2) and (3) differ only in the lateral treatment, which isolates the cost of the
fragment-breaking behaviour in (1).

    python tools/effect_of_boundary.py --runs 150
"""

import argparse
import math
import sys
import time

import _compat
import numpy as np

COMPARISON = [
    (0.005, 0.075, "paper: 0.098 (49/500) / older run 0.084"),
    (0.006, 0.218, "paper: 0.218"),
    (0.007, 0.490, "paper: 0.490 / older run 0.440"),
    (0.010, 0.998, "paper: 0.998"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--phie", type=float, nargs="+", default=[0.005, 0.006, 0.007, 0.010])
    args = ap.parse_args()

    from percolation import Model, n_for_fraction, simulate

    print(f"runs={args.runs} seed={args.seed}\n")
    print(f"{'phi':>7} {'N_A':>5} {'legacy':>16} {'walls':>16} {'periodic':>16} {'paper':>8}")
    print("-" * 78)

    for phi in args.phie:
        row = []
        for policy in ("legacy", "walls", "periodic"):
            if policy == "legacy":
                old = _compat.load("problem2")
                n = n_for_fraction(phi, Model(), "rod")
                import random as _random
                _random.seed(args.seed)
                t0 = time.time()
                k = 0
                for _ in range(args.runs):
                    k += old.single_simulation(n)[0]
                dt = time.time() - t0
            else:
                m = Model(lateral="periodic" if policy == "periodic" else "walls")
                n = n_for_fraction(phi, m, "rod")
                rng = np.random.default_rng(args.seed)
                t0 = time.time()
                k = 0
                for _ in range(args.runs):
                    k += int(simulate(m, n, 0, rng).conducting)
                dt = time.time() - t0
            p = k / args.runs
            se = math.sqrt(p * (1 - p) / args.runs) if 0 < p < 1 else 0.0
            row.append(f"{p:.3f}+/-{1.96*se:.3f}")
            if policy == "legacy":
                ref = {0.005: 0.098, 0.006: 0.218, 0.007: 0.490, 0.010: 0.998}.get(phi, float("nan"))
        print(f"{phi*100:6.2f}% {n:5d} {row[0]:>16} {row[1]:>16} {row[2]:>16} {ref:8.3f}")

    print("\nReading:")
    print("  * 'periodic' is the physically consistent default of the new engine.")
    print("  * 'walls' isolates the lateral treatment: it still differs from 'legacy'")
    print("    because the legacy code disconnects a rod that crosses a lateral face")
    print("    from itself, which shortens roughly a quarter of all rods.")
    print("  * 'legacy' additionally mirrors material across the electrodes.")


if __name__ == "__main__":
    sys.exit(main())
