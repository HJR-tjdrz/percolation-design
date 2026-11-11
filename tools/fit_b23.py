# -*- coding: utf-8 -*-
"""Fit the B2 (delta sensitivity) and B3 (sphere boundary) supplement data.

    python tools/fit_b23.py [data/supplement]

Prints the 50 % and 90 % level sets for each tunnelling cut-off and for each
lateral boundary convention.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import fit as fm  # noqa: E402

SUP = os.path.join(ROOT, "data", "supplement")


def _fit_block(rows, group_key):
    out = {}
    for g in sorted({r[group_key] for r in rows}):
        sub = sorted([r for r in rows if r[group_key] == g], key=lambda r: r["phi"])
        phi = np.array([r["phi"] for r in sub])
        k = np.array([r["k"] for r in sub])
        n = np.array([r["runs"] for r in sub])
        res = fm.bootstrap_levels(phi, k, n, ps=(0.5, 0.9), n_boot=600)
        a, b = res["levels"][0.5], res["levels"][0.9]
        out[g] = (a, b)
        print("  %-10s phi50=%.4f%% [%.4f,%.4f]   phi90=%.4f%% [%.4f,%.4f]"
              % (g, a["value"] * 100, a["lo"] * 100, a["hi"] * 100,
                 b["value"] * 100, b["lo"] * 100, b["hi"] * 100))
    return out


def main():
    sup = sys.argv[1] if len(sys.argv) > 1 else SUP

    b2 = fm.load_jsonl(os.path.join(sup, "B2_delta_sensitivity", "b2_delta.jsonl"))
    print("B2  delta sensitivity (rods):")
    r2 = _fit_block(b2, "delta")
    v = [r2[d][1]["value"] * 100 for d in sorted(r2)]
    print("  90%% level across delta 1-3 nm: %.3f -> %.3f  (%.1f%% relative)"
          % (v[0], v[-1], (v[-1] / v[0] - 1) * 100))

    b3 = fm.load_jsonl(os.path.join(sup, "B3_sphere_boundary", "b3_sphere_boundary.jsonl"))
    print("B3  sphere boundary:")
    r3 = _fit_block(b3, "lateral")
    v3 = {g: r3[g][1]["value"] * 100 for g in r3}
    print("  periodic %.3f  vs  walls %.3f  ->  %+.1f%% relative"
          % (v3["periodic"], v3["walls"], (v3["walls"] / v3["periodic"] - 1) * 100))
    return 0


if __name__ == "__main__":
    sys.exit(main())
