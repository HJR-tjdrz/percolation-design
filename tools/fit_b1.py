# -*- coding: utf-8 -*-
"""Fit the B1 rod-thickness data and print the level sets with bootstrap intervals.

    python tools/fit_b1.py [data/supplement/B1_rod_thickness/b1_rod_thickness.jsonl]
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import fit as fm  # noqa: E402

DEFAULT = os.path.join(ROOT, "data", "supplement", "B1_rod_thickness",
                       "b1_rod_thickness.jsonl")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    rows = fm.load_jsonl(path)
    sizes = sorted({r["L"] for r in rows})
    out = {}
    for L in sizes:
        sub = sorted([r for r in rows if r["L"] == L], key=lambda r: r["phi"])
        phi = np.array([r["phi"] for r in sub])
        k = np.array([r["k"] for r in sub])
        n = np.array([r["runs"] for r in sub])
        res = fm.bootstrap_levels(phi, k, n, ps=(0.5, 0.9), n_boot=600)
        a, b = res["levels"][0.5], res["levels"][0.9]
        out[L] = (a, b)
        print("L=%5.0f  phi50=%.4f%% [%.4f, %.4f]   phi90=%.4f%% [%.4f, %.4f]"
              % (L, a["value"] * 100, a["lo"] * 100, a["hi"] * 100,
                 b["value"] * 100, b["lo"] * 100, b["hi"] * 100))
    v = [out[L][1]["value"] * 100 for L in sizes]
    h = [out[L][0]["value"] * 100 for L in sizes]
    print("90%% level: %.3f -> %.3f  (%.1f%% relative)" % (v[0], v[-1], (v[-1] / v[0] - 1) * 100))
    print("50%% level: %.3f -> %.3f  (%.1f%% relative)" % (h[0], h[-1], (h[-1] / h[0] - 1) * 100))
    return 0


if __name__ == "__main__":
    sys.exit(main())
