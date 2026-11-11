# -*- coding: utf-8 -*-
"""Threshold extraction from a Monte Carlo percolation curve.

The conduction probability is modelled as a two-parameter logistic

    P(phi) = 1 / (1 + exp(-k (phi - phi_c)))

which is fitted on the logit scale by weighted least squares:

    logit(P_i) = k * phi_i - k * phi_c

with weights ``w_i = n_i p_i (1 - p_i)`` (the inverse variance of the logit,
delta method).  This is a closed-form fit -- no optimiser, no scipy -- and it is
well behaved for the sample sizes used here (n >= 300).  Points with p = 0 or
p = 1 carry no logit information and are handled with the usual continuity
correction (add 0.5 successes / failures) rather than dropped, which keeps the
extrapolation anchored at both ends.

Uncertainty on ``phi_c`` and on the level-set ``phi*(p)`` comes from a
parametric bootstrap: resample ``k_i ~ Binomial(n_i, P_fit(phi_i))``, refit,
repeat.

    python tools/fit.py results/e1b_rod.jsonl --ps 0.5 0.9
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np

__all__ = ["logistic", "fit_logistic", "bootstrap_levels", "load_jsonl"]


def logistic(phi, phi_c, k):
    return 1.0 / (1.0 + np.exp(-k * (phi - phi_c)))


def _wls_fit(phi, k_succ, n):
    """Weighted least squares on the logit scale.  Returns (phi_c, k)."""
    phi = np.asarray(phi, dtype=float)
    k_succ = np.asarray(k_succ, dtype=float)
    n = np.asarray(n, dtype=float)

    # continuity correction keeps 0 and 1 informative
    p = (k_succ + 0.5) / (n + 1.0)
    y = np.log(p / (1.0 - p))
    w = n * p * (1.0 - p)
    w = np.maximum(w, 1e-9)

    sw = w.sum()
    mx = (w * phi).sum() / sw
    my = (w * y).sum() / sw
    sxx = (w * (phi - mx) ** 2).sum()
    if sxx <= 0:
        raise ValueError("degenerate design: all phi identical")
    b = (w * (phi - mx) * (y - my)).sum() / sxx
    a = my - b * mx
    if b <= 0:
        raise ValueError("fit slope is not positive; the curve is not monotone")
    return -a / b, b


def fit_logistic(phi, k_succ, n):
    """Fit and return a dict with phi_c, k, the curve and a goodness-of-fit."""
    phi_c, k = _wls_fit(phi, k_succ, n)
    p_hat = logistic(np.asarray(phi, float), phi_c, k)
    p_obs = np.asarray(k_succ, float) / np.asarray(n, float)
    resid = p_obs - p_hat
    ss_tot = ((p_obs - p_obs.mean()) ** 2).sum()
    r2 = 1.0 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else float("nan")
    return {"phi_c": float(phi_c), "k": float(k), "r2": float(r2),
            "rmse": float(np.sqrt((resid ** 2).mean())),
            "curve": p_hat, "resid": resid}


def level(fit, p):
    """Composition at which the fitted probability equals ``p``."""
    return fit["phi_c"] + math.log(p / (1.0 - p)) / fit["k"]


def bootstrap_levels(phi, k_succ, n, ps=(0.5, 0.9), n_boot=2000, seed=0):
    """Parametric bootstrap CI for phi_c and for each level-set ``phi*(p)``."""
    phi = np.asarray(phi, float)
    k_succ = np.asarray(k_succ, float)
    n = np.asarray(n, float)
    base = fit_logistic(phi, k_succ, n)
    rng = np.random.default_rng(seed)

    draws = {p: [] for p in ps}
    draws["phi_c"] = []
    p_fit = logistic(phi, base["phi_c"], base["k"])
    for _ in range(n_boot):
        k_star = rng.binomial(n.astype(int), p_fit).astype(float)
        try:
            f = fit_logistic(phi, k_star, n)
        except ValueError:
            continue
        draws["phi_c"].append(f["phi_c"])
        for p in ps:
            draws[p].append(level(f, p))

    out = {"fit": {key: base[key] for key in ("phi_c", "k", "r2", "rmse")},
           "levels": {}, "n_boot_ok": len(draws["phi_c"])}
    out["levels"]["phi_c"] = {"value": base["phi_c"],
                              "lo": float(np.percentile(draws["phi_c"], 2.5)),
                              "hi": float(np.percentile(draws["phi_c"], 97.5))}
    for p in ps:
        arr = np.array(draws[p])
        out["levels"][p] = {"value": level(base, p),
                            "lo": float(np.percentile(arr, 2.5)),
                            "hi": float(np.percentile(arr, 97.5)),
                            "boot_sd": float(arr.std())}
    return out


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r.get("phi", 0.0))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--ps", type=float, nargs="+", default=[0.5, 0.9])
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args(argv)

    rows = load_jsonl(args.path)
    phi = [r["phi"] for r in rows]
    ks = [r["k"] for r in rows]
    ns = [r["runs"] for r in rows]
    res = bootstrap_levels(phi, ks, ns, ps=args.ps, n_boot=args.boot)

    print(f"{args.path}  ({len(rows)} points, R2={res['fit']['r2']:.4f}, "
          f"RMSE={res['fit']['rmse']:.4f})")
    print(f"  k        = {res['fit']['k']:.1f}")
    print(f"  phi_c    = {res['levels']['phi_c']['value']:.5f} "
          f"[{res['levels']['phi_c']['lo']:.5f}, {res['levels']['phi_c']['hi']:.5f}]")
    for p in args.ps:
        lv = res["levels"][p]
        print(f"  phi*({p:.2f}) = {lv['value']:.5f} [{lv['lo']:.5f}, {lv['hi']:.5f}]"
              f"   (boot sd {lv['boot_sd']:.5f})")
    print(f"  bootstrap draws kept: {res['n_boot_ok']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
