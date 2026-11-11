# -*- coding: utf-8 -*-
"""Supplement experiments requested by the reviewers (B1 and B3).

    python tools/run_supplement.py b1 --runs 400
    python tools/run_supplement.py b3 --runs 200

Writes to ``project_analyze/11.7论文第一次自审及修改/11.7论文修改时补充的内容比如实验图片等等/``
so that tonight's data does not mix with the original ``results/`` set that the
submitted manuscript quotes.  Each point is flushed to disk immediately, so the
job can be interrupted and resumed with ``--resume``.

B1  rod thickness dependence   (reviewer P0-4: the main result uses rods but the
     thickness study was spheres only)
B3  sphere boundary sensitivity (reviewer P0-2: only rods were tested under both
     lateral conventions)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from percolation import Model, estimate  # noqa: E402

OUT_ROOT = os.path.join(ROOT, "data", "supplement")


def _dir(sub: str) -> str:
    d = os.path.join(OUT_ROOT, sub)
    os.makedirs(d, exist_ok=True)
    return d


def _load_done(path: str) -> dict:
    done = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done[rec["key"]] = rec
    return done


def _append(path: str, rec: dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


# ----------------------------------------------------------------------
def job_b1(args):
    """Rod threshold versus cell thickness."""
    d = _dir("B1_rod_thickness")
    path = os.path.join(d, "b1_rod_thickness.jsonl")
    done = _load_done(path) if args.resume else {}
    sizes = args.sizes or [6000.0, 7500.0, 10000.0]
    phis = args.phis or [round(0.0040 + 0.0010 * i, 4) for i in range(8)]
    print(f"B1  sizes={sizes}  phis={phis}  K={args.runs}")
    t_all = time.time()
    for L in sizes:
        m = Model(L=L, lateral="periodic")          # rod_length stays 5000 nm
        for phi in phis:
            key = f"L{L:.0f}_p{phi:.4f}"
            if key in done:
                print(f"  L={L:6.0f}  phi={phi*100:.2f}%  (cached) p={done[key]['p']:.3f}")
                continue
            est = estimate(m, phi_rod=phi, n_runs=args.runs, seed=args.seed)
            rec = dict(job="b1", key=key, L=L, phi=phi, n=est.n_rod, k=est.k,
                       runs=args.runs, p=est.p_hat, lo=est.ci_low, hi=est.ci_high,
                       seconds=est.wall_time, L_over_rod=L / 5000.0,
                       timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
            _append(path, rec)
            print(f"  L={L:6.0f} ({L/5000:4.1f} rod)  phi={phi*100:.2f}%  "
                  f"N={est.n_rod:4d}  p={est.p_hat:.3f}  "
                  f"{est.wall_time/args.runs*1000:.0f} ms/sim", flush=True)
    print(f"-> {path}   ({time.time()-t_all:.0f} s)")
    return path


def job_b3(args):
    """Sphere threshold under both lateral conventions."""
    d = _dir("B3_sphere_boundary")
    path = os.path.join(d, "b3_sphere_boundary.jsonl")
    done = _load_done(path) if args.resume else {}
    phis = args.phis or [0.30, 0.32, 0.34, 0.36, 0.38]
    conventions = ["periodic", "walls"]
    print(f"B3  phis={phis}  conventions={conventions}  K={args.runs}")
    t_all = time.time()
    for lat in conventions:
        m = Model(lateral=lat)
        for phi in phis:
            key = f"{lat}_p{phi:.4f}"
            if key in done:
                continue
            est = estimate(m, phi_sphere=phi, n_runs=args.runs, seed=args.seed)
            rec = dict(job="b3", key=key, lateral=lat, phi=phi, n=est.n_sphere,
                       k=est.k, runs=args.runs, p=est.p_hat, lo=est.ci_low,
                       hi=est.ci_high, seconds=est.wall_time,
                       timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
            _append(path, rec)
            print(f"  {lat:8s}  phi={phi*100:.2f}%  n={est.n_sphere:5d}  "
                  f"p={est.p_hat:.3f}  {est.wall_time/args.runs:.2f} s/sim", flush=True)
    print(f"-> {path}   ({time.time()-t_all:.0f} s)")
    return path


def job_b2(args):
    """Sensitivity of the rod threshold to the tunnelling cut-off delta."""
    d = _dir("B2_delta_sensitivity")
    path = os.path.join(d, "b2_delta.jsonl")
    done = _load_done(path) if args.resume else {}
    deltas = args.deltas or [1.0, 1.8, 3.0]
    phis = args.phis or [round(0.0040 + 0.0010 * i, 4) for i in range(8)]
    print(f"B2  deltas={deltas}  phis={phis}  K={args.runs}")
    t_all = time.time()
    for delta in deltas:
        m = Model(delta=delta, lateral="periodic")
        for phi in phis:
            key = f"d{delta:.1f}_p{phi:.4f}"
            if key in done:
                continue
            est = estimate(m, phi_rod=phi, n_runs=args.runs, seed=args.seed)
            rec = dict(job="b2", key=key, delta=delta, phi=phi, n=est.n_rod,
                       k=est.k, runs=args.runs, p=est.p_hat, lo=est.ci_low,
                       hi=est.ci_high, seconds=est.wall_time,
                       timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
            _append(path, rec)
            print(f"  delta={delta:.1f} nm  phi={phi*100:.2f}%  N={est.n_rod:4d}  "
                  f"p={est.p_hat:.3f}  {est.wall_time/args.runs*1000:.0f} ms/sim",
                  flush=True)
    print(f"-> {path}   ({time.time()-t_all:.0f} s)")
    return path


JOBS = {"b1": job_b1, "b2": job_b2, "b3": job_b3}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job", choices=sorted(JOBS))
    ap.add_argument("--runs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--phis", type=float, nargs="+", default=None)
    ap.add_argument("--sizes", type=float, nargs="+", default=None)
    ap.add_argument("--deltas", type=float, nargs="+", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    JOBS[args.job](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
