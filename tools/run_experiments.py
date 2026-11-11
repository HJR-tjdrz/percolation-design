# -*- coding: utf-8 -*-
"""Experiment runner: E1a / E1b / E2 / E3.

Every job appends one JSON record per composition to ``results/<job>.jsonl``
immediately after finishing that point, so an interrupted run loses at most one
point and can be resumed with ``--resume``.

    python tools/run_experiments.py e1b-rod    --runs 2000
    python tools/run_experiments.py e1b-sphere --runs 500
    python tools/run_experiments.py e1a        --runs 200
    python tools/run_experiments.py e2         --runs 500
    python tools/run_experiments.py e3

All times are wall-clock seconds on the machine that ran the job; they are
recorded in the output so the paper can quote measured throughput.
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

from percolation import Model, estimate, n_for_fraction, simulate  # noqa: E402

RESULTS = os.path.join(ROOT, "results")


def _out_path(name: str) -> str:
    os.makedirs(RESULTS, exist_ok=True)
    return os.path.join(RESULTS, name + ".jsonl")


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


def _record(job, key, **kwargs):
    rec = dict(job=job, key=key, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"), **kwargs)
    return rec


# ---------------------------------------------------------------------------
def job_e1b_rod(args):
    """Pure-rod percolation curve in the device geometry."""
    m = Model(lateral="periodic")
    path = _out_path("e1b_rod")
    done = _load_done(path) if args.resume else {}
    phis = args.phis or [round(0.0040 + 0.0005 * i, 4) for i in range(15)]
    print(f"E1b-rod  {len(phis)} points x K={args.runs}  lateral={m.lateral}")
    for phi in phis:
        key = f"{phi:.4f}"
        if key in done:
            print(f"  {phi*100:.2f}%  (cached) p={done[key]['p']:.3f}")
            continue
        est = estimate(m, phi_rod=phi, n_runs=args.runs, seed=args.seed)
        rec = _record("e1b_rod", key, phi=phi, n=est.n_rod, k=est.k, runs=args.runs,
                      p=est.p_hat, lo=est.ci_low, hi=est.ci_high,
                      cost=est.cost, seconds=est.wall_time,
                      model={"L": m.L, "lateral": m.lateral, "rod_x": m.rod_x})
        _append(path, rec)
        print(f"  {phi*100:.2f}%  N={est.n_rod:4d}  p={est.p_hat:.4f} "
              f"[{est.ci_low:.3f},{est.ci_high:.3f}]  {est.wall_time/args.runs*1000:.0f} ms/sim",
              flush=True)
    return path


def job_e1b_sphere(args):
    """Pure-sphere percolation curve in the device geometry."""
    m = Model(lateral="periodic")
    path = _out_path("e1b_sphere")
    done = _load_done(path) if args.resume else {}
    phis = args.phis or [round(0.20 + 0.02 * i, 3) for i in range(14)]
    print(f"E1b-sphere  {len(phis)} points x K={args.runs}")
    for phi in phis:
        key = f"{phi:.4f}"
        if key in done:
            print(f"  {phi*100:.1f}%  (cached) p={done[key]['p']:.3f}")
            continue
        est = estimate(m, phi_sphere=phi, n_runs=args.runs, seed=args.seed)
        rec = _record("e1b_sphere", key, phi=phi, n=est.n_sphere, k=est.k, runs=args.runs,
                      p=est.p_hat, lo=est.ci_low, hi=est.ci_high,
                      cost=est.cost, seconds=est.wall_time,
                      model={"L": m.L, "lateral": m.lateral})
        _append(path, rec)
        print(f"  {phi*100:.1f}%  n={est.n_sphere:6d}  p={est.p_hat:.4f} "
              f"[{est.ci_low:.3f},{est.ci_high:.3f}]  {est.wall_time/args.runs:.2f} s/sim",
              flush=True)
    return path


def job_e1a(args):
    """Finite-size study: how the measured threshold moves with cell size.

    The cell has conducting plates on the two x faces and periodic laterals, so
    the estimator is the spanning probability.  As L grows the measured
    threshold should approach the bulk continuum value.
    """
    path = _out_path("e1a")
    done = _load_done(path) if args.resume else {}
    sizes = args.sizes or [3000.0, 4500.0, 6000.0, 8000.0, 10000.0]
    print(f"E1a  sizes={sizes}  K={args.runs}")
    for L in sizes:
        # sphere-only study: the rod length is irrelevant but must obey L
        m = Model(L=L, rod_length=min(5000.0, 0.5 * L), lateral="periodic")
        # same reduced density grid for every size
        for phi in (0.20, 0.24, 0.28, 0.32, 0.36, 0.40):
            key = f"L{L:.0f}_s{phi:.2f}"
            if key in done:
                continue
            est = estimate(m, phi_sphere=phi, n_runs=args.runs, seed=args.seed)
            rec = _record("e1a", key, L=L, phi=phi, n=est.n_sphere, k=est.k,
                          runs=args.runs, p=est.p_hat, lo=est.ci_low, hi=est.ci_high,
                          seconds=est.wall_time, L_over_sigma=L / 400.0)
            _append(path, rec)
            print(f"  L={L:5.0f} ({L/400:4.1f} sigma)  phi={phi:.2f}  n={est.n_sphere:5d}  "
                  f"p={est.p_hat:.3f}  {est.wall_time/args.runs:.2f} s/sim", flush=True)
    return path


def job_e2(args):
    """Boundary-condition sensitivity at fixed compositions."""
    path = _out_path("e2")
    done = _load_done(path) if args.resume else {}
    phis = args.phis or [0.005, 0.006, 0.007, 0.008, 0.009]
    combos = [("periodic", "clip"), ("walls", "clip"), ("periodic", "wrap")]
    print(f"E2  {len(phis)} x {len(combos)} conventions  K={args.runs}")
    for lateral, rod_x in combos:
        m = Model(lateral=lateral, rod_x=rod_x)
        for phi in phis:
            key = f"{lateral}_{rod_x}_{phi:.4f}"
            if key in done:
                continue
            est = estimate(m, phi_rod=phi, n_runs=args.runs, seed=args.seed)
            rec = _record("e2", key, lateral=lateral, rod_x=rod_x, phi=phi,
                          n=est.n_rod, k=est.k, runs=args.runs, p=est.p_hat,
                          lo=est.ci_low, hi=est.ci_high, seconds=est.wall_time)
            _append(path, rec)
            print(f"  {lateral:8s} rod_x={rod_x:4s}  phi={phi*100:.1f}%  p={est.p_hat:.3f}",
                  flush=True)
    return path


def job_e3(args):
    """Throughput versus particle count, new engine against the legacy one."""
    path = os.path.join(RESULTS, "e3.json")
    m = Model(lateral="periodic")
    sizes = [100, 200, 400, 600, 800, 1200, 1600]
    rows = []
    for n in sizes:
        reps = max(3, min(20, int(2000 / n)))
        rng = np.random.default_rng(99)
        t0 = time.perf_counter()
        for _ in range(reps):
            simulate(m, n, 0, rng)
        dt = (time.perf_counter() - t0) / reps
        rows.append({"n_rod": n, "seconds_per_sim": dt, "reps": reps, "impl": "new"})
        print(f"  new  N={n:5d}  {dt*1000:8.2f} ms/sim", flush=True)

    # legacy implementation, same particle counts
    sys.path.insert(0, HERE)
    import _compat
    old = _compat.load("problem2")
    import random as _random
    for n in [100, 200, 400, 600, 800]:
        reps = max(3, min(10, int(800 / n)))
        _random.seed(99)
        t0 = time.perf_counter()
        for _ in range(reps):
            old.single_simulation(n)
        dt = (time.perf_counter() - t0) / reps
        rows.append({"n_rod": n, "seconds_per_sim": dt, "reps": reps, "impl": "legacy"})
        print(f"  old  N={n:5d}  {dt*1000:8.2f} ms/sim", flush=True)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "model": {"L": m.L, "lateral": m.lateral}}, fh, indent=1)
    return path


# ---------------------------------------------------------------------------
def _thresholds():
    """Read the calibrated thresholds written by summarize.py, with fallbacks."""
    path = os.path.join(RESULTS, "numbers.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def job_e4(args):
    """Two-dimensional composition sweep: the data behind Fig. 4.

    A 5x5 grid on multiples of the two measured thresholds, plus the two edges
    so that the pure-filler curves are included.  K is deliberately lower than
    in E1b: this sweep is for locating the 90% level set, and the boundary is
    refined afterwards in E5.
    """
    th = _thresholds()
    a90 = th.get("phi_A90", 0.00745)
    b90 = th.get("phi_B90", 0.40)
    fa = args.fa or [0.0, 0.25, 0.5, 0.75, 1.0]
    fb = args.fb or [0.0, 0.25, 0.5, 0.75, 1.0]

    path = _out_path("e4")
    done = _load_done(path) if args.resume else {}
    m = Model(lateral="periodic")
    print(f"E4  {len(fa)}x{len(fb)} grid  K={args.runs}  "
          f"phi_A90={a90:.5f}  phi_B90={b90:.3f}")
    for xa in fa:
        for xb in fb:
            if xa == 0 and xb == 0:
                continue
            pa, pb = xa * a90, xb * b90
            key = f"a{pa:.5f}_b{pb:.4f}"
            if key in done:
                continue
            est = estimate(m, phi_rod=pa, phi_sphere=pb, n_runs=args.runs, seed=args.seed)
            rec = _record("e4", key, phi_a=pa, phi_b=pb, fa=xa, fb=xb,
                          n_a=est.n_rod, n_b=est.n_sphere, k=est.k, runs=args.runs,
                          p=est.p_hat, lo=est.ci_low, hi=est.ci_high,
                          cost=est.cost, seconds=est.wall_time)
            _append(path, rec)
            print(f"  fA={xa:.2f} fB={xb:.2f}  N=({est.n_rod:4d},{est.n_sphere:5d})  "
                  f"p={est.p_hat:.3f}  cost={est.cost:6.2f}  "
                  f"{est.wall_time/args.runs:.2f} s/sim", flush=True)
    return path


def job_e5(args):
    """High-precision comparison of a few pre-specified compositions.

    The candidates are chosen from the E4 sweep *before* this run, so the only
    selection effect is that of having looked at K=150 data; this is stated as a
    limitation in the paper.  All candidates are evaluated with the same K and
    the difference in conduction probability is tested with a two-proportion
    z-test.
    """
    th = _thresholds()
    a90 = th.get("phi_A90", 0.00745)
    b90 = th.get("phi_B90", 0.40)
    if args.abs:
        cands = [(args.abs[i], args.abs[i + 1]) for i in range(0, len(args.abs) - 1, 2)]
        absolute = True
    else:
        cands = args.cands or [(1.00, 0.00), (0.90, 0.10), (0.80, 0.20), (0.70, 0.30)]
        absolute = False
    path = _out_path("e5")
    done = _load_done(path) if args.resume else {}
    m = Model(lateral="periodic")
    print(f"E5  {len(cands)} candidates  K={args.runs}  "
          f"{'absolute compositions' if absolute else 'multiples of thresholds'}")
    for fa, fb in cands:
        pa, pb = (fa, fb) if absolute else (fa * a90, fb * b90)
        key = f"a{pa:.5f}_b{pb:.4f}"
        if key in done:
            continue
        est = estimate(m, phi_rod=pa, phi_sphere=pb, n_runs=args.runs, seed=args.seed)
        rec = _record("e5", key, phi_a=pa, phi_b=pb, fa=fa, fb=fb,
                      n_a=est.n_rod, n_b=est.n_sphere, k=est.k, runs=args.runs,
                      p=est.p_hat, lo=est.ci_low, hi=est.ci_high,
                      cost=est.cost, seconds=est.wall_time)
        _append(path, rec)
        print(f"  A={pa*100:.3f}%  B={pb*100:.2f}%  N=({est.n_rod:4d},{est.n_sphere:5d})  "
              f"p={est.p_hat:.4f} [{est.ci_low:.4f},{est.ci_high:.4f}]  "
              f"cost={est.cost:6.3f}", flush=True)
    return path


JOBS = {
    "e1b-rod": job_e1b_rod,
    "e1b-sphere": job_e1b_sphere,
    "e1a": job_e1a,
    "e2": job_e2,
    "e3": job_e3,
    "e4": job_e4,
    "e5": job_e5,
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job", choices=sorted(JOBS))
    ap.add_argument("--runs", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--phis", type=float, nargs="+", default=None)
    ap.add_argument("--sizes", type=float, nargs="+", default=None)
    ap.add_argument("--fa", type=float, nargs="+", default=None)
    ap.add_argument("--fb", type=float, nargs="+", default=None)
    ap.add_argument("--cands", type=float, nargs="+", default=None,
                    help="flat list fa1 fb1 fa2 fb2 ... (multiples of the thresholds)")
    ap.add_argument("--abs", type=float, nargs="+", default=None,
                    help="flat list of absolute volume fractions phi_a1 phi_b1 ...")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)

    if args.cands:
        args.cands = [(args.cands[i], args.cands[i + 1])
                      for i in range(0, len(args.cands) - 1, 2)]

    t0 = time.time()
    path = JOBS[args.job](args)
    print(f"\n-> {path}   ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
