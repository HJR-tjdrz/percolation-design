# -*- coding: utf-8 -*-
"""Figures for the reviewer-requested supplement experiments (B1, B2, B3).

    python tools/make_supplement_figures.py            # all available
    python tools/make_supplement_figures.py b1 b3

Reads the JSONL files under
``project_analyze/11.7论文第一次自审及修改/11.7论文修改时补充的内容比如实验图片等等/``
and writes PDF + SVG + 300 dpi PNG next to them.  Missing experiments are
skipped, so this is safe to run while a job is still filling in points.
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fit as fitmod

SUP = os.path.join(ROOT, "data", "supplement")
FIGS = os.path.join(ROOT, "figures")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "legend.frameon": False, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01, "svg.fonttype": "none", "pdf.fonttype": 42,
})

C_MAIN, C_ACC, C_ALT = "#1f3b73", "#c0392b", "#2e7d32"


def _save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"))
    fig.savefig(os.path.join(FIGS, f"{name}.png"), dpi=300)
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf/.svg/.png")


def _rows(sub, fname):
    p = os.path.join(SUP, sub, fname)
    if not os.path.exists(p):
        return []
    return fitmod.load_jsonl(p)


def fig_b1():
    rows = _rows("B1_rod_thickness", "b1_rod_thickness.jsonl")
    if not rows:
        print("  b1: no data")
        return
    sizes = sorted({r["L"] for r in rows})
    half, nine, lo9, hi9 = [], [], [], []
    for L in sizes:
        sub = sorted([r for r in rows if r["L"] == L], key=lambda r: r["phi"])
        phi = np.array([r["phi"] for r in sub])
        k = np.array([r["k"] for r in sub])
        n = np.array([r["runs"] for r in sub])
        res = fitmod.bootstrap_levels(phi, k, n, ps=(0.5, 0.9), n_boot=400)
        half.append(res["levels"][0.5]["value"] * 100)
        nine.append(res["levels"][0.9]["value"] * 100)
        lo9.append(res["levels"][0.9]["lo"] * 100)
        hi9.append(res["levels"][0.9]["hi"] * 100)

    x = np.array(sizes) / 5000.0
    nine = np.array(nine)
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    ax.errorbar(x, nine, yerr=[np.maximum(0, nine - np.array(lo9)),
                               np.maximum(0, np.array(hi9) - nine)],
                fmt="o-", ms=4, lw=1.0, color=C_MAIN, ecolor=C_MAIN,
                elinewidth=0.8, capsize=2, label=r"$\phi_A^*(0.9)$")
    ax.plot(x, half, "s--", ms=3.6, lw=0.9, color=C_ACC,
            label=r"$\phi_A^*(0.5)$")
    for xi, yi in zip(x, nine):
        ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                    xytext=(4, 5), fontsize=6.2, color=C_MAIN)
    for xi, yi in zip(x, half):
        ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                    xytext=(4, -9), fontsize=6.2, color=C_ACC)
    ax.set_xlabel(r"cell thickness  $L/L_{\mathrm{rod}}$")
    ax.set_ylabel(r"threshold  [vol %]")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.1f}" for v in x])
    ax.set_xlim(x.min() - 0.10, x.max() + 0.22)
    ax.legend(loc="center right", handletextpad=0.4, borderaxespad=0.2)
    _save(fig, "figB1_rod_thickness")


def fig_b2():
    rows = _rows("B2_delta_sensitivity", "b2_delta.jsonl")
    if not rows:
        print("  b2: no data")
        return
    deltas = sorted({r["delta"] for r in rows})
    nine = []
    for dd in deltas:
        sub = sorted([r for r in rows if r["delta"] == dd], key=lambda r: r["phi"])
        phi = np.array([r["phi"] for r in sub])
        k = np.array([r["k"] for r in sub])
        n = np.array([r["runs"] for r in sub])
        res = fitmod.bootstrap_levels(phi, k, n, ps=(0.9,), n_boot=300)
        nine.append(res["levels"][0.9]["value"] * 100)
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    ax.plot(deltas, nine, "o-", ms=4, lw=1.0, color=C_ALT)
    for dd, yy in zip(deltas, nine):
        ax.annotate(f"{yy:.2f}", (dd, yy), textcoords="offset points",
                    xytext=(5, 4), fontsize=6.4, color=C_ALT)
    ax.set_xlabel(r"tunnelling cut-off $\delta$  [nm]")
    ax.set_ylabel(r"$\phi_A^*(0.9)$  [vol %]")
    _save(fig, "figB2_delta")


def fig_b3():
    rows = _rows("B3_sphere_boundary", "b3_sphere_boundary.jsonl")
    if not rows:
        print("  b3: no data")
        return
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for lat, col, mk in (("periodic", C_MAIN, "o"), ("walls", C_ACC, "s")):
        sub = sorted([r for r in rows if r["lateral"] == lat], key=lambda r: r["phi"])
        if not sub:
            continue
        phi = np.array([r["phi"] for r in sub]) * 100
        p = np.array([r["p"] for r in sub])
        lo = np.array([r["lo"] for r in sub])
        hi = np.array([r["hi"] for r in sub])
        ax.errorbar(phi, p, yerr=[np.maximum(0, p - lo), np.maximum(0, hi - p)],
                    fmt=mk + "-", ms=3.6, lw=0.9, color=col, ecolor=col,
                    elinewidth=0.7, capsize=0, mfc="white", mew=0.8,
                    label=("periodic laterals" if lat == "periodic"
                           else "insulating laterals"))
        try:
            res = fitmod.bootstrap_levels(np.array([r["phi"] for r in sub]),
                                          np.array([r["k"] for r in sub]),
                                          np.array([r["runs"] for r in sub]),
                                          ps=(0.9,), n_boot=300)
            print(f"    {lat}: phi_B*(0.9) = {res['levels'][0.9]['value']*100:.2f} vol %")
        except ValueError:
            pass
    ax.axhline(0.9, color="#888888", lw=0.6, ls=":")
    ax.set_xlabel(r"sphere volume fraction $\phi_B$  [vol %]")
    ax.set_ylabel(r"conduction probability $P$")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="lower right", handletextpad=0.4, borderaxespad=0.2)
    _save(fig, "figB3_sphere_boundary")


JOBS = {"b1": fig_b1, "b2": fig_b2, "b3": fig_b3}


def main(argv=None):
    names = [a for a in (argv or sys.argv[1:]) if not a.startswith("-")] or list(JOBS)
    for n in names:
        if n in JOBS:
            print(f"{n}:")
            JOBS[n]()
        else:
            print(f"unknown {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
