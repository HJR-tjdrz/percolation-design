# -*- coding: utf-8 -*-
"""Publication figures from the stored experiment results.

    python tools/make_figures.py            # everything that has data
    python tools/make_figures.py fig3 fig5  # a subset

Writes vector PDF and SVG into ``figures/``.  Sizes follow the IEEE two-column
grid: single column 3.5 in, double column 7.16 in, base font 8 pt, serif to match
the template.
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import fit as fitmod

RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figures")

# ---------------------------------------------------------------- style
GOLDEN = 0.618
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.grid": False,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

C_MAIN = "#1f3b73"
C_ACC = "#c0392b"
C_ALT = "#2e7d32"
C_GREY = "#6b6b6b"


def _save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"))
    prev = os.path.join(FIGS, "preview")
    os.makedirs(prev, exist_ok=True)
    fig.savefig(os.path.join(prev, f"{name}.png"), dpi=200)
    # raster copy at print resolution, used by the DOCX builder
    hires = os.path.join(FIGS, "png")
    os.makedirs(hires, exist_ok=True)
    fig.savefig(os.path.join(hires, f"{name}.png"), dpi=300)
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf / .svg  (+ png at 200/300 dpi)")


def _load_jsonl(name):
    path = os.path.join(RESULTS, name + ".jsonl")
    if not os.path.exists(path):
        return []
    return fitmod.load_jsonl(path)


# ---------------------------------------------------------------- Fig. 2
def fig2_validation():
    """Finite-size study: the measured threshold approaches the bulk value."""
    rows = _load_jsonl("e1a")
    if not rows:
        print("  fig2: no e1a data yet")
        return

    rows = [r for r in rows if r.get("lateral", "periodic") != "walls"]
    sizes = sorted({r["L"] for r in rows})
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.55))

    phis_fit = np.linspace(0.15, 0.45, 200)
    cols = plt.cm.viridis(np.linspace(0.12, 0.88, len(sizes)))
    fitted = []
    for L, c in zip(sizes, cols):
        sub = sorted([r for r in rows if r["L"] == L], key=lambda r: r["phi"])
        phi = np.array([r["phi"] for r in sub])
        k = np.array([r["k"] for r in sub])
        n = np.array([r["runs"] for r in sub])
        p = k / n
        lo = np.array([r["lo"] for r in sub])
        hi = np.array([r["hi"] for r in sub])
        ax[0].errorbar(phi, p, yerr=[np.maximum(0.0, p - lo), np.maximum(0.0, hi - p)],
                       fmt="o", ms=2.6,
                       color=c, ecolor=c, elinewidth=0.7, capsize=0,
                       mfc="white", mew=0.8, label=f"$L/\\sigma$={L/400:.0f}")
        try:
            res = fitmod.bootstrap_levels(phi, k, n, ps=(0.5,), n_boot=300)
            ax[0].plot(phis_fit, fitmod.logistic(phis_fit, res["fit"]["phi_c"],
                                                 res["fit"]["k"]), "-", color=c, lw=0.9)
            fitted.append((L, res["levels"][0.5]["value"],
                           res["levels"][0.5]["lo"], res["levels"][0.5]["hi"]))
        except Exception as exc:  # noqa: BLE001
            print(f"    fit failed at L={L}: {exc}")

    ax[0].set_xlabel(r"sphere volume fraction $\phi_B$")
    ax[0].set_ylabel(r"spanning probability $P$")
    ax[0].set_xlim(0.17, 0.45)
    ax[0].set_ylim(-0.03, 1.03)
    ax[0].legend(loc="lower right", handletextpad=0.4, borderaxespad=0.2,
                 title="cell thickness", title_fontsize=6.5)
    ax[0].set_title("(a) the spanning curve hardly moves with thickness",
                    fontsize=7.5, pad=2)

    if fitted:
        Ls = np.array([f[0] for f in fitted], float) / 400.0
        ph = np.array([f[1] for f in fitted], float)
        lo = np.array([f[2] for f in fitted], float)
        hi = np.array([f[3] for f in fitted], float)
        ax[1].errorbar(Ls, ph, yerr=[np.maximum(0.0, ph - lo), np.maximum(0.0, hi - ph)],
                       fmt="o", ms=3.6, color=C_MAIN, ecolor=C_MAIN, elinewidth=0.8,
                       capsize=2, label="plate-to-plate spanning")
        mean = float(np.mean(ph))
        ax[1].axhline(mean, color=C_MAIN, lw=0.8, alpha=0.6)
        ax[1].text(5.6, mean + 0.004, f"mean {mean:.3f}", fontsize=6.5, color=C_MAIN)
        ax[1].axhline(0.2895, color=C_ACC, lw=0.9, ls=":",
                      label="bulk, fully periodic (0.2895)")
        ax[1].set_xlabel(r"cell thickness $L/\sigma$")
        ax[1].set_ylabel(r"measured $\phi_c$")
        ax[1].set_xlim(5, 28)
        ax[1].set_ylim(0.27, 0.37)
        ax[1].legend(loc="lower right", handletextpad=0.4, borderaxespad=0.2)
        ax[1].set_title("(b) the offset is a criterion difference, not finite size",
                        fontsize=7.5, pad=2)

    fig.tight_layout(pad=0.3, w_pad=1.2)
    _save(fig, "fig2_validation")
    if fitted:
        vals = [f[1] for f in fitted]
        print(f"    phi_c mean {np.mean(vals):.4f}, spread {max(vals)-min(vals):.4f}, "
              f"ratio to bulk {np.mean(vals)/0.2895:.3f}")


# ---------------------------------------------------------------- Fig. 2 (rev2)
SUP = os.path.join(ROOT, "data", "supplement")


def _sup_rows(sub, fname):
    p = os.path.join(SUP, sub, fname)
    if not os.path.exists(p):
        return []
    return fitmod.load_jsonl(p)


def fig2_validation_v2():
    """Thickness dependence of both fillers: spheres (a) and rods (b).

    Panel (a) answers 'does the measured sphere threshold drift with cell size'.
    Panel (b) is the rod study added after the reviewers pointed out that the
    design uses rods while only spheres had been tested.
    """
    sph = _load_jsonl("e1a")
    rod = _sup_rows("B1_rod_thickness", "b1_rod_thickness.jsonl")
    if not sph and not rod:
        print("  fig2: no data")
        return

    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.45))

    if sph:
        sizes = sorted({r["L"] for r in sph})
        Ls, ph, lo, hi = [], [], [], []
        for L in sizes:
            sub = sorted([r for r in sph if r["L"] == L], key=lambda r: r["phi"])
            phi = np.array([r["phi"] for r in sub])
            k = np.array([r["k"] for r in sub])
            n = np.array([r["runs"] for r in sub])
            try:
                res = fitmod.bootstrap_levels(phi, k, n, ps=(0.5,), n_boot=250)
            except ValueError:
                continue
            lv = res["levels"][0.5]
            Ls.append(L / 400.0)
            ph.append(lv["value"])
            lo.append(lv["lo"])
            hi.append(lv["hi"])
        Ls = np.array(Ls); ph = np.array(ph)
        lo = np.array(lo); hi = np.array(hi)
        ax[0].errorbar(Ls, ph, yerr=[np.maximum(0, ph - lo), np.maximum(0, hi - ph)],
                       fmt="o", ms=3.4, color=C_MAIN, ecolor=C_MAIN,
                       elinewidth=0.8, capsize=2)
        if len(ph):
            ax[0].axhline(ph.mean(), color=C_MAIN, lw=0.8, alpha=0.6)
            ax[0].text(27.2, ph.mean() + 0.005, f"mean {ph.mean():.3f}",
                       fontsize=6.4, color=C_MAIN, ha="right")
        ax[0].axhline(0.2895, color=C_ACC, lw=0.9, ls=":")
        ax[0].set_xlabel(r"cell thickness $L/\sigma$")
        ax[0].set_ylabel(r"$\phi_B^*(0.5)$")
        ax[0].set_xlim(5, 28)
        ax[0].text(6.0, 0.2925, "bulk, fully periodic (0.2895)",
                   fontsize=6.2, color=C_ACC)
        ax[0].set_title("(a) spheres: threshold does not drift", fontsize=7.4, pad=2)

    if rod:
        sizes = sorted({r["L"] for r in rod})
        h, n9, n9lo, n9hi = [], [], [], []
        for L in sizes:
            sub = sorted([r for r in rod if r["L"] == L], key=lambda r: r["phi"])
            phi = np.array([r["phi"] for r in sub])
            k = np.array([r["k"] for r in sub])
            n = np.array([r["runs"] for r in sub])
            res = fitmod.bootstrap_levels(phi, k, n, ps=(0.5, 0.9), n_boot=400)
            h.append(res["levels"][0.5]["value"] * 100)
            n9.append(res["levels"][0.9]["value"] * 100)
            n9lo.append(res["levels"][0.9]["lo"] * 100)
            n9hi.append(res["levels"][0.9]["hi"] * 100)
        x = np.array(sizes) / 5000.0
        n9 = np.array(n9)
        ax[1].errorbar(x, n9, yerr=[np.maximum(0, n9 - np.array(n9lo)),
                                    np.maximum(0, np.array(n9hi) - n9)],
                       fmt="o-", ms=3.6, lw=1.0, color=C_MAIN, ecolor=C_MAIN,
                       elinewidth=0.8, capsize=2, label=r"$\phi_A^*(0.9)$")
        ax[1].plot(x, h, "s--", ms=3.2, lw=0.9, color=C_ACC,
                   label=r"$\phi_A^*(0.5)$")
        ax[1].set_xlabel(r"cell thickness $L/L_{\mathrm{rod}}$")
        ax[1].set_ylabel(r"threshold  [vol %]")
        ax[1].set_xticks(x)
        ax[1].set_xticklabels([f"{v:.1f}" for v in x])
        ax[1].set_xlim(x.min() - 0.10, x.max() + 0.16)
        ax[1].set_ylim(0.46, 0.80)
        ax[1].set_ylim(0.46, 0.80)
        ax[1].legend(loc="lower right", handletextpad=0.4, borderaxespad=0.2)
        ax[1].set_title("(b) rods: 50 % moves, 90 % does not", fontsize=7.4, pad=2)
        print(f"    rod 90%: {n9[0]:.3f} -> {n9[-1]:.3f} "
              f"({(n9[-1]/n9[0]-1)*100:.1f}% relative); "
              f"50%: {h[0]:.3f} -> {h[-1]:.3f} ({(h[-1]/h[0]-1)*100:.1f}%)")

    fig.tight_layout(pad=0.3, w_pad=1.4)
    _save(fig, "fig2_validation")


# ---------------------------------------------------------------- Fig. 3
def fig3_curves():
    """Pure-filler percolation curves in the device geometry."""
    rod = _load_jsonl("e1b_rod")
    sph = _load_jsonl("e1b_sphere")
    if not rod and not sph:
        print("  fig3: no e1b data yet")
        return

    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.55))
    summary = {}

    for panel, rows, colour, label, unit in (
            (0, rod, C_MAIN, "filler A (spherocylinder)", r"$\phi_A$"),
            (1, sph, C_ALT, "filler B (sphere)", r"$\phi_B$")):
        if not rows:
            continue
        phi = np.array([r["phi"] for r in rows])
        k = np.array([r["k"] for r in rows])
        n = np.array([r["runs"] for r in rows])
        p = k / n
        lo = np.array([r["lo"] for r in rows])
        hi = np.array([r["hi"] for r in rows])
        ax[panel].errorbar(phi * 100, p, yerr=[np.maximum(0.0, p - lo),
                                               np.maximum(0.0, hi - p)],
                           fmt="o", ms=2.8,
                           color=colour, ecolor=colour, elinewidth=0.7, capsize=0,
                           mfc="white", mew=0.8, label=label)
        try:
            res = fitmod.bootstrap_levels(phi, k, n, ps=(0.5, 0.9), n_boot=1000)
            xs = np.linspace(phi.min(), phi.max(), 200)
            ax[panel].plot(xs * 100, fitmod.logistic(xs, res["fit"]["phi_c"],
                                                     res["fit"]["k"]), "-",
                           color=colour, lw=1.0, label="logistic fit")
            for target, ls in ((0.5, "--"), (0.9, ":")):
                lv = res["levels"][target]
                ax[panel].axhline(target, color=C_GREY, lw=0.6, ls=ls)
                ax[panel].axvline(lv["value"] * 100, color=C_ACC, lw=0.7, ls=ls)
            ax[panel].annotate(f"$\\phi^*(0.9)$={res['levels'][0.9]['value']*100:.2f}%",
                               xy=(res["levels"][0.9]["value"] * 100, 0.9),
                               xytext=(0.30, 0.80), textcoords="axes fraction",
                               fontsize=7, color=C_ACC,
                               arrowprops=dict(arrowstyle="->", lw=0.6, color=C_ACC))
            summary[unit] = {str(t): res["levels"][t] for t in (0.5, 0.9)}
            summary[unit]["r2"] = res["fit"]["r2"]
        except Exception as exc:  # noqa: BLE001
            print(f"    fit failed ({unit}): {exc}")
        ax[panel].set_xlabel(unit + "  [vol %]")
        ax[panel].set_ylabel(r"conduction probability $P$")
        ax[panel].set_ylim(-0.03, 1.03)
        ax[panel].legend(loc="lower right", handletextpad=0.4, borderaxespad=0.2)

    fig.tight_layout(pad=0.3, w_pad=1.2)
    _save(fig, "fig3_percolation_curves")

    with open(os.path.join(RESULTS, "thresholds.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    for unit, d in summary.items():
        print(f"    {unit}: phi*(0.5)={d['0.5']['value']:.5f}, "
              f"phi*(0.9)={d['0.9']['value']:.5f}  (R2={d['r2']:.4f})")


# ---------------------------------------------------------------- Fig. 5
def fig5_design_map():
    """Lambda plane: which filler is economically preferable."""
    th = {}
    path = os.path.join(RESULTS, "thresholds.json")
    if os.path.exists(path):
        th = json.load(open(path, encoding="utf-8"))
    phi_a = th.get(r"$\phi_A$", {}).get("0.9", {}).get("value", 0.0074)
    phi_b = th.get(r"$\phi_B$", {}).get("0.9", {}).get("value", 0.38)

    c_ratio = np.linspace(4, 60, 300)          # c_A / c_B
    phi_b_grid = np.linspace(0.05, 0.50, 300)  # B threshold
    R = c_ratio[:, None] * phi_a / phi_b_grid[None, :]   # Lambda * (c_A/c_B normalised)

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    cs = ax.contourf(phi_b_grid * 100, c_ratio, np.log10(R),
                     levels=np.linspace(-2, 1.2, 17), cmap="RdBu_r")
    ax.contour(phi_b_grid * 100, c_ratio, R, levels=[1.0], colors="k", linewidths=1.0)
    ax.plot([15], [21], "o", ms=5, mfc="white", mec="k", mew=0.9)
    ax.annotate("bulk literature\n15 vol%", (15, 21), xytext=(19, 30),
                fontsize=6.5, arrowprops=dict(arrowstyle="-", lw=0.5))
    ax.plot([phi_b * 100], [21], "s", ms=5, color="#f1c40f", mec="k", mew=0.8)
    ax.annotate(f"measured here\n{phi_b*100:.0f} vol%", (phi_b * 100, 21),
                xytext=(phi_b * 100 - 2, 9), ha="right", fontsize=6.5,
                arrowprops=dict(arrowstyle="-", lw=0.5))
    ax.set_xlabel(r"$\phi_B^*(0.9)$  [vol %]")
    ax.set_ylabel(r"cost ratio $c_A/c_B$")
    ax.set_xlim(5, 50)
    ax.set_ylim(4, 60)
    cb = fig.colorbar(cs, ax=ax, pad=0.02)
    cb.set_label(r"$\log_{10}\Lambda$")
    cb.ax.tick_params(labelsize=6)
    fig.tight_layout(pad=0.2)
    _save(fig, "fig5_design_map")
    print(f"    using phi_A*(0.9)={phi_a:.4f}, phi_B*(0.9)={phi_b:.3f}, "
          f"Lambda={1.05*phi_a/(0.05*phi_b):.3f}")


# ---------------------------------------------------------------- Fig. 1
def fig1_schematic():
    """Schematic of the cell and of the two electrode conventions.

    Left: a front view of the cell with both fillers and one conducting path.
    Right: what happens to a filler that crosses an electrode under clipping
    (the convention used here) and under mirroring (the unsound one).
    """
    rng = np.random.default_rng(4)
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.45),
                           gridspec_kw={"width_ratios": [1.35, 1.0]})

    # ---------------- left: the cell -------------------------------
    a = ax[0]
    W, H = 1.0, 0.72
    a.add_patch(plt.Rectangle((0, 0), W, H, fc="#fafafa", ec="k", lw=0.8, zorder=1))
    for xpos in (0.0, W):
        a.add_patch(plt.Rectangle((xpos - 0.017, -0.02), 0.034, H + 0.04,
                                  fc="#3a3a3a", ec="k", lw=0.3, zorder=5))

    ys = np.array([0.30, 0.38, 0.30, 0.36, 0.28, 0.34])
    xs = np.linspace(0.02, W - 0.02, ys.size)
    a.plot(xs, ys, "-", color=C_ACC, lw=1.7, solid_capstyle="round",
           zorder=6, alpha=0.9)

    for _ in range(26):
        cx, cy = rng.uniform(0.04, W - 0.04), rng.uniform(0.04, H - 0.04)
        ang = rng.uniform(0, np.pi)
        half = 0.16
        dx, dy = half * np.cos(ang), half * np.sin(ang)
        a.plot([cx - dx, cx + dx], [cy - dy, cy + dy], "-", color=C_MAIN,
               lw=0.9, solid_capstyle="round", zorder=3, alpha=0.7)
    for _ in range(28):
        cx, cy = rng.uniform(0.04, W - 0.04), rng.uniform(0.04, H - 0.04)
        a.add_patch(plt.Circle((cx, cy), 0.020, fc="#d8ecd8", ec=C_ALT,
                               lw=0.5, zorder=2, alpha=0.95))

    a.text(-0.055, H / 2, "electrode", rotation=90, ha="center", va="center",
           fontsize=6.5, color="#3a3a3a")
    a.text(W + 0.055, H / 2, "electrode", rotation=-90, ha="center", va="center",
           fontsize=6.5, color="#3a3a3a")
    a.text(W / 2, -0.075, "insulating lateral faces, periodic",
           ha="center", va="top", fontsize=6.5, color=C_GREY)
    a.annotate("conducting path", (W * 0.62, 0.335), (W * 0.60, 0.575),
               fontsize=6.5, color=C_ACC, ha="center",
               arrowprops=dict(arrowstyle="->", lw=0.6, color=C_ACC))
    a.plot([], [], "-", color=C_MAIN, lw=1.1, label="filler A (rod)")
    a.plot([], [], "o", ms=3.6, color="#d8ecd8", mec=C_ALT, label="filler B (sphere)")
    a.legend(loc="lower center", bbox_to_anchor=(0.5, -0.26), ncol=2,
             handletextpad=0.4, columnspacing=1.4, borderaxespad=0.0,
             labelspacing=0.25)
    a.set_xlim(-0.11, W + 0.11)
    a.set_ylim(-0.12, H + 0.03)
    a.set_aspect("equal")
    a.axis("off")
    a.set_title("(a) micro-cell between two electrodes", fontsize=7.5, pad=1)

    # ---------------- right: the two conventions --------------------
    b = ax[1]
    strip_h = 0.26
    for k, (title, mirrored) in enumerate((("clipped  (used here)", False),
                                           ("mirrored  (unsound)", True))):
        y0 = 0.62 if k == 0 else 0.07
        b.add_patch(plt.Rectangle((0.0, y0), 1.0, strip_h, fc="white", ec="k",
                                  lw=0.6, ls=(0, (1, 1.4)), zorder=1))
        b.add_patch(plt.Rectangle((-0.012, y0 - 0.015), 0.024, strip_h + 0.03,
                                  fc="#3a3a3a", ec="k", lw=0.2, zorder=4))
        b.add_patch(plt.Rectangle((0.988, y0 - 0.015), 0.024, strip_h + 0.03,
                                  fc="#3a3a3a", ec="k", lw=0.2, zorder=4))
        # the rod inside the cell, touching the right electrode
        y_in, y_out = y0 + 0.06, y0 + 0.20
        b.plot([0.50, 1.00], [y_in, y_out], "-", color=C_MAIN, lw=1.4,
               solid_capstyle="round", zorder=3)
        # the part that would be beyond the electrode
        slope = (y_out - y_in) / 0.50
        y_end = y_out + slope * 0.16
        b.plot([1.00, 1.16], [y_out, y_end], "-", color=C_MAIN, lw=1.4,
               ls=(0, (1.6, 1.2)), alpha=0.35, zorder=3)
        b.text(0.0, y0 - 0.045, title, fontsize=6.8, va="top")
        if mirrored:
            # the same overhang, translated by one cell length onto the near side
            # of the LEFT electrode
            b.plot([0.0, 0.16], [y_in, y_in + slope * 0.16], ls=(0, (1.6, 1.2)),
                   color=C_MAIN, lw=1.4, zorder=3)
            b.annotate("", (0.14, y_in + slope * 0.14), (0.93, y_out + slope * 0.10),
                       arrowprops=dict(arrowstyle="->", lw=0.7, color=C_ACC,
                                       connectionstyle="arc3,rad=0.32"))
            b.text(0.52, y0 + strip_h + 0.030,
                   "the overhang is re-attached here, so this one rod now\n"
                   "touches both electrodes: every realisation conducts",
                   ha="center", va="bottom", fontsize=5.9, color=C_ACC)
        else:
            b.plot([1.00, 1.00], [y_out - 0.02, y_out + 0.045], "-",
                   color=C_GREY, lw=0.7)
            b.text(0.52, y0 + strip_h + 0.030,
                   "the overhang is discarded; contact is with this electrode only",
                   ha="center", va="bottom", fontsize=5.9, color=C_GREY)
    b.set_xlim(-0.22, 1.22)
    b.set_ylim(0.02, 1.02)
    b.set_aspect("equal")
    b.axis("off")
    b.set_title("(b) filler crossing a current-carrying electrode", fontsize=7.5, pad=1)

    fig.tight_layout(pad=0.2, w_pad=0.4)
    _save(fig, "fig1_schematic")


# ---------------------------------------------------------------- Fig. 4
def fig4_compositions():
    """Two-dimensional sweep: how good is the linear strength constraint?"""
    rows = _load_jsonl("e4")
    if not rows:
        print("  fig4: no e4 data yet")
        return
    th = json.load(open(os.path.join(RESULTS, "numbers.json"), encoding="utf-8"))
    a50 = th.get("phi_A50", 0.00607)
    b50 = th.get("phi_B50", 0.33513)
    a90 = th.get("phi_A90", 0.00753)
    b90 = th.get("phi_B90", 0.35438)
    eta = a90 / a50
    lam_cross = (1.05 * a90) / (0.05 * b90)

    va = np.array([r["phi_a"] for r in rows]) * 100
    vb = np.array([r["phi_b"] for r in rows]) * 100
    p = np.array([r["p"] for r in rows])
    cost = np.array([r["cost"] for r in rows])

    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.7),
                           gridspec_kw={"width_ratios": [1.05, 1.0]})

    # ---- (a) probability in the composition plane -------------------
    sc = ax[0].scatter(va, vb, c=p, vmin=0, vmax=1, s=34, cmap="viridis",
                       edgecolor="k", linewidth=0.35, zorder=4)
    xs = np.linspace(0, a90 * 100, 50)
    ax[0].plot(xs, b90 * 100 * (1 - xs / (a90 * 100)), "-", color=C_ACC, lw=1.2,
               zorder=5, label="linear 90 % contour")
    xs5 = np.linspace(0, a50 * 100, 50)
    ax[0].plot(xs5, b50 * 100 * (1 - xs5 / (a50 * 100)), "--", color=C_GREY, lw=1.0,
               zorder=5, label="linear 50 % contour")
    ax[0].axvline(a90 * 100, color=C_MAIN, lw=0.7, ls=":", zorder=3)
    ax[0].plot([a90 * 100], [0], "*", ms=11, color="#f1c40f", mec="k", mew=0.7,
               zorder=6, label="linear-programming optimum")
    ax[0].set_xlabel(r"filler A  $\phi_A$  [vol %]")
    ax[0].set_ylabel(r"filler B  $\phi_B$  [vol %]")
    ax[0].set_xlim(0, 0.86)
    ax[0].set_ylim(-1, 42)
    ax[0].legend(loc="upper right", handletextpad=0.5, borderaxespad=0.3)
    cb = fig.colorbar(sc, ax=ax[0], pad=0.02)
    cb.set_label(r"measured $P$", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax[0].set_title("(a) measured probability and the surrogate", fontsize=7.5, pad=2)

    # ---- (b) collapse onto the linear strength ----------------------
    s = va / (a90 * 100) + vb / (b90 * 100)
    order = np.argsort(s)
    ax[1].plot(s[order], p[order], "o", ms=3.4, color=C_MAIN, mfc="white",
               mew=0.8, label="all compositions")
    ax[1].axhline(0.9, color=C_ACC, lw=0.8, ls=":")
    ax[1].axvline(1.0, color=C_ACC, lw=0.9)
    ax[1].text(1.03, 0.12, r"$s$ = 1", fontsize=7, color=C_ACC)
    ax[1].text(0.42, 0.93, r"$P$ = 0.9", fontsize=7, color=C_ACC)
    ax[1].set_xlabel(r"linear strength  $\phi_A/\phi_A^*(0.9) + \phi_B/\phi_B^*(0.9)$")
    ax[1].set_ylabel(r"measured $P$")
    ax[1].set_xlim(0.0, 1.6)
    ax[1].set_ylim(-0.03, 1.03)
    ax[1].legend(loc="lower right", handletextpad=0.4, borderaxespad=0.2)
    ax[1].set_title("(b) all compositions collapse onto one strength axis",
                    fontsize=7.5, pad=2)

    fig.tight_layout(pad=0.3, w_pad=1.3)
    _save(fig, "fig4_compositions")

    # ---- numbers for the text --------------------------------------
    near = np.abs(p - 0.9) <= 0.12
    if near.any():
        print(f"    compositions with P in [0.78,1.02]: s in "
              f"[{s[near].min():.2f}, {s[near].max():.2f}], eta={eta:.3f}")
    feasible = p >= 0.9
    if feasible.any():
        best = int(np.argmin(np.where(feasible, cost, np.inf)))
        print(f"    cheapest feasible grid point: A={va[best]:.2f}% B={vb[best]:.2f}% "
              f"P={p[best]:.3f} cost={cost[best]:.2f}")
    print(f"    Lambda = {lam_cross:.3f}, eta = {eta:.3f}")


JOBS = {"fig1": fig1_schematic, "fig2": fig2_validation_v2,
        "fig3": fig3_curves, "fig4": fig4_compositions, "fig5": fig5_design_map}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    names = [a for a in argv if not a.startswith("-")] or list(JOBS)
    for name in names:
        if name not in JOBS:
            print(f"unknown figure {name!r}")
            continue
        print(f"{name}:")
        JOBS[name]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
