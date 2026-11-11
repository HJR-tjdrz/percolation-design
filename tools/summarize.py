# -*- coding: utf-8 -*-
"""Build the single source of truth for every number that appears in the paper.

    python tools/summarize.py

Writes ``results/numbers.md`` (human readable) and ``results/numbers.json``
(machine readable).  The paper must never contain a hand-typed experimental
number; it quotes this file.
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

import fit as fitmod

RESULTS = os.path.join(ROOT, "results")

C_A, C_B = 1.05, 0.05
V_ROD = math.pi * 30.0 ** 2 * 5000.0
V_SPH = 4.0 / 3.0 * math.pi * 200.0 ** 3
V_CELL = 1000.0  # um^3


def cost_from_phi(phi, kind):
    vol_um3 = phi * V_CELL  # since V_cell = 1000 um^3 by construction
    return vol_um3 * (C_A if kind == "rod" else C_B)


def load(name):
    path = os.path.join(RESULTS, name + ".jsonl")
    if not os.path.exists(path):
        return []
    return fitmod.load_jsonl(path)


def thresholds(name, ps=(0.5, 0.9), boot=3000):
    rows = load(name)
    if len(rows) < 3:
        return None
    phi = np.array([r["phi"] for r in rows])
    k = np.array([r["k"] for r in rows])
    n = np.array([r["runs"] for r in rows])
    try:
        return fitmod.bootstrap_levels(phi, k, n, ps=ps, n_boot=boot)
    except ValueError as exc:
        print(f"  fit failed for {name}: {exc}")
        return None


def main():
    out = {}
    lines = ["# numbers.md — 论文里所有实验数字的唯一来源", "",
             "> 由 `tools/summarize.py` 自动生成，**不要手改**。",
             "> 论文中任何实验数字都必须能在这里找到。", ""]

    # ---------------- thresholds -------------------------------------
    th_a = thresholds("e1b_rod")
    th_b = thresholds("e1b_sphere")

    lines.append("## 1. 标定阈值（器件几何，横向周期 + 电极截断）")
    lines.append("")
    lines.append("| 量 | 值 | 95% CI | R² | 来源 |")
    lines.append("|---|---|---|---|---|")
    for tag, th, kind in (("A (棒)", th_a, "rod"), ("B (球)", th_b, "sphere")):
        if not th:
            lines.append(f"| φ_{tag} | 待跑 | — | — | e1b |")
            continue
        for p in (0.5, 0.9):
            lv = th["levels"][p]
            lines.append(f"| φ_{tag}*({p}) | {lv['value']*100:.3f}% | "
                         f"[{lv['lo']*100:.3f}, {lv['hi']*100:.3f}]% | "
                         f"{th['fit']['r2']:.4f} | e1b |")
        out[tag] = th

    # ---------------- economics ----------------------------------------
    lines += ["", "## 2. 成本与判据", ""]
    if th_a:
        pa = th_a["levels"][0.9]["value"]
        out["phi_A50"] = th_a["levels"][0.5]["value"]
        out["phi_A90"] = pa
        if th_b:
            out["phi_B50"] = th_b["levels"][0.5]["value"]
            out["phi_B90"] = th_b["levels"][0.9]["value"]
        cost_a = cost_from_phi(pa, "rod")
        n_a = pa * V_CELL / (V_ROD / 1e9)
        lines.append(f"- 纯 A 在 90% 处的成本：**{cost_a:.2f} 元** "
                     f"（φ={pa*100:.3f}%，N≈{n_a:.0f} 根）")
        out["cost_A"] = cost_a
        if th_b:
            pb = th_b["levels"][0.9]["value"]
            cost_b = cost_from_phi(pb, "sphere")
            lam = (C_A * pa) / (C_B * pb)
            eta = th_a["levels"][0.9]["value"] / th_a["levels"][0.5]["value"]
            cross = (C_A / C_B) * pa
            lines += [
                f"- 纯 B 在 90% 处的成本：**{cost_b:.2f} 元**（φ={pb*100:.2f}%）",
                f"- **Λ = c_A φ_A\\*(0.9) / (c_B φ_B\\*(0.9)) = {lam:.3f}**  "
                f"→ A 便宜 {1/lam:.2f} 倍",
                f"- B 具竞争力的门槛：φ_B\\*(0.9) < (c_A/c_B)·φ_A\\*(0.9) = "
                f"**{cross*100:.1f}%**（实测 {pb*100:.1f}%，差 {cross/pb:.1f} 倍多一点）",
                f"- **η = φ_A\\*(0.9)/φ_A\\*(0.5) = {eta:.3f}**（论文里那个「安全系数」的正确定义）",
            ]
            out.update({"cost_B": cost_b, "Lambda": lam, "eta": eta,
                        "crossover_phiB": cross})
            if th_b["levels"][0.5]:
                out["phiB_c"] = th_b["levels"][0.5]["value"]

    # ---------------- E5 contour test ----------------------------------
    rows = load("e5")
    if rows:
        lines += ["", "## 4. 等值线验证（E5，K=2000）", "",
                  "§ 三个配比都落在线性代理的 90% 等值线上（强度 ≈ η），比较实测 P。", "",
                  "| φ_A | φ_B | 线性强度 | P̂ | 95% CI | 成本 |", "|---|---|---|---|---|---|"]
        a50 = out.get("phi_A50", 0.00607)
        b50 = out.get("phi_B50", 0.33513)
        for r in sorted(rows, key=lambda r: r["phi_b"]):
            s = r["phi_a"] / a50 + r["phi_b"] / b50
            lines.append(f"| {r['phi_a']*100:.3f}% | {r['phi_b']*100:.2f}% | {s:.3f} | "
                         f"{r['p']:.4f} | [{r['lo']:.4f}, {r['hi']:.4f}] | {r['cost']:.2f} |")
        out["e5"] = [{k: r[k] for k in ("phi_a", "phi_b", "p", "lo", "hi", "cost")}
                     for r in rows]

    rows = load("e2")
    if rows:
        lines += ["", "## 3. 边界条件敏感性（Table 2）", "",
                  "| φ_A | periodic + clip | walls + clip | mirror-contact |",
                  "|---|---|---|---|"]
        phis = sorted({r["phi"] for r in rows})
        table = {}
        for phi in phis:
            cells = []
            for key in ("periodic_clip", "walls_clip", "periodic_wrap"):
                hit = [r for r in rows if r["key"] == f"{key}_{phi:.4f}"]
                if hit:
                    r = hit[0]
                    cells.append(f"{r['p']:.3f} [{r['lo']:.3f},{r['hi']:.3f}]")
                    table.setdefault(key, {})[f"{phi:.4f}"] = r["p"]
                else:
                    cells.append("—")
            lines.append(f"| {phi*100:.1f}% | " + " | ".join(cells) + " |")
        out["e2"] = table
        lines += ["", "> `mirror-contact` 一列全为 1.000：把「穿过电极的部分算作对面电极的接触」",
                  "> 当作接触规则时，任何一根碰到电极的棒都会同时挂上两个电极，构型必然短路。",
                  "> 这不是竞赛代码的行为（那里的镜像碎片是独立节点），但它说明这个约定本身不成立。"]

    # ---------------- throughput ---------------------------------------
    p = os.path.join(RESULTS, "e3.json")
    if os.path.exists(p):
        d = json.load(open(p, encoding="utf-8"))["rows"]
        new = {r["n_rod"]: r["seconds_per_sim"] for r in d if r["impl"] == "new"}
        old = {r["n_rod"]: r["seconds_per_sim"] for r in d if r["impl"] == "legacy"}
        lines += ["", "## 4. 吞吐量（Table 3，Anaconda Python 3.9，同一进程）", "",
                  "| N_A | 旧实现 ms/次 | 新实现 ms/次 | 加速 |", "|---|---|---|---|"]
        for n in sorted(set(new) & set(old)):
            lines.append(f"| {n} | {old[n]*1000:.0f} | {new[n]*1000:.0f} | "
                         f"{old[n]/new[n]:.1f}× |")
        for n in sorted(set(new) - set(old)):
            lines.append(f"| {n} | — | {new[n]*1000:.0f} | — |")
        ns = np.array(sorted(new))
        ts = np.array([new[n] for n in ns])
        slope = np.polyfit(np.log(ns), np.log(ts), 1)[0]
        lines.append(f"\n拟合标度指数（新实现，N=200–1600）：**{slope:.2f}**")
        out["throughput"] = {"new": new, "legacy": old, "scaling_exponent": float(slope)}

    # ---------------- finite size --------------------------------------
    rows = load("e1a")
    if rows:
        lines += ["", "## 5. 有限尺寸（Fig. 2）", "",
                  "| L/σ | φ_c(0.5) |", "|---|---|"]
        for L in sorted({r["L"] for r in rows}):
            sub = [r for r in rows if r["L"] == L]
            phi = np.array([r["phi"] for r in sub])
            k = np.array([r["k"] for r in sub])
            n = np.array([r["runs"] for r in sub])
            try:
                f = fitmod.fit_logistic(phi, k, n)
                lines.append(f"| {L/400:.0f} | {f['phi_c']:.4f} |")
            except ValueError:
                lines.append(f"| {L/400:.0f} | 拟合失败 |")

    md = "\n".join(lines) + "\n"
    with open(os.path.join(RESULTS, "numbers.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    with open(os.path.join(RESULTS, "numbers.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False, default=float)
    print(md)
    print("-> results/numbers.md, results/numbers.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
