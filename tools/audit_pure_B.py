# -*- coding: utf-8 -*-
"""
audit_pure_B.py —— 纯球体（介质 B）的贯穿型逾渗阈值扫描。

这是整篇论文的地基：论文用的 phi_B,c = 15% 来自文献，而发布版程序从未模拟过纯 B。
本脚本用 problem4.single_simulation_mixture(0, nB) 扫一遍。

用法
----
    # 先摸底（快，每个点 4-8 次）
    python tools/audit_pure_B.py --runs 4 --phie 0.15 0.22 0.28 0.34

    # 正式标定（慢！发布版球-球是 O(n^2) 纯 Python 循环，34% 时约 39 s/次）
    python tools/audit_pure_B.py --runs 200 --phie 0.24 0.26 0.28 0.30 0.32 0.34 0.36 0.38

性能警告
--------
发布版 problem4.py 的球-球判定是双重 Python 循环，n_B=10146 时约 39 s/次。
    34% x 200 次 ≈ 2.2 小时;  38% x 200 次 ≈ 3.0 小时。
正式标定前请先做 E0（向量化 + cell list），可提速 100 倍以上。

本机实测（runs=4~8, seed=2026）
------------------------------
    phi_B = 15%   n_B= 4476   P = 0.00   7.0 s/sim
    phi_B = 22%   n_B= 6565   P = 0.00  21.9 s/sim
    phi_B = 28%   n_B= 8356   P = 0.00  35.6 s/sim
    phi_B = 34%   n_B=10146   P = 0.25  38.9 s/sim
=> 论文的 15% 与自家模型差了一个数量级;  贯穿型阈值约在 34%-40%。
"""

import argparse
import math
import random
import sys
import time

import _compat


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--phie", type=float, nargs="+",
                    default=[0.15, 0.22, 0.28, 0.34])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--code-dir", default=None)
    ap.add_argument("--yes", action="store_true", help="跳过耗时确认")
    args = ap.parse_args()

    Q = _compat.load("problem4", args.code_dir)

    total = 0.0
    plan = []
    for vB in args.phie:
        nB = round(vB * Q.V_MICRO / Q.V_B)
        # 经验耗时：n^2 * 3.8e-7 s
        est = nB * nB * 3.8e-7
        plan.append((vB, nB, est))
        total += est * args.runs

    print(f"V_B = {Q.V_B:.3e} nm^3,  box = {Q.V_MICRO:.3e} nm^3")
    print(f"{'phi_B':>7} {'n_B':>7} {'est s/sim':>10}")
    for vB, nB, est in plan:
        print(f"{vB*100:6.1f}% {nB:7d} {est:10.1f}")
    print(f"总计约 {total/60:.0f} 分钟（{args.runs} runs/point）")

    if total > 1800 and not args.yes:
        print("\n[!] 预计超过 30 分钟。确认请加 --yes，或先做 E0 向量化。")
        return 1

    print()
    print(f"{'phi_B':>7} {'n_B':>7} {'P_hat':>7} {'95%CI':>8} {'k/n':>10} {'s/sim':>8}")
    print("-" * 56)
    for vB, nB, _ in plan:
        random.seed(args.seed)
        t0 = time.time()
        k = 0
        for _ in range(args.runs):
            k += Q.single_simulation_mixture(0, nB)
        dt = (time.time() - t0) / args.runs
        p = k / args.runs
        se = math.sqrt(p * (1 - p) / args.runs) if 0 < p < 1 else 0.0
        print(f"{vB*100:6.1f}% {nB:7d} {p:7.4f} {1.96*se:8.4f} "
              f"{k:3d}/{args.runs:<6d} {dt:8.1f}", flush=True)

    print("\n提醒：发布版对球体不做任何边界处理（不 wrap、不 clip），")
    print("      而圆柱做三轴环绕。这个不对称会同时压低 B 的连通性、抬高 A 的连通性。")
    print("      E1b 必须用统一、物理正确的边界重跑。")


if __name__ == "__main__":
    sys.exit(main())
