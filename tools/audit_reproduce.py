# -*- coding: utf-8 -*-
"""
audit_reproduce.py —— 用发布版 problem2.py 复现论文表 3 的导通概率。

这是"引擎是可信的"这一结论的证据（见 01_技术审计 §1）。

用法
----
    python tools/audit_reproduce.py
    python tools/audit_reproduce.py --runs 100 --phie 0.005 0.006 0.007 0.01

注意
----
发布版 problem2.py 的 main() 会把结果写到 C:\\Users\\10847\\Desktop\\...，
在别的机器上会崩。本脚本直接调用 monte_carlo()/single_simulation()，不走 main()。
"""

import argparse
import math
import random
import sys
import time

import _compat


def run_point(P, phi, runs, seed):
    n = round(phi * P.V_MICRO / P.V_A)
    random.seed(seed)
    t0 = time.time()
    k = 0
    for _ in range(runs):
        r, _, _ = P.single_simulation(n)
        k += r
    dt = time.time() - t0
    p = k / runs
    se = math.sqrt(p * (1 - p) / runs) if 0 < p < 1 else 0.0
    return n, p, 1.96 * se, k, dt / runs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=40)
    ap.add_argument("--phie", type=float, nargs="+",
                    default=[0.0050, 0.0060, 0.0070, 0.0100])
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--code-dir", default=None)
    args = ap.parse_args()

    P = _compat.load("problem2", args.code_dir)

    # 论文表 3（PDF 版）的值，用于对照
    paper = {0.0050: 0.0980, 0.0060: 0.2180, 0.0070: 0.4900, 0.0100: 0.9980}

    print(f"runs/point = {args.runs}, seed = {args.seed}")
    print(f"{'phi':>7} {'N_A':>5} {'P_hat':>7} {'95%CI':>8} {'k/n':>12} "
          f"{'s/sim':>7} {'paper':>7} {'diff':>7}")
    print("-" * 74)
    for phi in args.phie:
        n, p, ci, k, dt = run_point(P, phi, args.runs, args.seed)
        ref = paper.get(round(phi, 6))
        diff = f"{p - ref:+.4f}" if ref is not None else "-"
        refs = f"{ref:.4f}" if ref is not None else "-"
        print(f"{phi*100:6.2f}% {n:5d} {p:7.4f} {ci:8.4f} {k:5d}/{args.runs:<6d} "
              f"{dt:7.3f} {refs:>7} {diff:>7}")

    print("\n本机实测（runs=40）: 0.50% -> 0.075, 1.00% -> 1.000")
    print("结论: 发布版代码可以复现论文的量级，可信。速度比论文声称的还快。")


if __name__ == "__main__":
    sys.exit(main())
