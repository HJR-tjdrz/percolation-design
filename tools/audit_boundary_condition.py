# -*- coding: utf-8 -*-
"""
audit_boundary_condition.py —— 对照"三轴环绕"与"电极截断"两种边界条件下的导通概率。

背景
----
发布版 apply_boundary_truncation() 在 x, y, z 三个轴上都做"切开 + 搬到对面"。
但 x = ±5000 是电极：一根圆柱从右电极穿出后，其外侧部分被平移到 x=-5000，
正好贴在左电极上。这等于凭空给左电极接了一段导体（见 01_技术审计 §S3）。

本脚本把边界处理替换成物理一致的版本（横向周期、纵向在电极处截断），
其余一切不变（同一随机种子、同一几何、同一导通判据），直接比较 P̂。

用法
----
    python tools/audit_boundary_condition.py            # 默认 150 次/点
    python tools/audit_boundary_condition.py --runs 500 --phie 0.006 0.007 0.008

输出
----
    每个体积分数下两种边界处理的 P̂ ± 95%CI，以及耗时。
    结果可直接填进 01_技术审计 §S3 的表，或论文的 Table 2。
"""

import argparse
import math
import random
import sys
import time

import _compat


# ----------------------------------------------------------------------
# 电极截断版边界处理
# ----------------------------------------------------------------------
def make_clip_x(module):
    """返回一个 apply_boundary_truncation 的替代实现：x 截断、y/z 周期。"""
    P = module
    HALF = P.CUBE_HALF
    FULL = P.CUBE_FULL

    def _shift(seg, axis, s):
        if axis == 1:
            return P.Segment(seg.x1, seg.y1 + s, seg.z1, seg.x2, seg.y2 + s, seg.z2)
        return P.Segment(seg.x1, seg.y1, seg.z1 + s, seg.x2, seg.y2, seg.z2 + s)

    def _axis(seg, axis):
        """y 或 z 轴的切开+环绕（复用原模块的切割函数）。"""
        half = HALF
        a1, a2 = (seg.y1, seg.y2) if axis == 1 else (seg.z1, seg.z2)
        if -half <= a1 <= half and -half <= a2 <= half:
            return [seg]
        tp = []
        for b in (-half, half):
            den = a2 - a1
            if abs(den) > 1e-20:
                t = (b - a1) / den
                if 0.0 < t < 1.0:
                    tp.append((t, b))
        tp.sort()
        if len(tp) == 0:
            s = -FULL if a1 > half else FULL
            return [_shift(seg, axis, s)]
        if len(tp) == 1:
            t0 = tp[0][0]
            inner, outer = P._cut_seg(seg, t0, axis, a1, half)
            return [inner, P._wrap_seg(outer, axis)]
        t0, t1 = tp[0][0], tp[1][0]
        return [P._wrap_seg(P._cut_first(seg, t0), axis),
                P._cut_segment_by_params(seg, t0, t1),
                P._wrap_seg(P._cut_second(seg, t1), axis)]

    def _clip_x(seg):
        """把线段裁到 x ∈ [-HALF, HALF]；完全在外则丢弃。"""
        x1, x2 = seg.x1, seg.x2
        if (x1 < -HALF and x2 < -HALF) or (x1 > HALF and x2 > HALF):
            return []
        ts = [0.0, 1.0]
        den = x2 - x1
        if abs(den) > 1e-20:
            for b in (-HALF, HALF):
                t = (b - x1) / den
                if 0.0 < t < 1.0:
                    ts.append(t)
        ts.sort()
        for a, b in zip(ts[:-1], ts[1:]):
            xm = x1 + 0.5 * (a + b) * den
            if -HALF - 1e-9 <= xm <= HALF + 1e-9:
                pa = P.Segment(x1 + a * den, seg.y1 + a * (seg.y2 - seg.y1),
                               seg.z1 + a * (seg.z2 - seg.z1), 0, 0, 0)
                pb = P.Segment(x1 + b * den, seg.y1 + b * (seg.y2 - seg.y1),
                               seg.z1 + b * (seg.z2 - seg.z1), 0, 0, 0)
                return [P.Segment(pa.x1, pa.y1, pa.z1, pb.x1, pb.y1, pb.z1)]
        return []

    def trunc(cyl):
        segs = _clip_x(cyl.seg)
        for axis in (1, 2):
            nxt = []
            for s in segs:
                nxt.extend(_axis(s, axis))
            segs = nxt
        return segs

    return trunc


# ----------------------------------------------------------------------
def run(module, phi, runs, seed):
    n = round(phi * module.V_MICRO / module.V_A)
    random.seed(seed)
    t0 = time.time()
    k = 0
    for _ in range(runs):
        r, _, _ = module.single_simulation(n)
        k += r
    dt = time.time() - t0
    p = k / runs
    se = math.sqrt(p * (1 - p) / runs) if 0 < p < 1 else 0.0
    return n, p, 1.96 * se, k, dt


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=150)
    ap.add_argument("--phie", type=float, nargs="+", default=[0.0060, 0.0070])
    ap.add_argument("--seed", type=int, default=424242)
    ap.add_argument("--code-dir", default=None)
    args = ap.parse_args()

    P = _compat.load("problem2", args.code_dir)
    original = P.apply_boundary_truncation
    clip = make_clip_x(P)

    print(f"runs/point = {args.runs}, seed = {args.seed}")
    print(f"{'phi':>8} {'N':>5} {'boundary':<26} {'P_hat':>7} {'95%CI':>8} {'k/n':>11} {'sec':>6}")
    print("-" * 78)

    for phi in args.phie:
        for label, fn in (("wrap_all (as released)", original),
                          ("clip_x   (electrode)", clip)):
            P.apply_boundary_truncation = fn
            n, p, ci, k, dt = run(P, phi, args.runs, args.seed)
            print(f"{phi*100:7.2f}% {n:5d} {label:<26} {p:7.4f} {ci:8.4f} "
                  f"{k:5d}/{args.runs:<5d} {dt:6.1f}")
        print("-" * 78)

    P.apply_boundary_truncation = original
    print("\n预期（本机实测）:")
    print("  0.60%: wrap 0.2067 +/-0.0648  vs  clip 0.1933 +/-0.0632")
    print("  0.70%: wrap 0.5000 +/-0.0800  vs  clip 0.4533 +/-0.0797")
    print("=> 环绕版系统性高估 P，阈值相对偏移约 1%-2%。方向确认，幅度不大。")


if __name__ == "__main__":
    sys.exit(main())
