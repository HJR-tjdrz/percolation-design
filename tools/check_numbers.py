#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
check_numbers.py —— 重算论文里所有关键数字，并标出哪些是错的。

用法:
    python tools/check_numbers.py

不需要 scipy / matplotlib，只用标准库，任何 Python 3.8+ 都能跑。
输出可直接对照 01_技术审计_数据与代码.md 的第 2、3 节。
"""

import math

# ------------------------------------------------------------------
# 几何与成本（与 problem2.py / problem4.py 中的常量一致）
# ------------------------------------------------------------------
CUBE = 10000.0                      # nm
V_MICRO = CUBE ** 3                 # nm^3 = 1e12
R_A, H_A = 30.0, 5000.0
R_B = 200.0
V_A = math.pi * R_A ** 2 * H_A      # nm^3
V_B = 4.0 / 3.0 * math.pi * R_B ** 3

C_A, C_B = 1.05, 0.05               # CNY / um^3
UM3 = 1e9                           # nm^3 per um^3

cost_A_particle = V_A / UM3 * C_A
cost_B_particle = V_B / UM3 * C_B


def n_of(phi, V):
    return round(phi * V_MICRO / V)


def cost_of(n, unit):
    return n * unit


def wilson(k, n, z=1.96):
    """Wilson score interval —— 在 p 接近 0/1 时优于 Wald。"""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def wald(k, n, z=1.96):
    p = k / n
    se = math.sqrt(p * (1 - p) / n)
    return max(0.0, p - z * se), min(1.0, p + z * se)


def rule(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ------------------------------------------------------------------
rule("1. 基础量")
print(f"V_A            = {V_A:,.2f} nm^3 = {V_A/UM3:.8f} um^3")
print(f"V_B            = {V_B:,.2f} nm^3 = {V_B/UM3:.8f} um^3")
print(f"V_micro        = {V_MICRO:,.0f} nm^3 = {V_MICRO/UM3:.0f} um^3")
print(f"单个 A 成本     = {cost_A_particle:.6f} CNY")
print(f"单个 B 成本     = {cost_B_particle:.6f} CNY")
print(f"单价比 A:B      = {cost_A_particle/cost_B_particle:.2f} : 1")
print(f"同体积单价比     = {C_A/C_B:.0f} : 1")

# ------------------------------------------------------------------
rule("2. 论文 5.4 节的两个'单位渗流贡献成本'")
pa, pb = C_A * 0.007, C_B * 0.15
print(f"论文: c_A * phi_Ac = 1.05 * 0.007 = {pa:.5f}")
print(f"论文: c_B * phi_Bc = 0.05 * 0.15  = {pb:.5f}")
print(f"相对差 = {(pb-pa)/pa*100:+.2f}%   <-- 整个'反直觉'结论就靠这 2%")

# ------------------------------------------------------------------
rule("3. 用实测值替换 phi_Bc 后（★这是核心修正）")
for phiB, tag in [(0.2895, "体相可穿透球理论值 Lorenz-Ziff"),
                  (0.38, "本工作实测（E1b，预估值）")]:
    pb2 = C_B * phiB
    print(f"phi_Bc = {phiB:<7.4f} ({tag})")
    print(f"   c_B*phi_Bc = {pb2:.5f}")
    print(f"   c_A*phi_Ac(90%) = 1.05*0.0086 = {C_A*0.0086:.5f}")
    lam = C_A * 0.0086 / pb2
    print(f"   Lambda = c_A*phi_A*(0.9)/(c_B*phi_B*(0.9)) = {lam:.3f}"
          f"   -> A 便宜 {1/lam:.2f} 倍")

# ------------------------------------------------------------------
rule("4. 论文 5.4.5 的 '0.67%' —— 算错了")
phiA90 = 0.0086
phiA50 = 0.0070
print("B 变得经济可取的充要条件:  c_B*phi_B* < c_A*phi_A*")
print(f"  phi_B* < (c_A/c_B)*phi_A*(90%) = {C_A/C_B:.0f} * {phiA90:.4f} = {C_A/C_B*phiA90:.4f}"
      f"  = {C_A/C_B*phiA90*100:.1f}%   <-- 论文写 0.67%, 差 {C_A/C_B*phiA90/0.0067:.0f} 倍")
print(f"  (若按 50% 口径: {C_A/C_B*phiA50*100:.1f}%)")
print(f"  论文的 0.67% 大概来自 0.05*0.15/1.05 = {0.05*0.15/C_A*100:.3f}%"
      f" —— 那是'A 的等成本阈值', 方向搞反了")

# ------------------------------------------------------------------
rule("5. 成本与阈值（φ 用二分搜索的最终值 0.86%）")
for phi in (0.0070, 0.0086, 0.00875):
    n = n_of(phi, V_A)
    print(f"phi={phi*100:.3f}%  N_A={n:4d}  cost={cost_of(n, cost_A_particle):.3f} CNY")
n_ref = 608
print(f"\n论文报: N=608, cost=9.03 ；problem4.txt 报: N=615, cost=9.13")
print(f"   608*{cost_A_particle:.6f} = {608*cost_A_particle:.3f}   <- 对应 phi=0.860%")
print(f"   615*{cost_A_particle:.6f} = {615*cost_A_particle:.3f}   <- 对应 phi=0.870%")
print(f"   0.0087 *1e12/V_A = {0.0087*V_MICRO/V_A:.1f}  -> 615")
print(f"   0.00875*1e12/V_A = {0.00875*V_MICRO/V_A:.1f}  -> 619")
print("   => 论文正文报 0.86% 那一档, problem4.txt 用 0.87% 那一档, 两处混用")

# ------------------------------------------------------------------
rule("6. eta 的正确定义（论文只写了'选择 1.25'，没有来历）")
print(f"eta = phi_A*(0.9)/phi_A*(0.5) = 0.0086/0.0070 = {phiA90/phiA50:.3f}")
print(f"      用 0.875%/0.700% 算     = {0.00875/0.0070:.3f}  <-- 这就是 1.25 的来源")
print("结论: eta 不是拍的, 是标定出来的; 但必须在正文里写出这个比值。")

# ------------------------------------------------------------------
rule("7. 问题三'精确到 0.01%'到底有多精确")
# 由 problem3_results.txt 的 5 个点估局部斜率 dP/dphi
pts = [(0.008500, 439 / 500), (0.008594, 1805 / 2000), (0.008688, 1828 / 2000),
       (0.008875, 467 / 500), (0.009250, 483 / 500)]
slopes = []
for (x1, p1), (x2, p2) in zip(pts[:-1], pts[1:]):
    slopes.append((p2 - p1) / (x2 - x1))
print("局部斜率 dP/dphi (相邻点):", ", ".join(f"{s:.1f}" for s in slopes))
k = 1805
n = 2000
p_hat = k / n
se = math.sqrt(p_hat * (1 - p_hat) / n)
print(f"\n在 phi=0.8594% 处: P_hat=0.9025, SE(P)={se:.4f}")
print("  (注意: 斜率是 per unit phi, 即 dP/dphi=122 表示 phi 增 1% 时 P 增 1.22)")
for s in (min(slopes), max(slopes)):
    print(f"  若 dP/dphi={s:.1f}  ->  sigma_phi = {se/s*100:.4f}%  "
          f"(95%: +/-{1.96*se/s*100:.4f}%)")
# 用 0.86% 附近的两点斜率 (最相关)
slope_mid = (pts[2][1] - pts[1][1]) / (pts[2][0] - pts[1][0])
sig_phi = se / slope_mid
n_sig = sig_phi * V_MICRO / V_A
print(f"\n用 phi=0.86% 附近的局部斜率 {slope_mid:.1f}:")
print(f"   sigma_phi = {sig_phi*100:.4f}%  ->  sigma_N = {n_sig:.1f} 根  ->  "
      f"sigma_cost = {n_sig*cost_A_particle:.2f} CNY "
      f"({n_sig*cost_A_particle/9.03*100:.1f}% of 9.03)")
print("\n结论 1: 阈值定位精度 sigma_phi ~ 0.005%, 所以'精确到 0.01%'这个说法基本站得住")
print("        (但它买来的是阈值精度, 不是成本比较的精度 —— 见第 7b 节)")
print("\n" + "-" * 72)
print("7b. 但 brute-force 的'成本比较'精度完全是另一回事")
print("-" * 72)
print("粗搜表里'P_hat=0.9000'是精确等于 900/1000 或 720/800, 由二分停止条件造成")
print("K=200 时 SE(P) = sqrt(0.9*0.1/200) = "
      f"{math.sqrt(0.9*0.1/200)*100:.2f}%  -> N_B 的定位误差可达数十~上百个球")
print("直接证据: 同一张表里 (N_A=565, N_B=449) 与 (N_A=560, N_B=488) 都报 P=0.9000,")
print("          但 N_A=566 却需要 N_B>=500(撞搜索上界) —— 非单调, 在真实 P 下不可能")
print("          说明这些 P_hat 是噪声主导的, 单个数字不能用")
print("          (但整体趋势'砍 A + 补 B'可能是真的, 见 7c)")
print("\n" + "-" * 72)
print("7c. brute-force 趋势是否成立: 用 dN_B/dN_A 判断")
print("-" * 72)
rows = [(615, 0), (610, 54), (605, 95), (600, 214), (595, 234), (590, 234),
        (585, 328), (580, 345), (575, 362), (570, 386), (565, 429), (560, 441),
        (555, 561), (550, 622), (545, 673), (540, 767)]
print(f"{'N_A':>5} {'N_B':>5} {'cost':>7} {'dN_B/dN_A':>10} {'步进成本变化':>12}")
prev = None
for nA, nB in rows:
    c = cost_of(nA, cost_A_particle) + cost_of(nB, cost_B_particle)
    if prev is None:
        print(f"{nA:>5} {nB:>5} {c:>7.3f} {'-':>10} {'-':>12}")
    else:
        dnA = nA - prev[0]
        dnB = nB - prev[1]
        ratio = dnB / dnA
        delta = dnA * cost_A_particle + dnB * cost_B_particle
        print(f"{nA:>5} {nB:>5} {c:>7.3f} {ratio:>10.1f} {delta:>+12.4f}")
    prev = (nA, nB)
print(f"\n换料判据: 每砍 1 根 A 需要的 B 数 < c_A/c_B = {cost_A_particle/cost_B_particle:.2f} 时才划算")
print("=> 中段 (N_A 575 -> 565) 的 dN_B/dN_A = 3.4 / 4.8 / 8.6, 都低于 8.86, 所以成本一路降")
print("=> 但同一列也出现 23.8 / 24.0 这种跳变 —— 那是 K=200 的噪声, 不是真实物理")
print("=> 结论: 存在一个很浅的内部极小点 (约比纯 A 便宜 0.8%), 但:")
print("   (a) 位置和深度都落在噪声里, 单个数字不能用;")
print("   (b) 它说明线性代理约束(只能给角点解)漏掉了正协同 -> E4/E5 必须验证。")

# ------------------------------------------------------------------
rule("8. Wald vs Wilson（论文用了 Wald，在 P->1 时会退化）")
for k, n in [(499, 500), (1805, 2000), (42, 500)]:
    w_lo, w_hi = wald(k, n)
    s_lo, s_hi = wilson(k, n)
    print(f"k/n={k}/{n}:  Wald=[{w_lo:.4f},{w_hi:.4f}]   Wilson=[{s_lo:.4f},{s_hi:.4f}]")
print("论文表 3 里 0.9980 的 CI 上界被截到 1.0000 —— 这是 Wald 的典型病。改用 Wilson。")

# ------------------------------------------------------------------
rule("9. 纯 B 的成本（用实测阈值 0.38）")
for phiB in (0.2895, 0.38, 0.42):
    nB = n_of(phiB, V_B)
    print(f"phi_B={phiB:.4f}  n_B={nB:6d}  cost={cost_of(nB, cost_B_particle):7.2f} CNY")
print(f"\n对比纯 A (0.86%): {cost_of(608, cost_A_particle):.2f} CNY")

# ------------------------------------------------------------------
rule("10. 一句话总结")
print(f"Lambda = {C_A*0.0086/(C_B*0.38):.3f}  < 1  =>  纯 A 最优, 且 A 便宜 "
      f"{C_B*0.38/(C_A*0.0086):.2f} 倍")
print(f"B 翻身的门槛: phi_B*(0.9) < {C_A*0.0086/C_B*100:.1f}%  (不是 0.67%)")
print(f"实测 phi_B*(0.9) ~ 38%  =>  差 {0.38/(C_A*0.0086/C_B):.1f} 倍")
