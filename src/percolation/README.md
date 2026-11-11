# percolation

一个用于**薄层微构体贯穿型逾渗**的蒙特卡洛引擎（numpy 向量化）。

用于 ICCMS 2027 投稿工作的 E0 产物：把竞赛期间的 `problem1–4.py` 合并、修正、加速，
并提供可复现的命令行接口。

---

## 安装与测试

```bash
pip install -r requirements.txt      # numpy 即可；测试可选 pytest

# 测试（有 pytest 就用 pytest，没有就用自带 runner）
python src/percolation/tests/run_tests.py --all
```

## 30 秒上手

```python
import numpy as np
from percolation import Model, simulate, estimate, cost_of

m = Model()                                  # 10 um 立方体，5 um x 60 nm 棒 + 400 nm 球
simulate(m, n_rod=500).conducting            # 一次实现
estimate(m, phi_rod=0.0074, n_runs=2000, seed=0)   # 概率 + Wilson 区间
```

## 模型

| 项 | 取值 |
|---|---|
| 微构体 | 边长 `L = 10000 nm` 的立方体 |
| 电极 | `x = ±L/2`；其余四面绝缘 |
| 介质 A | 胶囊体（capsule），轴长 5000 nm，半径 30 nm |
| 介质 B | 球体，半径 200 nm |
| 接触判据 | 表面最短距离 ≤ 1.8 nm |
| 填料分布 | 中心均匀、取向各向同性、允许重叠（soft-core） |

距离核：
`d_AA = max(0, d_seg − 2r_A)`，`d_AS = max(0, d_pt−seg − r_A − r_B)`，`d_SS = max(0, |Δc| − 2r_B)`。

## 边界条件（这是本引擎相对竞赛代码的主要修正）

| 面 | 处理 |
|---|---|
| 横向 y, z | 周期（最小镜像） |
| 纵向 x（电极） | **截断**：穿过电极的填料不再出现在对面电极上 |

`Model(rod_x="wrap")` 保留了竞赛代码的做法（把穿出部分镜像到对面电极），
只用于复现与对照。**它不是物理正确的电极处理**，见论文的边界条件一节。

`Model(lateral="walls")` 关闭横向周期性（绝缘硬壁），用于敏感性分析。

## 相对竞赛代码改了什么

1. **修**：横向切开后，同一根棒的碎片没有并回同一个连通分量 —— 约四分之一的棒被
   人为截短。新引擎用最小镜像处理，棒保持完整。
2. **修**：穿过电极的填料被镜像到对面电极，凭空生成贴在电极上的导体。
3. **修**：球体完全不做边界处理（与棒的环绕不对称）。
4. **修**：置信区间用 Wilson（原来用 Wald，在 P→1 时上界被截到 1.0000）。
5. **快**：O(N²) 纯 Python 双重循环 → 向量化 + 按 x 排序的滑动窗口。
   实测 0.04–0.15 s/次（ρ_rod = 0.5%–0.9%），比原来快 5–15 倍。
6. **加**：`Model` 集中参数，结果可写入 JSON（含 seed / 用时 / 版本）。

## 目录

```
src/percolation/
├── geometry.py     # 线段–线段、点–线段、点–点距离（向量化，对拍标量实现）
├── engine.py       # 采样、边界、邻域搜索、并查集、导通判定
├── estimate.py     # 概率估计、Wilson 区间、成本
└── tests/
    ├── test_geometry.py   # 与 problem2.py 的标量实现逐点对拍（1e-12 容差）
    ├── test_engine.py     # 物理 sanity + 边界语义 + 粗回归
    └── run_tests.py       # 没有 pytest 时用的极简 runner
```

## 已验证

- `segment_segment_distance` 与竞赛代码的标量实现（Ericson 分支）在 3000 组随机线段上
  **完全一致**（rtol 1e-12）。
- `point_segment_distance` 在 2000 组随机点上完全一致。
- 取向采样的各向同性：`E|u_x| = 0.5`、`E u_x² = 1/3`（40000 个样本内误差 < 0.01）。
- 与竞赛报告 Problem-2 曲线在同一边界约定下一致（legacy 模式）。

## 还没有做（列出来以免被误认为已做）

- 电导率计算（当前只判通/不通）—— 不在本文范围
- 硬核（不可重叠）模型 —— 当前是 soft-core
- 球–球的 cell list —— 目前用按 x 排序的窗口，n ≈ 1.1e4 时约 0.3 s/次
- 并行 —— 单进程；实验靠多进程脚本并行

## 引用

若使用本代码，请引用随论文提交的 Zenodo DOI（E8 阶段生成）。
