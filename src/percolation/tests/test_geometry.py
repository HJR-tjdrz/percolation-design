# -*- coding: utf-8 -*-
"""Kernel tests against an independent scalar implementation.

The scalar routines below are copied from the competition code
(``problem2.py``: ``dist_segment_to_segment`` / ``dist_point_segment``) so that
the vectorised kernels are checked against the implementation that produced the
published numbers, not against themselves.
"""

import numpy as np

try:  # real pytest when available
    import pytest
except ImportError:  # pragma: no cover - fallback for a bare interpreter
    import _pytest_compat as pytest

from percolation.geometry import point_point_distance, point_segment_distance, segment_segment_distance


# ---------------------------------------------------------------------------
# scalar reference implementation (verbatim logic from problem2.py)
# ---------------------------------------------------------------------------
def _clamp(v, lo=0.0, hi=1.0):
    return lo if v < lo else (hi if v > hi else v)


def ref_seg_seg(s1, s2):
    (x1, y1, z1), (x2, y2, z2) = s1
    (x3, y3, z3), (x4, y4, z4) = s2
    ux, uy, uz = x2 - x1, y2 - y1, z2 - z1
    vx, vy, vz = x4 - x3, y4 - y3, z4 - z3
    wx, wy, wz = x1 - x3, y1 - y3, z1 - z3
    a = ux * ux + uy * uy + uz * uz
    b = ux * vx + uy * vy + uz * vz
    c = vx * vx + vy * vy + vz * vz
    d = ux * wx + uy * wy + uz * wz
    e = vx * wx + vy * wy + vz * wz
    denom = a * c - b * b

    def sqd(s, t):
        cx = (x1 + s * ux) - (x3 + t * vx)
        cy = (y1 + s * uy) - (y3 + t * vy)
        cz = (z1 + s * uz) - (z3 + t * vz)
        return cx * cx + cy * cy + cz * cz

    if denom > 1e-20:
        s = (b * e - c * d) / denom
        t = (a * e - b * d) / denom
        if 0.0 <= s <= 1.0 and 0.0 <= t <= 1.0:
            return np.sqrt(sqd(s, t))

    best = float("inf")
    for tt in (_clamp(e / c), _clamp((e + b) / c)):
        best = min(best, sqd(0.0, tt), sqd(1.0, tt))
    for ss in (_clamp(-d / a), _clamp((b - d) / a)):
        best = min(best, sqd(ss, 0.0), sqd(ss, 1.0))
    for ss in (0.0, 1.0):
        for tt in (0.0, 1.0):
            best = min(best, sqd(ss, tt))
    return np.sqrt(best)


def ref_point_seg(p, a, b):
    ab = np.subtract(b, a)
    ap = np.subtract(p, a)
    t = float(np.dot(ap, ab) / max(np.dot(ab, ab), 1e-20))
    t = _clamp(t)
    closest = np.add(a, t * ab)
    return float(np.linalg.norm(np.subtract(p, closest)))


# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(20261104)


def test_segment_distance_matches_reference(rng):
    n = 3000
    pts = rng.uniform(-5000, 5000, size=(n, 4, 3))
    vec = segment_segment_distance(pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3])
    ref = np.array([ref_seg_seg((pts[i, 0], pts[i, 1]), (pts[i, 2], pts[i, 3]))
                    for i in range(n)])
    assert np.allclose(vec, ref, rtol=1e-12, atol=1e-9), (
        f"max abs diff = {np.abs(vec - ref).max():.3e}")


def test_segment_distance_degenerate_and_parallel(rng):
    # zero-length segments (point-point), exactly parallel, exactly crossing
    pts = rng.uniform(-100, 100, size=(200, 3))
    same = np.einsum("ij,ij->i", pts, pts) ** 0.5
    got = segment_segment_distance(pts, pts, np.zeros_like(pts), np.zeros_like(pts))
    assert np.allclose(got, same, rtol=1e-12, atol=1e-9)

    a = np.array([[0.0, 0.0, 0.0]])
    b = np.array([[10.0, 0.0, 0.0]])
    c = np.array([[0.0, 5.0, 0.0]])
    d = np.array([[10.0, 5.0, 0.0]])
    assert abs(float(segment_segment_distance(a, b, c, d)) - 5.0) < 1e-12

    # crossing segments -> zero distance
    e = np.array([[0.0, -5.0, 0.0]])
    f = np.array([[0.0, 5.0, 0.0]])
    assert float(segment_segment_distance(a, b, e, f)) < 1e-12


def test_point_segment_matches_reference(rng):
    n = 2000
    p = rng.uniform(-5000, 5000, size=(n, 3))
    a = rng.uniform(-5000, 5000, size=(n, 3))
    b = rng.uniform(-5000, 5000, size=(n, 3))
    got = point_segment_distance(p, a, b)
    ref = np.array([ref_point_seg(p[i], a[i], b[i]) for i in range(n)])
    assert np.allclose(got, ref, rtol=1e-12, atol=1e-9)


def test_known_values():
    assert abs(float(point_point_distance(np.zeros(3), np.array([3.0, 4.0, 0.0]))) - 5.0) < 1e-12
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([10.0, 0.0, 0.0])
    assert abs(float(point_segment_distance(np.array([5.0, 0.0, 0.0]), a, b))) < 1e-12
    assert abs(float(point_segment_distance(np.array([-2.0, 0.0, 0.0]), a, b)) - 2.0) < 1e-12
