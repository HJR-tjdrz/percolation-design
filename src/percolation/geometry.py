# -*- coding: utf-8 -*-
"""Vectorised geometry kernels for soft-core continuum percolation.

All functions are written for broadcasting: shapes such as ``(..., 3)`` for
vectors and ``(...)`` for scalars.  Nothing here allocates O(N^2) memory on its
own; the caller blocks the work.

Conventions
-----------
A spherocylinder (filler A) is represented by its axis end points ``p1`` and
``p2`` plus a radius ``r``.  The surface-to-surface distance between two such
capsules is ``max(0, d_axis - r_i - r_j)``.  A sphere (filler B) is a point
plus a radius.

The "soft-core" (penetrable) model is used throughout: particles may overlap,
which is the standard idealisation for randomly dispersed fillers at low
volume fraction.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "segment_segment_distance",
    "point_segment_distance",
    "point_point_distance",
]

_EPS = 1e-12


def segment_segment_distance(p1, p2, q1, q2):
    """Shortest distance between two 3-D segments.

    Parameters
    ----------
    p1, p2, q1, q2 : array_like, shape (..., 3)
        End points of the two segments.  All four must be broadcastable.

    Returns
    -------
    ndarray, shape (...)
        Euclidean distance between the closest points.

    Notes
    -----
    This is the branch-based algorithm of Ericson, *Real-Time Collision
    Detection*, section 5.1.9, vectorised with ``np.where``.  The unconstrained
    closest-point pair is computed first; then the four boundary cases
    (s = 0, s = 1, t = 0, t = 1) are handled in the same order as the scalar
    version, which guarantees identical results.
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)

    u = p2 - p1
    v = q2 - q1
    w = p1 - q1

    a = np.einsum("...i,...i->...", u, u)
    b = np.einsum("...i,...i->...", u, v)
    c = np.einsum("...i,...i->...", v, v)
    d = np.einsum("...i,...i->...", u, w)
    e = np.einsum("...i,...i->...", v, w)

    a = np.maximum(a, _EPS)
    c = np.maximum(c, _EPS)
    big = a * c - b * b

    parallel = big <= _EPS * np.maximum(a * c, 1.0)

    # unconstrained candidates
    s_n = np.where(parallel, 0.0, b * e - c * d)
    s_d = np.where(parallel, 1.0, big)
    t_n = np.where(parallel, e, a * e - b * d)
    t_d = np.where(parallel, c, big)

    # --- region s = 0 -------------------------------------------------
    m = s_n < 0.0
    s_n = np.where(m, 0.0, s_n)
    t_n = np.where(m, e, t_n)
    t_d = np.where(m, c, t_d)

    # --- region s = 1 -------------------------------------------------
    m = s_n > s_d
    s_n = np.where(m, s_d, s_n)
    t_n = np.where(m, e + b, t_n)
    t_d = np.where(m, c, t_d)

    # --- region t = 0 -------------------------------------------------
    m = t_n < 0.0
    sn_lo = np.where(-d < 0.0, 0.0, np.where(-d > a, s_d, -d))
    sd_lo = np.where((-d < 0.0) | (-d > a), s_d, a)
    s_n = np.where(m, sn_lo, s_n)
    s_d = np.where(m, sd_lo, s_d)
    t_n = np.where(m, 0.0, t_n)

    # --- region t = 1 -------------------------------------------------
    m = (t_n > t_d)
    q = -d + b
    sn_hi = np.where(q < 0.0, 0.0, np.where(q > a, s_d, q))
    sd_hi = np.where((q < 0.0) | (q > a), s_d, a)
    s_n = np.where(m, sn_hi, s_n)
    s_d = np.where(m, sd_hi, s_d)
    t_n = np.where(m, t_d, t_n)

    s = np.clip(s_n / s_d, 0.0, 1.0)
    t = np.clip(t_n / t_d, 0.0, 1.0)

    diff = (p1 + s[..., None] * u) - (q1 + t[..., None] * v)
    return np.sqrt(np.einsum("...i,...i->...", diff, diff))


def point_segment_distance(c, p1, p2):
    """Shortest distance from point(s) ``c`` to segment(s) ``p1``-``p2``."""
    c = np.asarray(c, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    u = p2 - p1
    w = c - p1
    a = np.maximum(np.einsum("...i,...i->...", u, u), _EPS)
    t = np.clip(np.einsum("...i,...i->...", w, u) / a, 0.0, 1.0)
    diff = w - t[..., None] * u
    return np.sqrt(np.einsum("...i,...i->...", diff, diff))


def point_point_distance(p, q):
    """Euclidean distance between broadcastable point arrays."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    d = p - q
    return np.sqrt(np.einsum("...i,...i->...", d, d))
