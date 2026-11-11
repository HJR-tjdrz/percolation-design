# -*- coding: utf-8 -*-
"""Monte Carlo engine for through-thickness percolation in a thin micro-cell.

Model
-----
A cubic cell of side ``L``.  The faces at ``x = -L/2`` and ``x = +L/2`` are
electrodes; the four lateral faces are insulating.  Two fillers are supported:

* filler A -- a spherocylinder (capsule) of axis length ``h`` and radius ``r_a``;
* filler B -- a sphere of radius ``r_b``.

Two objects are electrically connected when the shortest distance between their
surfaces is ``<= delta``.  The cell conducts when the two electrodes are in the
same connected component.

Boundary treatment
------------------
Lateral faces (y, z) are periodic, implemented by the minimum-image convention
on the pair distances.  The electrode faces are **not** periodic: material that
crosses an electrode is inside the metal and contributes nothing beyond the
contact itself.  This is the physically consistent choice; a commonly used
alternative (mirroring the crossing part onto the opposite electrode) is
available as ``rod_x="wrap"`` for reproducing the original competition code and
is discussed in the paper's boundary-condition section.

Notes
-----
Particles are soft-core (penetrable) with uniform random centres and
isotropically sampled orientations -- the standard idealisation for randomly
dispersed fillers well below the packing limit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .geometry import point_point_distance, point_segment_distance, segment_segment_distance

__all__ = ["Model", "SimulationResult", "simulate", "n_for_fraction", "EQUIVALENT_SIZE"]

# Diameter of the sphere; used as the natural length unit for finite-size plots.
EQUIVALENT_SIZE = 400.0  # nm


@dataclass(frozen=True)
class Model:
    """Geometry and contact rule.  Distances in nm, costs in CNY/um^3."""

    L: float = 10000.0
    rod_length: float = 5000.0
    rod_radius: float = 30.0
    sphere_radius: float = 200.0
    delta: float = 1.8

    # boundary policy on the electrode axis: "clip" (default) or "wrap" (legacy)
    rod_x: str = "clip"
    sphere_x: str = "none"  # "none" == clipped at the electrode, same as clip

    # lateral faces: "periodic" (default, minimum image) or "walls" (insulating)
    lateral: str = "periodic"

    # ---- derived -------------------------------------------------------
    @property
    def half_L(self) -> float:
        return 0.5 * self.L

    @property
    def rod_half_length(self) -> float:
        return 0.5 * self.rod_length

    @property
    def volume_cell(self) -> float:
        return self.L ** 3

    @property
    def volume_rod(self) -> float:
        return np.pi * self.rod_radius ** 2 * self.rod_length

    @property
    def volume_sphere(self) -> float:
        return 4.0 / 3.0 * np.pi * self.sphere_radius ** 3

    # surface-to-surface cutoffs converted to axis distances
    @property
    def cutoff_aa(self) -> float:
        return 2.0 * self.rod_radius + self.delta

    @property
    def cutoff_as(self) -> float:
        return self.rod_radius + self.sphere_radius + self.delta

    @property
    def cutoff_ss(self) -> float:
        return 2.0 * self.sphere_radius + self.delta

    def __post_init__(self):
        if self.rod_x not in ("clip", "wrap"):
            raise ValueError(f"rod_x must be 'clip' or 'wrap', got {self.rod_x!r}")
        if self.lateral not in ("periodic", "walls"):
            raise ValueError(f"lateral must be 'periodic' or 'walls', got {self.lateral!r}")
        if self.rod_length > self.L:
            raise ValueError("rod_length must not exceed L (minimum-image assumption)")


@dataclass
class SimulationResult:
    conducting: bool
    n_rod: int = 0
    n_sphere: int = 0
    n_edges: int = 0
    n_contact_left: int = 0
    n_contact_right: int = 0
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# particle generation
# ---------------------------------------------------------------------------
def _isotropic_directions(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniformly distributed unit vectors on the sphere (no pole bias)."""
    if n == 0:
        return np.zeros((0, 3))
    cos_t = rng.uniform(-1.0, 1.0, size=n)
    sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t * cos_t))
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    return np.stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], axis=1)


def _random_rods(model: Model, n: int, rng: np.random.Generator):
    half = model.half_L
    centres = rng.uniform(-half, half, size=(n, 3))
    dirs = _isotropic_directions(n, rng)
    ext = model.rod_half_length
    p1 = centres - ext * dirs
    p2 = centres + ext * dirs
    return centres, dirs, p1, p2


def _random_spheres(model: Model, n: int, rng: np.random.Generator) -> np.ndarray:
    half = model.half_L
    return rng.uniform(-half, half, size=(n, 3))


def n_for_fraction(phi: float, model: Model, kind: str) -> int:
    """Number of particles that realises volume fraction ``phi``."""
    vol = model.volume_rod if kind == "rod" else model.volume_sphere
    return int(round(phi * model.volume_cell / vol))


# ---------------------------------------------------------------------------
# minimum image and candidate search
# ---------------------------------------------------------------------------
def _lateral_shift(ca: np.ndarray, cb: np.ndarray, L: float, periodic: bool = True) -> np.ndarray:
    """Integer-image shift (in nm) that maps ``cb`` next to ``ca`` laterally.

    Only y and z are periodic; x (the electrode axis) is left untouched.
    With ``periodic=False`` the lateral faces are insulating hard walls.
    """
    shift = np.zeros_like(cb)
    if not periodic:
        return shift
    dy = cb[:, 1] - ca[:, 1]
    dz = cb[:, 2] - ca[:, 2]
    shift[:, 1] = -L * np.round(dy / L)
    shift[:, 2] = -L * np.round(dz / L)
    return shift


def _pairs_by_x_window(xa: np.ndarray, xb: np.ndarray, reach: float):
    """Index pairs (ia, ib) with ``|xa - xb| <= reach``.

    The x axis is not periodic, so a one-dimensional sweep is exact.  This is
    what keeps the expensive distance kernels off the O(N^2) pair list.
    """
    if xa.size == 0 or xb.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    order = np.argsort(xb, kind="stable")
    xb_sorted = xb[order]
    lo = np.searchsorted(xb_sorted, xa - reach, side="left")
    hi = np.searchsorted(xb_sorted, xa + reach, side="right")
    counts = hi - lo
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    ia = np.repeat(np.arange(xa.size, dtype=np.int64), counts)
    # ragged range expansion
    offsets = np.arange(total, dtype=np.int64)
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    ib = order[np.repeat(lo, counts) + (offsets - starts)]
    return ia, ib


def _blocked_pairs(xa, xb, need, block_elems=1_500_000):
    """Yield ``(ia, ib)`` blocks whose x separation is below ``need``."""
    ia, ib = _pairs_by_x_window(xa, xb, need)
    n = ia.size
    if n == 0:
        return
    step = max(1, min(n, int(block_elems)))
    for start in range(0, n, step):
        sl = slice(start, min(start + step, n))
        yield ia[sl], ib[sl]


# ---------------------------------------------------------------------------
# union-find
# ---------------------------------------------------------------------------
class _DSU:
    __slots__ = ("parent",)

    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = int(p[x])
        return x

    def union(self, a, b) -> None:
        p = self.parent
        for u, v in zip(np.atleast_1d(a).tolist(), np.atleast_1d(b).tolist()):
            ru, rv = self.find(u), self.find(v)
            if ru != rv:
                p[rv] = ru

    def connected(self, a: int, b: int) -> bool:
        return self.find(a) == self.find(b)


# ---------------------------------------------------------------------------
# electrode contact
# ---------------------------------------------------------------------------
def _rod_face_contacts(p1, p2, model: Model):
    """(touch_left, touch_right) boolean arrays for the rod axes."""
    r, d, half = model.rod_radius, model.delta, model.half_L
    xmin = np.minimum(p1[:, 0], p2[:, 0])
    xmax = np.maximum(p1[:, 0], p2[:, 0])
    left = (xmin + half) <= (r + d)
    right = (half - xmax) <= (r + d)
    if model.rod_x == "wrap":
        # legacy: the part that exits one electrode is mirrored onto the other,
        # so a rod crossing +L/2 also touches the left electrode (and vice versa)
        left = left | ((half - xmax) <= (r + d))
        right = right | ((xmin + half) <= (r + d))
    return left, right


def _sphere_face_contacts(centres, model: Model):
    r, d, half = model.sphere_radius, model.delta, model.half_L
    reach = r + d
    left = (centres[:, 0] + half) <= reach
    right = (half - centres[:, 0]) <= reach
    return left, right


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def simulate(model: Model, n_rod: int, n_sphere: int = 0,
             rng: Optional[np.random.Generator] = None) -> SimulationResult:
    """Run one Monte Carlo realisation and return whether the cell conducts."""
    if rng is None:
        rng = np.random.default_rng()

    centres_a, dirs_a, p1_a, p2_a = _random_rods(model, n_rod, rng)
    centres_b = _random_spheres(model, n_sphere, rng)

    n_a = p1_a.shape[0]
    periodic = model.lateral == "periodic"
    n_b = centres_b.shape[0]
    iS, iT = n_a + n_b, n_a + n_b + 1
    dsu = _DSU(n_a + n_b + 2)

    # ---- electrode contacts ------------------------------------------
    touch_l_a, touch_r_a = _rod_face_contacts(p1_a, p2_a, model)
    touch_l_b, touch_r_b = _sphere_face_contacts(centres_b, model)

    for idx in np.nonzero(touch_l_a)[0]:
        dsu.union(iS, int(idx))
    for idx in np.nonzero(touch_r_a)[0]:
        dsu.union(iT, int(idx))
    for idx in np.nonzero(touch_l_b)[0]:
        dsu.union(iS, int(n_a + idx))
    for idx in np.nonzero(touch_r_b)[0]:
        dsu.union(iT, int(n_a + idx))

    n_edges = 0

    # ---- rod-rod ------------------------------------------------------
    if n_a > 1:
        xa = centres_a[:, 0]
        reach = model.cutoff_aa + model.rod_length
        for ia, ib in _blocked_pairs(xa, xa, reach):
            keep = ib > ia
            if not keep.any():
                continue
            ia, ib = ia[keep], ib[keep]
            shift = _lateral_shift(centres_a[ia], centres_a[ib], model.L, periodic)
            q1 = p1_a[ib] + shift
            q2 = p2_a[ib] + shift
            dist = segment_segment_distance(p1_a[ia], p2_a[ia], q1, q2)
            hit = dist <= model.cutoff_aa
            if hit.any():
                dsu.union(ia[hit], ib[hit])
                n_edges += int(hit.sum())

    # ---- rod-sphere ---------------------------------------------------
    if n_a > 0 and n_b > 0:
        xa = centres_a[:, 0]
        reach = model.cutoff_as + model.rod_half_length
        for ia, ib in _blocked_pairs(xa, centres_b[:, 0], reach):
            shift = _lateral_shift(centres_a[ia], centres_b[ib], model.L, periodic)
            c = centres_b[ib] + shift
            dist = point_segment_distance(c, p1_a[ia], p2_a[ia])
            hit = dist <= model.cutoff_as
            if hit.any():
                dsu.union(ia[hit], n_a + ib[hit])
                n_edges += int(hit.sum())

    # ---- sphere-sphere ------------------------------------------------
    if n_b > 1:
        xb = centres_b[:, 0]
        reach = model.cutoff_ss
        for ia, ib in _blocked_pairs(xb, xb, reach):
            keep = ib > ia
            if not keep.any():
                continue
            ia, ib = ia[keep], ib[keep]
            shift = _lateral_shift(centres_b[ia], centres_b[ib], model.L, periodic)
            dist = point_point_distance(centres_b[ia], centres_b[ib] + shift)
            hit = dist <= model.cutoff_ss
            if hit.any():
                dsu.union(n_a + ia[hit], n_a + ib[hit])
                n_edges += int(hit.sum())

    return SimulationResult(
        conducting=bool(dsu.connected(iS, iT)),
        n_rod=n_a,
        n_sphere=n_b,
        n_edges=n_edges,
        n_contact_left=int(touch_l_a.sum() + touch_l_b.sum()),
        n_contact_right=int(touch_r_a.sum() + touch_r_b.sum()),
    )
