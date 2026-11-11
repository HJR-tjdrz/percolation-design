# -*- coding: utf-8 -*-
"""Physical sanity checks and statistical regression for the engine."""

import numpy as np

try:  # real pytest when available
    import pytest
except ImportError:  # pragma: no cover - fallback for a bare interpreter
    import _pytest_compat as pytest

from percolation import Model, estimate, n_for_fraction, simulate
from percolation.engine import _random_rods, _rod_face_contacts, _sphere_face_contacts


def test_empty_cell_does_not_conduct():
    m = Model()
    assert simulate(m, 0, 0, np.random.default_rng(1)).conducting is False


def test_rod_crossing_an_electrode_touches_both_under_legacy_wrap():
    """The legacy mirror convention is what creates the artificial short.

    A rod that only exits through the right electrode must touch the right
    electrode in both conventions, but only the legacy convention also reports
    contact with the left electrode.
    """
    p1 = np.array([[4000.0, 0.0, 0.0]])
    p2 = np.array([[9000.0, 0.0, 0.0]])

    left_clip, right_clip = _rod_face_contacts(p1, p2, Model(rod_x="clip"))
    left_wrap, right_wrap = _rod_face_contacts(p1, p2, Model(rod_x="wrap"))

    assert (left_clip[0], right_clip[0]) == (False, True)
    assert (left_wrap[0], right_wrap[0]) == (True, True)


def test_rod_fully_inside_touches_neither_electrode():
    p1 = np.array([[0.0, 0.0, 0.0]])
    p2 = np.array([[3000.0, 0.0, 0.0]])
    left, right = _rod_face_contacts(p1, p2, Model())
    assert not left[0] and not right[0]


def test_sphere_contact_uses_surface_distance():
    m = Model(sphere_radius=200.0)
    # centre 201.0 nm from the left face -> surface 1.0 nm away -> touching
    c = np.array([[-5000.0 + 201.0, 0.0, 0.0]])
    left, right = _sphere_face_contacts(c, m)
    assert left[0] and not right[0]

    # centre 210 nm away -> surface 10 nm away -> not touching
    c2 = np.array([[-5000.0 + 210.0, 0.0, 0.0]])
    left2, _ = _sphere_face_contacts(c2, m)
    assert not left2[0]


def test_dense_small_cell_conducts():
    m = Model(L=1000.0, rod_length=100.0, sphere_radius=300.0)
    assert simulate(m, 0, n_for_fraction(0.5, m, "sphere"),
                    np.random.default_rng(5)).conducting is True


def test_sparse_small_cell_does_not_conduct():
    m = Model(L=1000.0, rod_length=100.0, sphere_radius=50.0)
    for seed in range(20):
        assert simulate(m, 0, 2, np.random.default_rng(seed)).conducting is False


def test_short_isolated_rod_does_not_conduct():
    m = Model(rod_length=500.0)
    for seed in range(20):
        assert simulate(m, 1, 0, np.random.default_rng(seed)).conducting is False


def test_cluster_away_from_electrodes_does_not_conduct():
    # one sphere in the middle: touches nothing
    m = Model()
    assert simulate(m, 0, 1, np.random.default_rng(5)).conducting is False


def test_n_for_fraction_roundtrip():
    m = Model()
    for phi in (0.002, 0.005, 0.01):
        n = n_for_fraction(phi, m, "rod")
        assert n == round(phi * m.L ** 3 / m.volume_rod)
    # the values quoted in the competition report
    assert n_for_fraction(0.005, m, "rod") == 354
    assert n_for_fraction(0.0086, m, "rod") in (607, 608)


def test_rod_generation_is_isotropic():
    m = Model()
    rng = np.random.default_rng(11)
    _, dirs, _, _ = _random_rods(m, 40000, rng)
    # for isotropic unit vectors the mean of |u_x| is 1/2 and <u_x^2> is 1/3
    assert abs(np.abs(dirs[:, 0]).mean() - 0.5) < 0.01
    assert abs((dirs[:, 0] ** 2).mean() - 1.0 / 3.0) < 0.01
    assert np.allclose(np.linalg.norm(dirs, axis=1), 1.0, atol=1e-12)


@pytest.mark.slow
def test_rod_percolation_curve_regression():
    """Order-of-magnitude regression against the published Problem-2 curve."""
    m = Model()
    low = estimate(m, phi_rod=0.005, n_runs=150, seed=12345)
    high = estimate(m, phi_rod=0.010, n_runs=30, seed=12345)
    assert 0.02 <= low.p_hat <= 0.16, low
    assert high.p_hat >= 0.97, high
    # the transition must be monotone
    mid = estimate(m, phi_rod=0.007, n_runs=150, seed=12345)
    assert low.p_hat < mid.p_hat < high.p_hat


@pytest.mark.slow
def test_pure_sphere_has_no_conduction_at_low_fraction():
    m = Model()
    res = estimate(m, phi_sphere=0.15, n_runs=2, seed=7)
    assert res.p_hat == 0.0
