"""Surface interpolation: node exactness, continuity, extrapolation, arb checks."""

from __future__ import annotations

import numpy as np
import pytest

from localvol import CubicSpline1D, ImpliedVolSurface


def _smile_surface() -> ImpliedVolSurface:
    ks = np.linspace(-0.4, 0.4, 9)
    ts = np.array([0.25, 0.5, 1.0, 2.0])
    vols = np.array([[0.20 + 0.05 * k * k * 4 - 0.08 * k for k in ks] for _ in ts])
    return ImpliedVolSurface(ks, ts, vols)


def test_spline_exact_at_nodes_and_smooth():
    x = np.linspace(-1.0, 2.0, 11)
    y = np.sin(x) + 0.3 * x * x
    sp = CubicSpline1D(x, y)
    np.testing.assert_allclose(sp(x), y, atol=1e-14)
    # C0/C1 continuity across an interior node
    xn = x[5]
    eps = 1e-7
    assert sp(xn + eps) == pytest.approx(sp(xn - eps), abs=1e-6)
    left = (sp(xn) - sp(xn - eps)) / eps
    right = (sp(xn + eps) - sp(xn)) / eps
    assert left == pytest.approx(right, abs=1e-5)


def test_spline_degenerate_and_validation():
    assert CubicSpline1D(np.array([1.0]), np.array([3.0]))(99.0) == 3.0
    lin = CubicSpline1D(np.array([0.0, 2.0]), np.array([1.0, 5.0]))
    assert lin(1.0) == pytest.approx(3.0)
    with pytest.raises(ValueError, match="increasing"):
        CubicSpline1D(np.array([0.0, 0.0, 1.0]), np.zeros(3))
    with pytest.raises(ValueError):
        CubicSpline1D(np.array([0.0, 1.0]), np.array([np.nan, 1.0]))


def test_surface_exact_at_nodes():
    srf = _smile_surface()
    for j, t in enumerate(srf.expiries):
        for i, k in enumerate(srf.k_nodes):
            w = srf.total_variance(float(k), float(t))
            assert w == pytest.approx(srf.vols[j, i] ** 2 * t, abs=1e-14)
            assert srf.implied_vol(float(k), float(t)) == pytest.approx(srf.vols[j, i], abs=1e-12)


def test_surface_continuity_near_nodes():
    """Property-style: w is continuous in both k and T at every pillar/node."""
    srf = _smile_surface()
    eps = 1e-8
    for t in srf.expiries:
        for k in srf.k_nodes:
            up = srf.total_variance(float(k) + eps, float(t))
            dn = srf.total_variance(float(k) - eps, float(t))
            assert up == pytest.approx(dn, abs=1e-6)
        mid = srf.total_variance(0.01, float(t))
        assert srf.total_variance(0.01, float(t) + eps) == pytest.approx(mid, abs=1e-6)
        assert srf.total_variance(0.01, float(t) - eps) == pytest.approx(mid, abs=1e-6)


def test_flat_k_extrapolation():
    srf = _smile_surface()
    w_edge = srf.total_variance(srf.k_max, 1.0)
    assert srf.total_variance(srf.k_max + 5.0, 1.0) == pytest.approx(w_edge, abs=1e-15)
    assert srf.total_variance(srf.k_min - 5.0, 1.0) == pytest.approx(
        srf.total_variance(srf.k_min, 1.0), abs=1e-15
    )


def test_time_extrapolation_rules():
    srf = _smile_surface()
    # below first pillar: proportional total variance (constant implied vol)
    iv1 = srf.implied_vol(0.05, float(srf.expiries[0]))
    assert srf.implied_vol(0.05, 0.05) == pytest.approx(iv1, abs=1e-12)
    assert srf.total_variance(0.05, 0.0) == 0.0
    assert srf.implied_vol(0.05, 0.0) == pytest.approx(iv1, abs=1e-12)  # short-end limit
    # beyond last pillar: linear continuation of last-interval slope
    t1, t2 = float(srf.expiries[-2]), float(srf.expiries[-1])
    w1, w2 = srf.total_variance(0.1, t1), srf.total_variance(0.1, t2)
    slope = (w2 - w1) / (t2 - t1)
    assert srf.total_variance(0.1, t2 + 1.0) == pytest.approx(w2 + slope, rel=1e-12)
    # vectorised query agrees with scalar loop
    ks = np.array([-0.3, 0.0, 0.2])
    np.testing.assert_allclose(
        srf.total_variance(ks, 0.8), [srf.total_variance(float(k), 0.8) for k in ks], atol=1e-15
    )


def test_single_expiry_warns_and_flat_forward():
    ks = np.array([-0.2, 0.0, 0.2])
    with pytest.warns(UserWarning, match="single expiry"):
        srf = ImpliedVolSurface(ks, np.array([1.0]), np.array([[0.25, 0.2, 0.22]]))
    assert srf.single_expiry
    # flat forward variance on both sides of the pillar
    assert srf.implied_vol(0.0, 0.5) == pytest.approx(0.2, abs=1e-12)
    assert srf.implied_vol(0.0, 3.0) == pytest.approx(0.2, abs=1e-12)


def test_calendar_arbitrage_detected_not_fatal():
    ks = np.array([-0.1, 0.0, 0.1])
    ts = np.array([0.5, 1.0])
    vols = np.array([[0.30, 0.30, 0.30], [0.10, 0.10, 0.30]])  # w drops at 2 nodes
    with pytest.warns(UserWarning, match="calendar"):
        srf = ImpliedVolSurface(ks, ts, vols)
    assert srf.calendar_violations == 2


def test_surface_validation():
    ks = np.array([-0.1, 0.0, 0.1])
    ts = np.array([0.5, 1.0])
    good = np.full((2, 3), 0.2)
    with pytest.raises(ValueError, match="shape"):
        ImpliedVolSurface(ks, ts, np.full((3, 2), 0.2))
    with pytest.raises(ValueError, match="increasing"):
        ImpliedVolSurface(ks[::-1], ts, good)
    with pytest.raises(ValueError, match="positive"):
        ImpliedVolSurface(ks, np.array([-0.5, 1.0]), good)
    with pytest.raises(ValueError, match="positive"):
        ImpliedVolSurface(ks, ts, np.full((2, 3), -0.2))
    with pytest.raises(ValueError, match="finite"):
        ImpliedVolSurface(ks, ts, np.full((2, 3), np.inf))
    srf = _smile_surface()
    with pytest.raises(ValueError):
        srf.total_variance(0.0, -1.0)
    with pytest.raises(ValueError):
        srf.total_variance(float("nan"), 1.0)


def test_from_csv_round_trip(bundled_surface, data_dir):
    assert bundled_surface.expiries.tolist() == [0.25, 0.5, 1.0, 2.0]
    assert bundled_surface.k_nodes.size == 15
    assert bundled_surface.calendar_violations == 0
    with pytest.raises(ValueError, match="header"):
        ImpliedVolSurface.from_csv(data_dir / "generate_data.py")


def _write_csv(path, rows):
    path.write_text("T,k,iv\n" + "".join(f"{t},{k},{iv}\n" for t, k, iv in rows))


def test_csv_rejects_duplicate_missing_and_short_rows(tmp_path):
    """MIN-4 / MIN-5: duplicates and gaps are named; a short row is a
    ValueError, not a TypeError/IndexError."""
    base = [(0.5, -0.1, 0.22), (0.5, 0.0, 0.2), (0.5, 0.1, 0.21),
            (1.0, -0.1, 0.23), (1.0, 0.0, 0.21), (1.0, 0.1, 0.22)]
    good = tmp_path / "good.csv"
    _write_csv(good, base)
    srf = ImpliedVolSurface.from_csv(good)
    assert srf.expiries.tolist() == [0.5, 1.0] and srf.k_nodes.size == 3

    dup = tmp_path / "dup.csv"
    _write_csv(dup, base + [(1.0, 0.0, 0.25)])
    with pytest.raises(ValueError, match="duplicate"):
        ImpliedVolSurface.from_csv(dup)

    missing = tmp_path / "missing.csv"
    _write_csv(missing, base[:-1])
    with pytest.raises(ValueError, match="rectangular"):
        ImpliedVolSurface.from_csv(missing)

    short = tmp_path / "short.csv"
    short.write_text("T,k,iv\n0.5,-0.1,0.22\n0.5,0.0\n")
    with pytest.raises(ValueError, match="3 columns"):
        ImpliedVolSurface.from_csv(short)

    extra = tmp_path / "extra.csv"
    extra.write_text("T,k,iv,x\n0.5,-0.1,0.22,1\n")
    with pytest.raises(ValueError, match="header"):
        ImpliedVolSurface.from_csv(extra)

    nonnum = tmp_path / "nonnum.csv"
    nonnum.write_text("T,k,iv\n0.5,-0.1,abc\n")
    with pytest.raises(ValueError, match="non-numeric"):
        ImpliedVolSurface.from_csv(nonnum)

    nonfinite = tmp_path / "nonfinite.csv"
    nonfinite.write_text("T,k,iv\n0.5,-0.1,nan\n")
    with pytest.raises(ValueError, match="non-finite"):
        ImpliedVolSurface.from_csv(nonfinite)

    empty = tmp_path / "empty.csv"
    empty.write_text("T,k,iv\n")
    with pytest.raises(ValueError, match="empty"):
        ImpliedVolSurface.from_csv(empty)

    # Blank lines and CRLF endings are tolerated.
    crlf = tmp_path / "crlf.csv"
    crlf.write_bytes(b"T,k,iv\r\n" + "".join(f"{t},{k},{iv}\r\n" for t, k, iv in base).encode() + b"\r\n")
    assert ImpliedVolSurface.from_csv(crlf).k_nodes.size == 3


def test_negative_total_variance_overshoot_is_counted(bundled_surface):
    """MIN-15: a ragged slice whose spline dips to w <= 0 between nodes is
    reported via negative_w_count + warning; clean surfaces report 0."""
    assert bundled_surface.negative_w_count == 0
    ks = np.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
    # tiny vols next to large ones -> natural spline overshoots below zero
    row = [1.0, 0.001, 0.001, 1.0, 0.001, 0.001, 1.0]
    with pytest.warns(UserWarning, match="overshoot"):
        srf = ImpliedVolSurface(ks, np.array([0.5, 1.0]), np.array([row, row]))
    assert srf.negative_w_count > 0
    # and implied_vol indeed reads 0 there rather than raising
    probe = np.linspace(-0.3, 0.3, 601)
    w = np.asarray(srf.total_variance(probe, 1.0))
    assert np.any(w < 0.0)
    assert float(np.min(np.asarray(srf.implied_vol(probe, 1.0)))) == 0.0
