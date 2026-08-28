"""Dupire local vol: flat-surface identity, clamping, skew/smile behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from localvol import CAP, FLOOR, DupireLocalVol, ImpliedVolSurface


def test_flat_surface_localvol_equals_implied(flat_surface):
    """Property-style golden anchor: sigma_loc == 20% to 1e-6 everywhere."""
    lv = DupireLocalVol(flat_surface)
    for k in np.linspace(-0.45, 0.45, 9):
        for t in [0.05, 0.25, 0.6, 1.0, 1.7, 2.4]:
            assert float(lv.vol(float(k), float(t))) == pytest.approx(0.20, abs=1e-6)
    assert lv.floor_count == 0 and lv.cap_count == 0


def test_bundled_surface_sane_interior(bundled_localvol):
    """Skewed surface: local vol finite, in band, decreasing across the skew."""
    lv = bundled_localvol
    ks = np.linspace(-0.25, 0.25, 11)
    vols = np.asarray(lv.vol(ks, 1.0))
    assert np.all(np.isfinite(vols))
    assert np.all((vols >= FLOOR) & (vols <= CAP))
    # negative-rho SSVI: local vol decreases from put wing to call wing
    assert vols[0] > vols[-1]
    # local skew is roughly twice the implied skew — at least steeper:
    srf = lv.surface
    imp = np.asarray(srf.implied_vol(ks, 1.0))
    assert (vols[0] - vols[-1]) > (imp[0] - imp[-1])


def test_vectorised_matches_scalar(bundled_localvol):
    ks = np.array([-0.3, -0.1, 0.0, 0.15, 0.3])
    vec = np.asarray(bundled_localvol.vol(ks, 0.8))
    sca = np.array([float(bundled_localvol.vol(float(k), 0.8)) for k in ks])
    np.testing.assert_allclose(vec, sca, atol=1e-15)


def test_floor_engaged_on_extreme_skew():
    """A calendar-flat (dw/dT ~ 0) steep surface must clamp, not crash."""
    ks = np.linspace(-0.5, 0.5, 11)
    ts = np.array([0.5, 1.0])
    # nearly constant total variance in T -> numerator ~ 0 -> floor engages
    v1 = 0.2 + 0.5 * np.abs(ks)
    v2 = np.sqrt((v1**2 * 0.5 + 1e-9) / 1.0)
    srf = ImpliedVolSurface(ks, ts, np.vstack([v1, v2]))
    lv = DupireLocalVol(srf)
    with np.errstate(all="ignore"):
        vols = np.asarray(lv.vol(ks, 0.75))
    assert np.all(vols >= FLOOR - 1e-15)
    assert lv.floor_count > 0
    assert "floor" in lv.violation_report
    lv.reset_counters()
    assert lv.floor_count == 0 and lv.cap_count == 0


def test_cap_engaged_on_butterfly_violating_wing():
    """Over-steep smile (butterfly arb): denominator <= 0 -> capped at 500%."""
    ks = np.linspace(-0.5, 0.5, 21)
    ts = np.array([0.5, 1.0])
    v = 0.10 + 1.0 * np.abs(ks)  # V-shaped smile: wings violate |dw/dk| bound
    srf = ImpliedVolSurface(ks, ts, np.vstack([v, v * 1.05]))
    lv = DupireLocalVol(srf)
    with np.errstate(all="ignore"):
        vols = np.asarray(lv.vol(np.linspace(-0.45, 0.45, 41), 0.75))
    assert np.all(vols <= CAP + 1e-15)
    assert lv.cap_count > 0


def test_fx_symmetric_smile():
    """FX-style symmetric smile: local vol symmetric-ish and smile-shaped."""
    ks = np.linspace(-0.3, 0.3, 13)
    ts = np.array([0.25, 0.5, 1.0])
    vols = np.array([[0.10 + 0.15 * k * k for k in ks] for _ in ts])
    srf = ImpliedVolSurface(ks, ts, vols * np.array([1.0, 1.0, 1.0])[:, None])
    lv = DupireLocalVol(srf)
    v_pos = float(lv.vol(0.15, 0.5))
    v_neg = float(lv.vol(-0.15, 0.5))
    v_atm = float(lv.vol(0.0, 0.5))
    assert v_pos == pytest.approx(v_neg, rel=0.15)  # near-symmetric
    assert v_pos > v_atm and v_neg > v_atm          # smile shape survives


def test_validation():
    with pytest.raises(ValueError):
        DupireLocalVol("not a surface")  # type: ignore[arg-type]
    ks = np.array([-0.1, 0.0, 0.1])
    srf = ImpliedVolSurface(ks, np.array([0.5, 1.0]), np.full((2, 3), 0.2))
    lv = DupireLocalVol(srf)
    with pytest.raises(ValueError):
        lv.vol(0.0, -1.0)
    with pytest.raises(ValueError):
        lv.vol(float("inf"), 1.0)
