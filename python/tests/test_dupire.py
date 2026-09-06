"""Dupire local vol: flat-surface identity, clamping, skew/smile behaviour."""

from __future__ import annotations

import math

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


def test_no_clamp_inside_quoted_box_and_continuous_at_wings(bundled_surface):
    """CRIT-1 regression: the wing-clamped stencil must never cap inside the
    quoted box, and local vol must be continuous across the last quoted
    strike (before the fix vol(0.4995..0.5005, 1) was capped at 500%)."""
    lv = DupireLocalVol(bundled_surface)
    ks = np.arange(-0.5, 0.5 + 1e-9, 0.005)
    for t in [0.1, 0.5, 0.75, 1.0, 2.0, 3.0]:
        vols = np.asarray(lv.vol(ks, t))
        assert np.all((vols > FLOOR) & (vols < CAP)), f"T={t}: {vols.min()}..{vols.max()}"
        for edge in (bundled_surface.k_min, bundled_surface.k_max):
            lo = float(lv.vol(edge - 1e-4, t))
            hi = float(lv.vol(edge + 1e-4, t))
            assert abs(lo - hi) < 1e-3, f"T={t} k={edge}: {lo} vs {hi}"
            # constant in k beyond k_max - DK (clamped stencil)
            far = float(lv.vol(edge + math.copysign(1.0, edge), t))
            assert far == pytest.approx(hi if edge > 0 else lo, abs=1e-15)
    assert lv.floor_count == 0 and lv.cap_count == 0
    # Reference point at the wing for T = 1: 0.197 (was 5.0 before the fix).
    assert float(lv.vol(0.5, 1.0)) == pytest.approx(0.19703445, abs=1e-6)


def test_stencil_clamp_only_when_box_wider_than_2dk():
    """A degenerate box narrower than 2 DK is left unclamped (no crash)."""
    ks = np.array([-0.0005, 0.0, 0.0005])
    ts = np.array([0.5, 1.0])
    srf = ImpliedVolSurface(ks, ts, np.full((2, 3), 0.2))
    lv = DupireLocalVol(srf)
    assert float(lv.vol(0.0, 0.75)) == pytest.approx(0.2, abs=1e-6)
    assert float(lv.vol(3.0, 0.75)) == pytest.approx(0.2, abs=1e-6)


def test_term_structure_local_vol_reprices_forward_variance():
    """Flat-in-k surface with iv = {0.15, 0.20, 0.25} at T = {0.25, 0.5, 1}:
    local variance is the piecewise-constant forward variance."""
    ks = np.linspace(-0.5, 0.5, 11)
    ts = np.array([0.25, 0.5, 1.0])
    vols = np.array([[0.15] * 11, [0.20] * 11, [0.25] * 11])
    srf = ImpliedVolSurface(ks, ts, vols)
    lv = DupireLocalVol(srf)
    # (0.20^2*0.5 - 0.15^2*0.25) / 0.25 = 0.0575 -> 0.239792
    assert float(lv.vol(0.0, 0.4)) == pytest.approx(math.sqrt(0.0575), abs=1e-6)
    assert float(lv.vol(0.3, 0.4)) == pytest.approx(math.sqrt(0.0575), abs=1e-6)
    # (0.25^2*1 - 0.20^2*0.5) / 0.5 = 0.085 -> 0.291548
    assert float(lv.vol(0.0, 0.75)) == pytest.approx(math.sqrt(0.085), abs=1e-6)
    # below the first pillar: flat forward variance = first-pillar vol
    assert float(lv.vol(0.0, 0.1)) == pytest.approx(0.15, abs=1e-6)
    # beyond the last pillar: last interval's slope continues
    assert float(lv.vol(0.0, 2.0)) == pytest.approx(math.sqrt(0.085), abs=1e-6)
    assert lv.floor_count == 0 and lv.cap_count == 0


def test_term_structure_pde_round_trip_within_3bp():
    """End to end: pillar kinks + t_mid freezing + time interpolation."""
    from localvol import Market, implied_vol, price_european_pde

    ks = np.linspace(-0.5, 0.5, 11)
    ts = np.array([0.25, 0.5, 1.0])
    vols = np.array([[0.15] * 11, [0.20] * 11, [0.25] * 11])
    srf = ImpliedVolSurface(ks, ts, vols)
    lv = DupireLocalVol(srf)
    mkt = Market(100.0, 0.03, 0.01)
    for t in (0.75, 1.0):
        iv_in = float(srf.implied_vol(0.0, t))
        price = price_european_pde(mkt, 100.0, t, lv.vol, num_space=200, num_time=200, sigma_ref=iv_in)
        iv_out = implied_vol(price, 100.0, 100.0, 0.03, 0.01, t, True)
        assert abs(iv_out - iv_in) * 1e4 < 3.0, f"T={t}: {(iv_out - iv_in) * 1e4:.2f} bp"


def test_t0_equals_short_time_limit(bundled_localvol):
    """MAJ-1 regression: vol(k, 0) is the Berestycki-Busca-Florent limit
    sigma_imp / (1 - k sigma_imp'/sigma_imp), continuous with T -> 0+
    (before the fix vol(-0.3, 0) = 0.309 vs vol(-0.3, 1e-6) = 0.452)."""
    lv = bundled_localvol
    srf = lv.surface
    for k in (-0.3, -0.1, 0.0, 0.2, 0.45):
        v0 = float(lv.vol(k, 0.0))
        assert abs(v0 - float(lv.vol(k, 1e-6))) < 1e-3
        assert abs(v0 - float(lv.vol(k, 1e-4))) < 1e-3
        # independent BBF evaluation from the short-end implied slice
        s = float(srf.implied_vol(k, 0.0))
        sp = (float(srf.implied_vol(k + 1e-3, 0.0)) - float(srf.implied_vol(k - 1e-3, 0.0))) / 2e-3
        assert v0 == pytest.approx(s / (1.0 - k * sp / s), abs=1e-12)
    assert float(lv.vol(-0.3, 0.0)) == pytest.approx(0.45240187, abs=1e-6)
    assert float(lv.vol(0.0, 0.0)) == pytest.approx(0.2, abs=1e-12)  # ATM: s' cancels


def test_t0_limit_caps_on_butterfly_violating_slope():
    """1 - k s'/s <= 0 at T = 0 must cap (and count), never divide by zero."""
    ks = np.linspace(-0.5, 0.5, 21)
    # Strongly convex parabola: at |k| = 0.5, k s'/s = 0.5 * 4 / 1.05 = 1.9 > 1.
    # (A linear or mildly convex smile never violates it: s - k s' is the
    # tangent's intercept at k = 0, which convexity keeps above s(0).)
    v = 0.05 + 4.0 * ks * ks
    srf = ImpliedVolSurface(ks, np.array([0.5, 1.0]), np.vstack([v, v * 1.05]))
    lv = DupireLocalVol(srf)
    vols = np.asarray(lv.vol(ks, 0.0))
    assert np.all(vols <= CAP + 1e-15) and np.all(vols >= FLOOR - 1e-15)
    assert lv.cap_count > 0
    assert float(lv.vol(0.5, 0.0)) == CAP
    assert float(lv.vol(0.0, 0.0)) == pytest.approx(0.05, abs=1e-9)
