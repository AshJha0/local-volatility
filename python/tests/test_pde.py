"""PDE pricer: BS agreement, Rannacher damping, convergence order, American."""

from __future__ import annotations

import math

import numpy as np
import pytest

from localvol import (
    Market,
    bs_price,
    price_american_put_pde,
    price_european_pde,
)


@pytest.mark.parametrize(
    "s,k,r,q,sig,t,call",
    [
        (100.0, 100.0, 0.05, 0.02, 0.20, 1.0, True),    # equity ATM call, q > 0
        (100.0, 80.0, 0.05, 0.00, 0.20, 1.0, True),     # ITM call
        (100.0, 120.0, 0.05, 0.00, 0.30, 0.5, False),   # ITM put
        (1.10, 1.05, 0.03, 0.01, 0.10, 0.5, False),     # FX put (GK rd/rf)
        (100.0, 110.0, -0.01, 0.00, 0.25, 2.0, True),   # negative rate
        (100.0, 100.0, 0.00, 0.03, 0.15, 0.25, True),   # short-dated, q > r
    ],
)
def test_flat_vol_matches_black_scholes(s, k, r, q, sig, t, call):
    exact = bs_price(s, k, r, q, sig, t, call)
    approx = price_european_pde(Market(s, r, q), k, t, sig, is_call=call, num_space=200, num_time=200)
    assert approx == pytest.approx(exact, rel=1e-3)


def test_deep_itm_otm_and_degenerate():
    mkt = Market(100.0, 0.02, 0.0)
    # deep OTM: tiny but nonnegative; deep ITM: pinned to forward asymptote
    otm = price_european_pde(mkt, 300.0, 0.5, 0.2, is_call=True)
    assert 0.0 <= otm < 1e-3
    itm = price_european_pde(mkt, 10.0, 0.5, 0.2, is_call=True)
    assert itm == pytest.approx(bs_price(100, 10, 0.02, 0.0, 0.2, 0.5, True), rel=1e-3)
    # T = 0 -> intrinsic; sigma = 0 -> discounted forward intrinsic; K = 0
    assert price_european_pde(mkt, 90.0, 0.0, 0.2) == 10.0
    assert price_european_pde(mkt, 90.0, 1.0, 0.0) == pytest.approx(
        bs_price(100, 90, 0.02, 0.0, 0.0, 1.0, True)
    )
    assert price_european_pde(mkt, 0.0, 1.0, 0.2, is_call=True) == pytest.approx(100.0)
    assert price_european_pde(mkt, 0.0, 1.0, 0.2, is_call=False) == 0.0


def test_put_call_parity_on_grid():
    mkt = Market(100.0, 0.03, 0.01)
    c = price_european_pde(mkt, 105.0, 1.0, 0.2, is_call=True, num_space=200, num_time=100)
    p = price_european_pde(mkt, 105.0, 1.0, 0.2, is_call=False, num_space=200, num_time=100)
    rhs = 100 * math.exp(-0.01) - 105 * math.exp(-0.03)
    assert c - p == pytest.approx(rhs, abs=2e-2)


def test_rannacher_damping_no_gamma_oscillation():
    """Discrete gamma near the strike must stay essentially nonnegative.

    Plain CN leaves an oscillating gamma near the payoff kink; the Rannacher
    start damps it.  We assert the most negative second difference of the
    final price grid in a band around the strike is negligible relative to
    the peak curvature.
    """
    mkt = Market(100.0, 0.05, 0.0)
    res = price_european_pde(mkt, 100.0, 0.25, 0.2, num_space=200, num_time=50, return_grid=True)
    x, v = res.x, res.values
    band = (np.exp(x) > 70) & (np.exp(x) < 140)
    curv = np.diff(v, 2)[band[1:-1]]
    assert curv.max() > 0.0
    assert curv.min() > -1e-4 * curv.max()


def test_grid_convergence_second_order():
    """Halving dx and dt twice: observed order must land in [1.5, 2.5]."""
    mkt = Market(100.0, 0.05, 0.0)
    exact = bs_price(100, 100, 0.05, 0.0, 0.2, 1.0, True)
    errs = []
    for m in (50, 100, 200):
        approx = price_european_pde(mkt, 100.0, 1.0, 0.2, num_space=m, num_time=m)
        errs.append(abs(approx - exact))
    orders = [math.log2(errs[i] / errs[i + 1]) for i in range(2)]
    avg = sum(orders) / len(orders)
    assert 1.5 <= avg <= 2.5, f"observed order {avg:.2f} (steps: {orders})"


def test_localvol_pde_with_callable(bundled_localvol):
    """Local-vol PDE ~ BS at the surface's own implied vol (ATM, small err)."""
    mkt = Market(100.0, 0.0, 0.0)
    srf = bundled_localvol.surface
    price = price_european_pde(
        mkt, 100.0, 1.0, bundled_localvol.vol, num_space=200, num_time=200,
        sigma_ref=float(srf.implied_vol(0.0, 1.0)),
    )
    ref = bs_price(100, 100, 0.0, 0.0, float(srf.implied_vol(0.0, 1.0)), 1.0, True)
    assert price == pytest.approx(ref, rel=5e-3)  # ~1bp vol agreement


def test_american_put_flat():
    mkt = Market(100.0, 0.05, 0.0)
    eur = price_european_pde(mkt, 100.0, 1.0, 0.2, is_call=False, num_space=150, num_time=100)
    ame = price_american_put_pde(mkt, 100.0, 1.0, 0.2, num_space=150, num_time=100)
    # early-exercise premium nonnegative, and American >= intrinsic everywhere
    assert ame >= eur - 1e-10
    assert ame >= max(100.0 - 100.0, 0.0)
    # with q = 0 and r > 0 the premium is strictly positive for a put
    assert ame - eur > 0.01


def test_american_zero_rate_equals_european():
    """r = 0, q = 0: American put is never exercised early -> equals European."""
    mkt = Market(100.0, 0.0, 0.0)
    eur = price_european_pde(mkt, 110.0, 1.0, 0.25, is_call=False, num_space=150, num_time=100)
    ame = price_american_put_pde(mkt, 110.0, 1.0, 0.25, num_space=150, num_time=100)
    assert ame == pytest.approx(eur, rel=2e-3)


def test_american_localvol_above_european(bundled_localvol):
    mkt = Market(100.0, 0.04, 0.0)
    kwargs = dict(num_space=150, num_time=100, sigma_ref=0.2)
    eur = price_european_pde(mkt, 105.0, 1.0, bundled_localvol.vol, is_call=False, **kwargs)
    ame = price_american_put_pde(mkt, 105.0, 1.0, bundled_localvol.vol, **kwargs)
    assert ame >= eur - 1e-10


def test_sigma_zero_american_is_deterministic_optimum():
    # r > 0, sigma = 0: exercising immediately is optimal for an ITM put
    mkt = Market(80.0, 0.05, 0.0)
    assert price_american_put_pde(mkt, 100.0, 1.0, 0.0) == pytest.approx(20.0, abs=1e-6)


def test_validation():
    mkt = Market(100.0, 0.05, 0.0)
    with pytest.raises(ValueError, match="even"):
        price_european_pde(mkt, 100.0, 1.0, 0.2, num_space=201)
    with pytest.raises(ValueError, match="num_time"):
        price_european_pde(mkt, 100.0, 1.0, 0.2, num_time=0)
    with pytest.raises(ValueError, match="strike"):
        price_european_pde(mkt, -5.0, 1.0, 0.2)
    with pytest.raises(ValueError, match="expiry"):
        price_european_pde(mkt, 100.0, -1.0, 0.2)
    with pytest.raises(ValueError, match="flat vol"):
        price_european_pde(mkt, 100.0, 1.0, -0.2)
    with pytest.raises(ValueError, match="nsd"):
        price_european_pde(mkt, 100.0, 1.0, 0.2, nsd=0.0)
    with pytest.raises(ValueError, match="omega"):
        price_american_put_pde(mkt, 100.0, 1.0, 0.2, omega=2.5)
    with pytest.raises(ValueError, match="market"):
        price_european_pde("spot", 100.0, 1.0, 0.2)  # type: ignore[arg-type]


def test_psor_nonconvergence_warns_and_returns_finite():
    """max_iter = 1 cannot converge: a RuntimeWarning per time step, and the
    returned price is finite and non-negative (report, don't crash)."""
    mkt = Market(100.0, 0.05, 0.0)
    with pytest.warns(RuntimeWarning, match="PSOR did not converge"):
        price = price_american_put_pde(mkt, 100.0, 1.0, 0.2, num_space=100, num_time=20, max_iter=1)
    assert math.isfinite(price) and price >= 0.0
    assert price == pytest.approx(6.4776, abs=1e-3)


def test_american_put_deep_itm_equals_intrinsic():
    """Deep ITM: the lower Dirichlet boundary / obstacle pin V = K - S."""
    price = price_american_put_pde(Market(50.0, 0.05, 0.0), 100.0, 1.0, 0.2, num_space=100, num_time=50)
    assert abs(price - 50.0) < 1e-8


def test_american_put_negative_rate_equals_european():
    """r < 0, q = 0: early exercise is never optimal for a put."""
    mkt = Market(100.0, -0.02, 0.0)
    eur = price_european_pde(mkt, 100.0, 1.0, 0.2, is_call=False, num_space=100, num_time=50)
    ame = price_american_put_pde(mkt, 100.0, 1.0, 0.2, num_space=100, num_time=50)
    assert abs(ame - eur) < 1e-6
    assert ame >= eur - 1e-12


def test_localvol_pde_grid_convergence(bundled_localvol):
    """Observed order vs a 400x400 reference lies in [0.8, 2.5] under local
    vol (frozen-coefficient CN is formally first order in time)."""
    mkt = Market(100.0, 0.0, 0.0)
    ref = price_european_pde(mkt, 100.0, 1.0, bundled_localvol.vol, num_space=400, num_time=400, sigma_ref=0.2)
    errs = [
        abs(price_european_pde(mkt, 100.0, 1.0, bundled_localvol.vol, num_space=m, num_time=m, sigma_ref=0.2) - ref)
        for m in (50, 100, 200)
    ]
    orders = [math.log2(errs[i] / errs[i + 1]) for i in range(2)]
    for o in orders:
        assert 0.8 <= o <= 2.5, f"orders {orders}, errors {errs}"


def test_peclet_violation_switches_to_upwind_and_stays_monotone():
    """MAJ-5 regression: 1% vol with 10% carry violates |mu| h <= 2a on the
    default grid.  Central differencing then yields a negative put price
    (-2.95e-6) and grid values down to -0.084; the upwind switch keeps the
    put price and every grid value >= 0 and the grid monotone."""
    mkt = Market(100.0, 0.10, 0.0)
    res = price_european_pde(mkt, 90.0, 1.0, 0.01, is_call=False, num_space=100, num_time=50, return_grid=True)
    assert res.price >= 0.0
    assert res.price < 1e-6  # Black-Scholes value is ~1e-95
    assert np.all(res.values >= 0.0)
    assert np.all(np.diff(res.values) <= 1e-14)  # put nonincreasing in S
    call = price_european_pde(mkt, 90.0, 1.0, 0.01, is_call=True, num_space=100, num_time=50, return_grid=True)
    assert np.all(np.diff(call.values) >= -1e-14)
    assert call.price == pytest.approx(bs_price(100.0, 90.0, 0.10, 0.0, 0.01, 1.0, True), rel=2e-3)
    amer = price_american_put_pde(mkt, 90.0, 1.0, 0.01, num_space=100, num_time=50)
    assert 0.0 <= amer < 1e-6


def test_upwind_coefficients_are_m_matrix_and_consistent():
    """Direct check of the coefficient rule: central where |mu| h <= 2a,
    upwind elsewhere; off-diagonals >= 0 and row sum -r in both regimes."""
    from localvol.pde import _coefficients

    h, r, q = 0.01, 0.05, 0.0
    sig = np.array([0.2, 0.01])           # first central, second upwind (mu h = 5e-4 > 2a = 1e-4)
    lower, center, upper = _coefficients(sig, r, q, h)
    a = 0.5 * sig**2
    mu = r - q - a
    np.testing.assert_allclose(lower[0], a[0] / h**2 - mu[0] / (2 * h))
    np.testing.assert_allclose(upper[0], a[0] / h**2 + mu[0] / (2 * h))
    np.testing.assert_allclose(lower[1], a[1] / h**2)                  # mu > 0: no lower upwind term
    np.testing.assert_allclose(upper[1], a[1] / h**2 + mu[1] / h)
    np.testing.assert_allclose(center[1], -2 * a[1] / h**2 - mu[1] / h - r)
    assert np.all(lower >= 0.0) and np.all(upper >= 0.0)
    np.testing.assert_allclose(lower + center + upper, -r, atol=1e-12)


def test_pde_scalar_and_bad_callables():
    """MIN-8 / MAJ-4: scalar-returning callables are broadcast; NaN/negative
    output is rejected with ValueError, not an IndexError or a solver error."""
    mkt = Market(100.0, 0.05, 0.0)
    flat = price_european_pde(mkt, 100.0, 1.0, 0.2, num_space=100, num_time=50)
    scal = price_european_pde(mkt, 100.0, 1.0, lambda k, t: 0.2, num_space=100, num_time=50)
    assert scal == flat
    with pytest.raises(ValueError, match="non-finite/negative"):
        price_european_pde(mkt, 100.0, 1.0, lambda k, t: np.nan * k, num_space=100, num_time=50, sigma_ref=0.2)
    with pytest.raises(ValueError, match="non-finite/negative"):
        price_european_pde(mkt, 100.0, 1.0, lambda k, t: -0.2, num_space=100, num_time=50, sigma_ref=0.2)
    with pytest.raises(ValueError, match="non-finite/negative"):
        price_american_put_pde(mkt, 100.0, 1.0, lambda k, t: np.full_like(k, np.inf), num_space=100, num_time=50, sigma_ref=0.2)
    with pytest.raises(ValueError, match="shape"):
        price_european_pde(mkt, 100.0, 1.0, lambda k, t: np.zeros(7), num_space=100, num_time=50, sigma_ref=0.2)
    with pytest.raises(ValueError, match="sigma_ref"):
        price_european_pde(mkt, 100.0, 1.0, lambda k, t: 0.2, sigma_ref=-1.0)


def test_psor_rejects_nan_tol_and_bad_iters():
    """MIN-12: tol = NaN must be rejected eagerly, not spin every time step."""
    mkt = Market(100.0, 0.05, 0.0)
    with pytest.raises(ValueError, match="tol"):
        price_american_put_pde(mkt, 100.0, 1.0, 0.2, tol=float("nan"))
    with pytest.raises(ValueError, match="tol"):
        price_american_put_pde(mkt, 100.0, 1.0, 0.2, tol=0.0)
    with pytest.raises(ValueError, match="tol"):
        price_american_put_pde(mkt, 100.0, 1.0, 0.2, max_iter=0)
    with pytest.raises(ValueError, match="tol"):
        price_american_put_pde(mkt, 100.0, 1.0, 0.2, tol=float("inf"))


def test_pde_determinism():
    mkt = Market(100.0, 0.03, 0.01)
    a = price_european_pde(mkt, 105.0, 0.7, 0.23, num_space=120, num_time=60)
    b = price_european_pde(mkt, 105.0, 0.7, 0.23, num_space=120, num_time=60)
    assert a == b
