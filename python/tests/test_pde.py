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
