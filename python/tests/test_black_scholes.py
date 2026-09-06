"""Black-Scholes analytics: known values, parity, Greeks vs FD, edge cases."""

from __future__ import annotations

import math

import pytest

from localvol import (
    Market,
    bs_delta,
    bs_gamma,
    bs_price,
    bs_vega,
    implied_vol,
)


def test_known_atm_value():
    # Classic textbook number: S=K=100, r=5%, q=0, sigma=20%, T=1.
    assert bs_price(100, 100, 0.05, 0.0, 0.2, 1.0, True) == pytest.approx(10.450584, abs=1e-6)


def test_put_call_parity_grid():
    """Property-style: parity C - P = S e^{-qT} - K e^{-rT} across a grid."""
    for s in [80.0, 100.0, 123.4]:
        for k in [60.0, 100.0, 140.0]:
            for r, q in [(0.05, 0.0), (0.01, 0.03), (-0.005, 0.0)]:
                for t in [0.1, 1.0, 3.0]:
                    c = bs_price(s, k, r, q, 0.25, t, True)
                    p = bs_price(s, k, r, q, 0.25, t, False)
                    lhs = c - p
                    rhs = s * math.exp(-q * t) - k * math.exp(-r * t)
                    assert lhs == pytest.approx(rhs, abs=1e-10)


def test_garman_kohlhagen_via_market_fx():
    mkt = Market.fx(1.20, rd=0.03, rf=0.01)
    # GK call == BS call with q = rf
    assert bs_price(mkt.spot, 1.25, mkt.rate, mkt.dividend, 0.10, 0.5, True) == pytest.approx(
        bs_price(1.20, 1.25, 0.03, 0.01, 0.10, 0.5, True)
    )
    assert mkt.forward(0.5) == pytest.approx(1.20 * math.exp(0.02 * 0.5))


def test_edge_cases():
    assert bs_price(100, 90, 0.05, 0.0, 0.2, 0.0, True) == 10.0  # expiry -> intrinsic
    assert bs_price(100, 110, 0.05, 0.0, 0.2, 0.0, True) == 0.0
    # sigma=0 -> discounted forward intrinsic
    f = 100 * math.exp(0.05)
    assert bs_price(100, 90, 0.05, 0.0, 0.0, 1.0, True) == pytest.approx(math.exp(-0.05) * (f - 90))
    assert bs_price(100, 0.0, 0.03, 0.01, 0.2, 1.0, True) == pytest.approx(100 * math.exp(-0.01))
    assert bs_price(100, 0.0, 0.03, 0.01, 0.2, 1.0, False) == 0.0
    # deep ITM/OTM: values pinned to asymptotics
    assert bs_price(100, 1e6, 0.0, 0.0, 0.2, 1.0, True) == pytest.approx(0.0, abs=1e-12)
    deep_itm = bs_price(100, 1.0, 0.0, 0.0, 0.2, 1.0, True)
    assert deep_itm == pytest.approx(99.0, abs=1e-8)


def test_input_validation():
    with pytest.raises(ValueError):
        bs_price(-1.0, 100, 0.0, 0.0, 0.2, 1.0)
    with pytest.raises(ValueError):
        bs_price(100, -5.0, 0.0, 0.0, 0.2, 1.0)
    with pytest.raises(ValueError):
        bs_price(100, 100, 0.0, 0.0, -0.2, 1.0)
    with pytest.raises(ValueError):
        bs_price(100, 100, 0.0, 0.0, 0.2, -1.0)
    with pytest.raises(ValueError):
        bs_price(float("nan"), 100, 0.0, 0.0, 0.2, 1.0)
    with pytest.raises(ValueError):
        Market(-1.0, 0.0, 0.0)


def test_greeks_vs_finite_differences():
    s, k, r, q, sig, t = 105.0, 100.0, 0.03, 0.01, 0.22, 0.75
    eps = 1e-4
    d_fd = (bs_price(s + eps, k, r, q, sig, t) - bs_price(s - eps, k, r, q, sig, t)) / (2 * eps)
    g_fd = (bs_price(s + eps, k, r, q, sig, t) - 2 * bs_price(s, k, r, q, sig, t) + bs_price(s - eps, k, r, q, sig, t)) / eps**2
    v_fd = (bs_price(s, k, r, q, sig + eps, t) - bs_price(s, k, r, q, sig - eps, t)) / (2 * eps)
    assert bs_delta(s, k, r, q, sig, t) == pytest.approx(d_fd, abs=1e-6)
    assert bs_gamma(s, k, r, q, sig, t) == pytest.approx(g_fd, abs=1e-5)
    assert bs_vega(s, k, r, q, sig, t) == pytest.approx(v_fd, abs=1e-4)


def test_implied_vol_round_trip():
    for sig in [0.05, 0.2, 0.8]:
        for k in [70.0, 100.0, 130.0]:
            price = bs_price(100, k, 0.02, 0.01, sig, 1.5, True)
            iv = implied_vol(price, 100, k, 0.02, 0.01, 1.5, True)
            assert iv == pytest.approx(sig, abs=1e-8)


def test_implied_vol_rejects_arbitrage_and_bad_inputs():
    with pytest.raises(ValueError, match="bounds"):
        implied_vol(1000.0, 100, 100, 0.0, 0.0, 1.0, True)  # above S
    with pytest.raises(ValueError, match="bounds"):
        implied_vol(-1.0, 100, 100, 0.0, 0.0, 1.0, True)  # below intrinsic
    with pytest.raises(ValueError):
        implied_vol(5.0, 100, 100, 0.0, 0.0, 0.0, True)  # T = 0


def test_implied_vol_rejects_bad_bracket_and_iterations():
    """MIN-7: non-default lo/hi/iterations are validated."""
    price = bs_price(100, 100, 0.02, 0.01, 0.2, 1.0, True)
    with pytest.raises(ValueError, match="bracket"):
        implied_vol(price, 100, 100, 0.02, 0.01, 1.0, True, lo=5.0, hi=1e-9)
    with pytest.raises(ValueError, match="bracket"):
        implied_vol(price, 100, 100, 0.02, 0.01, 1.0, True, lo=0.0, hi=1.0)
    with pytest.raises(ValueError, match="bracket"):
        implied_vol(price, 100, 100, 0.02, 0.01, 1.0, True, lo=1e-9, hi=float("inf"))
    with pytest.raises(ValueError, match="iterations"):
        implied_vol(price, 100, 100, 0.02, 0.01, 1.0, True, iterations=0)
    with pytest.raises(ValueError, match="price must be finite"):
        implied_vol(float("nan"), 100, 100, 0.02, 0.01, 1.0, True)
    # the default contract: exactly 100 halvings of [1e-9, 5]
    assert implied_vol(price, 100, 100, 0.02, 0.01, 1.0, True) == pytest.approx(0.2, abs=1e-10)


def test_market_log_forward_validation_and_value():
    """MIN-6: log_forward validates expiry in every port."""
    mkt = Market(100.0, 0.03, 0.01)
    assert mkt.log_forward(2.0) == pytest.approx(math.log(100.0) + 0.02 * 2.0, abs=1e-15)
    assert mkt.log_forward(0.0) == math.log(100.0)
    with pytest.raises(ValueError):
        mkt.log_forward(-1.0)
    with pytest.raises(ValueError):
        mkt.log_forward(float("nan"))
    with pytest.raises(ValueError):
        mkt.forward(float("inf"))
    with pytest.raises(ValueError):
        Market(100.0, float("nan"), 0.0)
    with pytest.raises(ValueError):
        Market(100.0, 0.0, float("inf"))
