"""Black-Scholes / Garman-Kohlhagen analytics.

All formulas are written in terms of ``(r, q)``; for FX simply pass
``r = rd`` and ``q = rf`` (Garman-Kohlhagen).  The normal CDF is computed via
``erfc`` for accuracy in the far tails, which every target language can
reproduce bit-for-bit closely enough for the golden tolerances.
"""

from __future__ import annotations

import math

__all__ = [
    "norm_cdf",
    "norm_pdf",
    "bs_price",
    "bs_delta",
    "bs_gamma",
    "bs_vega",
    "implied_vol",
]

_SQRT2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def norm_cdf(x: float) -> float:
    """Standard normal CDF via ``erfc`` (tail-accurate)."""
    return 0.5 * math.erfc(-x / _SQRT2)


def norm_pdf(x: float) -> float:
    """Standard normal density."""
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)


def _validate_common(
    spot: float, strike: float, rate: float, dividend: float, sigma: float, expiry: float
) -> None:
    for name, v in (
        ("spot", spot),
        ("strike", strike),
        ("rate", rate),
        ("dividend", dividend),
        ("sigma", sigma),
        ("expiry", expiry),
    ):
        if not math.isfinite(v):
            raise ValueError(f"{name} must be finite, got {v!r}")
    if spot <= 0.0:
        raise ValueError(f"spot must be > 0, got {spot!r}")
    if strike < 0.0:
        raise ValueError(f"strike must be >= 0, got {strike!r}")
    if sigma < 0.0:
        raise ValueError(f"sigma must be >= 0, got {sigma!r}")
    if expiry < 0.0:
        raise ValueError(f"expiry must be >= 0, got {expiry!r}")


def bs_price(
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    sigma: float,
    expiry: float,
    is_call: bool = True,
) -> float:
    """European vanilla price under Black-Scholes / Garman-Kohlhagen.

    Edge cases (all handled without branching surprises):

    * ``expiry == 0``      -> intrinsic value ``max(phi (S - K), 0)``.
    * ``sigma == 0``       -> discounted forward intrinsic
      ``exp(-r T) max(phi (F - K), 0)``: the deterministic limit.
    * ``strike == 0``      -> call is worth the dividend-discounted spot
      ``S exp(-q T)``; put is worthless.
    """
    _validate_common(spot, strike, rate, dividend, sigma, expiry)
    phi = 1.0 if is_call else -1.0
    if expiry == 0.0:
        return max(phi * (spot - strike), 0.0)
    df_r = math.exp(-rate * expiry)
    df_q = math.exp(-dividend * expiry)
    forward = spot * math.exp((rate - dividend) * expiry)
    if strike == 0.0:
        return spot * df_q if is_call else 0.0
    if sigma == 0.0:
        return df_r * max(phi * (forward - strike), 0.0)
    st = sigma * math.sqrt(expiry)
    d1 = (math.log(forward / strike) + 0.5 * st * st) / st
    d2 = d1 - st
    return phi * (spot * df_q * norm_cdf(phi * d1) - strike * df_r * norm_cdf(phi * d2))


def _d1(spot: float, strike: float, rate: float, dividend: float, sigma: float, expiry: float) -> float:
    st = sigma * math.sqrt(expiry)
    forward = spot * math.exp((rate - dividend) * expiry)
    return (math.log(forward / strike) + 0.5 * st * st) / st


def bs_delta(
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    sigma: float,
    expiry: float,
    is_call: bool = True,
) -> float:
    """Spot delta ``exp(-q T) N(phi d1) * phi`` (requires ``sigma, T, K > 0``)."""
    _validate_common(spot, strike, rate, dividend, sigma, expiry)
    if expiry <= 0.0 or sigma <= 0.0 or strike <= 0.0:
        raise ValueError("bs_delta requires expiry > 0, sigma > 0 and strike > 0")
    phi = 1.0 if is_call else -1.0
    d1 = _d1(spot, strike, rate, dividend, sigma, expiry)
    return phi * math.exp(-dividend * expiry) * norm_cdf(phi * d1)


def bs_gamma(
    spot: float, strike: float, rate: float, dividend: float, sigma: float, expiry: float
) -> float:
    """Spot gamma ``exp(-q T) n(d1) / (S sigma sqrt(T))`` (call == put)."""
    _validate_common(spot, strike, rate, dividend, sigma, expiry)
    if expiry <= 0.0 or sigma <= 0.0 or strike <= 0.0:
        raise ValueError("bs_gamma requires expiry > 0, sigma > 0 and strike > 0")
    d1 = _d1(spot, strike, rate, dividend, sigma, expiry)
    return math.exp(-dividend * expiry) * norm_pdf(d1) / (spot * sigma * math.sqrt(expiry))


def bs_vega(
    spot: float, strike: float, rate: float, dividend: float, sigma: float, expiry: float
) -> float:
    """Vega ``S exp(-q T) n(d1) sqrt(T)`` per unit of vol (call == put)."""
    _validate_common(spot, strike, rate, dividend, sigma, expiry)
    if expiry <= 0.0 or sigma <= 0.0 or strike <= 0.0:
        raise ValueError("bs_vega requires expiry > 0, sigma > 0 and strike > 0")
    d1 = _d1(spot, strike, rate, dividend, sigma, expiry)
    return spot * math.exp(-dividend * expiry) * norm_pdf(d1) * math.sqrt(expiry)


def implied_vol(
    price: float,
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    expiry: float,
    is_call: bool = True,
    lo: float = 1e-9,
    hi: float = 5.0,
    iterations: int = 100,
) -> float:
    """Invert Black-Scholes by bisection on ``[lo, hi]``.

    Bisection is chosen deliberately: it is monotone-safe (vega can be tiny in
    the wings), needs no derivative, and — crucially for the cross-language
    golden values — is trivially reproducible in C++/Rust/Java: 100 halvings
    of a fixed bracket give ~1e-10 vol accuracy deterministically.

    Raises ValueError if the price violates the static no-arbitrage bounds for
    the given forward, or if inputs are non-finite / negative where forbidden.
    """
    _validate_common(spot, strike, rate, dividend, 0.0, expiry)
    if not math.isfinite(price):
        raise ValueError(f"price must be finite, got {price!r}")
    if expiry <= 0.0:
        raise ValueError("implied_vol requires expiry > 0")
    if strike <= 0.0:
        raise ValueError("implied_vol requires strike > 0")
    lower = bs_price(spot, strike, rate, dividend, 0.0, expiry, is_call)
    upper = spot * math.exp(-dividend * expiry) if is_call else strike * math.exp(-rate * expiry)
    eps = 1e-12 * max(1.0, spot)
    if price < lower - eps or price > upper + eps:
        raise ValueError(
            f"price {price!r} outside no-arbitrage bounds [{lower:.10g}, {upper:.10g}]"
        )
    a, b = lo, hi
    for _ in range(iterations):
        mid = 0.5 * (a + b)
        if bs_price(spot, strike, rate, dividend, mid, expiry, is_call) < price:
            a = mid
        else:
            b = mid
    return 0.5 * (a + b)
