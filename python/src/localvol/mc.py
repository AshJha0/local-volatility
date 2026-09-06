"""Local-volatility Monte Carlo: log-Euler scheme with antithetic variates.

Scheme
------
Simulate ``X = ln S`` on a uniform time grid ``t_n = n dt``, ``dt = T / N``:

    X_{n+1} = X_n + (r - q - 1/2 sigma_n^2) dt + sigma_n sqrt(dt) Z_n

with ``sigma_n = sigma(k_n, t_n)`` looked up at the *start* of the step, at
forward log-moneyness ``k_n = X_n - ln F(t_n)`` — the same coordinates the
Dupire surface uses, so PDE and MC discretise the identical diffusion.
Log-Euler (Euler on ``ln S``) is preferred to Euler on ``S`` because it keeps
``S > 0`` exactly and its weak error for vanillas is comparable.

Antithetic variates: paths come in pairs driven by ``+Z`` and ``-Z``.  The
estimator averages each pair first, and the standard error is computed over
the ``n_paths / 2`` *pair means* — using raw per-path deviations would
overstate the error because paired payoffs are negatively correlated.

Barrier (up-and-out call)
-------------------------
Two monitoring modes:

* **Discrete** (``brownian_bridge=False``): a path is dead once ``X_n``
  ends a step at or above ``ln B``.  This overprices the continuous barrier
  (the well-known O(sqrt(dt)) discrete-monitoring bias) because excursions
  above the barrier *between* grid points go unseen.
* **Brownian-bridge correction** (``brownian_bridge=True``, default): for a
  step from ``x0`` to ``x1`` with both below ``b = ln B``, the probability
  that the Brownian bridge between them crossed the barrier is

      p_cross = exp(-2 (b - x0)(b - x1) / (sigma^2 dt)),

  and each path carries a multiplicative survival weight
  ``prod (1 - p_cross)``.  This removes the leading-order bias without
  extra random numbers (keeping the estimator deterministic per seed) and
  strictly lowers the price versus discrete monitoring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Union

import numpy as np
from numpy.typing import NDArray

from .market import Market

__all__ = ["MCResult", "price_european_mc", "price_up_out_call_mc"]

VolInput = Union[float, Callable[[NDArray[np.float64], float], NDArray[np.float64]]]


@dataclass(frozen=True)
class MCResult:
    """Monte Carlo estimate with its standard error."""

    price: float
    stderr: float

    def within(self, reference: float, n_se: float = 3.0) -> bool:
        """True when ``reference`` lies inside ``n_se`` standard errors."""
        return abs(self.price - reference) <= n_se * max(self.stderr, 1e-300)


def _validate_mc(
    market: Market, strike: float, expiry: float, n_paths: int, n_steps: int, antithetic: bool
) -> None:
    if not isinstance(market, Market):
        raise ValueError("market must be a Market instance")
    if not math.isfinite(strike) or strike < 0.0:
        raise ValueError(f"strike must be finite and >= 0, got {strike!r}")
    if not math.isfinite(expiry) or expiry <= 0.0:
        raise ValueError(f"expiry must be finite and > 0, got {expiry!r}")
    if antithetic:
        if n_paths % 2 != 0:
            raise ValueError("antithetic sampling requires an even n_paths")
        if n_paths < 4:
            raise ValueError(
                f"n_paths must be >= 4 with antithetic sampling (at least two pair "
                f"means are needed for a standard error), got {n_paths!r}"
            )
    elif n_paths < 2:
        raise ValueError(
            f"n_paths must be >= 2 (at least two samples are needed for a standard "
            f"error), got {n_paths!r}"
        )
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps!r}")


def _sigma_lookup(vol: VolInput):
    """Wrap ``vol`` as a checked ``sigma(k_array, t) -> array`` callable.

    A user callable is validated on every call: its output is broadcast to
    the shape of ``k`` and must be finite and ``>= 0`` (a NaN/inf/negative
    sigma would otherwise propagate silently into ``price``/``stderr``).
    """
    if callable(vol):
        user_fn = vol

        def checked(k: NDArray[np.float64], t: float) -> NDArray[np.float64]:
            sig = np.asarray(user_fn(k, t), dtype=np.float64)
            try:
                sig = np.broadcast_to(sig, k.shape)
            except ValueError as exc:
                raise ValueError(
                    f"vol callable returned shape {sig.shape}, expected {k.shape} at t={t:g}"
                ) from exc
            if not (np.all(np.isfinite(sig)) and np.all(sig >= 0.0)):
                raise ValueError(f"vol callable returned non-finite/negative sigma at t={t:g}")
            return sig

        return checked
    sigma = float(vol)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"flat vol must be finite and > 0 for MC, got {sigma!r}")

    def flat(k: NDArray[np.float64], t: float) -> NDArray[np.float64]:
        return np.full_like(k, sigma)

    return flat


def _simulate_terminal(
    market: Market,
    expiry: float,
    vol: VolInput,
    n_paths: int,
    n_steps: int,
    seed: int,
    antithetic: bool,
) -> NDArray[np.float64]:
    """Terminal log-spots, antithetic halves stacked as [+Z block, -Z block]."""
    sigma_fn = _sigma_lookup(vol)
    rng = np.random.default_rng(seed)
    dt = expiry / n_steps
    sq = math.sqrt(dt)
    drift_rq = (market.rate - market.dividend) * dt
    n_base = n_paths // 2 if antithetic else n_paths
    x = np.full(n_base, math.log(market.spot))
    xa = x.copy() if antithetic else None
    for n in range(n_steps):
        t = n * dt
        lf = market.log_forward(t)
        z = rng.standard_normal(n_base)
        sig = np.asarray(sigma_fn(x - lf, t), dtype=np.float64)
        x = x + drift_rq - 0.5 * sig * sig * dt + sig * sq * z
        if antithetic:
            siga = np.asarray(sigma_fn(xa - lf, t), dtype=np.float64)
            xa = xa + drift_rq - 0.5 * siga * siga * dt - siga * sq * z
    return np.concatenate([x, xa]) if antithetic else x


def _estimate(payoffs: NDArray[np.float64], antithetic: bool) -> MCResult:
    if antithetic:
        half = payoffs.size // 2
        samples = 0.5 * (payoffs[:half] + payoffs[half:])
    else:
        samples = payoffs
    mean = float(np.mean(samples))
    stderr = float(np.std(samples, ddof=1) / math.sqrt(samples.size))
    return MCResult(mean, stderr)


def price_european_mc(
    market: Market,
    strike: float,
    expiry: float,
    vol: VolInput,
    is_call: bool = True,
    n_paths: int = 20000,
    n_steps: int = 100,
    seed: int = 42,
    antithetic: bool = True,
) -> MCResult:
    """European vanilla by log-Euler MC (flat vol or ``sigma(k, t)`` callable)."""
    _validate_mc(market, strike, expiry, n_paths, n_steps, antithetic)
    x_t = _simulate_terminal(market, expiry, vol, n_paths, n_steps, seed, antithetic)
    phi = 1.0 if is_call else -1.0
    payoff = np.maximum(phi * (np.exp(x_t) - strike), 0.0)
    df = math.exp(-market.rate * expiry)
    return _estimate(df * payoff, antithetic)


def price_up_out_call_mc(
    market: Market,
    strike: float,
    barrier: float,
    expiry: float,
    vol: VolInput,
    n_paths: int = 20000,
    n_steps: int = 100,
    seed: int = 42,
    antithetic: bool = True,
    brownian_bridge: bool = True,
) -> MCResult:
    """Up-and-out call under local (or flat) vol, optional bridge correction.

    Requires ``S0 < barrier`` (otherwise the option is born knocked out and
    worth exactly 0, which is returned with zero standard error).
    """
    _validate_mc(market, strike, expiry, n_paths, n_steps, antithetic)
    if not math.isfinite(barrier) or barrier <= 0.0:
        raise ValueError(f"barrier must be finite and > 0, got {barrier!r}")
    if market.spot >= barrier:
        return MCResult(0.0, 0.0)

    sigma_fn = _sigma_lookup(vol)
    rng = np.random.default_rng(seed)
    dt = expiry / n_steps
    sq = math.sqrt(dt)
    drift_rq = (market.rate - market.dividend) * dt
    b = math.log(barrier)
    n_base = n_paths // 2 if antithetic else n_paths

    def run_block(sign: float, z_all: NDArray[np.float64]) -> NDArray[np.float64]:
        x = np.full(n_base, math.log(market.spot))
        weight = np.ones(n_base)
        for n in range(n_steps):
            t = n * dt
            sig = np.asarray(sigma_fn(x - market.log_forward(t), t), dtype=np.float64)
            x_new = x + drift_rq - 0.5 * sig * sig * dt + sign * sig * sq * z_all[n]
            hit = x_new >= b
            weight[hit] = 0.0
            if brownian_bridge:
                alive = ~hit & (weight > 0.0)
                # P[bridge from x to x_new crosses b] for x, x_new < b.  A
                # zero step vol gives exp(-inf) = 0: no crossing possible.
                with np.errstate(divide="ignore"):
                    p = np.exp(
                        -2.0 * (b - x[alive]) * (b - x_new[alive]) / (sig[alive] ** 2 * dt)
                    )
                weight[alive] *= 1.0 - p
            x = x_new
        payoff = weight * np.maximum(np.exp(x) - strike, 0.0)
        return payoff

    z_all = rng.standard_normal((n_steps, n_base))
    payoff = run_block(+1.0, z_all)
    if antithetic:
        payoff = np.concatenate([payoff, run_block(-1.0, z_all)])
    df = math.exp(-market.rate * expiry)
    return _estimate(df * payoff, antithetic)
