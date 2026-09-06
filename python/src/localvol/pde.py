"""Log-spot finite-difference pricer: Crank-Nicolson with Rannacher start.

The PDE
-------
In ``x = ln S`` the backward pricing equation under local volatility
``sigma(S, t)`` reads (``tau = T - t`` is time remaining):

    dV/dtau = 1/2 sigma^2 d2V/dx2 + (r - q - 1/2 sigma^2) dV/dx - r V

discretised on a **uniform** x-grid with central differences and marched in
``tau`` with a theta-scheme (theta = 1 backward Euler, theta = 1/2
Crank-Nicolson).  Tridiagonal systems are solved with the native Thomas
kernel (:mod:`localvol.tridiag`).

Mesh Peclet condition and upwinding
-----------------------------------
Central differencing of the drift term keeps the theta-scheme matrix an
M-matrix (monotone, diagonally dominant) only while ``|mu_i| h <= 2 a_i``
at every node.  With a flat vol on the default grid this always holds, but
under *local* vol the width rule uses ``sigma_ref`` while each node uses its
own ``sigma_i`` — a node floored at 1% by the Dupire clamp with a 1-5% carry
violates it.  At such nodes (and only there) the first derivative switches
to first-order **upwind**: ``lower_i = alpha_i + max(-mu_i, 0)/h``,
``upper_i = alpha_i + max(mu_i, 0)/h``, ``center_i = -2 alpha_i - |mu_i|/h
- r``.  Off-diagonals then stay non-negative, the row sum stays ``-r``, and
the scheme cannot produce the spurious oscillations of a non-monotone
matrix.  The switch is exact-arithmetic identical to central differencing
whenever the condition holds, so flat-vol results are unaffected.

Why Rannacher?
--------------
Crank-Nicolson is unconditionally *stable* but only neutrally damped: its
amplification factor tends to -1 for high-frequency modes, so the
non-smooth payoff (kink at the strike, and for barriers a discontinuity)
excites spurious oscillations in the price — and worse, in gamma — near the
strike that decay only slowly.  Rannacher's fix is to open with fully
implicit (backward Euler) steps, whose amplification factor tends to 0,
killing precisely those high-frequency components; after that CN's
second-order accuracy takes over.  Following the spec, the **first time step
of size dt is replaced by two implicit half-steps of size dt/2**, and the
remaining ``N - 1`` steps are Crank-Nicolson.  With this start the scheme
recovers clean second-order convergence in (dx, dt) for vanilla payoffs
(verified by the grid-convergence test) and a non-oscillatory gamma.

Grid construction (the exact rule ports must copy)
--------------------------------------------------
``x_i = ln S0 + (i - M/2) h`` for ``i = 0..M`` (``M`` even), with half-width

    W = |ln(K/S0)| + nsd * sigma_ref * sqrt(T) + |r - q| * T,   h = 2 W / M.

``ln S0`` is then *exactly* the middle node, so the price is read off with
no interpolation: ``price = V[M/2]``.  The strike is always inside the grid,
the width scales with the diffusion scale ``sigma_ref sqrt(T)`` (default
``nsd = 6`` standard deviations) plus the drift displacement.

Boundary conditions: Dirichlet with discounted asymptotics, e.g. for a call
``V(x_max, tau) = S_max e^{-q tau} - K e^{-r tau}`` (deep ITM value) and
``V(x_min, tau) = 0``.

The local-vol coefficient for the step from ``tau`` to ``tau + dts`` is
frozen at the step midpoint ``t_mid = T - tau - dts/2`` and evaluated at
forward log-moneyness ``k_i = x_i - ln F(t_mid)`` — the same coordinates the
Dupire surface is built in.

American exercise
-----------------
American puts use **PSOR** (projected successive over-relaxation) on the
same theta-scheme systems: each Thomas solve is replaced by an SOR iteration
projected onto the obstacle ``V >= payoff``.  PSOR was chosen over a penalty
/ operator-splitting scheme because it solves the discrete linear
complementarity problem to a controllable tolerance with no penalty
parameter to tune, and warm-starting from the previous time level keeps the
iteration count small on these grids.  Non-convergence is reported via a
warning (not an exception), per the project error policy.
"""

from __future__ import annotations

import math
import warnings
from typing import Callable, Optional, Union

import numpy as np
from numpy.typing import NDArray

from .black_scholes import bs_price
from .market import Market
from .tridiag import thomas_solve

__all__ = ["PDEResult", "price_european_pde", "price_american_put_pde"]

VolInput = Union[float, Callable[[NDArray[np.float64], float], NDArray[np.float64]]]


class PDEResult:
    """Price plus the final grid (for diagnostics such as gamma inspection)."""

    __slots__ = ("price", "x", "values")

    def __init__(self, price: float, x: NDArray[np.float64], values: NDArray[np.float64]) -> None:
        self.price = price
        self.x = x
        self.values = values


def _validate(
    market: Market,
    strike: float,
    expiry: float,
    num_space: int,
    num_time: int,
    nsd: float,
) -> None:
    if not isinstance(market, Market):
        raise ValueError("market must be a Market instance")
    if not math.isfinite(strike) or strike < 0.0:
        raise ValueError(f"strike must be finite and >= 0, got {strike!r}")
    if not math.isfinite(expiry) or expiry < 0.0:
        raise ValueError(f"expiry must be finite and >= 0, got {expiry!r}")
    if num_space < 4 or num_space % 2 != 0:
        raise ValueError(f"num_space must be an even integer >= 4, got {num_space!r}")
    if num_time < 1:
        raise ValueError(f"num_time must be >= 1, got {num_time!r}")
    if not math.isfinite(nsd) or nsd <= 0.0:
        raise ValueError(f"nsd must be finite and > 0, got {nsd!r}")


def _checked_vol_fn(vol_fn: Callable) -> Callable[[NDArray[np.float64], float], NDArray[np.float64]]:
    """Wrap a user ``sigma(k_array, t)`` callable: broadcast a scalar result
    to the node shape and reject non-finite / negative output eagerly (a bad
    coefficient would otherwise surface as an opaque Thomas-solver error or,
    for a negative sigma, silently enter the scheme squared)."""

    def checked(k: NDArray[np.float64], t: float) -> NDArray[np.float64]:
        sig = np.asarray(vol_fn(k, t), dtype=np.float64)
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


def _resolve_vol(
    vol: VolInput, expiry: float, sigma_ref: Optional[float]
) -> tuple[Optional[float], Optional[Callable], float]:
    """Return (flat_sigma or None, checked callable or None, sigma_ref)."""
    if callable(vol):
        fn = _checked_vol_fn(vol)
        if sigma_ref is None:
            sigma_ref = float(fn(np.array([0.0]), expiry)[0])
        if not math.isfinite(sigma_ref) or sigma_ref <= 0.0:
            raise ValueError(f"sigma_ref must be finite and > 0, got {sigma_ref!r}")
        return None, fn, sigma_ref
    sigma = float(vol)
    if not math.isfinite(sigma) or sigma < 0.0:
        raise ValueError(f"flat vol must be finite and >= 0, got {sigma!r}")
    return sigma, None, sigma if sigma > 0.0 else 1.0


def _coefficients(
    sigma_nodes: NDArray[np.float64], r: float, q: float, h: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Interior-node coefficients ``(lower, center, upper)`` of the spatial
    operator ``L``: ``(L V)_i = lower_i V_{i-1} + center_i V_i + upper_i V_{i+1}``.

    Central differences for the first derivative wherever the mesh Peclet
    condition ``|mu_i| h <= 2 a_i`` holds; where it fails (floored local
    vol with strong carry) the node switches to first-order **upwind** so
    that both off-diagonals stay ``>= 0`` and the theta-scheme matrix keeps
    its M-matrix / monotonicity property.  Both branches have row sum ``-r``.
    """
    a = 0.5 * sigma_nodes * sigma_nodes            # diffusion, interior nodes
    mu = (r - q) - a                               # log-spot drift
    alpha = a / (h * h)
    beta = mu / (2.0 * h)

    central = np.abs(mu) * h <= 2.0 * a
    lower = np.where(central, alpha - beta, alpha + np.maximum(-mu, 0.0) / h)
    upper = np.where(central, alpha + beta, alpha + np.maximum(mu, 0.0) / h)
    center = np.where(central, -2.0 * alpha - r, -2.0 * alpha - np.abs(mu) / h - r)
    return lower, center, upper


def _theta_step(
    v: NDArray[np.float64],
    sigma_nodes: NDArray[np.float64],
    r: float,
    q: float,
    h: float,
    dts: float,
    theta: float,
    v0_new: float,
    vm_new: float,
) -> NDArray[np.float64]:
    """One theta-scheme step for the interior nodes; returns the full new level.

    ``v`` is the level at the current tau (boundaries included);
    ``v0_new``/``vm_new`` are the Dirichlet values at the new tau level.
    """
    lower, center, upper = _coefficients(sigma_nodes, r, q, h)

    vi = v[1:-1]
    lv = lower * v[:-2] + center * vi + upper * v[2:]
    rhs = vi + (1.0 - theta) * dts * lv

    sub = -theta * dts * lower[1:]
    diag = 1.0 - theta * dts * center
    sup = -theta * dts * upper[:-1]
    rhs = rhs.copy()
    rhs[0] += theta * dts * lower[0] * v0_new
    rhs[-1] += theta * dts * upper[-1] * vm_new

    out = np.empty_like(v)
    out[1:-1] = thomas_solve(sub, diag, sup, rhs)
    out[0] = v0_new
    out[-1] = vm_new
    return out


def _psor_step(
    v: NDArray[np.float64],
    sigma_nodes: NDArray[np.float64],
    r: float,
    q: float,
    h: float,
    dts: float,
    theta: float,
    v0_new: float,
    vm_new: float,
    obstacle: NDArray[np.float64],
    omega: float,
    tol: float,
    max_iter: int,
) -> NDArray[np.float64]:
    """One theta-scheme step solved as an LCP by projected SOR."""
    lower, center, upper = _coefficients(sigma_nodes, r, q, h)

    vi = v[1:-1]
    lv = lower * v[:-2] + center * vi + upper * v[2:]
    rhs = vi + (1.0 - theta) * dts * lv
    sub = -theta * dts * lower          # sub[i] multiplies x[i-1] in row i
    diag = 1.0 - theta * dts * center
    sup = -theta * dts * upper          # sup[i] multiplies x[i+1] in row i
    rhs = rhs.copy()
    rhs[0] += theta * dts * lower[0] * v0_new
    rhs[-1] += theta * dts * upper[-1] * vm_new

    n = rhs.size
    x = np.maximum(vi, obstacle[1:-1]).copy()  # warm start, feasible
    obs = obstacle[1:-1]
    converged = False
    for _ in range(max_iter):
        err = 0.0
        for i in range(n):
            acc = rhs[i]
            if i > 0:
                acc -= sub[i] * x[i - 1]
            if i < n - 1:
                acc -= sup[i] * x[i + 1]
            gs = acc / diag[i]
            xn = max(obs[i], x[i] + omega * (gs - x[i]))
            err = max(err, abs(xn - x[i]))
            x[i] = xn
        if err < tol:
            converged = True
            break
    if not converged:
        warnings.warn(
            f"PSOR did not converge to {tol:g} within {max_iter} iterations "
            f"(last update {err:g}); continuing with current iterate",
            RuntimeWarning,
            stacklevel=2,
        )
    out = np.empty_like(v)
    out[1:-1] = x
    out[0] = v0_new
    out[-1] = vm_new
    return out


def _build_grid(
    market: Market, strike: float, expiry: float, sigma_ref: float, num_space: int, nsd: float
) -> tuple[NDArray[np.float64], float]:
    x0 = math.log(market.spot)
    half_width = (
        abs(math.log(strike / market.spot))
        + nsd * sigma_ref * math.sqrt(expiry)
        + abs(market.rate - market.dividend) * expiry
    )
    h = 2.0 * half_width / num_space
    i = np.arange(num_space + 1, dtype=np.float64)
    x = x0 + (i - num_space / 2.0) * h
    return x, h


def _time_steps(expiry: float, num_time: int) -> list[tuple[float, float]]:
    """Rannacher schedule: list of (theta, dts) sub-steps summing to expiry.

    The first dt-interval is two backward-Euler half-steps; the remaining
    ``num_time - 1`` intervals are single Crank-Nicolson steps.
    """
    dt = expiry / num_time
    steps = [(1.0, 0.5 * dt), (1.0, 0.5 * dt)]
    steps.extend((0.5, dt) for _ in range(num_time - 1))
    return steps


def price_european_pde(
    market: Market,
    strike: float,
    expiry: float,
    vol: VolInput,
    is_call: bool = True,
    num_space: int = 200,
    num_time: int = 200,
    nsd: float = 6.0,
    sigma_ref: Optional[float] = None,
    return_grid: bool = False,
) -> Union[float, PDEResult]:
    """Price a European vanilla by Crank-Nicolson with Rannacher start.

    Args:
        market: spot / rate / dividend (or FX rd/rf via :meth:`Market.fx`).
        strike: option strike (K = 0 degenerates to a forward-like payoff).
        expiry: maturity in years; ``expiry == 0`` returns intrinsic.
        vol: either a flat volatility (float) or a callable
            ``sigma(k_array, t)`` with ``k = ln(level / F(t))`` — e.g.
            :meth:`localvol.dupire.DupireLocalVol.vol`.
        is_call: payoff type.
        num_space: number of spatial intervals M (even, >= 4); M+1 nodes.
        num_time: number of time intervals N (>= 1).
        nsd: grid half-width in standard deviations (default 6).
        sigma_ref: reference vol for the grid width; defaults to the flat vol
            or, for callables, ``vol(0, T)`` (the ATM local vol at expiry).
        return_grid: when True, return a :class:`PDEResult` with the final
            (t = 0) grid for diagnostics.

    Returns:
        The price at ``S0`` (grid midpoint — no interpolation involved).
    """
    _validate(market, strike, expiry, num_space, num_time, nsd)
    phi = 1.0 if is_call else -1.0
    if expiry == 0.0:
        return max(phi * (market.spot - strike), 0.0)
    flat_sigma, vol_fn, sigma_ref = _resolve_vol(vol, expiry, sigma_ref)
    if flat_sigma is not None and flat_sigma == 0.0:
        # Deterministic limit: PDE degenerates to pure discounting.
        return bs_price(market.spot, strike, market.rate, market.dividend, 0.0, expiry, is_call)
    if strike == 0.0:
        # Degenerate payoff: call pays S_T, put pays 0; both are analytic.
        return market.spot * math.exp(-market.dividend * expiry) if is_call else 0.0

    r, q = market.rate, market.dividend
    x, h = _build_grid(market, strike, expiry, sigma_ref, num_space, nsd)
    s_nodes = np.exp(x)
    v = np.maximum(phi * (s_nodes - strike), 0.0)

    tau = 0.0
    for theta, dts in _time_steps(expiry, num_time):
        t_mid = expiry - tau - 0.5 * dts
        if vol_fn is None:
            sigma_nodes = np.full(num_space - 1, flat_sigma)
        else:
            k_nodes = x[1:-1] - market.log_forward(t_mid)
            sigma_nodes = np.asarray(vol_fn(k_nodes, t_mid), dtype=np.float64)
        tau_new = tau + dts
        if is_call:
            v0_new = 0.0
            vm_new = max(s_nodes[-1] * math.exp(-q * tau_new) - strike * math.exp(-r * tau_new), 0.0)
        else:
            v0_new = max(strike * math.exp(-r * tau_new) - s_nodes[0] * math.exp(-q * tau_new), 0.0)
            vm_new = 0.0
        v = _theta_step(v, sigma_nodes, r, q, h, dts, theta, v0_new, vm_new)
        tau = tau_new

    price = float(v[num_space // 2])
    if return_grid:
        return PDEResult(price, x, v)
    return price


def price_american_put_pde(
    market: Market,
    strike: float,
    expiry: float,
    vol: VolInput,
    num_space: int = 200,
    num_time: int = 200,
    nsd: float = 6.0,
    sigma_ref: Optional[float] = None,
    omega: float = 1.5,
    tol: float = 1e-8,
    max_iter: int = 10000,
    return_grid: bool = False,
) -> Union[float, PDEResult]:
    """Price an American put by PSOR on the Rannacher/CN scheme.

    The obstacle is the immediate-exercise value ``max(K - S, 0)`` applied at
    every node and every time level; boundaries are ``V = K - S_min`` at the
    lower edge (deep ITM exercise) and 0 at the upper edge.

    PSOR parameters: relaxation ``omega`` in (0, 2), sup-norm tolerance
    ``tol``, iteration cap ``max_iter``.  Non-convergence warns and proceeds.
    """
    _validate(market, strike, expiry, num_space, num_time, nsd)
    if not (0.0 < omega < 2.0):
        raise ValueError(f"omega must lie in (0, 2), got {omega!r}")
    # `not (tol > 0)` also rejects NaN, which `tol <= 0` would let through.
    if not (math.isfinite(tol) and tol > 0.0) or max_iter < 1:
        raise ValueError(f"tol must be finite and > 0 and max_iter >= 1, got {tol!r}, {max_iter!r}")
    if strike <= 0.0:
        raise ValueError("American put requires strike > 0")
    if expiry == 0.0:
        return max(strike - market.spot, 0.0)
    flat_sigma, vol_fn, sigma_ref = _resolve_vol(vol, expiry, sigma_ref)
    if flat_sigma is not None and flat_sigma == 0.0:
        # sigma = 0: optimal exercise is deterministic; value is the maximum
        # over exercise dates of the discounted deterministic payoff.
        f = lambda t: max(strike - market.spot * math.exp((market.rate - market.dividend) * t), 0.0) * math.exp(-market.rate * t)
        ts = np.linspace(0.0, expiry, 2001)
        return float(max(f(float(t)) for t in ts))

    r, q = market.rate, market.dividend
    x, h = _build_grid(market, strike, expiry, sigma_ref, num_space, nsd)
    s_nodes = np.exp(x)
    obstacle = np.maximum(strike - s_nodes, 0.0)
    v = obstacle.copy()

    tau = 0.0
    for theta, dts in _time_steps(expiry, num_time):
        t_mid = expiry - tau - 0.5 * dts
        if vol_fn is None:
            sigma_nodes = np.full(num_space - 1, flat_sigma)
        else:
            k_nodes = x[1:-1] - market.log_forward(t_mid)
            sigma_nodes = np.asarray(vol_fn(k_nodes, t_mid), dtype=np.float64)
        v0_new = obstacle[0]  # deep ITM: exercise value K - S_min
        vm_new = 0.0
        v = _psor_step(
            v, sigma_nodes, r, q, h, dts, theta, v0_new, vm_new,
            obstacle, omega, tol, max_iter,
        )
        tau += dts

    price = float(v[num_space // 2])
    if return_grid:
        return PDEResult(price, x, v)
    return price
