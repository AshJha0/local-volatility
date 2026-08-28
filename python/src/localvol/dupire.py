"""Dupire local volatility in total-variance (Gatheral) form.

The math
--------
Dupire's original formula gives the local variance as a ratio of strike/time
derivatives of *call prices*.  Substituting the Black-Scholes representation
of those prices in terms of total implied variance ``w(k, T)`` (with
``k = ln(K / F(T))`` the forward log-moneyness) yields Gatheral's equivalent
form, which is far better conditioned numerically because it differentiates
the smooth surface ``w`` instead of near-degenerate call prices:

    sigma_loc^2(k, T) =
                      dw/dT
        -------------------------------------------------------------
        1 - (k/w) dw/dk + 1/4 (-1/4 - 1/w + k^2/w^2) (dw/dk)^2
          + 1/2 d2w/dk2

The numerator ``dw/dT`` is the *forward variance*; positivity is calendar
no-arbitrage.  The denominator is (up to a positive factor) the probability
density term ``g(k)``; positivity is butterfly no-arbitrage.  A flat surface
(``w`` independent of k, linear in T) collapses the formula to
``sigma_loc = implied vol`` identically — the sanity anchor used by the
golden tests.

Numerics
--------
Derivatives are central finite differences applied to the interpolated
surface with **fixed steps** ``DK = 1e-3`` (log-moneyness) and ``DT = 1e-4``
(years); at ``T <= DT`` the time derivative switches to a forward difference
so it never samples negative expiries.  These exact step sizes are part of
the cross-language contract (see API_SPEC.md): with the interpolation scheme
pinned down, ports reproduce the golden local-vol values to well under the
1e-4 tolerance.

Robustness: the local vol is clamped to ``[FLOOR, CAP] = [1%, 500%]``.
A clamp fires when the raw formula leaves that band or is undefined —
numerator <= 0 (calendar arbitrage / flat forward variance) floors, and
denominator <= 0 (butterfly arbitrage in the wings of an over-steep smile)
caps.  Every clamp is counted in :attr:`DupireLocalVol.floor_count` /
:attr:`DupireLocalVol.cap_count` so callers can *report* surface quality
instead of crashing mid-pricing.
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np
from numpy.typing import NDArray

from .surface import ImpliedVolSurface

__all__ = ["DupireLocalVol"]

FloatOrArray = Union[float, NDArray[np.float64]]

DK = 1.0e-3   # central-difference step in log-moneyness
DT = 1.0e-4   # central-difference step in expiry (years)
FLOOR = 0.01  # 1% local-vol floor
CAP = 5.00    # 500% local-vol cap
_W_EPS = 1e-12


class DupireLocalVol:
    """Local-volatility function derived from an :class:`ImpliedVolSurface`.

    Use :meth:`vol` as the ``sigma(k, t)`` callable expected by the PDE and
    Monte Carlo engines (vectorised over ``k``).
    """

    def __init__(self, surface: ImpliedVolSurface) -> None:
        if not isinstance(surface, ImpliedVolSurface):
            raise ValueError("DupireLocalVol requires an ImpliedVolSurface")
        self.surface = surface
        self.floor_count = 0
        self.cap_count = 0

    def reset_counters(self) -> None:
        """Zero the floor/cap violation counters."""
        self.floor_count = 0
        self.cap_count = 0

    def local_variance(self, k: FloatOrArray, expiry: float) -> FloatOrArray:
        """Clamped local variance ``sigma_loc^2(k, T)``; see :meth:`vol`."""
        if not math.isfinite(expiry) or expiry < 0.0:
            raise ValueError(f"expiry must be finite and >= 0, got {expiry!r}")
        scalar = np.isscalar(k) or (isinstance(k, np.ndarray) and k.ndim == 0)
        kq = np.atleast_1d(np.asarray(k, dtype=np.float64))
        if not np.all(np.isfinite(kq)):
            raise ValueError("k must be finite")
        srf = self.surface

        w = np.asarray(srf.total_variance(kq, expiry), dtype=np.float64)
        w_up = np.asarray(srf.total_variance(kq + DK, expiry), dtype=np.float64)
        w_dn = np.asarray(srf.total_variance(kq - DK, expiry), dtype=np.float64)
        dwdk = (w_up - w_dn) / (2.0 * DK)
        d2wdk2 = (w_up - 2.0 * w + w_dn) / (DK * DK)
        if expiry > DT:
            wt_up = np.asarray(srf.total_variance(kq, expiry + DT), dtype=np.float64)
            wt_dn = np.asarray(srf.total_variance(kq, expiry - DT), dtype=np.float64)
            dwdt = (wt_up - wt_dn) / (2.0 * DT)
        else:
            wt_up = np.asarray(srf.total_variance(kq, expiry + DT), dtype=np.float64)
            dwdt = (wt_up - w) / DT

        ws = np.maximum(w, _W_EPS)  # guard 1/w and k/w at (near-)zero variance
        denom = (
            1.0
            - (kq / ws) * dwdk
            + 0.25 * (-0.25 - 1.0 / ws + (kq * kq) / (ws * ws)) * dwdk * dwdk
            + 0.5 * d2wdk2
        )

        var = np.empty_like(ws)
        bad_denom = denom <= 0.0
        bad_num = dwdt <= 0.0
        ok = ~(bad_denom | bad_num)
        var[ok] = dwdt[ok] / denom[ok]
        var[bad_num] = FLOOR * FLOOR          # no forward variance -> floor
        var[bad_denom & ~bad_num] = CAP * CAP  # butterfly-arb wing -> cap

        vol = np.sqrt(var)
        floors = int(np.count_nonzero(vol < FLOOR)) + int(np.count_nonzero(bad_num))
        caps = int(np.count_nonzero(vol > CAP)) + int(np.count_nonzero(bad_denom & ~bad_num))
        self.floor_count += floors
        self.cap_count += caps
        vol = np.clip(vol, FLOOR, CAP)
        out = vol * vol
        return float(out[0]) if scalar else out

    def vol(self, k: FloatOrArray, expiry: float) -> FloatOrArray:
        """Local volatility ``sigma_loc(k, T)``, clamped to ``[1%, 500%]``.

        ``k`` is forward log-moneyness ``ln(level / F(T))``; scalar or array.
        """
        var = self.local_variance(k, expiry)
        if np.isscalar(var):
            return math.sqrt(var)
        return np.sqrt(var)

    @property
    def violation_report(self) -> str:
        """Human-readable clamp summary for demos/logs."""
        return (
            f"local-vol clamps: floor({FLOOR:.0%}) hit {self.floor_count}x, "
            f"cap({CAP:.0%}) hit {self.cap_count}x"
        )
