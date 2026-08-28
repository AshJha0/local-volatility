"""Implied-volatility surface in (log-moneyness, expiry) total-variance form.

Coordinates
-----------
The surface is parametrised by *forward* log-moneyness ``k = ln(K / F(T))``
and stores **total implied variance** ``w(k, T) = iv(k, T)^2 * T``.  Working
in ``(k, w)`` has three payoffs:

* the Dupire formula takes its cleanest (Gatheral) form in these variables;
* calendar-arbitrage is simply monotonicity of ``w`` in ``T`` at fixed ``k``;
* the surface is independent of rates/dividends, so one CSV serves equity
  and FX alike (the pricer converts strikes via its own forward curve).

Interpolation ("bicubic-lite")
------------------------------
* **In k** (per expiry pillar): a **natural cubic spline** through the node
  total variances.  Natural (zero second derivative at both end nodes)
  because it is fully determined by the nodes — no tension/tangent
  parameters — and its C^2 smoothness is exactly what the Dupire formula's
  ``d2w/dk2`` term needs.  The spline coefficients come from a tridiagonal
  system solved with the same Thomas kernel the PDE uses, so every language
  port reproduces it exactly.
* **In T** (at fixed k): **linear interpolation of total variance** between
  pillar expiries.  Linear-in-w is the standard "no calendar arbitrage
  between pillars" rule: if ``w`` is nondecreasing at the pillars it stays
  nondecreasing everywhere in between.

Extrapolation
-------------
* ``k`` beyond the node range: **flat** — ``k`` is clamped to
  ``[k_min, k_max]`` before the spline is evaluated (constant-vol wings).
* ``T`` below the first pillar: ``w(k, T) = w(k, T1) * T / T1``
  (flat forward variance, i.e. constant implied vol down to T=0).
* ``T`` beyond the last pillar: linear in ``w`` continuing the slope of the
  last pillar interval, floored at 0 so ``w`` never decreases:
  ``w(k, T) = w(k, Tn) + max(slope, 0) * (T - Tn)`` with
  ``slope = (w(k, Tn) - w(k, T_{n-1})) / (Tn - T_{n-1})``.
* **Single expiry**: flat-forward-variance is assumed on both sides
  (``w(k, T) = w(k, T1) * T / T1`` for all T); a warning is emitted because
  ``dw/dT`` then rests on an assumption rather than data.

Calendar arbitrage is checked node-by-node at construction: any
``w(k_i, T_{j+1}) < w(k_i, T_j) - 1e-12`` is counted in
:attr:`ImpliedVolSurface.calendar_violations` and reported via a warning —
detected and reported, not silently repaired, so callers can decide.
"""

from __future__ import annotations

import csv
import math
import warnings
from pathlib import Path
from typing import Union

import numpy as np
from numpy.typing import NDArray

from .tridiag import thomas_solve

__all__ = ["CubicSpline1D", "ImpliedVolSurface"]

FloatOrArray = Union[float, NDArray[np.float64]]

_CAL_TOL = 1e-12


class CubicSpline1D:
    """Natural cubic spline through ``(x_i, y_i)`` with vectorised evaluation.

    Degenerate node counts degrade gracefully: 1 node -> constant,
    2 nodes -> linear.  Evaluation clamps the query to ``[x_0, x_{n-1}]``
    (flat extrapolation), matching the surface's wing rule.
    """

    def __init__(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.ndim != 1 or x.shape != y.shape:
            raise ValueError(f"spline: x and y must be equal-length 1-D arrays, got {x.shape} vs {y.shape}")
        if x.size == 0:
            raise ValueError("spline: need at least one node")
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise ValueError("spline: non-finite node data")
        if x.size > 1 and not np.all(np.diff(x) > 0.0):
            raise ValueError("spline: x nodes must be strictly increasing")
        self.x = x.copy()
        self.y = y.copy()
        n = x.size
        # m[i] = second derivative of the spline at node i ("moments").
        self.m = np.zeros(n, dtype=np.float64)
        if n > 2:
            h = np.diff(x)  # length n-1
            # Natural spline: m[0] = m[n-1] = 0; interior moments solve a
            # symmetric tridiagonal system (classical formulation).
            sub = h[1:-1] / 6.0                      # length n-3
            diag = (h[:-1] + h[1:]) / 3.0            # length n-2
            sup = h[1:-1] / 6.0                      # length n-3
            slopes = np.diff(y) / h                  # length n-1
            rhs = slopes[1:] - slopes[:-1]           # length n-2
            self.m[1:-1] = thomas_solve(sub, diag, sup, rhs)

    def __call__(self, xq: FloatOrArray) -> FloatOrArray:
        """Evaluate the spline; scalar in -> scalar out, array in -> array out."""
        scalar = np.isscalar(xq) or (isinstance(xq, np.ndarray) and xq.ndim == 0)
        q = np.atleast_1d(np.asarray(xq, dtype=np.float64))
        if not np.all(np.isfinite(q)):
            raise ValueError("spline: non-finite query")
        x, y, m = self.x, self.y, self.m
        n = x.size
        if n == 1:
            out = np.full_like(q, y[0])
            return float(out[0]) if scalar else out
        qc = np.clip(q, x[0], x[-1])
        i = np.clip(np.searchsorted(x, qc, side="right") - 1, 0, n - 2)
        h = x[i + 1] - x[i]
        t = (qc - x[i]) / h
        # Hermite-style cubic in terms of node values and moments:
        # s(x) = y_i (1-t) + y_{i+1} t
        #      + h^2/6 [ ((1-t)^3 - (1-t)) m_i + (t^3 - t) m_{i+1} ]
        u = 1.0 - t
        out = (
            y[i] * u
            + y[i + 1] * t
            + (h * h / 6.0) * ((u * u * u - u) * m[i] + (t * t * t - t) * m[i + 1])
        )
        return float(out[0]) if scalar else out


class ImpliedVolSurface:
    """Total-variance implied surface with the interpolation rules above.

    Args:
        k_nodes: strictly increasing forward log-moneyness nodes (length >= 1).
        expiries: strictly increasing positive pillar expiries (length >= 1).
        vols: implied vols, shape ``(len(expiries), len(k_nodes))``, all > 0.

    Attributes:
        calendar_violations: number of node pairs where total variance
            decreases with expiry (calendar arbitrage), detected at build time.
        single_expiry: True when only one pillar was supplied (flat forward
            variance assumed in T; a UserWarning is emitted).
    """

    def __init__(
        self,
        k_nodes: NDArray[np.float64],
        expiries: NDArray[np.float64],
        vols: NDArray[np.float64],
    ) -> None:
        k = np.asarray(k_nodes, dtype=np.float64)
        t = np.asarray(expiries, dtype=np.float64)
        v = np.asarray(vols, dtype=np.float64)
        if k.ndim != 1 or k.size == 0:
            raise ValueError("k_nodes must be a non-empty 1-D array")
        if t.ndim != 1 or t.size == 0:
            raise ValueError("expiries must be a non-empty 1-D array")
        if v.shape != (t.size, k.size):
            raise ValueError(f"vols must have shape (n_expiries, n_k) = {(t.size, k.size)}, got {v.shape}")
        if not (np.all(np.isfinite(k)) and np.all(np.isfinite(t)) and np.all(np.isfinite(v))):
            raise ValueError("surface inputs must be finite")
        if k.size > 1 and not np.all(np.diff(k) > 0.0):
            raise ValueError("k_nodes must be strictly increasing")
        if not np.all(t > 0.0):
            raise ValueError("expiries must be strictly positive")
        if t.size > 1 and not np.all(np.diff(t) > 0.0):
            raise ValueError("expiries must be strictly increasing")
        if not np.all(v > 0.0):
            raise ValueError("implied vols must be strictly positive")

        self.k_nodes = k.copy()
        self.expiries = t.copy()
        self.vols = v.copy()
        # Total variance at the nodes: w = iv^2 * T.
        self.w_nodes = v * v * t[:, None]

        # Calendar-arbitrage detection (report, don't crash).
        self.calendar_violations = 0
        if t.size > 1:
            dec = self.w_nodes[1:, :] < self.w_nodes[:-1, :] - _CAL_TOL
            self.calendar_violations = int(np.count_nonzero(dec))
            if self.calendar_violations > 0:
                warnings.warn(
                    f"calendar arbitrage: total variance decreases in T at "
                    f"{self.calendar_violations} node(s)",
                    UserWarning,
                    stacklevel=2,
                )

        self.single_expiry = t.size == 1
        if self.single_expiry:
            warnings.warn(
                "single expiry pillar: dw/dT uses the flat-forward-variance assumption",
                UserWarning,
                stacklevel=2,
            )

        self._splines = [CubicSpline1D(k, self.w_nodes[j]) for j in range(t.size)]

    # ------------------------------------------------------------------ I/O
    @classmethod
    def from_csv(cls, path: Union[str, Path]) -> "ImpliedVolSurface":
        """Load from a CSV with header ``T,k,iv`` (one row per grid node).

        The file must contain a full rectangular grid: every (T, k) pair
        exactly once.
        """
        rows: list[tuple[float, float, float]] = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or {"T", "k", "iv"} - set(reader.fieldnames):
                raise ValueError(f"{path}: CSV must have header columns T,k,iv")
            for rec in reader:
                rows.append((float(rec["T"]), float(rec["k"]), float(rec["iv"])))
        if not rows:
            raise ValueError(f"{path}: empty surface CSV")
        ts = np.array(sorted({r[0] for r in rows}))
        ks = np.array(sorted({r[1] for r in rows}))
        vols = np.full((ts.size, ks.size), np.nan)
        ti = {t: j for j, t in enumerate(ts)}
        ki = {k: i for i, k in enumerate(ks)}
        for t, k, iv in rows:
            vols[ti[t], ki[k]] = iv
        if np.any(np.isnan(vols)):
            raise ValueError(f"{path}: surface grid is not rectangular (missing (T,k) pairs)")
        return cls(ks, ts, vols)

    # -------------------------------------------------------------- queries
    def total_variance(self, k: FloatOrArray, expiry: float) -> FloatOrArray:
        """Total implied variance ``w(k, T)`` under the documented rules.

        ``k`` may be a scalar or an array; ``expiry`` is a scalar >= 0.
        ``w(k, 0) = 0`` exactly.
        """
        if not math.isfinite(expiry) or expiry < 0.0:
            raise ValueError(f"expiry must be finite and >= 0, got {expiry!r}")
        scalar = np.isscalar(k) or (isinstance(k, np.ndarray) and k.ndim == 0)
        kq = np.atleast_1d(np.asarray(k, dtype=np.float64))
        if not np.all(np.isfinite(kq)):
            raise ValueError("k must be finite")
        t = self.expiries
        if expiry == 0.0:
            out = np.zeros_like(kq)
            return float(out[0]) if scalar else out
        if self.single_expiry or expiry <= t[0]:
            # Flat forward variance: w scales proportionally with T.  For a
            # single-expiry surface this rule applies on both sides of T1.
            w = self._splines[0](kq) * (expiry / t[0])
        elif expiry >= t[-1]:
            w_last = self._splines[-1](kq)
            w_prev = self._splines[-2](kq)
            slope = (w_last - w_prev) / (t[-1] - t[-2])
            w = w_last + np.maximum(slope, 0.0) * (expiry - t[-1])
        else:
            j = int(np.searchsorted(t, expiry, side="right") - 1)
            lam = (expiry - t[j]) / (t[j + 1] - t[j])
            w = (1.0 - lam) * self._splines[j](kq) + lam * self._splines[j + 1](kq)
        w = np.asarray(w, dtype=np.float64)
        return float(w[0]) if scalar else w

    def implied_vol(self, k: FloatOrArray, expiry: float) -> FloatOrArray:
        """Implied vol ``sqrt(w(k,T)/T)``; at ``T == 0`` the short-end limit
        ``sqrt(w(k, T1)/T1)`` is returned (proportional rule's limit)."""
        if not math.isfinite(expiry) or expiry < 0.0:
            raise ValueError(f"expiry must be finite and >= 0, got {expiry!r}")
        if expiry == 0.0:
            w1 = self.total_variance(k, float(self.expiries[0]))
            return np.sqrt(np.asarray(w1) / self.expiries[0]) if not np.isscalar(w1) else math.sqrt(w1 / self.expiries[0])
        w = self.total_variance(k, expiry)
        if np.isscalar(w):
            return math.sqrt(max(w, 0.0) / expiry)
        return np.sqrt(np.maximum(w, 0.0) / expiry)

    @property
    def k_min(self) -> float:
        return float(self.k_nodes[0])

    @property
    def k_max(self) -> float:
        return float(self.k_nodes[-1])
