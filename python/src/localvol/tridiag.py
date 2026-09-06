"""Native Thomas tridiagonal solver.

Solves ``A x = d`` where ``A`` is tridiagonal with sub-diagonal ``a`` (length
``n-1``), diagonal ``b`` (length ``n``) and super-diagonal ``c`` (length
``n-1``).  The Thomas algorithm is Gaussian elimination without pivoting in
O(n): it is stable for the diagonally-dominant systems produced by the
Crank-Nicolson discretisation (the diagonal ``1 + theta dt (2a/h^2 + r)``
dominates the off-diagonals for any dt, h > 0 whenever both off-diagonals
are non-negative).  Central differencing gives that only under the mesh
Peclet condition ``|mu| h <= 2 a``, which a flat vol on the default grid
satisfies but a floored local vol with a few percent of carry does not;
the PDE therefore switches those nodes to upwind differencing (see
:mod:`localvol.pde`), which restores non-negative off-diagonals and hence
the diagonal dominance this kernel relies on.

No SciPy banded solver is used anywhere in the pricing path — this module is
the single linear-algebra kernel, so the C++/Rust/Java ports reproduce it
line for line.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["thomas_solve"]

_PIVOT_TOL = 1e-300


def thomas_solve(
    sub: NDArray[np.float64],
    diag: NDArray[np.float64],
    sup: NDArray[np.float64],
    rhs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve a tridiagonal system by the Thomas algorithm.

    Args:
        sub:  sub-diagonal ``a_1..a_{n-1}`` (length ``n-1``); ``sub[i]``
              multiplies ``x[i]`` in row ``i+1``.
        diag: main diagonal ``b_0..b_{n-1}`` (length ``n``).
        sup:  super-diagonal ``c_0..c_{n-2}`` (length ``n-1``); ``sup[i]``
              multiplies ``x[i+1]`` in row ``i``.
        rhs:  right-hand side (length ``n``).

    Returns:
        Solution vector ``x`` (length ``n``), a fresh float64 array.

    Raises:
        ValueError: on inconsistent lengths, empty system, non-finite input,
            or a (near-)zero pivot encountered during elimination.
    """
    a = np.asarray(sub, dtype=np.float64)
    b = np.asarray(diag, dtype=np.float64)
    c = np.asarray(sup, dtype=np.float64)
    d = np.asarray(rhs, dtype=np.float64)
    n = b.shape[0]
    if n == 0:
        raise ValueError("thomas_solve: empty system")
    if a.shape != (max(n - 1, 0),) or c.shape != (max(n - 1, 0),) or d.shape != (n,):
        raise ValueError(
            f"thomas_solve: inconsistent shapes sub={a.shape}, diag={b.shape}, "
            f"sup={c.shape}, rhs={d.shape}"
        )
    if not (
        np.all(np.isfinite(a))
        and np.all(np.isfinite(b))
        and np.all(np.isfinite(c))
        and np.all(np.isfinite(d))
    ):
        raise ValueError("thomas_solve: non-finite input")

    # Forward elimination: c'_i and d'_i overwrite copies of c and d.
    cp = np.empty(n, dtype=np.float64)  # modified super-diagonal (cp[n-1] unused)
    dp = np.empty(n, dtype=np.float64)  # modified rhs
    piv = b[0]
    if abs(piv) < _PIVOT_TOL:
        raise ValueError("thomas_solve: zero pivot at row 0")
    cp[0] = c[0] / piv if n > 1 else 0.0
    dp[0] = d[0] / piv
    for i in range(1, n):
        piv = b[i] - a[i - 1] * cp[i - 1]
        if abs(piv) < _PIVOT_TOL:
            raise ValueError(f"thomas_solve: zero pivot at row {i}")
        cp[i] = c[i] / piv if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / piv

    # Back substitution.
    x = np.empty(n, dtype=np.float64)
    x[n - 1] = dp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x
