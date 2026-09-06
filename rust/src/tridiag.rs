//! Native Thomas tridiagonal solver.
//!
//! Solves `A x = d` where `A` is tridiagonal with sub-diagonal `sub`
//! (length `n-1`), diagonal `diag` (length `n`) and super-diagonal `sup`
//! (length `n-1`).  The Thomas algorithm is Gaussian elimination without
//! pivoting in O(n): it is stable for the diagonally-dominant systems
//! produced by the Crank-Nicolson discretisation (the diagonal
//! `1 + theta dt (2a/h^2 + r)` dominates the off-diagonals whenever both
//! off-diagonals are non-negative).  Central differencing gives that only
//! under the mesh Peclet condition `|mu| h <= 2 a`, which a flat vol on the
//! default grid satisfies but a floored local vol with a few percent of
//! carry does not; the PDE therefore switches those nodes to upwind
//! differencing (see [`crate::pde`]), which restores non-negative
//! off-diagonals and hence the diagonal dominance this kernel relies on.
//!
//! No library banded solver is used anywhere in the pricing path — this is
//! the single linear-algebra kernel of the crate (also used for the cubic
//! spline moments), ported line for line from the Python reference.

use crate::error::{invalid, LocalVolError, Result};

const PIVOT_TOL: f64 = 1e-300;

/// Solve a tridiagonal system by the Thomas algorithm.
///
/// * `sub`:  sub-diagonal `a_1..a_{n-1}` (length `n-1`); `sub[i]`
///   multiplies `x[i]` in row `i+1`.
/// * `diag`: main diagonal `b_0..b_{n-1}` (length `n`).
/// * `sup`:  super-diagonal `c_0..c_{n-2}` (length `n-1`); `sup[i]`
///   multiplies `x[i+1]` in row `i`.
/// * `rhs`:  right-hand side (length `n`).
///
/// Returns the solution vector `x` (length `n`).  Errors on inconsistent
/// lengths, an empty system, non-finite input, or a (near-)zero pivot.
pub fn thomas_solve(sub: &[f64], diag: &[f64], sup: &[f64], rhs: &[f64]) -> Result<Vec<f64>> {
    let n = diag.len();
    if n == 0 {
        return Err(invalid("thomas_solve: empty system"));
    }
    let off = n - 1;
    if sub.len() != off || sup.len() != off || rhs.len() != n {
        return Err(invalid(format!(
            "thomas_solve: inconsistent shapes sub={}, diag={}, sup={}, rhs={}",
            sub.len(),
            n,
            sup.len(),
            rhs.len()
        )));
    }
    let finite = |v: &[f64]| v.iter().all(|x| x.is_finite());
    if !(finite(sub) && finite(diag) && finite(sup) && finite(rhs)) {
        return Err(invalid("thomas_solve: non-finite input"));
    }

    // Forward elimination: cp / dp are the modified super-diagonal and rhs.
    let mut cp = vec![0.0_f64; n];
    let mut dp = vec![0.0_f64; n];
    let mut piv = diag[0];
    if piv.abs() < PIVOT_TOL {
        return Err(LocalVolError::Numerical("thomas_solve: zero pivot at row 0".into()));
    }
    cp[0] = if n > 1 { sup[0] / piv } else { 0.0 };
    dp[0] = rhs[0] / piv;
    for i in 1..n {
        piv = diag[i] - sub[i - 1] * cp[i - 1];
        if piv.abs() < PIVOT_TOL {
            return Err(LocalVolError::Numerical(format!(
                "thomas_solve: zero pivot at row {i}"
            )));
        }
        cp[i] = if i < n - 1 { sup[i] / piv } else { 0.0 };
        dp[i] = (rhs[i] - sub[i - 1] * dp[i - 1]) / piv;
    }

    // Back substitution.
    let mut x = vec![0.0_f64; n];
    x[n - 1] = dp[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = dp[i] - cp[i] * x[i + 1];
    }
    Ok(x)
}
