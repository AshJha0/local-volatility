package com.quant.localvol;

/**
 * Native Thomas tridiagonal solver.
 *
 * <p>Solves {@code A x = d} where {@code A} is tridiagonal with sub-diagonal
 * {@code a} (length {@code n-1}), diagonal {@code b} (length {@code n}) and
 * super-diagonal {@code c} (length {@code n-1}). The Thomas algorithm is
 * Gaussian elimination without pivoting in O(n): it is stable for the
 * diagonally-dominant systems produced by the Crank-Nicolson discretisation
 * (the diagonal {@code 1 + theta dt (2a/h^2 + r)} dominates the off-diagonals
 * whenever the scheme's local Peclet condition {@code |mu| h <= 2 a} holds,
 * which the PDE grids used here satisfy by construction).
 *
 * <p>No library banded solver is used anywhere in the pricing path — this is
 * the single linear-algebra kernel, ported line for line from the Python
 * reference.
 */
public final class Tridiag {

    private static final double PIVOT_TOL = 1e-300;

    private Tridiag() {
    }

    /**
     * Solve a tridiagonal system by the Thomas algorithm.
     *
     * @param sub  sub-diagonal (length {@code n-1}); {@code sub[i]} multiplies
     *             {@code x[i]} in row {@code i+1}
     * @param diag main diagonal (length {@code n})
     * @param sup  super-diagonal (length {@code n-1}); {@code sup[i]} multiplies
     *             {@code x[i+1]} in row {@code i}
     * @param rhs  right-hand side (length {@code n})
     * @return solution vector {@code x} (length {@code n}), freshly allocated
     * @throws IllegalArgumentException on inconsistent lengths, empty system,
     *         non-finite input, or a (near-)zero pivot during elimination
     */
    public static double[] thomasSolve(double[] sub, double[] diag, double[] sup, double[] rhs) {
        if (sub == null || diag == null || sup == null || rhs == null) {
            throw new IllegalArgumentException("thomasSolve: null input array");
        }
        int n = diag.length;
        if (n == 0) {
            throw new IllegalArgumentException("thomasSolve: empty system");
        }
        int off = Math.max(n - 1, 0);
        if (sub.length != off || sup.length != off || rhs.length != n) {
            throw new IllegalArgumentException(String.format(
                    "thomasSolve: inconsistent shapes sub=%d, diag=%d, sup=%d, rhs=%d",
                    sub.length, diag.length, sup.length, rhs.length));
        }
        requireFinite(sub);
        requireFinite(diag);
        requireFinite(sup);
        requireFinite(rhs);

        // Forward elimination: cp/dp hold the modified super-diagonal and rhs.
        double[] cp = new double[n];
        double[] dp = new double[n];
        double piv = diag[0];
        if (Math.abs(piv) < PIVOT_TOL) {
            throw new IllegalArgumentException("thomasSolve: zero pivot at row 0");
        }
        cp[0] = n > 1 ? sup[0] / piv : 0.0;
        dp[0] = rhs[0] / piv;
        for (int i = 1; i < n; i++) {
            piv = diag[i] - sub[i - 1] * cp[i - 1];
            if (Math.abs(piv) < PIVOT_TOL) {
                throw new IllegalArgumentException("thomasSolve: zero pivot at row " + i);
            }
            cp[i] = i < n - 1 ? sup[i] / piv : 0.0;
            dp[i] = (rhs[i] - sub[i - 1] * dp[i - 1]) / piv;
        }

        // Back substitution.
        double[] x = new double[n];
        x[n - 1] = dp[n - 1];
        for (int i = n - 2; i >= 0; i--) {
            x[i] = dp[i] - cp[i] * x[i + 1];
        }
        return x;
    }

    private static void requireFinite(double[] a) {
        for (double v : a) {
            if (!Double.isFinite(v)) {
                throw new IllegalArgumentException("thomasSolve: non-finite input");
            }
        }
    }
}
