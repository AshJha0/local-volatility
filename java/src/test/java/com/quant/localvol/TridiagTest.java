package com.quant.localvol;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

import java.util.Random;
import org.junit.Test;

/** Thomas solver vs a dense Gaussian-elimination check, plus validation. */
public class TridiagTest {

    /** Dense partial-pivoting Gaussian elimination — independent test oracle. */
    private static double[] denseSolve(double[][] a, double[] b) {
        int n = b.length;
        double[][] m = new double[n][n + 1];
        for (int i = 0; i < n; i++) {
            System.arraycopy(a[i], 0, m[i], 0, n);
            m[i][n] = b[i];
        }
        for (int col = 0; col < n; col++) {
            int piv = col;
            for (int row = col + 1; row < n; row++) {
                if (Math.abs(m[row][col]) > Math.abs(m[piv][col])) {
                    piv = row;
                }
            }
            double[] tmp = m[piv];
            m[piv] = m[col];
            m[col] = tmp;
            for (int row = col + 1; row < n; row++) {
                double f = m[row][col] / m[col][col];
                for (int c = col; c <= n; c++) {
                    m[row][c] -= f * m[col][c];
                }
            }
        }
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            double acc = m[i][n];
            for (int j = i + 1; j < n; j++) {
                acc -= m[i][j] * x[j];
            }
            x[i] = acc / m[i][i];
        }
        return x;
    }

    @Test
    public void matchesDenseSolveOnRandomSystems() {
        Random rng = new Random(7L);
        for (int n : new int[]{1, 2, 3, 5, 17, 64}) {
            double[] sub = new double[Math.max(n - 1, 0)];
            double[] diag = new double[n];
            double[] sup = new double[Math.max(n - 1, 0)];
            double[] rhs = new double[n];
            for (int i = 0; i < n; i++) {
                // Diagonally dominant, like the CN systems the PDE produces.
                diag[i] = 3.0 + rng.nextDouble();
                rhs[i] = rng.nextDouble() * 2.0 - 1.0;
                if (i < n - 1) {
                    sub[i] = rng.nextDouble() - 0.5;
                    sup[i] = rng.nextDouble() - 0.5;
                }
            }
            double[][] dense = new double[n][n];
            for (int i = 0; i < n; i++) {
                dense[i][i] = diag[i];
                if (i > 0) {
                    dense[i][i - 1] = sub[i - 1];
                }
                if (i < n - 1) {
                    dense[i][i + 1] = sup[i];
                }
            }
            double[] x = Tridiag.thomasSolve(sub, diag, sup, rhs);
            double[] ref = denseSolve(dense, rhs);
            for (int i = 0; i < n; i++) {
                assertEquals("n=" + n + " x[" + i + "]", ref[i], x[i], 1e-12);
            }
        }
    }

    @Test
    public void solvesKnownSystemExactly() {
        // [2 1; 1 2] x = [3; 3] -> x = [1, 1]
        double[] x = Tridiag.thomasSolve(new double[]{1.0}, new double[]{2.0, 2.0},
                new double[]{1.0}, new double[]{3.0, 3.0});
        assertEquals(1.0, x[0], 1e-15);
        assertEquals(1.0, x[1], 1e-15);
    }

    @Test
    public void rejectsBadInput() {
        assertThrows(IllegalArgumentException.class,
                () -> Tridiag.thomasSolve(new double[0], new double[0], new double[0], new double[0]));
        assertThrows(IllegalArgumentException.class,
                () -> Tridiag.thomasSolve(new double[]{1.0}, new double[]{1.0, 1.0},
                        new double[0], new double[]{1.0, 1.0}));
        assertThrows(IllegalArgumentException.class,
                () -> Tridiag.thomasSolve(new double[]{1.0}, new double[]{1.0, Double.NaN},
                        new double[]{1.0}, new double[]{1.0, 1.0}));
        // zero pivot
        assertThrows(IllegalArgumentException.class,
                () -> Tridiag.thomasSolve(new double[]{1.0}, new double[]{0.0, 1.0},
                        new double[]{1.0}, new double[]{1.0, 1.0}));
    }
}
