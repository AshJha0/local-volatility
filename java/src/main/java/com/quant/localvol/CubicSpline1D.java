package com.quant.localvol;

/**
 * Natural cubic spline through {@code (x_i, y_i)}.
 *
 * <p>Natural (zero second derivative at both end nodes) because it is fully
 * determined by the nodes — no tension/tangent parameters — and its C^2
 * smoothness is exactly what the Dupire formula's {@code d2w/dk2} term needs.
 * The interior second-derivative "moments" solve a tridiagonal system handled
 * by the same Thomas kernel the PDE uses, so every language port reproduces
 * the interpolant bit-for-bit closely.
 *
 * <p>Degenerate node counts degrade gracefully: 1 node → constant,
 * 2 nodes → linear. Evaluation clamps the query into {@code [x_0, x_{n-1}]}
 * (flat extrapolation), matching the implied surface's wing rule.
 */
public final class CubicSpline1D {

    private final double[] x;
    private final double[] y;
    private final double[] m; // second derivative at each node ("moments")

    /**
     * Build the natural spline.
     *
     * @param x strictly increasing nodes (length >= 1)
     * @param y values at the nodes (same length as {@code x})
     * @throws IllegalArgumentException on empty/mismatched/non-finite/non-increasing input
     */
    public CubicSpline1D(double[] x, double[] y) {
        if (x == null || y == null || x.length != y.length) {
            throw new IllegalArgumentException("spline: x and y must be equal-length arrays");
        }
        if (x.length == 0) {
            throw new IllegalArgumentException("spline: need at least one node");
        }
        for (int i = 0; i < x.length; i++) {
            if (!Double.isFinite(x[i]) || !Double.isFinite(y[i])) {
                throw new IllegalArgumentException("spline: non-finite node data");
            }
        }
        for (int i = 1; i < x.length; i++) {
            if (!(x[i] > x[i - 1])) {
                throw new IllegalArgumentException("spline: x nodes must be strictly increasing");
            }
        }
        this.x = x.clone();
        this.y = y.clone();
        int n = x.length;
        this.m = new double[n];
        if (n > 2) {
            // Natural spline: m[0] = m[n-1] = 0; the n-2 interior moments solve
            // (h_{i-1}/6) m_{i-1} + ((h_{i-1}+h_i)/3) m_i + (h_i/6) m_{i+1}
            //     = (y_{i+1}-y_i)/h_i - (y_i-y_{i-1})/h_{i-1}
            double[] h = new double[n - 1];
            for (int i = 0; i < n - 1; i++) {
                h[i] = x[i + 1] - x[i];
            }
            int p = n - 2;
            double[] sub = new double[p - 1];
            double[] diag = new double[p];
            double[] sup = new double[p - 1];
            double[] rhs = new double[p];
            for (int i = 0; i < p; i++) {
                diag[i] = (h[i] + h[i + 1]) / 3.0;
                rhs[i] = (y[i + 2] - y[i + 1]) / h[i + 1] - (y[i + 1] - y[i]) / h[i];
                if (i < p - 1) {
                    sub[i] = h[i + 1] / 6.0;
                    sup[i] = h[i + 1] / 6.0;
                }
            }
            double[] interior = Tridiag.thomasSolve(sub, diag, sup, rhs);
            System.arraycopy(interior, 0, m, 1, p);
        }
    }

    /**
     * Evaluate the spline at {@code xq} (clamped to the node range).
     *
     * @param xq query point, finite
     * @return the interpolated value
     */
    public double value(double xq) {
        if (!Double.isFinite(xq)) {
            throw new IllegalArgumentException("spline: non-finite query");
        }
        int n = x.length;
        if (n == 1) {
            return y[0];
        }
        double q = Math.min(Math.max(xq, x[0]), x[n - 1]);
        // Locate i with x[i] <= q <= x[i+1] (right end maps to the last interval).
        int i = locate(q);
        double h = x[i + 1] - x[i];
        double t = (q - x[i]) / h;
        double u = 1.0 - t;
        // Hermite-style cubic in terms of node values and moments:
        // s = y_i u + y_{i+1} t + h^2/6 [ (u^3 - u) m_i + (t^3 - t) m_{i+1} ]
        return y[i] * u + y[i + 1] * t
                + (h * h / 6.0) * ((u * u * u - u) * m[i] + (t * t * t - t) * m[i + 1]);
    }

    private int locate(double q) {
        int lo = 0;
        int hi = x.length - 1;
        while (hi - lo > 1) {
            int mid = (lo + hi) >>> 1;
            if (x[mid] <= q) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return lo;
    }
}
