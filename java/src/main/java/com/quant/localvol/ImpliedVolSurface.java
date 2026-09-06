package com.quant.localvol;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.TreeSet;

/**
 * Implied-volatility surface in (log-moneyness, expiry) total-variance form.
 *
 * <p>The surface is parametrised by <b>forward</b> log-moneyness
 * {@code k = ln(K / F(T))} and stores <b>total implied variance</b>
 * {@code w(k, T) = iv(k, T)^2 * T}. Working in {@code (k, w)} has three
 * payoffs: the Dupire formula takes its cleanest (Gatheral) form in these
 * variables; calendar arbitrage is simply monotonicity of {@code w} in
 * {@code T} at fixed {@code k}; and the surface is independent of
 * rates/dividends, so one CSV serves equity and FX alike.
 *
 * <p><b>Interpolation ("bicubic-lite")</b>: a natural cubic spline in
 * {@code k} per expiry pillar (see {@link CubicSpline1D}); linear
 * interpolation of total variance in {@code T} between pillars (the standard
 * "no calendar arbitrage between pillars" rule).
 *
 * <p><b>Extrapolation</b>: flat in {@code k} beyond the wings (queries are
 * clamped); {@code w(k, T) = w(k, T1) * T / T1} below the first pillar (flat
 * forward variance); linear-in-w continuation with the last-interval slope
 * floored at 0 beyond the last pillar. A single-expiry surface uses the
 * flat-forward rule for all {@code T} (with a warning, since {@code dw/dT}
 * then rests on an assumption).
 *
 * <p>Calendar arbitrage is checked node-by-node at construction: any
 * {@code w[j+1][i] < w[j][i] - 1e-12} is counted in
 * {@link #calendarViolations()} and reported via a warning on stderr —
 * detected and reported, not silently repaired. The same policy applies to
 * spline overshoot: every node interval of every pillar is probed at 15
 * interior points and any {@code w <= 0} is counted in
 * {@link #negativeWCount()} ({@link #impliedVol} reads 0% there). The
 * surface is immutable after construction and safe to share across threads.
 */
public final class ImpliedVolSurface {

    private static final double CAL_TOL = 1e-12;
    /** Spline-overshoot scan: probes per node interval (minus one). */
    private static final int NEG_W_SUBDIV = 16;

    private final double[] kNodes;
    private final double[] expiries;
    private final double[][] vols;
    private final CubicSpline1D[] splines; // one spline in k per expiry row (of total variance)
    private final int calendarViolations;
    private final long negativeWCount;
    private final boolean singleExpiry;

    /**
     * Build the surface from a rectangular grid of implied vols.
     *
     * @param kNodes   strictly increasing forward log-moneyness nodes (length >= 1)
     * @param expiries strictly increasing positive pillar expiries (length >= 1)
     * @param vols     implied vols, {@code vols[j][i]} at {@code (expiries[j], kNodes[i])}, all > 0
     * @throws IllegalArgumentException on empty, non-finite, non-monotone or non-positive input
     */
    public ImpliedVolSurface(double[] kNodes, double[] expiries, double[][] vols) {
        if (kNodes == null || kNodes.length == 0) {
            throw new IllegalArgumentException("k_nodes must be a non-empty array");
        }
        if (expiries == null || expiries.length == 0) {
            throw new IllegalArgumentException("expiries must be a non-empty array");
        }
        if (vols == null || vols.length != expiries.length) {
            throw new IllegalArgumentException(String.format(
                    "vols must have shape (n_expiries, n_k) = (%d, %d)", expiries.length, kNodes.length));
        }
        for (double[] row : vols) {
            if (row == null || row.length != kNodes.length) {
                throw new IllegalArgumentException(String.format(
                        "vols must have shape (n_expiries, n_k) = (%d, %d)", expiries.length, kNodes.length));
            }
        }
        for (double k : kNodes) {
            if (!Double.isFinite(k)) {
                throw new IllegalArgumentException("surface inputs must be finite");
            }
        }
        for (double t : expiries) {
            if (!Double.isFinite(t)) {
                throw new IllegalArgumentException("surface inputs must be finite");
            }
        }
        for (double[] row : vols) {
            for (double v : row) {
                if (!Double.isFinite(v)) {
                    throw new IllegalArgumentException("surface inputs must be finite");
                }
            }
        }
        for (int i = 1; i < kNodes.length; i++) {
            if (!(kNodes[i] > kNodes[i - 1])) {
                throw new IllegalArgumentException("k_nodes must be strictly increasing");
            }
        }
        for (double t : expiries) {
            if (!(t > 0.0)) {
                throw new IllegalArgumentException("expiries must be strictly positive");
            }
        }
        for (int j = 1; j < expiries.length; j++) {
            if (!(expiries[j] > expiries[j - 1])) {
                throw new IllegalArgumentException("expiries must be strictly increasing");
            }
        }
        for (double[] row : vols) {
            for (double v : row) {
                if (!(v > 0.0)) {
                    throw new IllegalArgumentException("implied vols must be strictly positive");
                }
            }
        }

        this.kNodes = kNodes.clone();
        this.expiries = expiries.clone();
        this.vols = new double[expiries.length][];
        for (int j = 0; j < expiries.length; j++) {
            this.vols[j] = vols[j].clone();
        }

        // Total variance at the nodes: w = iv^2 * T.
        int n = expiries.length;
        int mK = kNodes.length;
        double[][] wNodes = new double[n][mK];
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < mK; i++) {
                wNodes[j][i] = vols[j][i] * vols[j][i] * expiries[j];
            }
        }

        // Calendar-arbitrage detection (report, don't crash).
        int violations = 0;
        for (int j = 0; j + 1 < n; j++) {
            for (int i = 0; i < mK; i++) {
                if (wNodes[j + 1][i] < wNodes[j][i] - CAL_TOL) {
                    violations++;
                }
            }
        }
        this.calendarViolations = violations;
        if (violations > 0) {
            System.err.println("warning: calendar arbitrage: total variance decreases in T at "
                    + violations + " node(s)");
        }

        this.singleExpiry = n == 1;
        if (singleExpiry) {
            System.err.println(
                    "warning: single expiry pillar: dw/dT uses the flat-forward-variance assumption");
        }

        this.splines = new CubicSpline1D[n];
        for (int j = 0; j < n; j++) {
            this.splines[j] = new CubicSpline1D(this.kNodes, wNodes[j]);
        }

        // Spline-overshoot detection (report, don't crash): probe every node
        // interval of every pillar at NEG_W_SUBDIV - 1 interior points and
        // count w <= 0 (impliedVol() would silently read 0% there).
        long negatives = 0;
        for (CubicSpline1D sp : splines) {
            for (int i = 0; i + 1 < mK; i++) {
                double h = this.kNodes[i + 1] - this.kNodes[i];
                for (int p = 1; p < NEG_W_SUBDIV; p++) {
                    if (sp.value(this.kNodes[i] + h * ((double) p / NEG_W_SUBDIV)) <= 0.0) {
                        negatives++;
                    }
                }
            }
        }
        this.negativeWCount = negatives;
        if (negatives > 0) {
            System.err.println("warning: spline overshoot: total variance <= 0 at " + negatives
                    + " probe point(s) between nodes (implied vol reads as 0% there)");
        }
    }

    /**
     * Load a surface from a CSV with header exactly {@code T,k,iv} (one row
     * per grid node, three columns). The file must contain a full rectangular
     * grid: every {@code (T, k)} pair exactly once (row order irrelevant).
     *
     * @param path CSV file path
     * @return the surface
     * @throws IllegalArgumentException on a missing/invalid header, an
     *         unreadable or empty file, a short/long/non-numeric/non-finite
     *         row, a duplicated {@code (T,k)} pair, or a non-rectangular grid
     */
    public static ImpliedVolSurface fromCsv(Path path) {
        List<double[]> rows = new ArrayList<>();
        try (BufferedReader reader = Files.newBufferedReader(path)) {
            String header = reader.readLine();
            if (header == null) {
                throw new IllegalArgumentException(path + ": empty surface CSV");
            }
            String[] cols = header.split(",", -1);
            if (cols.length != 3 || !cols[0].trim().equals("T") || !cols[1].trim().equals("k")
                    || !cols[2].trim().equals("iv")) {
                throw new IllegalArgumentException(path + ": CSV must have header columns T,k,iv");
            }
            String line;
            int lineno = 1;
            while ((line = reader.readLine()) != null) {
                lineno++;
                line = line.trim();
                if (line.isEmpty()) {
                    continue;
                }
                String[] parts = line.split(",", -1);
                if (parts.length != 3) {
                    throw new IllegalArgumentException(path + ": line " + lineno
                            + ": expected 3 columns (T,k,iv), got " + parts.length);
                }
                double[] row = new double[3];
                for (int c = 0; c < 3; c++) {
                    try {
                        row[c] = Double.parseDouble(parts[c].trim());
                    } catch (NumberFormatException e) {
                        throw new IllegalArgumentException(path + ": line " + lineno
                                + ": non-numeric CSV row '" + line + "'", e);
                    }
                    if (!Double.isFinite(row[c])) {
                        throw new IllegalArgumentException(path + ": line " + lineno
                                + ": non-finite CSV value '" + line + "'");
                    }
                }
                rows.add(row);
            }
        } catch (IOException e) {
            throw new IllegalArgumentException(path + ": cannot read surface CSV: " + e.getMessage(), e);
        }
        if (rows.isEmpty()) {
            throw new IllegalArgumentException(path + ": empty surface CSV");
        }
        TreeSet<Double> tSet = new TreeSet<>();
        TreeSet<Double> kSet = new TreeSet<>();
        for (double[] row : rows) {
            tSet.add(row[0]);
            kSet.add(row[1]);
        }
        double[] ts = tSet.stream().mapToDouble(Double::doubleValue).toArray();
        double[] ks = kSet.stream().mapToDouble(Double::doubleValue).toArray();
        double[][] grid = new double[ts.length][ks.length];
        boolean[][] seen = new boolean[ts.length][ks.length];
        for (double[] row : rows) {
            int j = indexOf(ts, row[0]);
            int i = indexOf(ks, row[1]);
            if (seen[j][i]) {
                throw new IllegalArgumentException(
                        path + ": duplicate (T,k) row T=" + row[0] + ", k=" + row[1]);
            }
            grid[j][i] = row[2];
            seen[j][i] = true;
        }
        for (boolean[] rowSeen : seen) {
            for (boolean s : rowSeen) {
                if (!s) {
                    throw new IllegalArgumentException(
                            path + ": surface grid is not rectangular (missing (T,k) pairs)");
                }
            }
        }
        return new ImpliedVolSurface(ks, ts, grid);
    }

    private static int indexOf(double[] sorted, double v) {
        for (int i = 0; i < sorted.length; i++) {
            if (sorted[i] == v) {
                return i;
            }
        }
        throw new IllegalStateException("value not found in sorted axis: " + v);
    }

    /**
     * Total implied variance {@code w(k, T)} under the documented rules;
     * {@code w(k, 0) = 0} exactly.
     *
     * @param k      forward log-moneyness (finite; clamped to the wing nodes)
     * @param expiry maturity in years, finite and {@code >= 0}
     * @return the total variance
     */
    public double totalVariance(double k, double expiry) {
        if (!Double.isFinite(expiry) || expiry < 0.0) {
            throw new IllegalArgumentException("expiry must be finite and >= 0, got " + expiry);
        }
        if (!Double.isFinite(k)) {
            throw new IllegalArgumentException("k must be finite");
        }
        if (expiry == 0.0) {
            return 0.0;
        }
        double[] t = expiries;
        int n = t.length;
        if (singleExpiry || expiry <= t[0]) {
            // Flat forward variance: w scales proportionally with T.  For a
            // single-expiry surface this rule applies on both sides of T1.
            return splines[0].value(k) * (expiry / t[0]);
        }
        if (expiry >= t[n - 1]) {
            double wLast = splines[n - 1].value(k);
            double wPrev = splines[n - 2].value(k);
            double slope = (wLast - wPrev) / (t[n - 1] - t[n - 2]);
            return wLast + Math.max(slope, 0.0) * (expiry - t[n - 1]);
        }
        int j = pillarIndex(expiry);
        double lam = (expiry - t[j]) / (t[j + 1] - t[j]);
        return (1.0 - lam) * splines[j].value(k) + lam * splines[j + 1].value(k);
    }

    private int pillarIndex(double expiry) {
        // largest j with expiries[j] <= expiry (called only for interior T)
        int j = 0;
        while (j + 1 < expiries.length && expiries[j + 1] <= expiry) {
            j++;
        }
        return j;
    }

    /**
     * Implied vol {@code sqrt(w(k,T)/T)}; at {@code T == 0} the short-end
     * limit {@code sqrt(w(k, T1)/T1)} is returned (proportional rule's limit).
     *
     * @param k      forward log-moneyness
     * @param expiry maturity in years, finite and {@code >= 0}
     * @return the Black implied vol
     */
    public double impliedVol(double k, double expiry) {
        if (!Double.isFinite(expiry) || expiry < 0.0) {
            throw new IllegalArgumentException("expiry must be finite and >= 0, got " + expiry);
        }
        if (expiry == 0.0) {
            double w1 = totalVariance(k, expiries[0]);
            return Math.sqrt(w1 / expiries[0]);
        }
        double w = totalVariance(k, expiry);
        return Math.sqrt(Math.max(w, 0.0) / expiry);
    }

    /** @return number of node pairs where total variance decreases with expiry */
    public int calendarViolations() {
        return calendarViolations;
    }

    /**
     * @return number of probe points (15 per node interval, per pillar) where
     *         the natural spline overshoots to total variance {@code <= 0}
     *         ({@link #impliedVol} reads 0% there); detected at construction
     *         and reported on stderr, never repaired; 0 for a clean surface
     */
    public long negativeWCount() {
        return negativeWCount;
    }

    /** @return true when only one pillar expiry was supplied */
    public boolean singleExpiry() {
        return singleExpiry;
    }

    /** @return number of pillar expiries */
    public int numExpiries() {
        return expiries.length;
    }

    /** @return number of log-moneyness nodes */
    public int numKNodes() {
        return kNodes.length;
    }

    /** @return smallest log-moneyness node */
    public double kMin() {
        return kNodes[0];
    }

    /** @return largest log-moneyness node */
    public double kMax() {
        return kNodes[kNodes.length - 1];
    }
}
