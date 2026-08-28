package com.quant.localvol;

import java.util.Random;

/**
 * Local-volatility Monte Carlo: log-Euler scheme with antithetic variates.
 *
 * <p><b>Scheme</b>: simulate {@code X = ln S} on a uniform time grid
 * {@code t_n = n dt}, {@code dt = T / N}:
 *
 * <pre>
 * X_{n+1} = X_n + (r - q - 1/2 sigma_n^2) dt + sigma_n sqrt(dt) Z_n
 * </pre>
 *
 * with {@code sigma_n = sigma(k_n, t_n)} looked up at the <b>start</b> of the
 * step, at forward log-moneyness {@code k_n = X_n - ln F(t_n)} — the same
 * coordinates the Dupire surface uses, so PDE and MC discretise the identical
 * diffusion. Log-Euler keeps {@code S > 0} exactly.
 *
 * <p><b>Antithetic variates</b>: paths come in pairs driven by {@code +Z} and
 * {@code -Z} (each mirrored path recomputes its own sigma from its own
 * state). The estimator averages each pair first, and the standard error is
 * computed over the {@code n_paths/2} pair means — raw per-path deviations
 * would overstate the error because paired payoffs are negatively correlated.
 *
 * <p><b>Barrier (up-and-out call)</b>: discrete monitoring kills a path once
 * a step ends at or above {@code ln B}; the optional Brownian-bridge
 * correction (default in the golden case) additionally multiplies each
 * surviving step's weight by {@code 1 - exp(-2 (b - X_n)(b - X_{n+1}) /
 * (sigma_n^2 dt))} — the exact bridge crossing probability — removing the
 * O(sqrt(dt)) discrete-monitoring bias with no extra random numbers.
 *
 * <p><b>RNG</b>: {@code java.util.Random} Gaussians with a fixed seed. Per
 * the contract each language uses its own generator, so MC golden
 * comparisons are statistical (tolerances sized at 4 standard errors).
 */
public final class MonteCarlo {

    private MonteCarlo() {
    }

    /**
     * Monte Carlo estimate with its standard error.
     *
     * @param price  discounted-payoff estimate
     * @param stderr standard error of the estimate
     */
    public record McResult(double price, double stderr) {

        /**
         * @param reference reference value
         * @param nSe       band half-width in standard errors
         * @return true when {@code reference} lies inside {@code nSe} standard errors
         */
        public boolean within(double reference, double nSe) {
            return Math.abs(price - reference) <= nSe * Math.max(stderr, 1e-300);
        }
    }

    private static void validateMc(Market market, double strike, double expiry,
                                   int nPaths, int nSteps, boolean antithetic) {
        if (market == null) {
            throw new IllegalArgumentException("market must be a Market instance");
        }
        if (!Double.isFinite(strike) || strike < 0.0) {
            throw new IllegalArgumentException("strike must be finite and >= 0, got " + strike);
        }
        if (!Double.isFinite(expiry) || expiry <= 0.0) {
            throw new IllegalArgumentException("expiry must be finite and > 0, got " + expiry);
        }
        if (nPaths < 2) {
            throw new IllegalArgumentException("n_paths must be >= 2, got " + nPaths);
        }
        if (antithetic && nPaths % 2 != 0) {
            throw new IllegalArgumentException("antithetic sampling requires an even n_paths");
        }
        if (nSteps < 1) {
            throw new IllegalArgumentException("n_steps must be >= 1, got " + nSteps);
        }
    }

    /** Wrap a flat volatility as a {@link LocalVolFn}; must be finite and positive. */
    private static LocalVolFn flatVol(double sigma) {
        if (!Double.isFinite(sigma) || sigma <= 0.0) {
            throw new IllegalArgumentException("flat vol must be finite and > 0 for MC, got " + sigma);
        }
        return (k, t) -> sigma;
    }

    /** Pair-mean (or raw) estimator with sample-std (ddof = 1) standard error. */
    private static McResult estimate(double[] payoffs, boolean antithetic) {
        double[] samples;
        if (antithetic) {
            int half = payoffs.length / 2;
            samples = new double[half];
            for (int i = 0; i < half; i++) {
                samples[i] = 0.5 * (payoffs[i] + payoffs[half + i]);
            }
        } else {
            samples = payoffs;
        }
        int n = samples.length;
        double mean = 0.0;
        for (double s : samples) {
            mean += s;
        }
        mean /= n;
        double ss = 0.0;
        for (double s : samples) {
            double d = s - mean;
            ss += d * d;
        }
        double stderr = Math.sqrt(ss / (n - 1)) / Math.sqrt(n);
        return new McResult(mean, stderr);
    }

    /**
     * European vanilla by log-Euler MC with a flat volatility.
     *
     * @param market     market data
     * @param strike     option strike
     * @param expiry     maturity in years ({@code > 0})
     * @param sigma      flat volatility ({@code > 0})
     * @param isCall     payoff type
     * @param nPaths     number of paths (even when antithetic)
     * @param nSteps     number of time steps
     * @param seed       RNG seed
     * @param antithetic use antithetic variates
     * @return price and standard error
     */
    public static McResult priceEuropean(Market market, double strike, double expiry, double sigma,
                                         boolean isCall, int nPaths, int nSteps, long seed,
                                         boolean antithetic) {
        validateMc(market, strike, expiry, nPaths, nSteps, antithetic);
        return europeanCore(market, strike, expiry, flatVol(sigma), isCall,
                nPaths, nSteps, seed, antithetic);
    }

    /**
     * European vanilla by log-Euler MC under a local-vol function.
     *
     * @param market     market data
     * @param strike     option strike
     * @param expiry     maturity in years ({@code > 0})
     * @param vol        local-vol function {@code sigma(k, t)}
     * @param isCall     payoff type
     * @param nPaths     number of paths (even when antithetic)
     * @param nSteps     number of time steps
     * @param seed       RNG seed
     * @param antithetic use antithetic variates
     * @return price and standard error
     */
    public static McResult priceEuropean(Market market, double strike, double expiry,
                                         LocalVolFn vol, boolean isCall, int nPaths, int nSteps,
                                         long seed, boolean antithetic) {
        validateMc(market, strike, expiry, nPaths, nSteps, antithetic);
        if (vol == null) {
            throw new IllegalArgumentException("vol function must not be null");
        }
        return europeanCore(market, strike, expiry, vol, isCall, nPaths, nSteps, seed, antithetic);
    }

    private static McResult europeanCore(Market market, double strike, double expiry,
                                         LocalVolFn vol, boolean isCall, int nPaths, int nSteps,
                                         long seed, boolean antithetic) {
        Random rng = new Random(seed);
        double dt = expiry / nSteps;
        double sq = Math.sqrt(dt);
        double driftRq = (market.rate() - market.dividend()) * dt;
        int nBase = antithetic ? nPaths / 2 : nPaths;
        double x0 = Math.log(market.spot());
        double[] x = new double[nBase];
        double[] xa = antithetic ? new double[nBase] : null;
        java.util.Arrays.fill(x, x0);
        if (xa != null) {
            java.util.Arrays.fill(xa, x0);
        }
        double[] z = new double[nBase];
        for (int n = 0; n < nSteps; n++) {
            double t = n * dt;
            double lf = market.logForward(t);
            for (int p = 0; p < nBase; p++) {
                z[p] = rng.nextGaussian();
            }
            for (int p = 0; p < nBase; p++) {
                double sig = vol.vol(x[p] - lf, t);
                x[p] += driftRq - 0.5 * sig * sig * dt + sig * sq * z[p];
            }
            if (xa != null) {
                for (int p = 0; p < nBase; p++) {
                    double sig = vol.vol(xa[p] - lf, t);
                    xa[p] += driftRq - 0.5 * sig * sig * dt - sig * sq * z[p];
                }
            }
        }
        double phi = isCall ? 1.0 : -1.0;
        double df = Math.exp(-market.rate() * expiry);
        int total = antithetic ? 2 * nBase : nBase;
        double[] payoff = new double[total];
        for (int p = 0; p < nBase; p++) {
            payoff[p] = df * Math.max(phi * (Math.exp(x[p]) - strike), 0.0);
        }
        if (xa != null) {
            for (int p = 0; p < nBase; p++) {
                payoff[nBase + p] = df * Math.max(phi * (Math.exp(xa[p]) - strike), 0.0);
            }
        }
        return estimate(payoff, antithetic);
    }

    /**
     * Up-and-out call by MC with a flat volatility (optional bridge correction).
     *
     * @param market         market data
     * @param strike         option strike
     * @param barrier        knock-out barrier {@code B > 0}; {@code S0 >= B}
     *                       returns price = stderr = 0
     * @param expiry         maturity ({@code > 0})
     * @param sigma          flat volatility ({@code > 0})
     * @param nPaths         number of paths (even when antithetic)
     * @param nSteps         number of time steps
     * @param seed           RNG seed
     * @param antithetic     use antithetic variates
     * @param brownianBridge apply the bridge crossing-probability correction
     * @return price and standard error
     */
    public static McResult priceUpOutCall(Market market, double strike, double barrier,
                                          double expiry, double sigma, int nPaths, int nSteps,
                                          long seed, boolean antithetic, boolean brownianBridge) {
        validateMc(market, strike, expiry, nPaths, nSteps, antithetic);
        return upOutCore(market, strike, barrier, expiry, flatVol(sigma),
                nPaths, nSteps, seed, antithetic, brownianBridge);
    }

    /**
     * Up-and-out call by MC under a local-vol function (optional bridge
     * correction). Same contract as the flat-vol overload.
     *
     * @param market         market data
     * @param strike         option strike
     * @param barrier        knock-out barrier {@code B > 0}
     * @param expiry         maturity ({@code > 0})
     * @param vol            local-vol function
     * @param nPaths         number of paths
     * @param nSteps         number of time steps
     * @param seed           RNG seed
     * @param antithetic     use antithetic variates
     * @param brownianBridge apply the bridge correction
     * @return price and standard error
     */
    public static McResult priceUpOutCall(Market market, double strike, double barrier,
                                          double expiry, LocalVolFn vol, int nPaths, int nSteps,
                                          long seed, boolean antithetic, boolean brownianBridge) {
        validateMc(market, strike, expiry, nPaths, nSteps, antithetic);
        if (vol == null) {
            throw new IllegalArgumentException("vol function must not be null");
        }
        return upOutCore(market, strike, barrier, expiry, vol,
                nPaths, nSteps, seed, antithetic, brownianBridge);
    }

    private static McResult upOutCore(Market market, double strike, double barrier, double expiry,
                                      LocalVolFn vol, int nPaths, int nSteps, long seed,
                                      boolean antithetic, boolean brownianBridge) {
        if (!Double.isFinite(barrier) || barrier <= 0.0) {
            throw new IllegalArgumentException("barrier must be finite and > 0, got " + barrier);
        }
        if (market.spot() >= barrier) {
            return new McResult(0.0, 0.0); // born knocked out
        }
        Random rng = new Random(seed);
        double dt = expiry / nSteps;
        double sq = Math.sqrt(dt);
        double driftRq = (market.rate() - market.dividend()) * dt;
        double b = Math.log(barrier);
        int nBase = antithetic ? nPaths / 2 : nPaths;

        // Pre-draw all normals so the antithetic block reuses the same Z.
        double[][] zAll = new double[nSteps][nBase];
        for (int n = 0; n < nSteps; n++) {
            for (int p = 0; p < nBase; p++) {
                zAll[n][p] = rng.nextGaussian();
            }
        }

        int total = antithetic ? 2 * nBase : nBase;
        double[] payoff = new double[total];
        runBarrierBlock(market, strike, expiry, vol, nSteps, nBase, dt, sq, driftRq, b,
                brownianBridge, +1.0, zAll, payoff, 0);
        if (antithetic) {
            runBarrierBlock(market, strike, expiry, vol, nSteps, nBase, dt, sq, driftRq, b,
                    brownianBridge, -1.0, zAll, payoff, nBase);
        }
        double df = Math.exp(-market.rate() * expiry);
        for (int i = 0; i < total; i++) {
            payoff[i] *= df;
        }
        return estimate(payoff, antithetic);
    }

    private static void runBarrierBlock(Market market, double strike, double expiry,
                                        LocalVolFn vol, int nSteps, int nBase, double dt,
                                        double sq, double driftRq, double b,
                                        boolean brownianBridge, double sign, double[][] zAll,
                                        double[] payoffOut, int offset) {
        double x0 = Math.log(market.spot());
        double[] x = new double[nBase];
        double[] weight = new double[nBase];
        java.util.Arrays.fill(x, x0);
        java.util.Arrays.fill(weight, 1.0);
        for (int n = 0; n < nSteps; n++) {
            double t = n * dt;
            double lf = market.logForward(t);
            for (int p = 0; p < nBase; p++) {
                double sig = vol.vol(x[p] - lf, t);
                double xNew = x[p] + driftRq - 0.5 * sig * sig * dt + sign * sig * sq * zAll[n][p];
                if (xNew >= b) {
                    weight[p] = 0.0;
                } else if (brownianBridge && weight[p] > 0.0) {
                    // P[bridge from x to xNew crosses b] for x, xNew < b.
                    double pCross = Math.exp(-2.0 * (b - x[p]) * (b - xNew) / (sig * sig * dt));
                    weight[p] *= 1.0 - pCross;
                }
                x[p] = xNew;
            }
        }
        for (int p = 0; p < nBase; p++) {
            payoffOut[offset + p] = weight[p] * Math.max(Math.exp(x[p]) - strike, 0.0);
        }
    }
}
