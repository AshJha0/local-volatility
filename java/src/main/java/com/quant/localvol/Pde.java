package com.quant.localvol;

/**
 * Log-spot finite-difference pricer: Crank-Nicolson with Rannacher start.
 *
 * <p>In {@code x = ln S} the backward pricing equation under local volatility
 * {@code sigma(k, t)} reads ({@code tau = T - t} is time remaining):
 *
 * <pre>
 * dV/dtau = 1/2 sigma^2 d2V/dx2 + (r - q - 1/2 sigma^2) dV/dx - r V
 * </pre>
 *
 * discretised on a <b>uniform</b> x-grid with central differences and marched
 * in {@code tau} with a theta-scheme (theta = 1 backward Euler, theta = 1/2
 * Crank-Nicolson). Tridiagonal systems are solved with the native Thomas
 * kernel ({@link Tridiag}).
 *
 * <p><b>Why Rannacher?</b> Crank-Nicolson is unconditionally stable but only
 * neutrally damped: its amplification factor tends to −1 for high-frequency
 * modes, so the payoff kink excites spurious oscillations in the price — and
 * worse, in gamma — near the strike that decay only slowly ("dt too big vs
 * dx" makes them visible even though the scheme never blows up). Rannacher's
 * fix opens with fully implicit steps, whose amplification factor tends to 0,
 * killing precisely those modes; per the contract, the <b>first time step of
 * size dt is replaced by two backward-Euler half-steps of size dt/2</b>, and
 * the remaining {@code N - 1} steps are Crank-Nicolson. Second-order
 * convergence is preserved (verified by the grid-convergence test).
 *
 * <p><b>Grid rule</b> (exact, part of the cross-language contract):
 * {@code x_i = ln S0 + (i - M/2) h} for {@code i = 0..M} with half-width
 * {@code W = |ln(K/S0)| + nsd * sigma_ref * sqrt(T) + |r - q| * T},
 * {@code h = 2 W / M}. {@code ln S0} is exactly the middle node, so the
 * price is read off with no interpolation.
 *
 * <p><b>American exercise</b>: American puts use <b>PSOR</b> (projected SOR)
 * on the same theta-scheme systems. PSOR was chosen over a penalty /
 * operator-splitting scheme because it solves the discrete linear
 * complementarity problem to a controllable tolerance with no penalty
 * parameter to tune, and warm-starting from the previous time level keeps
 * the iteration count small. Non-convergence is reported via a warning on
 * stderr (not an exception), per the project error policy.
 */
public final class Pde {

    /** Default number of spatial intervals. */
    public static final int DEFAULT_NUM_SPACE = 200;
    /** Default number of time intervals. */
    public static final int DEFAULT_NUM_TIME = 200;
    /** Default grid half-width in standard deviations. */
    public static final double DEFAULT_NSD = 6.0;
    /** Default PSOR relaxation parameter. */
    public static final double DEFAULT_OMEGA = 1.5;
    /** Default PSOR sup-norm tolerance. */
    public static final double DEFAULT_TOL = 1e-8;
    /** Default PSOR iteration cap. */
    public static final int DEFAULT_MAX_ITER = 10000;

    private Pde() {
    }

    /**
     * Price plus the final grid (for diagnostics such as gamma inspection).
     * Analytic short-circuits (T = 0, sigma = 0, K = 0) return empty grids.
     *
     * @param price  the option price at {@code S0} (grid midpoint)
     * @param x      log-spot grid nodes (empty for analytic short-circuits)
     * @param values option values on the grid at t = 0
     */
    public record Result(double price, double[] x, double[] values) {
    }

    // ------------------------------------------------------------------ European

    /**
     * European vanilla with a flat volatility (defaults 200x200, nsd = 6).
     *
     * @param market market data
     * @param strike option strike (K = 0 degenerates to a forward-like payoff)
     * @param expiry maturity in years; 0 returns intrinsic
     * @param sigma  flat Black volatility, {@code >= 0}
     * @param isCall payoff type
     * @return the price at {@code S0}
     */
    public static double priceEuropean(Market market, double strike, double expiry,
                                       double sigma, boolean isCall) {
        return priceEuropean(market, strike, expiry, sigma, isCall,
                DEFAULT_NUM_SPACE, DEFAULT_NUM_TIME, DEFAULT_NSD);
    }

    /**
     * European vanilla with a flat volatility.
     *
     * @param market   market data
     * @param strike   option strike
     * @param expiry   maturity in years
     * @param sigma    flat Black volatility, {@code >= 0}
     * @param isCall   payoff type
     * @param numSpace number of spatial intervals M (even, >= 4)
     * @param numTime  number of time intervals N (>= 1)
     * @param nsd      grid half-width in standard deviations (> 0)
     * @return the price at {@code S0}
     */
    public static double priceEuropean(Market market, double strike, double expiry, double sigma,
                                       boolean isCall, int numSpace, int numTime, double nsd) {
        return europeanGrid(market, strike, expiry, sigma, isCall, numSpace, numTime, nsd).price();
    }

    /**
     * European vanilla with a flat volatility, returning the final grid.
     * Same arguments as {@link #priceEuropean(Market, double, double, double,
     * boolean, int, int, double)}.
     *
     * @param market   market data
     * @param strike   option strike
     * @param expiry   maturity in years
     * @param sigma    flat Black volatility
     * @param isCall   payoff type
     * @param numSpace number of spatial intervals
     * @param numTime  number of time intervals
     * @param nsd      grid half-width in standard deviations
     * @return price and final (t = 0) grid
     */
    public static Result europeanGrid(Market market, double strike, double expiry, double sigma,
                                      boolean isCall, int numSpace, int numTime, double nsd) {
        validate(market, strike, expiry, numSpace, numTime, nsd);
        if (!Double.isFinite(sigma) || sigma < 0.0) {
            throw new IllegalArgumentException("flat vol must be finite and >= 0, got " + sigma);
        }
        double sigmaRef = sigma > 0.0 ? sigma : 1.0;
        return europeanCore(market, strike, expiry, sigma, null, sigmaRef,
                isCall, numSpace, numTime, nsd);
    }

    /**
     * European vanilla under a local-volatility function (defaults 200x200,
     * nsd = 6). {@code sigmaRef = NaN} selects the default reference vol
     * {@code vol(0, T)} — the ATM local vol at expiry.
     *
     * @param market   market data
     * @param strike   option strike
     * @param expiry   maturity in years
     * @param vol      local-vol function {@code sigma(k, t)}, e.g. {@link DupireLocalVol}
     * @param sigmaRef reference vol for the grid width (NaN → {@code vol(0, T)})
     * @param isCall   payoff type
     * @return the price at {@code S0}
     */
    public static double priceEuropean(Market market, double strike, double expiry,
                                       LocalVolFn vol, double sigmaRef, boolean isCall) {
        return priceEuropean(market, strike, expiry, vol, sigmaRef, isCall,
                DEFAULT_NUM_SPACE, DEFAULT_NUM_TIME, DEFAULT_NSD);
    }

    /**
     * European vanilla under a local-volatility function.
     *
     * @param market   market data
     * @param strike   option strike
     * @param expiry   maturity in years
     * @param vol      local-vol function {@code sigma(k, t)}
     * @param sigmaRef reference vol for the grid width (NaN → {@code vol(0, T)})
     * @param isCall   payoff type
     * @param numSpace number of spatial intervals M (even, >= 4)
     * @param numTime  number of time intervals N (>= 1)
     * @param nsd      grid half-width in standard deviations (> 0)
     * @return the price at {@code S0}
     */
    public static double priceEuropean(Market market, double strike, double expiry, LocalVolFn vol,
                                       double sigmaRef, boolean isCall,
                                       int numSpace, int numTime, double nsd) {
        validate(market, strike, expiry, numSpace, numTime, nsd);
        if (vol == null) {
            throw new IllegalArgumentException("vol function must not be null");
        }
        double ref = resolveSigmaRef(vol, expiry, sigmaRef);
        return europeanCore(market, strike, expiry, null, vol, ref,
                isCall, numSpace, numTime, nsd).price();
    }

    // ------------------------------------------------------------------ American

    /**
     * American put by PSOR with default grid and PSOR settings.
     *
     * @param market market data
     * @param strike option strike ({@code > 0})
     * @param expiry maturity in years
     * @param sigma  flat Black volatility, {@code >= 0}
     * @return the price at {@code S0}
     */
    public static double priceAmericanPut(Market market, double strike, double expiry, double sigma) {
        return priceAmericanPut(market, strike, expiry, sigma,
                DEFAULT_NUM_SPACE, DEFAULT_NUM_TIME, DEFAULT_NSD,
                DEFAULT_OMEGA, DEFAULT_TOL, DEFAULT_MAX_ITER);
    }

    /**
     * American put by PSOR on the Rannacher/CN scheme, flat volatility.
     *
     * <p>The obstacle is the immediate-exercise value {@code max(K - S, 0)};
     * boundaries are {@code V = K - S_min} at the lower edge (deep ITM
     * exercise) and 0 at the upper edge.
     *
     * @param market   market data
     * @param strike   option strike ({@code > 0})
     * @param expiry   maturity in years
     * @param sigma    flat Black volatility
     * @param numSpace number of spatial intervals M (even, >= 4)
     * @param numTime  number of time intervals N (>= 1)
     * @param nsd      grid half-width in standard deviations
     * @param omega    PSOR relaxation, in (0, 2)
     * @param tol      PSOR sup-norm tolerance (> 0)
     * @param maxIter  PSOR iteration cap (>= 1); hitting it warns and proceeds
     * @return the price at {@code S0}
     */
    public static double priceAmericanPut(Market market, double strike, double expiry, double sigma,
                                          int numSpace, int numTime, double nsd,
                                          double omega, double tol, int maxIter) {
        validateAmerican(market, strike, expiry, numSpace, numTime, nsd, omega, tol, maxIter);
        if (!Double.isFinite(sigma) || sigma < 0.0) {
            throw new IllegalArgumentException("flat vol must be finite and >= 0, got " + sigma);
        }
        if (expiry == 0.0) {
            return Math.max(strike - market.spot(), 0.0);
        }
        if (sigma == 0.0) {
            // sigma = 0: optimal exercise is deterministic; value is the maximum
            // over exercise dates of the discounted deterministic payoff
            // (reference maximises over 2001 equidistant t).
            double best = 0.0;
            for (int i = 0; i <= 2000; i++) {
                double t = expiry * i / 2000.0;
                double payoff = Math.max(
                        strike - market.spot() * Math.exp((market.rate() - market.dividend()) * t), 0.0);
                best = Math.max(best, payoff * Math.exp(-market.rate() * t));
            }
            return best;
        }
        return americanCore(market, strike, expiry, sigma, null, sigma,
                numSpace, numTime, nsd, omega, tol, maxIter).price();
    }

    /**
     * American put by PSOR under a local-volatility function with default
     * grid and PSOR settings. {@code sigmaRef = NaN} → {@code vol(0, T)}.
     *
     * @param market   market data
     * @param strike   option strike ({@code > 0})
     * @param expiry   maturity in years
     * @param vol      local-vol function
     * @param sigmaRef reference vol for the grid width (NaN → {@code vol(0, T)})
     * @return the price at {@code S0}
     */
    public static double priceAmericanPut(Market market, double strike, double expiry,
                                          LocalVolFn vol, double sigmaRef) {
        return priceAmericanPut(market, strike, expiry, vol, sigmaRef,
                DEFAULT_NUM_SPACE, DEFAULT_NUM_TIME, DEFAULT_NSD,
                DEFAULT_OMEGA, DEFAULT_TOL, DEFAULT_MAX_ITER);
    }

    /**
     * American put by PSOR under a local-volatility function.
     *
     * @param market   market data
     * @param strike   option strike ({@code > 0})
     * @param expiry   maturity in years
     * @param vol      local-vol function
     * @param sigmaRef reference vol for the grid width (NaN → {@code vol(0, T)})
     * @param numSpace number of spatial intervals M (even, >= 4)
     * @param numTime  number of time intervals N (>= 1)
     * @param nsd      grid half-width in standard deviations
     * @param omega    PSOR relaxation, in (0, 2)
     * @param tol      PSOR sup-norm tolerance (> 0)
     * @param maxIter  PSOR iteration cap (>= 1)
     * @return the price at {@code S0}
     */
    public static double priceAmericanPut(Market market, double strike, double expiry,
                                          LocalVolFn vol, double sigmaRef,
                                          int numSpace, int numTime, double nsd,
                                          double omega, double tol, int maxIter) {
        validateAmerican(market, strike, expiry, numSpace, numTime, nsd, omega, tol, maxIter);
        if (vol == null) {
            throw new IllegalArgumentException("vol function must not be null");
        }
        if (expiry == 0.0) {
            return Math.max(strike - market.spot(), 0.0);
        }
        double ref = resolveSigmaRef(vol, expiry, sigmaRef);
        return americanCore(market, strike, expiry, null, vol, ref,
                numSpace, numTime, nsd, omega, tol, maxIter).price();
    }

    // ------------------------------------------------------------------ internals

    private static void validate(Market market, double strike, double expiry,
                                 int numSpace, int numTime, double nsd) {
        if (market == null) {
            throw new IllegalArgumentException("market must be a Market instance");
        }
        if (!Double.isFinite(strike) || strike < 0.0) {
            throw new IllegalArgumentException("strike must be finite and >= 0, got " + strike);
        }
        if (!Double.isFinite(expiry) || expiry < 0.0) {
            throw new IllegalArgumentException("expiry must be finite and >= 0, got " + expiry);
        }
        if (numSpace < 4 || numSpace % 2 != 0) {
            throw new IllegalArgumentException("num_space must be an even integer >= 4, got " + numSpace);
        }
        if (numTime < 1) {
            throw new IllegalArgumentException("num_time must be >= 1, got " + numTime);
        }
        if (!Double.isFinite(nsd) || nsd <= 0.0) {
            throw new IllegalArgumentException("nsd must be finite and > 0, got " + nsd);
        }
    }

    private static void validateAmerican(Market market, double strike, double expiry,
                                         int numSpace, int numTime, double nsd,
                                         double omega, double tol, int maxIter) {
        validate(market, strike, expiry, numSpace, numTime, nsd);
        if (!(omega > 0.0 && omega < 2.0)) {
            throw new IllegalArgumentException("omega must lie in (0, 2), got " + omega);
        }
        if (tol <= 0.0 || maxIter < 1) {
            throw new IllegalArgumentException("tol must be > 0 and max_iter >= 1");
        }
        if (strike <= 0.0) {
            throw new IllegalArgumentException("American put requires strike > 0");
        }
    }

    private static double resolveSigmaRef(LocalVolFn vol, double expiry, double sigmaRef) {
        double ref = Double.isNaN(sigmaRef) ? vol.vol(0.0, expiry) : sigmaRef;
        if (!Double.isFinite(ref) || ref <= 0.0) {
            throw new IllegalArgumentException("sigma_ref must be finite and > 0, got " + ref);
        }
        return ref;
    }

    /**
     * Grid rule: {@code W = |ln(K/S0)| + nsd sigma_ref sqrt(T) + |r - q| T},
     * {@code h = 2W/M}, {@code x_i = ln S0 + (i - M/2) h}.
     */
    private static double[] buildGrid(Market market, double strike, double expiry,
                                      double sigmaRef, int numSpace, double nsd) {
        double x0 = Math.log(market.spot());
        double halfWidth = Math.abs(Math.log(strike / market.spot()))
                + nsd * sigmaRef * Math.sqrt(expiry)
                + Math.abs(market.rate() - market.dividend()) * expiry;
        double h = 2.0 * halfWidth / numSpace;
        double[] x = new double[numSpace + 1];
        for (int i = 0; i <= numSpace; i++) {
            x[i] = x0 + (i - numSpace / 2.0) * h;
        }
        return x;
    }

    /**
     * Rannacher schedule: the first dt-interval is two backward-Euler
     * half-steps (theta = 1, dt/2 each), then N-1 Crank-Nicolson steps
     * (theta = 1/2, dt). Returned as {theta, dts} pairs.
     */
    private static double[][] timeSteps(double expiry, int numTime) {
        double dt = expiry / numTime;
        double[][] steps = new double[numTime + 1][2];
        steps[0] = new double[]{1.0, 0.5 * dt};
        steps[1] = new double[]{1.0, 0.5 * dt};
        for (int i = 0; i < numTime - 1; i++) {
            steps[i + 2] = new double[]{0.5, dt};
        }
        return steps;
    }

    private static double[] volNodes(double[] x, Double flatSigma, LocalVolFn fn,
                                     Market market, double tMid) {
        int n = x.length - 2; // interior nodes
        double[] sigma = new double[n];
        if (fn == null) {
            java.util.Arrays.fill(sigma, flatSigma);
        } else {
            double lf = market.logForward(tMid);
            for (int i = 0; i < n; i++) {
                sigma[i] = fn.vol(x[i + 1] - lf, tMid);
            }
        }
        return sigma;
    }

    /**
     * Discrete theta-scheme coefficients for the interior nodes:
     * {@code lower_i = alpha_i - beta_i}, {@code upper_i = alpha_i + beta_i},
     * {@code center_i = -2 alpha_i - r} with {@code alpha = a/h^2},
     * {@code beta = mu/(2h)}, {@code a = sigma^2/2}, {@code mu = r - q - a}.
     */
    private static void coefficients(double[] sigmaNodes, double r, double q, double h,
                                     double[] lower, double[] center, double[] upper) {
        int n = sigmaNodes.length;
        for (int i = 0; i < n; i++) {
            double a = 0.5 * sigmaNodes[i] * sigmaNodes[i];
            double mu = (r - q) - a;
            double alpha = a / (h * h);
            double beta = mu / (2.0 * h);
            lower[i] = alpha - beta;
            upper[i] = alpha + beta;
            center[i] = -2.0 * alpha - r;
        }
    }

    /** RHS of the theta scheme, including the explicit part and boundary terms. */
    private static double[] thetaRhs(double[] v, double[] lower, double[] center, double[] upper,
                                     double dts, double theta, double v0New, double vmNew) {
        int n = v.length - 2;
        double[] rhs = new double[n];
        for (int i = 0; i < n; i++) {
            double lv = lower[i] * v[i] + center[i] * v[i + 1] + upper[i] * v[i + 2];
            rhs[i] = v[i + 1] + (1.0 - theta) * dts * lv;
        }
        rhs[0] += theta * dts * lower[0] * v0New;
        rhs[n - 1] += theta * dts * upper[n - 1] * vmNew;
        return rhs;
    }

    /** One theta-scheme step for the interior nodes; returns the full new level. */
    private static double[] thetaStep(double[] v, double[] sigmaNodes, double r, double q,
                                      double h, double dts, double theta,
                                      double v0New, double vmNew) {
        int n = v.length - 2;
        double[] lower = new double[n];
        double[] center = new double[n];
        double[] upper = new double[n];
        coefficients(sigmaNodes, r, q, h, lower, center, upper);
        double[] rhs = thetaRhs(v, lower, center, upper, dts, theta, v0New, vmNew);

        double[] sub = new double[n - 1];
        double[] diag = new double[n];
        double[] sup = new double[n - 1];
        for (int i = 0; i < n; i++) {
            diag[i] = 1.0 - theta * dts * center[i];
            if (i > 0) {
                sub[i - 1] = -theta * dts * lower[i];
            }
            if (i < n - 1) {
                sup[i] = -theta * dts * upper[i];
            }
        }
        double[] interior = Tridiag.thomasSolve(sub, diag, sup, rhs);
        double[] out = new double[v.length];
        System.arraycopy(interior, 0, out, 1, n);
        out[0] = v0New;
        out[n + 1] = vmNew;
        return out;
    }

    /**
     * One theta-scheme step solved as a linear complementarity problem by
     * projected SOR with obstacle {@code psi = max(K - S, 0)}. Warm start from
     * {@code max(V, psi)} (feasible); ascending Gauss-Seidel sweeps with
     * over-relaxation {@code omega} and projection onto the obstacle; stop at
     * sup-norm update {@code < tol}. Non-convergence warns and proceeds.
     */
    private static double[] psorStep(double[] v, double[] sigmaNodes, double r, double q,
                                     double h, double dts, double theta,
                                     double v0New, double vmNew, double[] obstacle,
                                     double omega, double tol, int maxIter) {
        int n = v.length - 2;
        double[] lower = new double[n];
        double[] center = new double[n];
        double[] upper = new double[n];
        coefficients(sigmaNodes, r, q, h, lower, center, upper);
        double[] rhs = thetaRhs(v, lower, center, upper, dts, theta, v0New, vmNew);

        // Row-indexed matrix entries: sub[i] multiplies x[i-1], sup[i] x[i+1].
        double[] sub = new double[n];
        double[] diag = new double[n];
        double[] sup = new double[n];
        for (int i = 0; i < n; i++) {
            sub[i] = -theta * dts * lower[i];
            diag[i] = 1.0 - theta * dts * center[i];
            sup[i] = -theta * dts * upper[i];
        }

        double[] x = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = Math.max(v[i + 1], obstacle[i + 1]); // warm start, feasible
        }
        boolean converged = false;
        double err = 0.0;
        for (int iter = 0; iter < maxIter; iter++) {
            err = 0.0;
            for (int i = 0; i < n; i++) {
                double acc = rhs[i];
                if (i > 0) {
                    acc -= sub[i] * x[i - 1];
                }
                if (i < n - 1) {
                    acc -= sup[i] * x[i + 1];
                }
                double gs = acc / diag[i];
                double xn = Math.max(obstacle[i + 1], x[i] + omega * (gs - x[i]));
                err = Math.max(err, Math.abs(xn - x[i]));
                x[i] = xn;
            }
            if (err < tol) {
                converged = true;
                break;
            }
        }
        if (!converged) {
            System.err.printf(
                    "warning: PSOR did not converge to %g within %d iterations "
                            + "(last update %g); continuing with current iterate%n",
                    tol, maxIter, err);
        }
        double[] out = new double[v.length];
        System.arraycopy(x, 0, out, 1, n);
        out[0] = v0New;
        out[n + 1] = vmNew;
        return out;
    }

    private static Result europeanCore(Market market, double strike, double expiry,
                                       Double flatSigma, LocalVolFn fn, double sigmaRef,
                                       boolean isCall, int numSpace, int numTime, double nsd) {
        double phi = isCall ? 1.0 : -1.0;
        if (expiry == 0.0) {
            return new Result(Math.max(phi * (market.spot() - strike), 0.0),
                    new double[0], new double[0]);
        }
        if (fn == null && flatSigma == 0.0) {
            // Deterministic limit: PDE degenerates to pure discounting.
            return new Result(BlackScholes.bsPrice(market.spot(), strike, market.rate(),
                    market.dividend(), 0.0, expiry, isCall), new double[0], new double[0]);
        }
        if (strike == 0.0) {
            // Degenerate payoff: call pays S_T, put pays 0; both are analytic.
            double p = isCall ? market.spot() * Math.exp(-market.dividend() * expiry) : 0.0;
            return new Result(p, new double[0], new double[0]);
        }

        double r = market.rate();
        double q = market.dividend();
        double[] x = buildGrid(market, strike, expiry, sigmaRef, numSpace, nsd);
        double h = x[1] - x[0];
        double[] sNodes = new double[x.length];
        double[] v = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            sNodes[i] = Math.exp(x[i]);
            v[i] = Math.max(phi * (sNodes[i] - strike), 0.0);
        }

        double tau = 0.0;
        for (double[] step : timeSteps(expiry, numTime)) {
            double theta = step[0];
            double dts = step[1];
            double tMid = expiry - tau - 0.5 * dts;
            double[] sigmaNodes = volNodes(x, flatSigma, fn, market, tMid);
            double tauNew = tau + dts;
            double v0New;
            double vmNew;
            if (isCall) {
                v0New = 0.0;
                vmNew = Math.max(sNodes[numSpace] * Math.exp(-q * tauNew)
                        - strike * Math.exp(-r * tauNew), 0.0);
            } else {
                v0New = Math.max(strike * Math.exp(-r * tauNew)
                        - sNodes[0] * Math.exp(-q * tauNew), 0.0);
                vmNew = 0.0;
            }
            v = thetaStep(v, sigmaNodes, r, q, h, dts, theta, v0New, vmNew);
            tau = tauNew;
        }
        return new Result(v[numSpace / 2], x, v);
    }

    private static Result americanCore(Market market, double strike, double expiry,
                                       Double flatSigma, LocalVolFn fn, double sigmaRef,
                                       int numSpace, int numTime, double nsd,
                                       double omega, double tol, int maxIter) {
        double r = market.rate();
        double q = market.dividend();
        double[] x = buildGrid(market, strike, expiry, sigmaRef, numSpace, nsd);
        double h = x[1] - x[0];
        double[] obstacle = new double[x.length];
        double[] v = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            obstacle[i] = Math.max(strike - Math.exp(x[i]), 0.0);
            v[i] = obstacle[i];
        }

        double tau = 0.0;
        for (double[] step : timeSteps(expiry, numTime)) {
            double theta = step[0];
            double dts = step[1];
            double tMid = expiry - tau - 0.5 * dts;
            double[] sigmaNodes = volNodes(x, flatSigma, fn, market, tMid);
            double v0New = obstacle[0]; // deep ITM: exercise value K - S_min
            double vmNew = 0.0;
            v = psorStep(v, sigmaNodes, r, q, h, dts, theta, v0New, vmNew,
                    obstacle, omega, tol, maxIter);
            tau += dts;
        }
        return new Result(v[numSpace / 2], x, v);
    }
}
