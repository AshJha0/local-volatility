package com.quant.localvol;

/**
 * Black-Scholes / Garman-Kohlhagen analytics.
 *
 * <p>All formulas are written in terms of {@code (r, q)}; for FX simply pass
 * {@code r = rd} and {@code q = rf} (Garman-Kohlhagen). The normal CDF is
 * computed via {@code erfc} for accuracy in the far tails; Java's standard
 * library has no {@code erfc}, so a compact double-precision implementation
 * (Maclaurin series with positive terms for small arguments, Legendre
 * continued fraction for the tail) is bundled — relative error is below
 * 1e-13 across the range the pricers use.
 */
public final class BlackScholes {

    private static final double SQRT2 = Math.sqrt(2.0);
    private static final double SQRT_PI = Math.sqrt(Math.PI);
    private static final double INV_SQRT_2PI = 1.0 / Math.sqrt(2.0 * Math.PI);

    private BlackScholes() {
    }

    /**
     * Complementary error function, {@code erfc(x) = 1 - erf(x)}.
     *
     * <p>For {@code |x| < 2} the non-alternating Maclaurin series
     * {@code erf(x) = (2x e^{-x^2}/sqrt(pi)) * sum (2x^2)^n / (2n+1)!!} is
     * used (every term positive — no cancellation); for {@code x >= 2} the
     * Legendre continued fraction
     * {@code erfc(x) = (e^{-x^2}/sqrt(pi)) / (x + (1/2)/(x + 1/(x + (3/2)/(x + ...))))}
     * evaluated bottom-up. Negative arguments reflect via
     * {@code erfc(-x) = 2 - erfc(x)}.
     *
     * @param x argument
     * @return {@code erfc(x)}
     */
    public static double erfc(double x) {
        if (Double.isNaN(x)) {
            return Double.NaN;
        }
        double ax = Math.abs(x);
        double result;
        if (ax < 2.0) {
            // erf via all-positive-term series; erfc = 1 - erf loses no
            // relative accuracy for erfc down to erfc(2) ~ 4.7e-3.
            double x2 = 2.0 * ax * ax;
            double term = 1.0;
            double sum = 1.0;
            for (int n = 1; n < 200; n++) {
                term *= x2 / (2.0 * n + 1.0);
                sum += term;
                if (term < 1e-18 * sum) {
                    break;
                }
            }
            double erf = 2.0 * ax * Math.exp(-ax * ax) / SQRT_PI * sum;
            result = 1.0 - erf;
        } else {
            // Continued fraction, evaluated bottom-up with fixed depth: for
            // x >= 2 sixty levels are far beyond double-precision convergence.
            double f = ax;
            for (int n = 60; n >= 1; n--) {
                f = ax + (0.5 * n) / f;
            }
            result = Math.exp(-ax * ax) / (SQRT_PI * f);
        }
        return x >= 0.0 ? result : 2.0 - result;
    }

    /**
     * Standard normal CDF via {@code erfc} (tail-accurate).
     *
     * @param x quantile
     * @return {@code N(x)}
     */
    public static double normCdf(double x) {
        return 0.5 * erfc(-x / SQRT2);
    }

    /**
     * Standard normal density.
     *
     * @param x quantile
     * @return {@code n(x)}
     */
    public static double normPdf(double x) {
        return INV_SQRT_2PI * Math.exp(-0.5 * x * x);
    }

    private static void validateCommon(double spot, double strike, double rate,
                                       double dividend, double sigma, double expiry) {
        if (!Double.isFinite(spot)) {
            throw new IllegalArgumentException("spot must be finite, got " + spot);
        }
        if (!Double.isFinite(strike)) {
            throw new IllegalArgumentException("strike must be finite, got " + strike);
        }
        if (!Double.isFinite(rate)) {
            throw new IllegalArgumentException("rate must be finite, got " + rate);
        }
        if (!Double.isFinite(dividend)) {
            throw new IllegalArgumentException("dividend must be finite, got " + dividend);
        }
        if (!Double.isFinite(sigma)) {
            throw new IllegalArgumentException("sigma must be finite, got " + sigma);
        }
        if (!Double.isFinite(expiry)) {
            throw new IllegalArgumentException("expiry must be finite, got " + expiry);
        }
        if (spot <= 0.0) {
            throw new IllegalArgumentException("spot must be > 0, got " + spot);
        }
        if (strike < 0.0) {
            throw new IllegalArgumentException("strike must be >= 0, got " + strike);
        }
        if (sigma < 0.0) {
            throw new IllegalArgumentException("sigma must be >= 0, got " + sigma);
        }
        if (expiry < 0.0) {
            throw new IllegalArgumentException("expiry must be >= 0, got " + expiry);
        }
    }

    /**
     * European vanilla price under Black-Scholes / Garman-Kohlhagen.
     *
     * <p>Edge cases: {@code expiry == 0} → intrinsic {@code max(phi (S - K), 0)};
     * {@code sigma == 0} → discounted forward intrinsic
     * {@code exp(-r T) max(phi (F - K), 0)}; {@code strike == 0} → call is the
     * dividend-discounted spot {@code S exp(-q T)}, put is worthless.
     *
     * @param spot     spot {@code S0 > 0}
     * @param strike   strike {@code K >= 0}
     * @param rate     discount rate {@code r} ({@code rd} for FX)
     * @param dividend carry rate {@code q} ({@code rf} for FX)
     * @param sigma    Black volatility, {@code >= 0}
     * @param expiry   maturity in years, {@code >= 0}
     * @param isCall   payoff type
     * @return the option price
     */
    public static double bsPrice(double spot, double strike, double rate, double dividend,
                                 double sigma, double expiry, boolean isCall) {
        validateCommon(spot, strike, rate, dividend, sigma, expiry);
        double phi = isCall ? 1.0 : -1.0;
        if (expiry == 0.0) {
            return Math.max(phi * (spot - strike), 0.0);
        }
        double dfR = Math.exp(-rate * expiry);
        double dfQ = Math.exp(-dividend * expiry);
        double forward = spot * Math.exp((rate - dividend) * expiry);
        if (strike == 0.0) {
            return isCall ? spot * dfQ : 0.0;
        }
        if (sigma == 0.0) {
            return dfR * Math.max(phi * (forward - strike), 0.0);
        }
        double st = sigma * Math.sqrt(expiry);
        double d1 = (Math.log(forward / strike) + 0.5 * st * st) / st;
        double d2 = d1 - st;
        return phi * (spot * dfQ * normCdf(phi * d1) - strike * dfR * normCdf(phi * d2));
    }

    private static double d1(double spot, double strike, double rate, double dividend,
                             double sigma, double expiry) {
        double st = sigma * Math.sqrt(expiry);
        double forward = spot * Math.exp((rate - dividend) * expiry);
        return (Math.log(forward / strike) + 0.5 * st * st) / st;
    }

    /**
     * Spot delta {@code phi exp(-q T) N(phi d1)}.
     *
     * @param spot spot; {@code > 0}
     * @param strike strike; {@code > 0}
     * @param rate discount rate
     * @param dividend carry rate
     * @param sigma volatility; {@code > 0}
     * @param expiry maturity; {@code > 0}
     * @param isCall payoff type
     * @return the delta
     */
    public static double bsDelta(double spot, double strike, double rate, double dividend,
                                 double sigma, double expiry, boolean isCall) {
        validateCommon(spot, strike, rate, dividend, sigma, expiry);
        if (expiry <= 0.0 || sigma <= 0.0 || strike <= 0.0) {
            throw new IllegalArgumentException("bsDelta requires expiry > 0, sigma > 0 and strike > 0");
        }
        double phi = isCall ? 1.0 : -1.0;
        double d1 = d1(spot, strike, rate, dividend, sigma, expiry);
        return phi * Math.exp(-dividend * expiry) * normCdf(phi * d1);
    }

    /**
     * Spot gamma {@code exp(-q T) n(d1) / (S sigma sqrt(T))} (call == put).
     *
     * @param spot spot; {@code > 0}
     * @param strike strike; {@code > 0}
     * @param rate discount rate
     * @param dividend carry rate
     * @param sigma volatility; {@code > 0}
     * @param expiry maturity; {@code > 0}
     * @return the gamma
     */
    public static double bsGamma(double spot, double strike, double rate, double dividend,
                                 double sigma, double expiry) {
        validateCommon(spot, strike, rate, dividend, sigma, expiry);
        if (expiry <= 0.0 || sigma <= 0.0 || strike <= 0.0) {
            throw new IllegalArgumentException("bsGamma requires expiry > 0, sigma > 0 and strike > 0");
        }
        double d1 = d1(spot, strike, rate, dividend, sigma, expiry);
        return Math.exp(-dividend * expiry) * normPdf(d1) / (spot * sigma * Math.sqrt(expiry));
    }

    /**
     * Vega {@code S exp(-q T) n(d1) sqrt(T)} per unit of vol (call == put).
     *
     * @param spot spot; {@code > 0}
     * @param strike strike; {@code > 0}
     * @param rate discount rate
     * @param dividend carry rate
     * @param sigma volatility; {@code > 0}
     * @param expiry maturity; {@code > 0}
     * @return the vega
     */
    public static double bsVega(double spot, double strike, double rate, double dividend,
                                double sigma, double expiry) {
        validateCommon(spot, strike, rate, dividend, sigma, expiry);
        if (expiry <= 0.0 || sigma <= 0.0 || strike <= 0.0) {
            throw new IllegalArgumentException("bsVega requires expiry > 0, sigma > 0 and strike > 0");
        }
        double d1 = d1(spot, strike, rate, dividend, sigma, expiry);
        return spot * Math.exp(-dividend * expiry) * normPdf(d1) * Math.sqrt(expiry);
    }

    /**
     * Invert Black-Scholes by bisection on {@code [1e-9, 5.0]} with exactly
     * 100 halvings (the cross-language contract: deterministic ~1e-10 vol
     * accuracy, no vega/convergence corner cases).
     *
     * @param price    observed option price
     * @param spot     spot; {@code > 0}
     * @param strike   strike; {@code > 0}
     * @param rate     discount rate
     * @param dividend carry rate
     * @param expiry   maturity; {@code > 0}
     * @param isCall   payoff type
     * @return the implied Black volatility
     * @throws IllegalArgumentException if the price violates static
     *         no-arbitrage bounds or inputs are invalid
     */
    public static double impliedVol(double price, double spot, double strike, double rate,
                                    double dividend, double expiry, boolean isCall) {
        validateCommon(spot, strike, rate, dividend, 0.0, expiry);
        if (!Double.isFinite(price)) {
            throw new IllegalArgumentException("price must be finite, got " + price);
        }
        if (expiry <= 0.0) {
            throw new IllegalArgumentException("impliedVol requires expiry > 0");
        }
        if (strike <= 0.0) {
            throw new IllegalArgumentException("impliedVol requires strike > 0");
        }
        double lower = bsPrice(spot, strike, rate, dividend, 0.0, expiry, isCall);
        double upper = isCall ? spot * Math.exp(-dividend * expiry)
                              : strike * Math.exp(-rate * expiry);
        double eps = 1e-12 * Math.max(1.0, spot);
        if (price < lower - eps || price > upper + eps) {
            throw new IllegalArgumentException(String.format(
                    "price %s outside no-arbitrage bounds [%.10g, %.10g]", price, lower, upper));
        }
        double a = 1e-9;
        double b = 5.0;
        for (int i = 0; i < 100; i++) {
            double mid = 0.5 * (a + b);
            if (bsPrice(spot, strike, rate, dividend, mid, expiry, isCall) < price) {
                a = mid;
            } else {
                b = mid;
            }
        }
        return 0.5 * (a + b);
    }
}
