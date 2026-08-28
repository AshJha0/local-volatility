package com.quant.localvol;

/**
 * Market description shared by all pricers.
 *
 * <p>A single record covers both asset classes:
 * <ul>
 *   <li><b>Equity</b>: {@code rate} is the risk-free rate {@code r} and
 *       {@code dividend} is the continuous dividend yield {@code q}.</li>
 *   <li><b>FX (Garman-Kohlhagen)</b>: {@code rate} is the domestic rate
 *       {@code rd} and {@code dividend} plays the role of the foreign rate
 *       {@code rf}. Under Garman-Kohlhagen the foreign rate enters every
 *       formula exactly where the dividend yield enters the equity formula,
 *       so one parametrisation serves both; use {@link #fx} to make the
 *       intent explicit at call sites.</li>
 * </ul>
 * The forward is {@code F(T) = S * exp((rate - dividend) * T)} in both cases.
 *
 * @param spot     spot price {@code S0} (equity price or FX rate); strictly positive
 * @param rate     continuously-compounded discount rate {@code r} ({@code rd} for FX); may be negative
 * @param dividend continuous dividend yield {@code q} ({@code rf} for FX); may be negative
 */
public record Market(double spot, double rate, double dividend) {

    /** Validates finiteness of every field and positivity of the spot. */
    public Market {
        if (!Double.isFinite(spot)) {
            throw new IllegalArgumentException("Market.spot must be finite, got " + spot);
        }
        if (!Double.isFinite(rate)) {
            throw new IllegalArgumentException("Market.rate must be finite, got " + rate);
        }
        if (!Double.isFinite(dividend)) {
            throw new IllegalArgumentException("Market.dividend must be finite, got " + dividend);
        }
        if (spot <= 0.0) {
            throw new IllegalArgumentException("Market.spot must be > 0, got " + spot);
        }
    }

    /**
     * Garman-Kohlhagen market: domestic rate {@code rd}, foreign rate {@code rf}.
     *
     * @param spot FX spot (domestic units per foreign unit)
     * @param rd   domestic rate
     * @param rf   foreign rate
     * @return the equivalent {@code Market}
     */
    public static Market fx(double spot, double rd, double rf) {
        return new Market(spot, rd, rf);
    }

    /**
     * Forward {@code F(T) = S0 * exp((r - q) T)}.
     *
     * @param expiry maturity in years, finite and {@code >= 0}
     * @return the forward level
     */
    public double forward(double expiry) {
        if (!Double.isFinite(expiry) || expiry < 0.0) {
            throw new IllegalArgumentException("expiry must be finite and >= 0, got " + expiry);
        }
        return spot * Math.exp((rate - dividend) * expiry);
    }

    /**
     * {@code ln F(T)} — convenient for log-moneyness lookups.
     *
     * @param expiry maturity in years, finite and {@code >= 0}
     * @return the log-forward
     */
    public double logForward(double expiry) {
        if (!Double.isFinite(expiry) || expiry < 0.0) {
            throw new IllegalArgumentException("expiry must be finite and >= 0, got " + expiry);
        }
        return Math.log(spot) + (rate - dividend) * expiry;
    }
}
