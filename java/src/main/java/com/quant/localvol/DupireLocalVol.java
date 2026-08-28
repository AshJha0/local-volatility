package com.quant.localvol;

/**
 * Dupire local volatility in total-variance (Gatheral) form.
 *
 * <p>Substituting the Black-Scholes representation of call prices in terms of
 * total implied variance {@code w(k, T)} (with {@code k = ln(K / F(T))} the
 * forward log-moneyness) into Dupire's formula yields Gatheral's form, far
 * better conditioned numerically because it differentiates the smooth surface
 * {@code w} instead of near-degenerate call prices:
 *
 * <pre>
 * sigma_loc^2(k, T) = (dw/dT) /
 *     ( 1 - (k/w) dw/dk + 1/4 (-1/4 - 1/w + k^2/w^2) (dw/dk)^2 + 1/2 d2w/dk2 )
 * </pre>
 *
 * The numerator {@code dw/dT} is the forward variance; positivity is calendar
 * no-arbitrage. The denominator is (up to a positive factor) the probability
 * density term; positivity is butterfly no-arbitrage. A flat surface
 * collapses the formula to {@code sigma_loc == implied vol} identically — the
 * sanity anchor used by the golden tests.
 *
 * <p><b>Numerics</b>: derivatives are central finite differences with
 * <b>fixed steps</b> {@code DK = 1e-3} and {@code DT = 1e-4}; at
 * {@code T <= DT} the time derivative switches to a forward difference. These
 * exact step sizes are part of the cross-language contract.
 *
 * <p><b>Robustness</b>: the local vol is clamped to {@code [1%, 500%]}.
 * Numerator {@code <= 0} (calendar arbitrage) floors; denominator
 * {@code <= 0} (butterfly arbitrage in over-steep wings) caps. Every clamp is
 * counted in {@link #floorCount()} / {@link #capCount()} so callers can
 * report surface quality instead of crashing mid-pricing.
 */
public final class DupireLocalVol implements LocalVolFn {

    /** Central-difference step in log-moneyness. */
    public static final double DK = 1.0e-3;
    /** Central-difference step in expiry (years). */
    public static final double DT = 1.0e-4;
    /** 1% local-vol floor. */
    public static final double FLOOR = 0.01;
    /** 500% local-vol cap. */
    public static final double CAP = 5.00;

    private static final double W_EPS = 1e-12;

    private final ImpliedVolSurface surface;
    private long floorCount;
    private long capCount;

    /**
     * @param surface the implied surface to differentiate; must not be null
     */
    public DupireLocalVol(ImpliedVolSurface surface) {
        if (surface == null) {
            throw new IllegalArgumentException("DupireLocalVol requires an ImpliedVolSurface");
        }
        this.surface = surface;
    }

    /** @return the underlying implied surface */
    public ImpliedVolSurface surface() {
        return surface;
    }

    /** Zero the floor/cap violation counters. */
    public void resetCounters() {
        floorCount = 0;
        capCount = 0;
    }

    /** @return cumulative number of floor clamps since the last reset */
    public long floorCount() {
        return floorCount;
    }

    /** @return cumulative number of cap clamps since the last reset */
    public long capCount() {
        return capCount;
    }

    /**
     * Clamped local variance {@code sigma_loc^2(k, T)}.
     *
     * @param k      forward log-moneyness
     * @param expiry maturity in years, finite and {@code >= 0}
     * @return the clamped local variance
     */
    public double localVariance(double k, double expiry) {
        if (!Double.isFinite(expiry) || expiry < 0.0) {
            throw new IllegalArgumentException("expiry must be finite and >= 0, got " + expiry);
        }
        if (!Double.isFinite(k)) {
            throw new IllegalArgumentException("k must be finite");
        }
        double w = surface.totalVariance(k, expiry);
        double wUp = surface.totalVariance(k + DK, expiry);
        double wDn = surface.totalVariance(k - DK, expiry);
        double dwdk = (wUp - wDn) / (2.0 * DK);
        double d2wdk2 = (wUp - 2.0 * w + wDn) / (DK * DK);
        double dwdt;
        if (expiry > DT) {
            double wtUp = surface.totalVariance(k, expiry + DT);
            double wtDn = surface.totalVariance(k, expiry - DT);
            dwdt = (wtUp - wtDn) / (2.0 * DT);
        } else {
            double wtUp = surface.totalVariance(k, expiry + DT);
            dwdt = (wtUp - w) / DT;
        }

        double ws = Math.max(w, W_EPS); // guard 1/w and k/w at (near-)zero variance
        double denom = 1.0
                - (k / ws) * dwdk
                + 0.25 * (-0.25 - 1.0 / ws + (k * k) / (ws * ws)) * dwdk * dwdk
                + 0.5 * d2wdk2;

        double vol;
        if (dwdt <= 0.0) {
            vol = FLOOR;      // no forward variance -> floor
            floorCount++;
        } else if (denom <= 0.0) {
            vol = CAP;        // butterfly-arb wing -> cap
            capCount++;
        } else {
            vol = Math.sqrt(dwdt / denom);
            if (vol < FLOOR) {
                floorCount++;
                vol = FLOOR;
            } else if (vol > CAP) {
                capCount++;
                vol = CAP;
            }
        }
        return vol * vol;
    }

    /**
     * Local volatility {@code sigma_loc(k, T)}, clamped to {@code [1%, 500%]}.
     *
     * @param k forward log-moneyness {@code ln(level / F(t))}
     * @param t maturity/time in years, finite and {@code >= 0}
     * @return the clamped local volatility
     */
    @Override
    public double vol(double k, double t) {
        return Math.sqrt(localVariance(k, t));
    }

    /** @return human-readable clamp summary for demos/logs */
    public String violationReport() {
        return String.format("local-vol clamps: floor(%.0f%%) hit %dx, cap(%.0f%%) hit %dx",
                FLOOR * 100.0, floorCount, CAP * 100.0, capCount);
    }
}
