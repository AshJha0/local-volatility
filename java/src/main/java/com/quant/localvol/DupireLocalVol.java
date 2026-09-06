package com.quant.localvol;

import java.util.concurrent.atomic.LongAdder;

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
 * {@code 0 < T <= DT} the time derivative switches to a forward difference.
 * These exact step sizes are part of the cross-language contract.
 *
 * <p><b>Stencil clamp at the quoted wings</b>: the surface is flat in
 * {@code k} beyond the last quoted strikes, so {@code w} is only C^0 there (a
 * natural spline has {@code w'' = 0} but {@code w' != 0} at the end node). A
 * central stencil straddling that kink would read {@code d2w/dk2 ~ -w'/DK}
 * and cap the vol at 500% <i>at the quoted wing</i>. The query is therefore
 * clamped into {@code [kMin + DK, kMax - DK]} before differencing (whenever
 * the box is wider than {@code 2 DK}) and the clamped {@code k} is used in
 * every term: local vol is constant in {@code k} beyond {@code kMax - DK} and
 * continuous across the wing node.
 *
 * <p><b>Short-expiry limit</b>: {@code w(k, 0) = 0} exactly, so
 * {@code vol(k, 0)} is defined as the {@code T -> 0+} limit (Berestycki,
 * Busca, Florent 2002) with {@code s(k) = impliedVol(k, 0)}:
 * {@code sigma_loc(k, 0) = s / (1 - k s'/s)}, {@code s'} by the same central
 * {@code DK} step. The {@code T > 0} branch converges to it (flat forward
 * variance below the first pillar), so {@code vol} is continuous at
 * {@code T = 0}.
 *
 * <p><b>Thread safety</b>: {@link #vol} / {@link #localVariance} are safe to
 * call concurrently on one shared instance (the surface is immutable; the
 * counters are {@link LongAdder}s, so their total is exact while a read taken
 * mid-computation is only a snapshot). Call {@link #resetCounters()} between
 * phases, not during one.
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
    private final LongAdder floorCount = new LongAdder();
    private final LongAdder capCount = new LongAdder();

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
        floorCount.reset();
        capCount.reset();
    }

    /** @return cumulative number of floor clamps since the last reset */
    public long floorCount() {
        return floorCount.sum();
    }

    /** @return cumulative number of cap clamps since the last reset */
    public long capCount() {
        return capCount.sum();
    }

    /**
     * Shared clamp policy: {@code num <= 0} floors, else {@code den <= 0}
     * caps, else {@code sqrt(num/den)} clipped into {@code [FLOOR, CAP]};
     * every hit counted. Returns the clamped local <i>variance</i>.
     */
    private double clampAndCount(double num, double den) {
        double vol;
        if (num <= 0.0) {
            vol = FLOOR;      // no forward variance -> floor
            floorCount.increment();
        } else if (den <= 0.0) {
            vol = CAP;        // butterfly arbitrage -> cap
            capCount.increment();
        } else {
            vol = Math.sqrt(num / den);
            if (vol < FLOOR) {
                floorCount.increment();
                vol = FLOOR;
            } else if (vol > CAP) {
                capCount.increment();
                vol = CAP;
            }
        }
        return vol * vol;
    }

    /** Berestycki-Busca-Florent {@code T -> 0+} limit at an already-clamped k. */
    private double shortTimeVariance(double k) {
        double s = surface.impliedVol(k, 0.0);
        double sUp = surface.impliedVol(k + DK, 0.0);
        double sDn = surface.impliedVol(k - DK, 0.0);
        double dsdk = (sUp - sDn) / (2.0 * DK);
        double ss = Math.max(s, W_EPS); // guard s == 0 (spline overshoot to w <= 0)
        double denom = 1.0 - k * dsdk / ss;
        // Same clamp policy as T > 0 with num = s^2 and den = denom |denom|:
        // sqrt(num/den) = s/denom when denom > 0, s == 0 floors, denom <= 0 caps.
        return clampAndCount(s * s, denom * Math.abs(denom));
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
        // Clamp into [kMin + DK, kMax - DK] so that no central stencil
        // straddles the C^0 kink of the flat wing extrapolation.
        double kMin = surface.kMin();
        double kMax = surface.kMax();
        if (kMax - kMin > 2.0 * DK) {
            k = Math.min(Math.max(k, kMin + DK), kMax - DK);
        }
        if (expiry == 0.0) {
            return shortTimeVariance(k);
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

        return clampAndCount(dwdt, denom);
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
                FLOOR * 100.0, floorCount(), CAP * 100.0, capCount());
    }
}
