package com.quant.localvol;

/**
 * A local-volatility function {@code sigma(k, t)} as consumed by the PDE and
 * Monte Carlo engines.
 *
 * <p>The {@code k} argument is always <b>forward</b> log-moneyness
 * {@code ln(level / F(t))} — the same coordinates the Dupire surface is
 * built in, so PDE and MC discretise the identical diffusion.
 */
@FunctionalInterface
public interface LocalVolFn {

    /**
     * Local volatility at forward log-moneyness {@code k} and calendar time
     * {@code t}.
     *
     * @param k forward log-moneyness {@code ln(level / F(t))}
     * @param t calendar time in years
     * @return the local volatility (positive)
     */
    double vol(double k, double t);
}
