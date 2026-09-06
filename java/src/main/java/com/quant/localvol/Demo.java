package com.quant.localvol;

import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * localvol demo: Dupire round-trip consistency check.
 *
 * <p>Pipeline: bundled SSVI implied surface → Dupire local vol → reprice
 * European vanillas by PDE (and spot-check by Monte Carlo) → invert back to
 * implied vols → report the error against the input surface in basis points.
 *
 * <p>Run: {@code bash demo.sh} (optionally pass the data directory, default
 * {@code ../data} relative to the working directory).
 */
public final class Demo {

    private static final double INTERIOR_TARGET_BP = 30.0;

    private Demo() {
    }

    /**
     * Entry point.
     *
     * @param args optional: data directory (default {@code ../data})
     */
    public static void main(String[] args) {
        Path data = Paths.get(args.length > 0 ? args[0] : "../data");
        String rule = "=".repeat(72);
        System.out.println(rule);
        System.out.println("localvol demo - Dupire local volatility round-trip consistency (Java)");
        System.out.println(rule);

        ImpliedVolSurface surface = ImpliedVolSurface.fromCsv(data.resolve("implied_surface.csv"));
        DupireLocalVol lv = new DupireLocalVol(surface);
        Market mkt = new Market(100.0, 0.02, 0.01); // equity-style carry
        System.out.printf("surface: %d expiries x %d strikes, calendar violations: %d%n",
                surface.numExpiries(), surface.numKNodes(), surface.calendarViolations());
        System.out.printf("market : S0=%.2f  r=%.2f%%  q=%.2f%%%n%n",
                mkt.spot(), mkt.rate() * 100.0, mkt.dividend() * 100.0);

        // ---- round trip: implied in -> local vol -> PDE price -> implied out
        double[] expiries = {0.5, 1.0, 1.5, 2.0};
        double[] strikes = {80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0};
        System.out.println("Round-trip implied-vol error (PDE 200x200), basis points:");
        StringBuilder head = new StringBuilder("  T\\K   ");
        for (double k : strikes) {
            head.append(String.format("%8.0f", k));
        }
        System.out.println(head);
        double maxErrBp = 0.0;
        double maxAtK = 0.0;
        double maxAtT = 0.0;
        lv.resetCounters(); // once: the clamp report below covers the whole sweep
        for (double t : expiries) {
            StringBuilder row = new StringBuilder(String.format("  %-5.2f", t));
            double sigmaRef = surface.impliedVol(0.0, t);
            for (double strike : strikes) {
                double k = Math.log(strike / mkt.forward(t));
                double ivIn = surface.impliedVol(k, t);
                double price = Pde.priceEuropean(mkt, strike, t, lv, sigmaRef, true, 200, 200, 6.0);
                double ivOut = BlackScholes.impliedVol(price, mkt.spot(), strike,
                        mkt.rate(), mkt.dividend(), t, true);
                double errBp = Math.abs(ivOut - ivIn) * 1e4;
                row.append(String.format("%8.1f", errBp));
                boolean interior = Math.abs(k) <= 0.30; // wings excluded per spec
                if (interior && errBp > maxErrBp) {
                    maxErrBp = errBp;
                    maxAtK = strike;
                    maxAtT = t;
                }
            }
            System.out.println(row);
        }
        System.out.printf("%nmax interior |vol error|: %.2f bp at K=%.0f, T=%.2f  "
                        + "(target < %.0f bp, |k| <= 0.30)%n",
                maxErrBp, maxAtK, maxAtT, INTERIOR_TARGET_BP);
        System.out.println("round-trip check: " + (maxErrBp < INTERIOR_TARGET_BP ? "PASS" : "FAIL"));
        System.out.println(lv.violationReport());

        // ---- PDE vs MC cross-check at the ATM pillar
        double t1 = 1.0;
        double strike1 = 100.0;
        double sigmaRef1 = surface.impliedVol(0.0, t1);
        double pde = Pde.priceEuropean(mkt, strike1, t1, lv, sigmaRef1, true, 200, 200, 6.0);
        MonteCarlo.McResult mc = MonteCarlo.priceEuropean(
                mkt, strike1, t1, lv, true, 20000, 100, 42L, true);
        System.out.printf("%nPDE vs MC (K=100, T=1): PDE=%.4f  MC=%.4f +/- %.4f  |diff|=%.2f SE (%s)%n",
                pde, mc.price(), mc.stderr(), Math.abs(pde - mc.price()) / mc.stderr(),
                mc.within(pde, 3.0) ? "OK" : "OUTSIDE 3 SE");

        // ---- American and barrier flavours under the same local vol
        double amer = Pde.priceAmericanPut(mkt, 100.0, 1.0, lv, sigmaRef1);
        double eurPut = Pde.priceEuropean(mkt, 100.0, 1.0, lv, sigmaRef1, false, 200, 200, 6.0);
        MonteCarlo.McResult uo = MonteCarlo.priceUpOutCall(
                mkt, 100.0, 130.0, 1.0, lv, 20000, 100, 42L, true, true);
        System.out.printf("American put (PSOR)     : %.4f  (European %.4f, premium %+.4f)%n",
                amer, eurPut, amer - eurPut);
        System.out.printf("Up-and-out call B=130 MC: %.4f +/- %.4f (vanilla %.4f; "
                        + "barrier <= vanilla: %s)%n",
                uo.price(), uo.stderr(), pde, uo.price() <= pde ? "OK" : "VIOLATED");

        System.out.println("\ndone.");
    }
}
