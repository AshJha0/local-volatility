package com.quant.localvol;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.nio.file.Paths;
import org.junit.Test;

/** PDE pricer: BS agreement, Rannacher damping, convergence order, American. */
public class PdeTest {

    private static DupireLocalVol bundledLocalVol() {
        return new DupireLocalVol(ImpliedVolSurface.fromCsv(
                Paths.get("..", "data", "implied_surface.csv")));
    }

    @Test
    public void flatVolMatchesBlackScholes() {
        double[][] cases = {
                // s, k, r, q, sigma, t, call(1/0)
                {100.0, 100.0, 0.05, 0.02, 0.20, 1.0, 1},   // equity ATM call, q > 0
                {100.0, 80.0, 0.05, 0.00, 0.20, 1.0, 1},    // ITM call
                {100.0, 120.0, 0.05, 0.00, 0.30, 0.5, 0},   // ITM put
                {1.10, 1.05, 0.03, 0.01, 0.10, 0.5, 0},     // FX put (GK rd/rf)
                {100.0, 110.0, -0.01, 0.00, 0.25, 2.0, 1},  // negative rate
                {100.0, 100.0, 0.00, 0.03, 0.15, 0.25, 1},  // short-dated, q > r
        };
        for (double[] c : cases) {
            boolean call = c[6] != 0;
            double exact = BlackScholes.bsPrice(c[0], c[1], c[2], c[3], c[4], c[5], call);
            double approx = Pde.priceEuropean(new Market(c[0], c[2], c[3]), c[1], c[5], c[4],
                    call, 200, 200, 6.0);
            assertEquals("PDE vs BS " + java.util.Arrays.toString(c),
                    exact, approx, 1e-3 * Math.abs(exact));
        }
    }

    @Test
    public void fxGarmanKohlhagenViaFactory() {
        Market fx = Market.fx(1.25, 0.045, 0.02);
        double exact = BlackScholes.bsPrice(1.25, 1.30, 0.045, 0.02, 0.12, 1.0, true);
        double approx = Pde.priceEuropean(fx, 1.30, 1.0, 0.12, true, 200, 200, 6.0);
        assertEquals(exact, approx, 1e-3 * exact);
    }

    @Test
    public void deepItmOtmAndDegenerate() {
        Market mkt = new Market(100.0, 0.02, 0.0);
        double otm = Pde.priceEuropean(mkt, 300.0, 0.5, 0.2, true);
        assertTrue("deep OTM tiny but nonnegative, got " + otm, otm >= 0.0 && otm < 1e-3);
        double itm = Pde.priceEuropean(mkt, 10.0, 0.5, 0.2, true);
        double itmRef = BlackScholes.bsPrice(100, 10, 0.02, 0.0, 0.2, 0.5, true);
        assertEquals(itmRef, itm, 1e-3 * itmRef);
        // T = 0 -> intrinsic; sigma = 0 -> discounted forward intrinsic; K = 0
        assertEquals(10.0, Pde.priceEuropean(mkt, 90.0, 0.0, 0.2, true), 0.0);
        assertEquals(BlackScholes.bsPrice(100, 90, 0.02, 0.0, 0.0, 1.0, true),
                Pde.priceEuropean(mkt, 90.0, 1.0, 0.0, true), 1e-12);
        assertEquals(100.0, Pde.priceEuropean(mkt, 0.0, 1.0, 0.2, true), 1e-12);
        assertEquals(0.0, Pde.priceEuropean(mkt, 0.0, 1.0, 0.2, false), 0.0);
    }

    @Test
    public void putCallParityOnGrid() {
        Market mkt = new Market(100.0, 0.03, 0.01);
        double c = Pde.priceEuropean(mkt, 105.0, 1.0, 0.2, true, 200, 100, 6.0);
        double p = Pde.priceEuropean(mkt, 105.0, 1.0, 0.2, false, 200, 100, 6.0);
        double rhs = 100 * Math.exp(-0.01) - 105 * Math.exp(-0.03);
        assertEquals(rhs, c - p, 2e-2);
    }

    @Test
    public void rannacherDampingNoGammaOscillation() {
        // Discrete gamma near the strike must stay essentially nonnegative:
        // plain CN leaves an oscillating gamma at the payoff kink; the
        // Rannacher start damps it.
        Market mkt = new Market(100.0, 0.05, 0.0);
        Pde.Result res = Pde.europeanGrid(mkt, 100.0, 0.25, 0.2, true, 200, 50, 6.0);
        double[] x = res.x();
        double[] v = res.values();
        double maxCurv = Double.NEGATIVE_INFINITY;
        double minCurv = Double.POSITIVE_INFINITY;
        for (int i = 1; i < v.length - 1; i++) {
            double s = Math.exp(x[i]);
            if (s > 70.0 && s < 140.0) {
                double curv = v[i + 1] - 2.0 * v[i] + v[i - 1];
                maxCurv = Math.max(maxCurv, curv);
                minCurv = Math.min(minCurv, curv);
            }
        }
        assertTrue("peak curvature positive", maxCurv > 0.0);
        assertTrue("most negative curvature " + minCurv + " bounded by peak " + maxCurv,
                minCurv > -1e-4 * maxCurv);
    }

    @Test
    public void gridConvergenceSecondOrder() {
        // Halving dx and dt twice: observed order must land in [1.5, 2.5].
        Market mkt = new Market(100.0, 0.05, 0.0);
        double exact = BlackScholes.bsPrice(100, 100, 0.05, 0.0, 0.2, 1.0, true);
        double[] errs = new double[3];
        int[] ms = {50, 100, 200};
        for (int i = 0; i < 3; i++) {
            double approx = Pde.priceEuropean(mkt, 100.0, 1.0, 0.2, true, ms[i], ms[i], 6.0);
            errs[i] = Math.abs(approx - exact);
        }
        double order1 = Math.log(errs[0] / errs[1]) / Math.log(2.0);
        double order2 = Math.log(errs[1] / errs[2]) / Math.log(2.0);
        double avg = 0.5 * (order1 + order2);
        assertTrue("observed order " + avg + " in [1.5, 2.5]", avg >= 1.5 && avg <= 2.5);
    }

    @Test
    public void localVolPdeWithCallable() {
        // Local-vol PDE ~ BS at the surface's own ATM implied vol (small err).
        DupireLocalVol lv = bundledLocalVol();
        Market mkt = new Market(100.0, 0.0, 0.0);
        double atmVol = lv.surface().impliedVol(0.0, 1.0);
        double price = Pde.priceEuropean(mkt, 100.0, 1.0, lv, atmVol, true, 200, 200, 6.0);
        double ref = BlackScholes.bsPrice(100, 100, 0.0, 0.0, atmVol, 1.0, true);
        assertEquals(ref, price, 5e-3 * ref); // ~1bp vol agreement
    }

    @Test
    public void americanPutFlat() {
        Market mkt = new Market(100.0, 0.05, 0.0);
        double eur = Pde.priceEuropean(mkt, 100.0, 1.0, 0.2, false, 150, 100, 6.0);
        double ame = Pde.priceAmericanPut(mkt, 100.0, 1.0, 0.2, 150, 100, 6.0,
                Pde.DEFAULT_OMEGA, Pde.DEFAULT_TOL, Pde.DEFAULT_MAX_ITER);
        // early-exercise premium nonnegative; American >= intrinsic
        assertTrue("American >= European", ame >= eur - 1e-10);
        assertTrue("American >= intrinsic", ame >= 0.0);
        // with q = 0 and r > 0 the premium is strictly positive for a put
        assertTrue("premium strictly positive, got " + (ame - eur), ame - eur > 0.01);
    }

    @Test
    public void americanZeroRateEqualsEuropean() {
        // r = 0, q = 0: American put is never exercised early -> equals European.
        Market mkt = new Market(100.0, 0.0, 0.0);
        double eur = Pde.priceEuropean(mkt, 110.0, 1.0, 0.25, false, 150, 100, 6.0);
        double ame = Pde.priceAmericanPut(mkt, 110.0, 1.0, 0.25, 150, 100, 6.0,
                Pde.DEFAULT_OMEGA, Pde.DEFAULT_TOL, Pde.DEFAULT_MAX_ITER);
        assertEquals(eur, ame, 2e-3 * eur);
    }

    @Test
    public void americanLocalVolAboveEuropean() {
        DupireLocalVol lv = bundledLocalVol();
        Market mkt = new Market(100.0, 0.04, 0.0);
        double eur = Pde.priceEuropean(mkt, 105.0, 1.0, lv, 0.2, false, 150, 100, 6.0);
        double ame = Pde.priceAmericanPut(mkt, 105.0, 1.0, lv, 0.2, 150, 100, 6.0,
                Pde.DEFAULT_OMEGA, Pde.DEFAULT_TOL, Pde.DEFAULT_MAX_ITER);
        assertTrue("local-vol American >= European", ame >= eur - 1e-10);
    }

    @Test
    public void sigmaZeroAmericanIsDeterministicOptimum() {
        // r > 0, sigma = 0: exercising immediately is optimal for an ITM put.
        Market mkt = new Market(80.0, 0.05, 0.0);
        assertEquals(20.0, Pde.priceAmericanPut(mkt, 100.0, 1.0, 0.0), 1e-6);
    }

    @Test
    public void validation() {
        Market mkt = new Market(100.0, 0.05, 0.0);
        assertThrows(IllegalArgumentException.class, // odd num_space
                () -> Pde.priceEuropean(mkt, 100.0, 1.0, 0.2, true, 201, 200, 6.0));
        assertThrows(IllegalArgumentException.class, // num_space too small
                () -> Pde.priceEuropean(mkt, 100.0, 1.0, 0.2, true, 2, 200, 6.0));
        assertThrows(IllegalArgumentException.class, // num_time < 1
                () -> Pde.priceEuropean(mkt, 100.0, 1.0, 0.2, true, 200, 0, 6.0));
        assertThrows(IllegalArgumentException.class, // negative strike
                () -> Pde.priceEuropean(mkt, -5.0, 1.0, 0.2, true));
        assertThrows(IllegalArgumentException.class, // negative expiry
                () -> Pde.priceEuropean(mkt, 100.0, -1.0, 0.2, true));
        assertThrows(IllegalArgumentException.class, // negative flat vol
                () -> Pde.priceEuropean(mkt, 100.0, 1.0, -0.2, true));
        assertThrows(IllegalArgumentException.class, // bad nsd
                () -> Pde.priceEuropean(mkt, 100.0, 1.0, 0.2, true, 200, 200, 0.0));
        assertThrows(IllegalArgumentException.class, // omega outside (0, 2)
                () -> Pde.priceAmericanPut(mkt, 100.0, 1.0, 0.2, 200, 200, 6.0, 2.5, 1e-8, 100));
        assertThrows(IllegalArgumentException.class, // American needs K > 0
                () -> Pde.priceAmericanPut(mkt, 0.0, 1.0, 0.2));
        assertThrows(IllegalArgumentException.class, // null market
                () -> Pde.priceEuropean(null, 100.0, 1.0, 0.2, true));
        assertThrows(IllegalArgumentException.class, // null vol function
                () -> Pde.priceEuropean(mkt, 100.0, 1.0, (LocalVolFn) null, 0.2, true));
    }
}
