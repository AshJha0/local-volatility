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

    @Test
    public void psorNonConvergenceWarnsAndReturnsFinite() {
        // max_iter = 1 cannot converge: a stderr warning per time step and a
        // finite, non-negative price (report, don't crash).
        Market mkt = new Market(100.0, 0.05, 0.0);
        java.io.PrintStream old = System.err;
        java.io.ByteArrayOutputStream buf = new java.io.ByteArrayOutputStream();
        double price;
        try {
            System.setErr(new java.io.PrintStream(buf, true));
            price = Pde.priceAmericanPut(mkt, 100.0, 1.0, 0.2, 100, 20, 6.0,
                    Pde.DEFAULT_OMEGA, Pde.DEFAULT_TOL, 1);
        } finally {
            System.setErr(old);
        }
        assertTrue("warning emitted", buf.toString().contains("PSOR did not converge"));
        assertTrue(Double.isFinite(price) && price >= 0.0);
        assertEquals(6.4776, price, 1e-3);
    }

    @Test
    public void americanPutDeepItmEqualsIntrinsic() {
        double price = Pde.priceAmericanPut(new Market(50.0, 0.05, 0.0), 100.0, 1.0, 0.2, 100, 50, 6.0,
                Pde.DEFAULT_OMEGA, Pde.DEFAULT_TOL, Pde.DEFAULT_MAX_ITER);
        assertEquals(50.0, price, 1e-8);
    }

    @Test
    public void americanPutNegativeRateEqualsEuropean() {
        // r < 0, q = 0: early exercise is never optimal for a put.
        Market mkt = new Market(100.0, -0.02, 0.0);
        double eur = Pde.priceEuropean(mkt, 100.0, 1.0, 0.2, false, 100, 50, 6.0);
        double ame = Pde.priceAmericanPut(mkt, 100.0, 1.0, 0.2, 100, 50, 6.0,
                Pde.DEFAULT_OMEGA, Pde.DEFAULT_TOL, Pde.DEFAULT_MAX_ITER);
        assertEquals(eur, ame, 1e-6);
        assertTrue(ame >= eur - 1e-12);
    }

    @Test
    public void localVolGridConvergence() {
        // Observed order vs a 400x400 reference in [0.8, 2.5] under local vol.
        DupireLocalVol lv = bundledLocalVol();
        Market mkt = new Market(100.0, 0.0, 0.0);
        double ref = Pde.priceEuropean(mkt, 100.0, 1.0, lv, 0.2, true, 400, 400, 6.0);
        int[] ms = {50, 100, 200};
        double[] errs = new double[3];
        for (int i = 0; i < 3; i++) {
            errs[i] = Math.abs(Pde.priceEuropean(mkt, 100.0, 1.0, lv, 0.2, true, ms[i], ms[i], 6.0) - ref);
        }
        for (int i = 0; i < 2; i++) {
            double order = Math.log(errs[i] / errs[i + 1]) / Math.log(2.0);
            assertTrue("order " + order + " errors " + java.util.Arrays.toString(errs),
                    order >= 0.8 && order <= 2.5);
        }
    }

    @Test
    public void pecletViolationSwitchesToUpwindAndStaysMonotone() {
        // MAJ-5: 1% vol with 10% carry violates |mu| h <= 2a on the default
        // grid. Central differencing yields a negative put price (-2.95e-6)
        // and grid values down to -0.084; upwinding keeps everything >= 0
        // and monotone.
        Market mkt = new Market(100.0, 0.10, 0.0);
        Pde.Result put = Pde.europeanGrid(mkt, 90.0, 1.0, 0.01, false, 100, 50, 6.0);
        assertTrue("put " + put.price(), put.price() >= 0.0 && put.price() < 1e-6); // BS ~1e-95
        double[] v = put.values();
        for (int i = 0; i < v.length; i++) {
            assertTrue("node " + i, v[i] >= 0.0);
            if (i > 0) {
                assertTrue("node " + i, v[i] - v[i - 1] <= 1e-14); // nonincreasing in S
            }
        }
        Pde.Result call = Pde.europeanGrid(mkt, 90.0, 1.0, 0.01, true, 100, 50, 6.0);
        double[] c = call.values();
        for (int i = 1; i < c.length; i++) {
            assertTrue("node " + i, c[i] - c[i - 1] >= -1e-14);
        }
        double bs = BlackScholes.bsPrice(100.0, 90.0, 0.10, 0.0, 0.01, 1.0, true);
        assertEquals(bs, call.price(), 2e-3 * bs);
        double amer = Pde.priceAmericanPut(mkt, 90.0, 1.0, 0.01, 100, 50, 6.0,
                Pde.DEFAULT_OMEGA, Pde.DEFAULT_TOL, Pde.DEFAULT_MAX_ITER);
        assertTrue("American " + amer, amer >= 0.0 && amer < 1e-6);
    }

    @Test
    public void rejectsBadVolFunctionsAndNanTol() {
        // MAJ-4 / MIN-12
        Market mkt = new Market(100.0, 0.05, 0.0);
        LocalVolFn nan = (k, t) -> Double.NaN;
        LocalVolFn neg = (k, t) -> -0.2;
        LocalVolFn inf = (k, t) -> Double.POSITIVE_INFINITY;
        assertThrows(IllegalArgumentException.class,
                () -> Pde.priceEuropean(mkt, 100.0, 1.0, nan, 0.2, true, 100, 50, 6.0));
        assertThrows(IllegalArgumentException.class,
                () -> Pde.priceEuropean(mkt, 100.0, 1.0, neg, 0.2, true, 100, 50, 6.0));
        assertThrows(IllegalArgumentException.class,
                () -> Pde.priceAmericanPut(mkt, 100.0, 1.0, inf, 0.2, 100, 50, 6.0,
                        Pde.DEFAULT_OMEGA, Pde.DEFAULT_TOL, Pde.DEFAULT_MAX_ITER));
        assertThrows(IllegalArgumentException.class, // default sigmaRef from a NaN function
                () -> Pde.priceEuropean(mkt, 100.0, 1.0, nan, Double.NaN, true, 100, 50, 6.0));
        assertThrows(IllegalArgumentException.class,
                () -> Pde.priceEuropean(mkt, 100.0, 1.0, (k, t) -> 0.2, -1.0, true, 100, 50, 6.0));
        // a constant function equals the flat-vol path exactly
        assertEquals(Pde.priceEuropean(mkt, 100.0, 1.0, 0.2, true, 100, 50, 6.0),
                Pde.priceEuropean(mkt, 100.0, 1.0, (k, t) -> 0.2, 0.2, true, 100, 50, 6.0), 0.0);
        assertThrows(IllegalArgumentException.class,
                () -> Pde.priceAmericanPut(mkt, 100.0, 1.0, 0.2, 100, 50, 6.0, 1.5, Double.NaN, 100));
        assertThrows(IllegalArgumentException.class,
                () -> Pde.priceAmericanPut(mkt, 100.0, 1.0, 0.2, 100, 50, 6.0, 1.5, Double.POSITIVE_INFINITY, 100));
        assertThrows(IllegalArgumentException.class,
                () -> Pde.priceAmericanPut(mkt, 100.0, 1.0, 0.2, 100, 50, 6.0, 1.5, 1e-8, 0));
    }

    @Test
    public void pdeIsDeterministic() {
        Market mkt = new Market(100.0, 0.03, 0.01);
        assertEquals(Pde.priceEuropean(mkt, 105.0, 0.7, 0.23, true, 120, 60, 6.0),
                Pde.priceEuropean(mkt, 105.0, 0.7, 0.23, true, 120, 60, 6.0), 0.0);
    }
}
