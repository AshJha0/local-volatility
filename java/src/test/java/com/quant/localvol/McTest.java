package com.quant.localvol;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.nio.file.Paths;
import org.junit.Test;

/** Monte Carlo: statistical agreement with BS/PDE, barrier ordering, validation. */
public class McTest {

    @Test
    public void flatVolMatchesBlackScholesWithinFourSe() {
        Market mkt = new Market(100.0, 0.03, 0.01);
        double exact = BlackScholes.bsPrice(100, 105, 0.03, 0.01, 0.2, 1.0, true);
        MonteCarlo.McResult res = MonteCarlo.priceEuropean(
                mkt, 105.0, 1.0, 0.2, true, 20000, 100, 42L, true);
        assertTrue("MC stderr positive", res.stderr() > 0.0);
        assertTrue("MC within 4 SE of BS: " + res.price() + " vs " + exact
                + " (SE " + res.stderr() + ")", res.within(exact, 4.0));
    }

    @Test
    public void putFlatVolWithinFourSe() {
        Market mkt = new Market(100.0, 0.02, 0.0);
        double exact = BlackScholes.bsPrice(100, 95, 0.02, 0.0, 0.25, 0.5, false);
        MonteCarlo.McResult res = MonteCarlo.priceEuropean(
                mkt, 95.0, 0.5, 0.25, false, 20000, 50, 7L, true);
        assertTrue("put MC within 4 SE", res.within(exact, 4.0));
    }

    @Test
    public void antitheticReducesStandardError() {
        Market mkt = new Market(100.0, 0.03, 0.0);
        MonteCarlo.McResult anti = MonteCarlo.priceEuropean(
                mkt, 100.0, 1.0, 0.2, true, 10000, 50, 11L, true);
        MonteCarlo.McResult raw = MonteCarlo.priceEuropean(
                mkt, 100.0, 1.0, 0.2, true, 10000, 50, 11L, false);
        assertTrue("antithetic SE " + anti.stderr() + " < raw SE " + raw.stderr(),
                anti.stderr() < raw.stderr());
    }

    @Test
    public void localVolMcMatchesPdeWithinThreeSe() {
        ImpliedVolSurface s = ImpliedVolSurface.fromCsv(
                Paths.get("..", "data", "implied_surface.csv"));
        DupireLocalVol lv = new DupireLocalVol(s);
        Market mkt = new Market(100.0, 0.0, 0.0);
        double pde = Pde.priceEuropean(mkt, 100.0, 1.0, lv, 0.2, true, 200, 200, 6.0);
        MonteCarlo.McResult mc = MonteCarlo.priceEuropean(
                mkt, 100.0, 1.0, lv, true, 20000, 100, 42L, true);
        assertTrue("local-vol MC " + mc.price() + " within 3 SE of PDE " + pde,
                mc.within(pde, 3.0));
    }

    @Test
    public void barrierBelowVanillaAndBridgeBelowDiscrete() {
        Market mkt = new Market(100.0, 0.02, 0.0);
        MonteCarlo.McResult vanilla = MonteCarlo.priceEuropean(
                mkt, 100.0, 1.0, 0.2, true, 20000, 200, 42L, true);
        MonteCarlo.McResult discrete = MonteCarlo.priceUpOutCall(
                mkt, 100.0, 130.0, 1.0, 0.2, 20000, 200, 42L, true, false);
        MonteCarlo.McResult bridged = MonteCarlo.priceUpOutCall(
                mkt, 100.0, 130.0, 1.0, 0.2, 20000, 200, 42L, true, true);
        assertTrue("bridged " + bridged.price() + " <= discrete " + discrete.price(),
                bridged.price() <= discrete.price());
        assertTrue("discrete barrier <= vanilla", discrete.price() <= vanilla.price());
        assertTrue("bridged barrier <= vanilla", bridged.price() <= vanilla.price());
        assertTrue("barrier price positive", bridged.price() > 0.0);
    }

    @Test
    public void bornKnockedOutIsWorthless() {
        Market mkt = new Market(140.0, 0.02, 0.0);
        MonteCarlo.McResult res = MonteCarlo.priceUpOutCall(
                mkt, 100.0, 130.0, 1.0, 0.2, 2000, 10, 1L, true, true);
        assertEquals(0.0, res.price(), 0.0);
        assertEquals(0.0, res.stderr(), 0.0);
    }

    @Test
    public void validation() {
        Market mkt = new Market(100.0, 0.02, 0.0);
        assertThrows(IllegalArgumentException.class, // odd paths with antithetic
                () -> MonteCarlo.priceEuropean(mkt, 100.0, 1.0, 0.2, true, 10001, 10, 1L, true));
        assertThrows(IllegalArgumentException.class, // n_paths < 2
                () -> MonteCarlo.priceEuropean(mkt, 100.0, 1.0, 0.2, true, 1, 10, 1L, false));
        assertThrows(IllegalArgumentException.class, // n_steps < 1
                () -> MonteCarlo.priceEuropean(mkt, 100.0, 1.0, 0.2, true, 100, 0, 1L, true));
        assertThrows(IllegalArgumentException.class, // expiry <= 0
                () -> MonteCarlo.priceEuropean(mkt, 100.0, 0.0, 0.2, true, 100, 10, 1L, true));
        assertThrows(IllegalArgumentException.class, // flat vol <= 0 for MC
                () -> MonteCarlo.priceEuropean(mkt, 100.0, 1.0, 0.0, true, 100, 10, 1L, true));
        assertThrows(IllegalArgumentException.class, // bad barrier
                () -> MonteCarlo.priceUpOutCall(mkt, 100.0, -130.0, 1.0, 0.2,
                        100, 10, 1L, true, true));
        assertThrows(IllegalArgumentException.class, // negative strike
                () -> MonteCarlo.priceEuropean(mkt, -100.0, 1.0, 0.2, true, 100, 10, 1L, true));
    }

    private static DupireLocalVol bundledLocalVol() {
        return new DupireLocalVol(ImpliedVolSurface.fromCsv(
                Paths.get("..", "data", "implied_surface.csv")));
    }

    @Test
    public void localVolMcMatchesPdeWithCarry() {
        // MAJ-7: with r != q the lookup k = X_n - ln F(t_n) is exercised
        // (a ln S0 lookup lands 4-5 SE away from the PDE).
        DupireLocalVol lv = bundledLocalVol();
        Market[] markets = {new Market(100.0, 0.05, 0.02), Market.fx(1.10, 0.03, -0.01)};
        for (Market mkt : markets) {
            double strike = mkt.spot();
            double pde = Pde.priceEuropean(mkt, strike, 1.0, lv, lv.surface().impliedVol(0.0, 1.0),
                    true, 200, 200, 6.0);
            MonteCarlo.McResult res = MonteCarlo.priceEuropean(
                    mkt, strike, 1.0, lv, true, 20000, 100, 987L, true);
            assertTrue("S0=" + mkt.spot() + ": MC " + res.price() + " +/- " + res.stderr()
                    + " vs PDE " + pde, res.within(pde, 3.0));
        }
    }

    @Test
    public void putCallParityUnderLocalVol() {
        DupireLocalVol lv = bundledLocalVol();
        Market mkt = new Market(100.0, 0.03, 0.01);
        MonteCarlo.McResult c = MonteCarlo.priceEuropean(mkt, 100.0, 1.0, lv, true, 20000, 100, 5L, true);
        MonteCarlo.McResult p = MonteCarlo.priceEuropean(mkt, 100.0, 1.0, lv, false, 20000, 100, 5L, true);
        double parity = 100.0 * Math.exp(-0.01) - 100.0 * Math.exp(-0.03);
        assertTrue("parity", Math.abs(c.price() - p.price() - parity)
                < 3.0 * Math.sqrt(c.stderr() * c.stderr() + p.stderr() * p.stderr()));
    }

    @Test
    public void rejectsTinyPathCounts() {
        // MAJ-8: n_paths = 2 with antithetic used to return stderr = NaN.
        Market mkt = new Market(100.0, 0.02, 0.0);
        assertThrows(IllegalArgumentException.class,
                () -> MonteCarlo.priceEuropean(mkt, 100.0, 1.0, 0.2, true, 2, 5, 1L, true));
        assertThrows(IllegalArgumentException.class,
                () -> MonteCarlo.priceEuropean(mkt, 100.0, 1.0, 0.2, true, 1, 5, 1L, false));
        assertThrows(IllegalArgumentException.class,
                () -> MonteCarlo.priceUpOutCall(mkt, 100.0, 130.0, 1.0, 0.2, 2, 5, 1L, true, true));
        // deep ITM strike so every path pays and the sample spread is non-zero
        MonteCarlo.McResult four = MonteCarlo.priceEuropean(mkt, 50.0, 1.0, 0.2, true, 4, 5, 1L, true);
        assertTrue(Double.isFinite(four.price()) && Double.isFinite(four.stderr()) && four.stderr() > 0.0);
        MonteCarlo.McResult two = MonteCarlo.priceEuropean(mkt, 50.0, 1.0, 0.2, true, 2, 5, 1L, false);
        assertTrue(Double.isFinite(two.stderr()) && two.stderr() > 0.0);
    }

    @Test
    public void rejectsNonFiniteOrNegativeVolFunction() {
        // MAJ-4: a bad function used to yield price = stderr = NaN.
        Market mkt = new Market(100.0, 0.02, 0.0);
        LocalVolFn nan = (k, t) -> Double.NaN;
        LocalVolFn neg = (k, t) -> -0.2;
        LocalVolFn inf = (k, t) -> Double.POSITIVE_INFINITY;
        assertThrows(IllegalArgumentException.class,
                () -> MonteCarlo.priceEuropean(mkt, 100.0, 1.0, nan, true, 100, 5, 1L, true));
        assertThrows(IllegalArgumentException.class,
                () -> MonteCarlo.priceEuropean(mkt, 100.0, 1.0, neg, true, 100, 5, 1L, true));
        assertThrows(IllegalArgumentException.class,
                () -> MonteCarlo.priceEuropean(mkt, 100.0, 1.0, inf, true, 100, 5, 1L, true));
        assertThrows(IllegalArgumentException.class,
                () -> MonteCarlo.priceUpOutCall(mkt, 100.0, 130.0, 1.0, nan, 100, 5, 1L, true, true));
        assertThrows(IllegalArgumentException.class,
                () -> MonteCarlo.priceEuropean(mkt, 100.0, 1.0, (LocalVolFn) null, true, 100, 5, 1L, true));
        // sigma == 0 from a function is allowed: every path is the forward.
        Market m2 = new Market(100.0, 0.03, 0.01);
        LocalVolFn zero = (k, t) -> 0.0;
        MonteCarlo.McResult res = MonteCarlo.priceEuropean(m2, 90.0, 1.0, zero, true, 8, 4, 1L, true);
        assertEquals(Math.exp(-0.03) * (100.0 * Math.exp(0.02) - 90.0), res.price(), 1e-12);
        assertEquals(0.0, res.stderr(), 0.0);
        MonteCarlo.McResult uo = MonteCarlo.priceUpOutCall(m2, 90.0, 130.0, 1.0, zero, 8, 4, 1L, true, true);
        assertEquals(res.price(), uo.price(), 1e-12); // never crosses: no bridge weight
        // a constant function equals the flat-vol path exactly (same draws)
        MonteCarlo.McResult flat = MonteCarlo.priceEuropean(mkt, 100.0, 1.0, 0.2, true, 2000, 10, 3L, true);
        MonteCarlo.McResult fn = MonteCarlo.priceEuropean(mkt, 100.0, 1.0, (k, t) -> 0.2, true, 2000, 10, 3L, true);
        assertEquals(flat.price(), fn.price(), 0.0);
        assertEquals(flat.stderr(), fn.stderr(), 0.0);
    }

    @Test
    public void mcIsDeterministicPerSeed() {
        // T-14: same seed twice -> identical price/stderr; different seed differs.
        Market mkt = new Market(100.0, 0.02, 0.0);
        MonteCarlo.McResult a = MonteCarlo.priceEuropean(mkt, 100.0, 1.0, 0.2, true, 2000, 20, 11L, true);
        MonteCarlo.McResult b = MonteCarlo.priceEuropean(mkt, 100.0, 1.0, 0.2, true, 2000, 20, 11L, true);
        MonteCarlo.McResult c = MonteCarlo.priceEuropean(mkt, 100.0, 1.0, 0.2, true, 2000, 20, 12L, true);
        assertEquals(a.price(), b.price(), 0.0);
        assertEquals(a.stderr(), b.stderr(), 0.0);
        assertTrue(a.price() != c.price());
        assertEquals("L64X128MixRandom", MonteCarlo.RNG_ALGORITHM);
    }
}
