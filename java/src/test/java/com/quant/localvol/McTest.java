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
}
