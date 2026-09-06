package com.quant.localvol;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.nio.file.Paths;
import org.junit.Test;

/**
 * Round-trip consistency: implied surface -> Dupire -> PDE -> implied vols.
 * The full sweep lives in the demo; here the demo's 4 x 7 grid is asserted:
 * max interior error under 5 bp (the wing-clamped Dupire stencil brought the
 * worst cell from 16.4 bp to 2.7 bp) and no clamp fires on any PDE grid.
 */
public class RoundTripTest {

    private static ImpliedVolSurface bundledSurface() {
        return ImpliedVolSurface.fromCsv(Paths.get("..", "data", "implied_surface.csv"));
    }

    @Test
    public void interiorSubsetUnder5bp() {
        ImpliedVolSurface srf = bundledSurface();
        DupireLocalVol lv = new DupireLocalVol(srf);
        Market mkt = new Market(100.0, 0.02, 0.01); // equity carry exercises the forward logic
        for (double expiry : new double[]{0.5, 1.0}) {
            double sigmaRef = srf.impliedVol(0.0, expiry);
            for (double strike : new double[]{90.0, 100.0, 110.0}) {
                double k = Math.log(strike / mkt.forward(expiry));
                double ivIn = srf.impliedVol(k, expiry);
                double price = Pde.priceEuropean(mkt, strike, expiry, lv, sigmaRef, true, 200, 200, 6.0);
                double ivOut = BlackScholes.impliedVol(price, mkt.spot(), strike, mkt.rate(),
                        mkt.dividend(), expiry, true);
                double errBp = Math.abs(ivOut - ivIn) * 1e4;
                assertTrue("K=" + strike + " T=" + expiry + ": " + errBp + " bp", errBp < 30.0);
                assertTrue("K=" + strike + " T=" + expiry + ": " + errBp + " bp (post-CRIT-1 budget)",
                        errBp < 5.0);
            }
        }
    }

    @Test
    public void fullDemoGridUnder5bpAndNoClamps() {
        ImpliedVolSurface srf = bundledSurface();
        DupireLocalVol lv = new DupireLocalVol(srf);
        Market mkt = new Market(100.0, 0.02, 0.01);
        double worst = 0.0;
        for (double expiry : new double[]{0.5, 1.0, 1.5, 2.0}) {
            double sigmaRef = srf.impliedVol(0.0, expiry);
            for (double strike : new double[]{80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0}) {
                double k = Math.log(strike / mkt.forward(expiry));
                double ivIn = srf.impliedVol(k, expiry);
                double price = Pde.priceEuropean(mkt, strike, expiry, lv, sigmaRef, true, 200, 200, 6.0);
                double ivOut = BlackScholes.impliedVol(price, mkt.spot(), strike, mkt.rate(),
                        mkt.dividend(), expiry, true);
                if (Math.abs(k) <= 0.30) {
                    worst = Math.max(worst, Math.abs(ivOut - ivIn) * 1e4);
                }
            }
        }
        assertTrue("max interior error " + worst + " bp", worst < 5.0);
        assertEquals(2.73, worst, 0.05); // K=80, T=2 (documented value)
        assertEquals(0, lv.floorCount());
        assertEquals(0, lv.capCount());
    }
}
