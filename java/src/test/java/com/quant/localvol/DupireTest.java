package com.quant.localvol;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.nio.file.Paths;
import org.junit.Test;

/** Dupire local vol: flat-surface identity, clamps, counters, validation. */
public class DupireTest {

    private static ImpliedVolSurface flatSurface() {
        return ImpliedVolSurface.fromCsv(Paths.get("..", "data", "flat_surface.csv"));
    }

    @Test
    public void flatSurfaceLocalVolEqualsImpliedEverywhere() {
        DupireLocalVol lv = new DupireLocalVol(flatSurface());
        // property-style grid loop: sigma_loc == 0.20 to 1e-6 at every (k, T)
        for (double k = -0.4; k <= 0.4001; k += 0.1) {
            for (double t : new double[]{0.1, 0.25, 0.5, 1.0, 1.7, 2.0, 3.0}) {
                assertEquals("flat local vol at k=" + k + " T=" + t,
                        0.20, lv.vol(k, t), 1e-6);
            }
        }
        assertEquals(0, lv.floorCount());
        assertEquals(0, lv.capCount());
    }

    @Test
    public void bundledSurfaceInteriorVolsAreSane() {
        ImpliedVolSurface s = ImpliedVolSurface.fromCsv(
                Paths.get("..", "data", "implied_surface.csv"));
        DupireLocalVol lv = new DupireLocalVol(s);
        // skewed smile: local vol higher on the put wing than the call wing
        double putWing = lv.vol(-0.2, 1.0);
        double callWing = lv.vol(0.2, 1.0);
        assertTrue("skew: put wing " + putWing + " > call wing " + callWing,
                putWing > callWing);
        for (double k = -0.3; k <= 0.3001; k += 0.05) {
            double v = lv.vol(k, 1.0);
            assertTrue("interior local vol in a sane band, got " + v,
                    v > 0.05 && v < 1.0);
        }
    }

    @Test
    public void floorEngagesOnCalendarArbSurfaceAndIsCounted() {
        // total variance decreasing in T -> dw/dT <= 0 -> floored at 1%
        double[] k = {-0.1, 0.0, 0.1};
        double[] t = {0.5, 1.0};
        double[][] v = {{0.3, 0.3, 0.3}, {0.1, 0.1, 0.1}};
        ImpliedVolSurface s = new ImpliedVolSurface(k, t, v);
        DupireLocalVol lv = new DupireLocalVol(s);
        assertEquals(DupireLocalVol.FLOOR, lv.vol(0.0, 0.75), 0.0);
        assertTrue(lv.floorCount() > 0);
        lv.resetCounters();
        assertEquals(0, lv.floorCount());
        assertEquals(0, lv.capCount());
    }

    @Test
    public void capEngagesOnExtremeSkew() {
        // absurdly steep smile in k -> denominator <= 0 in the wing -> capped
        double[] k = {-0.2, -0.1, 0.0, 0.1, 0.2};
        double[] t = {0.5, 1.0};
        double[][] v = new double[2][5];
        for (int j = 0; j < 2; j++) {
            for (int i = 0; i < 5; i++) {
                v[j][i] = 0.15 + 2.5 * Math.abs(k[i]); // ~250%/unit skew
            }
        }
        ImpliedVolSurface s = new ImpliedVolSurface(k, t, v);
        DupireLocalVol lv = new DupireLocalVol(s);
        for (double kk = -0.19; kk <= 0.19; kk += 0.01) {
            lv.vol(kk, 0.75);
        }
        assertTrue("cap counter engaged on extreme skew", lv.capCount() > 0);
    }

    @Test
    public void validation() {
        assertThrows(IllegalArgumentException.class, () -> new DupireLocalVol(null));
        DupireLocalVol lv = new DupireLocalVol(flatSurface());
        assertThrows(IllegalArgumentException.class, () -> lv.vol(Double.NaN, 1.0));
        assertThrows(IllegalArgumentException.class, () -> lv.vol(0.0, -0.5));
        assertThrows(IllegalArgumentException.class,
                () -> lv.vol(0.0, Double.POSITIVE_INFINITY));
    }
}
