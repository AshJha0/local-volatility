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

    private static ImpliedVolSurface bundledSurface() {
        return ImpliedVolSurface.fromCsv(Paths.get("..", "data", "implied_surface.csv"));
    }

    @Test
    public void noCapInsideQuotedBoxAndContinuousAtWings() {
        // CRIT-1: before the stencil clamp vol(0.4995..0.5005, 1) was 5.0.
        ImpliedVolSurface srf = bundledSurface();
        DupireLocalVol lv = new DupireLocalVol(srf);
        for (double t : new double[]{0.1, 0.5, 0.75, 1.0, 2.0, 3.0}) {
            for (int i = 0; i <= 200; i++) {
                double k = -0.5 + 0.005 * i;
                double v = lv.vol(k, t);
                assertTrue("k=" + k + " T=" + t + ": " + v, v > DupireLocalVol.FLOOR && v < DupireLocalVol.CAP);
            }
            for (double edge : new double[]{srf.kMin(), srf.kMax()}) {
                double lo = lv.vol(edge - 1e-4, t);
                double hi = lv.vol(edge + 1e-4, t);
                assertTrue("T=" + t + " k=" + edge + ": " + lo + " vs " + hi, Math.abs(lo - hi) < 1e-3);
                // constant in k beyond kMax - DK (clamped stencil)
                double far = lv.vol(edge + Math.signum(edge), t);
                assertEquals(edge > 0 ? hi : lo, far, 0.0);
            }
        }
        assertEquals(0, lv.floorCount());
        assertEquals(0, lv.capCount());
        assertEquals(0.19703445, lv.vol(0.5, 1.0), 1e-6);
    }

    @Test
    public void termStructureLocalVolIsForwardVariance() {
        // Flat in k, iv = {0.15, 0.20, 0.25} at T = {0.25, 0.5, 1}.
        double[] ks = new double[11];
        for (int i = 0; i < 11; i++) {
            ks[i] = -0.5 + 0.1 * i;
        }
        double[] ts = {0.25, 0.5, 1.0};
        double[][] v = new double[3][11];
        java.util.Arrays.fill(v[0], 0.15);
        java.util.Arrays.fill(v[1], 0.20);
        java.util.Arrays.fill(v[2], 0.25);
        DupireLocalVol lv = new DupireLocalVol(new ImpliedVolSurface(ks, ts, v));
        assertEquals(Math.sqrt(0.0575), lv.vol(0.0, 0.4), 1e-6);   // (0.02 - 0.005625)/0.25
        assertEquals(Math.sqrt(0.0575), lv.vol(0.3, 0.4), 1e-6);
        assertEquals(Math.sqrt(0.085), lv.vol(0.0, 0.75), 1e-6);   // (0.0625 - 0.02)/0.5
        assertEquals(0.15, lv.vol(0.0, 0.1), 1e-6);                // below first pillar
        assertEquals(Math.sqrt(0.085), lv.vol(0.0, 2.0), 1e-6);    // beyond last pillar
        assertEquals(0, lv.floorCount());
        assertEquals(0, lv.capCount());
    }

    @Test
    public void zeroExpiryIsShortTimeLimit() {
        // MAJ-1: before the fix vol(-0.3, 0) = 0.309 vs vol(-0.3, 1e-6) = 0.452.
        ImpliedVolSurface srf = bundledSurface();
        DupireLocalVol lv = new DupireLocalVol(srf);
        for (double k : new double[]{-0.3, -0.1, 0.0, 0.2, 0.45}) {
            double v0 = lv.vol(k, 0.0);
            assertTrue("k=" + k, Math.abs(v0 - lv.vol(k, 1e-6)) < 1e-3);
            assertTrue("k=" + k, Math.abs(v0 - lv.vol(k, 1e-4)) < 1e-3);
            // independent BBF evaluation from the short-end implied slice
            double s = srf.impliedVol(k, 0.0);
            double sp = (srf.impliedVol(k + 1e-3, 0.0) - srf.impliedVol(k - 1e-3, 0.0)) / 2e-3;
            assertEquals("k=" + k, s / (1.0 - k * sp / s), v0, 1e-12);
        }
        assertEquals(0.45240187, lv.vol(-0.3, 0.0), 1e-6);
        assertEquals(0.2, lv.vol(0.0, 0.0), 1e-12);
    }

    private static ImpliedVolSurface stronglyConvexSurface() {
        // At |k| = 0.5, k s'/s = 0.5 * 4 / 1.05 = 1.9 > 1 -> denominator <= 0 -> cap.
        double[] ks = new double[21];
        double[][] v = new double[2][21];
        for (int i = 0; i < 21; i++) {
            ks[i] = -0.5 + 0.05 * i;
            v[0][i] = 0.05 + 4.0 * ks[i] * ks[i];
            v[1][i] = v[0][i] * 1.05;
        }
        return new ImpliedVolSurface(ks, new double[]{0.5, 1.0}, v);
    }

    @Test
    public void zeroExpiryCapsOnStronglyConvexSmile() {
        DupireLocalVol lv = new DupireLocalVol(stronglyConvexSurface());
        for (int i = 0; i < 21; i++) {
            double v = lv.vol(-0.5 + 0.05 * i, 0.0);
            assertTrue(v <= DupireLocalVol.CAP && v >= DupireLocalVol.FLOOR);
        }
        assertTrue(lv.capCount() > 0);
        assertEquals(DupireLocalVol.CAP, lv.vol(0.5, 0.0), 0.0);
        assertEquals(0.05, lv.vol(0.0, 0.0), 1e-9);
    }

    @Test
    public void countersAreThreadSafe() throws InterruptedException {
        // MAJ-3: four threads hammering one shared object over a cap-firing
        // region must produce exactly the single-threaded count.
        DupireLocalVol lv = new DupireLocalVol(stronglyConvexSurface());
        Runnable work = () -> {
            for (int i = 0; i < 100000; i++) {
                lv.vol(-0.5 + (i % 101) * 0.01, 0.75);
            }
        };
        work.run();
        long singleFloor = lv.floorCount();
        long singleCap = lv.capCount();
        assertTrue(singleCap > 0);
        lv.resetCounters();
        Thread[] pool = new Thread[4];
        for (int t = 0; t < 4; t++) {
            pool[t] = new Thread(work);
            pool[t].start();
        }
        for (Thread th : pool) {
            th.join();
        }
        assertEquals(4 * singleFloor, lv.floorCount());
        assertEquals(4 * singleCap, lv.capCount());
    }
}
