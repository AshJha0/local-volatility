package com.quant.localvol;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.nio.file.Paths;
import org.junit.Test;

/** Implied surface: node exactness, continuity, extrapolation, arbitrage checks. */
public class SurfaceTest {

    private static ImpliedVolSurface tinySurface() {
        double[] k = {-0.4, -0.2, 0.0, 0.2, 0.4};
        double[] t = {0.5, 1.0, 2.0};
        double[][] v = new double[t.length][k.length];
        for (int j = 0; j < t.length; j++) {
            for (int i = 0; i < k.length; i++) {
                v[j][i] = 0.2 + 0.05 * k[i] * k[i] - 0.02 * k[i]; // smile with skew
            }
        }
        return new ImpliedVolSurface(k, t, v);
    }

    @Test
    public void splineExactAtNodesAndContinuous() {
        double[] x = {0.0, 1.0, 2.5, 3.0, 4.7};
        double[] y = {1.0, -2.0, 0.5, 3.0, -1.0};
        CubicSpline1D s = new CubicSpline1D(x, y);
        for (int i = 0; i < x.length; i++) {
            assertEquals("node " + i, y[i], s.value(x[i]), 1e-14);
        }
        // continuity across each interior node
        for (int i = 1; i < x.length - 1; i++) {
            double left = s.value(x[i] - 1e-9);
            double right = s.value(x[i] + 1e-9);
            assertEquals("continuity at node " + i, left, right, 1e-7);
        }
        // flat extrapolation beyond the ends
        assertEquals(y[0], s.value(-5.0), 1e-14);
        assertEquals(y[4], s.value(99.0), 1e-14);
    }

    @Test
    public void splineDegenerateSizes() {
        CubicSpline1D constant = new CubicSpline1D(new double[]{1.0}, new double[]{7.0});
        assertEquals(7.0, constant.value(-3.0), 0.0);
        CubicSpline1D linear = new CubicSpline1D(new double[]{0.0, 2.0}, new double[]{0.0, 4.0});
        assertEquals(2.0, linear.value(1.0), 1e-14);
    }

    @Test
    public void totalVarianceExactAtNodes() {
        ImpliedVolSurface s = tinySurface();
        double iv = 0.2 + 0.05 * 0.04 - 0.02 * 0.2; // at k = 0.2
        assertEquals(iv * iv * 1.0, s.totalVariance(0.2, 1.0), 1e-14);
        assertEquals(0.0, s.totalVariance(0.1, 0.0), 0.0); // w(k, 0) = 0 exactly
    }

    @Test
    public void wingExtrapolationIsFlat() {
        ImpliedVolSurface s = tinySurface();
        assertEquals(s.totalVariance(0.4, 1.0), s.totalVariance(2.0, 1.0), 1e-14);
        assertEquals(s.totalVariance(-0.4, 1.0), s.totalVariance(-9.0, 1.0), 1e-14);
    }

    @Test
    public void shortEndIsFlatForwardVariance() {
        ImpliedVolSurface s = tinySurface();
        double w1 = s.totalVariance(0.0, 0.5);
        assertEquals(w1 * (0.25 / 0.5), s.totalVariance(0.0, 0.25), 1e-14);
        // implied vol constant below the first pillar
        assertEquals(s.impliedVol(0.0, 0.5), s.impliedVol(0.0, 0.1), 1e-12);
        assertEquals(s.impliedVol(0.0, 0.5), s.impliedVol(0.0, 0.0), 1e-12);
    }

    @Test
    public void longEndExtrapolationLinearInTotalVariance() {
        ImpliedVolSurface s = tinySurface();
        double w2 = s.totalVariance(0.0, 2.0);
        double w1 = s.totalVariance(0.0, 1.0);
        double slope = (w2 - w1) / 1.0;
        assertEquals(w2 + slope * 1.0, s.totalVariance(0.0, 3.0), 1e-13);
    }

    @Test
    public void interiorTIsLinearInTotalVariance() {
        ImpliedVolSurface s = tinySurface();
        double wA = s.totalVariance(0.1, 1.0);
        double wB = s.totalVariance(0.1, 2.0);
        assertEquals(0.5 * (wA + wB), s.totalVariance(0.1, 1.5), 1e-13);
        // monotone in T (calendar-consistent input): property-style loop
        double prev = 0.0;
        for (double t = 0.1; t <= 3.0; t += 0.1) {
            double w = s.totalVariance(0.05, t);
            assertTrue("w nondecreasing in T at t=" + t, w >= prev - 1e-12);
            prev = w;
        }
    }

    @Test
    public void calendarArbitrageDetectedNotRaised() {
        double[] k = {-0.1, 0.0, 0.1};
        double[] t = {0.5, 1.0};
        // second expiry has much lower vol -> total variance decreases
        double[][] v = {{0.3, 0.3, 0.3}, {0.1, 0.1, 0.1}};
        ImpliedVolSurface s = new ImpliedVolSurface(k, t, v);
        assertEquals(3, s.calendarViolations());
    }

    @Test
    public void singleExpiryFlagAndProportionalRule() {
        double[] k = {-0.1, 0.0, 0.1};
        double[] t = {1.0};
        double[][] v = {{0.2, 0.2, 0.2}};
        ImpliedVolSurface s = new ImpliedVolSurface(k, t, v);
        assertTrue(s.singleExpiry());
        assertEquals(0.04, s.totalVariance(0.0, 1.0), 1e-14);
        assertEquals(0.08, s.totalVariance(0.0, 2.0), 1e-14); // flat-forward on both sides
    }

    @Test
    public void loadsBundledCsvAndIsRectangular() {
        ImpliedVolSurface s = ImpliedVolSurface.fromCsv(
                Paths.get("..", "data", "implied_surface.csv"));
        assertEquals(4, s.numExpiries());
        assertEquals(15, s.numKNodes());
        assertEquals(0, s.calendarViolations());
        assertEquals(-0.5, s.kMin(), 1e-12);
        assertEquals(0.5, s.kMax(), 1e-12);
    }

    @Test
    public void validation() {
        double[] k = {-0.1, 0.0, 0.1};
        double[] t = {0.5, 1.0};
        double[][] good = {{0.2, 0.2, 0.2}, {0.2, 0.2, 0.2}};
        assertThrows(IllegalArgumentException.class,
                () -> new ImpliedVolSurface(new double[0], t, good));
        assertThrows(IllegalArgumentException.class,
                () -> new ImpliedVolSurface(new double[]{0.1, 0.0, -0.1}, t, good)); // decreasing k
        assertThrows(IllegalArgumentException.class,
                () -> new ImpliedVolSurface(k, new double[]{-0.5, 1.0}, good)); // T <= 0
        assertThrows(IllegalArgumentException.class,
                () -> new ImpliedVolSurface(k, t, new double[][]{{0.2, 0.2, 0.2}, {0.2, -0.2, 0.2}}));
        assertThrows(IllegalArgumentException.class,
                () -> new ImpliedVolSurface(k, t, new double[][]{{0.2, 0.2}, {0.2, 0.2}})); // shape
        ImpliedVolSurface s = tinySurface();
        assertThrows(IllegalArgumentException.class, () -> s.totalVariance(Double.NaN, 1.0));
        assertThrows(IllegalArgumentException.class, () -> s.totalVariance(0.0, -1.0));
    }
}
