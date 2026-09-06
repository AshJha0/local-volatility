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

    private static java.nio.file.Path writeTempCsv(String name, String body) throws java.io.IOException {
        java.nio.file.Path p = java.nio.file.Files.createTempFile("localvol_" + name, ".csv");
        p.toFile().deleteOnExit();
        java.nio.file.Files.writeString(p, body);
        return p;
    }

    private static final String BASE_CSV =
            "T,k,iv\n0.5,-0.1,0.22\n0.5,0.0,0.2\n0.5,0.1,0.21\n1.0,-0.1,0.23\n1.0,0.0,0.21\n1.0,0.1,0.22\n";

    @Test
    public void csvRejectsDuplicateMissingAndShortRows() throws java.io.IOException {
        // MIN-4 / MIN-5: duplicates and gaps are named; short/extra/non-numeric
        // rows are IllegalArgumentException, never ArrayIndexOutOfBounds.
        ImpliedVolSurface good = ImpliedVolSurface.fromCsv(writeTempCsv("good", BASE_CSV));
        assertEquals(2, good.numExpiries());
        assertEquals(3, good.numKNodes());
        IllegalArgumentException dup = assertThrows(IllegalArgumentException.class,
                () -> ImpliedVolSurface.fromCsv(writeTempCsv("dup", BASE_CSV + "1.0,0.0,0.25\n")));
        assertTrue(dup.getMessage(), dup.getMessage().contains("duplicate"));
        String missing = BASE_CSV.substring(0, BASE_CSV.lastIndexOf("1.0,0.1"));
        IllegalArgumentException gap = assertThrows(IllegalArgumentException.class,
                () -> ImpliedVolSurface.fromCsv(writeTempCsv("missing", missing)));
        assertTrue(gap.getMessage(), gap.getMessage().contains("rectangular"));
        String[][] bad = {
                {"short", "T,k,iv\n0.5,-0.1,0.22\n0.5,0.0\n"},
                {"extra", "T,k,iv,x\n0.5,-0.1,0.22,1\n"},
                {"long", "T,k,iv\n0.5,-0.1,0.22,1\n"},
                {"nonnum", "T,k,iv\n0.5,-0.1,abc\n"},
                {"nonfinite", "T,k,iv\n0.5,-0.1,NaN\n"},
                {"empty", "T,k,iv\n"},
                {"negiv", "T,k,iv\n0.5,0.0,-0.2\n"},
        };
        for (String[] c : bad) {
            java.nio.file.Path p = writeTempCsv(c[0], c[1]);
            assertThrows(c[0], IllegalArgumentException.class, () -> ImpliedVolSurface.fromCsv(p));
        }
        assertThrows(IllegalArgumentException.class,
                () -> ImpliedVolSurface.fromCsv(Paths.get("..", "data", "does_not_exist.csv")));
        // Blank lines, spaces and CRLF endings are tolerated.
        StringBuilder crlf = new StringBuilder("T,k,iv\r\n");
        for (String line : BASE_CSV.split("\n")) {
            if (!line.startsWith("T,")) {
                crlf.append(" ").append(line).append(" \r\n");
            }
        }
        crlf.append("\r\n");
        assertEquals(3, ImpliedVolSurface.fromCsv(writeTempCsv("crlf", crlf.toString())).numKNodes());
    }

    @Test
    public void negativeTotalVarianceOvershootIsCounted() {
        // MIN-15: a ragged slice whose spline dips to w <= 0 between nodes is
        // reported via negativeWCount(); clean surfaces report 0.
        ImpliedVolSurface clean = ImpliedVolSurface.fromCsv(Paths.get("..", "data", "implied_surface.csv"));
        assertEquals(0, clean.negativeWCount());
        double[] ks = {-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3};
        double[] row = {1.0, 0.001, 0.001, 1.0, 0.001, 0.001, 1.0};
        ImpliedVolSurface ragged = new ImpliedVolSurface(ks, new double[]{0.5, 1.0}, new double[][]{row, row});
        assertTrue(ragged.negativeWCount() > 0);
        double minW = 1.0;
        double minIv = 1.0;
        for (int i = 0; i <= 600; i++) {
            double k = -0.3 + 0.001 * i;
            minW = Math.min(minW, ragged.totalVariance(k, 1.0));
            minIv = Math.min(minIv, ragged.impliedVol(k, 1.0));
        }
        assertTrue(minW < 0.0);
        assertEquals(0.0, minIv, 0.0); // reads 0% there rather than throwing
    }

    @Test
    public void marketLogForwardValidationAndValue() {
        // MIN-6: logForward validates expiry in every port.
        Market mkt = new Market(100.0, 0.03, 0.01);
        assertEquals(Math.log(100.0) + 0.02 * 2.0, mkt.logForward(2.0), 1e-15);
        assertEquals(Math.log(100.0), mkt.logForward(0.0), 0.0);
        assertThrows(IllegalArgumentException.class, () -> mkt.logForward(-1.0));
        assertThrows(IllegalArgumentException.class, () -> mkt.logForward(Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> mkt.forward(Double.POSITIVE_INFINITY));
        assertThrows(IllegalArgumentException.class, () -> new Market(100.0, Double.NaN, 0.0));
        assertThrows(IllegalArgumentException.class, () -> new Market(100.0, 0.0, Double.POSITIVE_INFINITY));
        assertThrows(IllegalArgumentException.class, () -> new Market(0.0, 0.0, 0.0));
    }
}
