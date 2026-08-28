package com.quant.localvol;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

import org.junit.Test;

/** Black-Scholes analytics: values, parity, FD Greeks, inversion, edge cases. */
public class BlackScholesTest {

    @Test
    public void normCdfReferenceValues() {
        // reference values accurate to ~1e-16 (standard tables / scipy)
        assertEquals(0.5, BlackScholes.normCdf(0.0), 1e-15);
        assertEquals(0.8413447460685429, BlackScholes.normCdf(1.0), 1e-13);
        assertEquals(0.15865525393145707, BlackScholes.normCdf(-1.0), 1e-13);
        assertEquals(0.9772498680518208, BlackScholes.normCdf(2.0), 1e-13);
        assertEquals(0.9986501019683699, BlackScholes.normCdf(3.0), 1e-13);
        // far tail relative accuracy (erfc-based, no cancellation)
        assertEquals(9.865876450376946e-10, BlackScholes.normCdf(-6.0), 1e-19);
    }

    @Test
    public void atmCallReferenceValue() {
        // classic S=K=100, r=5%, q=0, sigma=20%, T=1
        assertEquals(10.450583572185565,
                BlackScholes.bsPrice(100, 100, 0.05, 0.0, 0.2, 1.0, true), 1e-9);
    }

    @Test
    public void putCallParityOnGrid() {
        double s = 100.0;
        double r = 0.03;
        double q = 0.015;
        for (double strike : new double[]{60, 80, 100, 120, 150}) {
            for (double t : new double[]{0.1, 0.5, 1.0, 2.0}) {
                for (double sigma : new double[]{0.1, 0.2, 0.5}) {
                    double c = BlackScholes.bsPrice(s, strike, r, q, sigma, t, true);
                    double p = BlackScholes.bsPrice(s, strike, r, q, sigma, t, false);
                    double rhs = s * Math.exp(-q * t) - strike * Math.exp(-r * t);
                    assertEquals("parity K=" + strike + " T=" + t + " sig=" + sigma,
                            rhs, c - p, 1e-10);
                }
            }
        }
    }

    @Test
    public void greeksMatchFiniteDifferences() {
        double s = 100.0;
        double k = 95.0;
        double r = 0.04;
        double q = 0.01;
        double sigma = 0.25;
        double t = 0.75;
        double ds = 1e-4;
        double dsig = 1e-5;
        double up = BlackScholes.bsPrice(s + ds, k, r, q, sigma, t, true);
        double dn = BlackScholes.bsPrice(s - ds, k, r, q, sigma, t, true);
        double mid = BlackScholes.bsPrice(s, k, r, q, sigma, t, true);
        assertEquals((up - dn) / (2 * ds),
                BlackScholes.bsDelta(s, k, r, q, sigma, t, true), 1e-6);
        assertEquals((up - 2 * mid + dn) / (ds * ds),
                BlackScholes.bsGamma(s, k, r, q, sigma, t), 1e-4);
        double vUp = BlackScholes.bsPrice(s, k, r, q, sigma + dsig, t, true);
        double vDn = BlackScholes.bsPrice(s, k, r, q, sigma - dsig, t, true);
        assertEquals((vUp - vDn) / (2 * dsig),
                BlackScholes.bsVega(s, k, r, q, sigma, t), 1e-5);
    }

    @Test
    public void edgeCases() {
        // T = 0 -> intrinsic
        assertEquals(10.0, BlackScholes.bsPrice(110, 100, 0.05, 0.0, 0.2, 0.0, true), 0.0);
        assertEquals(0.0, BlackScholes.bsPrice(110, 100, 0.05, 0.0, 0.2, 0.0, false), 0.0);
        // sigma = 0 -> discounted forward intrinsic
        double f = 100 * Math.exp(0.05);
        assertEquals(Math.exp(-0.05) * (f - 90),
                BlackScholes.bsPrice(100, 90, 0.05, 0.0, 0.0, 1.0, true), 1e-12);
        // K = 0 -> call = S e^{-qT}, put = 0
        assertEquals(100 * Math.exp(-0.02),
                BlackScholes.bsPrice(100, 0.0, 0.05, 0.02, 0.2, 1.0, true), 1e-12);
        assertEquals(0.0, BlackScholes.bsPrice(100, 0.0, 0.05, 0.02, 0.2, 1.0, false), 0.0);
        // deep ITM / OTM monotone in strike (property loop)
        double prev = Double.POSITIVE_INFINITY;
        for (double strike = 10; strike <= 300; strike += 10) {
            double c = BlackScholes.bsPrice(100, strike, 0.02, 0.0, 0.2, 1.0, true);
            org.junit.Assert.assertTrue("call decreasing in K at " + strike, c <= prev + 1e-12);
            prev = c;
        }
    }

    @Test
    public void impliedVolRoundTrip() {
        for (double sigma : new double[]{0.05, 0.2, 0.8}) {
            for (double strike : new double[]{80.0, 100.0, 125.0}) {
                double price = BlackScholes.bsPrice(100, strike, 0.03, 0.01, sigma, 1.5, true);
                double iv = BlackScholes.impliedVol(price, 100, strike, 0.03, 0.01, 1.5, true);
                assertEquals("round trip sigma=" + sigma + " K=" + strike, sigma, iv, 1e-8);
            }
        }
        // put round trip too (GK-style rates)
        double p = BlackScholes.bsPrice(1.10, 1.05, 0.03, 0.01, 0.10, 0.5, false);
        assertEquals(0.10, BlackScholes.impliedVol(p, 1.10, 1.05, 0.03, 0.01, 0.5, false), 1e-8);
    }

    @Test
    public void impliedVolRejectsArbitragePrices() {
        // above the upper bound S e^{-qT}
        assertThrows(IllegalArgumentException.class,
                () -> BlackScholes.impliedVol(101.0, 100, 100, 0.0, 0.0, 1.0, true));
        // below intrinsic lower bound
        assertThrows(IllegalArgumentException.class,
                () -> BlackScholes.impliedVol(0.0, 100, 50, 0.05, 0.0, 1.0, true));
        assertThrows(IllegalArgumentException.class,
                () -> BlackScholes.impliedVol(5.0, 100, 100, 0.0, 0.0, 0.0, true)); // T = 0
    }

    @Test
    public void validation() {
        assertThrows(IllegalArgumentException.class,
                () -> BlackScholes.bsPrice(-1, 100, 0.0, 0.0, 0.2, 1.0, true));
        assertThrows(IllegalArgumentException.class,
                () -> BlackScholes.bsPrice(100, -1, 0.0, 0.0, 0.2, 1.0, true));
        assertThrows(IllegalArgumentException.class,
                () -> BlackScholes.bsPrice(100, 100, 0.0, 0.0, -0.2, 1.0, true));
        assertThrows(IllegalArgumentException.class,
                () -> BlackScholes.bsPrice(100, 100, 0.0, 0.0, 0.2, -1.0, true));
        assertThrows(IllegalArgumentException.class,
                () -> BlackScholes.bsPrice(100, 100, Double.NaN, 0.0, 0.2, 1.0, true));
        assertThrows(IllegalArgumentException.class,
                () -> BlackScholes.bsDelta(100, 100, 0.0, 0.0, 0.0, 1.0, true)); // sigma = 0
    }
}
