// Black-Scholes analytics: known values, put-call parity over a grid,
// finite-difference Greeks, edge cases, bisection implied vol.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>

#include "localvol/black_scholes.hpp"
#include "localvol/market.hpp"

using localvol::bs_delta;
using localvol::bs_gamma;
using localvol::bs_price;
using localvol::bs_vega;
using localvol::implied_vol;
using localvol::norm_cdf;

TEST(BlackScholes, KnownAtmValue) {
    // Classic textbook value: S=K=100, r=5%, q=0, sigma=20%, T=1.
    EXPECT_NEAR(bs_price(100, 100, 0.05, 0.0, 0.2, 1.0, true), 10.450583572185565, 1e-9);
}

TEST(BlackScholes, NormCdfSymmetryAndTails) {
    EXPECT_DOUBLE_EQ(norm_cdf(0.0), 0.5);
    EXPECT_NEAR(norm_cdf(1.959963984540054), 0.975, 1e-9);
    EXPECT_NEAR(norm_cdf(-8.0) + norm_cdf(8.0), 1.0, 1e-15);
    EXPECT_GT(norm_cdf(-37.0), 0.0);  // erfc keeps the far tail alive
}

TEST(BlackScholes, PutCallParityGrid) {
    // Property-style loop: C - P = S e^{-qT} - K e^{-rT} across a grid.
    const double s = 100.0, r = 0.03, q = 0.015;
    for (double strike : {60.0, 80.0, 100.0, 120.0, 150.0}) {
        for (double t : {0.1, 0.5, 1.0, 2.0}) {
            for (double sig : {0.1, 0.25, 0.5}) {
                const double c = bs_price(s, strike, r, q, sig, t, true);
                const double p = bs_price(s, strike, r, q, sig, t, false);
                const double rhs = s * std::exp(-q * t) - strike * std::exp(-r * t);
                EXPECT_NEAR(c - p, rhs, 1e-10)
                    << "K=" << strike << " T=" << t << " sig=" << sig;
            }
        }
    }
}

TEST(BlackScholes, GarmanKohlhagenFxPut) {
    // FX put via r=rd, q=rf must equal the same formula by construction and
    // satisfy parity with the GK forward.
    const double c = bs_price(1.10, 1.05, 0.03, 0.01, 0.10, 0.5, true);
    const double p = bs_price(1.10, 1.05, 0.03, 0.01, 0.10, 0.5, false);
    const double rhs = 1.10 * std::exp(-0.01 * 0.5) - 1.05 * std::exp(-0.03 * 0.5);
    EXPECT_NEAR(c - p, rhs, 1e-12);
    EXPECT_GT(p, 0.0);
}

TEST(BlackScholes, EdgeCases) {
    EXPECT_DOUBLE_EQ(bs_price(100, 90, 0.05, 0.0, 0.2, 0.0, true), 10.0);   // T=0 intrinsic
    EXPECT_DOUBLE_EQ(bs_price(100, 110, 0.05, 0.0, 0.2, 0.0, true), 0.0);
    const double f = 100.0 * std::exp(0.05);
    EXPECT_NEAR(bs_price(100, 90, 0.05, 0.0, 0.0, 1.0, true),
                std::exp(-0.05) * (f - 90.0), 1e-12);                        // sigma=0
    EXPECT_NEAR(bs_price(100, 0.0, 0.05, 0.02, 0.2, 1.0, true),
                100.0 * std::exp(-0.02), 1e-12);                             // K=0 call
    EXPECT_DOUBLE_EQ(bs_price(100, 0.0, 0.05, 0.02, 0.2, 1.0, false), 0.0);  // K=0 put
}

TEST(BlackScholes, GreeksMatchFiniteDifferences) {
    const double s = 100, k = 105, r = 0.02, q = 0.01, sig = 0.25, t = 0.75;
    const double eps = 1e-4;
    const double dfd =
        (bs_price(s + eps, k, r, q, sig, t, true) - bs_price(s - eps, k, r, q, sig, t, true)) /
        (2 * eps);
    EXPECT_NEAR(bs_delta(s, k, r, q, sig, t, true), dfd, 1e-6);
    const double gfd = (bs_price(s + eps, k, r, q, sig, t, true) -
                        2 * bs_price(s, k, r, q, sig, t, true) +
                        bs_price(s - eps, k, r, q, sig, t, true)) /
                       (eps * eps);
    EXPECT_NEAR(bs_gamma(s, k, r, q, sig, t), gfd, 1e-5);
    const double veps = 1e-5;
    const double vfd = (bs_price(s, k, r, q, sig + veps, t, true) -
                        bs_price(s, k, r, q, sig - veps, t, true)) /
                       (2 * veps);
    EXPECT_NEAR(bs_vega(s, k, r, q, sig, t), vfd, 1e-5);
    // Put delta from parity.
    EXPECT_NEAR(bs_delta(s, k, r, q, sig, t, false),
                bs_delta(s, k, r, q, sig, t, true) - std::exp(-q * t), 1e-12);
}

TEST(BlackScholes, ImpliedVolRoundTrip) {
    for (double sig : {0.08, 0.2, 0.45, 1.2}) {
        for (double strike : {80.0, 100.0, 125.0}) {
            const double price = bs_price(100, strike, 0.03, 0.01, sig, 0.9, true);
            const double iv = implied_vol(price, 100, strike, 0.03, 0.01, 0.9, true);
            EXPECT_NEAR(iv, sig, 1e-8) << "sig=" << sig << " K=" << strike;
        }
    }
    // Puts too.
    const double pp = bs_price(100, 95, 0.02, 0.0, 0.3, 0.5, false);
    EXPECT_NEAR(implied_vol(pp, 100, 95, 0.02, 0.0, 0.5, false), 0.3, 1e-8);
}

TEST(BlackScholes, ImpliedVolRejectsOutOfBoundsPrices) {
    // Below the sigma=0 lower bound.
    EXPECT_THROW(implied_vol(0.0, 100, 50, 0.05, 0.0, 1.0, true), std::invalid_argument);
    // Above the discounted-spot upper bound.
    EXPECT_THROW(implied_vol(101.0, 100, 100, 0.0, 0.0, 1.0, true), std::invalid_argument);
    EXPECT_THROW(implied_vol(5.0, 100, 100, 0.05, 0.0, 0.0, true), std::invalid_argument);
    EXPECT_THROW(implied_vol(5.0, 100, 0.0, 0.05, 0.0, 1.0, true), std::invalid_argument);
}

TEST(BlackScholes, RejectsInvalidInputs) {
    const double nan = std::nan("");
    EXPECT_THROW(bs_price(-1, 100, 0.0, 0.0, 0.2, 1.0), std::invalid_argument);
    EXPECT_THROW(bs_price(100, -1, 0.0, 0.0, 0.2, 1.0), std::invalid_argument);
    EXPECT_THROW(bs_price(100, 100, 0.0, 0.0, -0.2, 1.0), std::invalid_argument);
    EXPECT_THROW(bs_price(100, 100, 0.0, 0.0, 0.2, -1.0), std::invalid_argument);
    EXPECT_THROW(bs_price(100, 100, nan, 0.0, 0.2, 1.0), std::invalid_argument);
    EXPECT_THROW(bs_gamma(100, 100, 0.0, 0.0, 0.0, 1.0), std::invalid_argument);
    EXPECT_THROW(bs_vega(100, 100, 0.0, 0.0, 0.2, 0.0), std::invalid_argument);
}

TEST(BlackScholes, ImpliedVolRejectsBadBracketAndIterations) {
    // MIN-7: non-default lo/hi/iterations are validated.
    const double price = localvol::bs_price(100, 100, 0.02, 0.01, 0.2, 1.0, true);
    EXPECT_THROW(localvol::implied_vol(price, 100, 100, 0.02, 0.01, 1.0, true, 5.0, 1e-9),
                 std::invalid_argument);
    EXPECT_THROW(localvol::implied_vol(price, 100, 100, 0.02, 0.01, 1.0, true, 0.0, 1.0),
                 std::invalid_argument);
    EXPECT_THROW(localvol::implied_vol(price, 100, 100, 0.02, 0.01, 1.0, true, 1e-9,
                                       std::numeric_limits<double>::infinity()),
                 std::invalid_argument);
    EXPECT_THROW(localvol::implied_vol(price, 100, 100, 0.02, 0.01, 1.0, true, 1e-9, 5.0, 0),
                 std::invalid_argument);
    EXPECT_THROW(localvol::implied_vol(std::nan(""), 100, 100, 0.02, 0.01, 1.0, true),
                 std::invalid_argument);
    EXPECT_NEAR(localvol::implied_vol(price, 100, 100, 0.02, 0.01, 1.0, true), 0.2, 1e-10);
}

TEST(BlackScholes, MarketLogForwardValidationAndValue) {
    // MIN-6: log_forward validates expiry in every port.
    const localvol::Market mkt(100.0, 0.03, 0.01);
    EXPECT_NEAR(mkt.log_forward(2.0), std::log(100.0) + 0.02 * 2.0, 1e-15);
    EXPECT_DOUBLE_EQ(mkt.log_forward(0.0), std::log(100.0));
    EXPECT_THROW(mkt.log_forward(-1.0), std::invalid_argument);
    EXPECT_THROW(mkt.log_forward(std::nan("")), std::invalid_argument);
    EXPECT_THROW(mkt.forward(std::numeric_limits<double>::infinity()), std::invalid_argument);
    EXPECT_THROW(localvol::Market(100.0, std::nan(""), 0.0), std::invalid_argument);
    EXPECT_THROW(localvol::Market(100.0, 0.0, std::numeric_limits<double>::infinity()),
                 std::invalid_argument);
}
