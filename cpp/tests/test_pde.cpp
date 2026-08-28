// PDE pricer: BS agreement, parity, Rannacher damping, convergence order,
// American exercise properties, validation.

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "localvol/black_scholes.hpp"
#include "localvol/dupire.hpp"
#include "localvol/market.hpp"
#include "localvol/pde.hpp"
#include "localvol/surface.hpp"

using localvol::bs_price;
using localvol::DupireLocalVol;
using localvol::ImpliedVolSurface;
using localvol::Market;
using localvol::PdeResult;
using localvol::PdeSettings;
using localvol::price_american_put_pde;
using localvol::price_european_pde;
using localvol::price_european_pde_grid;
using localvol::PsorSettings;

namespace {
const std::string kDataDir = LOCALVOL_DATA_DIR;
}

TEST(Pde, FlatVolMatchesBlackScholes) {
    struct Case {
        double s, k, r, q, sig, t;
        bool call;
    };
    const Case cases[] = {
        {100.0, 100.0, 0.05, 0.02, 0.20, 1.0, true},   // equity ATM call, q > 0
        {100.0, 80.0, 0.05, 0.00, 0.20, 1.0, true},    // ITM call
        {100.0, 120.0, 0.05, 0.00, 0.30, 0.5, false},  // ITM put
        {1.10, 1.05, 0.03, 0.01, 0.10, 0.5, false},    // FX put (GK rd/rf)
        {100.0, 110.0, -0.01, 0.00, 0.25, 2.0, true},  // negative rate
        {100.0, 100.0, 0.00, 0.03, 0.15, 0.25, true},  // short-dated, q > r
    };
    for (const Case& c : cases) {
        const double exact = bs_price(c.s, c.k, c.r, c.q, c.sig, c.t, c.call);
        const double approx = price_european_pde(Market(c.s, c.r, c.q), c.k, c.t, c.sig, c.call);
        EXPECT_NEAR(approx, exact, 1e-3 * std::abs(exact))
            << "S=" << c.s << " K=" << c.k << " call=" << c.call;
    }
}

TEST(Pde, DeepMoneynessAndDegenerateInputs) {
    const Market mkt(100.0, 0.02, 0.0);
    const double otm = price_european_pde(mkt, 300.0, 0.5, 0.2, true);
    EXPECT_GE(otm, 0.0);
    EXPECT_LT(otm, 1e-3);
    const double itm = price_european_pde(mkt, 10.0, 0.5, 0.2, true);
    EXPECT_NEAR(itm, bs_price(100, 10, 0.02, 0.0, 0.2, 0.5, true), 1e-3 * itm);
    // T = 0 -> intrinsic; sigma = 0 -> discounted forward intrinsic; K = 0.
    EXPECT_DOUBLE_EQ(price_european_pde(mkt, 90.0, 0.0, 0.2, true), 10.0);
    EXPECT_NEAR(price_european_pde(mkt, 90.0, 1.0, 0.0, true),
                bs_price(100, 90, 0.02, 0.0, 0.0, 1.0, true), 1e-12);
    EXPECT_NEAR(price_european_pde(mkt, 0.0, 1.0, 0.2, true), 100.0, 1e-12);
    EXPECT_DOUBLE_EQ(price_european_pde(mkt, 0.0, 1.0, 0.2, false), 0.0);
}

TEST(Pde, PutCallParityOnGrid) {
    const Market mkt(100.0, 0.03, 0.01);
    PdeSettings s;
    s.num_time = 100;
    const double c = price_european_pde(mkt, 105.0, 1.0, 0.2, true, s);
    const double p = price_european_pde(mkt, 105.0, 1.0, 0.2, false, s);
    const double rhs = 100.0 * std::exp(-0.01) - 105.0 * std::exp(-0.03);
    EXPECT_NEAR(c - p, rhs, 2e-2);
}

TEST(Pde, RannacherDampsGammaOscillation) {
    // The most negative second difference of the final grid near the strike
    // must be negligible relative to the peak curvature.
    const Market mkt(100.0, 0.05, 0.0);
    PdeSettings s;
    s.num_time = 50;
    const PdeResult res = price_european_pde_grid(mkt, 100.0, 0.25, 0.2, true, s);
    double max_curv = 0.0, min_curv = 0.0;
    for (std::size_t i = 1; i + 1 < res.values.size(); ++i) {
        const double spot = std::exp(res.x[i]);
        if (spot <= 70.0 || spot >= 140.0) continue;
        const double curv = res.values[i + 1] - 2.0 * res.values[i] + res.values[i - 1];
        max_curv = std::max(max_curv, curv);
        min_curv = std::min(min_curv, curv);
    }
    EXPECT_GT(max_curv, 0.0);
    EXPECT_GT(min_curv, -1e-4 * max_curv);
}

TEST(Pde, GridConvergenceIsSecondOrder) {
    // Halving dx and dt twice: observed order must land in [1.5, 2.5].
    const Market mkt(100.0, 0.05, 0.0);
    const double exact = bs_price(100, 100, 0.05, 0.0, 0.2, 1.0, true);
    std::vector<double> errs;
    for (int m : {50, 100, 200}) {
        PdeSettings s;
        s.num_space = m;
        s.num_time = m;
        errs.push_back(std::abs(price_european_pde(mkt, 100.0, 1.0, 0.2, true, s) - exact));
    }
    const double o1 = std::log2(errs[0] / errs[1]);
    const double o2 = std::log2(errs[1] / errs[2]);
    const double avg = 0.5 * (o1 + o2);
    EXPECT_GE(avg, 1.5) << "orders " << o1 << ", " << o2;
    EXPECT_LE(avg, 2.5) << "orders " << o1 << ", " << o2;
}

TEST(Pde, LocalVolCallableNearBsAtSurfaceVol) {
    const ImpliedVolSurface srf = ImpliedVolSurface::from_csv(kDataDir + "/implied_surface.csv");
    const DupireLocalVol lv(srf);
    const Market mkt(100.0, 0.0, 0.0);
    const double sigma_ref = srf.implied_vol(0.0, 1.0);
    const double price = price_european_pde(
        mkt, 100.0, 1.0, [&lv](double k, double t) { return lv.vol(k, t); }, true, PdeSettings{},
        sigma_ref);
    const double ref = bs_price(100, 100, 0.0, 0.0, sigma_ref, 1.0, true);
    EXPECT_NEAR(price, ref, 5e-3 * ref);  // ~1bp vol agreement
}

TEST(Pde, AmericanPutPremiumAndProperties) {
    const Market mkt(100.0, 0.05, 0.0);
    PdeSettings s;
    s.num_space = 150;
    s.num_time = 100;
    const double eur = price_european_pde(mkt, 100.0, 1.0, 0.2, false, s);
    const double ame = price_american_put_pde(mkt, 100.0, 1.0, 0.2, s);
    EXPECT_GE(ame, eur - 1e-10);   // early-exercise premium >= 0
    EXPECT_GE(ame, 0.0);           // >= intrinsic (ATM: 0)
    EXPECT_GT(ame - eur, 0.01);    // strictly positive premium for r > 0, q = 0
}

TEST(Pde, AmericanEqualsEuropeanAtZeroRates) {
    // r = 0, q = 0: an American put is never exercised early.
    const Market mkt(100.0, 0.0, 0.0);
    PdeSettings s;
    s.num_space = 150;
    s.num_time = 100;
    const double eur = price_european_pde(mkt, 110.0, 1.0, 0.25, false, s);
    const double ame = price_american_put_pde(mkt, 110.0, 1.0, 0.25, s);
    EXPECT_NEAR(ame, eur, 2e-3 * eur);
}

TEST(Pde, AmericanLocalVolAboveEuropean) {
    const ImpliedVolSurface srf = ImpliedVolSurface::from_csv(kDataDir + "/implied_surface.csv");
    const DupireLocalVol lv(srf);
    const auto vol = [&lv](double k, double t) { return lv.vol(k, t); };
    const Market mkt(100.0, 0.04, 0.0);
    PdeSettings s;
    s.num_space = 150;
    s.num_time = 100;
    const double eur = price_european_pde(mkt, 105.0, 1.0, vol, false, s, 0.2);
    const double ame = price_american_put_pde(mkt, 105.0, 1.0, vol, s, PsorSettings{}, 0.2);
    EXPECT_GE(ame, eur - 1e-10);
}

TEST(Pde, SigmaZeroAmericanIsDeterministicOptimum) {
    // r > 0, sigma = 0: immediate exercise is optimal for a deep ITM put.
    const Market mkt(80.0, 0.05, 0.0);
    EXPECT_NEAR(price_american_put_pde(mkt, 100.0, 1.0, 0.0), 20.0, 1e-6);
    // T = 0 short-circuit.
    EXPECT_DOUBLE_EQ(price_american_put_pde(mkt, 100.0, 0.0, 0.2), 20.0);
}

TEST(Pde, RejectsInvalidInputs) {
    const Market mkt(100.0, 0.05, 0.0);
    PdeSettings s;
    s.num_space = 201;  // odd
    EXPECT_THROW(price_european_pde(mkt, 100.0, 1.0, 0.2, true, s), std::invalid_argument);
    s.num_space = 2;  // < 4
    EXPECT_THROW(price_european_pde(mkt, 100.0, 1.0, 0.2, true, s), std::invalid_argument);
    s = PdeSettings{};
    s.num_time = 0;
    EXPECT_THROW(price_european_pde(mkt, 100.0, 1.0, 0.2, true, s), std::invalid_argument);
    s = PdeSettings{};
    s.nsd = 0.0;
    EXPECT_THROW(price_european_pde(mkt, 100.0, 1.0, 0.2, true, s), std::invalid_argument);
    EXPECT_THROW(price_european_pde(mkt, -5.0, 1.0, 0.2, true), std::invalid_argument);
    EXPECT_THROW(price_european_pde(mkt, 100.0, -1.0, 0.2, true), std::invalid_argument);
    EXPECT_THROW(price_european_pde(mkt, 100.0, 1.0, -0.2, true), std::invalid_argument);
    PsorSettings bad;
    bad.omega = 2.5;
    EXPECT_THROW(price_american_put_pde(mkt, 100.0, 1.0, 0.2, PdeSettings{}, bad),
                 std::invalid_argument);
    bad = PsorSettings{};
    bad.tol = 0.0;
    EXPECT_THROW(price_american_put_pde(mkt, 100.0, 1.0, 0.2, PdeSettings{}, bad),
                 std::invalid_argument);
    EXPECT_THROW(price_american_put_pde(mkt, 0.0, 1.0, 0.2), std::invalid_argument);
    EXPECT_THROW(Market(-1.0, 0.0, 0.0), std::invalid_argument);
    EXPECT_THROW(Market(100.0, std::nan(""), 0.0), std::invalid_argument);
}
