// Monte Carlo: BS agreement within SE bands, antithetic constraints,
// barrier ordering (bridge <= discrete <= vanilla), validation.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "localvol/black_scholes.hpp"
#include "localvol/dupire.hpp"
#include "localvol/market.hpp"
#include "localvol/mc.hpp"
#include "localvol/pde.hpp"
#include "localvol/surface.hpp"

using localvol::bs_price;
using localvol::DupireLocalVol;
using localvol::ImpliedVolSurface;
using localvol::Market;
using localvol::McResult;
using localvol::McSettings;
using localvol::PdeSettings;
using localvol::price_european_mc;
using localvol::price_european_pde;
using localvol::price_up_out_call_mc;
using localvol::VolFn;

namespace {
const std::string kDataDir = LOCALVOL_DATA_DIR;

McSettings settings(int n_paths, int n_steps, std::uint64_t seed = 42) {
    McSettings s;
    s.n_paths = n_paths;
    s.n_steps = n_steps;
    s.seed = seed;
    return s;
}
}  // namespace

TEST(Mc, FlatVolMatchesBlackScholesWithin3Se) {
    const Market mkt(100.0, 0.03, 0.01);
    const double exact = bs_price(100, 105, 0.03, 0.01, 0.2, 1.0, true);
    const McResult res = price_european_mc(mkt, 105.0, 1.0, 0.2, true, settings(20000, 50));
    EXPECT_GT(res.std_err, 0.0);
    EXPECT_TRUE(res.within(exact, 3.0))
        << "MC " << res.price << " +/- " << res.std_err << " vs BS " << exact;
}

TEST(Mc, PutFlavourAndDeterminism) {
    const Market mkt(100.0, 0.02, 0.0);
    const double exact = bs_price(100, 95, 0.02, 0.0, 0.25, 0.5, false);
    const McResult a = price_european_mc(mkt, 95.0, 0.5, 0.25, false, settings(10000, 50));
    const McResult b = price_european_mc(mkt, 95.0, 0.5, 0.25, false, settings(10000, 50));
    EXPECT_TRUE(a.within(exact, 4.0));
    EXPECT_DOUBLE_EQ(a.price, b.price);  // fixed seed -> deterministic
    EXPECT_DOUBLE_EQ(a.std_err, b.std_err);
}

TEST(Mc, AntitheticReducesStandardError) {
    const Market mkt(100.0, 0.03, 0.0);
    McSettings anti = settings(20000, 25);
    McSettings plain = settings(20000, 25);
    plain.antithetic = false;
    const McResult ra = price_european_mc(mkt, 100.0, 1.0, 0.2, true, anti);
    const McResult rp = price_european_mc(mkt, 100.0, 1.0, 0.2, true, plain);
    EXPECT_LT(ra.std_err, rp.std_err);  // ATM payoff: pairing helps
}

TEST(Mc, CallableVolMatchesFlatVol) {
    const Market mkt(100.0, 0.01, 0.0);
    const McResult flat = price_european_mc(mkt, 100.0, 1.0, 0.2, true, settings(4000, 20));
    const McResult fn = price_european_mc(
        mkt, 100.0, 1.0, [](double, double) { return 0.2; }, true, settings(4000, 20));
    EXPECT_DOUBLE_EQ(flat.price, fn.price);  // identical draws, identical sigma
}

TEST(Mc, BarrierOrderingBridgeDiscreteVanilla) {
    const Market mkt(100.0, 0.02, 0.0);
    const McSettings s = settings(20000, 100);
    const McResult vanilla = price_european_mc(mkt, 100.0, 1.0, 0.2, true, s);
    const McResult discrete = price_up_out_call_mc(mkt, 100.0, 130.0, 1.0, 0.2, s, false);
    const McResult bridged = price_up_out_call_mc(mkt, 100.0, 130.0, 1.0, 0.2, s, true);
    EXPECT_LE(bridged.price, discrete.price);  // bridge removes upward bias
    EXPECT_LE(discrete.price, vanilla.price);  // barrier <= vanilla
    EXPECT_GT(bridged.price, 0.0);
}

TEST(Mc, BarrierBornKnockedOutIsZero) {
    const Market mkt(140.0, 0.02, 0.0);
    const McResult res = price_up_out_call_mc(mkt, 100.0, 130.0, 1.0, 0.2, settings(1000, 10));
    EXPECT_DOUBLE_EQ(res.price, 0.0);
    EXPECT_DOUBLE_EQ(res.std_err, 0.0);
}

TEST(Mc, RejectsInvalidInputs) {
    const Market mkt(100.0, 0.02, 0.0);
    EXPECT_THROW(price_european_mc(mkt, 100.0, 0.0, 0.2, true, settings(1000, 10)),
                 std::invalid_argument);  // expiry must be > 0
    EXPECT_THROW(price_european_mc(mkt, -1.0, 1.0, 0.2, true, settings(1000, 10)),
                 std::invalid_argument);
    EXPECT_THROW(price_european_mc(mkt, 100.0, 1.0, 0.0, true, settings(1000, 10)),
                 std::invalid_argument);  // flat vol must be > 0 for MC
    EXPECT_THROW(price_european_mc(mkt, 100.0, 1.0, 0.2, true, settings(1001, 10)),
                 std::invalid_argument);  // antithetic needs even n_paths
    EXPECT_THROW(price_european_mc(mkt, 100.0, 1.0, 0.2, true, settings(1000, 0)),
                 std::invalid_argument);
    McSettings tiny = settings(1, 10);
    tiny.antithetic = false;
    EXPECT_THROW(price_european_mc(mkt, 100.0, 1.0, 0.2, true, tiny), std::invalid_argument);
    EXPECT_THROW(price_up_out_call_mc(mkt, 100.0, -130.0, 1.0, 0.2, settings(1000, 10)),
                 std::invalid_argument);
}

TEST(Mc, LocalVolMatchesPdeWithCarry) {
    // MAJ-7: with r != q the lookup k = X_n - ln F(t_n) is exercised
    // (a ln S0 lookup lands 4-5 SE away from the PDE).
    const ImpliedVolSurface srf = ImpliedVolSurface::from_csv(kDataDir + "/implied_surface.csv");
    const DupireLocalVol lv(srf);
    const VolFn vol = [&lv](double k, double t) { return lv.vol(k, t); };
    const Market markets[] = {Market(100.0, 0.05, 0.02), Market::fx(1.10, 0.03, -0.01)};
    for (const Market& mkt : markets) {
        const double strike = mkt.spot();
        const double pde =
            price_european_pde(mkt, strike, 1.0, vol, true, PdeSettings{}, srf.implied_vol(0.0, 1.0));
        const McResult res = price_european_mc(mkt, strike, 1.0, vol, true, settings(20000, 100, 987));
        EXPECT_TRUE(res.within(pde, 3.0))
            << "S0=" << mkt.spot() << ": MC " << res.price << " +/- " << res.std_err << " vs PDE " << pde;
    }
}

TEST(Mc, PutCallParityUnderLocalVol) {
    const ImpliedVolSurface srf = ImpliedVolSurface::from_csv(kDataDir + "/implied_surface.csv");
    const DupireLocalVol lv(srf);
    const VolFn vol = [&lv](double k, double t) { return lv.vol(k, t); };
    const Market mkt(100.0, 0.03, 0.01);
    const McSettings s = settings(20000, 100, 5);
    const McResult c = price_european_mc(mkt, 100.0, 1.0, vol, true, s);
    const McResult p = price_european_mc(mkt, 100.0, 1.0, vol, false, s);
    const double parity = 100.0 * std::exp(-0.01) - 100.0 * std::exp(-0.03);
    EXPECT_LT(std::abs(c.price - p.price - parity),
              3.0 * std::sqrt(c.std_err * c.std_err + p.std_err * p.std_err));
}

TEST(Mc, RejectsTinyPathCounts) {
    // MAJ-8: n_paths = 2 with antithetic used to return std_err = NaN.
    const Market mkt(100.0, 0.02, 0.0);
    EXPECT_THROW(price_european_mc(mkt, 100.0, 1.0, 0.2, true, settings(2, 5)),
                 std::invalid_argument);
    McSettings one = settings(1, 5);
    one.antithetic = false;
    EXPECT_THROW(price_european_mc(mkt, 100.0, 1.0, 0.2, true, one), std::invalid_argument);
    EXPECT_THROW(price_up_out_call_mc(mkt, 100.0, 130.0, 1.0, 0.2, settings(2, 5)),
                 std::invalid_argument);
    // deep ITM strike so every path pays and the sample spread is non-zero
    const McResult four = price_european_mc(mkt, 50.0, 1.0, 0.2, true, settings(4, 5));
    EXPECT_TRUE(std::isfinite(four.price));
    EXPECT_TRUE(std::isfinite(four.std_err));
    EXPECT_GT(four.std_err, 0.0);
    McSettings two = settings(2, 5);
    two.antithetic = false;
    const McResult r2 = price_european_mc(mkt, 50.0, 1.0, 0.2, true, two);
    EXPECT_TRUE(std::isfinite(r2.std_err));
    EXPECT_GT(r2.std_err, 0.0);
}

TEST(Mc, RejectsNonFiniteOrNegativeVolCallable) {
    // MAJ-4: a bad callable used to yield price = std_err = NaN.
    const Market mkt(100.0, 0.02, 0.0);
    const McSettings s = settings(100, 5);
    const VolFn nan_fn = [](double, double) { return std::nan(""); };
    const VolFn neg_fn = [](double, double) { return -0.2; };
    const VolFn inf_fn = [](double, double) { return std::numeric_limits<double>::infinity(); };
    EXPECT_THROW(price_european_mc(mkt, 100.0, 1.0, nan_fn, true, s), std::invalid_argument);
    EXPECT_THROW(price_european_mc(mkt, 100.0, 1.0, neg_fn, true, s), std::invalid_argument);
    EXPECT_THROW(price_european_mc(mkt, 100.0, 1.0, inf_fn, true, s), std::invalid_argument);
    EXPECT_THROW(price_up_out_call_mc(mkt, 100.0, 130.0, 1.0, nan_fn, s), std::invalid_argument);
    EXPECT_THROW(price_european_mc(mkt, 100.0, 1.0, VolFn{}, true, s), std::invalid_argument);
    // sigma == 0 from a callable is allowed: every path is the forward.
    const Market m2(100.0, 0.03, 0.01);
    const VolFn zero = [](double, double) { return 0.0; };
    const McResult res = price_european_mc(m2, 90.0, 1.0, zero, true, settings(8, 4));
    EXPECT_NEAR(res.price, std::exp(-0.03) * (100.0 * std::exp(0.02) - 90.0), 1e-12);
    EXPECT_DOUBLE_EQ(res.std_err, 0.0);
    const McResult uo = price_up_out_call_mc(m2, 90.0, 130.0, 1.0, zero, settings(8, 4), true);
    EXPECT_NEAR(uo.price, res.price, 1e-12);  // never crosses: no bridge weight
}

TEST(Mc, DifferentSeedsDiffer) {
    const Market mkt(100.0, 0.02, 0.0);
    const McResult a = price_european_mc(mkt, 100.0, 1.0, 0.2, true, settings(2000, 20, 11));
    const McResult b = price_european_mc(mkt, 100.0, 1.0, 0.2, true, settings(2000, 20, 12));
    EXPECT_NE(a.price, b.price);
}
