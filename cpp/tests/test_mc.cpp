// Monte Carlo: BS agreement within SE bands, antithetic constraints,
// barrier ordering (bridge <= discrete <= vanilla), validation.

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "localvol/black_scholes.hpp"
#include "localvol/market.hpp"
#include "localvol/mc.hpp"

using localvol::bs_price;
using localvol::Market;
using localvol::McResult;
using localvol::McSettings;
using localvol::price_european_mc;
using localvol::price_up_out_call_mc;

namespace {
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
