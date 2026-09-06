// Dupire local vol: flat-surface identity, clamp counters, validation.

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "localvol/dupire.hpp"
#include "localvol/surface.hpp"

using localvol::DupireLocalVol;
using localvol::ImpliedVolSurface;

namespace {
const std::string kDataDir = LOCALVOL_DATA_DIR;
}

TEST(Dupire, FlatSurfaceGivesFlatLocalVolEverywhere) {
    // Property-style grid loop: a flat implied surface must collapse the
    // Gatheral formula to sigma_loc == implied vol identically (to 1e-6).
    const ImpliedVolSurface srf = ImpliedVolSurface::from_csv(kDataDir + "/flat_surface.csv");
    const DupireLocalVol lv(srf);
    for (double k : {-0.45, -0.2, 0.0, 0.15, 0.45}) {
        for (double t : {0.1, 0.25, 0.7, 1.0, 1.9, 2.5}) {
            EXPECT_NEAR(lv.vol(k, t), 0.20, 1e-6) << "k=" << k << " T=" << t;
        }
    }
    EXPECT_EQ(lv.floor_count(), 0);
    EXPECT_EQ(lv.cap_count(), 0);
}

TEST(Dupire, BundledSurfaceSkewShape) {
    // Equity-style skew: local vol on the put wing exceeds the call wing.
    const ImpliedVolSurface srf = ImpliedVolSurface::from_csv(kDataDir + "/implied_surface.csv");
    const DupireLocalVol lv(srf);
    const double put_wing = lv.vol(-0.25, 1.0);
    const double atm = lv.vol(0.0, 1.0);
    const double call_wing = lv.vol(0.25, 1.0);
    EXPECT_GT(put_wing, atm);
    EXPECT_LT(call_wing, atm);
    for (double v : {put_wing, atm, call_wing}) {
        EXPECT_GE(v, DupireLocalVol::kFloor);
        EXPECT_LE(v, DupireLocalVol::kCap);
    }
}

TEST(Dupire, FloorEngagesOnCalendarArbitrageAndIsCounted) {
    // Total variance *decreasing* in T at k=0 -> dw/dT <= 0 -> floor.
    std::vector<double> k = {-0.3, 0.0, 0.3};
    std::vector<double> t = {0.5, 1.0};
    std::vector<std::vector<double>> v = {
        {0.30, 0.30, 0.30},
        {0.10, 0.10, 0.10},  // w drops from 0.045 to 0.01
    };
    const ImpliedVolSurface srf(k, t, v);
    EXPECT_GT(srf.calendar_violations(), 0);
    const DupireLocalVol lv(srf);
    EXPECT_DOUBLE_EQ(lv.vol(0.0, 0.75), DupireLocalVol::kFloor);
    EXPECT_GE(lv.floor_count(), 1);
    lv.reset_counters();
    EXPECT_EQ(lv.floor_count(), 0);
    EXPECT_EQ(lv.cap_count(), 0);
}

TEST(Dupire, ReportStringMentionsCounts) {
    const ImpliedVolSurface srf = ImpliedVolSurface::from_csv(kDataDir + "/flat_surface.csv");
    const DupireLocalVol lv(srf);
    (void)lv.vol(0.0, 1.0);
    const std::string rep = lv.violation_report();
    EXPECT_NE(rep.find("floor"), std::string::npos);
    EXPECT_NE(rep.find("cap"), std::string::npos);
}

TEST(Dupire, RejectsInvalidQueries) {
    const ImpliedVolSurface srf = ImpliedVolSurface::from_csv(kDataDir + "/flat_surface.csv");
    const DupireLocalVol lv(srf);
    EXPECT_THROW(lv.vol(0.0, -0.5), std::invalid_argument);
    EXPECT_THROW(lv.vol(std::nan(""), 1.0), std::invalid_argument);
    EXPECT_THROW(lv.local_variance(0.0, std::nan("")), std::invalid_argument);
}

TEST(Dupire, TemporaryBindingIsACompileError) {
    // MAJ-2: `DupireLocalVol lv(ImpliedVolSurface::from_csv(path));` would
    // dangle; the rvalue overload is deleted so the code does not compile.
    static_assert(!std::is_constructible_v<DupireLocalVol, ImpliedVolSurface>,
                  "DupireLocalVol must not accept a temporary surface");
    static_assert(std::is_constructible_v<DupireLocalVol, const ImpliedVolSurface&>,
                  "DupireLocalVol must accept an lvalue surface");
    static_assert(!std::is_copy_constructible_v<DupireLocalVol>, "shared by reference");
}

TEST(Dupire, NoCapInsideQuotedBoxAndContinuousAtWings) {
    // CRIT-1: before the stencil clamp vol(0.4995..0.5005, 1) was 5.0.
    const ImpliedVolSurface srf = ImpliedVolSurface::from_csv(kDataDir + "/implied_surface.csv");
    const DupireLocalVol lv(srf);
    for (double t : {0.1, 0.5, 0.75, 1.0, 2.0, 3.0}) {
        for (int i = 0; i <= 200; ++i) {
            const double k = -0.5 + 0.005 * i;
            const double v = lv.vol(k, t);
            EXPECT_GT(v, DupireLocalVol::kFloor) << "k=" << k << " T=" << t;
            EXPECT_LT(v, DupireLocalVol::kCap) << "k=" << k << " T=" << t;
        }
        for (double edge : {srf.k_min(), srf.k_max()}) {
            const double lo = lv.vol(edge - 1e-4, t);
            const double hi = lv.vol(edge + 1e-4, t);
            EXPECT_LT(std::abs(lo - hi), 1e-3) << "T=" << t << " k=" << edge;
            // constant in k beyond k_max - DK (clamped stencil)
            const double far = lv.vol(edge + (edge > 0 ? 1.0 : -1.0), t);
            EXPECT_DOUBLE_EQ(far, edge > 0 ? hi : lo);
        }
    }
    EXPECT_EQ(lv.floor_count(), 0);
    EXPECT_EQ(lv.cap_count(), 0);
    EXPECT_NEAR(lv.vol(0.5, 1.0), 0.19703445, 1e-6);
}

TEST(Dupire, TermStructureLocalVolIsForwardVariance) {
    // Flat in k, iv = {0.15, 0.20, 0.25} at T = {0.25, 0.5, 1}.
    std::vector<double> ks;
    for (int i = 0; i < 11; ++i) ks.push_back(-0.5 + 0.1 * i);
    const std::vector<double> ts = {0.25, 0.5, 1.0};
    const std::vector<std::vector<double>> v = {
        std::vector<double>(11, 0.15), std::vector<double>(11, 0.20), std::vector<double>(11, 0.25)};
    const ImpliedVolSurface srf(ks, ts, v);
    const DupireLocalVol lv(srf);
    EXPECT_NEAR(lv.vol(0.0, 0.4), std::sqrt(0.0575), 1e-6);   // (0.02 - 0.005625)/0.25
    EXPECT_NEAR(lv.vol(0.3, 0.4), std::sqrt(0.0575), 1e-6);
    EXPECT_NEAR(lv.vol(0.0, 0.75), std::sqrt(0.085), 1e-6);   // (0.0625 - 0.02)/0.5
    EXPECT_NEAR(lv.vol(0.0, 0.1), 0.15, 1e-6);                // below first pillar
    EXPECT_NEAR(lv.vol(0.0, 2.0), std::sqrt(0.085), 1e-6);    // beyond last pillar
    EXPECT_EQ(lv.floor_count(), 0);
    EXPECT_EQ(lv.cap_count(), 0);
}

TEST(Dupire, ZeroExpiryIsShortTimeLimit) {
    // MAJ-1: before the fix vol(-0.3, 0) = 0.309 vs vol(-0.3, 1e-6) = 0.452.
    const ImpliedVolSurface srf = ImpliedVolSurface::from_csv(kDataDir + "/implied_surface.csv");
    const DupireLocalVol lv(srf);
    for (double k : {-0.3, -0.1, 0.0, 0.2, 0.45}) {
        const double v0 = lv.vol(k, 0.0);
        EXPECT_LT(std::abs(v0 - lv.vol(k, 1e-6)), 1e-3) << "k=" << k;
        EXPECT_LT(std::abs(v0 - lv.vol(k, 1e-4)), 1e-3) << "k=" << k;
        // independent BBF evaluation from the short-end implied slice
        const double s = srf.implied_vol(k, 0.0);
        const double sp = (srf.implied_vol(k + 1e-3, 0.0) - srf.implied_vol(k - 1e-3, 0.0)) / 2e-3;
        EXPECT_NEAR(v0, s / (1.0 - k * sp / s), 1e-12) << "k=" << k;
    }
    EXPECT_NEAR(lv.vol(-0.3, 0.0), 0.45240187, 1e-6);
    EXPECT_NEAR(lv.vol(0.0, 0.0), 0.2, 1e-12);
}

TEST(Dupire, ZeroExpiryCapsOnStronglyConvexSmile) {
    // At |k| = 0.5, k s'/s = 0.5 * 4 / 1.05 = 1.9 > 1 -> denominator <= 0 -> cap.
    std::vector<double> ks;
    std::vector<double> v1;
    for (int i = 0; i < 21; ++i) {
        const double k = -0.5 + 0.05 * i;
        ks.push_back(k);
        v1.push_back(0.05 + 4.0 * k * k);
    }
    std::vector<double> v2;
    for (double v : v1) v2.push_back(v * 1.05);
    const ImpliedVolSurface srf(ks, {0.5, 1.0}, {v1, v2});
    const DupireLocalVol lv(srf);
    for (double k : ks) {
        const double v = lv.vol(k, 0.0);
        EXPECT_LE(v, DupireLocalVol::kCap);
        EXPECT_GE(v, DupireLocalVol::kFloor);
    }
    EXPECT_GT(lv.cap_count(), 0);
    EXPECT_DOUBLE_EQ(lv.vol(0.5, 0.0), DupireLocalVol::kCap);
    EXPECT_NEAR(lv.vol(0.0, 0.0), 0.05, 1e-9);
}

TEST(Dupire, CountersAreThreadSafe) {
    // MAJ-3: four threads hammering one shared object over a cap-firing
    // region must produce exactly the single-threaded count.
    std::vector<double> ks;
    std::vector<double> v1;
    for (int i = 0; i < 21; ++i) {
        const double k = -0.5 + 0.05 * i;
        ks.push_back(k);
        v1.push_back(0.05 + 4.0 * k * k);
    }
    std::vector<double> v2;
    for (double v : v1) v2.push_back(v * 1.05);
    const ImpliedVolSurface srf(ks, {0.5, 1.0}, {v1, v2});
    const DupireLocalVol lv(srf);
    const int per_thread = 100000;
    auto work = [&lv, per_thread]() {
        for (int i = 0; i < per_thread; ++i) {
            const double k = -0.5 + (i % 101) * 0.01;
            (void)lv.vol(k, 0.75);
        }
    };
    work();
    const long single_floor = lv.floor_count();
    const long single_cap = lv.cap_count();
    EXPECT_GT(single_cap, 0);
    lv.reset_counters();
    std::vector<std::thread> pool;
    for (int t = 0; t < 4; ++t) pool.emplace_back(work);
    for (auto& th : pool) th.join();
    EXPECT_EQ(lv.floor_count(), 4 * single_floor);
    EXPECT_EQ(lv.cap_count(), 4 * single_cap);
}
