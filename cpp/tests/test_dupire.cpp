// Dupire local vol: flat-surface identity, clamp counters, validation.

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>
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
