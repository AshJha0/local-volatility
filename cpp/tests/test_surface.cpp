// Surface & spline: exactness at nodes, continuity near nodes, wing and
// temporal extrapolation rules, calendar-arbitrage detection, validation.

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "localvol/surface.hpp"

using localvol::CubicSpline1D;
using localvol::ImpliedVolSurface;

namespace {

const std::string kDataDir = LOCALVOL_DATA_DIR;

ImpliedVolSurface make_surface() {
    // Two expiries, four strikes, gentle smile, calendar-consistent.
    std::vector<double> k = {-0.4, -0.1, 0.1, 0.4};
    std::vector<double> t = {0.5, 1.0};
    std::vector<std::vector<double>> v = {
        {0.25, 0.21, 0.20, 0.23},
        {0.24, 0.215, 0.205, 0.225},
    };
    return ImpliedVolSurface(k, t, v);
}

}  // namespace

TEST(Spline, ExactAtNodesAndContinuous) {
    const std::vector<double> x = {-1.0, -0.2, 0.3, 0.9, 2.0};
    const std::vector<double> y = {1.5, 0.7, 0.9, 2.2, -0.3};
    const CubicSpline1D s(x, y);
    for (std::size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(s(x[i]), y[i], 1e-14);  // interpolation, not smoothing
    }
    // C0/C1-style continuity probe just left/right of each interior node.
    const double eps = 1e-7;
    for (std::size_t i = 1; i + 1 < x.size(); ++i) {
        EXPECT_NEAR(s(x[i] - eps), s(x[i] + eps), 1e-5);
        const double dl = (s(x[i]) - s(x[i] - eps)) / eps;
        const double dr = (s(x[i] + eps) - s(x[i])) / eps;
        EXPECT_NEAR(dl, dr, 1e-4);  // matching first derivative
    }
}

TEST(Spline, FlatExtrapolationAndDegenerateSizes) {
    const CubicSpline1D s({0.0, 1.0, 2.0}, {1.0, 3.0, 2.0});
    EXPECT_DOUBLE_EQ(s(-5.0), 1.0);  // clamped left
    EXPECT_DOUBLE_EQ(s(10.0), 2.0);  // clamped right
    const CubicSpline1D lin({0.0, 2.0}, {1.0, 5.0});
    EXPECT_NEAR(lin(0.5), 2.0, 1e-14);  // 2 nodes -> linear
    const CubicSpline1D cst({1.0}, {7.0});
    EXPECT_DOUBLE_EQ(cst(123.0), 7.0);  // 1 node -> constant
}

TEST(Spline, RejectsBadInput) {
    EXPECT_THROW(CubicSpline1D({}, {}), std::invalid_argument);
    EXPECT_THROW(CubicSpline1D({0.0, 1.0}, {1.0}), std::invalid_argument);
    EXPECT_THROW(CubicSpline1D({0.0, 0.0}, {1.0, 2.0}), std::invalid_argument);
    EXPECT_THROW(CubicSpline1D({0.0, std::nan("")}, {1.0, 2.0}), std::invalid_argument);
    const CubicSpline1D s({0.0, 1.0}, {0.0, 1.0});
    EXPECT_THROW(s(std::nan("")), std::invalid_argument);
}

TEST(Surface, NodeExactnessAndZeroExpiry) {
    const ImpliedVolSurface srf = make_surface();
    // w at the nodes equals iv^2 * T exactly.
    EXPECT_NEAR(srf.total_variance(-0.1, 0.5), 0.21 * 0.21 * 0.5, 1e-15);
    EXPECT_NEAR(srf.total_variance(0.4, 1.0), 0.225 * 0.225 * 1.0, 1e-15);
    EXPECT_DOUBLE_EQ(srf.total_variance(0.0, 0.0), 0.0);  // w(k, 0) = 0
    // implied_vol at T=0 returns the short-end limit.
    EXPECT_NEAR(srf.implied_vol(-0.1, 0.0), 0.21, 1e-12);
}

TEST(Surface, TemporalInterpolationIsLinearInW) {
    const ImpliedVolSurface srf = make_surface();
    const double w1 = srf.total_variance(0.05, 0.5);
    const double w2 = srf.total_variance(0.05, 1.0);
    EXPECT_NEAR(srf.total_variance(0.05, 0.75), 0.5 * (w1 + w2), 1e-14);
    // Below the first pillar: proportional in T (flat forward variance).
    EXPECT_NEAR(srf.total_variance(0.05, 0.25), w1 * 0.5, 1e-14);
    // Beyond the last pillar: linear continuation with slope >= 0.
    const double slope = (w2 - w1) / 0.5;
    EXPECT_NEAR(srf.total_variance(0.05, 1.5), w2 + slope * 0.5, 1e-14);
}

TEST(Surface, FlatWingExtrapolation) {
    const ImpliedVolSurface srf = make_surface();
    EXPECT_DOUBLE_EQ(srf.total_variance(-3.0, 1.0), srf.total_variance(-0.4, 1.0));
    EXPECT_DOUBLE_EQ(srf.total_variance(9.0, 1.0), srf.total_variance(0.4, 1.0));
}

TEST(Surface, ExtrapolationSlopeFlooredAtZero) {
    // Second pillar with *lower* total variance -> slope < 0 -> floored.
    std::vector<double> k = {0.0};
    std::vector<double> t = {1.0, 2.0};
    std::vector<std::vector<double>> v = {{0.30}, {0.20}};  // w: 0.09 -> 0.08
    const ImpliedVolSurface srf(k, t, v);
    EXPECT_EQ(srf.calendar_violations(), 1);
    EXPECT_NEAR(srf.total_variance(0.0, 3.0), srf.total_variance(0.0, 2.0), 1e-15);
}

TEST(Surface, SingleExpiryFlatForwardRule) {
    std::vector<double> k = {-0.2, 0.0, 0.2};
    std::vector<double> t = {1.0};
    std::vector<std::vector<double>> v = {{0.22, 0.20, 0.21}};
    const ImpliedVolSurface srf(k, t, v);
    EXPECT_TRUE(srf.single_expiry());
    const double w1 = srf.total_variance(0.0, 1.0);
    EXPECT_NEAR(srf.total_variance(0.0, 0.5), 0.5 * w1, 1e-15);
    EXPECT_NEAR(srf.total_variance(0.0, 2.0), 2.0 * w1, 1e-15);  // both sides of T1
}

TEST(Surface, CalendarViolationsCounted) {
    std::vector<double> k = {-0.1, 0.1};
    std::vector<double> t = {0.5, 1.0};
    // w row 0: 0.02, 0.02; row 1 first node: 0.01 < 0.02 -> one violation.
    std::vector<std::vector<double>> v = {{0.2, 0.2}, {0.1, 0.2}};
    const ImpliedVolSurface srf(k, t, v);
    EXPECT_EQ(srf.calendar_violations(), 1);
    const ImpliedVolSurface clean = make_surface();
    EXPECT_EQ(clean.calendar_violations(), 0);
}

TEST(Surface, LoadsBundledCsvAndIsRectangular) {
    const ImpliedVolSurface srf = ImpliedVolSurface::from_csv(kDataDir + "/implied_surface.csv");
    EXPECT_EQ(srf.expiries().size(), 4u);
    EXPECT_EQ(srf.k_nodes().size(), 15u);
    EXPECT_EQ(srf.calendar_violations(), 0);  // bundled surface is arb-free
    EXPECT_NEAR(srf.k_min(), -0.5, 1e-9);
    EXPECT_NEAR(srf.k_max(), 0.5, 1e-9);
    const ImpliedVolSurface flat = ImpliedVolSurface::from_csv(kDataDir + "/flat_surface.csv");
    for (double k : {-0.5, -0.17, 0.0, 0.33, 0.5}) {
        for (double t : {0.1, 0.25, 1.0, 3.0}) {
            EXPECT_NEAR(flat.implied_vol(k, t), 0.20, 1e-12);
        }
    }
}

TEST(Surface, RejectsBadInput) {
    std::vector<double> k = {-0.1, 0.1};
    std::vector<double> t = {0.5, 1.0};
    std::vector<std::vector<double>> v = {{0.2, 0.2}, {0.2, 0.2}};
    EXPECT_THROW(ImpliedVolSurface({}, t, v), std::invalid_argument);
    EXPECT_THROW(ImpliedVolSurface(k, {}, {}), std::invalid_argument);
    EXPECT_THROW(ImpliedVolSurface(k, t, {{0.2, 0.2}}), std::invalid_argument);  // shape
    EXPECT_THROW(ImpliedVolSurface({0.1, -0.1}, t, v), std::invalid_argument);   // k order
    EXPECT_THROW(ImpliedVolSurface(k, {1.0, 0.5}, v), std::invalid_argument);    // T order
    EXPECT_THROW(ImpliedVolSurface(k, {0.0, 1.0}, v), std::invalid_argument);    // T <= 0
    EXPECT_THROW(ImpliedVolSurface(k, t, {{0.2, -0.2}, {0.2, 0.2}}), std::invalid_argument);
    EXPECT_THROW(ImpliedVolSurface::from_csv(kDataDir + "/does_not_exist.csv"),
                 std::invalid_argument);
    const ImpliedVolSurface srf(k, t, v);
    EXPECT_THROW(srf.total_variance(0.0, -1.0), std::invalid_argument);
    EXPECT_THROW(srf.total_variance(std::nan(""), 1.0), std::invalid_argument);
}
