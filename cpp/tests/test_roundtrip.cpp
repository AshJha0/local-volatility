// Round-trip consistency: implied surface -> Dupire -> PDE -> implied vols.
// The full sweep lives in the demo; here the demo's 4 x 7 grid is asserted:
// max interior error < 5 bp (the wing-clamped Dupire stencil brought the
// worst cell from 16.4 bp to 2.7 bp) and no clamp fires on any PDE grid.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <string>

#include "localvol/black_scholes.hpp"
#include "localvol/dupire.hpp"
#include "localvol/market.hpp"
#include "localvol/pde.hpp"
#include "localvol/surface.hpp"

using localvol::DupireLocalVol;
using localvol::implied_vol;
using localvol::ImpliedVolSurface;
using localvol::Market;
using localvol::PdeSettings;
using localvol::price_european_pde;
using localvol::VolFn;

namespace {
const std::string kDataDir = LOCALVOL_DATA_DIR;
}

TEST(RoundTrip, InteriorSubsetUnder5bp) {
    const ImpliedVolSurface srf = ImpliedVolSurface::from_csv(kDataDir + "/implied_surface.csv");
    const DupireLocalVol lv(srf);
    const VolFn vol = [&lv](double k, double t) { return lv.vol(k, t); };
    const Market mkt(100.0, 0.02, 0.01);  // equity carry exercises the forward logic
    for (double expiry : {0.5, 1.0}) {
        const double sigma_ref = srf.implied_vol(0.0, expiry);
        for (double strike : {90.0, 100.0, 110.0}) {
            const double k = std::log(strike / mkt.forward(expiry));
            const double iv_in = srf.implied_vol(k, expiry);
            const double price =
                price_european_pde(mkt, strike, expiry, vol, true, PdeSettings{}, sigma_ref);
            const double iv_out =
                implied_vol(price, mkt.spot(), strike, mkt.rate(), mkt.dividend(), expiry, true);
            const double err_bp = std::abs(iv_out - iv_in) * 1e4;
            EXPECT_LT(err_bp, 30.0) << "K=" << strike << " T=" << expiry;
            EXPECT_LT(err_bp, 5.0) << "K=" << strike << " T=" << expiry << " (post-CRIT-1 budget)";
        }
    }
}

TEST(RoundTrip, FullDemoGridUnder5bpAndNoClamps) {
    const ImpliedVolSurface srf = ImpliedVolSurface::from_csv(kDataDir + "/implied_surface.csv");
    const DupireLocalVol lv(srf);
    const VolFn vol = [&lv](double k, double t) { return lv.vol(k, t); };
    const Market mkt(100.0, 0.02, 0.01);
    double worst = 0.0;
    for (double expiry : {0.5, 1.0, 1.5, 2.0}) {
        const double sigma_ref = srf.implied_vol(0.0, expiry);
        for (double strike : {80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0}) {
            const double k = std::log(strike / mkt.forward(expiry));
            const double iv_in = srf.implied_vol(k, expiry);
            const double price =
                price_european_pde(mkt, strike, expiry, vol, true, PdeSettings{}, sigma_ref);
            const double iv_out =
                implied_vol(price, mkt.spot(), strike, mkt.rate(), mkt.dividend(), expiry, true);
            if (std::abs(k) <= 0.30) worst = std::max(worst, std::abs(iv_out - iv_in) * 1e4);
        }
    }
    EXPECT_LT(worst, 5.0);
    EXPECT_NEAR(worst, 2.73, 0.05);  // K=80, T=2 (documented value)
    EXPECT_EQ(lv.floor_count(), 0);
    EXPECT_EQ(lv.cap_count(), 0);
}
