/// \file demo.cpp
/// \brief localvol demo: Dupire round-trip consistency check.
///
/// Pipeline: bundled SSVI implied surface -> Dupire local vol -> reprice
/// European vanillas by PDE (and spot-check by Monte Carlo) -> invert back
/// to implied vols -> report the error against the input surface in bp.

#include <cmath>
#include <cstdio>
#include <string>

#include "localvol/black_scholes.hpp"
#include "localvol/dupire.hpp"
#include "localvol/market.hpp"
#include "localvol/mc.hpp"
#include "localvol/pde.hpp"
#include "localvol/surface.hpp"

#ifndef LOCALVOL_DATA_DIR
#define LOCALVOL_DATA_DIR "../data"
#endif

namespace {
constexpr double kInteriorTargetBp = 30.0;
}

int main() {
    using namespace localvol;

    std::printf("%s\n", std::string(72, '=').c_str());
    std::printf("localvol demo — Dupire local volatility round-trip consistency (C++)\n");
    std::printf("%s\n", std::string(72, '=').c_str());

    const std::string data_dir = LOCALVOL_DATA_DIR;
    const ImpliedVolSurface surface =
        ImpliedVolSurface::from_csv(data_dir + "/implied_surface.csv");
    const DupireLocalVol lv(surface);
    const Market mkt(100.0, 0.02, 0.01);  // equity-style carry
    std::printf("surface: %zu expiries x %zu strikes, calendar violations: %d\n",
                surface.expiries().size(), surface.k_nodes().size(),
                surface.calendar_violations());
    std::printf("market : S0=%.2f  r=%.2f%%  q=%.2f%%\n\n", mkt.spot(), 100.0 * mkt.rate(),
                100.0 * mkt.dividend());

    const VolFn vol_fn = [&lv](double k, double t) { return lv.vol(k, t); };

    // ---- round trip: implied in -> local vol -> PDE price -> implied out
    const double expiries[] = {0.5, 1.0, 1.5, 2.0};
    const double strikes[] = {80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0};
    std::printf("Round-trip implied-vol error (PDE 200x200), basis points:\n");
    std::printf("  T\\K   ");
    for (double strike : strikes) std::printf("%8.0f", strike);
    std::printf("\n");
    double max_err_bp = 0.0;
    double max_at_k = 0.0, max_at_t = 0.0;
    for (double t : expiries) {
        lv.reset_counters();
        std::printf("  %-5.2f", t);
        const double sigma_ref = surface.implied_vol(0.0, t);
        for (double strike : strikes) {
            const double k = std::log(strike / mkt.forward(t));
            const double iv_in = surface.implied_vol(k, t);
            const double price =
                price_european_pde(mkt, strike, t, vol_fn, true, PdeSettings{}, sigma_ref);
            const double iv_out =
                implied_vol(price, mkt.spot(), strike, mkt.rate(), mkt.dividend(), t, true);
            const double err_bp = std::abs(iv_out - iv_in) * 1e4;
            std::printf("%8.1f", err_bp);
            const bool interior = std::abs(k) <= 0.30;  // wings excluded per spec
            if (interior && err_bp > max_err_bp) {
                max_err_bp = err_bp;
                max_at_k = strike;
                max_at_t = t;
            }
        }
        std::printf("\n");
    }
    std::printf("\nmax interior |vol error|: %.2f bp at K=%.0f, T=%.2f  (target < %.0f bp, "
                "|k| <= 0.30)\n",
                max_err_bp, max_at_k, max_at_t, kInteriorTargetBp);
    std::printf("round-trip check: %s\n", max_err_bp < kInteriorTargetBp ? "PASS" : "FAIL");
    std::printf("%s\n", lv.violation_report().c_str());

    // ---- PDE vs MC cross-check at the ATM pillar
    const double t1 = 1.0, k1 = 100.0;
    const double sr1 = surface.implied_vol(0.0, t1);
    const double pde = price_european_pde(mkt, k1, t1, vol_fn, true, PdeSettings{}, sr1);
    McSettings mcs;
    mcs.n_paths = 20000;
    mcs.n_steps = 100;
    mcs.seed = 42;
    const McResult mc = price_european_mc(mkt, k1, t1, vol_fn, true, mcs);
    std::printf("\nPDE vs MC (K=100, T=1): PDE=%.4f  MC=%.4f +/- %.4f  |diff|=%.2f SE (%s)\n",
                pde, mc.price, mc.std_err, std::abs(pde - mc.price) / mc.std_err,
                mc.within(pde, 3.0) ? "OK" : "OUTSIDE 3 SE");

    // ---- American and barrier flavours under the same local vol
    const double amer =
        price_american_put_pde(mkt, 100.0, 1.0, vol_fn, PdeSettings{}, PsorSettings{}, sr1);
    const double eur_put = price_european_pde(mkt, 100.0, 1.0, vol_fn, false, PdeSettings{}, sr1);
    const McResult uo = price_up_out_call_mc(mkt, 100.0, 130.0, 1.0, vol_fn, mcs, true);
    std::printf("American put (PSOR)     : %.4f  (European %.4f, premium %+.4f)\n", amer,
                eur_put, amer - eur_put);
    std::printf("Up-and-out call B=130 MC: %.4f +/- %.4f (vanilla %.4f; barrier <= vanilla: %s)\n",
                uo.price, uo.std_err, pde, uo.price <= pde ? "OK" : "VIOLATED");

    std::printf("\ndone.\n");
    return 0;
}
