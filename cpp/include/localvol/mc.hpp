#ifndef LOCALVOL_MC_HPP
#define LOCALVOL_MC_HPP

/// \file mc.hpp
/// \brief Local-volatility Monte Carlo: log-Euler scheme with antithetic
///        variates; up-and-out barrier with Brownian-bridge correction.
///
/// Simulate X = ln S on a uniform grid t_n = n dt, dt = T / N:
///
///   X_{n+1} = X_n + (r - q - sigma_n^2/2) dt + sigma_n sqrt(dt) Z_n
///
/// with sigma_n = sigma(X_n - ln F(t_n), t_n) looked up at the *start* of
/// the step — the same forward log-moneyness coordinates the Dupire surface
/// uses, so PDE and MC discretise the identical diffusion. Log-Euler keeps
/// S > 0 exactly.
///
/// Antithetic variates: paths come in pairs driven by +Z and -Z (each
/// mirrored path recomputes its own sigma from its own state). The
/// estimator averages each pair first; the standard error is computed over
/// the n_paths/2 pair means — raw per-path deviations would overstate it
/// because paired payoffs are negatively correlated.
///
/// Barrier: discrete monitoring kills a path once a step ends at or above
/// ln B; the optional Brownian-bridge correction additionally multiplies a
/// surviving step's weight by 1 - exp(-2 (b-X_n)(b-X_{n+1}) / (sigma_n^2 dt))
/// — the exact bridge crossing probability — removing the O(sqrt(dt))
/// discrete-monitoring bias with no extra random numbers. The bridged price
/// is <= the discrete one; both are <= the vanilla.

#include <cstdint>

#include "localvol/market.hpp"
#include "localvol/pde.hpp"  // VolFn

namespace localvol {

/// Monte Carlo estimate with its standard error.
struct McResult {
    double price = 0.0;
    double std_err = 0.0;

    /// True when \p reference lies inside \p n_se standard errors.
    bool within(double reference, double n_se = 3.0) const;
};

/// Simulation parameters. The RNG is std::mt19937_64 driving
/// std::normal_distribution<double>. The normal transform is
/// implementation-defined (libstdc++, libc++ and MSVC produce different
/// streams for the same seed), so runs are deterministic for a fixed
/// toolchain only; MC golden comparisons are statistical (tolerances sized
/// at 4 SE) and each language fixes its own seed.
struct McSettings {
    int n_paths = 20000;      ///< total paths: >= 4 and even when antithetic, else >= 2.
    int n_steps = 100;        ///< Euler steps (>= 1).
    std::uint64_t seed = 42;  ///< fixed seed — deterministic runs.
    bool antithetic = true;   ///< +Z/-Z pairing (requires even n_paths >= 4).
};

/// European vanilla by log-Euler MC under a flat volatility (> 0).
/// \throws std::invalid_argument on invalid market/strike/expiry/settings.
McResult price_european_mc(const Market& market, double strike, double expiry,
                           double sigma, bool is_call = true,
                           const McSettings& settings = McSettings{});

/// European vanilla by log-Euler MC under a sigma(k, t) callable. The
/// callable's output is checked on every lookup: NaN, inf or a negative
/// sigma throws std::invalid_argument instead of propagating into the
/// estimate (sigma == 0 is allowed: a deterministic step).
McResult price_european_mc(const Market& market, double strike, double expiry,
                           const VolFn& vol, bool is_call = true,
                           const McSettings& settings = McSettings{});

/// Up-and-out call under flat vol; \p brownian_bridge selects the bridge
/// correction (default on — the golden configuration). Requires S0 < barrier,
/// otherwise the option is born knocked out and (0, 0) is returned.
McResult price_up_out_call_mc(const Market& market, double strike,
                              double barrier, double expiry, double sigma,
                              const McSettings& settings = McSettings{},
                              bool brownian_bridge = true);

/// Up-and-out call under a sigma(k, t) callable.
McResult price_up_out_call_mc(const Market& market, double strike,
                              double barrier, double expiry, const VolFn& vol,
                              const McSettings& settings = McSettings{},
                              bool brownian_bridge = true);

}  // namespace localvol

#endif  // LOCALVOL_MC_HPP
