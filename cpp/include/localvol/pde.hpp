#ifndef LOCALVOL_PDE_HPP
#define LOCALVOL_PDE_HPP

/// \file pde.hpp
/// \brief Log-spot finite-difference pricer: Crank-Nicolson with Rannacher
///        start; American puts via PSOR.
///
/// In x = ln S the backward pricing equation reads (tau = T - t):
///
///   dV/dtau = a d2V/dx2 + mu dV/dx - r V,  a = sigma^2/2,  mu = r - q - a
///
/// discretised on a uniform x-grid with central differences and marched in
/// tau with a theta-scheme; tridiagonal systems use the native Thomas kernel.
///
/// Why Rannacher: CN is unconditionally stable but only neutrally damped —
/// its amplification factor tends to -1 for high-frequency modes, so the
/// payoff kink excites persistent oscillations in price and (worse) gamma
/// near the strike ("dt too big vs dx" makes them visible even though the
/// scheme never blows up). The first dt-interval is therefore replaced by
/// two backward-Euler half-steps, whose amplification tends to 0, killing
/// precisely those modes; second-order accuracy is preserved (observed
/// convergence order ~ 2 on grid halving, tested in [1.5, 2.5]).
///
/// Mesh Peclet condition: central differencing of the drift keeps the
/// theta-scheme matrix an M-matrix only while |mu_i| h <= 2 a_i. Under a
/// flat vol on the default grid this holds; under local vol a node floored
/// at 1% with a few percent of carry violates it. Such nodes (and only
/// those) switch to first-order upwind for the first derivative, keeping
/// both off-diagonals >= 0 (no spurious oscillations / negative prices)
/// at the cost of O(|mu| h / 2) numerical diffusion there.
///
/// Grid rule (exact, ports must copy): with M = num_space even,
///   W = |ln(K/S0)| + nsd * sigma_ref * sqrt(T) + |r - q| * T,  h = 2W/M,
///   x_i = ln S0 + (i - M/2) h, i = 0..M
/// so ln S0 is exactly node M/2 and the price is read there — no
/// interpolation.
///
/// American exercise uses PSOR (projected SOR), chosen over penalty /
/// operator-splitting because it solves the discrete linear complementarity
/// problem to an explicit tolerance with no penalty parameter, and warm
/// starts keep the iteration count small. Non-convergence warns (stderr)
/// and proceeds, per the project error policy.

#include <functional>
#include <optional>
#include <vector>

#include "localvol/market.hpp"

namespace localvol {

/// sigma(k, t) callable: k is forward log-moneyness ln(level / F(t)).
using VolFn = std::function<double(double, double)>;

/// PDE grid parameters.
struct PdeSettings {
    int num_space = 200;  ///< spatial intervals M (even, >= 4); M+1 nodes.
    int num_time = 200;   ///< time intervals N (>= 1).
    double nsd = 6.0;     ///< grid half-width in standard deviations.
};

/// PSOR parameters for the American solver (also the golden settings).
struct PsorSettings {
    double omega = 1.5;    ///< relaxation, must lie in (0, 2).
    double tol = 1e-8;     ///< sup-norm update tolerance.
    int max_iter = 10000;  ///< iteration cap; hitting it warns and proceeds.
};

/// Price plus the final (t = 0) grid, for diagnostics such as gamma checks.
struct PdeResult {
    double price = 0.0;
    std::vector<double> x;       ///< log-spot nodes, size M+1.
    std::vector<double> values;  ///< option values at the nodes, size M+1.
};

/// European vanilla, flat volatility. Matches Black-Scholes to ~1e-3 rel
/// on the default 200x200 grid.
/// \throws std::invalid_argument on invalid market/strike/expiry/settings
///         or a negative / non-finite sigma.
double price_european_pde(const Market& market, double strike, double expiry,
                          double sigma, bool is_call = true,
                          const PdeSettings& settings = PdeSettings{});

/// European vanilla under a local-volatility function. The callable's
/// output is checked at every node and time step: NaN, inf or a negative
/// sigma throws std::invalid_argument.
/// \param sigma_ref reference vol for the grid width; defaults to
///        vol(0, T) — the ATM local vol at expiry. Must be finite and > 0.
double price_european_pde(const Market& market, double strike, double expiry,
                          const VolFn& vol, bool is_call = true,
                          const PdeSettings& settings = PdeSettings{},
                          std::optional<double> sigma_ref = std::nullopt);

/// As price_european_pde (flat vol) but returning the final grid.
PdeResult price_european_pde_grid(const Market& market, double strike,
                                  double expiry, double sigma,
                                  bool is_call = true,
                                  const PdeSettings& settings = PdeSettings{});

/// American put via PSOR on the Rannacher/CN scheme, flat volatility.
/// Obstacle max(K - S, 0) at every node and time level; boundaries
/// V = K - S_min (immediate exercise) and 0.
/// \throws std::invalid_argument additionally when strike <= 0 or the PSOR
///         parameters are out of range.
double price_american_put_pde(const Market& market, double strike,
                              double expiry, double sigma,
                              const PdeSettings& settings = PdeSettings{},
                              const PsorSettings& psor = PsorSettings{});

/// American put via PSOR under a local-volatility function.
double price_american_put_pde(const Market& market, double strike,
                              double expiry, const VolFn& vol,
                              const PdeSettings& settings = PdeSettings{},
                              const PsorSettings& psor = PsorSettings{},
                              std::optional<double> sigma_ref = std::nullopt);

}  // namespace localvol

#endif  // LOCALVOL_PDE_HPP
