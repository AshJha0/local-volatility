#ifndef LOCALVOL_BLACK_SCHOLES_HPP
#define LOCALVOL_BLACK_SCHOLES_HPP

/// \file black_scholes.hpp
/// \brief Black-Scholes / Garman-Kohlhagen analytics.
///
/// All formulas are written in terms of (r, q); for FX pass r = rd and
/// q = rf (Garman-Kohlhagen). The normal CDF is computed via erfc for
/// accuracy in the far tails.

namespace localvol {

/// Standard normal CDF N(x) = 0.5 * erfc(-x / sqrt(2)) (tail-accurate).
double norm_cdf(double x);

/// Standard normal density.
double norm_pdf(double x);

/// European vanilla price under Black-Scholes / Garman-Kohlhagen.
///
/// Edge cases: expiry == 0 -> intrinsic max(phi(S-K), 0);
/// sigma == 0 -> discounted forward intrinsic exp(-rT) max(phi(F-K), 0);
/// strike == 0 -> call S exp(-qT), put 0.
/// \throws std::invalid_argument on non-finite input, spot <= 0,
///         strike < 0, sigma < 0 or expiry < 0.
double bs_price(double spot, double strike, double rate, double dividend,
                double sigma, double expiry, bool is_call = true);

/// Spot delta exp(-qT) N(phi d1) * phi (requires sigma, T, K > 0).
double bs_delta(double spot, double strike, double rate, double dividend,
                double sigma, double expiry, bool is_call = true);

/// Spot gamma exp(-qT) n(d1) / (S sigma sqrt(T)) (call == put).
double bs_gamma(double spot, double strike, double rate, double dividend,
                double sigma, double expiry);

/// Vega S exp(-qT) n(d1) sqrt(T) per unit of vol (call == put).
double bs_vega(double spot, double strike, double rate, double dividend,
               double sigma, double expiry);

/// Invert Black-Scholes by bisection on [lo, hi].
///
/// Bisection is mandated (not Newton) so every language port gets the same
/// ~1e-10-accurate result deterministically: exactly \p iterations halvings
/// of the fixed bracket, returning the midpoint. Prices outside the static
/// no-arbitrage bounds (with slack 1e-12 * max(1, spot)) are rejected.
/// The defaults ([1e-9, 5.0], 100 halvings) are the cross-language golden
/// contract; other values are validated (0 < lo < hi finite, iterations
/// >= 1) but leave that contract.
/// \throws std::invalid_argument on invalid inputs or out-of-bounds price.
double implied_vol(double price, double spot, double strike, double rate,
                   double dividend, double expiry, bool is_call = true,
                   double lo = 1e-9, double hi = 5.0, int iterations = 100);

}  // namespace localvol

#endif  // LOCALVOL_BLACK_SCHOLES_HPP
