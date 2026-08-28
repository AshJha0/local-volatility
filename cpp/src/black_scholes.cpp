#include "localvol/black_scholes.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace localvol {

namespace {

const double kSqrt2 = std::sqrt(2.0);
const double kInvSqrt2Pi = 1.0 / std::sqrt(2.0 * std::acos(-1.0));

void validate_common(double spot, double strike, double rate, double dividend, double sigma,
                     double expiry) {
    if (!std::isfinite(spot) || !std::isfinite(strike) || !std::isfinite(rate) ||
        !std::isfinite(dividend) || !std::isfinite(sigma) || !std::isfinite(expiry)) {
        throw std::invalid_argument("black_scholes: all inputs must be finite");
    }
    if (spot <= 0.0) throw std::invalid_argument("black_scholes: spot must be > 0");
    if (strike < 0.0) throw std::invalid_argument("black_scholes: strike must be >= 0");
    if (sigma < 0.0) throw std::invalid_argument("black_scholes: sigma must be >= 0");
    if (expiry < 0.0) throw std::invalid_argument("black_scholes: expiry must be >= 0");
}

double d1_of(double spot, double strike, double rate, double dividend, double sigma,
             double expiry) {
    const double st = sigma * std::sqrt(expiry);
    const double forward = spot * std::exp((rate - dividend) * expiry);
    return (std::log(forward / strike) + 0.5 * st * st) / st;
}

}  // namespace

double norm_cdf(double x) { return 0.5 * std::erfc(-x / kSqrt2); }

double norm_pdf(double x) { return kInvSqrt2Pi * std::exp(-0.5 * x * x); }

double bs_price(double spot, double strike, double rate, double dividend, double sigma,
                double expiry, bool is_call) {
    validate_common(spot, strike, rate, dividend, sigma, expiry);
    const double phi = is_call ? 1.0 : -1.0;
    if (expiry == 0.0) return std::max(phi * (spot - strike), 0.0);
    const double df_r = std::exp(-rate * expiry);
    const double df_q = std::exp(-dividend * expiry);
    const double forward = spot * std::exp((rate - dividend) * expiry);
    if (strike == 0.0) return is_call ? spot * df_q : 0.0;
    if (sigma == 0.0) return df_r * std::max(phi * (forward - strike), 0.0);
    const double st = sigma * std::sqrt(expiry);
    const double d1 = (std::log(forward / strike) + 0.5 * st * st) / st;
    const double d2 = d1 - st;
    return phi * (spot * df_q * norm_cdf(phi * d1) - strike * df_r * norm_cdf(phi * d2));
}

double bs_delta(double spot, double strike, double rate, double dividend, double sigma,
                double expiry, bool is_call) {
    validate_common(spot, strike, rate, dividend, sigma, expiry);
    if (expiry <= 0.0 || sigma <= 0.0 || strike <= 0.0) {
        throw std::invalid_argument("bs_delta requires expiry > 0, sigma > 0 and strike > 0");
    }
    const double phi = is_call ? 1.0 : -1.0;
    const double d1 = d1_of(spot, strike, rate, dividend, sigma, expiry);
    return phi * std::exp(-dividend * expiry) * norm_cdf(phi * d1);
}

double bs_gamma(double spot, double strike, double rate, double dividend, double sigma,
                double expiry) {
    validate_common(spot, strike, rate, dividend, sigma, expiry);
    if (expiry <= 0.0 || sigma <= 0.0 || strike <= 0.0) {
        throw std::invalid_argument("bs_gamma requires expiry > 0, sigma > 0 and strike > 0");
    }
    const double d1 = d1_of(spot, strike, rate, dividend, sigma, expiry);
    return std::exp(-dividend * expiry) * norm_pdf(d1) / (spot * sigma * std::sqrt(expiry));
}

double bs_vega(double spot, double strike, double rate, double dividend, double sigma,
               double expiry) {
    validate_common(spot, strike, rate, dividend, sigma, expiry);
    if (expiry <= 0.0 || sigma <= 0.0 || strike <= 0.0) {
        throw std::invalid_argument("bs_vega requires expiry > 0, sigma > 0 and strike > 0");
    }
    const double d1 = d1_of(spot, strike, rate, dividend, sigma, expiry);
    return spot * std::exp(-dividend * expiry) * norm_pdf(d1) * std::sqrt(expiry);
}

double implied_vol(double price, double spot, double strike, double rate, double dividend,
                   double expiry, bool is_call, double lo, double hi, int iterations) {
    validate_common(spot, strike, rate, dividend, 0.0, expiry);
    if (!std::isfinite(price)) throw std::invalid_argument("implied_vol: price must be finite");
    if (expiry <= 0.0) throw std::invalid_argument("implied_vol requires expiry > 0");
    if (strike <= 0.0) throw std::invalid_argument("implied_vol requires strike > 0");
    // Static no-arbitrage bounds: sigma=0 price below, discounted asset /
    // strike above (with tiny slack for round-off).
    const double lower = bs_price(spot, strike, rate, dividend, 0.0, expiry, is_call);
    const double upper =
        is_call ? spot * std::exp(-dividend * expiry) : strike * std::exp(-rate * expiry);
    const double eps = 1e-12 * std::max(1.0, spot);
    if (price < lower - eps || price > upper + eps) {
        std::ostringstream os;
        os << "implied_vol: price " << price << " outside no-arbitrage bounds [" << lower << ", "
           << upper << "]";
        throw std::invalid_argument(os.str());
    }
    // Exactly `iterations` halvings of a fixed bracket: deterministic and
    // monotone-safe (vega can be tiny in the wings); ~1e-10 vol accuracy.
    double a = lo, b = hi;
    for (int it = 0; it < iterations; ++it) {
        const double mid = 0.5 * (a + b);
        if (bs_price(spot, strike, rate, dividend, mid, expiry, is_call) < price) {
            a = mid;
        } else {
            b = mid;
        }
    }
    return 0.5 * (a + b);
}

}  // namespace localvol
