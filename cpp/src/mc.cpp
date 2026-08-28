#include "localvol/mc.hpp"

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace localvol {

namespace {

void validate_mc(const Market& /*market*/, double strike, double expiry, const McSettings& s) {
    if (!std::isfinite(strike) || strike < 0.0) {
        throw std::invalid_argument("mc: strike must be finite and >= 0");
    }
    if (!std::isfinite(expiry) || expiry <= 0.0) {
        throw std::invalid_argument("mc: expiry must be finite and > 0");
    }
    if (s.n_paths < 2) throw std::invalid_argument("mc: n_paths must be >= 2");
    if (s.antithetic && s.n_paths % 2 != 0) {
        throw std::invalid_argument("mc: antithetic sampling requires an even n_paths");
    }
    if (s.n_steps < 1) throw std::invalid_argument("mc: n_steps must be >= 1");
}

VolFn flat_vol_fn(double sigma) {
    if (!std::isfinite(sigma) || sigma <= 0.0) {
        throw std::invalid_argument("mc: flat vol must be finite and > 0");
    }
    return [sigma](double /*k*/, double /*t*/) { return sigma; };
}

/// Mean / standard error over samples; with antithetic on, the samples are
/// pair means (paired payoffs are negatively correlated, so per-path
/// deviations would overstate the error).
McResult estimate(const std::vector<double>& payoff_plus,
                  const std::vector<double>& payoff_minus, bool antithetic, double df) {
    std::vector<double> samples;
    samples.reserve(payoff_plus.size());
    if (antithetic) {
        for (std::size_t i = 0; i < payoff_plus.size(); ++i) {
            samples.push_back(0.5 * df * (payoff_plus[i] + payoff_minus[i]));
        }
    } else {
        for (double p : payoff_plus) samples.push_back(df * p);
    }
    const std::size_t n = samples.size();
    double mean = 0.0;
    for (double v : samples) mean += v;
    mean /= static_cast<double>(n);
    double ss = 0.0;
    for (double v : samples) ss += (v - mean) * (v - mean);
    const double sd = std::sqrt(ss / static_cast<double>(n - 1));  // ddof = 1
    return McResult{mean, sd / std::sqrt(static_cast<double>(n))};
}

McResult european_mc(const Market& market, double strike, double expiry, const VolFn& vol,
                     bool is_call, const McSettings& s) {
    validate_mc(market, strike, expiry, s);
    std::mt19937_64 rng(s.seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    const double dt = expiry / s.n_steps;
    const double sq = std::sqrt(dt);
    const double drift_rq = (market.rate() - market.dividend()) * dt;
    const double x0 = std::log(market.spot());
    const int n_base = s.antithetic ? s.n_paths / 2 : s.n_paths;
    const double phi = is_call ? 1.0 : -1.0;

    std::vector<double> pay_p(static_cast<std::size_t>(n_base));
    std::vector<double> pay_m(s.antithetic ? static_cast<std::size_t>(n_base) : 0);
    for (int p = 0; p < n_base; ++p) {
        double x = x0;
        double xa = x0;
        for (int n = 0; n < s.n_steps; ++n) {
            const double t = n * dt;
            const double lf = market.log_forward(t);
            const double z = normal(rng);
            // Start-of-step lookup at forward log-moneyness X_n - ln F(t_n);
            // the mirrored path recomputes sigma from its own state.
            const double sig = vol(x - lf, t);
            x += drift_rq - 0.5 * sig * sig * dt + sig * sq * z;
            if (s.antithetic) {
                const double siga = vol(xa - lf, t);
                xa += drift_rq - 0.5 * siga * siga * dt - siga * sq * z;
            }
        }
        pay_p[static_cast<std::size_t>(p)] = std::max(phi * (std::exp(x) - strike), 0.0);
        if (s.antithetic) {
            pay_m[static_cast<std::size_t>(p)] = std::max(phi * (std::exp(xa) - strike), 0.0);
        }
    }
    return estimate(pay_p, pay_m, s.antithetic, std::exp(-market.rate() * expiry));
}

McResult barrier_mc(const Market& market, double strike, double barrier, double expiry,
                    const VolFn& vol, const McSettings& s, bool brownian_bridge) {
    validate_mc(market, strike, expiry, s);
    if (!std::isfinite(barrier) || barrier <= 0.0) {
        throw std::invalid_argument("mc: barrier must be finite and > 0");
    }
    if (market.spot() >= barrier) return McResult{0.0, 0.0};  // born knocked out

    std::mt19937_64 rng(s.seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    const double dt = expiry / s.n_steps;
    const double sq = std::sqrt(dt);
    const double drift_rq = (market.rate() - market.dividend()) * dt;
    const double x0 = std::log(market.spot());
    const double b = std::log(barrier);
    const int n_base = s.antithetic ? s.n_paths / 2 : s.n_paths;

    std::vector<double> pay_p(static_cast<std::size_t>(n_base));
    std::vector<double> pay_m(s.antithetic ? static_cast<std::size_t>(n_base) : 0);
    for (int p = 0; p < n_base; ++p) {
        double x = x0, xa = x0;
        double w = 1.0, wa = 1.0;  // survival weights
        for (int n = 0; n < s.n_steps; ++n) {
            const double t = n * dt;
            const double lf = market.log_forward(t);
            const double z = normal(rng);
            {
                const double sig = vol(x - lf, t);
                const double x_new = x + drift_rq - 0.5 * sig * sig * dt + sig * sq * z;
                if (x_new >= b) {
                    w = 0.0;  // discrete knock-out
                } else if (brownian_bridge && w > 0.0) {
                    // Exact bridge crossing probability for x, x_new < b.
                    const double pc = std::exp(-2.0 * (b - x) * (b - x_new) / (sig * sig * dt));
                    w *= 1.0 - pc;
                }
                x = x_new;
            }
            if (s.antithetic) {
                const double sig = vol(xa - lf, t);
                const double x_new = xa + drift_rq - 0.5 * sig * sig * dt - sig * sq * z;
                if (x_new >= b) {
                    wa = 0.0;
                } else if (brownian_bridge && wa > 0.0) {
                    const double pc = std::exp(-2.0 * (b - xa) * (b - x_new) / (sig * sig * dt));
                    wa *= 1.0 - pc;
                }
                xa = x_new;
            }
        }
        pay_p[static_cast<std::size_t>(p)] = w * std::max(std::exp(x) - strike, 0.0);
        if (s.antithetic) {
            pay_m[static_cast<std::size_t>(p)] = wa * std::max(std::exp(xa) - strike, 0.0);
        }
    }
    return estimate(pay_p, pay_m, s.antithetic, std::exp(-market.rate() * expiry));
}

}  // namespace

bool McResult::within(double reference, double n_se) const {
    return std::abs(price - reference) <= n_se * std::max(std_err, 1e-300);
}

McResult price_european_mc(const Market& market, double strike, double expiry, double sigma,
                           bool is_call, const McSettings& settings) {
    return european_mc(market, strike, expiry, flat_vol_fn(sigma), is_call, settings);
}

McResult price_european_mc(const Market& market, double strike, double expiry, const VolFn& vol,
                           bool is_call, const McSettings& settings) {
    if (!vol) throw std::invalid_argument("mc: vol callable must be non-empty");
    return european_mc(market, strike, expiry, vol, is_call, settings);
}

McResult price_up_out_call_mc(const Market& market, double strike, double barrier, double expiry,
                              double sigma, const McSettings& settings, bool brownian_bridge) {
    return barrier_mc(market, strike, barrier, expiry, flat_vol_fn(sigma), settings,
                      brownian_bridge);
}

McResult price_up_out_call_mc(const Market& market, double strike, double barrier, double expiry,
                              const VolFn& vol, const McSettings& settings,
                              bool brownian_bridge) {
    if (!vol) throw std::invalid_argument("mc: vol callable must be non-empty");
    return barrier_mc(market, strike, barrier, expiry, vol, settings, brownian_bridge);
}

}  // namespace localvol
