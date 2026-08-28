#include "localvol/pde.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "localvol/black_scholes.hpp"
#include "localvol/tridiag.hpp"

namespace localvol {

namespace {

void validate(const Market& /*market*/, double strike, double expiry, const PdeSettings& s) {
    if (!std::isfinite(strike) || strike < 0.0) {
        throw std::invalid_argument("pde: strike must be finite and >= 0");
    }
    if (!std::isfinite(expiry) || expiry < 0.0) {
        throw std::invalid_argument("pde: expiry must be finite and >= 0");
    }
    if (s.num_space < 4 || s.num_space % 2 != 0) {
        throw std::invalid_argument("pde: num_space must be an even integer >= 4");
    }
    if (s.num_time < 1) {
        throw std::invalid_argument("pde: num_time must be >= 1");
    }
    if (!std::isfinite(s.nsd) || s.nsd <= 0.0) {
        throw std::invalid_argument("pde: nsd must be finite and > 0");
    }
}

/// Resolved volatility input: either a flat sigma or a callable, plus the
/// grid-width reference vol.
struct ResolvedVol {
    bool flat = true;
    double sigma = 0.0;        // valid when flat
    const VolFn* fn = nullptr; // valid when !flat
    double sigma_ref = 0.0;
};

ResolvedVol resolve_flat(double sigma) {
    if (!std::isfinite(sigma) || sigma < 0.0) {
        throw std::invalid_argument("pde: flat vol must be finite and >= 0");
    }
    ResolvedVol rv;
    rv.flat = true;
    rv.sigma = sigma;
    rv.sigma_ref = sigma > 0.0 ? sigma : 1.0;
    return rv;
}

ResolvedVol resolve_fn(const VolFn& vol, double expiry, std::optional<double> sigma_ref) {
    if (!vol) throw std::invalid_argument("pde: vol callable must be non-empty");
    ResolvedVol rv;
    rv.flat = false;
    rv.fn = &vol;
    // Default reference vol: the ATM local vol at expiry.
    rv.sigma_ref = sigma_ref.has_value() ? *sigma_ref : vol(0.0, expiry);
    if (!std::isfinite(rv.sigma_ref) || rv.sigma_ref <= 0.0) {
        throw std::invalid_argument("pde: sigma_ref must be finite and > 0");
    }
    return rv;
}

/// Uniform log-spot grid with ln(S0) exactly at node M/2 (see header).
std::pair<std::vector<double>, double> build_grid(const Market& market, double strike,
                                                  double expiry, double sigma_ref,
                                                  int num_space, double nsd) {
    const double x0 = std::log(market.spot());
    const double half_width = std::abs(std::log(strike / market.spot())) +
                              nsd * sigma_ref * std::sqrt(expiry) +
                              std::abs(market.rate() - market.dividend()) * expiry;
    const double h = 2.0 * half_width / num_space;
    std::vector<double> x(static_cast<std::size_t>(num_space) + 1);
    for (int i = 0; i <= num_space; ++i) {
        x[static_cast<std::size_t>(i)] = x0 + (i - num_space / 2.0) * h;
    }
    return {std::move(x), h};
}

/// Rannacher schedule: (theta, dts) sub-steps summing to expiry. The first
/// dt-interval is two backward-Euler half-steps; the remaining num_time - 1
/// intervals are single Crank-Nicolson steps.
std::vector<std::pair<double, double>> time_steps(double expiry, int num_time) {
    const double dt = expiry / num_time;
    std::vector<std::pair<double, double>> steps;
    steps.reserve(static_cast<std::size_t>(num_time) + 1);
    steps.emplace_back(1.0, 0.5 * dt);
    steps.emplace_back(1.0, 0.5 * dt);
    for (int n = 0; n < num_time - 1; ++n) steps.emplace_back(0.5, dt);
    return steps;
}

/// Interior-node theta-scheme coefficients for one sub-step.
struct StepCoeffs {
    std::vector<double> lower, center, upper;  // size M-1 (interior nodes)
};

StepCoeffs step_coeffs(const std::vector<double>& sigma_nodes, double r, double q, double h) {
    const std::size_t n = sigma_nodes.size();
    StepCoeffs c;
    c.lower.resize(n);
    c.center.resize(n);
    c.upper.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double a = 0.5 * sigma_nodes[i] * sigma_nodes[i];  // diffusion
        const double mu = (r - q) - a;                           // log-spot drift
        const double alpha = a / (h * h);
        const double beta = mu / (2.0 * h);
        c.lower[i] = alpha - beta;   // coefficient of V_{i-1}
        c.upper[i] = alpha + beta;   // coefficient of V_{i+1}
        c.center[i] = -2.0 * alpha - r;  // coefficient of V_i
    }
    return c;
}

/// One theta-scheme step (interior solve by Thomas); returns the new level,
/// boundaries set to their Dirichlet values at the new tau.
std::vector<double> theta_step(const std::vector<double>& v,
                               const std::vector<double>& sigma_nodes, double r, double q,
                               double h, double dts, double theta, double v0_new,
                               double vm_new) {
    const std::size_t n = sigma_nodes.size();  // interior nodes, = v.size() - 2
    const StepCoeffs c = step_coeffs(sigma_nodes, r, q, h);

    std::vector<double> rhs(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double lv = c.lower[i] * v[i] + c.center[i] * v[i + 1] + c.upper[i] * v[i + 2];
        rhs[i] = v[i + 1] + (1.0 - theta) * dts * lv;
    }
    std::vector<double> sub(n - 1), diag(n), sup(n - 1);
    for (std::size_t i = 0; i < n; ++i) diag[i] = 1.0 - theta * dts * c.center[i];
    for (std::size_t i = 0; i + 1 < n; ++i) {
        sub[i] = -theta * dts * c.lower[i + 1];
        sup[i] = -theta * dts * c.upper[i];
    }
    // Fold the new-level Dirichlet boundaries into the first/last rows.
    rhs[0] += theta * dts * c.lower[0] * v0_new;
    rhs[n - 1] += theta * dts * c.upper[n - 1] * vm_new;

    const std::vector<double> xin = thomas_solve(sub, diag, sup, rhs);
    std::vector<double> out(v.size());
    out.front() = v0_new;
    out.back() = vm_new;
    std::copy(xin.begin(), xin.end(), out.begin() + 1);
    return out;
}

/// One theta-scheme step solved as a linear complementarity problem by
/// projected SOR (obstacle = immediate exercise value).
std::vector<double> psor_step(const std::vector<double>& v,
                              const std::vector<double>& sigma_nodes, double r, double q,
                              double h, double dts, double theta, double v0_new, double vm_new,
                              const std::vector<double>& obstacle, const PsorSettings& psor) {
    const std::size_t n = sigma_nodes.size();
    const StepCoeffs c = step_coeffs(sigma_nodes, r, q, h);

    std::vector<double> rhs(n), sub(n), diag(n), sup(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double lv = c.lower[i] * v[i] + c.center[i] * v[i + 1] + c.upper[i] * v[i + 2];
        rhs[i] = v[i + 1] + (1.0 - theta) * dts * lv;
        sub[i] = -theta * dts * c.lower[i];   // multiplies x[i-1] in row i
        diag[i] = 1.0 - theta * dts * c.center[i];
        sup[i] = -theta * dts * c.upper[i];   // multiplies x[i+1] in row i
    }
    rhs[0] += theta * dts * c.lower[0] * v0_new;
    rhs[n - 1] += theta * dts * c.upper[n - 1] * vm_new;

    // Warm start from the previous level projected onto the obstacle
    // (feasible), which keeps PSOR iteration counts small.
    std::vector<double> x(n);
    for (std::size_t i = 0; i < n; ++i) x[i] = std::max(v[i + 1], obstacle[i + 1]);

    bool converged = false;
    double err = 0.0;
    for (int it = 0; it < psor.max_iter; ++it) {
        err = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            double acc = rhs[i];
            if (i > 0) acc -= sub[i] * x[i - 1];
            if (i + 1 < n) acc -= sup[i] * x[i + 1];
            const double gs = acc / diag[i];
            const double xn = std::max(obstacle[i + 1], x[i] + psor.omega * (gs - x[i]));
            err = std::max(err, std::abs(xn - x[i]));
            x[i] = xn;
        }
        if (err < psor.tol) {
            converged = true;
            break;
        }
    }
    if (!converged) {
        std::cerr << "localvol warning: PSOR did not converge to " << psor.tol << " within "
                  << psor.max_iter << " iterations (last update " << err
                  << "); continuing with current iterate\n";
    }

    std::vector<double> out(v.size());
    out.front() = v0_new;
    out.back() = vm_new;
    std::copy(x.begin(), x.end(), out.begin() + 1);
    return out;
}

/// Volatility per interior node for the sub-step from tau to tau + dts,
/// frozen at the midpoint calendar time t_mid = T - tau - dts/2 and looked
/// up at forward log-moneyness x_i - ln F(t_mid).
std::vector<double> sigma_at_nodes(const ResolvedVol& rv, const Market& market,
                                   const std::vector<double>& x, double t_mid) {
    const std::size_t n = x.size() - 2;
    std::vector<double> sig(n);
    if (rv.flat) {
        std::fill(sig.begin(), sig.end(), rv.sigma);
    } else {
        const double lf = market.log_forward(t_mid);
        for (std::size_t i = 0; i < n; ++i) sig[i] = (*rv.fn)(x[i + 1] - lf, t_mid);
    }
    return sig;
}

PdeResult european_engine(const Market& market, double strike, double expiry,
                          const ResolvedVol& rv, bool is_call, const PdeSettings& s) {
    const double r = market.rate();
    const double q = market.dividend();
    const double phi = is_call ? 1.0 : -1.0;
    auto [x, h] = build_grid(market, strike, expiry, rv.sigma_ref, s.num_space, s.nsd);
    const std::size_t np = x.size();
    std::vector<double> sn(np);
    for (std::size_t i = 0; i < np; ++i) sn[i] = std::exp(x[i]);
    std::vector<double> v(np);
    for (std::size_t i = 0; i < np; ++i) v[i] = std::max(phi * (sn[i] - strike), 0.0);

    double tau = 0.0;
    for (const auto& [theta, dts] : time_steps(expiry, s.num_time)) {
        const double t_mid = expiry - tau - 0.5 * dts;
        const std::vector<double> sig = sigma_at_nodes(rv, market, x, t_mid);
        const double tau_new = tau + dts;
        // Dirichlet boundaries: discounted asymptotics at the new tau.
        double v0_new, vm_new;
        if (is_call) {
            v0_new = 0.0;
            vm_new = std::max(sn.back() * std::exp(-q * tau_new) - strike * std::exp(-r * tau_new),
                              0.0);
        } else {
            v0_new = std::max(strike * std::exp(-r * tau_new) - sn.front() * std::exp(-q * tau_new),
                              0.0);
            vm_new = 0.0;
        }
        v = theta_step(v, sig, r, q, h, dts, theta, v0_new, vm_new);
        tau = tau_new;
    }

    PdeResult res;
    res.price = v[static_cast<std::size_t>(s.num_space) / 2];  // ln S0 is exactly the middle node
    res.x = std::move(x);
    res.values = std::move(v);
    return res;
}

double american_engine(const Market& market, double strike, double expiry,
                       const ResolvedVol& rv, const PdeSettings& s, const PsorSettings& psor) {
    const double r = market.rate();
    const double q = market.dividend();
    auto [x, h] = build_grid(market, strike, expiry, rv.sigma_ref, s.num_space, s.nsd);
    const std::size_t np = x.size();
    std::vector<double> obstacle(np);
    for (std::size_t i = 0; i < np; ++i) obstacle[i] = std::max(strike - std::exp(x[i]), 0.0);
    std::vector<double> v = obstacle;

    double tau = 0.0;
    for (const auto& [theta, dts] : time_steps(expiry, s.num_time)) {
        const double t_mid = expiry - tau - 0.5 * dts;
        const std::vector<double> sig = sigma_at_nodes(rv, market, x, t_mid);
        // Boundaries: deep ITM immediate-exercise value below, 0 above.
        v = psor_step(v, sig, r, q, h, dts, theta, obstacle.front(), 0.0, obstacle, psor);
        tau += dts;
    }
    return v[static_cast<std::size_t>(s.num_space) / 2];
}

void validate_psor(const PsorSettings& psor) {
    if (!(psor.omega > 0.0 && psor.omega < 2.0)) {
        throw std::invalid_argument("pde: omega must lie in (0, 2)");
    }
    if (!(psor.tol > 0.0) || psor.max_iter < 1) {
        throw std::invalid_argument("pde: tol must be > 0 and max_iter >= 1");
    }
}

/// sigma = 0 American put: optimal exercise is deterministic; maximise the
/// discounted deterministic payoff over 2001 equidistant exercise dates.
double american_sigma0(const Market& market, double strike, double expiry) {
    const int n = 2000;
    double best = 0.0;
    for (int i = 0; i <= n; ++i) {
        const double t = expiry * i / n;
        const double s_t = market.spot() * std::exp((market.rate() - market.dividend()) * t);
        best = std::max(best, std::exp(-market.rate() * t) * std::max(strike - s_t, 0.0));
    }
    return best;
}

}  // namespace

double price_european_pde(const Market& market, double strike, double expiry, double sigma,
                          bool is_call, const PdeSettings& settings) {
    return price_european_pde_grid(market, strike, expiry, sigma, is_call, settings).price;
}

PdeResult price_european_pde_grid(const Market& market, double strike, double expiry,
                                  double sigma, bool is_call, const PdeSettings& settings) {
    validate(market, strike, expiry, settings);
    const double phi = is_call ? 1.0 : -1.0;
    if (expiry == 0.0) {
        return PdeResult{std::max(phi * (market.spot() - strike), 0.0), {}, {}};
    }
    const ResolvedVol rv = resolve_flat(sigma);
    if (rv.sigma == 0.0) {
        // Deterministic limit: the PDE degenerates to pure discounting.
        return PdeResult{bs_price(market.spot(), strike, market.rate(), market.dividend(), 0.0,
                                  expiry, is_call),
                         {},
                         {}};
    }
    if (strike == 0.0) {
        // Degenerate payoff: the call pays S_T, the put pays 0.
        return PdeResult{
            is_call ? market.spot() * std::exp(-market.dividend() * expiry) : 0.0, {}, {}};
    }
    return european_engine(market, strike, expiry, rv, is_call, settings);
}

double price_european_pde(const Market& market, double strike, double expiry, const VolFn& vol,
                          bool is_call, const PdeSettings& settings,
                          std::optional<double> sigma_ref) {
    validate(market, strike, expiry, settings);
    const double phi = is_call ? 1.0 : -1.0;
    if (expiry == 0.0) return std::max(phi * (market.spot() - strike), 0.0);
    const ResolvedVol rv = resolve_fn(vol, expiry, sigma_ref);
    if (strike == 0.0) {
        return is_call ? market.spot() * std::exp(-market.dividend() * expiry) : 0.0;
    }
    return european_engine(market, strike, expiry, rv, is_call, settings).price;
}

double price_american_put_pde(const Market& market, double strike, double expiry, double sigma,
                              const PdeSettings& settings, const PsorSettings& psor) {
    validate(market, strike, expiry, settings);
    validate_psor(psor);
    if (strike <= 0.0) throw std::invalid_argument("pde: American put requires strike > 0");
    if (expiry == 0.0) return std::max(strike - market.spot(), 0.0);
    const ResolvedVol rv = resolve_flat(sigma);
    if (rv.sigma == 0.0) return american_sigma0(market, strike, expiry);
    return american_engine(market, strike, expiry, rv, settings, psor);
}

double price_american_put_pde(const Market& market, double strike, double expiry,
                              const VolFn& vol, const PdeSettings& settings,
                              const PsorSettings& psor, std::optional<double> sigma_ref) {
    validate(market, strike, expiry, settings);
    validate_psor(psor);
    if (strike <= 0.0) throw std::invalid_argument("pde: American put requires strike > 0");
    if (expiry == 0.0) return std::max(strike - market.spot(), 0.0);
    const ResolvedVol rv = resolve_fn(vol, expiry, sigma_ref);
    return american_engine(market, strike, expiry, rv, settings, psor);
}

}  // namespace localvol
