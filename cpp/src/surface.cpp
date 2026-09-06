#include "localvol/surface.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <limits>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>

#include "localvol/tridiag.hpp"

namespace localvol {

namespace {
constexpr double kCalTol = 1e-12;  // calendar-arbitrage detection tolerance
constexpr int kNegWSubdiv = 16;    // spline-overshoot scan: probes per interval (minus one)
}

// ---------------------------------------------------------------- spline

CubicSpline1D::CubicSpline1D(std::vector<double> x, std::vector<double> y)
    : x_(std::move(x)), y_(std::move(y)) {
    const std::size_t n = x_.size();
    if (n == 0 || y_.size() != n) {
        throw std::invalid_argument("spline: x and y must be equal-length non-empty arrays");
    }
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(x_[i]) || !std::isfinite(y_[i])) {
            throw std::invalid_argument("spline: non-finite node data");
        }
    }
    for (std::size_t i = 1; i < n; ++i) {
        if (!(x_[i] > x_[i - 1])) {
            throw std::invalid_argument("spline: x nodes must be strictly increasing");
        }
    }
    // Natural spline: m[0] = m[n-1] = 0; interior moments solve the classical
    // symmetric tridiagonal system, using the shared Thomas kernel so every
    // language port reproduces the coefficients exactly.
    m_.assign(n, 0.0);
    if (n > 2) {
        std::vector<double> h(n - 1);
        for (std::size_t i = 0; i + 1 < n; ++i) h[i] = x_[i + 1] - x_[i];
        const std::size_t p = n - 2;  // interior unknowns
        std::vector<double> sub(p - 1), diag(p), sup(p - 1), rhs(p);
        for (std::size_t i = 0; i < p; ++i) {
            diag[i] = (h[i] + h[i + 1]) / 3.0;
            rhs[i] = (y_[i + 2] - y_[i + 1]) / h[i + 1] - (y_[i + 1] - y_[i]) / h[i];
        }
        for (std::size_t i = 0; i + 1 < p; ++i) {
            sub[i] = h[i + 1] / 6.0;
            sup[i] = h[i + 1] / 6.0;
        }
        const std::vector<double> mm = thomas_solve(sub, diag, sup, rhs);
        for (std::size_t i = 0; i < p; ++i) m_[i + 1] = mm[i];
    }
}

double CubicSpline1D::operator()(double xq) const {
    if (!std::isfinite(xq)) {
        throw std::invalid_argument("spline: non-finite query");
    }
    const std::size_t n = x_.size();
    if (n == 1) return y_[0];
    // Clamp into the node range: flat extrapolation (constant-vol wings).
    const double q = std::min(std::max(xq, x_.front()), x_.back());
    // Interval i with x_i <= q <= x_{i+1} (i = n-2 at the right end).
    std::size_t i =
        static_cast<std::size_t>(std::upper_bound(x_.begin(), x_.end(), q) - x_.begin());
    i = (i == 0) ? 0 : i - 1;
    if (i > n - 2) i = n - 2;
    const double h = x_[i + 1] - x_[i];
    const double t = (q - x_[i]) / h;
    const double u = 1.0 - t;
    // s(x) = y_i u + y_{i+1} t + h^2/6 [ (u^3 - u) m_i + (t^3 - t) m_{i+1} ]
    return y_[i] * u + y_[i + 1] * t +
           (h * h / 6.0) * ((u * u * u - u) * m_[i] + (t * t * t - t) * m_[i + 1]);
}

// --------------------------------------------------------------- surface

ImpliedVolSurface::ImpliedVolSurface(std::vector<double> k_nodes,
                                     std::vector<double> expiries,
                                     std::vector<std::vector<double>> vols)
    : k_nodes_(std::move(k_nodes)), expiries_(std::move(expiries)), vols_(std::move(vols)) {
    const std::size_t m = k_nodes_.size();
    const std::size_t n = expiries_.size();
    if (m == 0) throw std::invalid_argument("surface: k_nodes must be non-empty");
    if (n == 0) throw std::invalid_argument("surface: expiries must be non-empty");
    if (vols_.size() != n) {
        throw std::invalid_argument("surface: vols must have one row per expiry");
    }
    for (const auto& row : vols_) {
        if (row.size() != m) {
            throw std::invalid_argument("surface: every vols row must have one entry per k node");
        }
    }
    for (double k : k_nodes_) {
        if (!std::isfinite(k)) throw std::invalid_argument("surface: k_nodes must be finite");
    }
    for (double t : expiries_) {
        if (!std::isfinite(t)) throw std::invalid_argument("surface: expiries must be finite");
        if (!(t > 0.0)) throw std::invalid_argument("surface: expiries must be strictly positive");
    }
    for (const auto& row : vols_) {
        for (double v : row) {
            if (!std::isfinite(v)) throw std::invalid_argument("surface: vols must be finite");
            if (!(v > 0.0)) {
                throw std::invalid_argument("surface: implied vols must be strictly positive");
            }
        }
    }
    for (std::size_t i = 1; i < m; ++i) {
        if (!(k_nodes_[i] > k_nodes_[i - 1])) {
            throw std::invalid_argument("surface: k_nodes must be strictly increasing");
        }
    }
    for (std::size_t j = 1; j < n; ++j) {
        if (!(expiries_[j] > expiries_[j - 1])) {
            throw std::invalid_argument("surface: expiries must be strictly increasing");
        }
    }

    // Total variance at the nodes: w = iv^2 * T.
    std::vector<std::vector<double>> w(n, std::vector<double>(m));
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < m; ++i) {
            w[j][i] = vols_[j][i] * vols_[j][i] * expiries_[j];
        }
    }

    // Calendar-arbitrage detection: report, don't crash.
    for (std::size_t j = 0; j + 1 < n; ++j) {
        for (std::size_t i = 0; i < m; ++i) {
            if (w[j + 1][i] < w[j][i] - kCalTol) ++calendar_violations_;
        }
    }
    if (calendar_violations_ > 0) {
        std::cerr << "localvol warning: calendar arbitrage — total variance decreases in T at "
                  << calendar_violations_ << " node(s)\n";
    }
    if (n == 1) {
        std::cerr << "localvol warning: single expiry pillar — dw/dT uses the "
                     "flat-forward-variance assumption\n";
    }

    splines_.reserve(n);
    for (std::size_t j = 0; j < n; ++j) {
        splines_.emplace_back(k_nodes_, w[j]);
    }

    // Spline-overshoot detection (report, don't crash): probe every node
    // interval of every pillar at kNegWSubdiv - 1 interior points and count
    // w <= 0 (implied_vol() would silently read 0% there).
    for (const CubicSpline1D& sp : splines_) {
        for (std::size_t i = 0; i + 1 < m; ++i) {
            const double h = k_nodes_[i + 1] - k_nodes_[i];
            for (int p = 1; p < kNegWSubdiv; ++p) {
                if (sp(k_nodes_[i] + h * (static_cast<double>(p) / kNegWSubdiv)) <= 0.0) {
                    ++negative_w_count_;
                }
            }
        }
    }
    if (negative_w_count_ > 0) {
        std::cerr << "localvol warning: spline overshoot — total variance <= 0 at "
                  << negative_w_count_ << " probe point(s) between nodes\n";
    }
}

namespace {

/// Locale-independent decimal parse of a trimmed CSV field (std::stod would
/// honour a comma-decimal process locale and read "0.25" as 0).
bool parse_double(std::string field, double& out) {
    const auto not_space = [](unsigned char c) { return !std::isspace(c); };
    field.erase(field.begin(), std::find_if(field.begin(), field.end(), not_space));
    field.erase(std::find_if(field.rbegin(), field.rend(), not_space).base(), field.end());
    if (field.empty()) return false;
    const char* first = field.data();
    const char* last = first + field.size();
    if (*first == '+') ++first;  // from_chars does not accept a leading '+'
    const auto res = std::from_chars(first, last, out);
    return res.ec == std::errc{} && res.ptr == last;
}

std::string trim(const std::string& s) {
    const auto not_space = [](unsigned char c) { return !std::isspace(c); };
    const auto b = std::find_if(s.begin(), s.end(), not_space);
    const auto e = std::find_if(s.rbegin(), s.rend(), not_space).base();
    return b < e ? std::string(b, e) : std::string();
}

}  // namespace

ImpliedVolSurface ImpliedVolSurface::from_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::invalid_argument(path + ": cannot open surface CSV");
    std::string line;
    if (!std::getline(in, line)) throw std::invalid_argument(path + ": empty surface CSV");
    // Strip a possible trailing CR (Windows line endings).
    if (!line.empty() && line.back() == '\r') line.pop_back();
    {
        std::istringstream hs(line);
        std::string c1, c2, c3, extra;
        const bool three = std::getline(hs, c1, ',') && std::getline(hs, c2, ',') &&
                           std::getline(hs, c3, ',') && !std::getline(hs, extra, ',');
        if (!three || trim(c1) != "T" || trim(c2) != "k" || trim(c3) != "iv") {
            throw std::invalid_argument(path + ": CSV must have header columns T,k,iv");
        }
    }
    struct Row {
        double t, k, iv;
    };
    std::vector<Row> rows;
    std::size_t lineno = 1;
    while (std::getline(in, line)) {
        ++lineno;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (trim(line).empty()) continue;
        std::istringstream ss(line);
        std::string f1, f2, f3, extra;
        const bool three = std::getline(ss, f1, ',') && std::getline(ss, f2, ',') &&
                           std::getline(ss, f3, ',') && !std::getline(ss, extra, ',');
        if (!three) {
            throw std::invalid_argument(path + ": line " + std::to_string(lineno) +
                                        ": expected 3 columns (T,k,iv) in '" + line + "'");
        }
        Row r{};
        if (!parse_double(f1, r.t) || !parse_double(f2, r.k) || !parse_double(f3, r.iv)) {
            throw std::invalid_argument(path + ": line " + std::to_string(lineno) +
                                        ": non-numeric CSV row '" + line + "'");
        }
        if (!std::isfinite(r.t) || !std::isfinite(r.k) || !std::isfinite(r.iv)) {
            throw std::invalid_argument(path + ": line " + std::to_string(lineno) +
                                        ": non-finite CSV value '" + line + "'");
        }
        rows.push_back(r);
    }
    if (rows.empty()) throw std::invalid_argument(path + ": empty surface CSV");

    // Distinct sorted axes; exact double equality is safe because identical
    // text parses to identical doubles.
    std::map<double, std::size_t> ti, ki;
    for (const Row& r : rows) {
        ti.emplace(r.t, 0);
        ki.emplace(r.k, 0);
    }
    std::vector<double> ts, ks;
    ts.reserve(ti.size());
    ks.reserve(ki.size());
    for (auto& kv : ti) {
        kv.second = ts.size();
        ts.push_back(kv.first);
    }
    for (auto& kv : ki) {
        kv.second = ks.size();
        ks.push_back(kv.first);
    }
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<std::vector<double>> vols(ts.size(), std::vector<double>(ks.size(), nan));
    std::vector<std::vector<char>> seen(ts.size(), std::vector<char>(ks.size(), 0));
    for (const Row& r : rows) {
        const std::size_t j = ti[r.t];
        const std::size_t i = ki[r.k];
        if (seen[j][i]) {
            std::ostringstream os;
            os << path << ": duplicate (T,k) row T=" << r.t << ", k=" << r.k;
            throw std::invalid_argument(os.str());
        }
        seen[j][i] = 1;
        vols[j][i] = r.iv;
    }
    for (const auto& row : seen) {
        for (char s : row) {
            if (!s) {
                throw std::invalid_argument(
                    path + ": surface grid is not rectangular (missing (T,k) pairs)");
            }
        }
    }
    return ImpliedVolSurface(std::move(ks), std::move(ts), std::move(vols));
}

double ImpliedVolSurface::total_variance(double k, double expiry) const {
    if (!std::isfinite(expiry) || expiry < 0.0) {
        throw std::invalid_argument("total_variance: expiry must be finite and >= 0");
    }
    if (!std::isfinite(k)) {
        throw std::invalid_argument("total_variance: k must be finite");
    }
    if (expiry == 0.0) return 0.0;
    const std::vector<double>& t = expiries_;
    const std::size_t n = t.size();
    // (Splines clamp k into [k_min, k_max] themselves: flat wings.)
    if (n == 1 || expiry <= t.front()) {
        // Flat forward variance: w scales proportionally with T. For a
        // single-expiry surface this rule applies on both sides of T1.
        return splines_[0](k) * (expiry / t.front());
    }
    if (expiry >= t.back()) {
        // Linear-in-T total variance beyond the last pillar, slope >= 0.
        const double w_last = splines_[n - 1](k);
        const double w_prev = splines_[n - 2](k);
        const double slope = (w_last - w_prev) / (t[n - 1] - t[n - 2]);
        return w_last + std::max(slope, 0.0) * (expiry - t.back());
    }
    // Interior: linear interpolation of total variance between pillars.
    const std::size_t j = static_cast<std::size_t>(
                              std::upper_bound(t.begin(), t.end(), expiry) - t.begin()) -
                          1;
    const double lam = (expiry - t[j]) / (t[j + 1] - t[j]);
    return (1.0 - lam) * splines_[j](k) + lam * splines_[j + 1](k);
}

double ImpliedVolSurface::implied_vol(double k, double expiry) const {
    if (!std::isfinite(expiry) || expiry < 0.0) {
        throw std::invalid_argument("implied_vol: expiry must be finite and >= 0");
    }
    if (expiry == 0.0) {
        // Short-end limit of the proportional rule: sqrt(w(k, T1) / T1).
        const double w1 = total_variance(k, expiries_.front());
        return std::sqrt(w1 / expiries_.front());
    }
    const double w = total_variance(k, expiry);
    return std::sqrt(std::max(w, 0.0) / expiry);
}

}  // namespace localvol
