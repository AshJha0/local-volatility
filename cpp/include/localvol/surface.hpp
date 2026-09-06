#ifndef LOCALVOL_SURFACE_HPP
#define LOCALVOL_SURFACE_HPP

/// \file surface.hpp
/// \brief Implied-volatility surface in (log-moneyness, expiry)
///        total-variance form.
///
/// The surface is parametrised by *forward* log-moneyness k = ln(K / F(T))
/// and stores total implied variance w(k, T) = iv(k, T)^2 * T. Working in
/// (k, w) has three payoffs: the Dupire formula takes its cleanest
/// (Gatheral) form; calendar arbitrage is simply monotonicity of w in T at
/// fixed k; and the surface is independent of rates/dividends, so one CSV
/// serves equity and FX alike.
///
/// Interpolation ("bicubic-lite"):
///  - in k, per expiry pillar: a natural cubic spline through the node total
///    variances (C^2 smoothness is exactly what Dupire's d2w/dk2 needs);
///  - in T, at fixed k: linear interpolation of total variance between
///    pillars — the standard "no calendar arbitrage between pillars" rule.
///
/// Extrapolation: flat in k beyond the wings (query clamped to the node
/// range); w * T/T1 below the first pillar (flat forward variance); linear
/// in w beyond the last pillar with the slope floored at 0.
///
/// Data-quality conditions are reported, never raised: calendar arbitrage
/// at the nodes (calendar_violations()) and spline overshoot to w <= 0
/// between nodes (negative_w_count()) are counted at construction and
/// logged to stderr. The object is immutable after construction and safe
/// to share across threads.

#include <string>
#include <vector>

namespace localvol {

/// Natural cubic spline through (x_i, y_i).
///
/// Degenerate node counts degrade gracefully: 1 node -> constant,
/// 2 nodes -> linear. Evaluation clamps the query into [x_0, x_{n-1}]
/// (flat extrapolation), matching the surface's wing rule. The moment
/// system is solved with the same Thomas kernel the PDE uses, so every
/// language port reproduces it exactly.
class CubicSpline1D {
public:
    /// \throws std::invalid_argument on empty/mismatched arrays, non-finite
    ///         data, or non-strictly-increasing x.
    CubicSpline1D(std::vector<double> x, std::vector<double> y);

    /// Evaluate the spline at \p xq (clamped into the node range).
    /// \throws std::invalid_argument on a non-finite query.
    double operator()(double xq) const;

private:
    std::vector<double> x_;
    std::vector<double> y_;
    std::vector<double> m_;  ///< second-derivative "moments"; m[0]=m[n-1]=0.
};

/// Total-variance implied surface with the interpolation rules above.
class ImpliedVolSurface {
public:
    /// \param k_nodes  strictly increasing forward log-moneyness nodes (>= 1).
    /// \param expiries strictly increasing positive pillar expiries (>= 1).
    /// \param vols     implied vols, vols[j][i] for (expiries[j], k_nodes[i]),
    ///                 all strictly positive.
    /// \throws std::invalid_argument on shape/monotonicity/positivity
    ///         violations or non-finite input.
    ///
    /// Calendar arbitrage (total variance decreasing in T at a node) is
    /// *reported, not raised*: the count is exposed via
    /// calendar_violations() and a warning is written to stderr.
    /// A single-expiry surface warns that dw/dT rests on the
    /// flat-forward-variance assumption.
    ImpliedVolSurface(std::vector<double> k_nodes,
                      std::vector<double> expiries,
                      std::vector<std::vector<double>> vols);

    /// Load from a CSV with header exactly `T,k,iv` (one row per grid node,
    /// three columns; full rectangular grid, every (T,k) pair exactly once,
    /// order irrelevant). Parsing is locale-independent (std::from_chars).
    /// \throws std::invalid_argument on a missing/invalid header, empty
    ///         file, unreadable path, a short/long/non-numeric/non-finite
    ///         row, a duplicated (T,k) pair, or a non-rectangular grid.
    static ImpliedVolSurface from_csv(const std::string& path);

    /// Total implied variance w(k, T) under the documented rules
    /// (w(k, 0) = 0 exactly).
    /// \throws std::invalid_argument on non-finite k, or T non-finite / < 0.
    double total_variance(double k, double expiry) const;

    /// Implied vol sqrt(w(k,T)/T); at T == 0 the short-end limit
    /// sqrt(w(k, T1)/T1) is returned.
    double implied_vol(double k, double expiry) const;

    /// Number of node pairs where total variance decreases with expiry
    /// (calendar arbitrage), detected at construction.
    int calendar_violations() const { return calendar_violations_; }

    /// Number of probe points (15 per node interval, per pillar) where the
    /// natural spline overshoots to total variance <= 0 (implied_vol reads
    /// 0% there). Detected at construction and reported on stderr, never
    /// repaired; 0 for a well-behaved surface.
    long negative_w_count() const { return negative_w_count_; }

    /// True when only one pillar expiry was supplied.
    bool single_expiry() const { return expiries_.size() == 1; }

    const std::vector<double>& k_nodes() const { return k_nodes_; }
    const std::vector<double>& expiries() const { return expiries_; }
    double k_min() const { return k_nodes_.front(); }
    double k_max() const { return k_nodes_.back(); }

private:
    std::vector<double> k_nodes_;
    std::vector<double> expiries_;
    std::vector<std::vector<double>> vols_;
    std::vector<CubicSpline1D> splines_;  ///< one spline in k per expiry row.
    int calendar_violations_ = 0;
    long negative_w_count_ = 0;
};

}  // namespace localvol

#endif  // LOCALVOL_SURFACE_HPP
