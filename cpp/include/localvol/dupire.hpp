#ifndef LOCALVOL_DUPIRE_HPP
#define LOCALVOL_DUPIRE_HPP

/// \file dupire.hpp
/// \brief Dupire local volatility in total-variance (Gatheral) form.
///
/// Substituting the Black-Scholes representation of call prices in terms of
/// total implied variance w(k, T) into Dupire's formula yields
///
///   sigma_loc^2(k,T) = (dw/dT) /
///       ( 1 - (k/w) dw/dk
///           + 1/4 (-1/4 - 1/w + k^2/w^2) (dw/dk)^2
///           + 1/2 d2w/dk2 )
///
/// which is far better conditioned than differentiating near-degenerate
/// call prices. The numerator is the forward variance (positivity =
/// calendar no-arbitrage); the denominator is the density term (positivity
/// = butterfly no-arbitrage). A flat surface collapses the formula to
/// sigma_loc == implied vol identically — the golden-test anchor.
///
/// Derivatives are central finite differences on the interpolated surface
/// with fixed steps DK = 1e-3 and DT = 1e-4 (forward difference in T when
/// T <= DT). These exact steps are part of the cross-language contract.

#include <string>

#include "localvol/surface.hpp"

namespace localvol {

/// Local-volatility function derived from an ImpliedVolSurface.
///
/// vol() is the sigma(k, t) callable expected by the PDE and Monte Carlo
/// engines; its k argument is always forward log-moneyness ln(level/F(t)).
/// The raw local vol is clamped into [FLOOR, CAP] = [1%, 500%]; every clamp
/// is *counted* (floor_count / cap_count), never raised, so callers can
/// report surface quality instead of crashing mid-pricing.
class DupireLocalVol {
public:
    static constexpr double kDk = 1.0e-3;    ///< FD step in log-moneyness.
    static constexpr double kDt = 1.0e-4;    ///< FD step in expiry (years).
    static constexpr double kFloor = 0.01;   ///< 1% local-vol floor.
    static constexpr double kCap = 5.0;      ///< 500% local-vol cap.

    /// The surface must outlive this object (a reference is kept).
    explicit DupireLocalVol(const ImpliedVolSurface& surface);

    /// Clamped local variance sigma_loc^2(k, T).
    /// \throws std::invalid_argument on non-finite k, or T non-finite / < 0.
    double local_variance(double k, double expiry) const;

    /// Local volatility sigma_loc(k, T), clamped to [1%, 500%].
    double vol(double k, double expiry) const;

    long floor_count() const { return floor_count_; }
    long cap_count() const { return cap_count_; }

    /// Zero the floor/cap violation counters.
    void reset_counters() const;

    /// Human-readable clamp summary for demos/logs.
    std::string violation_report() const;

    const ImpliedVolSurface& surface() const { return surface_; }

private:
    const ImpliedVolSurface& surface_;
    // Counters are mutable so a const DupireLocalVol can be bound as the
    // sigma(k,t) callable while still reporting clamp statistics.
    mutable long floor_count_ = 0;
    mutable long cap_count_ = 0;
};

}  // namespace localvol

#endif  // LOCALVOL_DUPIRE_HPP
