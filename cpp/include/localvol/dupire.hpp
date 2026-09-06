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
/// 0 < T <= DT). These exact steps are part of the cross-language contract.
///
/// Stencil clamp at the quoted wings: the surface is flat in k beyond the
/// last quoted strikes, so w is only C^0 there (natural spline: w'' = 0 but
/// w' != 0 at the end node). A central stencil straddling that kink would
/// read d2w/dk2 ~ -w'/DK and cap the vol at 500% *at the quoted wing*. The
/// query is therefore clamped into [k_min + DK, k_max - DK] before
/// differencing (whenever the box is wider than 2 DK) and the clamped k is
/// used in every term: local vol is constant in k beyond k_max - DK and
/// continuous across the wing node.
///
/// Short-expiry limit: w(k, 0) = 0 exactly, so vol(k, 0) is defined as the
/// T -> 0+ limit (Berestycki-Busca-Florent 2002) with s(k) = implied_vol(k, 0):
///   sigma_loc(k, 0) = s / (1 - k s'/s),   s' by the same central DK step.
/// The T > 0 branch converges to it (flat forward variance below the first
/// pillar), so vol is continuous at T = 0.

#include <atomic>
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
///
/// Thread safety: vol() / local_variance() are const and safe to call
/// concurrently from any number of threads on one shared object (the
/// surface is immutable; the counters are std::atomic and updated with
/// relaxed increments, so their *total* is exact but a read taken while
/// other threads are still evaluating is only a snapshot). reset_counters()
/// is safe too, but racing it against evaluations makes the count
/// meaningless — reset between phases, not during one.
class DupireLocalVol {
public:
    static constexpr double kDk = 1.0e-3;    ///< FD step in log-moneyness.
    static constexpr double kDt = 1.0e-4;    ///< FD step in expiry (years).
    static constexpr double kFloor = 0.01;   ///< 1% local-vol floor.
    static constexpr double kCap = 5.0;      ///< 500% local-vol cap.

    /// The surface must outlive this object (a reference is kept, not a
    /// copy). Keep the surface in a named variable.
    explicit DupireLocalVol(const ImpliedVolSurface& surface);
    /// Binding a temporary is a compile error: `DupireLocalVol lv(
    /// ImpliedVolSurface::from_csv(path));` would leave a dangling
    /// reference and every later vol() call would be undefined behaviour.
    explicit DupireLocalVol(ImpliedVolSurface&&) = delete;

    // Holds a reference and atomic counters: not copyable/movable (share it
    // by reference or pointer instead).
    DupireLocalVol(const DupireLocalVol&) = delete;
    DupireLocalVol& operator=(const DupireLocalVol&) = delete;

    /// Clamped local variance sigma_loc^2(k, T).
    /// \throws std::invalid_argument on non-finite k, or T non-finite / < 0.
    double local_variance(double k, double expiry) const;

    /// Local volatility sigma_loc(k, T), clamped to [1%, 500%].
    double vol(double k, double expiry) const;

    long floor_count() const { return floor_count_.load(std::memory_order_relaxed); }
    long cap_count() const { return cap_count_.load(std::memory_order_relaxed); }

    /// Zero the floor/cap violation counters.
    void reset_counters() const;

    /// Human-readable clamp summary for demos/logs.
    std::string violation_report() const;

    const ImpliedVolSurface& surface() const { return surface_; }

private:
    /// Shared clamp policy: num <= 0 floors, else den <= 0 caps, else
    /// sqrt(num/den) clipped into [FLOOR, CAP]; every hit counted. Returns
    /// the clamped local *variance*.
    double clamp_and_count(double num, double den) const;
    /// Berestycki-Busca-Florent T -> 0+ limit at an already-clamped k.
    double short_time_variance(double k) const;

    const ImpliedVolSurface& surface_;
    // Counters are mutable + atomic so a const DupireLocalVol can be bound
    // as the sigma(k,t) callable from several threads at once while still
    // reporting clamp statistics (relaxed: counts only, no ordering needed).
    mutable std::atomic<long> floor_count_{0};
    mutable std::atomic<long> cap_count_{0};
};

}  // namespace localvol

#endif  // LOCALVOL_DUPIRE_HPP
