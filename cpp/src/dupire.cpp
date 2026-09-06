#include "localvol/dupire.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace localvol {

namespace {
constexpr double kWEps = 1e-12;  // guards 1/w and k/w at (near-)zero variance
}

DupireLocalVol::DupireLocalVol(const ImpliedVolSurface& surface) : surface_(surface) {}

double DupireLocalVol::clamp_and_count(double num, double den) const {
    // Clamp policy: no forward variance -> floor; butterfly arbitrage -> cap;
    // otherwise clamp the raw vol into [FLOOR, CAP], counting every hit.
    double vol;
    if (num <= 0.0) {
        floor_count_.fetch_add(1, std::memory_order_relaxed);
        vol = kFloor;
    } else if (den <= 0.0) {
        cap_count_.fetch_add(1, std::memory_order_relaxed);
        vol = kCap;
    } else {
        vol = std::sqrt(num / den);
        if (vol < kFloor) {
            floor_count_.fetch_add(1, std::memory_order_relaxed);
            vol = kFloor;
        } else if (vol > kCap) {
            cap_count_.fetch_add(1, std::memory_order_relaxed);
            vol = kCap;
        }
    }
    return vol * vol;
}

double DupireLocalVol::short_time_variance(double k) const {
    // sigma_loc(k, 0) = s / (1 - k s'/s), s = implied_vol(k, 0).
    const ImpliedVolSurface& srf = surface_;
    const double s = srf.implied_vol(k, 0.0);
    const double s_up = srf.implied_vol(k + kDk, 0.0);
    const double s_dn = srf.implied_vol(k - kDk, 0.0);
    const double dsdk = (s_up - s_dn) / (2.0 * kDk);
    const double ss = std::max(s, kWEps);  // guard s == 0 (overshoot to w <= 0)
    const double denom = 1.0 - k * dsdk / ss;
    // Same clamp policy as T > 0 with num = s^2 and den = denom |denom|:
    // sqrt(num/den) = s/denom when denom > 0, s == 0 floors, denom <= 0 caps.
    return clamp_and_count(s * s, denom * std::abs(denom));
}

double DupireLocalVol::local_variance(double k, double expiry) const {
    if (!std::isfinite(expiry) || expiry < 0.0) {
        throw std::invalid_argument("local_variance: expiry must be finite and >= 0");
    }
    if (!std::isfinite(k)) {
        throw std::invalid_argument("local_variance: k must be finite");
    }
    const ImpliedVolSurface& srf = surface_;
    // Clamp into [k_min + DK, k_max - DK] so that no central stencil
    // straddles the C^0 kink of the flat wing extrapolation.
    const double kmin = srf.k_min();
    const double kmax = srf.k_max();
    if (kmax - kmin > 2.0 * kDk) {
        k = std::clamp(k, kmin + kDk, kmax - kDk);
    }
    if (expiry == 0.0) return short_time_variance(k);

    const double w = srf.total_variance(k, expiry);
    const double w_up = srf.total_variance(k + kDk, expiry);
    const double w_dn = srf.total_variance(k - kDk, expiry);
    const double dwdk = (w_up - w_dn) / (2.0 * kDk);
    const double d2wdk2 = (w_up - 2.0 * w + w_dn) / (kDk * kDk);
    double dwdt;
    if (expiry > kDt) {
        dwdt = (srf.total_variance(k, expiry + kDt) - srf.total_variance(k, expiry - kDt)) /
               (2.0 * kDt);
    } else {
        // Forward difference near T = 0: never samples negative expiries.
        dwdt = (srf.total_variance(k, expiry + kDt) - w) / kDt;
    }

    const double ws = std::max(w, kWEps);
    const double denom = 1.0 - (k / ws) * dwdk +
                         0.25 * (-0.25 - 1.0 / ws + (k * k) / (ws * ws)) * dwdk * dwdk +
                         0.5 * d2wdk2;
    return clamp_and_count(dwdt, denom);
}

double DupireLocalVol::vol(double k, double expiry) const {
    return std::sqrt(local_variance(k, expiry));
}

void DupireLocalVol::reset_counters() const {
    floor_count_.store(0, std::memory_order_relaxed);
    cap_count_.store(0, std::memory_order_relaxed);
}

std::string DupireLocalVol::violation_report() const {
    std::ostringstream os;
    os << "local-vol clamps: floor(1%) hit " << floor_count() << "x, cap(500%) hit "
       << cap_count() << "x";
    return os.str();
}

}  // namespace localvol
