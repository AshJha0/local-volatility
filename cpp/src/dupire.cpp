#include "localvol/dupire.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace localvol {

namespace {
constexpr double kWEps = 1e-12;  // guards 1/w and k/w at (near-)zero variance
}

DupireLocalVol::DupireLocalVol(const ImpliedVolSurface& surface) : surface_(surface) {}

double DupireLocalVol::local_variance(double k, double expiry) const {
    if (!std::isfinite(expiry) || expiry < 0.0) {
        throw std::invalid_argument("local_variance: expiry must be finite and >= 0");
    }
    if (!std::isfinite(k)) {
        throw std::invalid_argument("local_variance: k must be finite");
    }
    const ImpliedVolSurface& srf = surface_;

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

    // Clamp policy: no forward variance -> floor; butterfly-arb wing -> cap;
    // otherwise clamp the raw vol into [FLOOR, CAP], counting every hit.
    double vol;
    if (dwdt <= 0.0) {
        ++floor_count_;
        vol = kFloor;
    } else if (denom <= 0.0) {
        ++cap_count_;
        vol = kCap;
    } else {
        vol = std::sqrt(dwdt / denom);
        if (vol < kFloor) {
            ++floor_count_;
            vol = kFloor;
        } else if (vol > kCap) {
            ++cap_count_;
            vol = kCap;
        }
    }
    return vol * vol;
}

double DupireLocalVol::vol(double k, double expiry) const {
    return std::sqrt(local_variance(k, expiry));
}

void DupireLocalVol::reset_counters() const {
    floor_count_ = 0;
    cap_count_ = 0;
}

std::string DupireLocalVol::violation_report() const {
    std::ostringstream os;
    os << "local-vol clamps: floor(1%) hit " << floor_count_ << "x, cap(500%) hit " << cap_count_
       << "x";
    return os.str();
}

}  // namespace localvol
