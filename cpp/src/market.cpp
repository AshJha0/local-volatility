#include "localvol/market.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace localvol {

Market::Market(double spot, double rate, double dividend)
    : spot_(spot), rate_(rate), dividend_(dividend) {
    if (!std::isfinite(spot_) || !std::isfinite(rate_) || !std::isfinite(dividend_)) {
        throw std::invalid_argument("Market: spot, rate and dividend must be finite");
    }
    if (spot_ <= 0.0) {
        throw std::invalid_argument("Market: spot must be > 0, got " + std::to_string(spot_));
    }
}

Market Market::fx(double spot, double rd, double rf) { return Market(spot, rd, rf); }

double Market::forward(double expiry) const {
    if (!std::isfinite(expiry) || expiry < 0.0) {
        throw std::invalid_argument("Market::forward: expiry must be finite and >= 0");
    }
    return spot_ * std::exp((rate_ - dividend_) * expiry);
}

double Market::log_forward(double expiry) const {
    if (!std::isfinite(expiry) || expiry < 0.0) {
        throw std::invalid_argument("Market::log_forward: expiry must be finite and >= 0");
    }
    return std::log(spot_) + (rate_ - dividend_) * expiry;
}

}  // namespace localvol
