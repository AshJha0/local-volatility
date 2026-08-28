#ifndef LOCALVOL_MARKET_HPP
#define LOCALVOL_MARKET_HPP

/// \file market.hpp
/// \brief Market description shared by all pricers (equity and FX).
///
/// One struct covers both asset classes:
///  - Equity: \c rate is the risk-free rate r, \c dividend the continuous
///    dividend yield q.
///  - FX (Garman-Kohlhagen): \c rate is the domestic rate rd and \c dividend
///    plays the role of the foreign rate rf — under GK the foreign rate
///    enters every formula exactly where the equity dividend yield does.
///
/// The forward is F(T) = S * exp((rate - dividend) * T) in both cases.

namespace localvol {

/// Spot / rates container. Immutable after construction; validated eagerly.
class Market {
public:
    /// \param spot     Spot price S0 (equity price or FX rate). Must be > 0.
    /// \param rate     Continuously-compounded discount rate r (rd for FX).
    ///                 May be negative.
    /// \param dividend Continuous dividend yield q (rf for FX). May be negative.
    /// \throws std::invalid_argument on non-finite input or spot <= 0.
    Market(double spot, double rate, double dividend = 0.0);

    /// Garman-Kohlhagen market: domestic rate rd, foreign rate rf.
    static Market fx(double spot, double rd, double rf);

    /// Forward F(T) = S0 * exp((r - q) T). Requires expiry >= 0 and finite.
    double forward(double expiry) const;

    /// ln F(T) — convenient for log-moneyness lookups.
    double log_forward(double expiry) const;

    double spot() const { return spot_; }
    double rate() const { return rate_; }
    double dividend() const { return dividend_; }

private:
    double spot_;
    double rate_;
    double dividend_;
};

}  // namespace localvol

#endif  // LOCALVOL_MARKET_HPP
