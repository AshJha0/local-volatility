//! Market description shared by all pricers.
//!
//! A single struct covers both asset classes:
//!
//! * **Equity**: `rate` is the risk-free rate `r` and `dividend` is the
//!   continuous dividend yield `q`.
//! * **FX (Garman-Kohlhagen)**: `rate` is the domestic rate `rd` and
//!   `dividend` plays the role of the foreign rate `rf`.  Under
//!   Garman-Kohlhagen the foreign rate enters every formula exactly where
//!   the dividend yield enters the equity formula, so one parametrisation
//!   serves both; use [`Market::fx`] to make the intent explicit.
//!
//! The forward is `F(T) = S * exp((rate - dividend) * T)` in both cases.

use crate::error::{invalid, Result};

/// Spot / rates container (validated at construction, hence immutable).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Market {
    spot: f64,
    rate: f64,
    dividend: f64,
}

impl Market {
    /// Build a market from spot `S0 > 0`, discount rate `r` and carry rate
    /// `q` (both may be negative; all values must be finite).
    pub fn new(spot: f64, rate: f64, dividend: f64) -> Result<Self> {
        for (name, v) in [("spot", spot), ("rate", rate), ("dividend", dividend)] {
            if !v.is_finite() {
                return Err(invalid(format!("Market.{name} must be finite, got {v}")));
            }
        }
        if spot <= 0.0 {
            return Err(invalid(format!("Market.spot must be > 0, got {spot}")));
        }
        Ok(Market { spot, rate, dividend })
    }

    /// Garman-Kohlhagen market: domestic rate `rd`, foreign rate `rf`.
    pub fn fx(spot: f64, rd: f64, rf: f64) -> Result<Self> {
        Market::new(spot, rd, rf)
    }

    /// Spot `S0`.
    pub fn spot(&self) -> f64 {
        self.spot
    }

    /// Discount rate `r` (domestic rate for FX).
    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// Carry rate `q` (dividend yield, or foreign rate for FX).
    pub fn dividend(&self) -> f64 {
        self.dividend
    }

    /// Forward `F(T) = S0 * exp((r - q) T)`.  Requires `expiry >= 0`.
    pub fn forward(&self, expiry: f64) -> Result<f64> {
        if !expiry.is_finite() || expiry < 0.0 {
            return Err(invalid(format!("expiry must be finite and >= 0, got {expiry}")));
        }
        Ok(self.spot * ((self.rate - self.dividend) * expiry).exp())
    }

    /// `ln F(T)` — convenient for log-moneyness lookups.  Requires
    /// `expiry >= 0` (validated, like [`Market::forward`], in every port).
    pub fn log_forward(&self, expiry: f64) -> Result<f64> {
        if !expiry.is_finite() || expiry < 0.0 {
            return Err(invalid(format!("expiry must be finite and >= 0, got {expiry}")));
        }
        Ok(self.spot.ln() + (self.rate - self.dividend) * expiry)
    }
}
