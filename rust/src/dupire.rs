//! Dupire local volatility in total-variance (Gatheral) form.
//!
//! Substituting the Black-Scholes representation of call prices in terms of
//! total implied variance `w(k, T)` (with `k = ln(K / F(T))` the forward
//! log-moneyness) into Dupire's formula yields Gatheral's equivalent form,
//! far better conditioned because it differentiates the smooth surface `w`
//! instead of near-degenerate call prices:
//!
//! ```text
//! sigma_loc^2(k,T) = (dw/dT) /
//!     ( 1 - (k/w) dw/dk
//!         + 1/4 (-1/4 - 1/w + k^2/w^2) (dw/dk)^2
//!         + 1/2 d2w/dk2 )
//! ```
//!
//! The numerator `dw/dT` is the forward variance (positivity = calendar
//! no-arbitrage); the denominator is proportional to the density term
//! (positivity = butterfly no-arbitrage).  A flat surface collapses the
//! formula to `sigma_loc == implied vol` identically — the sanity anchor
//! used by the golden tests.
//!
//! Derivatives are central finite differences on the interpolated surface
//! with **fixed steps** `DK = 1e-3` and `DT = 1e-4`; at `T <= DT` the time
//! derivative switches to a forward difference so it never samples negative
//! expiries.  These exact steps are part of the cross-language contract.
//!
//! Robustness: the local vol is clamped to `[FLOOR, CAP] = [1%, 500%]`.
//! Numerator <= 0 floors, denominator <= 0 caps; every clamp is counted in
//! [`DupireLocalVol::floor_count`] / [`DupireLocalVol::cap_count`] so
//! callers can *report* surface quality instead of crashing mid-pricing.

use std::cell::Cell;

use crate::error::{invalid, Result};
use crate::surface::ImpliedVolSurface;

/// Central-difference step in log-moneyness.
pub const DK: f64 = 1.0e-3;
/// Central-difference step in expiry (years).
pub const DT: f64 = 1.0e-4;
/// 1% local-vol floor.
pub const FLOOR: f64 = 0.01;
/// 500% local-vol cap.
pub const CAP: f64 = 5.0;

const W_EPS: f64 = 1e-12;

/// Local-volatility function derived from an [`ImpliedVolSurface`].
///
/// Use [`DupireLocalVol::vol`] as the `sigma(k, t)` callable expected by
/// the PDE and Monte Carlo engines; its `k` argument is always **forward**
/// log-moneyness `ln(level / F(t))`.  The clamp counters use interior
/// mutability so the pricers can share the object immutably.
#[derive(Debug)]
pub struct DupireLocalVol {
    surface: ImpliedVolSurface,
    floor_count: Cell<u64>,
    cap_count: Cell<u64>,
}

impl DupireLocalVol {
    /// Wrap a surface (takes ownership; access it back via [`Self::surface`]).
    pub fn new(surface: ImpliedVolSurface) -> Self {
        DupireLocalVol {
            surface,
            floor_count: Cell::new(0),
            cap_count: Cell::new(0),
        }
    }

    /// The underlying implied surface.
    pub fn surface(&self) -> &ImpliedVolSurface {
        &self.surface
    }

    /// Cumulative number of floor clamps since construction / last reset.
    pub fn floor_count(&self) -> u64 {
        self.floor_count.get()
    }

    /// Cumulative number of cap clamps since construction / last reset.
    pub fn cap_count(&self) -> u64 {
        self.cap_count.get()
    }

    /// Zero the floor/cap violation counters.
    pub fn reset_counters(&self) {
        self.floor_count.set(0);
        self.cap_count.set(0);
    }

    /// Clamped local variance `sigma_loc^2(k, T)`; see [`Self::vol`].
    pub fn local_variance(&self, k: f64, expiry: f64) -> Result<f64> {
        if !expiry.is_finite() || expiry < 0.0 {
            return Err(invalid(format!("expiry must be finite and >= 0, got {expiry}")));
        }
        if !k.is_finite() {
            return Err(invalid("k must be finite"));
        }
        let srf = &self.surface;

        let w = srf.total_variance_unchecked(k, expiry);
        let w_up = srf.total_variance_unchecked(k + DK, expiry);
        let w_dn = srf.total_variance_unchecked(k - DK, expiry);
        let dwdk = (w_up - w_dn) / (2.0 * DK);
        let d2wdk2 = (w_up - 2.0 * w + w_dn) / (DK * DK);
        let dwdt = if expiry > DT {
            let wt_up = srf.total_variance_unchecked(k, expiry + DT);
            let wt_dn = srf.total_variance_unchecked(k, expiry - DT);
            (wt_up - wt_dn) / (2.0 * DT)
        } else {
            let wt_up = srf.total_variance_unchecked(k, expiry + DT);
            (wt_up - w) / DT
        };

        // Guard 1/w and k/w at (near-)zero total variance.
        let ws = w.max(W_EPS);
        let denom = 1.0 - (k / ws) * dwdk
            + 0.25 * (-0.25 - 1.0 / ws + (k * k) / (ws * ws)) * dwdk * dwdk
            + 0.5 * d2wdk2;

        let raw_var = if dwdt <= 0.0 {
            // No forward variance (calendar arb / flat-in-T) -> floor.
            self.floor_count.set(self.floor_count.get() + 1);
            FLOOR * FLOOR
        } else if denom <= 0.0 {
            // Butterfly-arbitrage wing -> cap.
            self.cap_count.set(self.cap_count.get() + 1);
            CAP * CAP
        } else {
            let vol = (dwdt / denom).sqrt();
            if vol < FLOOR {
                self.floor_count.set(self.floor_count.get() + 1);
            } else if vol > CAP {
                self.cap_count.set(self.cap_count.get() + 1);
            }
            let v = vol.clamp(FLOOR, CAP);
            v * v
        };
        Ok(raw_var)
    }

    /// Local volatility `sigma_loc(k, T)`, clamped to `[1%, 500%]`.
    /// `k` is forward log-moneyness `ln(level / F(T))`.
    pub fn vol(&self, k: f64, expiry: f64) -> Result<f64> {
        Ok(self.local_variance(k, expiry)?.sqrt())
    }

    /// Local vol evaluated at each `k` in `ks` (shared expiry).
    pub fn vol_slice(&self, ks: &[f64], expiry: f64) -> Result<Vec<f64>> {
        ks.iter().map(|&k| self.vol(k, expiry)).collect()
    }

    /// Human-readable clamp summary for demos/logs.
    pub fn violation_report(&self) -> String {
        format!(
            "local-vol clamps: floor({:.0}%) hit {}x, cap({:.0}%) hit {}x",
            FLOOR * 100.0,
            self.floor_count.get(),
            CAP * 100.0,
            self.cap_count.get()
        )
    }
}
