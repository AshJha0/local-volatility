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
//! with **fixed steps** `DK = 1e-3` and `DT = 1e-4`; at `0 < T <= DT` the
//! time derivative switches to a forward difference so it never samples
//! negative expiries.  These exact steps are part of the cross-language
//! contract.
//!
//! **Stencil clamp at the quoted wings.**  The surface is flat in `k`
//! beyond the last quoted strikes, so `w` is only C^0 there (a natural
//! spline has `w'' = 0` but `w' != 0` at the end node).  A central stencil
//! straddling that kink would read `d2w/dk2 ~ -w'/DK` and cap the vol at
//! 500% *at the quoted wing*.  The query is therefore clamped into
//! `[k_min + DK, k_max - DK]` before differencing (whenever the box is wider
//! than `2 DK`) and the clamped `k` is used in every term: local vol is
//! constant in `k` beyond `k_max - DK` and continuous across the wing node.
//!
//! **Short-expiry limit.**  `w(k, 0) = 0` exactly, so `vol(k, 0)` is defined
//! as the `T -> 0+` limit (Berestycki, Busca, Florent 2002) with
//! `s(k) = implied_vol(k, 0)`: `sigma_loc(k, 0) = s / (1 - k s'/s)`, `s'` by
//! the same central `DK` step.  The `T > 0` branch converges to it (flat
//! forward variance below the first pillar), so `vol` is continuous at
//! `T = 0`.
//!
//! Robustness: the local vol is clamped to `[FLOOR, CAP] = [1%, 500%]`.
//! Numerator <= 0 floors, denominator <= 0 caps; every clamp is counted in
//! [`DupireLocalVol::floor_count`] / [`DupireLocalVol::cap_count`] so
//! callers can *report* surface quality instead of crashing mid-pricing.
//!
//! **Thread safety.**  `DupireLocalVol` is `Send + Sync`: the surface is
//! immutable and the counters are `AtomicU64` with relaxed increments, so
//! one object can be shared across threads (e.g. a rayon strike grid).  The
//! *total* count is exact; a read taken while other threads are still
//! evaluating is only a snapshot, and `reset_counters` should be called
//! between phases, not during one.

use std::sync::atomic::{AtomicU64, Ordering};

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
/// log-moneyness `ln(level / F(t))`.  The clamp counters are atomic so the
/// pricers (and several threads) can share the object immutably.
#[derive(Debug)]
pub struct DupireLocalVol {
    surface: ImpliedVolSurface,
    floor_count: AtomicU64,
    cap_count: AtomicU64,
}

impl DupireLocalVol {
    /// Wrap a surface (takes ownership; access it back via [`Self::surface`]).
    pub fn new(surface: ImpliedVolSurface) -> Self {
        DupireLocalVol {
            surface,
            floor_count: AtomicU64::new(0),
            cap_count: AtomicU64::new(0),
        }
    }

    /// The underlying implied surface.
    pub fn surface(&self) -> &ImpliedVolSurface {
        &self.surface
    }

    /// Cumulative number of floor clamps since construction / last reset.
    pub fn floor_count(&self) -> u64 {
        self.floor_count.load(Ordering::Relaxed)
    }

    /// Cumulative number of cap clamps since construction / last reset.
    pub fn cap_count(&self) -> u64 {
        self.cap_count.load(Ordering::Relaxed)
    }

    /// Zero the floor/cap violation counters.
    pub fn reset_counters(&self) {
        self.floor_count.store(0, Ordering::Relaxed);
        self.cap_count.store(0, Ordering::Relaxed);
    }

    /// Shared clamp policy: `num <= 0` floors, else `den <= 0` caps, else
    /// `sqrt(num/den)` clipped into `[FLOOR, CAP]`; every hit counted.
    /// Returns the clamped local *variance*.
    fn clamp_and_count(&self, num: f64, den: f64) -> f64 {
        if num <= 0.0 {
            // No forward variance (calendar arb / flat-in-T) -> floor.
            self.floor_count.fetch_add(1, Ordering::Relaxed);
            FLOOR * FLOOR
        } else if den <= 0.0 {
            // Butterfly arbitrage -> cap.
            self.cap_count.fetch_add(1, Ordering::Relaxed);
            CAP * CAP
        } else {
            let vol = (num / den).sqrt();
            if vol < FLOOR {
                self.floor_count.fetch_add(1, Ordering::Relaxed);
            } else if vol > CAP {
                self.cap_count.fetch_add(1, Ordering::Relaxed);
            }
            let v = vol.clamp(FLOOR, CAP);
            v * v
        }
    }

    /// Berestycki-Busca-Florent `T -> 0+` limit at an already-clamped `k`:
    /// `sigma_loc(k, 0) = s / (1 - k s'/s)` with `s = implied_vol(k, 0)`.
    fn short_time_variance(&self, k: f64) -> f64 {
        let srf = &self.surface;
        let s = srf.implied_vol_unchecked(k, 0.0);
        let s_up = srf.implied_vol_unchecked(k + DK, 0.0);
        let s_dn = srf.implied_vol_unchecked(k - DK, 0.0);
        let dsdk = (s_up - s_dn) / (2.0 * DK);
        let ss = s.max(W_EPS); // guard s == 0 (spline overshoot to w <= 0)
        let denom = 1.0 - k * dsdk / ss;
        // Same clamp policy as T > 0 with num = s^2 and den = denom |denom|:
        // sqrt(num/den) = s/denom when denom > 0, s == 0 floors, denom <= 0 caps.
        self.clamp_and_count(s * s, denom * denom.abs())
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
        // Clamp into [k_min + DK, k_max - DK] so that no central stencil
        // straddles the C^0 kink of the flat wing extrapolation.
        let (kmin, kmax) = (srf.k_min(), srf.k_max());
        let k = if kmax - kmin > 2.0 * DK {
            k.clamp(kmin + DK, kmax - DK)
        } else {
            k
        };
        if expiry == 0.0 {
            return Ok(self.short_time_variance(k));
        }

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

        Ok(self.clamp_and_count(dwdt, denom))
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
            self.floor_count(),
            CAP * 100.0,
            self.cap_count()
        )
    }
}
