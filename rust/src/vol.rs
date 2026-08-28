//! Volatility input shared by the PDE and Monte Carlo engines.

use crate::dupire::DupireLocalVol;
use crate::error::{invalid, Result};

/// Volatility model handed to a pricer: either a flat Black vol or a
/// Dupire local-vol function.  The `k` argument of every lookup is
/// **forward** log-moneyness `ln(level / F(t))`.
#[derive(Clone, Copy)]
pub enum VolInput<'a> {
    /// Constant volatility (must be finite and `>= 0`; Monte Carlo
    /// additionally requires `> 0`).
    Flat(f64),
    /// Local volatility `sigma_loc(k, t)` from a Dupire surface.
    Local(&'a DupireLocalVol),
}

impl<'a> VolInput<'a> {
    /// The flat vol if this input is flat.
    pub(crate) fn flat(&self) -> Option<f64> {
        match self {
            VolInput::Flat(s) => Some(*s),
            VolInput::Local(_) => None,
        }
    }

    /// `sigma(k, t)` for a single point.
    pub fn sigma(&self, k: f64, t: f64) -> Result<f64> {
        match self {
            VolInput::Flat(s) => Ok(*s),
            VolInput::Local(lv) => lv.vol(k, t),
        }
    }

    /// `sigma(k, t)` for a slice of log-moneyness points at one time.
    pub fn sigmas(&self, ks: &[f64], t: f64) -> Result<Vec<f64>> {
        match self {
            VolInput::Flat(s) => Ok(vec![*s; ks.len()]),
            VolInput::Local(lv) => lv.vol_slice(ks, t),
        }
    }

    /// Validate a flat vol for PDE use (`>= 0`) and resolve the grid
    /// reference vol: the flat vol itself (1.0 when it is zero, since the
    /// PDE short-circuits before the grid is built), or for local vol the
    /// caller-supplied `sigma_ref`, defaulting to `vol(0, T)` — the ATM
    /// local vol at expiry.
    pub(crate) fn resolve_sigma_ref(&self, expiry: f64, sigma_ref: Option<f64>) -> Result<f64> {
        match self {
            VolInput::Flat(s) => {
                if !s.is_finite() || *s < 0.0 {
                    return Err(invalid(format!("flat vol must be finite and >= 0, got {s}")));
                }
                Ok(if *s > 0.0 { *s } else { 1.0 })
            }
            VolInput::Local(lv) => {
                let sr = match sigma_ref {
                    Some(v) => v,
                    None => lv.vol(0.0, expiry)?,
                };
                if !sr.is_finite() || sr <= 0.0 {
                    return Err(invalid(format!("sigma_ref must be finite and > 0, got {sr}")));
                }
                Ok(sr)
            }
        }
    }
}
