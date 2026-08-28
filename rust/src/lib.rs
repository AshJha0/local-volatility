//! # localvol — Dupire local volatility toolkit
//!
//! Rust port of the validated Python reference implementation (see
//! `API_SPEC.md` for the language-neutral numerical contract):
//!
//! * [`surface::ImpliedVolSurface`] — total-variance implied surface with
//!   natural-cubic-spline interpolation in log-moneyness and linear total
//!   variance in expiry (flat wings, proportional short end, linear tail);
//! * [`dupire::DupireLocalVol`] — Gatheral total-variance Dupire local vol
//!   with fixed finite-difference steps and counted `[1%, 500%]` clamps;
//! * [`pde`] — log-spot Crank-Nicolson pricer with Rannacher start (native
//!   Thomas solver), European vanillas and American puts via PSOR;
//! * [`mc`] — log-Euler Monte Carlo with antithetic variates and an
//!   up-and-out barrier with Brownian-bridge correction;
//! * [`black_scholes`] — Black-Scholes / Garman-Kohlhagen analytics with a
//!   deterministic bisection implied-vol inverter.
//!
//! Equity and FX share every formula: FX (Garman-Kohlhagen) uses
//! `rate = rd` and `dividend = rf` (see [`market::Market::fx`]).
//!
//! All fallible operations return [`error::Result`]; the library never
//! panics on bad input.

pub mod black_scholes;
pub mod dupire;
pub mod error;
pub mod market;
pub mod mc;
pub mod pde;
pub mod surface;
pub mod tridiag;
pub mod vol;

pub use black_scholes::{
    bs_delta, bs_gamma, bs_price, bs_vega, erfc, implied_vol, norm_cdf, norm_pdf,
};
pub use dupire::{DupireLocalVol, CAP, DK, DT, FLOOR};
pub use error::{LocalVolError, Result};
pub use market::Market;
pub use mc::{price_european_mc, price_up_out_call_mc, McResult, McSettings};
pub use pde::{
    price_american_put_pde, price_american_put_pde_grid, price_european_pde,
    price_european_pde_grid, PdeResult, PdeSettings, PsorSettings,
};
pub use surface::{CubicSpline1D, ImpliedVolSurface};
pub use tridiag::thomas_solve;
pub use vol::VolInput;
