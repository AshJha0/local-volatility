//! Local-volatility Monte Carlo: log-Euler scheme with antithetic variates.
//!
//! Simulate `X = ln S` on a uniform time grid `t_n = n dt`, `dt = T / N`:
//!
//! ```text
//! X_{n+1} = X_n + (r - q - 1/2 sigma_n^2) dt + sigma_n sqrt(dt) Z_n
//! ```
//!
//! with `sigma_n = sigma(k_n, t_n)` looked up at the *start* of the step,
//! at forward log-moneyness `k_n = X_n - ln F(t_n)` — the same coordinates
//! the Dupire surface uses, so PDE and MC discretise the identical
//! diffusion.  Log-Euler keeps `S > 0` exactly.
//!
//! **Antithetic variates**: paths come in pairs driven by `+Z` and `-Z`
//! (each mirrored path recomputes its own `sigma` from its own state).
//! The estimator averages each pair first, and the standard error is
//! computed over the `n_paths / 2` *pair means* — raw per-path deviations
//! would overstate the error because paired payoffs are negatively
//! correlated.
//!
//! **Up-and-out call**, two monitoring modes:
//!
//! * *Discrete* (`brownian_bridge = false`): a path is dead once a step
//!   ends at or above `ln B`; this overprices the continuous barrier (the
//!   O(sqrt(dt)) discrete-monitoring bias).
//! * *Brownian-bridge correction* (`brownian_bridge = true`, default in the
//!   golden case): for a step from `x0` to `x1`, both below `b = ln B`,
//!   the bridge crossing probability is
//!   `p = exp(-2 (b - x0)(b - x1) / (sigma^2 dt))` and each path carries a
//!   multiplicative survival weight `prod (1 - p)`.  This removes the
//!   leading-order bias with no extra random numbers (the estimator stays
//!   deterministic per seed) and strictly lowers the price versus discrete
//!   monitoring.
//!
//! The RNG is `StdRng` seeded from a fixed `u64`; per the cross-language
//! contract each language uses its own generator, so MC golden comparisons
//! are statistical (tolerances sized at 4 standard errors).

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

use crate::error::{invalid, Result};
use crate::market::Market;
use crate::vol::VolInput;

/// Monte Carlo estimate with its standard error.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct McResult {
    /// Discounted-payoff estimate.
    pub price: f64,
    /// Standard error of the estimate (over pair means when antithetic).
    pub stderr: f64,
}

impl McResult {
    /// True when `reference` lies inside `n_se` standard errors.
    pub fn within(&self, reference: f64, n_se: f64) -> bool {
        (self.price - reference).abs() <= n_se * self.stderr.max(1e-300)
    }
}

/// Simulation settings shared by the MC pricers.
#[derive(Debug, Clone, Copy)]
pub struct McSettings {
    /// Total number of paths (even when `antithetic`).
    pub n_paths: usize,
    /// Number of time steps.
    pub n_steps: usize,
    /// RNG seed (fixed -> deterministic estimate).
    pub seed: u64,
    /// Antithetic variates on/off.
    pub antithetic: bool,
}

impl Default for McSettings {
    fn default() -> Self {
        McSettings {
            n_paths: 20_000,
            n_steps: 100,
            seed: 42,
            antithetic: true,
        }
    }
}

fn validate_mc(strike: f64, expiry: f64, vol: &VolInput, s: &McSettings) -> Result<()> {
    if !strike.is_finite() || strike < 0.0 {
        return Err(invalid(format!("strike must be finite and >= 0, got {strike}")));
    }
    if !expiry.is_finite() || expiry <= 0.0 {
        return Err(invalid(format!("expiry must be finite and > 0, got {expiry}")));
    }
    if s.n_paths < 2 {
        return Err(invalid(format!("n_paths must be >= 2, got {}", s.n_paths)));
    }
    if s.antithetic && s.n_paths % 2 != 0 {
        return Err(invalid("antithetic sampling requires an even n_paths"));
    }
    if s.n_steps < 1 {
        return Err(invalid(format!("n_steps must be >= 1, got {}", s.n_steps)));
    }
    if let Some(sig) = vol.flat() {
        if !sig.is_finite() || sig <= 0.0 {
            return Err(invalid(format!("flat vol must be finite and > 0 for MC, got {sig}")));
        }
    }
    Ok(())
}

/// Pair-mean (antithetic) or plain sample estimator with its standard error.
fn estimate(payoffs: &[f64], antithetic: bool) -> McResult {
    let samples: Vec<f64> = if antithetic {
        let half = payoffs.len() / 2;
        (0..half).map(|i| 0.5 * (payoffs[i] + payoffs[half + i])).collect()
    } else {
        payoffs.to_vec()
    };
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let var = samples.iter().map(|&p| (p - mean) * (p - mean)).sum::<f64>() / (n - 1.0);
    McResult {
        price: mean,
        stderr: (var / n).sqrt(),
    }
}

/// European vanilla by log-Euler MC (flat vol or Dupire local vol).
pub fn price_european_mc(
    market: &Market,
    strike: f64,
    expiry: f64,
    vol: &VolInput,
    is_call: bool,
    settings: &McSettings,
) -> Result<McResult> {
    validate_mc(strike, expiry, vol, settings)?;
    let mut rng = StdRng::seed_from_u64(settings.seed);
    let dt = expiry / settings.n_steps as f64;
    let sq = dt.sqrt();
    let drift_rq = (market.rate() - market.dividend()) * dt;
    let anti = settings.antithetic;
    let n_base = if anti { settings.n_paths / 2 } else { settings.n_paths };

    let mut x = vec![market.spot().ln(); n_base];
    let mut xa = if anti { x.clone() } else { Vec::new() };
    let mut z = vec![0.0_f64; n_base];
    for n in 0..settings.n_steps {
        let t = n as f64 * dt;
        let lf = market.log_forward(t);
        for zi in z.iter_mut() {
            *zi = rng.sample(StandardNormal);
        }
        for i in 0..n_base {
            let sig = vol.sigma(x[i] - lf, t)?;
            x[i] += drift_rq - 0.5 * sig * sig * dt + sig * sq * z[i];
        }
        if anti {
            for i in 0..n_base {
                let sig = vol.sigma(xa[i] - lf, t)?;
                xa[i] += drift_rq - 0.5 * sig * sig * dt - sig * sq * z[i];
            }
        }
    }

    let phi = if is_call { 1.0 } else { -1.0 };
    let df = (-market.rate() * expiry).exp();
    let payoff = |xt: f64| df * (phi * (xt.exp() - strike)).max(0.0);
    let payoffs: Vec<f64> = x.iter().chain(xa.iter()).map(|&xt| payoff(xt)).collect();
    Ok(estimate(&payoffs, anti))
}

/// Up-and-out call under local (or flat) vol, optional Brownian-bridge
/// correction.  Requires `S0 < barrier` (otherwise the option is born
/// knocked out and worth exactly 0, returned with zero standard error).
pub fn price_up_out_call_mc(
    market: &Market,
    strike: f64,
    barrier: f64,
    expiry: f64,
    vol: &VolInput,
    settings: &McSettings,
    brownian_bridge: bool,
) -> Result<McResult> {
    validate_mc(strike, expiry, vol, settings)?;
    if !barrier.is_finite() || barrier <= 0.0 {
        return Err(invalid(format!("barrier must be finite and > 0, got {barrier}")));
    }
    if market.spot() >= barrier {
        return Ok(McResult { price: 0.0, stderr: 0.0 });
    }

    let mut rng = StdRng::seed_from_u64(settings.seed);
    let dt = expiry / settings.n_steps as f64;
    let sq = dt.sqrt();
    let drift_rq = (market.rate() - market.dividend()) * dt;
    let b = barrier.ln();
    let anti = settings.antithetic;
    let n_base = if anti { settings.n_paths / 2 } else { settings.n_paths };

    // One shared normal block so the antithetic pass reuses the same draws.
    let z_all: Vec<Vec<f64>> = (0..settings.n_steps)
        .map(|_| (0..n_base).map(|_| rng.sample(StandardNormal)).collect())
        .collect();

    let run_block = |sign: f64| -> Result<Vec<f64>> {
        let mut x = vec![market.spot().ln(); n_base];
        let mut weight = vec![1.0_f64; n_base];
        for (n, z) in z_all.iter().enumerate() {
            let t = n as f64 * dt;
            let lf = market.log_forward(t);
            for i in 0..n_base {
                let sig = vol.sigma(x[i] - lf, t)?;
                let x_new = x[i] + drift_rq - 0.5 * sig * sig * dt + sign * sig * sq * z[i];
                if x_new >= b {
                    weight[i] = 0.0;
                } else if brownian_bridge && weight[i] > 0.0 {
                    // P[bridge from x to x_new crosses b], both below b.
                    let p = (-2.0 * (b - x[i]) * (b - x_new) / (sig * sig * dt)).exp();
                    weight[i] *= 1.0 - p;
                }
                x[i] = x_new;
            }
        }
        Ok((0..n_base)
            .map(|i| weight[i] * (x[i].exp() - strike).max(0.0))
            .collect())
    };

    let mut payoffs = run_block(1.0)?;
    if anti {
        payoffs.extend(run_block(-1.0)?);
    }
    let df = (-market.rate() * expiry).exp();
    for p in payoffs.iter_mut() {
        *p *= df;
    }
    Ok(estimate(&payoffs, anti))
}
