//! Log-spot finite-difference pricer: Crank-Nicolson with Rannacher start.
//!
//! In `x = ln S` the backward pricing equation under local volatility
//! `sigma(S, t)` reads (`tau = T - t` is time remaining):
//!
//! ```text
//! dV/dtau = 1/2 sigma^2 d2V/dx2 + (r - q - 1/2 sigma^2) dV/dx - r V
//! ```
//!
//! discretised on a **uniform** x-grid with central differences and marched
//! in `tau` with a theta-scheme (theta = 1 backward Euler, theta = 1/2
//! Crank-Nicolson).  Tridiagonal systems are solved with the native Thomas
//! kernel.
//!
//! **Why Rannacher?**  Crank-Nicolson is unconditionally *stable* but only
//! neutrally damped: its amplification factor tends to -1 for
//! high-frequency modes, so the payoff kink at the strike excites spurious
//! oscillations in the price — and worse, in gamma — that decay only slowly
//! ("dt too big vs dx" makes them visible even though the scheme never
//! blows up).  Rannacher's fix is to open with fully implicit steps, whose
//! amplification factor tends to 0, killing precisely those modes: the
//! **first dt-interval is replaced by two backward-Euler half-steps of size
//! dt/2**, and the remaining `N - 1` steps are Crank-Nicolson.  Clean
//! second-order convergence in (dx, dt) is recovered (the grid-convergence
//! test asserts an observed order in [1.5, 2.5]) with a non-oscillatory
//! gamma.
//!
//! **Grid** (the exact cross-language rule): `x_i = ln S0 + (i - M/2) h`
//! for `i = 0..=M` (`M` even), with half-width
//! `W = |ln(K/S0)| + nsd * sigma_ref * sqrt(T) + |r - q| * T` and
//! `h = 2 W / M`.  `ln S0` is exactly the middle node, so the price is read
//! off with no interpolation: `price = V[M/2]`.
//!
//! **Mesh Peclet condition and upwinding.**  Central differencing of the
//! drift keeps the theta-scheme matrix an M-matrix (monotone, diagonally
//! dominant) only while `|mu_i| h <= 2 a_i` at every node.  With a flat vol
//! on the default grid this holds, but under *local* vol the width rule
//! uses `sigma_ref` while each node uses its own `sigma_i` — a node floored
//! at 1% by the Dupire clamp with a few percent of carry violates it.  At
//! such nodes (and only there) the first derivative switches to first-order
//! **upwind**: `lower_i = alpha_i + max(-mu_i, 0)/h`,
//! `upper_i = alpha_i + max(mu_i, 0)/h`, `center_i = -2 alpha_i - |mu_i|/h
//! - r`.  Off-diagonals stay non-negative, the row sum stays `-r`, and the
//! scheme cannot produce spurious oscillations or negative prices; the cost
//! is `O(|mu| h / 2)` numerical diffusion at those nodes.
//!
//! Boundary conditions are Dirichlet with discounted asymptotics.  The
//! local-vol coefficient for the step from `tau` to `tau + dts` is frozen
//! at the step midpoint `t_mid = T - tau - dts/2` and evaluated at forward
//! log-moneyness `k_i = x_i - ln F(t_mid)` — the same coordinates the
//! Dupire surface is built in.
//!
//! **American exercise** uses PSOR (projected successive over-relaxation)
//! on the same theta-scheme systems.  PSOR was chosen over a penalty /
//! operator-splitting scheme because it solves the discrete linear
//! complementarity problem to a controllable tolerance with no penalty
//! parameter to tune, and warm-starting from the previous time level keeps
//! the iteration count small on these grids.  Non-convergence is reported
//! via a log line (not an error), per the project error policy.

use crate::black_scholes::bs_price;
use crate::error::{invalid, Result};
use crate::market::Market;
use crate::tridiag::thomas_solve;
use crate::vol::VolInput;

/// Grid / scheme settings for the PDE pricers.
#[derive(Debug, Clone, Copy)]
pub struct PdeSettings {
    /// Number of spatial intervals `M` (even, >= 4); the grid has `M + 1` nodes.
    pub num_space: usize,
    /// Number of time intervals `N` (>= 1).
    pub num_time: usize,
    /// Grid half-width in standard deviations.
    pub nsd: f64,
    /// Reference vol for the grid width; `None` -> the flat vol, or for
    /// local vol the ATM local vol at expiry, `vol(0, T)`.
    pub sigma_ref: Option<f64>,
}

impl Default for PdeSettings {
    fn default() -> Self {
        PdeSettings {
            num_space: 200,
            num_time: 200,
            nsd: 6.0,
            sigma_ref: None,
        }
    }
}

/// PSOR parameters for the American-put solver.
#[derive(Debug, Clone, Copy)]
pub struct PsorSettings {
    /// Relaxation parameter, in `(0, 2)`.
    pub omega: f64,
    /// Sup-norm stopping tolerance on the iterate update.
    pub tol: f64,
    /// Iteration cap; hitting it logs a warning and proceeds.
    pub max_iter: usize,
}

impl Default for PsorSettings {
    fn default() -> Self {
        PsorSettings {
            omega: 1.5,
            tol: 1e-8,
            max_iter: 10_000,
        }
    }
}

/// Price plus the final grid (for diagnostics such as gamma inspection).
#[derive(Debug, Clone)]
pub struct PdeResult {
    /// Price at `S0` (grid midpoint — no interpolation involved).
    pub price: f64,
    /// Log-spot grid nodes `x_0 ..= x_M`.
    pub x: Vec<f64>,
    /// Option values on the grid at valuation time (`tau = T`).
    pub values: Vec<f64>,
}

fn validate(
    strike: f64,
    expiry: f64,
    settings: &PdeSettings,
) -> Result<()> {
    if !strike.is_finite() || strike < 0.0 {
        return Err(invalid(format!("strike must be finite and >= 0, got {strike}")));
    }
    if !expiry.is_finite() || expiry < 0.0 {
        return Err(invalid(format!("expiry must be finite and >= 0, got {expiry}")));
    }
    if settings.num_space < 4 || settings.num_space % 2 != 0 {
        return Err(invalid(format!(
            "num_space must be an even integer >= 4, got {}",
            settings.num_space
        )));
    }
    if settings.num_time < 1 {
        return Err(invalid(format!("num_time must be >= 1, got {}", settings.num_time)));
    }
    if !settings.nsd.is_finite() || settings.nsd <= 0.0 {
        return Err(invalid(format!("nsd must be finite and > 0, got {}", settings.nsd)));
    }
    Ok(())
}

/// Uniform log-spot grid centred on `ln S0`; returns `(nodes, h)`.
fn build_grid(
    market: &Market,
    strike: f64,
    expiry: f64,
    sigma_ref: f64,
    num_space: usize,
    nsd: f64,
) -> (Vec<f64>, f64) {
    let x0 = market.spot().ln();
    let half_width = (strike / market.spot()).ln().abs()
        + nsd * sigma_ref * expiry.sqrt()
        + (market.rate() - market.dividend()).abs() * expiry;
    let h = 2.0 * half_width / num_space as f64;
    let m2 = num_space as f64 / 2.0;
    let x = (0..=num_space).map(|i| x0 + (i as f64 - m2) * h).collect();
    (x, h)
}

/// Rannacher schedule: `(theta, dts)` sub-steps summing to `expiry`.
/// The first dt-interval is two backward-Euler half-steps; the remaining
/// `num_time - 1` intervals are single Crank-Nicolson steps.
fn time_steps(expiry: f64, num_time: usize) -> Vec<(f64, f64)> {
    let dt = expiry / num_time as f64;
    let mut steps = vec![(1.0, 0.5 * dt), (1.0, 0.5 * dt)];
    steps.extend(std::iter::repeat((0.5, dt)).take(num_time - 1));
    steps
}

/// Interior-node theta-scheme coefficients (`lower`, `center`, `upper`):
/// central differences where `|mu| h <= 2a`, first-order upwind elsewhere
/// (see the module docs); both branches have row sum `-r`.
fn coefficients(
    sigma_nodes: &[f64],
    r: f64,
    q: f64,
    h: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = sigma_nodes.len();
    let mut lower = vec![0.0; n];
    let mut center = vec![0.0; n];
    let mut upper = vec![0.0; n];
    for i in 0..n {
        let a = 0.5 * sigma_nodes[i] * sigma_nodes[i]; // diffusion
        let mu = (r - q) - a; // log-spot drift
        let alpha = a / (h * h);
        if mu.abs() * h <= 2.0 * a {
            let beta = mu / (2.0 * h);
            lower[i] = alpha - beta; // coefficient of V_{i-1}
            upper[i] = alpha + beta; // coefficient of V_{i+1}
            center[i] = -2.0 * alpha - r; // coefficient of V_i
        } else {
            lower[i] = alpha + (-mu).max(0.0) / h;
            upper[i] = alpha + mu.max(0.0) / h;
            center[i] = -2.0 * alpha - mu.abs() / h - r;
        }
    }
    (lower, center, upper)
}

/// Explicit part of the theta scheme: `rhs_i = V_i + (1-theta) dts (L V)_i`
/// plus the implicit boundary contributions folded into the first/last row.
#[allow(clippy::too_many_arguments)]
fn theta_rhs(
    v: &[f64],
    lower: &[f64],
    center: &[f64],
    upper: &[f64],
    dts: f64,
    theta: f64,
    v0_new: f64,
    vm_new: f64,
) -> Vec<f64> {
    let n = lower.len();
    let mut rhs = vec![0.0; n];
    for i in 0..n {
        let lv = lower[i] * v[i] + center[i] * v[i + 1] + upper[i] * v[i + 2];
        rhs[i] = v[i + 1] + (1.0 - theta) * dts * lv;
    }
    rhs[0] += theta * dts * lower[0] * v0_new;
    rhs[n - 1] += theta * dts * upper[n - 1] * vm_new;
    rhs
}

/// One theta-scheme step for the interior nodes; returns the full new level.
#[allow(clippy::too_many_arguments)]
fn theta_step(
    v: &[f64],
    sigma_nodes: &[f64],
    r: f64,
    q: f64,
    h: f64,
    dts: f64,
    theta: f64,
    v0_new: f64,
    vm_new: f64,
) -> Result<Vec<f64>> {
    let (lower, center, upper) = coefficients(sigma_nodes, r, q, h);
    let rhs = theta_rhs(v, &lower, &center, &upper, dts, theta, v0_new, vm_new);
    let n = lower.len();
    let sub: Vec<f64> = (1..n).map(|i| -theta * dts * lower[i]).collect();
    let diag: Vec<f64> = (0..n).map(|i| 1.0 - theta * dts * center[i]).collect();
    let sup: Vec<f64> = (0..n - 1).map(|i| -theta * dts * upper[i]).collect();
    let interior = thomas_solve(&sub, &diag, &sup, &rhs)?;

    let mut out = vec![0.0; v.len()];
    out[0] = v0_new;
    out[1..=n].copy_from_slice(&interior);
    out[v.len() - 1] = vm_new;
    Ok(out)
}

/// One theta-scheme step solved as an LCP by projected SOR.
#[allow(clippy::too_many_arguments)]
fn psor_step(
    v: &[f64],
    sigma_nodes: &[f64],
    r: f64,
    q: f64,
    h: f64,
    dts: f64,
    theta: f64,
    v0_new: f64,
    vm_new: f64,
    obstacle: &[f64],
    psor: &PsorSettings,
) -> Vec<f64> {
    let (lower, center, upper) = coefficients(sigma_nodes, r, q, h);
    let rhs = theta_rhs(v, &lower, &center, &upper, dts, theta, v0_new, vm_new);
    let n = lower.len();
    // Full-length off-diagonals: sub[i] multiplies x[i-1] in row i (sub[0]
    // unused), sup[i] multiplies x[i+1] in row i (sup[n-1] unused).
    let sub: Vec<f64> = (0..n).map(|i| -theta * dts * lower[i]).collect();
    let diag: Vec<f64> = (0..n).map(|i| 1.0 - theta * dts * center[i]).collect();
    let sup: Vec<f64> = (0..n).map(|i| -theta * dts * upper[i]).collect();

    // Warm start from the previous level, projected onto the obstacle.
    let obs = &obstacle[1..=n];
    let mut x: Vec<f64> = (0..n).map(|i| v[i + 1].max(obs[i])).collect();
    let mut converged = false;
    let mut err = 0.0_f64;
    for _ in 0..psor.max_iter {
        err = 0.0;
        for i in 0..n {
            let mut acc = rhs[i];
            if i > 0 {
                acc -= sub[i] * x[i - 1];
            }
            if i < n - 1 {
                acc -= sup[i] * x[i + 1];
            }
            let gs = acc / diag[i];
            let xn = obs[i].max(x[i] + psor.omega * (gs - x[i]));
            err = err.max((xn - x[i]).abs());
            x[i] = xn;
        }
        if err < psor.tol {
            converged = true;
            break;
        }
    }
    if !converged {
        eprintln!(
            "warning: PSOR did not converge to {:e} within {} iterations (last update {:e}); \
             continuing with current iterate",
            psor.tol, psor.max_iter, err
        );
    }
    let mut out = vec![0.0; v.len()];
    out[0] = v0_new;
    out[1..=n].copy_from_slice(&x);
    out[v.len() - 1] = vm_new;
    out
}

/// Price a European vanilla by Crank-Nicolson with Rannacher start,
/// returning the final grid for diagnostics.
pub fn price_european_pde_grid(
    market: &Market,
    strike: f64,
    expiry: f64,
    vol: &VolInput,
    is_call: bool,
    settings: &PdeSettings,
) -> Result<PdeResult> {
    validate(strike, expiry, settings)?;
    let phi = if is_call { 1.0 } else { -1.0 };
    let analytic = |price: f64| PdeResult {
        price,
        x: Vec::new(),
        values: Vec::new(),
    };
    if expiry == 0.0 {
        return Ok(analytic((phi * (market.spot() - strike)).max(0.0)));
    }
    let sigma_ref = vol.resolve_sigma_ref(expiry, settings.sigma_ref)?;
    if vol.flat() == Some(0.0) {
        // Deterministic limit: the PDE degenerates to pure discounting.
        let p = bs_price(
            market.spot(),
            strike,
            market.rate(),
            market.dividend(),
            0.0,
            expiry,
            is_call,
        )?;
        return Ok(analytic(p));
    }
    if strike == 0.0 {
        // Degenerate payoff: call pays S_T, put pays 0; both are analytic.
        let p = if is_call {
            market.spot() * (-market.dividend() * expiry).exp()
        } else {
            0.0
        };
        return Ok(analytic(p));
    }

    let (r, q) = (market.rate(), market.dividend());
    let m = settings.num_space;
    let (x, h) = build_grid(market, strike, expiry, sigma_ref, m, settings.nsd);
    let s_nodes: Vec<f64> = x.iter().map(|&xi| xi.exp()).collect();
    let mut v: Vec<f64> = s_nodes
        .iter()
        .map(|&s| (phi * (s - strike)).max(0.0))
        .collect();

    let mut tau = 0.0;
    for (theta, dts) in time_steps(expiry, settings.num_time) {
        let t_mid = expiry - tau - 0.5 * dts;
        let sigma_nodes = match vol {
            VolInput::Flat(s) => vec![*s; m - 1],
            VolInput::Local(_) => {
                let lf = market.log_forward(t_mid)?;
                let ks: Vec<f64> = x[1..m].iter().map(|&xi| xi - lf).collect();
                vol.sigmas(&ks, t_mid)?
            }
        };
        let tau_new = tau + dts;
        let (v0_new, vm_new) = if is_call {
            (
                0.0,
                (s_nodes[m] * (-q * tau_new).exp() - strike * (-r * tau_new).exp()).max(0.0),
            )
        } else {
            (
                (strike * (-r * tau_new).exp() - s_nodes[0] * (-q * tau_new).exp()).max(0.0),
                0.0,
            )
        };
        v = theta_step(&v, &sigma_nodes, r, q, h, dts, theta, v0_new, vm_new)?;
        tau = tau_new;
    }

    Ok(PdeResult {
        price: v[m / 2],
        x,
        values: v,
    })
}

/// Price a European vanilla by Crank-Nicolson with Rannacher start.
///
/// * `vol`: flat vol or Dupire local vol, see [`VolInput`].
/// * `settings.sigma_ref` defaults to the flat vol or, for local vol,
///   `vol(0, T)` — the ATM local vol at expiry.
pub fn price_european_pde(
    market: &Market,
    strike: f64,
    expiry: f64,
    vol: &VolInput,
    is_call: bool,
    settings: &PdeSettings,
) -> Result<f64> {
    Ok(price_european_pde_grid(market, strike, expiry, vol, is_call, settings)?.price)
}

/// Price an American put by PSOR on the Rannacher/CN scheme, returning the
/// final grid for diagnostics.
///
/// The obstacle is the immediate-exercise value `max(K - S, 0)` at every
/// node and time level; boundaries are `V = K - S_min` at the lower edge
/// (deep ITM exercise) and 0 at the upper edge.
pub fn price_american_put_pde_grid(
    market: &Market,
    strike: f64,
    expiry: f64,
    vol: &VolInput,
    settings: &PdeSettings,
    psor: &PsorSettings,
) -> Result<PdeResult> {
    validate(strike, expiry, settings)?;
    if !(psor.omega > 0.0 && psor.omega < 2.0) {
        return Err(invalid(format!("omega must lie in (0, 2), got {}", psor.omega)));
    }
    // `!(tol > 0)` also rejects NaN, which `tol <= 0` would let through.
    if !(psor.tol.is_finite() && psor.tol > 0.0) || psor.max_iter < 1 {
        return Err(invalid(format!(
            "tol must be finite and > 0 and max_iter >= 1, got {}, {}",
            psor.tol, psor.max_iter
        )));
    }
    if strike <= 0.0 {
        return Err(invalid("American put requires strike > 0"));
    }
    let analytic = |price: f64| PdeResult {
        price,
        x: Vec::new(),
        values: Vec::new(),
    };
    if expiry == 0.0 {
        return Ok(analytic((strike - market.spot()).max(0.0)));
    }
    let sigma_ref = vol.resolve_sigma_ref(expiry, settings.sigma_ref)?;
    if vol.flat() == Some(0.0) {
        // sigma = 0: exercise is deterministic; maximise the discounted
        // deterministic payoff over 2001 equidistant exercise dates.
        let n = 2000;
        let mut best = 0.0_f64;
        for i in 0..=n {
            let t = expiry * i as f64 / n as f64;
            let s_t = market.spot() * ((market.rate() - market.dividend()) * t).exp();
            let v = (strike - s_t).max(0.0) * (-market.rate() * t).exp();
            best = best.max(v);
        }
        return Ok(analytic(best));
    }

    let (r, q) = (market.rate(), market.dividend());
    let m = settings.num_space;
    let (x, h) = build_grid(market, strike, expiry, sigma_ref, m, settings.nsd);
    let s_nodes: Vec<f64> = x.iter().map(|&xi| xi.exp()).collect();
    let obstacle: Vec<f64> = s_nodes.iter().map(|&s| (strike - s).max(0.0)).collect();
    let mut v = obstacle.clone();

    let mut tau = 0.0;
    for (theta, dts) in time_steps(expiry, settings.num_time) {
        let t_mid = expiry - tau - 0.5 * dts;
        let sigma_nodes = match vol {
            VolInput::Flat(s) => vec![*s; m - 1],
            VolInput::Local(_) => {
                let lf = market.log_forward(t_mid)?;
                let ks: Vec<f64> = x[1..m].iter().map(|&xi| xi - lf).collect();
                vol.sigmas(&ks, t_mid)?
            }
        };
        let v0_new = obstacle[0]; // deep ITM: exercise value K - S_min
        let vm_new = 0.0;
        v = psor_step(
            &v, &sigma_nodes, r, q, h, dts, theta, v0_new, vm_new, &obstacle, psor,
        );
        tau += dts;
    }

    Ok(PdeResult {
        price: v[m / 2],
        x,
        values: v,
    })
}

/// Price an American put by PSOR (see [`price_american_put_pde_grid`]).
pub fn price_american_put_pde(
    market: &Market,
    strike: f64,
    expiry: f64,
    vol: &VolInput,
    settings: &PdeSettings,
    psor: &PsorSettings,
) -> Result<f64> {
    Ok(price_american_put_pde_grid(market, strike, expiry, vol, settings, psor)?.price)
}
