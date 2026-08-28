//! Black-Scholes / Garman-Kohlhagen analytics.
//!
//! All formulas are written in terms of `(r, q)`; for FX simply pass
//! `r = rd` and `q = rf` (Garman-Kohlhagen).  The normal CDF is computed
//! via `erfc` for accuracy in the far tails; since the Rust standard
//! library has no `erfc`, a self-contained double-precision implementation
//! is provided (positive-term Taylor series for small arguments, Lentz
//! continued fraction for the tail — both converge to full double
//! precision, comfortably inside the spec's 1e-12 relative-error budget).

use crate::error::{invalid, LocalVolError, Result};

const SQRT_2: f64 = std::f64::consts::SQRT_2;
const SQRT_PI: f64 = 1.772_453_850_905_516_1_f64;
const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7_f64;

/// Complementary error function to ~1e-15 relative accuracy.
///
/// * `|x| < 2`: `erfc(x) = 1 - erf(x)` with the **positive-term** series
///   `erf(x) = 2/sqrt(pi) * exp(-x^2) * sum_{n>=0} (2 x^2)^n x / (2n+1)!!`
///   (no alternating-sign cancellation, ~30 terms at the switch point).
/// * `x >= 2`: modified-Lentz evaluation of the classical continued
///   fraction `erfc(x) = exp(-x^2)/sqrt(pi) / (x + (1/2)/(x + 1/(x + (3/2)/(x + ...))))`.
/// * `x <= -2`: reflection `erfc(x) = 2 - erfc(-x)` (no cancellation:
///   the result is close to 2).
pub fn erfc(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.abs() < 2.0 {
        1.0 - erf_series(x)
    } else if x > 0.0 {
        erfc_tail(x)
    } else {
        2.0 - erfc_tail(-x)
    }
}

/// Positive-term Taylor/Kummer series for `erf`, valid for all finite `x`
/// but used only on `|x| < 2` where it converges in ~30 terms.
fn erf_series(x: f64) -> f64 {
    let x2 = x * x;
    let mut term = x; // (2 x^2)^n x / (2n+1)!!  at n = 0
    let mut sum = term;
    let mut n = 0u32;
    while term.abs() > 1e-18 * sum.abs() && n < 200 {
        n += 1;
        term *= 2.0 * x2 / (2.0 * f64::from(n) + 1.0);
        sum += term;
    }
    2.0 / SQRT_PI * (-x2).exp() * sum
}

/// Continued-fraction tail of `erfc` for `x >= 2` (modified Lentz).
fn erfc_tail(x: f64) -> f64 {
    const TINY: f64 = 1e-300;
    let mut f = x; // b0
    let mut c = f;
    let mut d = 0.0_f64;
    for n in 1..400 {
        let a = 0.5 * f64::from(n); // a_n = n / 2
        d = x + a * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = x + a / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() < 1e-17 {
            break;
        }
    }
    (-x * x).exp() / SQRT_PI / f
}

/// Standard normal CDF via `erfc` (tail-accurate).
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / SQRT_2)
}

/// Standard normal density.
pub fn norm_pdf(x: f64) -> f64 {
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

fn validate_common(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend: f64,
    sigma: f64,
    expiry: f64,
) -> Result<()> {
    for (name, v) in [
        ("spot", spot),
        ("strike", strike),
        ("rate", rate),
        ("dividend", dividend),
        ("sigma", sigma),
        ("expiry", expiry),
    ] {
        if !v.is_finite() {
            return Err(invalid(format!("{name} must be finite, got {v}")));
        }
    }
    if spot <= 0.0 {
        return Err(invalid(format!("spot must be > 0, got {spot}")));
    }
    if strike < 0.0 {
        return Err(invalid(format!("strike must be >= 0, got {strike}")));
    }
    if sigma < 0.0 {
        return Err(invalid(format!("sigma must be >= 0, got {sigma}")));
    }
    if expiry < 0.0 {
        return Err(invalid(format!("expiry must be >= 0, got {expiry}")));
    }
    Ok(())
}

/// European vanilla price under Black-Scholes / Garman-Kohlhagen.
///
/// Edge cases:
/// * `expiry == 0` -> intrinsic value `max(phi (S - K), 0)`;
/// * `sigma == 0`  -> discounted forward intrinsic `exp(-rT) max(phi (F - K), 0)`;
/// * `strike == 0` -> call `S exp(-qT)`, put `0`.
#[allow(clippy::too_many_arguments)]
pub fn bs_price(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend: f64,
    sigma: f64,
    expiry: f64,
    is_call: bool,
) -> Result<f64> {
    validate_common(spot, strike, rate, dividend, sigma, expiry)?;
    let phi = if is_call { 1.0 } else { -1.0 };
    if expiry == 0.0 {
        return Ok((phi * (spot - strike)).max(0.0));
    }
    let df_r = (-rate * expiry).exp();
    let df_q = (-dividend * expiry).exp();
    let forward = spot * ((rate - dividend) * expiry).exp();
    if strike == 0.0 {
        return Ok(if is_call { spot * df_q } else { 0.0 });
    }
    if sigma == 0.0 {
        return Ok(df_r * (phi * (forward - strike)).max(0.0));
    }
    let st = sigma * expiry.sqrt();
    let d1 = ((forward / strike).ln() + 0.5 * st * st) / st;
    let d2 = d1 - st;
    Ok(phi * (spot * df_q * norm_cdf(phi * d1) - strike * df_r * norm_cdf(phi * d2)))
}

fn d1(spot: f64, strike: f64, rate: f64, dividend: f64, sigma: f64, expiry: f64) -> f64 {
    let st = sigma * expiry.sqrt();
    let forward = spot * ((rate - dividend) * expiry).exp();
    ((forward / strike).ln() + 0.5 * st * st) / st
}

fn validate_greek(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend: f64,
    sigma: f64,
    expiry: f64,
    what: &str,
) -> Result<()> {
    validate_common(spot, strike, rate, dividend, sigma, expiry)?;
    if expiry <= 0.0 || sigma <= 0.0 || strike <= 0.0 {
        return Err(invalid(format!(
            "{what} requires expiry > 0, sigma > 0 and strike > 0"
        )));
    }
    Ok(())
}

/// Spot delta `phi exp(-q T) N(phi d1)` (requires `sigma, T, K > 0`).
pub fn bs_delta(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend: f64,
    sigma: f64,
    expiry: f64,
    is_call: bool,
) -> Result<f64> {
    validate_greek(spot, strike, rate, dividend, sigma, expiry, "bs_delta")?;
    let phi = if is_call { 1.0 } else { -1.0 };
    let d = d1(spot, strike, rate, dividend, sigma, expiry);
    Ok(phi * (-dividend * expiry).exp() * norm_cdf(phi * d))
}

/// Spot gamma `exp(-q T) n(d1) / (S sigma sqrt(T))` (call == put).
pub fn bs_gamma(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend: f64,
    sigma: f64,
    expiry: f64,
) -> Result<f64> {
    validate_greek(spot, strike, rate, dividend, sigma, expiry, "bs_gamma")?;
    let d = d1(spot, strike, rate, dividend, sigma, expiry);
    Ok((-dividend * expiry).exp() * norm_pdf(d) / (spot * sigma * expiry.sqrt()))
}

/// Vega `S exp(-q T) n(d1) sqrt(T)` per unit of vol (call == put).
pub fn bs_vega(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend: f64,
    sigma: f64,
    expiry: f64,
) -> Result<f64> {
    validate_greek(spot, strike, rate, dividend, sigma, expiry, "bs_vega")?;
    let d = d1(spot, strike, rate, dividend, sigma, expiry);
    Ok(spot * (-dividend * expiry).exp() * norm_pdf(d) * expiry.sqrt())
}

/// Invert Black-Scholes by bisection on the fixed bracket `[1e-9, 5.0]`
/// with exactly 100 halvings (the cross-language contract: deterministic
/// ~1e-10 vol accuracy with no vega/convergence corner cases).
///
/// Rejects prices outside the static no-arbitrage bounds
/// `[bs_price(sigma=0), call ? S e^{-qT} : K e^{-rT}]` with slack
/// `1e-12 * max(1, S)`.
pub fn implied_vol(
    price: f64,
    spot: f64,
    strike: f64,
    rate: f64,
    dividend: f64,
    expiry: f64,
    is_call: bool,
) -> Result<f64> {
    validate_common(spot, strike, rate, dividend, 0.0, expiry)?;
    if !price.is_finite() {
        return Err(invalid(format!("price must be finite, got {price}")));
    }
    if expiry <= 0.0 {
        return Err(invalid("implied_vol requires expiry > 0"));
    }
    if strike <= 0.0 {
        return Err(invalid("implied_vol requires strike > 0"));
    }
    let lower = bs_price(spot, strike, rate, dividend, 0.0, expiry, is_call)?;
    let upper = if is_call {
        spot * (-dividend * expiry).exp()
    } else {
        strike * (-rate * expiry).exp()
    };
    let eps = 1e-12 * spot.max(1.0);
    if price < lower - eps || price > upper + eps {
        return Err(LocalVolError::Numerical(format!(
            "price {price} outside no-arbitrage bounds [{lower:.10}, {upper:.10}]"
        )));
    }
    let (mut a, mut b) = (1e-9_f64, 5.0_f64);
    for _ in 0..100 {
        let mid = 0.5 * (a + b);
        if bs_price(spot, strike, rate, dividend, mid, expiry, is_call)? < price {
            a = mid;
        } else {
            b = mid;
        }
    }
    Ok(0.5 * (a + b))
}
