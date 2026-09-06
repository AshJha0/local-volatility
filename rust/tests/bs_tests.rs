//! Black-Scholes analytics: known values, parity, Greeks vs FD, edge cases.

mod common;

use common::assert_close;
use localvol::{bs_delta, bs_gamma, bs_price, bs_vega, erfc, implied_vol, norm_cdf, Market};

#[test]
fn known_atm_value() {
    // Classic textbook number: S=K=100, r=5%, q=0, sigma=20%, T=1.
    let p = bs_price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, true).unwrap();
    assert_close(p, 10.450584, 1e-6, "bs atm call");
}

#[test]
fn erfc_and_norm_cdf_reference_values() {
    // Spot values from an independent double-precision erfc (Python math.erfc).
    let refs = [
        (0.5, 0.4795001221869535),
        (1.9, 0.00720957076474253254),
        (2.5, 4.06952017444958919e-4),
        (4.0, 1.541725790028002e-8),
        (-1.2, 1.91031397822963545),
    ];
    for (x, want) in refs {
        let got = erfc(x);
        assert!(
            ((got - want) / want).abs() < 1e-12,
            "erfc({x}): {got} vs {want}"
        );
    }
    assert_close(norm_cdf(0.0), 0.5, 1e-15, "N(0)");
    assert_close(norm_cdf(1.6448536269514722), 0.95, 1e-12, "N(z_95)");
    assert_close(norm_cdf(-8.0), 6.22096057427178e-16, 1e-27, "deep tail");
}

#[test]
fn put_call_parity_grid() {
    // Property-style: parity C - P = S e^{-qT} - K e^{-rT} across a grid.
    for s in [80.0, 100.0, 123.4] {
        for k in [60.0, 100.0, 140.0] {
            for (r, q) in [(0.05, 0.0), (0.01, 0.03), (-0.005, 0.0)] {
                for t in [0.1, 1.0, 3.0] {
                    let c = bs_price(s, k, r, q, 0.25, t, true).unwrap();
                    let p = bs_price(s, k, r, q, 0.25, t, false).unwrap();
                    let rhs = s * (-q * t as f64).exp() - k * (-r * t as f64).exp();
                    assert_close(c - p, rhs, 1e-10, "parity");
                }
            }
        }
    }
}

#[test]
fn garman_kohlhagen_via_market_fx() {
    let mkt = Market::fx(1.20, 0.03, 0.01).unwrap();
    // GK call == BS call with q = rf.
    let a = bs_price(mkt.spot(), 1.25, mkt.rate(), mkt.dividend(), 0.10, 0.5, true).unwrap();
    let b = bs_price(1.20, 1.25, 0.03, 0.01, 0.10, 0.5, true).unwrap();
    assert_close(a, b, 1e-15, "GK == BS(q=rf)");
    assert_close(mkt.forward(0.5).unwrap(), 1.20 * (0.02_f64 * 0.5).exp(), 1e-15, "fx forward");
}

#[test]
fn edge_cases() {
    assert_eq!(bs_price(100.0, 90.0, 0.05, 0.0, 0.2, 0.0, true).unwrap(), 10.0);
    assert_eq!(bs_price(100.0, 110.0, 0.05, 0.0, 0.2, 0.0, true).unwrap(), 0.0);
    // sigma=0 -> discounted forward intrinsic
    let f = 100.0 * (0.05_f64).exp();
    assert_close(
        bs_price(100.0, 90.0, 0.05, 0.0, 0.0, 1.0, true).unwrap(),
        (-0.05_f64).exp() * (f - 90.0),
        1e-12,
        "sigma=0",
    );
    // K = 0
    assert_close(
        bs_price(100.0, 0.0, 0.03, 0.01, 0.2, 1.0, true).unwrap(),
        100.0 * (-0.01_f64).exp(),
        1e-12,
        "K=0 call",
    );
    assert_eq!(bs_price(100.0, 0.0, 0.03, 0.01, 0.2, 1.0, false).unwrap(), 0.0);
    // deep ITM/OTM pinned to asymptotics
    assert_close(bs_price(100.0, 1e6, 0.0, 0.0, 0.2, 1.0, true).unwrap(), 0.0, 1e-12, "deep OTM");
    assert_close(bs_price(100.0, 1.0, 0.0, 0.0, 0.2, 1.0, true).unwrap(), 99.0, 1e-8, "deep ITM");
}

#[test]
fn input_validation() {
    assert!(bs_price(-1.0, 100.0, 0.0, 0.0, 0.2, 1.0, true).is_err());
    assert!(bs_price(100.0, -5.0, 0.0, 0.0, 0.2, 1.0, true).is_err());
    assert!(bs_price(100.0, 100.0, 0.0, 0.0, -0.2, 1.0, true).is_err());
    assert!(bs_price(100.0, 100.0, 0.0, 0.0, 0.2, -1.0, true).is_err());
    assert!(bs_price(f64::NAN, 100.0, 0.0, 0.0, 0.2, 1.0, true).is_err());
    assert!(Market::new(-1.0, 0.0, 0.0).is_err());
    assert!(Market::new(100.0, f64::INFINITY, 0.0).is_err());
    assert!(Market::new(100.0, 0.0, 0.0).unwrap().forward(-1.0).is_err());
    assert!(bs_delta(100.0, 100.0, 0.0, 0.0, 0.0, 1.0, true).is_err()); // sigma = 0
    assert!(bs_gamma(100.0, 100.0, 0.0, 0.0, 0.2, 0.0).is_err()); // T = 0
}

#[test]
fn greeks_vs_finite_differences() {
    let (s, k, r, q, sig, t) = (105.0, 100.0, 0.03, 0.01, 0.22, 0.75);
    let eps = 1e-4;
    let p = |sp: f64, vol: f64| bs_price(sp, k, r, q, vol, t, true).unwrap();
    let d_fd = (p(s + eps, sig) - p(s - eps, sig)) / (2.0 * eps);
    let g_fd = (p(s + eps, sig) - 2.0 * p(s, sig) + p(s - eps, sig)) / (eps * eps);
    let v_fd = (p(s, sig + eps) - p(s, sig - eps)) / (2.0 * eps);
    assert_close(bs_delta(s, k, r, q, sig, t, true).unwrap(), d_fd, 1e-6, "delta vs FD");
    assert_close(bs_gamma(s, k, r, q, sig, t).unwrap(), g_fd, 1e-5, "gamma vs FD");
    assert_close(bs_vega(s, k, r, q, sig, t).unwrap(), v_fd, 1e-4, "vega vs FD");
}

#[test]
fn implied_vol_round_trip() {
    for sig in [0.05, 0.2, 0.8] {
        for k in [70.0, 100.0, 130.0] {
            let price = bs_price(100.0, k, 0.02, 0.01, sig, 1.5, true).unwrap();
            let iv = implied_vol(price, 100.0, k, 0.02, 0.01, 1.5, true).unwrap();
            assert_close(iv, sig, 1e-8, "implied vol round trip");
        }
    }
}

#[test]
fn implied_vol_rejects_arbitrage_and_bad_inputs() {
    let err = implied_vol(1000.0, 100.0, 100.0, 0.0, 0.0, 1.0, true).unwrap_err();
    assert!(err.to_string().contains("bounds"), "{err}"); // above S
    let err = implied_vol(-1.0, 100.0, 100.0, 0.0, 0.0, 1.0, true).unwrap_err();
    assert!(err.to_string().contains("bounds"), "{err}"); // below intrinsic
    assert!(implied_vol(5.0, 100.0, 100.0, 0.0, 0.0, 0.0, true).is_err()); // T = 0
    assert!(implied_vol(5.0, 100.0, 0.0, 0.0, 0.0, 1.0, true).is_err()); // K = 0
    assert!(implied_vol(f64::NAN, 100.0, 100.0, 0.0, 0.0, 1.0, true).is_err());
}

#[test]
fn market_log_forward_validation_and_value() {
    // MIN-6: log_forward validates expiry in every port.
    let mkt = localvol::Market::new(100.0, 0.03, 0.01).unwrap();
    let lf = mkt.log_forward(2.0).unwrap();
    assert!((lf - (100.0_f64.ln() + 0.02 * 2.0)).abs() < 1e-15);
    assert_eq!(mkt.log_forward(0.0).unwrap(), 100.0_f64.ln());
    assert!(mkt.log_forward(-1.0).is_err());
    assert!(mkt.log_forward(f64::NAN).is_err());
    assert!(mkt.forward(f64::INFINITY).is_err());
    assert!(localvol::Market::new(100.0, f64::NAN, 0.0).is_err());
    assert!(localvol::Market::new(100.0, 0.0, f64::INFINITY).is_err());
    assert!(localvol::Market::new(0.0, 0.0, 0.0).is_err());
}
