//! Round-trip consistency: implied surface -> Dupire -> PDE -> implied vols.
//!
//! The full sweep lives in the demo; here a representative interior subset
//! is asserted to stay under the 30 bp target (wings excluded per spec) and,
//! more tightly, under 5 bp — the wing-clamped Dupire stencil (CRIT-1)
//! brought the demo's worst interior cell from 16.4 bp to 2.7 bp.

mod common;

use common::bundled_localvol;
use localvol::{implied_vol, price_european_pde, Market, PdeSettings, VolInput};

#[test]
fn interior_round_trip_under_5bp() {
    let lv = bundled_localvol();
    let mkt = Market::new(100.0, 0.02, 0.01).unwrap(); // equity carry exercises the forward logic
    for expiry in [0.5, 1.0] {
        let sigma_ref = lv.surface().implied_vol(0.0, expiry).unwrap();
        let settings = PdeSettings {
            sigma_ref: Some(sigma_ref),
            ..PdeSettings::default()
        };
        for strike in [90.0, 100.0, 110.0] {
            let k = (strike / mkt.forward(expiry).unwrap()).ln();
            let iv_in = lv.surface().implied_vol(k, expiry).unwrap();
            let price =
                price_european_pde(&mkt, strike, expiry, &VolInput::Local(&lv), true, &settings)
                    .unwrap();
            let iv_out = implied_vol(
                price,
                mkt.spot(),
                strike,
                mkt.rate(),
                mkt.dividend(),
                expiry,
                true,
            )
            .unwrap();
            let err_bp = (iv_out - iv_in).abs() * 1e4;
            assert!(err_bp < 30.0, "K={strike} T={expiry}: {err_bp:.1}bp");
            assert!(err_bp < 5.0, "K={strike} T={expiry}: {err_bp:.2}bp (post-CRIT-1 budget)");
        }
    }
}

#[test]
fn full_demo_grid_round_trip_under_5bp_and_no_clamps() {
    // The demo's 4 x 7 sweep, asserted: max interior error < 5 bp and no
    // Dupire clamp fires anywhere on the PDE grids (r - q = 1%).
    let lv = bundled_localvol();
    let mkt = Market::new(100.0, 0.02, 0.01).unwrap();
    let mut worst = 0.0_f64;
    for expiry in [0.5, 1.0, 1.5, 2.0] {
        let settings = PdeSettings {
            sigma_ref: Some(lv.surface().implied_vol(0.0, expiry).unwrap()),
            ..PdeSettings::default()
        };
        for strike in [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0] {
            let k = (strike / mkt.forward(expiry).unwrap()).ln();
            let iv_in = lv.surface().implied_vol(k, expiry).unwrap();
            let price =
                price_european_pde(&mkt, strike, expiry, &VolInput::Local(&lv), true, &settings)
                    .unwrap();
            let iv_out = implied_vol(price, mkt.spot(), strike, mkt.rate(), mkt.dividend(), expiry, true)
                .unwrap();
            if k.abs() <= 0.30 {
                worst = worst.max((iv_out - iv_in).abs() * 1e4);
            }
        }
    }
    assert!(worst < 5.0, "max interior error {worst:.2} bp");
    assert!((worst - 2.73).abs() < 0.05, "documented worst cell (K=80, T=2): {worst:.2} bp");
    assert_eq!(lv.floor_count(), 0);
    assert_eq!(lv.cap_count(), 0);
}
