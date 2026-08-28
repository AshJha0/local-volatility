//! Round-trip consistency: implied surface -> Dupire -> PDE -> implied vols.
//!
//! The full sweep lives in the demo; here a representative interior subset
//! is asserted to stay under the 30bp target (wings excluded per spec).

mod common;

use common::bundled_localvol;
use localvol::{implied_vol, price_european_pde, Market, PdeSettings, VolInput};

#[test]
fn interior_round_trip_under_30bp() {
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
        }
    }
}
