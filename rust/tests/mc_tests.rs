//! Monte Carlo: agreement with BS/PDE, antithetic effect, barrier properties.

mod common;

use common::bundled_localvol;
use localvol::{
    bs_price, price_european_mc, price_european_pde, price_up_out_call_mc, Market, McSettings,
    PdeSettings, VolInput,
};

const SEED: u64 = 987;

fn mc(n_paths: usize, n_steps: usize) -> McSettings {
    McSettings {
        n_paths,
        n_steps,
        seed: SEED,
        antithetic: true,
    }
}

#[test]
fn flat_mc_matches_bs_within_3se() {
    let mkt = Market::new(100.0, 0.03, 0.01).unwrap();
    let exact = bs_price(100.0, 105.0, 0.03, 0.01, 0.2, 1.0, true).unwrap();
    let res =
        price_european_mc(&mkt, 105.0, 1.0, &VolInput::Flat(0.2), true, &mc(20_000, 50)).unwrap();
    assert!(res.stderr > 0.0);
    assert!(
        res.within(exact, 3.0),
        "{}±{} vs {exact}",
        res.price,
        res.stderr
    );
}

#[test]
fn flat_mc_put_and_fx() {
    let mkt = Market::fx(1.20, 0.03, 0.01).unwrap();
    let exact = bs_price(1.20, 1.25, 0.03, 0.01, 0.10, 0.5, false).unwrap();
    let res =
        price_european_mc(&mkt, 1.25, 0.5, &VolInput::Flat(0.10), false, &mc(20_000, 50)).unwrap();
    assert!(res.within(exact, 3.0), "{}±{} vs {exact}", res.price, res.stderr);
}

#[test]
fn mc_deterministic_given_seed() {
    let mkt = Market::new(100.0, 0.02, 0.0).unwrap();
    let s11 = McSettings { n_paths: 2000, n_steps: 20, seed: 11, antithetic: true };
    let s12 = McSettings { seed: 12, ..s11 };
    let flat = VolInput::Flat(0.2);
    let a = price_european_mc(&mkt, 100.0, 1.0, &flat, true, &s11).unwrap();
    let b = price_european_mc(&mkt, 100.0, 1.0, &flat, true, &s11).unwrap();
    let c = price_european_mc(&mkt, 100.0, 1.0, &flat, true, &s12).unwrap();
    assert_eq!(a.price, b.price);
    assert_eq!(a.stderr, b.stderr);
    assert_ne!(a.price, c.price);
}

#[test]
fn antithetic_reduces_stderr() {
    let mkt = Market::new(100.0, 0.02, 0.0).unwrap();
    let flat = VolInput::Flat(0.2);
    let anti = price_european_mc(&mkt, 100.0, 1.0, &flat, true, &mc(20_000, 25)).unwrap();
    let plain_settings = McSettings { antithetic: false, ..mc(20_000, 25) };
    let plain = price_european_mc(&mkt, 100.0, 1.0, &flat, true, &plain_settings).unwrap();
    assert!(
        anti.stderr < plain.stderr,
        "antithetic {} !< plain {}",
        anti.stderr,
        plain.stderr
    );
}

#[test]
fn localvol_mc_matches_pde_within_3se() {
    let lv = bundled_localvol();
    let mkt = Market::new(100.0, 0.0, 0.0).unwrap();
    let pde_settings = PdeSettings {
        sigma_ref: Some(0.2),
        ..PdeSettings::default()
    };
    let pde =
        price_european_pde(&mkt, 100.0, 1.0, &VolInput::Local(&lv), true, &pde_settings).unwrap();
    let res =
        price_european_mc(&mkt, 100.0, 1.0, &VolInput::Local(&lv), true, &mc(20_000, 100)).unwrap();
    assert!(
        res.within(pde, 3.0),
        "MC {}±{} vs PDE {pde}",
        res.price,
        res.stderr
    );
}

#[test]
fn barrier_below_vanilla_and_bridge_lowers_price() {
    let lv = bundled_localvol();
    let mkt = Market::new(100.0, 0.02, 0.0).unwrap();
    let settings = mc(10_000, 50);
    let vol = VolInput::Local(&lv);
    let vanilla = price_european_mc(&mkt, 100.0, 1.0, &vol, true, &settings).unwrap();
    let plain = price_up_out_call_mc(&mkt, 100.0, 125.0, 1.0, &vol, &settings, false).unwrap();
    let bridged = price_up_out_call_mc(&mkt, 100.0, 125.0, 1.0, &vol, &settings, true).unwrap();
    assert!(plain.price <= vanilla.price + 1e-12); // knock-out cannot add value
    assert!(bridged.price <= plain.price + 1e-12); // bridge sees more crossings
    assert!(bridged.price > 0.0);
}

#[test]
fn barrier_born_knocked_out() {
    let mkt = Market::new(130.0, 0.02, 0.0).unwrap();
    let res = price_up_out_call_mc(
        &mkt,
        100.0,
        125.0,
        1.0,
        &VolInput::Flat(0.2),
        &mc(1000, 10),
        true,
    )
    .unwrap();
    assert_eq!(res.price, 0.0);
    assert_eq!(res.stderr, 0.0);
}

#[test]
fn validation() {
    let mkt = Market::new(100.0, 0.02, 0.0).unwrap();
    let flat = VolInput::Flat(0.2);
    let odd = McSettings { n_paths: 1001, ..McSettings::default() };
    let err = price_european_mc(&mkt, 100.0, 1.0, &flat, true, &odd).unwrap_err();
    assert!(err.to_string().contains("even"), "{err}");
    let zero_steps = McSettings { n_steps: 0, ..McSettings::default() };
    assert!(price_european_mc(&mkt, 100.0, 1.0, &flat, true, &zero_steps).is_err());
    let s = McSettings::default();
    assert!(price_european_mc(&mkt, 100.0, 0.0, &flat, true, &s).is_err()); // T = 0
    assert!(price_european_mc(&mkt, 100.0, 1.0, &VolInput::Flat(0.0), true, &s).is_err());
    assert!(price_european_mc(&mkt, -1.0, 1.0, &flat, true, &s).is_err());
    assert!(price_up_out_call_mc(&mkt, 100.0, -5.0, 1.0, &flat, &s, true).is_err());
    let tiny = McSettings { n_paths: 1, ..McSettings::default() };
    assert!(price_european_mc(&mkt, 100.0, 1.0, &flat, true, &tiny).is_err());
}

#[test]
fn localvol_mc_matches_pde_with_carry() {
    // MAJ-7: with r != q the lookup k = X_n - ln F(t_n) is exercised
    // (a ln S0 lookup lands 4-5 SE away from the PDE).
    let lv = bundled_localvol();
    let markets = [
        Market::new(100.0, 0.05, 0.02).unwrap(),
        Market::fx(1.10, 0.03, -0.01).unwrap(),
    ];
    for mkt in markets {
        let strike = mkt.spot();
        let settings = PdeSettings {
            sigma_ref: Some(lv.surface().implied_vol(0.0, 1.0).unwrap()),
            ..PdeSettings::default()
        };
        let pde = price_european_pde(&mkt, strike, 1.0, &VolInput::Local(&lv), true, &settings)
            .unwrap();
        let res =
            price_european_mc(&mkt, strike, 1.0, &VolInput::Local(&lv), true, &mc(20_000, 100))
                .unwrap();
        assert!(
            res.within(pde, 3.0),
            "S0={}: MC {}±{} vs PDE {pde}",
            mkt.spot(),
            res.price,
            res.stderr
        );
    }
}

#[test]
fn mc_put_call_parity_under_local_vol() {
    let lv = bundled_localvol();
    let mkt = Market::new(100.0, 0.03, 0.01).unwrap();
    let s = McSettings { seed: 5, ..mc(20_000, 100) };
    let vol = VolInput::Local(&lv);
    let c = price_european_mc(&mkt, 100.0, 1.0, &vol, true, &s).unwrap();
    let p = price_european_mc(&mkt, 100.0, 1.0, &vol, false, &s).unwrap();
    let parity = 100.0 * (-0.01_f64).exp() - 100.0 * (-0.03_f64).exp();
    assert!(
        (c.price - p.price - parity).abs() < 3.0 * (c.stderr * c.stderr + p.stderr * p.stderr).sqrt(),
        "parity violated: {} - {} vs {parity}",
        c.price,
        p.price
    );
}

#[test]
fn mc_rejects_tiny_path_counts() {
    // MAJ-8: n_paths = 2 with antithetic used to return stderr = NaN.
    let mkt = Market::new(100.0, 0.02, 0.0).unwrap();
    let flat = VolInput::Flat(0.2);
    let err = price_european_mc(&mkt, 100.0, 1.0, &flat, true, &mc(2, 5)).unwrap_err();
    assert!(err.to_string().contains("n_paths"), "{err}");
    let one = McSettings { antithetic: false, ..mc(1, 5) };
    assert!(price_european_mc(&mkt, 100.0, 1.0, &flat, true, &one).is_err());
    assert!(price_up_out_call_mc(&mkt, 100.0, 130.0, 1.0, &flat, &mc(2, 5), true).is_err());
    // deep ITM strike so every path pays and the sample spread is non-zero
    let four = price_european_mc(&mkt, 50.0, 1.0, &flat, true, &mc(4, 5)).unwrap();
    assert!(four.price.is_finite() && four.stderr.is_finite() && four.stderr > 0.0);
    let two = McSettings { antithetic: false, ..mc(2, 5) };
    let r2 = price_european_mc(&mkt, 50.0, 1.0, &flat, true, &two).unwrap();
    assert!(r2.stderr.is_finite() && r2.stderr > 0.0);
}

#[test]
fn mc_rejects_nan_flat_vol_and_bad_steps() {
    let mkt = Market::new(100.0, 0.02, 0.0).unwrap();
    assert!(price_european_mc(&mkt, 100.0, 1.0, &VolInput::Flat(f64::NAN), true, &mc(100, 5)).is_err());
    assert!(price_european_mc(&mkt, 100.0, 1.0, &VolInput::Flat(-0.2), true, &mc(100, 5)).is_err());
    assert!(price_european_mc(&mkt, 100.0, f64::NAN, &VolInput::Flat(0.2), true, &mc(100, 5)).is_err());
    assert!(price_european_mc(&mkt, f64::INFINITY, 1.0, &VolInput::Flat(0.2), true, &mc(100, 5)).is_err());
    assert!(price_up_out_call_mc(&mkt, 100.0, f64::NAN, 1.0, &VolInput::Flat(0.2), &mc(100, 5), true).is_err());
}
