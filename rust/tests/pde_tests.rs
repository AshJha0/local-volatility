//! PDE pricer: BS agreement, Rannacher damping, convergence order, American.

mod common;

use common::{assert_close, bundled_localvol};
use localvol::{
    bs_price, price_american_put_pde, price_european_pde, price_european_pde_grid, Market,
    PdeSettings, PsorSettings, VolInput,
};

fn settings(m: usize, n: usize) -> PdeSettings {
    PdeSettings {
        num_space: m,
        num_time: n,
        ..PdeSettings::default()
    }
}

#[test]
fn flat_vol_matches_black_scholes() {
    // Equity (q > 0), ITM/OTM, FX Garman-Kohlhagen, negative rate, short-dated.
    let cases: [(f64, f64, f64, f64, f64, f64, bool); 6] = [
        (100.0, 100.0, 0.05, 0.02, 0.20, 1.0, true),
        (100.0, 80.0, 0.05, 0.00, 0.20, 1.0, true),
        (100.0, 120.0, 0.05, 0.00, 0.30, 0.5, false),
        (1.10, 1.05, 0.03, 0.01, 0.10, 0.5, false),
        (100.0, 110.0, -0.01, 0.00, 0.25, 2.0, true),
        (100.0, 100.0, 0.00, 0.03, 0.15, 0.25, true),
    ];
    for (s, k, r, q, sig, t, call) in cases {
        let exact = bs_price(s, k, r, q, sig, t, call).unwrap();
        let mkt = Market::new(s, r, q).unwrap();
        let approx =
            price_european_pde(&mkt, k, t, &VolInput::Flat(sig), call, &settings(200, 200))
                .unwrap();
        assert_close(approx, exact, 1e-3 * exact.abs(), "PDE vs BS");
    }
}

#[test]
fn deep_itm_otm_and_degenerate() {
    let mkt = Market::new(100.0, 0.02, 0.0).unwrap();
    let s = PdeSettings::default();
    // deep OTM: tiny but nonnegative; deep ITM: pinned to forward asymptote
    let otm = price_european_pde(&mkt, 300.0, 0.5, &VolInput::Flat(0.2), true, &s).unwrap();
    assert!((0.0..1e-3).contains(&otm), "deep OTM {otm}");
    let itm = price_european_pde(&mkt, 10.0, 0.5, &VolInput::Flat(0.2), true, &s).unwrap();
    let itm_bs = bs_price(100.0, 10.0, 0.02, 0.0, 0.2, 0.5, true).unwrap();
    assert_close(itm, itm_bs, 1e-3 * itm_bs, "deep ITM");
    // T = 0 -> intrinsic; sigma = 0 -> discounted forward intrinsic; K = 0
    assert_eq!(
        price_european_pde(&mkt, 90.0, 0.0, &VolInput::Flat(0.2), true, &s).unwrap(),
        10.0
    );
    let det = price_european_pde(&mkt, 90.0, 1.0, &VolInput::Flat(0.0), true, &s).unwrap();
    assert_close(
        det,
        bs_price(100.0, 90.0, 0.02, 0.0, 0.0, 1.0, true).unwrap(),
        1e-12,
        "sigma=0",
    );
    let k0 = price_european_pde(&mkt, 0.0, 1.0, &VolInput::Flat(0.2), true, &s).unwrap();
    assert_close(k0, 100.0, 1e-12, "K=0 call");
    assert_eq!(
        price_european_pde(&mkt, 0.0, 1.0, &VolInput::Flat(0.2), false, &s).unwrap(),
        0.0
    );
}

#[test]
fn put_call_parity_on_grid() {
    let mkt = Market::new(100.0, 0.03, 0.01).unwrap();
    let s = settings(200, 100);
    let c = price_european_pde(&mkt, 105.0, 1.0, &VolInput::Flat(0.2), true, &s).unwrap();
    let p = price_european_pde(&mkt, 105.0, 1.0, &VolInput::Flat(0.2), false, &s).unwrap();
    let rhs = 100.0 * (-0.01_f64).exp() - 105.0 * (-0.03_f64).exp();
    assert_close(c - p, rhs, 2e-2, "put-call parity");
}

#[test]
fn rannacher_damping_no_gamma_oscillation() {
    // Discrete gamma near the strike must stay essentially nonnegative:
    // plain CN leaves an oscillating gamma near the payoff kink; the
    // Rannacher start damps it.  Assert the most negative second difference
    // of the final grid in a band around the strike is negligible relative
    // to the peak curvature.
    let mkt = Market::new(100.0, 0.05, 0.0).unwrap();
    let res = price_european_pde_grid(
        &mkt,
        100.0,
        0.25,
        &VolInput::Flat(0.2),
        true,
        &settings(200, 50),
    )
    .unwrap();
    let (x, v) = (&res.x, &res.values);
    let mut min_curv = f64::INFINITY;
    let mut max_curv = f64::NEG_INFINITY;
    for i in 1..v.len() - 1 {
        let s = x[i].exp();
        if s > 70.0 && s < 140.0 {
            let curv = v[i + 1] - 2.0 * v[i] + v[i - 1];
            min_curv = min_curv.min(curv);
            max_curv = max_curv.max(curv);
        }
    }
    assert!(max_curv > 0.0);
    assert!(
        min_curv > -1e-4 * max_curv,
        "gamma oscillation: min {min_curv:.3e} vs max {max_curv:.3e}"
    );
}

#[test]
fn grid_convergence_second_order() {
    // Halving dx and dt twice: observed order must land in [1.5, 2.5].
    let mkt = Market::new(100.0, 0.05, 0.0).unwrap();
    let exact = bs_price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, true).unwrap();
    let mut errs = Vec::new();
    for m in [50usize, 100, 200] {
        let approx =
            price_european_pde(&mkt, 100.0, 1.0, &VolInput::Flat(0.2), true, &settings(m, m))
                .unwrap();
        errs.push((approx - exact).abs());
    }
    let orders: Vec<f64> = (0..2).map(|i| (errs[i] / errs[i + 1]).log2()).collect();
    let avg = 0.5 * (orders[0] + orders[1]);
    assert!(
        (1.5..=2.5).contains(&avg),
        "observed order {avg:.2} (steps: {orders:?})"
    );
}

#[test]
fn localvol_pde_with_surface() {
    // Local-vol PDE ~ BS at the surface's own implied vol (ATM, small err).
    let lv = bundled_localvol();
    let mkt = Market::new(100.0, 0.0, 0.0).unwrap();
    let atm_iv = lv.surface().implied_vol(0.0, 1.0).unwrap();
    let s = PdeSettings {
        sigma_ref: Some(atm_iv),
        ..PdeSettings::default()
    };
    let price = price_european_pde(&mkt, 100.0, 1.0, &VolInput::Local(&lv), true, &s).unwrap();
    let bs_ref = bs_price(100.0, 100.0, 0.0, 0.0, atm_iv, 1.0, true).unwrap();
    assert_close(price, bs_ref, 5e-3 * bs_ref, "local-vol ATM ~ BS"); // ~1bp vol agreement
}

#[test]
fn american_put_flat() {
    let mkt = Market::new(100.0, 0.05, 0.0).unwrap();
    let s = settings(150, 100);
    let psor = PsorSettings::default();
    let eur = price_european_pde(&mkt, 100.0, 1.0, &VolInput::Flat(0.2), false, &s).unwrap();
    let ame = price_american_put_pde(&mkt, 100.0, 1.0, &VolInput::Flat(0.2), &s, &psor).unwrap();
    // early-exercise premium nonnegative, and American >= intrinsic
    assert!(ame >= eur - 1e-10, "American {ame} < European {eur}");
    assert!(ame >= 0.0);
    // with q = 0 and r > 0 the premium is strictly positive for a put
    assert!(ame - eur > 0.01, "premium too small: {}", ame - eur);
}

#[test]
fn american_zero_rate_equals_european() {
    // r = 0, q = 0: an American put is never exercised early.
    let mkt = Market::new(100.0, 0.0, 0.0).unwrap();
    let s = settings(150, 100);
    let eur = price_european_pde(&mkt, 110.0, 1.0, &VolInput::Flat(0.25), false, &s).unwrap();
    let ame = price_american_put_pde(&mkt, 110.0, 1.0, &VolInput::Flat(0.25), &s, &PsorSettings::default())
        .unwrap();
    assert_close(ame, eur, 2e-3 * eur, "American == European at r=0");
}

#[test]
fn american_localvol_above_european() {
    let lv = bundled_localvol();
    let mkt = Market::new(100.0, 0.04, 0.0).unwrap();
    let s = PdeSettings {
        num_space: 150,
        num_time: 100,
        nsd: 6.0,
        sigma_ref: Some(0.2),
    };
    let eur = price_european_pde(&mkt, 105.0, 1.0, &VolInput::Local(&lv), false, &s).unwrap();
    let ame =
        price_american_put_pde(&mkt, 105.0, 1.0, &VolInput::Local(&lv), &s, &PsorSettings::default())
            .unwrap();
    assert!(ame >= eur - 1e-10, "American {ame} < European {eur}");
}

#[test]
fn sigma_zero_american_is_deterministic_optimum() {
    // r > 0, sigma = 0: exercising immediately is optimal for an ITM put.
    let mkt = Market::new(80.0, 0.05, 0.0).unwrap();
    let p = price_american_put_pde(
        &mkt,
        100.0,
        1.0,
        &VolInput::Flat(0.0),
        &PdeSettings::default(),
        &PsorSettings::default(),
    )
    .unwrap();
    assert_close(p, 20.0, 1e-6, "deterministic American");
}

#[test]
fn validation() {
    let mkt = Market::new(100.0, 0.05, 0.0).unwrap();
    let flat = VolInput::Flat(0.2);
    let bad = |s: PdeSettings| price_european_pde(&mkt, 100.0, 1.0, &flat, true, &s);
    let err = bad(settings(201, 200)).unwrap_err();
    assert!(err.to_string().contains("even"), "{err}");
    assert!(bad(settings(2, 200)).is_err());
    let err = bad(settings(200, 0)).unwrap_err();
    assert!(err.to_string().contains("num_time"), "{err}");
    let s = PdeSettings::default();
    assert!(price_european_pde(&mkt, -5.0, 1.0, &flat, true, &s).is_err());
    assert!(price_european_pde(&mkt, 100.0, -1.0, &flat, true, &s).is_err());
    assert!(price_european_pde(&mkt, 100.0, 1.0, &VolInput::Flat(-0.2), true, &s).is_err());
    let err = bad(PdeSettings { nsd: 0.0, ..s }).unwrap_err();
    assert!(err.to_string().contains("nsd"), "{err}");
    let err = price_american_put_pde(
        &mkt,
        100.0,
        1.0,
        &flat,
        &s,
        &PsorSettings { omega: 2.5, ..PsorSettings::default() },
    )
    .unwrap_err();
    assert!(err.to_string().contains("omega"), "{err}");
    assert!(price_american_put_pde(&mkt, 0.0, 1.0, &flat, &s, &PsorSettings::default()).is_err());
    assert!(price_american_put_pde(
        &mkt,
        100.0,
        1.0,
        &flat,
        &s,
        &PsorSettings { tol: 0.0, ..PsorSettings::default() }
    )
    .is_err());
}
