//! localvol demo: Dupire round-trip consistency check.
//!
//! Pipeline: bundled SSVI implied surface -> Dupire local vol -> reprice
//! European vanillas by PDE (and spot-check by Monte Carlo) -> invert back
//! to implied vols -> report the error against the input surface in basis
//! points.
//!
//! Run:  cd rust && cargo run --release --bin demo

use localvol::{
    implied_vol, price_american_put_pde, price_european_mc, price_european_pde,
    price_up_out_call_mc, DupireLocalVol, ImpliedVolSurface, Market, McSettings, PdeSettings,
    PsorSettings, Result, VolInput,
};

const INTERIOR_TARGET_BP: f64 = 30.0;

fn run() -> Result<()> {
    let line = "=".repeat(72);
    println!("{line}");
    println!("localvol demo — Dupire local volatility round-trip consistency");
    println!("{line}");

    let data = concat!(env!("CARGO_MANIFEST_DIR"), "/../data");
    let surface = ImpliedVolSurface::from_csv(format!("{data}/implied_surface.csv"))?;
    let mkt = Market::new(100.0, 0.02, 0.01)?; // equity-style carry
    println!(
        "surface: {} expiries x {} strikes, calendar violations: {}",
        surface.expiries().len(),
        surface.k_nodes().len(),
        surface.calendar_violations()
    );
    println!(
        "market : S0={:.2}  r={:.2}%  q={:.2}%\n",
        mkt.spot(),
        mkt.rate() * 100.0,
        mkt.dividend() * 100.0
    );
    let lv = DupireLocalVol::new(surface);

    // ---- round trip: implied in -> local vol -> PDE price -> implied out
    let expiries = [0.5, 1.0, 1.5, 2.0];
    let strikes = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0];
    println!("Round-trip implied-vol error (PDE 200x200), basis points:");
    print!("  T\\K   ");
    for k in strikes {
        print!("{k:>8.0}");
    }
    println!();
    let mut max_err_bp = 0.0_f64;
    let mut max_at = (0.0_f64, 0.0_f64);
    for t in expiries {
        lv.reset_counters();
        print!("  {t:<5.2}");
        let sigma_ref = lv.surface().implied_vol(0.0, t)?;
        for strike in strikes {
            let k = (strike / mkt.forward(t)?).ln();
            let iv_in = lv.surface().implied_vol(k, t)?;
            let settings = PdeSettings {
                sigma_ref: Some(sigma_ref),
                ..PdeSettings::default()
            };
            let price = price_european_pde(&mkt, strike, t, &VolInput::Local(&lv), true, &settings)?;
            let iv_out = implied_vol(price, mkt.spot(), strike, mkt.rate(), mkt.dividend(), t, true)?;
            let err_bp = (iv_out - iv_in).abs() * 1e4;
            print!("{err_bp:>8.1}");
            let interior = k.abs() <= 0.30; // wings excluded per spec
            if interior && err_bp > max_err_bp {
                max_err_bp = err_bp;
                max_at = (strike, t);
            }
        }
        println!();
    }
    println!(
        "\nmax interior |vol error|: {max_err_bp:.2} bp at K={:.0}, T={:.2}  \
         (target < {INTERIOR_TARGET_BP:.0} bp, |k| <= 0.30)",
        max_at.0, max_at.1
    );
    let status = if max_err_bp < INTERIOR_TARGET_BP { "PASS" } else { "FAIL" };
    println!("round-trip check: {status}");
    println!("{}", lv.violation_report());

    // ---- PDE vs MC cross-check at the ATM pillar
    let (t, strike) = (1.0, 100.0);
    let settings = PdeSettings {
        sigma_ref: Some(lv.surface().implied_vol(0.0, t)?),
        ..PdeSettings::default()
    };
    let pde = price_european_pde(&mkt, strike, t, &VolInput::Local(&lv), true, &settings)?;
    let mc_settings = McSettings {
        n_paths: 20_000,
        n_steps: 100,
        seed: 42,
        antithetic: true,
    };
    let mc = price_european_mc(&mkt, strike, t, &VolInput::Local(&lv), true, &mc_settings)?;
    println!(
        "\nPDE vs MC (K=100, T=1): PDE={pde:.4}  MC={:.4} +/- {:.4}  |diff|={:.2} SE ({})",
        mc.price,
        mc.stderr,
        (pde - mc.price).abs() / mc.stderr,
        if mc.within(pde, 3.0) { "OK" } else { "OUTSIDE 3 SE" }
    );

    // ---- American and barrier flavours under the same local vol
    let amer = price_american_put_pde(
        &mkt,
        100.0,
        1.0,
        &VolInput::Local(&lv),
        &settings,
        &PsorSettings::default(),
    )?;
    let eur_put = price_european_pde(&mkt, 100.0, 1.0, &VolInput::Local(&lv), false, &settings)?;
    let uo = price_up_out_call_mc(
        &mkt,
        100.0,
        130.0,
        1.0,
        &VolInput::Local(&lv),
        &mc_settings,
        true,
    )?;
    println!(
        "American put (PSOR)     : {amer:.4}  (European {eur_put:.4}, premium {:+.4})",
        amer - eur_put
    );
    println!(
        "Up-and-out call B=130 MC: {:.4} +/- {:.4} (vanilla {pde:.4}; barrier <= vanilla: {})",
        uo.price,
        uo.stderr,
        if uo.price <= pde { "OK" } else { "VIOLATED" }
    );

    println!("\ndone.");
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("demo failed: {e}");
        std::process::exit(1);
    }
}
