//! Golden-value suite: the Rust port must reproduce data/golden/golden.json.
//!
//! Case-name prefixes map to engines (the same dispatch every language uses):
//!   flat_dupire_*      -> Dupire local vol on data/flat_surface.csv
//!   pde_flat_*         -> flat-vol European PDE price
//!   dupire_bundled_*   -> Dupire local vol on data/implied_surface.csv
//!   pde_localvol_*     -> local-vol European PDE on the bundled surface
//!   american_put_*     -> American put PSOR PDE (reference: CRR binomial 5000)
//!   mc_flat_*          -> flat-vol MC vanilla (statistical tolerance, 4 SE)
//!   mc_localvol_*      -> local-vol MC vanilla (statistical tolerance)
//!   barrier_upout_*    -> up-and-out call MC with Brownian bridge (statistical)

mod common;

use std::collections::BTreeMap;

use serde::Deserialize;

use common::{assert_close, bundled_localvol, data_dir, flat_surface};
use localvol::{
    price_american_put_pde, price_european_mc, price_european_pde, price_up_out_call_mc,
    DupireLocalVol, Market, McSettings, PdeSettings, PsorSettings, VolInput,
};

/// Any fixed seed is allowed for the statistical cases (tolerances are 4 SE).
const SEED: u64 = 20260827;

#[derive(Deserialize)]
struct GoldenFile {
    cases: Vec<GoldenCase>,
}

#[derive(Deserialize)]
struct GoldenCase {
    name: String,
    inputs: BTreeMap<String, f64>,
    expect: BTreeMap<String, f64>,
    tol: f64,
}

impl GoldenCase {
    fn input(&self, key: &str) -> f64 {
        *self
            .inputs
            .get(key)
            .unwrap_or_else(|| panic!("{}: missing input {key}", self.name))
    }

    fn expect(&self, key: &str) -> f64 {
        *self
            .expect
            .get(key)
            .unwrap_or_else(|| panic!("{}: missing expect {key}", self.name))
    }

    fn market(&self) -> Market {
        Market::new(self.input("s"), self.input("r"), self.input("q")).expect("golden market")
    }

    fn pde_settings(&self) -> PdeSettings {
        PdeSettings {
            num_space: self.input("num_space") as usize,
            num_time: self.input("num_time") as usize,
            nsd: 6.0,
            sigma_ref: self.inputs.get("sigma_ref").copied(),
        }
    }

    fn mc_settings(&self) -> McSettings {
        McSettings {
            n_paths: self.input("n_paths") as usize,
            n_steps: self.input("n_steps") as usize,
            seed: SEED,
            antithetic: true,
        }
    }
}

fn load_cases() -> Vec<GoldenCase> {
    let path = data_dir().join("golden").join("golden.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let parsed: GoldenFile = serde_json::from_str(&text).expect("golden.json parses");
    parsed.cases
}

#[test]
fn golden_file_shape() {
    let cases = load_cases();
    assert_eq!(cases.len(), 15);
    for c in &cases {
        assert!(!c.inputs.is_empty(), "{}: empty inputs", c.name);
        assert_eq!(c.expect.len(), 1, "{}: expect one scalar output", c.name);
        assert!(c.tol > 0.0, "{}: non-positive tolerance", c.name);
    }
}

#[test]
fn all_golden_cases() {
    let flat = flat_surface();
    let lv = bundled_localvol();
    let mut checked = 0usize;
    for c in load_cases() {
        let name = c.name.as_str();
        if name.starts_with("flat_dupire_") {
            let got = DupireLocalVol::new(flat.clone())
                .vol(c.input("k"), c.input("T"))
                .unwrap();
            assert_close(got, c.expect("local_vol"), c.tol, name);
        } else if name.starts_with("dupire_bundled_") {
            let got = lv.vol(c.input("k"), c.input("T")).unwrap();
            assert_close(got, c.expect("local_vol"), c.tol, name);
        } else if name.starts_with("pde_flat_") {
            let got = price_european_pde(
                &c.market(),
                c.input("k"),
                c.input("T"),
                &VolInput::Flat(c.input("sigma")),
                c.input("call") != 0.0,
                &c.pde_settings(),
            )
            .unwrap();
            assert_close(got, c.expect("price"), c.tol, name);
        } else if name.starts_with("pde_localvol_") {
            let got = price_european_pde(
                &c.market(),
                c.input("k"),
                c.input("T"),
                &VolInput::Local(&lv),
                true,
                &c.pde_settings(),
            )
            .unwrap();
            assert_close(got, c.expect("price"), c.tol, name);
        } else if name.starts_with("american_put_") {
            let got = price_american_put_pde(
                &c.market(),
                c.input("k"),
                c.input("T"),
                &VolInput::Flat(c.input("sigma")),
                &c.pde_settings(),
                &PsorSettings::default(),
            )
            .unwrap();
            assert_close(got, c.expect("price"), c.tol, name);
        } else if name.starts_with("mc_flat_") {
            let res = price_european_mc(
                &c.market(),
                c.input("k"),
                c.input("T"),
                &VolInput::Flat(c.input("sigma")),
                true,
                &c.mc_settings(),
            )
            .unwrap();
            assert_close(res.price, c.expect("price"), c.tol, name);
        } else if name.starts_with("mc_localvol_") {
            let res = price_european_mc(
                &c.market(),
                c.input("k"),
                c.input("T"),
                &VolInput::Local(&lv),
                true,
                &c.mc_settings(),
            )
            .unwrap();
            assert_close(res.price, c.expect("price"), c.tol, name);
        } else if name.starts_with("barrier_upout_") {
            let res = price_up_out_call_mc(
                &c.market(),
                c.input("k"),
                c.input("b"),
                c.input("T"),
                &VolInput::Flat(c.input("sigma")),
                &c.mc_settings(),
                true,
            )
            .unwrap();
            assert_close(res.price, c.expect("price"), c.tol, name);
        } else {
            panic!("unrecognised golden case name: {name}");
        }
        checked += 1;
    }
    assert_eq!(checked, 15);
}
