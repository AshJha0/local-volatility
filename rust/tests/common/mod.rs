//! Shared helpers for the integration-test suites.

#![allow(dead_code)] // each test binary uses a different subset

use std::path::PathBuf;

use localvol::{DupireLocalVol, ImpliedVolSurface};

/// `<repo>/local-volatility/data`, resolved relative to the rust/ dir.
pub fn data_dir() -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../data"))
}

/// The bundled SSVI surface from `data/implied_surface.csv`.
pub fn bundled_surface() -> ImpliedVolSurface {
    ImpliedVolSurface::from_csv(data_dir().join("implied_surface.csv"))
        .expect("bundled surface loads")
}

/// The 20%-flat surface from `data/flat_surface.csv`.
pub fn flat_surface() -> ImpliedVolSurface {
    ImpliedVolSurface::from_csv(data_dir().join("flat_surface.csv")).expect("flat surface loads")
}

/// Dupire local vol on the bundled surface.
pub fn bundled_localvol() -> DupireLocalVol {
    DupireLocalVol::new(bundled_surface())
}

/// Assert `|got - expect| <= tol` with a readable failure message.
pub fn assert_close(got: f64, expect: f64, tol: f64, what: &str) {
    assert!(
        (got - expect).abs() <= tol,
        "{what}: got {got}, expected {expect} (tol {tol}, diff {:.3e})",
        (got - expect).abs()
    );
}
