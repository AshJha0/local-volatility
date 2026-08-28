//! Surface interpolation: node exactness, continuity, extrapolation, arb checks.

mod common;

use common::{assert_close, bundled_surface, data_dir};
use localvol::{CubicSpline1D, ImpliedVolSurface};

/// Smile-with-skew surface used across these tests.
fn smile_surface() -> ImpliedVolSurface {
    let ks: Vec<f64> = (0..9).map(|i| -0.4 + 0.1 * i as f64).collect();
    let ts = [0.25, 0.5, 1.0, 2.0];
    let row: Vec<f64> = ks.iter().map(|&k| 0.20 + 0.05 * k * k * 4.0 - 0.08 * k).collect();
    let vols: Vec<Vec<f64>> = ts.iter().map(|_| row.clone()).collect();
    ImpliedVolSurface::new(&ks, &ts, &vols).unwrap()
}

#[test]
fn spline_exact_at_nodes_and_smooth() {
    let x: Vec<f64> = (0..11).map(|i| -1.0 + 0.3 * i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&v| v.sin() + 0.3 * v * v).collect();
    let sp = CubicSpline1D::new(&x, &y).unwrap();
    for (xi, yi) in x.iter().zip(&y) {
        assert_close(sp.eval(*xi).unwrap(), *yi, 1e-14, "node exactness");
    }
    // C0/C1 continuity across an interior node.
    let xn = x[5];
    let eps = 1e-7;
    assert_close(sp.eval(xn + eps).unwrap(), sp.eval(xn - eps).unwrap(), 1e-6, "C0");
    let left = (sp.eval(xn).unwrap() - sp.eval(xn - eps).unwrap()) / eps;
    let right = (sp.eval(xn + eps).unwrap() - sp.eval(xn).unwrap()) / eps;
    assert_close(left, right, 1e-5, "C1");
}

#[test]
fn spline_degenerate_and_validation() {
    let constant = CubicSpline1D::new(&[1.0], &[3.0]).unwrap();
    assert_eq!(constant.eval(99.0).unwrap(), 3.0);
    let lin = CubicSpline1D::new(&[0.0, 2.0], &[1.0, 5.0]).unwrap();
    assert_close(lin.eval(1.0).unwrap(), 3.0, 1e-15, "2-node linear");
    let err = CubicSpline1D::new(&[0.0, 0.0, 1.0], &[0.0, 0.0, 0.0]).unwrap_err();
    assert!(err.to_string().contains("increasing"), "{err}");
    assert!(CubicSpline1D::new(&[0.0, 1.0], &[f64::NAN, 1.0]).is_err());
    assert!(CubicSpline1D::new(&[0.0, 1.0], &[1.0]).is_err());
    assert!(CubicSpline1D::new(&[], &[]).is_err());
    assert!(lin.eval(f64::NAN).is_err());
}

#[test]
fn surface_exact_at_nodes() {
    let srf = smile_surface();
    for (j, &t) in srf.expiries().to_vec().iter().enumerate() {
        for (i, &k) in srf.k_nodes().to_vec().iter().enumerate() {
            let iv = srf.vols()[j][i];
            assert_close(srf.total_variance(k, t).unwrap(), iv * iv * t, 1e-14, "w at node");
            assert_close(srf.implied_vol(k, t).unwrap(), iv, 1e-12, "iv at node");
        }
    }
}

#[test]
fn surface_continuity_near_nodes() {
    // Property-style: w is continuous in both k and T at every pillar/node.
    let srf = smile_surface();
    let eps = 1e-8;
    for &t in srf.expiries().to_vec().iter() {
        for &k in srf.k_nodes().to_vec().iter() {
            let up = srf.total_variance(k + eps, t).unwrap();
            let dn = srf.total_variance(k - eps, t).unwrap();
            assert_close(up, dn, 1e-6, "k-continuity");
        }
        let mid = srf.total_variance(0.01, t).unwrap();
        assert_close(srf.total_variance(0.01, t + eps).unwrap(), mid, 1e-6, "T-continuity up");
        assert_close(srf.total_variance(0.01, t - eps).unwrap(), mid, 1e-6, "T-continuity down");
    }
}

#[test]
fn flat_k_extrapolation() {
    let srf = smile_surface();
    let w_hi = srf.total_variance(srf.k_max(), 1.0).unwrap();
    assert_close(srf.total_variance(srf.k_max() + 5.0, 1.0).unwrap(), w_hi, 1e-15, "right wing");
    let w_lo = srf.total_variance(srf.k_min(), 1.0).unwrap();
    assert_close(srf.total_variance(srf.k_min() - 5.0, 1.0).unwrap(), w_lo, 1e-15, "left wing");
}

#[test]
fn time_extrapolation_rules() {
    let srf = smile_surface();
    // below first pillar: proportional total variance (constant implied vol)
    let t0 = srf.expiries()[0];
    let iv1 = srf.implied_vol(0.05, t0).unwrap();
    assert_close(srf.implied_vol(0.05, 0.05).unwrap(), iv1, 1e-12, "short end iv");
    assert_eq!(srf.total_variance(0.05, 0.0).unwrap(), 0.0);
    assert_close(srf.implied_vol(0.05, 0.0).unwrap(), iv1, 1e-12, "T=0 short-end limit");
    // beyond last pillar: linear continuation of the last-interval slope
    let n = srf.expiries().len();
    let (t1, t2) = (srf.expiries()[n - 2], srf.expiries()[n - 1]);
    let w1 = srf.total_variance(0.1, t1).unwrap();
    let w2 = srf.total_variance(0.1, t2).unwrap();
    let slope = (w2 - w1) / (t2 - t1);
    assert_close(
        srf.total_variance(0.1, t2 + 1.0).unwrap(),
        w2 + slope,
        1e-12 * (w2 + slope),
        "long-end linear w",
    );
}

#[test]
fn single_expiry_flat_forward() {
    let srf = ImpliedVolSurface::new(
        &[-0.2, 0.0, 0.2],
        &[1.0],
        &[vec![0.25, 0.2, 0.22]],
    )
    .unwrap();
    assert!(srf.single_expiry());
    // flat forward variance on both sides of the pillar
    assert_close(srf.implied_vol(0.0, 0.5).unwrap(), 0.2, 1e-12, "below pillar");
    assert_close(srf.implied_vol(0.0, 3.0).unwrap(), 0.2, 1e-12, "beyond pillar");
}

#[test]
fn calendar_arbitrage_detected_not_fatal() {
    let srf = ImpliedVolSurface::new(
        &[-0.1, 0.0, 0.1],
        &[0.5, 1.0],
        &[vec![0.30, 0.30, 0.30], vec![0.10, 0.10, 0.30]], // w drops at 2 nodes
    )
    .unwrap();
    assert_eq!(srf.calendar_violations(), 2);
}

#[test]
fn surface_validation() {
    let ks = [-0.1, 0.0, 0.1];
    let ts = [0.5, 1.0];
    let good = vec![vec![0.2; 3]; 2];
    let err = ImpliedVolSurface::new(&ks, &ts, &vec![vec![0.2; 2]; 3]).unwrap_err();
    assert!(err.to_string().contains("shape"), "{err}");
    let rev: Vec<f64> = ks.iter().rev().copied().collect();
    assert!(ImpliedVolSurface::new(&rev, &ts, &good).is_err());
    assert!(ImpliedVolSurface::new(&ks, &[-0.5, 1.0], &good).is_err());
    assert!(ImpliedVolSurface::new(&ks, &ts, &vec![vec![-0.2; 3]; 2]).is_err());
    assert!(ImpliedVolSurface::new(&ks, &ts, &vec![vec![f64::INFINITY; 3]; 2]).is_err());
    assert!(ImpliedVolSurface::new(&[], &ts, &good).is_err());
    let srf = smile_surface();
    assert!(srf.total_variance(0.0, -1.0).is_err());
    assert!(srf.total_variance(f64::NAN, 1.0).is_err());
    assert!(srf.implied_vol(0.0, f64::INFINITY).is_err());
}

#[test]
fn from_csv_round_trip() {
    let srf = bundled_surface();
    assert_eq!(srf.expiries(), &[0.25, 0.5, 1.0, 2.0]);
    assert_eq!(srf.k_nodes().len(), 15);
    assert_eq!(srf.calendar_violations(), 0);
    // node values survive the CSV round trip exactly
    let k0 = srf.k_nodes()[0];
    let t0 = srf.expiries()[0];
    assert_close(srf.implied_vol(k0, t0).unwrap(), srf.vols()[0][0], 1e-12, "csv node");
    // a non-surface file is rejected on its header
    let err = ImpliedVolSurface::from_csv(data_dir().join("generate_data.py")).unwrap_err();
    assert!(err.to_string().contains("header"), "{err}");
    assert!(ImpliedVolSurface::from_csv(data_dir().join("missing.csv")).is_err());
}
