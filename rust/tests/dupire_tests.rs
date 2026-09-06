//! Dupire local vol: flat-surface identity, clamping, skew/smile behaviour.

mod common;

use common::{assert_close, bundled_localvol, flat_surface};
use localvol::{DupireLocalVol, ImpliedVolSurface, CAP, FLOOR};

#[test]
fn flat_surface_localvol_equals_implied() {
    // Property-style golden anchor: sigma_loc == 20% to 1e-6 everywhere.
    let lv = DupireLocalVol::new(flat_surface());
    for i in 0..9 {
        let k = -0.45 + 0.1125 * i as f64;
        for t in [0.05, 0.25, 0.6, 1.0, 1.7, 2.4] {
            assert_close(lv.vol(k, t).unwrap(), 0.20, 1e-6, "flat local vol");
        }
    }
    assert_eq!(lv.floor_count(), 0);
    assert_eq!(lv.cap_count(), 0);
}

#[test]
fn bundled_surface_sane_interior() {
    // Skewed surface: local vol finite, in band, decreasing across the skew.
    let lv = bundled_localvol();
    let ks: Vec<f64> = (0..11).map(|i| -0.25 + 0.05 * i as f64).collect();
    let vols = lv.vol_slice(&ks, 1.0).unwrap();
    for &v in &vols {
        assert!(v.is_finite() && (FLOOR..=CAP).contains(&v), "vol {v} out of band");
    }
    // negative-rho SSVI: local vol decreases from put wing to call wing
    assert!(vols[0] > vols[10], "no skew: {} vs {}", vols[0], vols[10]);
    // local skew is roughly twice the implied skew — at least steeper:
    let imp0 = lv.surface().implied_vol(ks[0], 1.0).unwrap();
    let imp1 = lv.surface().implied_vol(ks[10], 1.0).unwrap();
    assert!((vols[0] - vols[10]) > (imp0 - imp1));
}

#[test]
fn slice_matches_scalar() {
    let lv = bundled_localvol();
    let ks = [-0.3, -0.1, 0.0, 0.15, 0.3];
    let vec = lv.vol_slice(&ks, 0.8).unwrap();
    for (i, &k) in ks.iter().enumerate() {
        assert_close(vec[i], lv.vol(k, 0.8).unwrap(), 1e-15, "slice vs scalar");
    }
}

#[test]
fn floor_engaged_on_extreme_skew() {
    // A calendar-flat (dw/dT ~ 0) steep surface must clamp, not fail.
    let ks: Vec<f64> = (0..11).map(|i| -0.5 + 0.1 * i as f64).collect();
    let ts = [0.5, 1.0];
    let v1: Vec<f64> = ks.iter().map(|&k| 0.2 + 0.5 * k.abs()).collect();
    // nearly constant total variance in T -> numerator ~ 0 -> floor engages
    let v2: Vec<f64> = v1.iter().map(|&v| ((v * v * 0.5 + 1e-9) / 1.0).sqrt()).collect();
    let srf = ImpliedVolSurface::new(&ks, &ts, &[v1, v2]).unwrap();
    let lv = DupireLocalVol::new(srf);
    for &k in &ks {
        let v = lv.vol(k, 0.75).unwrap();
        assert!(v >= FLOOR - 1e-15, "below floor: {v}");
    }
    assert!(lv.floor_count() > 0, "floor never engaged");
    assert!(lv.violation_report().contains("floor"));
    lv.reset_counters();
    assert_eq!(lv.floor_count(), 0);
    assert_eq!(lv.cap_count(), 0);
}

#[test]
fn cap_engaged_on_butterfly_violating_wing() {
    // Over-steep smile (butterfly arb): denominator <= 0 -> capped at 500%.
    let ks: Vec<f64> = (0..21).map(|i| -0.5 + 0.05 * i as f64).collect();
    let ts = [0.5, 1.0];
    let v1: Vec<f64> = ks.iter().map(|&k| 0.10 + k.abs()).collect(); // V-shaped smile
    let v2: Vec<f64> = v1.iter().map(|&v| v * 1.05).collect();
    let srf = ImpliedVolSurface::new(&ks, &ts, &[v1, v2]).unwrap();
    let lv = DupireLocalVol::new(srf);
    for i in 0..41 {
        let k = -0.45 + 0.0225 * i as f64;
        let v = lv.vol(k, 0.75).unwrap();
        assert!(v <= CAP + 1e-15, "above cap: {v}");
    }
    assert!(lv.cap_count() > 0, "cap never engaged");
}

#[test]
fn fx_symmetric_smile() {
    // FX-style symmetric smile: local vol symmetric-ish and smile-shaped.
    let ks: Vec<f64> = (0..13).map(|i| -0.3 + 0.05 * i as f64).collect();
    let ts = [0.25, 0.5, 1.0];
    let row: Vec<f64> = ks.iter().map(|&k| 0.10 + 0.15 * k * k).collect();
    let vols: Vec<Vec<f64>> = ts.iter().map(|_| row.clone()).collect();
    let lv = DupireLocalVol::new(ImpliedVolSurface::new(&ks, &ts, &vols).unwrap());
    let v_pos = lv.vol(0.15, 0.5).unwrap();
    let v_neg = lv.vol(-0.15, 0.5).unwrap();
    let v_atm = lv.vol(0.0, 0.5).unwrap();
    assert!(
        ((v_pos - v_neg) / v_neg).abs() < 0.15,
        "asymmetric: {v_pos} vs {v_neg}"
    );
    assert!(v_pos > v_atm && v_neg > v_atm, "smile shape lost");
}

#[test]
fn validation() {
    let srf = ImpliedVolSurface::new(
        &[-0.1, 0.0, 0.1],
        &[0.5, 1.0],
        &[vec![0.2; 3], vec![0.2; 3]],
    )
    .unwrap();
    let lv = DupireLocalVol::new(srf);
    assert!(lv.vol(0.0, -1.0).is_err());
    assert!(lv.vol(f64::INFINITY, 1.0).is_err());
    assert!(lv.local_variance(f64::NAN, 1.0).is_err());
}

#[test]
fn no_cap_inside_quoted_box_and_continuous_at_wings() {
    // CRIT-1: before the stencil clamp vol(0.4995..0.5005, 1) was 5.0.
    let lv = bundled_localvol();
    let (kmin, kmax) = (lv.surface().k_min(), lv.surface().k_max());
    for t in [0.1, 0.5, 0.75, 1.0, 2.0, 3.0] {
        for i in 0..=200 {
            let k = -0.5 + 0.005 * i as f64;
            let v = lv.vol(k, t).unwrap();
            assert!(v > FLOOR && v < CAP, "k={k} T={t}: {v}");
        }
        for edge in [kmin, kmax] {
            let lo = lv.vol(edge - 1e-4, t).unwrap();
            let hi = lv.vol(edge + 1e-4, t).unwrap();
            assert!((lo - hi).abs() < 1e-3, "T={t} k={edge}: {lo} vs {hi}");
            // constant in k beyond k_max - DK (clamped stencil)
            let far = lv.vol(edge + edge.signum(), t).unwrap();
            assert_eq!(far, if edge > 0.0 { hi } else { lo });
        }
    }
    assert_eq!(lv.floor_count(), 0);
    assert_eq!(lv.cap_count(), 0);
    assert_close(lv.vol(0.5, 1.0).unwrap(), 0.19703445, 1e-6, "wing vol at T=1");
}

#[test]
fn term_structure_local_vol_is_forward_variance() {
    // Flat in k, iv = {0.15, 0.20, 0.25} at T = {0.25, 0.5, 1}.
    let ks: Vec<f64> = (0..11).map(|i| -0.5 + 0.1 * i as f64).collect();
    let ts = [0.25, 0.5, 1.0];
    let vols = vec![vec![0.15; 11], vec![0.20; 11], vec![0.25; 11]];
    let lv = DupireLocalVol::new(ImpliedVolSurface::new(&ks, &ts, &vols).unwrap());
    assert_close(lv.vol(0.0, 0.4).unwrap(), 0.0575_f64.sqrt(), 1e-6, "(0.02-0.005625)/0.25");
    assert_close(lv.vol(0.3, 0.4).unwrap(), 0.0575_f64.sqrt(), 1e-6, "flat in k");
    assert_close(lv.vol(0.0, 0.75).unwrap(), 0.085_f64.sqrt(), 1e-6, "(0.0625-0.02)/0.5");
    assert_close(lv.vol(0.0, 0.1).unwrap(), 0.15, 1e-6, "below first pillar");
    assert_close(lv.vol(0.0, 2.0).unwrap(), 0.085_f64.sqrt(), 1e-6, "beyond last pillar");
    assert_eq!(lv.floor_count(), 0);
    assert_eq!(lv.cap_count(), 0);
}

#[test]
fn zero_expiry_is_short_time_limit() {
    // MAJ-1: before the fix vol(-0.3, 0) = 0.309 vs vol(-0.3, 1e-6) = 0.452.
    let lv = bundled_localvol();
    let srf = lv.surface();
    for k in [-0.3, -0.1, 0.0, 0.2, 0.45] {
        let v0 = lv.vol(k, 0.0).unwrap();
        assert!((v0 - lv.vol(k, 1e-6).unwrap()).abs() < 1e-3, "k={k}");
        assert!((v0 - lv.vol(k, 1e-4).unwrap()).abs() < 1e-3, "k={k}");
        // independent BBF evaluation from the short-end implied slice
        let s = srf.implied_vol(k, 0.0).unwrap();
        let sp = (srf.implied_vol(k + 1e-3, 0.0).unwrap() - srf.implied_vol(k - 1e-3, 0.0).unwrap())
            / 2e-3;
        assert_close(v0, s / (1.0 - k * sp / s), 1e-12, "BBF");
    }
    assert_close(lv.vol(-0.3, 0.0).unwrap(), 0.45240187, 1e-6, "vol(-0.3, 0)");
    assert_close(lv.vol(0.0, 0.0).unwrap(), 0.2, 1e-12, "ATM T=0");
}

fn strongly_convex_surface() -> ImpliedVolSurface {
    // At |k| = 0.5, k s'/s = 0.5 * 4 / 1.05 = 1.9 > 1 -> denominator <= 0 -> cap.
    let ks: Vec<f64> = (0..21).map(|i| -0.5 + 0.05 * i as f64).collect();
    let v1: Vec<f64> = ks.iter().map(|&k| 0.05 + 4.0 * k * k).collect();
    let v2: Vec<f64> = v1.iter().map(|&v| v * 1.05).collect();
    ImpliedVolSurface::new(&ks, &[0.5, 1.0], &[v1, v2]).unwrap()
}

#[test]
fn zero_expiry_caps_on_strongly_convex_smile() {
    let lv = DupireLocalVol::new(strongly_convex_surface());
    for i in 0..21 {
        let k = -0.5 + 0.05 * i as f64;
        let v = lv.vol(k, 0.0).unwrap();
        assert!((FLOOR..=CAP).contains(&v), "k={k}: {v}");
    }
    assert!(lv.cap_count() > 0);
    assert_eq!(lv.vol(0.5, 0.0).unwrap(), CAP);
    assert_close(lv.vol(0.0, 0.0).unwrap(), 0.05, 1e-9, "ATM T=0");
}

#[test]
fn counters_are_thread_safe_and_type_is_send_sync() {
    // MAJ-3: DupireLocalVol must be shareable across threads (rayon over
    // strikes) and the counters exact under concurrency.
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<DupireLocalVol>();
    assert_send_sync::<ImpliedVolSurface>();

    let lv = std::sync::Arc::new(DupireLocalVol::new(strongly_convex_surface()));
    let work = |lv: &DupireLocalVol| {
        for i in 0..100_000 {
            let k = -0.5 + (i % 101) as f64 * 0.01;
            lv.vol(k, 0.75).unwrap();
        }
    };
    work(&lv);
    let (single_floor, single_cap) = (lv.floor_count(), lv.cap_count());
    assert!(single_cap > 0);
    lv.reset_counters();
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let lv = std::sync::Arc::clone(&lv);
            std::thread::spawn(move || work(&lv))
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    assert_eq!(lv.floor_count(), 4 * single_floor);
    assert_eq!(lv.cap_count(), 4 * single_cap);
}
