//! Thomas solver vs a dense Gaussian-elimination reference, plus validation.

use localvol::thomas_solve;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Dense Gaussian elimination with partial pivoting — independent check
/// only (the pricing path uses the native Thomas kernel exclusively).
fn dense_solve(sub: &[f64], diag: &[f64], sup: &[f64], rhs: &[f64]) -> Vec<f64> {
    let n = diag.len();
    let mut a = vec![vec![0.0_f64; n]; n];
    let mut b = rhs.to_vec();
    for i in 0..n {
        a[i][i] = diag[i];
        if i + 1 < n {
            a[i + 1][i] = sub[i];
            a[i][i + 1] = sup[i];
        }
    }
    for col in 0..n {
        let pivot_row = (col..n)
            .max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))
            .unwrap();
        a.swap(col, pivot_row);
        b.swap(col, pivot_row);
        for row in col + 1..n {
            let f = a[row][col] / a[col][col];
            for k in col..n {
                a[row][k] -= f * a[col][k];
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut acc = b[i];
        for j in i + 1..n {
            acc -= a[i][j] * x[j];
        }
        x[i] = acc / a[i][i];
    }
    x
}

#[test]
fn matches_dense_solve_on_random_systems() {
    // Property-style: random diagonally-dominant systems of many sizes.
    let mut rng = StdRng::seed_from_u64(123);
    for n in [1usize, 2, 3, 5, 17, 64, 201] {
        for _ in 0..3 {
            let sub: Vec<f64> = (0..n - 1).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let sup: Vec<f64> = (0..n - 1).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let diag: Vec<f64> = (0..n)
                .map(|_| rng.gen_range(2.5..4.0) * if rng.gen_bool(0.5) { 1.0 } else { -1.0 })
                .collect();
            let rhs: Vec<f64> = (0..n).map(|_| rng.gen_range(-5.0..5.0)).collect();
            let x = thomas_solve(&sub, &diag, &sup, &rhs).unwrap();
            let x_ref = dense_solve(&sub, &diag, &sup, &rhs);
            for i in 0..n {
                assert!(
                    (x[i] - x_ref[i]).abs() <= 1e-12 * (1.0 + x_ref[i].abs()),
                    "n={n} i={i}: {} vs {}",
                    x[i],
                    x_ref[i]
                );
            }
        }
    }
}

#[test]
fn residual_is_tiny_on_a_large_system() {
    let mut rng = StdRng::seed_from_u64(7);
    let n = 120usize;
    let sub: Vec<f64> = (0..n - 1).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let sup: Vec<f64> = (0..n - 1).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let diag: Vec<f64> = (0..n).map(|_| 4.0 + rng.gen_range(0.0..1.0)).collect();
    let rhs: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let x = thomas_solve(&sub, &diag, &sup, &rhs).unwrap();
    for i in 0..n {
        let mut ax = diag[i] * x[i];
        if i > 0 {
            ax += sub[i - 1] * x[i - 1];
        }
        if i + 1 < n {
            ax += sup[i] * x[i + 1];
        }
        assert!((ax - rhs[i]).abs() < 1e-11, "row {i} residual {}", ax - rhs[i]);
    }
}

#[test]
fn identity_and_size_one() {
    let x = thomas_solve(&[0.0, 0.0], &[1.0, 1.0, 1.0], &[0.0, 0.0], &[1.0, 2.0, 3.0]).unwrap();
    assert_eq!(x, vec![1.0, 2.0, 3.0]);
    let y = thomas_solve(&[], &[2.0], &[], &[8.0]).unwrap();
    assert_eq!(y, vec![4.0]);
}

#[test]
fn shape_validation() {
    // bad sub length
    assert!(thomas_solve(&[0.0], &[1.0, 1.0, 1.0], &[0.0, 0.0], &[1.0, 1.0, 1.0]).is_err());
    // bad rhs length
    assert!(thomas_solve(&[0.0, 0.0], &[1.0, 1.0, 1.0], &[0.0, 0.0], &[1.0; 4]).is_err());
    // empty
    assert!(thomas_solve(&[], &[], &[], &[]).is_err());
}

#[test]
fn rejects_nonfinite_and_zero_pivot() {
    let err = thomas_solve(&[0.0], &[1.0, f64::NAN], &[0.0], &[1.0, 1.0]).unwrap_err();
    assert!(err.to_string().contains("non-finite"), "{err}");
    let err = thomas_solve(&[0.0], &[0.0, 1.0], &[0.0], &[1.0, 1.0]).unwrap_err();
    assert!(err.to_string().contains("pivot"), "{err}");
    // elimination-induced zero pivot: [[1,1],[1,1]] is singular
    let err = thomas_solve(&[1.0], &[1.0, 1.0], &[1.0], &[1.0, 1.0]).unwrap_err();
    assert!(err.to_string().contains("pivot"), "{err}");
}
