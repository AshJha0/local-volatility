//! Implied-volatility surface in (log-moneyness, expiry) total-variance form.
//!
//! The surface is parametrised by *forward* log-moneyness `k = ln(K / F(T))`
//! and stores **total implied variance** `w(k, T) = iv(k, T)^2 * T`.
//! Working in `(k, w)` has three payoffs:
//!
//! * the Dupire formula takes its cleanest (Gatheral) form in these variables;
//! * calendar-arbitrage is simply monotonicity of `w` in `T` at fixed `k`;
//! * the surface is independent of rates/dividends, so one CSV serves equity
//!   and FX alike (the pricer converts strikes via its own forward curve).
//!
//! Interpolation ("bicubic-lite"): a **natural cubic spline** in `k` per
//! expiry pillar (its C^2 smoothness is exactly what Dupire's `d2w/dk2`
//! needs; the moments come from a tridiagonal system solved with the same
//! Thomas kernel the PDE uses), and **linear total variance in T** between
//! pillars (the standard "no calendar arbitrage between pillars" rule).
//!
//! Extrapolation: flat in `k` beyond the wings (queries are clamped);
//! proportional total variance below the first pillar
//! (`w(k,T) = w(k,T1) * T/T1`, i.e. constant implied vol down to T=0);
//! linear-in-T continuation of the last-interval slope beyond the last
//! pillar, slope floored at 0 so `w` never decreases.  A single-expiry
//! surface uses the proportional rule on both sides and logs a warning
//! because `dw/dT` then rests on an assumption rather than data.
//!
//! Calendar arbitrage is checked node-by-node at construction: any
//! `w(k_i, T_{j+1}) < w(k_i, T_j) - 1e-12` is counted in
//! [`ImpliedVolSurface::calendar_violations`] and reported via a log line —
//! detected and reported, not silently repaired.  The same policy applies to
//! spline overshoot: every node interval of every pillar is probed at 15
//! interior points and any `w <= 0` is counted in
//! [`ImpliedVolSurface::negative_w_count`] (`implied_vol` reads 0% there).
//! The surface is immutable after construction and `Send + Sync`.

use std::path::Path;

use crate::error::{invalid, LocalVolError, Result};
use crate::tridiag::thomas_solve;

const CAL_TOL: f64 = 1e-12;
/// Spline-overshoot scan: probes per node interval (minus one).
const NEG_W_SUBDIV: usize = 16;

/// Natural cubic spline through `(x_i, y_i)`.
///
/// Degenerate node counts degrade gracefully: 1 node -> constant,
/// 2 nodes -> linear.  Evaluation clamps the query to `[x_0, x_{n-1}]`
/// (flat extrapolation), matching the surface's wing rule.
#[derive(Debug, Clone)]
pub struct CubicSpline1D {
    x: Vec<f64>,
    y: Vec<f64>,
    /// Second-derivative "moments" at the nodes; `m[0] = m[n-1] = 0`.
    m: Vec<f64>,
}

impl CubicSpline1D {
    /// Build the spline; `x` must be strictly increasing and both arrays
    /// finite, non-empty and of equal length.
    pub fn new(x: &[f64], y: &[f64]) -> Result<Self> {
        if x.len() != y.len() {
            return Err(invalid(format!(
                "spline: x and y must be equal-length, got {} vs {}",
                x.len(),
                y.len()
            )));
        }
        if x.is_empty() {
            return Err(invalid("spline: need at least one node"));
        }
        if !(x.iter().all(|v| v.is_finite()) && y.iter().all(|v| v.is_finite())) {
            return Err(invalid("spline: non-finite node data"));
        }
        if x.windows(2).any(|p| p[1] <= p[0]) {
            return Err(invalid("spline: x nodes must be strictly increasing"));
        }
        let n = x.len();
        let mut m = vec![0.0_f64; n];
        if n > 2 {
            // Natural spline: interior moments solve the classical
            // symmetric tridiagonal system (h_i = x_{i+1} - x_i):
            //   (h_{i-1}/6) m_{i-1} + ((h_{i-1}+h_i)/3) m_i + (h_i/6) m_{i+1}
            //       = (y_{i+1}-y_i)/h_i - (y_i-y_{i-1})/h_{i-1}
            let h: Vec<f64> = x.windows(2).map(|p| p[1] - p[0]).collect();
            let slopes: Vec<f64> = (0..n - 1).map(|i| (y[i + 1] - y[i]) / h[i]).collect();
            let sub: Vec<f64> = (1..n - 2).map(|i| h[i] / 6.0).collect();
            let diag: Vec<f64> = (0..n - 2).map(|i| (h[i] + h[i + 1]) / 3.0).collect();
            let sup: Vec<f64> = (1..n - 2).map(|i| h[i] / 6.0).collect();
            let rhs: Vec<f64> = (0..n - 2).map(|i| slopes[i + 1] - slopes[i]).collect();
            let interior = thomas_solve(&sub, &diag, &sup, &rhs)?;
            m[1..n - 1].copy_from_slice(&interior);
        }
        Ok(CubicSpline1D {
            x: x.to_vec(),
            y: y.to_vec(),
            m,
        })
    }

    /// Evaluate at `xq` (must be finite); the query is clamped into the
    /// node range (flat extrapolation).
    pub fn eval(&self, xq: f64) -> Result<f64> {
        if !xq.is_finite() {
            return Err(invalid("spline: non-finite query"));
        }
        Ok(self.eval_clamped(xq))
    }

    /// Hot-path evaluation without the finiteness check (callers validate).
    pub(crate) fn eval_clamped(&self, xq: f64) -> f64 {
        let n = self.x.len();
        if n == 1 {
            return self.y[0];
        }
        let q = xq.clamp(self.x[0], self.x[n - 1]);
        // Interval index: largest i with x[i] <= q, capped at n-2.
        let i = self.x.partition_point(|&v| v <= q).saturating_sub(1).min(n - 2);
        let h = self.x[i + 1] - self.x[i];
        let t = (q - self.x[i]) / h;
        let u = 1.0 - t;
        // s(x) = y_i u + y_{i+1} t + h^2/6 [ (u^3 - u) m_i + (t^3 - t) m_{i+1} ]
        self.y[i] * u
            + self.y[i + 1] * t
            + (h * h / 6.0) * ((u * u * u - u) * self.m[i] + (t * t * t - t) * self.m[i + 1])
    }
}

/// Total-variance implied surface with the interpolation rules above.
#[derive(Debug, Clone)]
pub struct ImpliedVolSurface {
    k_nodes: Vec<f64>,
    expiries: Vec<f64>,
    /// Implied vols, row `j` = expiry `T_j`, column `i` = node `k_i`.
    vols: Vec<Vec<f64>>,
    /// Total variance at the nodes: `w = iv^2 * T`.
    w_nodes: Vec<Vec<f64>>,
    splines: Vec<CubicSpline1D>,
    calendar_violations: usize,
    negative_w_count: usize,
    single_expiry: bool,
}

impl ImpliedVolSurface {
    /// Build from strictly increasing `k_nodes`, strictly increasing
    /// positive `expiries`, and `vols[j][i] > 0` of shape
    /// `(expiries.len(), k_nodes.len())`.
    pub fn new(k_nodes: &[f64], expiries: &[f64], vols: &[Vec<f64>]) -> Result<Self> {
        if k_nodes.is_empty() {
            return Err(invalid("k_nodes must be non-empty"));
        }
        if expiries.is_empty() {
            return Err(invalid("expiries must be non-empty"));
        }
        if vols.len() != expiries.len() || vols.iter().any(|row| row.len() != k_nodes.len()) {
            return Err(invalid(format!(
                "vols must have shape (n_expiries, n_k) = ({}, {})",
                expiries.len(),
                k_nodes.len()
            )));
        }
        let all_finite = k_nodes.iter().all(|v| v.is_finite())
            && expiries.iter().all(|v| v.is_finite())
            && vols.iter().flatten().all(|v| v.is_finite());
        if !all_finite {
            return Err(invalid("surface inputs must be finite"));
        }
        if k_nodes.windows(2).any(|p| p[1] <= p[0]) {
            return Err(invalid("k_nodes must be strictly increasing"));
        }
        if expiries.iter().any(|&t| t <= 0.0) {
            return Err(invalid("expiries must be strictly positive"));
        }
        if expiries.windows(2).any(|p| p[1] <= p[0]) {
            return Err(invalid("expiries must be strictly increasing"));
        }
        if vols.iter().flatten().any(|&v| v <= 0.0) {
            return Err(invalid("implied vols must be strictly positive"));
        }

        let w_nodes: Vec<Vec<f64>> = vols
            .iter()
            .zip(expiries)
            .map(|(row, &t)| row.iter().map(|&iv| iv * iv * t).collect())
            .collect();

        // Calendar-arbitrage detection (report, don't fail).
        let mut calendar_violations = 0usize;
        for j in 0..expiries.len().saturating_sub(1) {
            for i in 0..k_nodes.len() {
                if w_nodes[j + 1][i] < w_nodes[j][i] - CAL_TOL {
                    calendar_violations += 1;
                }
            }
        }
        if calendar_violations > 0 {
            eprintln!(
                "warning: calendar arbitrage: total variance decreases in T at \
                 {calendar_violations} node(s)"
            );
        }
        let single_expiry = expiries.len() == 1;
        if single_expiry {
            eprintln!(
                "warning: single expiry pillar: dw/dT uses the flat-forward-variance assumption"
            );
        }

        let splines = w_nodes
            .iter()
            .map(|row| CubicSpline1D::new(k_nodes, row))
            .collect::<Result<Vec<_>>>()?;

        // Spline-overshoot detection (report, don't fail): probe every node
        // interval of every pillar at NEG_W_SUBDIV - 1 interior points and
        // count w <= 0 (implied_vol would silently read 0% there).
        let mut negative_w_count = 0usize;
        for sp in &splines {
            for pair in k_nodes.windows(2) {
                let h = pair[1] - pair[0];
                for p in 1..NEG_W_SUBDIV {
                    if sp.eval_clamped(pair[0] + h * (p as f64 / NEG_W_SUBDIV as f64)) <= 0.0 {
                        negative_w_count += 1;
                    }
                }
            }
        }
        if negative_w_count > 0 {
            eprintln!(
                "warning: spline overshoot: total variance <= 0 at {negative_w_count} probe \
                 point(s) between nodes (implied vol reads as 0% there)"
            );
        }

        Ok(ImpliedVolSurface {
            k_nodes: k_nodes.to_vec(),
            expiries: expiries.to_vec(),
            vols: vols.to_vec(),
            w_nodes,
            splines,
            calendar_violations,
            negative_w_count,
            single_expiry,
        })
    }

    /// Load from a CSV with header exactly `T,k,iv` (one row per grid node,
    /// three columns).  The file must contain a full rectangular grid: every
    /// `(T, k)` pair exactly once (row order irrelevant); duplicates, gaps,
    /// short/long/non-numeric/non-finite rows are `InvalidInput` errors.
    pub fn from_csv<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| LocalVolError::Io(format!("{}: {e}", path.display())))?;
        let mut lines = text.lines();
        let header = lines
            .next()
            .ok_or_else(|| invalid(format!("{}: empty surface CSV", path.display())))?;
        let cols: Vec<&str> = header.split(',').map(str::trim).collect();
        if cols != ["T", "k", "iv"] {
            return Err(invalid(format!(
                "{}: CSV must have header columns T,k,iv",
                path.display()
            )));
        }
        let mut rows: Vec<(f64, f64, f64)> = Vec::new();
        for (ln, line) in lines.enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split(',').map(str::trim).collect();
            if parts.len() != 3 {
                return Err(invalid(format!(
                    "{}: line {}: expected 3 columns, got {}",
                    path.display(),
                    ln + 2,
                    parts.len()
                )));
            }
            let parse = |s: &str, what: &str| -> Result<f64> {
                s.parse::<f64>().map_err(|_| {
                    invalid(format!(
                        "{}: line {}: bad {what} value {s:?}",
                        path.display(),
                        ln + 2
                    ))
                })
            };
            let row = (
                parse(parts[0], "T")?,
                parse(parts[1], "k")?,
                parse(parts[2], "iv")?,
            );
            if !(row.0.is_finite() && row.1.is_finite() && row.2.is_finite()) {
                return Err(invalid(format!(
                    "{}: line {}: non-finite CSV value",
                    path.display(),
                    ln + 2
                )));
            }
            rows.push(row);
        }
        if rows.is_empty() {
            return Err(invalid(format!("{}: empty surface CSV", path.display())));
        }
        let mut ts: Vec<f64> = rows.iter().map(|r| r.0).collect();
        let mut ks: Vec<f64> = rows.iter().map(|r| r.1).collect();
        let sort_dedup = |v: &mut Vec<f64>| {
            v.sort_by(|a, b| a.total_cmp(b));
            v.dedup();
        };
        sort_dedup(&mut ts);
        sort_dedup(&mut ks);
        let mut vols = vec![vec![f64::NAN; ks.len()]; ts.len()];
        let mut seen = vec![vec![false; ks.len()]; ts.len()];
        for (t, k, iv) in rows {
            let j = ts.partition_point(|&v| v < t);
            let i = ks.partition_point(|&v| v < k);
            if seen[j][i] {
                return Err(invalid(format!(
                    "{}: duplicate (T,k) row T={t}, k={k}",
                    path.display()
                )));
            }
            seen[j][i] = true;
            vols[j][i] = iv;
        }
        if seen.iter().flatten().any(|&s| !s) {
            return Err(invalid(format!(
                "{}: surface grid is not rectangular (missing (T,k) pairs)",
                path.display()
            )));
        }
        ImpliedVolSurface::new(&ks, &ts, &vols)
    }

    /// Total implied variance `w(k, T)` under the documented rules;
    /// `w(k, 0) = 0` exactly.
    pub fn total_variance(&self, k: f64, expiry: f64) -> Result<f64> {
        if !expiry.is_finite() || expiry < 0.0 {
            return Err(invalid(format!("expiry must be finite and >= 0, got {expiry}")));
        }
        if !k.is_finite() {
            return Err(invalid("k must be finite"));
        }
        Ok(self.total_variance_unchecked(k, expiry))
    }

    /// Hot-path variant: `k` finite and `expiry >= 0` already guaranteed.
    pub(crate) fn total_variance_unchecked(&self, k: f64, expiry: f64) -> f64 {
        let t = &self.expiries;
        let n = t.len();
        if expiry == 0.0 {
            return 0.0;
        }
        if self.single_expiry || expiry <= t[0] {
            // Flat forward variance: w scales proportionally with T.  For a
            // single-expiry surface this rule applies on both sides of T1.
            self.splines[0].eval_clamped(k) * (expiry / t[0])
        } else if expiry >= t[n - 1] {
            let w_last = self.splines[n - 1].eval_clamped(k);
            let w_prev = self.splines[n - 2].eval_clamped(k);
            let slope = (w_last - w_prev) / (t[n - 1] - t[n - 2]);
            w_last + slope.max(0.0) * (expiry - t[n - 1])
        } else {
            let j = t.partition_point(|&v| v <= expiry) - 1;
            let lam = (expiry - t[j]) / (t[j + 1] - t[j]);
            (1.0 - lam) * self.splines[j].eval_clamped(k) + lam * self.splines[j + 1].eval_clamped(k)
        }
    }

    /// Implied vol `sqrt(w(k,T)/T)`; at `T == 0` the short-end limit
    /// `sqrt(w(k, T1)/T1)` is returned (the proportional rule's limit).
    pub fn implied_vol(&self, k: f64, expiry: f64) -> Result<f64> {
        if !expiry.is_finite() || expiry < 0.0 {
            return Err(invalid(format!("expiry must be finite and >= 0, got {expiry}")));
        }
        if !k.is_finite() {
            return Err(invalid("k must be finite"));
        }
        Ok(self.implied_vol_unchecked(k, expiry))
    }

    /// Hot-path variant: `k` finite and `expiry >= 0` already guaranteed.
    pub(crate) fn implied_vol_unchecked(&self, k: f64, expiry: f64) -> f64 {
        if expiry == 0.0 {
            let t0 = self.expiries[0];
            let w1 = self.total_variance_unchecked(k, t0);
            return (w1 / t0).sqrt();
        }
        let w = self.total_variance_unchecked(k, expiry);
        (w.max(0.0) / expiry).sqrt()
    }

    /// Node grid in `k` (strictly increasing).
    pub fn k_nodes(&self) -> &[f64] {
        &self.k_nodes
    }

    /// Pillar expiries (strictly increasing, positive).
    pub fn expiries(&self) -> &[f64] {
        &self.expiries
    }

    /// Node implied vols (row per expiry).
    pub fn vols(&self) -> &[Vec<f64>] {
        &self.vols
    }

    /// Node total variances `iv^2 T` (row per expiry).
    pub fn w_nodes(&self) -> &[Vec<f64>] {
        &self.w_nodes
    }

    /// Number of node pairs where total variance decreases with expiry
    /// (calendar arbitrage), detected at build time.
    pub fn calendar_violations(&self) -> usize {
        self.calendar_violations
    }

    /// Number of probe points (15 per node interval, per pillar) where the
    /// natural spline overshoots to total variance `<= 0` (`implied_vol`
    /// reads 0% there).  Detected at construction and logged, never
    /// repaired; 0 for a well-behaved surface.
    pub fn negative_w_count(&self) -> usize {
        self.negative_w_count
    }

    /// True when only one pillar was supplied (flat forward variance in T).
    pub fn single_expiry(&self) -> bool {
        self.single_expiry
    }

    /// Smallest `k` node.
    pub fn k_min(&self) -> f64 {
        self.k_nodes[0]
    }

    /// Largest `k` node.
    pub fn k_max(&self) -> f64 {
        *self.k_nodes.last().expect("non-empty by construction")
    }
}
