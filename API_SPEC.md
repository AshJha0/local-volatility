# localvol — API & Numerical Contract (language-neutral)

This document is the **normative** specification for the C++, Rust and Java
ports of the Python reference implementation in `python/src/localvol/`.
A port implemented from this document alone MUST reproduce every case in
`data/golden/golden.json` within its stated tolerance.  All arithmetic is
IEEE-754 double precision.

Names: C++ namespace `localvol`, Rust crate `localvol`,
Java package `com.quant.localvol`, Python package `localvol`.

---

## 1. Conventions and coordinates

| Symbol | Meaning |
|---|---|
| `S0` | spot (equity price or FX rate) |
| `r`  | discount rate (equity `r`, FX domestic rate `rd`) |
| `q`  | carry rate (equity dividend yield, FX foreign rate `rf`) |
| `F(t)` | forward `S0 * exp((r - q) * t)` |
| `k`  | **forward log-moneyness** `ln(K / F(T))` |
| `w(k, T)` | **total implied variance** `iv(k, T)^2 * T` |

Equity and FX (Garman-Kohlhagen) share every formula: FX uses `r = rd`,
`q = rf`.  The market type is `{spot > 0, rate, dividend}`; `rate` and
`dividend` may be negative; all fields must be finite.  `forward(T)` and
`log_forward(T)` both validate `T` finite and `>= 0` in every port.

**Time convention.** Every `T`, `t` and `expiry` is a bare year fraction;
the library carries no day-count convention, calendar, settlement lag or
date type.  Converting dates to year fractions (ACT/365F, business days,
…) is the caller's responsibility and must be done consistently for the
surface CSV, the pricer inputs and the rates.  Rates are continuously
compounded.

Normal CDF: `N(x) = 0.5 * erfc(-x / sqrt(2))` (use an `erfc` with relative
error ≤ 1e-12; every target language has one in its standard library or a
standard 20-line implementation).

### Error behaviour

Invalid input (non-finite values; `spot <= 0`; `strike < 0`; `sigma < 0`;
`expiry < 0`; non-increasing grids; non-positive vols; shape mismatches;
duplicated or missing CSV grid points; `num_space` odd or `< 4`;
`num_time < 1`; `omega` outside `(0,2)`; `tol` non-finite or `<= 0`;
`max_iter < 1`; `n_paths` below the minimum in §7; a user-supplied vol
callable returning a non-finite or negative sigma; price outside
no-arbitrage bounds in the implied-vol inverter, …) is rejected eagerly
with the language's idiomatic error:

* Python: `raise ValueError(message)`
* C++: `throw std::invalid_argument` (domain violations may use `std::domain_error`)
* Rust: `Result<T, LocalVolError>` (manual thiserror-style enum; never panic on bad input)
* Java: `throw IllegalArgumentException`

Three conditions are **reported, not raised**: calendar-arbitrage in the
input surface (violation count exposed on the surface object +
warning/log), spline overshoot to `w <= 0` between nodes (§3.3, count +
warning/log) and PSOR non-convergence (warning/log, computation proceeds).
Dupire clamps (§4) are counted, never raised.

---

## 2. Data files

### 2.1 Surface CSV (`data/implied_surface.csv`, `data/flat_surface.csv`)

Header exactly `T,k,iv` (three columns, whitespace around names ignored);
one row of exactly three numeric fields per grid node; full rectangular
grid (every `(T,k)` pair exactly once, order irrelevant).  `T` = expiry in
years, `k` = forward log-moneyness, `iv` = Black implied vol.  The bundled
surface samples an SSVI family (`theta_T = 0.04 T`, `rho = -0.5`,
`eta = 0.7`, `gamma = 0.5`) on expiries `{0.25, 0.5, 1, 2}` × 15
equidistant `k` in `[-0.5, 0.5]`; `flat_surface.csv` is 20% everywhere on
the same grid.  Loaders must: parse numbers locale-independently; tolerate
blank lines, surrounding spaces and CRLF; sort the distinct `T` and `k`
values ascending; and reject — with the idiomatic error, never an
index/parse crash — a wrong header, a row with other than three fields, a
non-numeric or non-finite field, a **duplicated** `(T,k)` pair and a
missing pair (non-rectangular grid).

### 2.2 Golden file (`data/golden/golden.json`)

```json
{ "cases": [ {"name": "...", "inputs": {...}, "expect": {...}, "tol": 1e-4} ] }
```

All values inside `inputs`/`expect` are flat scalars.  Cases whose name
contains `bundled` operate on `data/implied_surface.csv`; names starting
`flat_dupire_` operate on `data/flat_surface.csv`; all other cases are
self-contained.  Comparison is always **absolute**: `|got - expect| <= tol`
(relative tolerances were pre-multiplied at generation time).  Dispatch by
name prefix — the exact mapping is the table in §9.

---

## 3. Implied surface

Construction from nodes `k_0 < … < k_{m-1}`, `0 < T_0 < … < T_{n-1}`, vols
`iv[j][i] > 0`: store `w[j][i] = iv[j][i]^2 * T_j` and build **one natural
cubic spline in k per expiry row**.

### 3.1 Natural cubic spline (exact algorithm)

Given nodes `x_0 < … < x_{p-1}`, values `y_i`, define `h_i = x_{i+1} - x_i`.
Second-derivative "moments" `m_i` satisfy `m_0 = m_{p-1} = 0` and, for
`i = 1 … p-2`,

```
(h_{i-1}/6) m_{i-1} + ((h_{i-1}+h_i)/3) m_i + (h_i/6) m_{i+1}
    = (y_{i+1}-y_i)/h_i - (y_i-y_{i-1})/h_{i-1}
```

— a `(p-2)`-dimensional tridiagonal system solved with the Thomas kernel
(§6.1).  Degenerate sizes: `p = 1` → constant, `p = 2` → linear (`m = 0`).
Evaluation at `x` (after clamping `x` into `[x_0, x_{p-1}]`): find `i` with
`x_i <= x <= x_{i+1}` (`i = p-2` at the right end), set `t = (x-x_i)/h_i`,
`u = 1 - t`, and

```
s(x) = y_i u + y_{i+1} t + (h_i^2/6) [ (u^3 - u) m_i + (t^3 - t) m_{i+1} ]
```

### 3.2 Total variance `w(k, T)` (query rules, in this order)

1. Validate: `k` finite, `T` finite and `>= 0`.  `T == 0` → `w = 0`.
2. Clamp `k` to `[k_0, k_{m-1}]` (**flat wing extrapolation**).
3. **Single-expiry surface** (`n == 1`), or `T <= T_0`:
   `w = spline_0(k) * T / T_0` (flat forward variance; for a single-expiry
   surface this rule applies for all `T`, and a warning is emitted at build
   time because `dw/dT` then rests on an assumption).
4. `T >= T_{n-1}` (multi-expiry): with `s = (spline_{n-1}(k) - spline_{n-2}(k)) / (T_{n-1} - T_{n-2})`,
   `w = spline_{n-1}(k) + max(s, 0) * (T - T_{n-1})` (**linear-in-T total
   variance beyond the last pillar**, slope floored at 0).
5. Otherwise find `j` with `T_j <= T < T_{j+1}` and interpolate **linearly
   in total variance**: `lam = (T - T_j)/(T_{j+1} - T_j)`,
   `w = (1-lam) spline_j(k) + lam spline_{j+1}(k)`.

`implied_vol(k, T) = sqrt(w(k, T) / T)` for `T > 0`; at `T == 0` return the
short-end limit `sqrt(spline_0(clamped k) / T_0 … )` i.e.
`sqrt(w(k, T_0)/T_0)`.

### 3.3 Calendar-arbitrage check (at construction)

Count every node pair with `w[j+1][i] < w[j][i] - 1e-12`; expose the count
(`calendar_violations`) and warn/log if positive.  Do **not** repair or
throw.

### 3.4 Spline-overshoot scan (at construction)

A natural spline through ragged nodes can dip below zero between nodes,
which `implied_vol` would turn into a silent 0% (`sqrt(max(w,0)/T)`).  For
every pillar `j` and every node interval `[k_i, k_{i+1}]` evaluate
`spline_j(k_i + h_i * m / 16)` for `m = 1 … 15` and count the probes with
`w <= 0`; expose the count (`negative_w_count`) and warn/log if positive.
Do **not** repair or throw.  The surface object is immutable after
construction and safe to share across threads.

---

## 4. Dupire local volatility

Gatheral total-variance form.  With `w = w(k,T)` and derivatives below:

```
sigma_loc^2(k,T) = (dw/dT) /
    ( 1 - (k/w) dw/dk
        + 1/4 (-1/4 - 1/w + k^2/w^2) (dw/dk)^2
        + 1/2 d2w/dk2 )
```

**Step 0 — wing clamp of the query point (exact).**  With `k_0`, `k_{m-1}`
the first/last quoted strikes: if `k_{m-1} - k_0 > 2 DK`, replace `k` by
`clamp(k, k_0 + DK, k_{m-1} - DK)` *before* anything else and use the
clamped `k` in every term below (the stencil and the `k/w`, `k^2/w^2`
factors).  Rationale: the surface is flat in `k` beyond the quoted box, so
`w` is only C^0 at `k_0` and `k_{m-1}` (natural spline: `w'' = 0` but
`w' != 0`); a central stencil straddling that kink reads
`d2w/dk2 ~ -w'/DK`, the denominator goes negative and the vol is capped at
500% *at the quoted wing* — 2.5 sd from ATM on the bundled surface.  After
the clamp local vol is constant in `k` beyond `k_{m-1} - DK` and continuous
across the wing node; on the bundled surface no clamp fires anywhere on a
200×200 PDE grid.  (A degenerate box narrower than `2 DK` is left
unclamped.)

**Step 1 — `T == 0`: short-time limit (exact).**  `w(k, 0) = 0`, so the
formula is undefined there; `sigma_loc(k, 0)` is *defined* as the
`T -> 0+` limit (Berestycki, Busca, Florent 2002).  With
`s(k) = implied_vol(k, 0) = sqrt(w(k, T_0)/T_0)` (§3.2) and
`s'(k) = (s(k+DK) - s(k-DK)) / (2 DK)`:

```
den0 = 1 - k s'(k) / max(s(k), 1e-12)
sigma_loc(k, 0) = s(k) / den0
```

fed through the clamp policy below with `num = s^2` and
`den = den0 * |den0|` (so `sqrt(num/den) = s/den0` when `den0 > 0`,
`s == 0` floors and `den0 <= 0` caps).  Because the surface uses flat
forward variance below the first pillar, the `T > 0` branch converges to
this value as `T -> 0`; `vol` is continuous at `T = 0`.

**Step 2 — `T > 0`: finite differences on the interpolated surface — exact
steps:**

* `DK = 1e-3`:
  `dw/dk  = (w(k+DK,T) - w(k-DK,T)) / (2 DK)`,
  `d2w/dk2 = (w(k+DK,T) - 2 w(k,T) + w(k-DK,T)) / DK^2`.
* `DT = 1e-4`: if `T > DT`, central:
  `dw/dT = (w(k,T+DT) - w(k,T-DT)) / (2 DT)`;
  else forward: `dw/dT = (w(k,T+DT) - w(k,T)) / DT`.

**Guards & clamping** (evaluate numerator `num = dw/dT` and denominator
`den` with `w` replaced by `max(w, 1e-12)` inside `1/w`, `k/w`, `k^2/w^2`):

* `num <= 0`  → raw variance := `FLOOR^2` and count a **floor** hit.
* else `den <= 0` → raw variance := `CAP^2` and count a **cap** hit.
* else raw vol := `sqrt(num/den)`; if `< FLOOR` count floor, if `> CAP`
  count cap; clamp into `[FLOOR, CAP]`.

`FLOOR = 0.01` (1%), `CAP = 5.0` (500%).  Cumulative floor/cap counters are
exposed and resettable; they are reporting, never errors.  The counters
must be safe to increment from several threads at once (Python is
single-threaded per interpreter; C++ `std::atomic<long>`, Rust
`AtomicU64`, Java `LongAdder`, relaxed ordering): `vol()` on one shared
object is thread-safe, the counters' total is exact, and a read during a
concurrent sweep is a snapshot.  The C++ object stores a reference to the
surface and must refuse to bind a temporary (`ImpliedVolSurface&&`
constructor deleted).

**Time discontinuity (documented).**  Linear-in-`w` time interpolation
makes `dw/dT` piecewise constant, so `sigma_loc` *jumps* at every pillar
(bundled surface at `k = ln 0.8`: `0.3054 | 0.2985 | 0.2914` across
`T = 0.5 ± 3e-4`).  Exactly on a pillar the central `DT` stencil returns
the average of the two one-sided values; the golden cases therefore sit
`1e-3` off the pillars (§9) so that a port differentiating one-sidedly
still reproduces them.

`vol(k, t)` is the callable handed to the PDE/MC engines; its `k` argument
is always **forward** log-moneyness `ln(level / F(t))`.

---

## 5. Black-Scholes analytics

`bs_price(S,K,r,q,sigma,T,call)` with edge cases: `T == 0` → intrinsic
`max(phi(S-K),0)`; `sigma == 0` → `exp(-rT) max(phi(F-K),0)`;
`K == 0` → call `S exp(-qT)`, put `0`.  Otherwise standard formula on
`d1 = (ln(F/K) + sigma^2 T/2)/(sigma sqrt(T))`, `d2 = d1 - sigma sqrt(T)`.

**Implied vol inversion (used by the round-trip demo/tests): bisection.**
Bracket `[1e-9, 5.0]`, exactly 100 halvings, return the midpoint.  Before
iterating, reject prices outside
`[bs_price(sigma=0), call ? S e^{-qT} : K e^{-rT}]` with slack
`1e-12 * max(1, S)`.  Bisection is mandated (not Newton) so every port gets
the same ~1e-10-accurate result with no vega/convergence corner cases.
Ports that expose the bracket/iteration count as parameters (Python, C++)
must validate `0 < lo < hi` finite and `iterations >= 1`; non-default
values leave the golden contract.

---

## 6. PDE pricer (Crank-Nicolson + Rannacher, log-spot)

PDE in `x = ln S`, marched in remaining time `tau = T - t` from the payoff:

```
dV/dtau = a d2V/dx2 + mu dV/dx - r V,   a = sigma^2/2,  mu = r - q - sigma^2/2
```

### 6.1 Thomas tridiagonal solver (native — no library banded solver)

Inputs: sub-diagonal `a[0..n-2]` (`a[i]` multiplies `x[i]` in row `i+1`),
diagonal `b[0..n-1]`, super-diagonal `c[0..n-2]` (`c[i]` multiplies `x[i+1]`
in row `i`), rhs `d`.  Forward sweep `cp[0]=c[0]/b[0]`, `dp[0]=d[0]/b[0]`,
then `piv = b[i] - a[i-1] cp[i-1]`, `cp[i]=c[i]/piv`,
`dp[i]=(d[i]-a[i-1] dp[i-1])/piv`; back-substitute
`x[n-1]=dp[n-1]`, `x[i]=dp[i]-cp[i] x[i+1]`.  Reject `|pivot| < 1e-300`,
shape mismatches, non-finite input.  SciPy/Eigen/LAPACK banded solvers are
allowed **only** as an independent check inside tests.

### 6.2 Spatial grid (exact rule)

Inputs `num_space = M` (even, ≥ 4; default **200**), `nsd` (default
**6.0**), reference vol `sigma_ref` (flat vol, or for local vol: caller
supplies it, defaulting to `vol(0, T)` — the ATM local vol at expiry).

```
W = |ln(K/S0)| + nsd * sigma_ref * sqrt(T) + |r - q| * T
h = 2 W / M
x_i = ln(S0) + (i - M/2) h ,   i = 0 … M
```

`ln(S0)` is exactly node `M/2`; the final price is read at that node —
**no interpolation**.

### 6.3 Time stepping (Rannacher schedule — exact)

`num_time = N` (≥ 1; default **200**), `dt = T / N`.  The step sequence is:

1. two **backward-Euler** sub-steps (`theta = 1`) of size `dt/2`
   (these replace the first dt-interval — the Rannacher start), then
2. `N - 1` **Crank-Nicolson** steps (`theta = 1/2`) of size `dt`.

*Why:* CN is A-stable but not L-stable — its amplification factor → −1 for
stiff modes, so the payoff kink excites persistent gamma oscillations
("dt too big vs dx" makes them visible even though the scheme never blows
up).  The two implicit half-steps damp exactly those modes; second-order
accuracy is preserved (verified: observed convergence order ≈ 2 on
grid-halving, tested to lie in `[1.5, 2.5]`).

For the sub-step from `tau` to `tau + dts` the volatility is frozen at the
**midpoint calendar time** `t_mid = T - tau - dts/2` and evaluated at each
interior node as `sigma_i = vol(x_i - ln F(t_mid), t_mid)`; flat vol uses
`sigma_i = sigma` everywhere.

### 6.4 Discrete step (theta-scheme, exact linear system)

For interior nodes `i = 1 … M-1` with `alpha_i = a_i/h^2`,
`beta_i = mu_i/(2h)`, **central** differencing wherever the mesh Peclet
condition `|mu_i| h <= 2 a_i` holds:

```
lower_i = alpha_i - beta_i,  upper_i = alpha_i + beta_i,  center_i = -2 alpha_i - r
```

and first-order **upwind** at every node where it fails (`|mu_i| h > 2 a_i`
— a node floored at 1% by the Dupire clamp with a few percent of carry, or
a very low flat vol with strong carry):

```
lower_i = alpha_i + max(-mu_i, 0)/h,  upper_i = alpha_i + max(mu_i, 0)/h,
center_i = -2 alpha_i - |mu_i|/h - r
```

Both branches have row sum `-r`; the upwind branch keeps both
off-diagonals `>= 0`, so the theta-scheme matrix stays an M-matrix
(monotone, diagonally dominant for the Thomas kernel) at the cost of
`O(|mu| h / 2)` numerical diffusion at those nodes.  The switch is
node-by-node and exact-arithmetic identical to central differencing when
the condition holds, so flat-vol results on the default grids are
unaffected.  Then:

```
LHS:  -theta dts lower_i V^{new}_{i-1}
      + (1 - theta dts center_i) V^{new}_i
      - theta dts upper_i V^{new}_{i+1}
RHS:  V_i + (1-theta) dts (lower_i V_{i-1} + center_i V_i + upper_i V_{i+1})
      [ + theta dts lower_1 V0_new   added to the first row
        + theta dts upper_{M-1} VM_new added to the last row ]
```

Solve with Thomas; then set the boundary nodes to their Dirichlet values.

### 6.5 Boundary conditions (discounted asymptotics, at the new `tau`)

* European call: `V(x_0) = 0`,
  `V(x_M) = max(S_M e^{-q tau} - K e^{-r tau}, 0)` with `S_M = e^{x_M}`.
* European put: `V(x_M) = 0`,
  `V(x_0) = max(K e^{-r tau} - S_0^{node} e^{-q tau}, 0)`.
* American put: `V(x_0) = max(K - e^{x_0}, 0)` (immediate exercise),
  `V(x_M) = 0`.

### 6.6 Analytic short-circuits (before any grid is built)

`T == 0` → intrinsic.  Flat `sigma == 0` → European: `bs_price(sigma=0)`;
American put: `max over t in [0,T] of e^{-rt} max(K - S0 e^{(r-q)t}, 0)`
(reference implementation maximises over 2001 equidistant `t`).  `K == 0`
→ call `S0 e^{-qT}`, put `0` (American put requires `K > 0`).

### 6.7 American put — PSOR (documented scheme choice)

Chosen over penalty/operator-splitting: PSOR solves the discrete linear
complementarity problem to an explicit tolerance with no penalty parameter,
and warm starts keep it cheap.  Each theta-scheme system of §6.4 (same
matrix, same rhs, boundaries from §6.5) is solved by projected SOR with
obstacle `psi_i = max(K - e^{x_i}, 0)`:

* start from `x = max(V_interior, psi_interior)` (warm start, feasible);
* sweep `i` ascending: Gauss-Seidel value
  `gs = (rhs_i - sub_i x_{i-1} - sup_i x_{i+1}) / diag_i`, update
  `x_i = max(psi_i, x_i + omega (gs - x_i))`;
* stop when the sup-norm update `< tol`.

Defaults (also the golden settings): `omega = 1.5`, `tol = 1e-8`,
`max_iter = 10000`.  Validation: `omega` in `(0, 2)`, `tol` finite and
`> 0` (`!(tol > 0)` form so NaN is rejected), `max_iter >= 1`.  Hitting
`max_iter` warns/logs and proceeds.

A user-supplied `vol(k, t)` callable (Python, C++, Java; Rust's `VolInput`
is a closed enum) is checked at every node and time step: a non-finite or
negative sigma is rejected with the idiomatic error, a scalar result is
broadcast to the node vector (Python).

---

## 7. Monte Carlo (log-Euler, antithetic)

Uniform grid `t_n = n dt`, `dt = T/N_steps`; `X` starts at `ln S0`:

```
sigma_n = vol(X_n - ln F(t_n), t_n)         (start-of-step lookup)
X_{n+1} = X_n + (r - q - sigma_n^2/2) dt + sigma_n sqrt(dt) Z_n
```

**Antithetic** (default on; requires even `n_paths >= 4`): `n_paths/2` base
paths driven by `+Z`, mirrored paths reuse the same `Z` with `-Z` (each
mirrored path recomputes its own `sigma` from its own state).  Estimator:
discounted payoff `e^{-rT} payoff(X_T)`; average the two members of each
pair first, then `price = mean(pair_means)`,
`stderr = sample_std(pair_means, ddof=1) / sqrt(n_pairs)`.  Without
antithetic, the same over the raw paths, requiring `n_paths >= 2`.  The
minimum path counts exist because the `ddof = 1` standard error of a
single sample (one pair mean) is `0/0`; smaller counts are rejected, named
by reason.  `n_steps >= 1`.  A user-supplied `sigma(k, t)` callable is
checked on every lookup (non-finite or negative → idiomatic error;
`sigma == 0` is allowed and gives a deterministic step and a zero bridge
crossing probability).

**RNG:** each language uses its own standard-normal generator with a fixed
seed of its choice — Python `numpy.random.default_rng(seed)` (PCG64,
ziggurat), C++ `std::mt19937_64` + `std::normal_distribution<double>`
(the normal transform is implementation-defined: libstdc++, libc++ and
MSVC give different streams), Rust `rand::rngs::StdRng` (ChaCha12 in
`rand` 0.8; documented as not stable across `rand` releases) +
`rand_distr::StandardNormal`, Java `L64X128MixRandom` (LXM family, JDK 17+)
with its `nextGaussian`.  Runs are therefore **deterministic for a fixed
toolchain**, not across toolchains or languages, and MC golden comparisons
are *statistical*: tolerances in `golden.json` were sized at 4 standard
errors of the stated `(n_paths, n_steps)` configuration plus, for the
barrier, a small residual-bias allowance.  `n_paths = 20000` for all golden
MC cases.

**Up-and-out call** (requires `S0 < B`, else price = stderr = 0): barrier
`b = ln B`; each path carries weight `w`, initially 1.

* Discrete monitoring: after each step, `X_{n+1} >= b` → `w = 0`.
* Brownian-bridge correction (default on, used by the golden case): in
  addition, for surviving steps with `X_n, X_{n+1} < b`:
  `w *= 1 - exp(-2 (b - X_n)(b - X_{n+1}) / (sigma_n^2 dt))` — the exact
  bridge crossing probability, removing the O(sqrt(dt)) discrete-monitoring
  bias with no extra random numbers.  The bridged price is ≤ the discrete
  one; both are ≤ the vanilla.

Payoff: `w * max(S_T - K, 0)`, discounted; estimator as above.

---

## 8. Round-trip consistency (demo requirement)

Load `data/implied_surface.csv`, build Dupire local vol, price European
calls by PDE (200×200, `sigma_ref = surface ATM vol at that expiry`) on a
strike/expiry grid `T in {0.5, 1, 1.5, 2} × K in {80, 90, 95, 100, 105,
110, 120}` with `Market(100, 0.02, 0.01)`, invert with §5's bisection, and
report the max absolute implied-vol error in the **interior** `|k| <= 0.30`
(wings excluded).  Target: `< 30 bp`; every port achieves **2.73 bp**
(worst cell `K = 80, T = 2`) with the clamp counters at zero over the whole
sweep — the test suites assert `< 5 bp` on the same grid.  (Before the
wing clamp of §4 the figure was 16.4 bp, ~85% of it caused by 500% caps at
the last quoted strike.)  The demo also cross-checks PDE vs MC within 3 SE
and prints the Dupire clamp counters, reset once before the sweep.

---

## 9. Golden cases (15) — dispatch, settings, tolerances

Engines: **DUP** = Dupire vol at `(k, T)`; **PDE-E** = European PDE §6;
**PDE-A** = American put PSOR §6.7; **MC** = §7 (fixed per-language seed);
statistical cases (†) compare against the *reference value* with a
4-SE-sized absolute tolerance.  `call: 1/0` selects call/put.
Bundled cases use `data/implied_surface.csv`; `flat_dupire_*` use
`data/flat_surface.csv`; the bundled market is `S0=100, r=0, q=0` so the
case names' strikes map to `k = ln(K/100)` (the `inputs` carry the exact
`k`).  All comparisons: `|got - expect| <= tol`.

| name | engine | key inputs | expect | tol |
|---|---|---|---|---|
| `flat_dupire_atm_T1` | DUP(flat csv) | k=0, T=1 | local_vol 0.2 | 1e-6 |
| `flat_dupire_wing_k03_T05` | DUP(flat csv) | k=0.3, T=0.5 | local_vol 0.2 | 1e-6 |
| `pde_flat_call_equity` | PDE-E | S=100,K=100,r=.05,q=.02,σ=.2,T=1, 200×200 | 9.2270055082 | 9.227e-3 |
| `pde_flat_put_fx_gk` | PDE-E | S=1.10,K=1.05,rd=.03,rf=.01,σ=.10,T=.5, 200×200 | 0.0090516712 | 9.05e-6 |
| `pde_flat_call_negrate` | PDE-E | S=100,K=110,r=−.01,q=0,σ=.25,T=2, 200×200 | 9.5537562254 | 9.554e-3 |
| `dupire_bundled_k80_T0501` | DUP | k=ln0.8, T=0.501 | 0.2913439835 | 1e-4 |
| `dupire_bundled_k95_T1001` | DUP | k=ln0.95, T=1.001 | 0.2091962672 | 1e-4 |
| `dupire_bundled_k100_T1001` | DUP | k=0, T=1.001 | 0.1944178773 | 1e-4 |
| `dupire_bundled_k110_T15` | DUP | k=ln1.1, T=1.5 | 0.1753267481 | 1e-4 |
| `dupire_bundled_k120_T075` | DUP | k=ln1.2, T=0.75 | 0.1622172926 | 1e-4 |
| `pde_localvol_bundled_k100_T1` | PDE-E + DUP | S=100,K=100,r=q=0,T=1, 200×200, `sigma_ref` from inputs | 7.9615846335 | 7.96e-3 |
| `american_put_flat_crr5000` | PDE-A | S=100,K=100,r=.05,q=0,σ=.2,T=1, 200×200; reference = CRR binomial 5000 steps (carries ~±5e-4 of binomial oscillation) | 6.0902194081 | 1e-2 |
| `mc_flat_call_vs_bs` † | MC | S=100,K=105,r=.03,q=.01,σ=.2,T=1, 20000×100 | BS 6.6380611953 | 0.285281 |
| `mc_localvol_bundled_k100_T1` † | MC + DUP | S=100,K=100,r=q=0,T=1, 20000×100 | PDE 7.9615846335 | 0.233524 |
| `barrier_upout_flat_bb` † | MC barrier, bridge on | S=100,K=100,B=130,r=.02,q=0,σ=.2,T=1, 20000×200 | analytic 3.1288387727 | 0.16144 |

The three `dupire_bundled_*` cases at `T = 0.501` and `T = 1.001` sit
`1e-3` beyond a pillar on purpose (§4, time discontinuity); the `inputs`
carry the exact `T`.

Every port's test suite must load the JSON (tiny hand-rolled reader is fine
for this flat schema), dispatch on the name prefixes
(`flat_dupire_`, `pde_flat_`, `dupire_bundled_`, `pde_localvol_`,
`american_put_`, `mc_flat_`, `mc_localvol_`, `barrier_upout_`) and assert
all 15 cases.

## 10. Fixed numeric constants (recap)

| Constant | Value | Where |
|---|---|---|
| Dupire `DK` / `DT` | `1e-3` / `1e-4` | §4 |
| Dupire wing clamp | query into `[k_0 + DK, k_{m-1} - DK]` when box `> 2 DK` | §4 |
| Dupire `T = 0` | BBF limit `s / (1 - k s'/s)` | §4 |
| Dupire clamp band | `[0.01, 5.0]` | §4 |
| `w` division guard | `1e-12` | §4 |
| Calendar tolerance | `1e-12` | §3.3 |
| Spline-overshoot scan | 15 probes per node interval, count `w <= 0` | §3.4 |
| Mesh Peclet switch | upwind where `|mu_i| h > 2 a_i` | §6.4 |
| MC minimum paths | `>= 4` even (antithetic), `>= 2` (plain) | §7 |
| PDE defaults | `M = 200`, `N = 200`, `nsd = 6.0` | §6 |
| Rannacher | first dt as 2 × BE(dt/2), then CN | §6.3 |
| PSOR | `omega = 1.5`, `tol = 1e-8`, `max_iter = 10000` | §6.7 |
| Implied-vol bisection | bracket `[1e-9, 5.0]`, 100 iterations | §5 |
| Thomas pivot guard | `1e-300` | §6.1 |
| MC defaults | antithetic on, start-of-step vol lookup | §7 |
