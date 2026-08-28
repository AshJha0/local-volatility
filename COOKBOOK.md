# localvol Cookbook

Task-oriented recipes with snippets in all four languages. Conventions:

* **Python** snippets are runnable as-is from the `python/` directory with
  `PYTHONPATH=src python3 <file>` (data lives at `../data`).
* **C++** snippets assume `#include <localvol/localvol.hpp>` and
  `using namespace localvol;` inside a `main` linked against the library.
* **Rust** snippets run inside a `fn main() -> Result<(), localvol::LocalVolError>`
  (every fallible API returns `Result`; `?` propagates).
* **Java** snippets live in a class importing `com.quant.localvol.*` and may
  throw `Exception` from `main`.

All engines share the coordinate contract: `k` is **forward** log-moneyness
`ln(K / F(T))` and surfaces store total variance `w = iv^2 * T`.

---

## 1. How do I load an implied surface from CSV and query it?

**Python**

```python
from localvol import ImpliedVolSurface

srf = ImpliedVolSurface.from_csv("../data/implied_surface.csv")
print(srf.expiries, srf.k_nodes.size)      # pillars: [0.25 0.5 1. 2.], 15 strikes
print(srf.implied_vol(0.0, 1.0))           # ATM vol at T=1 -> 0.20
print(srf.total_variance(-0.1, 0.75))      # w interpolated in (k, T)
print(srf.implied_vol(0.9, 1.0))           # k beyond the wing -> flat extrapolation
```

**C++**

```cpp
auto srf = ImpliedVolSurface::from_csv("../data/implied_surface.csv");
std::cout << srf.implied_vol(0.0, 1.0) << "\n";      // 0.20
std::cout << srf.total_variance(-0.1, 0.75) << "\n"; // interpolated w(k, T)
```

**Rust**

```rust
let srf = localvol::ImpliedVolSurface::from_csv("../data/implied_surface.csv")?;
println!("{}", srf.implied_vol(0.0, 1.0)?);      // 0.20
println!("{}", srf.total_variance(-0.1, 0.75)?); // interpolated w(k, T)
```

**Java**

```java
ImpliedVolSurface srf = ImpliedVolSurface.fromCsv("../data/implied_surface.csv");
System.out.println(srf.impliedVol(0.0, 1.0));      // 0.20
System.out.println(srf.totalVariance(-0.1, 0.75)); // interpolated w(k, T)
```

Queries below the first pillar use flat forward variance
(`w * T / T1`); beyond the last pillar, linear total variance with the slope
floored at 0; `k` outside the node range is clamped (flat wings).

---

## 2. How do I build Dupire local vol and watch the clamp counters?

**Python**

```python
from localvol import DupireLocalVol, ImpliedVolSurface

srf = ImpliedVolSurface.from_csv("../data/implied_surface.csv")
lv = DupireLocalVol(srf)
print(lv.vol(0.0, 1.0))          # 0.19441818... (golden dupire_bundled_k100_T1)
print(lv.vol(-0.8, 0.5))         # beyond the quoted wing: flat-clamped, ~0.2638
print(lv.floor_count, lv.cap_count)   # counters accumulate across ALL queries
lv.reset_counters()              # counters are cumulative and resettable
print(lv.violation_report)
```

**C++**

```cpp
DupireLocalVol lv(srf);
std::cout << lv.vol(0.0, 1.0) << "\n";              // 0.1944...
std::cout << lv.floor_count() << " " << lv.cap_count() << "\n";
lv.reset_counters();
```

**Rust**

```rust
let mut lv = localvol::DupireLocalVol::new(&srf);
println!("{}", lv.vol(0.0, 1.0)?);                  // 0.1944...
println!("{} {}", lv.floor_count(), lv.cap_count());
lv.reset_counters();
```

**Java**

```java
DupireLocalVol lv = new DupireLocalVol(srf);
System.out.println(lv.vol(0.0, 1.0));               // 0.1944...
System.out.println(lv.floorCount() + " " + lv.capCount());
lv.resetCounters();
```

Clamps to `[1%, 500%]` are *reporting*, never errors: floors mean no forward
variance (calendar issue), caps mean a non-positive denominator (butterfly
issue, typically extrapolation-wing artifacts). Interior clamps on quoted
strikes are a red flag; far-wing clamps during PDE sweeps are expected.

---

## 3. How do I price a European option with flat vol via PDE and check it against Black-Scholes?

**Python**

```python
from localvol import Market, bs_price, price_european_pde

mkt = Market(spot=100.0, rate=0.05, dividend=0.02)
pde = price_european_pde(mkt, 100.0, 1.0, 0.2, is_call=True,
                         num_space=200, num_time=200)
bs = bs_price(100.0, 100.0, 0.05, 0.02, 0.2, 1.0, True)
print(pde, bs, abs(pde - bs) / bs)   # 9.2270... , rel err < 1e-3
```

**C++**

```cpp
Market mkt{100.0, 0.05, 0.02};
double pde = price_european_pde(mkt, 100.0, 1.0, 0.2, /*is_call=*/true,
                                /*num_space=*/200, /*num_time=*/200);
double bs = bs_price(100.0, 100.0, 0.05, 0.02, 0.2, 1.0, true);
```

**Rust**

```rust
let mkt = localvol::Market::new(100.0, 0.05, 0.02)?;
let pde = localvol::price_european_pde(&mkt, 100.0, 1.0,
              &localvol::Vol::Flat(0.2), true, 200, 200, 6.0, None)?;
let bs = localvol::bs_price(100.0, 100.0, 0.05, 0.02, 0.2, 1.0, true)?;
```

**Java**

```java
Market mkt = new Market(100.0, 0.05, 0.02);
double pde = Pde.priceEuropean(mkt, 100.0, 1.0, 0.2, true, 200, 200);
double bs = BlackScholes.price(100.0, 100.0, 0.05, 0.02, 0.2, 1.0, true);
```

The 200x200 defaults reproduce the golden value `9.2270055082` to 1e-3
relative. The scheme is Crank-Nicolson with a Rannacher start (two implicit
half-steps), so the convergence order on grid-halving is ~2.

---

## 4. How do I price under the local-vol surface via PDE (and pick `sigma_ref`)?

**Python**

```python
from localvol import DupireLocalVol, ImpliedVolSurface, Market, price_european_pde

srf = ImpliedVolSurface.from_csv("../data/implied_surface.csv")
lv = DupireLocalVol(srf)
mkt = Market(spot=100.0, rate=0.0, dividend=0.0)
price = price_european_pde(mkt, 100.0, 1.0, lv.vol,          # sigma(k, t) callable
                           num_space=200, num_time=200,
                           sigma_ref=float(srf.implied_vol(0.0, 1.0)))
print(price)   # 7.9609... (golden pde_localvol_bundled_k100_T1)
```

**C++**

```cpp
DupireLocalVol lv(srf);
Market mkt{100.0, 0.0, 0.0};
double price = price_european_pde(mkt, 100.0, 1.0, lv, true, 200, 200,
                                  /*nsd=*/6.0, /*sigma_ref=*/srf.implied_vol(0.0, 1.0));
```

**Rust**

```rust
let mut lv = localvol::DupireLocalVol::new(&srf);
let mkt = localvol::Market::new(100.0, 0.0, 0.0)?;
let sigma_ref = srf.implied_vol(0.0, 1.0)?;
let price = localvol::price_european_pde(&mkt, 100.0, 1.0,
                &localvol::Vol::Local(&mut lv), true, 200, 200, 6.0, Some(sigma_ref))?;
```

**Java**

```java
DupireLocalVol lv = new DupireLocalVol(srf);
Market mkt = new Market(100.0, 0.0, 0.0);
double sigmaRef = srf.impliedVol(0.0, 1.0);
double price = Pde.priceEuropean(mkt, 100.0, 1.0, lv, true, 200, 200, 6.0, sigmaRef);
```

`sigma_ref` only sets the **grid width** (`nsd * sigma_ref * sqrt(T)` plus
strike and drift terms); it does not touch the dynamics. Default when
omitted: `vol(0, T)`, the ATM local vol at expiry. Prefer the surface's ATM
*implied* vol when repricing quoted vanillas — that is what the golden case
and the round-trip demo use.

---

## 5. How do I price an FX option (Garman-Kohlhagen)?

**Python**

```python
from localvol import Market, bs_price, price_european_pde

mkt = Market.fx(spot=1.10, rd=0.03, rf=0.01)   # EURUSD-style: domestic 3%, foreign 1%
pde = price_european_pde(mkt, 1.05, 0.5, 0.10, is_call=False)
bs = bs_price(1.10, 1.05, 0.03, 0.01, 0.10, 0.5, False)
print(pde, bs)   # 0.0090516712 to 1e-3 rel (golden pde_flat_put_fx_gk)
```

**C++**

```cpp
Market mkt = Market::fx(1.10, 0.03, 0.01);
double pde = price_european_pde(mkt, 1.05, 0.5, 0.10, /*is_call=*/false, 200, 200);
```

**Rust**

```rust
let mkt = localvol::Market::fx(1.10, 0.03, 0.01)?;
let pde = localvol::price_european_pde(&mkt, 1.05, 0.5,
              &localvol::Vol::Flat(0.10), false, 200, 200, 6.0, None)?;
```

**Java**

```java
Market mkt = Market.fx(1.10, 0.03, 0.01);
double pde = Pde.priceEuropean(mkt, 1.05, 0.5, 0.10, false, 200, 200);
```

Garman-Kohlhagen is Black-Scholes with `r = rd`, `q = rf`; every engine
(PDE, MC, Dupire) accepts the same `Market`, and the surface itself is
carry-free because it is stored in forward log-moneyness. Negative rates are
allowed on both legs.

---

## 6. How do I price an American put and extract the early-exercise premium?

**Python**

```python
from localvol import Market, price_american_put_pde, price_european_pde

mkt = Market(spot=100.0, rate=0.05, dividend=0.0)
amer = price_american_put_pde(mkt, 100.0, 1.0, 0.2, num_space=200, num_time=200)
euro = price_european_pde(mkt, 100.0, 1.0, 0.2, is_call=False,
                          num_space=200, num_time=200)
print(amer, euro, amer - euro)   # 6.09 vs 5.57, premium ~ +0.517, always >= 0
```

**C++**

```cpp
Market mkt{100.0, 0.05, 0.0};
double amer = price_american_put_pde(mkt, 100.0, 1.0, 0.2, 200, 200);
double euro = price_european_pde(mkt, 100.0, 1.0, 0.2, /*is_call=*/false, 200, 200);
```

**Rust**

```rust
let mkt = localvol::Market::new(100.0, 0.05, 0.0)?;
let amer = localvol::price_american_put_pde(&mkt, 100.0, 1.0,
               &localvol::Vol::Flat(0.2), 200, 200, 6.0, None)?;
```

**Java**

```java
Market mkt = new Market(100.0, 0.05, 0.0);
double amer = Pde.priceAmericanPut(mkt, 100.0, 1.0, 0.2, 200, 200);
```

The engine is PSOR (`omega = 1.5`, `tol = 1e-8`, `max_iter = 10000`) on the
same Rannacher/CN systems; non-convergence warns and proceeds rather than
throwing. Golden reference: CRR binomial 5000 steps = `6.0902194081`
(tol 1e-2). A local-vol American works the same way — pass `lv.vol`
(Python) / the local-vol object (natives) instead of the flat sigma.

---

## 7. How do I run local-vol Monte Carlo and cross-check the PDE?

**Python**

```python
from localvol import (DupireLocalVol, ImpliedVolSurface, Market,
                      price_european_mc, price_european_pde)

srf = ImpliedVolSurface.from_csv("../data/implied_surface.csv")
lv = DupireLocalVol(srf)
mkt = Market(spot=100.0, rate=0.0, dividend=0.0)
pde = price_european_pde(mkt, 100.0, 1.0, lv.vol, sigma_ref=0.2)
mc = price_european_mc(mkt, 100.0, 1.0, lv.vol,
                       n_paths=20000, n_steps=100, seed=42, antithetic=True)
print(f"PDE={pde:.4f} MC={mc.price:.4f} +/- {mc.stderr:.4f}")
assert mc.within(pde, 3.0)       # agreement within 3 standard errors
```

**C++**

```cpp
McResult mc = price_european_mc(mkt, 100.0, 1.0, lv, /*is_call=*/true,
                                20000, 100, /*seed=*/42, /*antithetic=*/true);
bool ok = std::abs(mc.price - pde) <= 3.0 * mc.stderr;
```

**Rust**

```rust
let mc = localvol::price_european_mc(&mkt, 100.0, 1.0,
             &localvol::Vol::Local(&mut lv), true, 20000, 100, 42, true)?;
assert!((mc.price - pde).abs() <= 3.0 * mc.stderr);
```

**Java**

```java
McResult mc = MonteCarlo.priceEuropean(mkt, 100.0, 1.0, lv, true,
                                       20000, 100, 42L, true);
boolean ok = Math.abs(mc.price() - pde) <= 3.0 * mc.stderr();
```

Antithetic sampling requires an even `n_paths`; the standard error is
computed over **pair means**, so `mc.stderr` is directly usable in an
N-standard-error test. Seeds are per-language; cross-language MC comparisons
must be statistical, never bitwise.

---

## 8. How do I price an up-and-out barrier, and what does the Brownian bridge change?

**Python**

```python
from localvol import Market, price_up_out_call_mc

mkt = Market(spot=100.0, rate=0.02, dividend=0.0)
discrete = price_up_out_call_mc(mkt, 100.0, 130.0, 1.0, 0.2,
                                n_paths=20000, n_steps=200, seed=42,
                                brownian_bridge=False)
bridged = price_up_out_call_mc(mkt, 100.0, 130.0, 1.0, 0.2,
                               n_paths=20000, n_steps=200, seed=42,
                               brownian_bridge=True)
print(discrete.price, bridged.price)   # bridged <= discrete; analytic 3.1288...
```

**C++**

```cpp
McResult uo = price_up_out_call_mc(mkt, 100.0, 130.0, 1.0, 0.2,
                                   20000, 200, 42, /*antithetic=*/true,
                                   /*brownian_bridge=*/true);
```

**Rust**

```rust
let uo = localvol::price_up_out_call_mc(&mkt, 100.0, 130.0, 1.0,
             &localvol::Vol::Flat(0.2), 20000, 200, 42, true, true)?;
```

**Java**

```java
McResult uo = MonteCarlo.priceUpOutCall(mkt, 100.0, 130.0, 1.0, 0.2,
                                        20000, 200, 42L, true, true);
```

Discrete monitoring misses crossings between steps and *overprices* the
knock-out by `O(sqrt(dt))`. The bridge multiplies each surviving step's path
weight by `1 - exp(-2 (b - x0)(b - x1) / (sigma^2 dt))`, removing that bias
with no extra randomness. Invariants worth asserting: bridged <= discrete
<= vanilla, and `S0 >= B` returns price 0 with zero standard error. Local
vol: pass the local-vol function; the bridge uses the frozen per-step vol.

---

## 9. How do I invert a price back to implied vol (and round-trip a point)?

**Python**

```python
from localvol import Market, implied_vol, price_european_pde, ImpliedVolSurface, DupireLocalVol
import math

srf = ImpliedVolSurface.from_csv("../data/implied_surface.csv")
lv = DupireLocalVol(srf)
mkt = Market(spot=100.0, rate=0.02, dividend=0.01)
strike, T = 95.0, 1.0
k = math.log(strike / mkt.forward(T))
price = price_european_pde(mkt, strike, T, lv.vol,
                           sigma_ref=float(srf.implied_vol(0.0, T)))
iv_out = implied_vol(price, mkt.spot, strike, mkt.rate, mkt.dividend, T, True)
iv_in = float(srf.implied_vol(k, T))
print(f"in={iv_in:.4%} out={iv_out:.4%} err={(iv_out - iv_in) * 1e4:.1f}bp")
```

**C++**

```cpp
double iv_out = implied_vol(price, mkt.spot, strike, mkt.rate, mkt.dividend,
                            T, /*is_call=*/true);
```

**Rust**

```rust
let iv_out = localvol::implied_vol(price, mkt.spot(), strike,
                                   mkt.rate(), mkt.dividend(), t, true)?;
```

**Java**

```java
double ivOut = BlackScholes.impliedVol(price, mkt.spot(), strike,
                                       mkt.rate(), mkt.dividend(), t, true);
```

The inverter is deliberately bisection — 100 halvings of `[1e-9, 5.0]` —
identical and deterministic in every language. Prices outside the static
no-arbitrage bounds are rejected eagerly (Python `ValueError`, C++
`std::invalid_argument`, Rust `Err(LocalVolError::...)`, Java
`IllegalArgumentException`).

---

## 10. How do I detect calendar arbitrage or handle a single-expiry surface?

**Python**

```python
import warnings
import numpy as np
from localvol import ImpliedVolSurface

# A surface where total variance DECREASES from T=1 to T=2 at every node:
k = np.array([-0.2, 0.0, 0.2])
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    bad = ImpliedVolSurface(k, np.array([1.0, 2.0]),
                            np.array([[0.25, 0.24, 0.25],
                                      [0.15, 0.15, 0.15]]))  # w drops in T
print(bad.calendar_violations)          # 3 — reported, not raised

single = ImpliedVolSurface(k, np.array([1.0]),
                           np.array([[0.21, 0.20, 0.21]]))   # warns: single expiry
print(single.total_variance(0.0, 2.0))  # flat-forward: w(k,1) * 2
```

**C++**

```cpp
ImpliedVolSurface bad(k_nodes, expiries, vols);   // logs a warning
if (bad.calendar_violations() > 0) { /* decide: refit, reject, or proceed */ }
```

**Rust**

```rust
let bad = localvol::ImpliedVolSurface::new(&k_nodes, &expiries, &vols)?;
if bad.calendar_violations() > 0 { /* report upstream */ }
```

**Java**

```java
ImpliedVolSurface bad = new ImpliedVolSurface(kNodes, expiries, vols);
if (bad.calendarViolations() > 0) { /* report upstream */ }
```

Both conditions follow the project's reporting policy: construction
succeeds, a warning is emitted, and a counter/flag
(`calendar_violations`, `single_expiry`) lets callers decide. Genuinely
invalid input — non-rectangular CSV grids, non-positive vols, non-increasing
node arrays — is rejected eagerly with the language's idiomatic error.

---

## 11. How do I solve a tridiagonal system with the Thomas kernel?

**Python**

```python
import numpy as np
from localvol import thomas_solve

# Solve A x = d for A = tri(sub, diag, sup), here a 4x4 system:
sub = np.array([1.0, 1.0, 1.0])        # sub[i] multiplies x[i] in row i+1
diag = np.array([4.0, 4.0, 4.0, 4.0])
sup = np.array([1.0, 1.0, 1.0])        # sup[i] multiplies x[i+1] in row i
d = np.array([5.0, 6.0, 6.0, 5.0])
x = thomas_solve(sub, diag, sup, d)
print(x)                                # [1. 1. 1. 1.]
```

**C++**

```cpp
std::vector<double> x = thomas_solve(sub, diag, sup, rhs);
```

**Rust**

```rust
let x = localvol::thomas_solve(&sub, &diag, &sup, &rhs)?;
```

**Java**

```java
double[] x = Tridiag.thomasSolve(sub, diag, sup, rhs);
```

This is the single linear-algebra kernel of the whole project — the spline
moment systems and every PDE time step go through it, which is what makes
the four ports numerically comparable. It rejects near-zero pivots
(`< 1e-300`), shape mismatches and non-finite input; it is not pivoted, so
use it only on diagonally dominant systems (the CN grids guarantee this via
the mesh Péclet condition).

---

## 12. How do I regenerate the data and run the golden suite everywhere?

```bash
# Regenerate CSVs + golden.json (deterministic; validates reference first)
python3 data/generate_data.py

# Python
cd python && PYTHONPATH=src pytest -q && cd ..

# C++
cd cpp && ./build.sh && ctest --test-dir build --output-on-failure && cd ..

# Rust
cd rust && cargo test && cd ..

# Java
cd java && ./build.sh && ./test.sh && cd ..
```

The generator refuses to write `golden.json` unless the Python reference
passes its own validation (flat-vol PDE vs Black-Scholes to 1e-3 relative;
flat-surface Dupire exactly 0.20 to 1e-6 across a sweep). Each language's
suite then loads the JSON, dispatches on the case-name prefix
(`flat_dupire_`, `pde_flat_`, `dupire_bundled_`, `pde_localvol_`,
`american_put_`, `mc_flat_`, `mc_localvol_`, `barrier_upout_`) and asserts
`|got - expect| <= tol` for all 15 cases. If you change any numerical rule
(interpolation, steps, schedule), regenerate the goldens *and* re-run all
four suites — the point of the file is to catch exactly such drift.
