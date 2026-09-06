# localvol Cookbook

Task-oriented recipes with snippets in all four languages. Every native
snippet below is extracted verbatim and compiled/run against the real
library as part of the documentation check (see §13), so the calls match
the actual APIs. Conventions:

* **Python** snippets are runnable as-is from the `python/` directory with
  `PYTHONPATH=src python3 <file>` (data lives at `../data`).
* **C++** snippets assume the headers
  `localvol/{market,tridiag,black_scholes,surface,dupire,pde,mc}.hpp`
  (there is no umbrella header), `using namespace localvol;`, and a `main`
  linked against `liblocalvol` built by `bash cpp/build.sh`. Grid, PSOR and
  MC parameters travel in small structs (`PdeSettings`, `PsorSettings`,
  `McSettings`); a local-vol object is passed as a `VolFn`
  (`std::function<double(double, double)>`) lambda.
* **Rust** snippets run inside `fn main() -> localvol::Result<()>` (every
  fallible API returns `Result`; `?` propagates). Volatility is the closed
  enum `VolInput::{Flat(f64), Local(&DupireLocalVol)}`; settings are
  `PdeSettings` / `PsorSettings` / `McSettings` structs with `Default`.
* **Java** snippets live in a class in package `com.quant.localvol` (or
  importing it) and may throw `Exception` from `main`. Static entry points:
  `Pde.priceEuropean` / `Pde.priceAmericanPut` / `Pde.europeanGrid`,
  `MonteCarlo.priceEuropean` / `MonteCarlo.priceUpOutCall`,
  `BlackScholes.bsPrice` / `BlackScholes.impliedVol`, `Tridiag.thomasSolve`.
  A `DupireLocalVol` implements `LocalVolFn` and is passed directly; the
  reference vol argument `sigmaRef` may be `Double.NaN` to select
  `vol(0, T)`.

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
print(srf.calendar_violations, srf.negative_w_count)   # data-quality counters: 0, 0
```

**C++**

```cpp
#include <iostream>
#include "localvol/surface.hpp"
using namespace localvol;

int main() {
    const ImpliedVolSurface srf = ImpliedVolSurface::from_csv("../data/implied_surface.csv");
    std::cout << srf.implied_vol(0.0, 1.0) << "\n";      // 0.20
    std::cout << srf.total_variance(-0.1, 0.75) << "\n"; // interpolated w(k, T)
    std::cout << srf.calendar_violations() << " " << srf.negative_w_count() << "\n";  // 0 0
}
```

**Rust**

```rust
use localvol::ImpliedVolSurface;

fn main() -> localvol::Result<()> {
    let srf = ImpliedVolSurface::from_csv("../data/implied_surface.csv")?;
    println!("{}", srf.implied_vol(0.0, 1.0)?);      // 0.20
    println!("{}", srf.total_variance(-0.1, 0.75)?); // interpolated w(k, T)
    println!("{} {}", srf.calendar_violations(), srf.negative_w_count()); // 0 0
    Ok(())
}
```

**Java**

```java
import java.nio.file.Paths;
import com.quant.localvol.ImpliedVolSurface;

public class Cook1 {
    public static void main(String[] args) {
        ImpliedVolSurface srf = ImpliedVolSurface.fromCsv(Paths.get("../data/implied_surface.csv"));
        System.out.println(srf.impliedVol(0.0, 1.0));      // 0.20
        System.out.println(srf.totalVariance(-0.1, 0.75)); // interpolated w(k, T)
        System.out.println(srf.calendarViolations() + " " + srf.negativeWCount()); // 0 0
    }
}
```

Queries below the first pillar use flat forward variance
(`w * T / T1`); beyond the last pillar, linear total variance with the slope
floored at 0; `k` outside the node range is clamped (flat wings). The loader
requires the exact header `T,k,iv`, three columns per row and every `(T, k)`
pair exactly once — duplicates, gaps and short rows are rejected with the
language's idiomatic error.

---

## 2. How do I build Dupire local vol and watch the clamp counters?

**Python**

```python
from localvol import DupireLocalVol, ImpliedVolSurface

srf = ImpliedVolSurface.from_csv("../data/implied_surface.csv")
lv = DupireLocalVol(srf)
print(lv.vol(0.0, 1.001))        # 0.19441788 (golden dupire_bundled_k100_T1001)
print(lv.vol(0.5, 1.0))          # 0.1970 at the last quoted strike (no cap: clamped stencil)
print(lv.vol(-0.8, 0.5))         # beyond the quoted wing: constant in k, 0.4089
print(lv.vol(-0.3, 0.0))         # T = 0: Berestycki-Busca-Florent limit, 0.4524
print(lv.floor_count, lv.cap_count)   # counters accumulate across ALL queries
lv.reset_counters()              # counters are cumulative and resettable
print(lv.violation_report)
```

**C++**

```cpp
#include <iostream>
#include "localvol/dupire.hpp"
#include "localvol/surface.hpp"
using namespace localvol;

int main() {
    // Keep the surface in a named variable: DupireLocalVol stores a reference
    // and binding a temporary is a compile error by design.
    const ImpliedVolSurface srf = ImpliedVolSurface::from_csv("../data/implied_surface.csv");
    const DupireLocalVol lv(srf);              // vol() is const and thread-safe
    std::cout << lv.vol(0.0, 1.001) << "\n";   // 0.1944...
    std::cout << lv.vol(-0.3, 0.0) << "\n";    // T = 0 short-time limit, 0.4524
    std::cout << lv.floor_count() << " " << lv.cap_count() << "\n";
    lv.reset_counters();
    std::cout << lv.violation_report() << "\n";
}
```

**Rust**

```rust
use localvol::{DupireLocalVol, ImpliedVolSurface};

fn main() -> localvol::Result<()> {
    let srf = ImpliedVolSurface::from_csv("../data/implied_surface.csv")?;
    let lv = DupireLocalVol::new(srf);           // takes the surface by value; lv.surface() gets it back
    println!("{}", lv.vol(0.0, 1.001)?);         // 0.1944...
    println!("{}", lv.vol(-0.3, 0.0)?);          // T = 0 short-time limit, 0.4524
    println!("{} {}", lv.floor_count(), lv.cap_count());
    lv.reset_counters();                         // &self: counters are atomic (Send + Sync)
    println!("{}", lv.violation_report());
    Ok(())
}
```

**Java**

```java
import java.nio.file.Paths;
import com.quant.localvol.DupireLocalVol;
import com.quant.localvol.ImpliedVolSurface;

public class Cook2 {
    public static void main(String[] args) {
        ImpliedVolSurface srf = ImpliedVolSurface.fromCsv(Paths.get("../data/implied_surface.csv"));
        DupireLocalVol lv = new DupireLocalVol(srf);
        System.out.println(lv.vol(0.0, 1.001));             // 0.1944...
        System.out.println(lv.vol(-0.3, 0.0));              // T = 0 short-time limit, 0.4524
        System.out.println(lv.floorCount() + " " + lv.capCount());
        lv.resetCounters();
        System.out.println(lv.violationReport());
    }
}
```

Clamps to `[1%, 500%]` are *reporting*, never errors: floors mean no forward
variance (calendar issue), caps mean a non-positive denominator (butterfly
issue). The finite-difference stencil is clamped inside the quoted box
(`[k_min + DK, k_max - DK]`), so the flat wing extrapolation cannot trigger
caps: on the bundled surface neither counter fires anywhere on a 200x200 PDE
grid. Any clamp on a smooth surface therefore means the *input* carries
arbitrage — re-fit it rather than shipping the number. `vol(k, 0)` is the
short-time limit `sigma_imp / (1 - k sigma_imp'/sigma_imp)`, continuous with
`T -> 0+`.

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
#include <cmath>
#include <iostream>
#include "localvol/black_scholes.hpp"
#include "localvol/market.hpp"
#include "localvol/pde.hpp"
using namespace localvol;

int main() {
    const Market mkt(100.0, 0.05, 0.02);
    PdeSettings grid;            // defaults: num_space = 200, num_time = 200, nsd = 6.0
    grid.num_space = 200;
    grid.num_time = 200;
    const double pde = price_european_pde(mkt, 100.0, 1.0, 0.2, /*is_call=*/true, grid);
    const double bs = bs_price(100.0, 100.0, 0.05, 0.02, 0.2, 1.0, true);
    std::cout << pde << " " << bs << " " << std::abs(pde - bs) / bs << "\n";  // rel err < 1e-3
}
```

**Rust**

```rust
use localvol::{bs_price, price_european_pde, Market, PdeSettings, VolInput};

fn main() -> localvol::Result<()> {
    let mkt = Market::new(100.0, 0.05, 0.02)?;
    let grid = PdeSettings { num_space: 200, num_time: 200, ..PdeSettings::default() };
    let pde = price_european_pde(&mkt, 100.0, 1.0, &VolInput::Flat(0.2), true, &grid)?;
    let bs = bs_price(100.0, 100.0, 0.05, 0.02, 0.2, 1.0, true)?;
    println!("{pde} {bs} {}", (pde - bs).abs() / bs); // rel err < 1e-3
    Ok(())
}
```

**Java**

```java
import com.quant.localvol.BlackScholes;
import com.quant.localvol.Market;
import com.quant.localvol.Pde;

public class Cook3 {
    public static void main(String[] args) {
        Market mkt = new Market(100.0, 0.05, 0.02);
        double pde = Pde.priceEuropean(mkt, 100.0, 1.0, 0.2, true, 200, 200, 6.0);
        double bs = BlackScholes.bsPrice(100.0, 100.0, 0.05, 0.02, 0.2, 1.0, true);
        System.out.println(pde + " " + bs + " " + Math.abs(pde - bs) / bs); // rel err < 1e-3
    }
}
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
print(price)   # 7.9616... (golden pde_localvol_bundled_k100_T1)
```

**C++**

```cpp
#include <iostream>
#include "localvol/dupire.hpp"
#include "localvol/market.hpp"
#include "localvol/pde.hpp"
#include "localvol/surface.hpp"
using namespace localvol;

int main() {
    const ImpliedVolSurface srf = ImpliedVolSurface::from_csv("../data/implied_surface.csv");
    const DupireLocalVol lv(srf);
    const VolFn vol = [&lv](double k, double t) { return lv.vol(k, t); };   // adapt to VolFn
    const Market mkt(100.0, 0.0, 0.0);
    const double price = price_european_pde(mkt, 100.0, 1.0, vol, /*is_call=*/true,
                                            PdeSettings{}, /*sigma_ref=*/srf.implied_vol(0.0, 1.0));
    std::cout << price << "\n";   // 7.9616...
}
```

**Rust**

```rust
use localvol::{price_european_pde, DupireLocalVol, ImpliedVolSurface, Market, PdeSettings, VolInput};

fn main() -> localvol::Result<()> {
    let lv = DupireLocalVol::new(ImpliedVolSurface::from_csv("../data/implied_surface.csv")?);
    let mkt = Market::new(100.0, 0.0, 0.0)?;
    let grid = PdeSettings { sigma_ref: Some(lv.surface().implied_vol(0.0, 1.0)?), ..PdeSettings::default() };
    let price = price_european_pde(&mkt, 100.0, 1.0, &VolInput::Local(&lv), true, &grid)?;
    println!("{price}"); // 7.9616...
    Ok(())
}
```

**Java**

```java
import java.nio.file.Paths;
import com.quant.localvol.DupireLocalVol;
import com.quant.localvol.ImpliedVolSurface;
import com.quant.localvol.Market;
import com.quant.localvol.Pde;

public class Cook4 {
    public static void main(String[] args) {
        ImpliedVolSurface srf = ImpliedVolSurface.fromCsv(Paths.get("../data/implied_surface.csv"));
        DupireLocalVol lv = new DupireLocalVol(srf);
        Market mkt = new Market(100.0, 0.0, 0.0);
        double sigmaRef = srf.impliedVol(0.0, 1.0);
        double price = Pde.priceEuropean(mkt, 100.0, 1.0, lv, sigmaRef, true, 200, 200, 6.0);
        System.out.println(price); // 7.9616...
    }
}
```

`sigma_ref` only sets the **grid width** (`nsd * sigma_ref * sqrt(T)` plus
strike and drift terms); it does not touch the dynamics. Default when
omitted (`std::nullopt` / `None` / `Double.NaN`): `vol(0, T)`, the ATM local
vol at expiry. Prefer the surface's ATM *implied* vol when repricing quoted
vanillas — that is what the golden case and the round-trip demo use. A user
callable is checked at every node: NaN, infinite or negative output is an
error, not a silent coefficient.

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
#include <iostream>
#include "localvol/market.hpp"
#include "localvol/pde.hpp"
using namespace localvol;

int main() {
    const Market mkt = Market::fx(1.10, 0.03, 0.01);
    const double pde = price_european_pde(mkt, 1.05, 0.5, 0.10, /*is_call=*/false);  // default 200x200
    std::cout << pde << "\n";   // 0.00905...
}
```

**Rust**

```rust
use localvol::{price_european_pde, Market, PdeSettings, VolInput};

fn main() -> localvol::Result<()> {
    let mkt = Market::fx(1.10, 0.03, 0.01)?;
    let pde = price_european_pde(&mkt, 1.05, 0.5, &VolInput::Flat(0.10), false, &PdeSettings::default())?;
    println!("{pde}"); // 0.00905...
    Ok(())
}
```

**Java**

```java
import com.quant.localvol.Market;
import com.quant.localvol.Pde;

public class Cook5 {
    public static void main(String[] args) {
        Market mkt = Market.fx(1.10, 0.03, 0.01);
        double pde = Pde.priceEuropean(mkt, 1.05, 0.5, 0.10, false);  // default 200x200, nsd 6
        System.out.println(pde); // 0.00905...
    }
}
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
print(amer, euro, amer - euro)   # 6.086 vs 5.570, premium +0.516, always >= 0
```

**C++**

```cpp
#include <iostream>
#include "localvol/market.hpp"
#include "localvol/pde.hpp"
using namespace localvol;

int main() {
    const Market mkt(100.0, 0.05, 0.0);
    PdeSettings grid;                 // 200x200
    PsorSettings psor;                // omega = 1.5, tol = 1e-8, max_iter = 10000
    const double amer = price_american_put_pde(mkt, 100.0, 1.0, 0.2, grid, psor);
    const double euro = price_european_pde(mkt, 100.0, 1.0, 0.2, /*is_call=*/false, grid);
    std::cout << amer << " " << euro << " " << amer - euro << "\n";
}
```

**Rust**

```rust
use localvol::{price_american_put_pde, price_european_pde, Market, PdeSettings, PsorSettings, VolInput};

fn main() -> localvol::Result<()> {
    let mkt = Market::new(100.0, 0.05, 0.0)?;
    let grid = PdeSettings::default();
    let amer = price_american_put_pde(&mkt, 100.0, 1.0, &VolInput::Flat(0.2), &grid, &PsorSettings::default())?;
    let euro = price_european_pde(&mkt, 100.0, 1.0, &VolInput::Flat(0.2), false, &grid)?;
    println!("{amer} {euro} {}", amer - euro);
    Ok(())
}
```

**Java**

```java
import com.quant.localvol.Market;
import com.quant.localvol.Pde;

public class Cook6 {
    public static void main(String[] args) {
        Market mkt = new Market(100.0, 0.05, 0.0);
        double amer = Pde.priceAmericanPut(mkt, 100.0, 1.0, 0.2);   // defaults 200x200, PSOR 1.5/1e-8/10000
        double euro = Pde.priceEuropean(mkt, 100.0, 1.0, 0.2, false, 200, 200, 6.0);
        System.out.println(amer + " " + euro + " " + (amer - euro));
    }
}
```

The engine is PSOR (`omega = 1.5`, `tol = 1e-8`, `max_iter = 10000`) on the
same Rannacher/CN systems; non-convergence warns and proceeds rather than
throwing. Golden reference: CRR binomial 5000 steps = `6.0902194081`
(tol 1e-2). A local-vol American works the same way — pass `lv.vol`
(Python), a `VolFn` lambda (C++), `VolInput::Local(&lv)` (Rust) or the
`DupireLocalVol` itself (Java) instead of the flat sigma.

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
#include <cstdio>
#include "localvol/dupire.hpp"
#include "localvol/market.hpp"
#include "localvol/mc.hpp"
#include "localvol/pde.hpp"
#include "localvol/surface.hpp"
using namespace localvol;

int main() {
    const ImpliedVolSurface srf = ImpliedVolSurface::from_csv("../data/implied_surface.csv");
    const DupireLocalVol lv(srf);
    const VolFn vol = [&lv](double k, double t) { return lv.vol(k, t); };
    const Market mkt(100.0, 0.0, 0.0);
    const double pde = price_european_pde(mkt, 100.0, 1.0, vol, true, PdeSettings{}, 0.2);
    McSettings mcs;                   // n_paths = 20000, n_steps = 100, seed = 42, antithetic = true
    mcs.n_paths = 20000;
    mcs.n_steps = 100;
    mcs.seed = 42;
    const McResult mc = price_european_mc(mkt, 100.0, 1.0, vol, /*is_call=*/true, mcs);
    std::printf("PDE=%.4f MC=%.4f +/- %.4f within 3 SE: %d\n", pde, mc.price, mc.std_err,
                mc.within(pde, 3.0));
}
```

**Rust**

```rust
use localvol::{price_european_mc, price_european_pde, DupireLocalVol, ImpliedVolSurface, Market,
               McSettings, PdeSettings, VolInput};

fn main() -> localvol::Result<()> {
    let lv = DupireLocalVol::new(ImpliedVolSurface::from_csv("../data/implied_surface.csv")?);
    let mkt = Market::new(100.0, 0.0, 0.0)?;
    let vol = VolInput::Local(&lv);
    let grid = PdeSettings { sigma_ref: Some(0.2), ..PdeSettings::default() };
    let pde = price_european_pde(&mkt, 100.0, 1.0, &vol, true, &grid)?;
    let mcs = McSettings { n_paths: 20_000, n_steps: 100, seed: 42, antithetic: true };
    let mc = price_european_mc(&mkt, 100.0, 1.0, &vol, true, &mcs)?;
    println!("PDE={pde:.4} MC={:.4} +/- {:.4} within 3 SE: {}", mc.price, mc.stderr, mc.within(pde, 3.0));
    Ok(())
}
```

**Java**

```java
import java.nio.file.Paths;
import com.quant.localvol.DupireLocalVol;
import com.quant.localvol.ImpliedVolSurface;
import com.quant.localvol.Market;
import com.quant.localvol.MonteCarlo;
import com.quant.localvol.Pde;

public class Cook7 {
    public static void main(String[] args) {
        DupireLocalVol lv = new DupireLocalVol(
                ImpliedVolSurface.fromCsv(Paths.get("../data/implied_surface.csv")));
        Market mkt = new Market(100.0, 0.0, 0.0);
        double pde = Pde.priceEuropean(mkt, 100.0, 1.0, lv, 0.2, true);
        MonteCarlo.McResult mc = MonteCarlo.priceEuropean(mkt, 100.0, 1.0, lv, true,
                20000, 100, 42L, true);
        System.out.printf("PDE=%.4f MC=%.4f +/- %.4f within 3 SE: %b%n",
                pde, mc.price(), mc.stderr(), mc.within(pde, 3.0));
    }
}
```

Antithetic sampling requires an even `n_paths >= 4` (two pair means at
least; `>= 2` without antithetics); the standard error is computed over
**pair means**, so `mc.stderr` is directly usable in an N-standard-error
test. Seeds are per-language and generators differ (NumPy PCG64, C++
`std::mt19937_64`, Rust `StdRng`, Java `L64X128MixRandom`); cross-language
MC comparisons must be statistical, never bitwise.

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
#include <cstdio>
#include "localvol/market.hpp"
#include "localvol/mc.hpp"
using namespace localvol;

int main() {
    const Market mkt(100.0, 0.02, 0.0);
    McSettings mcs;
    mcs.n_paths = 20000;
    mcs.n_steps = 200;
    mcs.seed = 42;
    const McResult discrete = price_up_out_call_mc(mkt, 100.0, 130.0, 1.0, 0.2, mcs, /*brownian_bridge=*/false);
    const McResult bridged = price_up_out_call_mc(mkt, 100.0, 130.0, 1.0, 0.2, mcs, /*brownian_bridge=*/true);
    std::printf("%.4f %.4f\n", discrete.price, bridged.price);   // bridged <= discrete; analytic 3.1288
}
```

**Rust**

```rust
use localvol::{price_up_out_call_mc, Market, McSettings, VolInput};

fn main() -> localvol::Result<()> {
    let mkt = Market::new(100.0, 0.02, 0.0)?;
    let mcs = McSettings { n_paths: 20_000, n_steps: 200, seed: 42, antithetic: true };
    let discrete = price_up_out_call_mc(&mkt, 100.0, 130.0, 1.0, &VolInput::Flat(0.2), &mcs, false)?;
    let bridged = price_up_out_call_mc(&mkt, 100.0, 130.0, 1.0, &VolInput::Flat(0.2), &mcs, true)?;
    println!("{:.4} {:.4}", discrete.price, bridged.price); // bridged <= discrete; analytic 3.1288
    Ok(())
}
```

**Java**

```java
import com.quant.localvol.Market;
import com.quant.localvol.MonteCarlo;

public class Cook8 {
    public static void main(String[] args) {
        Market mkt = new Market(100.0, 0.02, 0.0);
        MonteCarlo.McResult discrete = MonteCarlo.priceUpOutCall(mkt, 100.0, 130.0, 1.0, 0.2,
                20000, 200, 42L, true, false);
        MonteCarlo.McResult bridged = MonteCarlo.priceUpOutCall(mkt, 100.0, 130.0, 1.0, 0.2,
                20000, 200, 42L, true, true);
        System.out.printf("%.4f %.4f%n", discrete.price(), bridged.price()); // bridged <= discrete
    }
}
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
print(f"in={iv_in:.4%} out={iv_out:.4%} err={(iv_out - iv_in) * 1e4:.1f}bp")   # ~0.8 bp
```

**C++**

```cpp
#include <cmath>
#include <cstdio>
#include "localvol/black_scholes.hpp"
#include "localvol/dupire.hpp"
#include "localvol/market.hpp"
#include "localvol/pde.hpp"
#include "localvol/surface.hpp"
using namespace localvol;

int main() {
    const ImpliedVolSurface srf = ImpliedVolSurface::from_csv("../data/implied_surface.csv");
    const DupireLocalVol lv(srf);
    const VolFn vol = [&lv](double k, double t) { return lv.vol(k, t); };
    const Market mkt(100.0, 0.02, 0.01);
    const double strike = 95.0, T = 1.0;
    const double k = std::log(strike / mkt.forward(T));
    const double price = price_european_pde(mkt, strike, T, vol, true, PdeSettings{}, srf.implied_vol(0.0, T));
    const double iv_out = implied_vol(price, mkt.spot(), strike, mkt.rate(), mkt.dividend(), T, /*is_call=*/true);
    const double iv_in = srf.implied_vol(k, T);
    std::printf("in=%.4f out=%.4f err=%.1f bp\n", iv_in, iv_out, (iv_out - iv_in) * 1e4);
}
```

**Rust**

```rust
use localvol::{implied_vol, price_european_pde, DupireLocalVol, ImpliedVolSurface, Market, PdeSettings, VolInput};

fn main() -> localvol::Result<()> {
    let lv = DupireLocalVol::new(ImpliedVolSurface::from_csv("../data/implied_surface.csv")?);
    let mkt = Market::new(100.0, 0.02, 0.01)?;
    let (strike, t) = (95.0, 1.0);
    let k = (strike / mkt.forward(t)?).ln();
    let grid = PdeSettings { sigma_ref: Some(lv.surface().implied_vol(0.0, t)?), ..PdeSettings::default() };
    let price = price_european_pde(&mkt, strike, t, &VolInput::Local(&lv), true, &grid)?;
    let iv_out = implied_vol(price, mkt.spot(), strike, mkt.rate(), mkt.dividend(), t, true)?;
    let iv_in = lv.surface().implied_vol(k, t)?;
    println!("in={iv_in:.4} out={iv_out:.4} err={:.1} bp", (iv_out - iv_in) * 1e4);
    Ok(())
}
```

**Java**

```java
import java.nio.file.Paths;
import com.quant.localvol.BlackScholes;
import com.quant.localvol.DupireLocalVol;
import com.quant.localvol.ImpliedVolSurface;
import com.quant.localvol.Market;
import com.quant.localvol.Pde;

public class Cook9 {
    public static void main(String[] args) {
        ImpliedVolSurface srf = ImpliedVolSurface.fromCsv(Paths.get("../data/implied_surface.csv"));
        DupireLocalVol lv = new DupireLocalVol(srf);
        Market mkt = new Market(100.0, 0.02, 0.01);
        double strike = 95.0;
        double t = 1.0;
        double k = Math.log(strike / mkt.forward(t));
        double price = Pde.priceEuropean(mkt, strike, t, lv, srf.impliedVol(0.0, t), true);
        double ivOut = BlackScholes.impliedVol(price, mkt.spot(), strike, mkt.rate(), mkt.dividend(), t, true);
        double ivIn = srf.impliedVol(k, t);
        System.out.printf("in=%.4f out=%.4f err=%.1f bp%n", ivIn, ivOut, (ivOut - ivIn) * 1e4);
    }
}
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
#include <iostream>
#include <vector>
#include "localvol/surface.hpp"
using namespace localvol;

int main() {
    const std::vector<double> k_nodes = {-0.2, 0.0, 0.2};
    const std::vector<double> expiries = {1.0, 2.0};
    const std::vector<std::vector<double>> vols = {{0.25, 0.24, 0.25}, {0.15, 0.15, 0.15}};  // w drops in T
    const ImpliedVolSurface bad(k_nodes, expiries, vols);   // logs a warning to stderr
    if (bad.calendar_violations() > 0) { /* decide: refit, reject, or proceed */ }
    std::cout << bad.calendar_violations() << "\n";   // 3
    const ImpliedVolSurface single(k_nodes, {1.0}, {{0.21, 0.20, 0.21}});  // warns: single expiry
    std::cout << single.single_expiry() << " " << single.total_variance(0.0, 2.0) << "\n";  // 1, w(k,1)*2
}
```

**Rust**

```rust
use localvol::ImpliedVolSurface;

fn main() -> localvol::Result<()> {
    let k_nodes = [-0.2, 0.0, 0.2];
    let vols = vec![vec![0.25, 0.24, 0.25], vec![0.15, 0.15, 0.15]]; // w drops in T
    let bad = ImpliedVolSurface::new(&k_nodes, &[1.0, 2.0], &vols)?; // logs a warning
    if bad.calendar_violations() > 0 { /* report upstream */ }
    println!("{}", bad.calendar_violations()); // 3
    let single = ImpliedVolSurface::new(&k_nodes, &[1.0], &[vec![0.21, 0.20, 0.21]])?;
    println!("{} {}", single.single_expiry(), single.total_variance(0.0, 2.0)?); // true, w(k,1)*2
    Ok(())
}
```

**Java**

```java
import com.quant.localvol.ImpliedVolSurface;

public class Cook10 {
    public static void main(String[] args) {
        double[] kNodes = {-0.2, 0.0, 0.2};
        double[][] vols = {{0.25, 0.24, 0.25}, {0.15, 0.15, 0.15}}; // w drops in T
        ImpliedVolSurface bad = new ImpliedVolSurface(kNodes, new double[]{1.0, 2.0}, vols); // warns
        if (bad.calendarViolations() > 0) { /* report upstream */ }
        System.out.println(bad.calendarViolations()); // 3
        ImpliedVolSurface single = new ImpliedVolSurface(kNodes, new double[]{1.0},
                new double[][]{{0.21, 0.20, 0.21}}); // warns: single expiry
        System.out.println(single.singleExpiry() + " " + single.totalVariance(0.0, 2.0)); // true, w(k,1)*2
    }
}
```

All three data-quality conditions follow the project's reporting policy:
construction succeeds, a warning is emitted, and a counter/flag
(`calendar_violations`, `negative_w_count` for spline overshoot to
`w <= 0`, `single_expiry`) lets callers decide. Genuinely invalid input —
non-rectangular or duplicated CSV grids, non-positive vols, non-increasing
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
#include <iostream>
#include <vector>
#include "localvol/tridiag.hpp"
using namespace localvol;

int main() {
    const std::vector<double> sub = {1.0, 1.0, 1.0}, diag = {4.0, 4.0, 4.0, 4.0}, sup = {1.0, 1.0, 1.0};
    const std::vector<double> rhs = {5.0, 6.0, 6.0, 5.0};
    const std::vector<double> x = thomas_solve(sub, diag, sup, rhs);
    for (double v : x) std::cout << v << " ";   // 1 1 1 1
    std::cout << "\n";
}
```

**Rust**

```rust
use localvol::thomas_solve;

fn main() -> localvol::Result<()> {
    let (sub, diag, sup) = ([1.0, 1.0, 1.0], [4.0, 4.0, 4.0, 4.0], [1.0, 1.0, 1.0]);
    let rhs = [5.0, 6.0, 6.0, 5.0];
    let x = thomas_solve(&sub, &diag, &sup, &rhs)?;
    println!("{x:?}"); // [1.0, 1.0, 1.0, 1.0]
    Ok(())
}
```

**Java**

```java
import java.util.Arrays;
import com.quant.localvol.Tridiag;

public class Cook11 {
    public static void main(String[] args) {
        double[] sub = {1.0, 1.0, 1.0};
        double[] diag = {4.0, 4.0, 4.0, 4.0};
        double[] sup = {1.0, 1.0, 1.0};
        double[] rhs = {5.0, 6.0, 6.0, 5.0};
        double[] x = Tridiag.thomasSolve(sub, diag, sup, rhs);
        System.out.println(Arrays.toString(x)); // [1.0, 1.0, 1.0, 1.0]
    }
}
```

This is the single linear-algebra kernel of the whole project — the spline
moment systems and every PDE time step go through it, which is what makes
the four ports numerically comparable. It rejects near-zero pivots
(`< 1e-300`), shape mismatches and non-finite input; it is not pivoted, so
use it only on diagonally dominant systems. The PDE keeps its systems
diagonally dominant by switching to upwind differencing at any node where
the mesh Péclet condition `|mu| h <= 2a` fails (see `docs/ARCHITECTURE.md`
§5).

---

## 12. How do I share one local-vol object across threads?

`DupireLocalVol` is immutable apart from its clamp counters, which are
atomic (`std::atomic<long>` / `AtomicU64` / `LongAdder`), so one instance
can serve a parallel strike grid. The counters' *total* is exact; a read
taken while other threads are still evaluating is only a snapshot, and
`reset_counters()` belongs between phases, not inside one.

**C++**

```cpp
#include <cstdio>
#include <thread>
#include <vector>
#include "localvol/dupire.hpp"
#include "localvol/market.hpp"
#include "localvol/pde.hpp"
#include "localvol/surface.hpp"
using namespace localvol;

int main() {
    const ImpliedVolSurface srf = ImpliedVolSurface::from_csv("../data/implied_surface.csv");
    const DupireLocalVol lv(srf);                       // shared, const, thread-safe
    const Market mkt(100.0, 0.02, 0.01);
    const std::vector<double> strikes = {90.0, 100.0, 110.0, 120.0};
    std::vector<double> prices(strikes.size());
    std::vector<std::thread> pool;
    for (std::size_t i = 0; i < strikes.size(); ++i) {
        pool.emplace_back([&, i] {
            const VolFn vol = [&lv](double k, double t) { return lv.vol(k, t); };
            prices[i] = price_european_pde(mkt, strikes[i], 1.0, vol, true, PdeSettings{}, srf.implied_vol(0.0, 1.0));
        });
    }
    for (auto& t : pool) t.join();
    for (std::size_t i = 0; i < strikes.size(); ++i) std::printf("K=%.0f %.4f\n", strikes[i], prices[i]);
    std::printf("%s\n", lv.violation_report().c_str());
}
```

**Rust**

```rust
use localvol::{price_european_pde, DupireLocalVol, ImpliedVolSurface, Market, PdeSettings, VolInput};

fn main() -> localvol::Result<()> {
    let lv = DupireLocalVol::new(ImpliedVolSurface::from_csv("../data/implied_surface.csv")?); // Send + Sync
    let mkt = Market::new(100.0, 0.02, 0.01)?;
    let grid = PdeSettings { sigma_ref: Some(lv.surface().implied_vol(0.0, 1.0)?), ..PdeSettings::default() };
    let (lv_ref, mkt_ref, grid_ref) = (&lv, &mkt, &grid); // shared references move into each thread
    let prices: Vec<localvol::Result<f64>> = std::thread::scope(|s| {
        let handles: Vec<_> = [90.0, 100.0, 110.0, 120.0]
            .iter()
            .map(|&strike| {
                s.spawn(move || price_european_pde(mkt_ref, strike, 1.0, &VolInput::Local(lv_ref), true, grid_ref))
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("thread panicked")).collect()
    });
    for p in prices {
        println!("{:.4}", p?);
    }
    println!("{}", lv.violation_report());
    Ok(())
}
```

**Java**

```java
import java.nio.file.Paths;
import java.util.stream.DoubleStream;
import com.quant.localvol.DupireLocalVol;
import com.quant.localvol.ImpliedVolSurface;
import com.quant.localvol.Market;
import com.quant.localvol.Pde;

public class Cook12 {
    public static void main(String[] args) {
        ImpliedVolSurface srf = ImpliedVolSurface.fromCsv(Paths.get("../data/implied_surface.csv"));
        DupireLocalVol lv = new DupireLocalVol(srf);       // shared, thread-safe (LongAdder counters)
        Market mkt = new Market(100.0, 0.02, 0.01);
        double sigmaRef = srf.impliedVol(0.0, 1.0);
        double[] prices = DoubleStream.of(90.0, 100.0, 110.0, 120.0).parallel()
                .map(strike -> Pde.priceEuropean(mkt, strike, 1.0, lv, sigmaRef, true))
                .toArray();
        System.out.println(java.util.Arrays.toString(prices));
        System.out.println(lv.violationReport());
    }
}
```

---

## 13. How do I regenerate the data and run the golden suite everywhere?

```bash
# Regenerate CSVs + golden.json (deterministic; validates reference first)
python3 data/generate_data.py

# Python
cd python && PYTHONPATH=src python3 -m pytest -q && cd ..

# C++
cd cpp && bash build.sh && ctest --test-dir build --output-on-failure && cd ..

# Rust
cd rust && cargo test --release && cd ..

# Java
cd java && bash build.sh && bash test.sh && cd ..
```

The generator refuses to write `golden.json` unless the Python reference
passes its own validation (flat-vol PDE vs Black-Scholes to 1e-3 relative;
flat-surface Dupire exactly 0.20 to 1e-6 across a sweep; no Dupire clamp
anywhere inside the quoted box and continuity at the wings). Each language's
suite then loads the JSON, dispatches on the case-name prefix
(`flat_dupire_`, `pde_flat_`, `dupire_bundled_`, `pde_localvol_`,
`american_put_`, `mc_flat_`, `mc_localvol_`, `barrier_upout_`) and asserts
`|got - expect| <= tol` for all 15 cases. If you change any numerical rule
(interpolation, steps, schedule), regenerate the goldens *and* re-run all
four suites — the point of the file is to catch exactly such drift.

To re-verify the snippets in this file, extract each fenced `cpp` / `rust` /
`java` block into a scratch file and build it against the library (the C++
blocks link `cpp/build/liblocalvol.a`, the Rust blocks compile as `main.rs`
in a crate depending on `path = "rust"`, the Java blocks compile with
`-cp java/out`); every block is a complete program with a `main`.
