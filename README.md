# Dupire Local Volatility (Surface, PDE, Monte Carlo)

A four-language reference implementation of the full local-volatility pricing
pipeline: an implied-volatility surface in total-variance form, Dupire local
volatility (Gatheral formula), a Crank-Nicolson/Rannacher finite-difference
pricer with an American-put PSOR variant, and a log-Euler Monte Carlo engine
with a Brownian-bridge barrier correction. The Python package `localvol` is
the normative reference; the C++ (`localvol` namespace), Rust (`localvol`
crate) and Java (`com.quant.localvol`) ports mirror it and are held to the
same golden values in `data/golden/golden.json`.

The headline demo is a **round-trip consistency check**: start from a known
arbitrage-free SSVI implied surface, extract Dupire local vol, reprice
European vanillas by PDE, invert back to implied vol, and verify the surface
is recovered to better than 30 bp in the interior. Every port achieves
**2.73 bp** (worst cell K = 80, T = 2) with zero Dupire clamps on the whole
sweep; the test suites assert < 5 bp.

## Feature matrix

| Feature | Notes |
|---|---|
| Implied surface `w(k, T)` | Natural cubic spline in log-moneyness per expiry, linear total variance in T; flat wings, flat-forward-variance below the first pillar, floored-slope linear extrapolation beyond the last |
| Data-quality reporting | Calendar-arbitrage node check and spline-overshoot (`w <= 0`) scan at build time; counted and warned, never repaired or raised |
| Dupire local vol | Gatheral total-variance form, fixed-step central finite differences (`DK = 1e-3`, `DT = 1e-4`) with the stencil clamped inside the quoted box; `T = 0` defined as the Berestycki-Busca-Florent short-time limit; clamped to [1%, 500%] with thread-safe floor/cap counters |
| Black-Scholes / Garman-Kohlhagen | Price, delta, gamma, vega, bisection implied vol (100 halvings on [1e-9, 5]); equity `(r, q)` and FX `(rd, rf)` share every formula |
| PDE pricer | Log-spot theta scheme: Crank-Nicolson with Rannacher start (two backward-Euler half-steps), native Thomas solver, Dirichlet boundaries from discounted asymptotics, price read at the exact spot node; first-order upwind at any node violating the mesh Péclet condition (M-matrix preserved) |
| American put | PSOR (projected SOR) on the same theta-scheme systems; `omega = 1.5`, `tol = 1e-8`, warm-started; non-convergence warns, never throws |
| Monte Carlo | Log-Euler with start-of-step local-vol lookup, antithetic pairs (pair-mean standard error), fixed seeds; user vol callables checked for NaN/negative output |
| Barrier (up-and-out call) | Discrete monitoring plus optional Brownian-bridge survival weights removing the O(sqrt(dt)) monitoring bias |
| Edge cases | T = 0 (surface, Dupire, PDE), sigma = 0, K = 0, negative rates, single-expiry surface (warns), strikes beyond the grid, clamp engagement on arbitrage-violating input, deep-ITM American, non-converged PSOR |
| Input validation | NaN/inf/negative/empty/unsorted/duplicate inputs rejected at every public entry point with the language's idiomatic error (`ValueError`, `std::invalid_argument`, `Err(LocalVolError)`, `IllegalArgumentException`) |
| Cross-language goldens | 15 named cases covering every engine; MC cases use 4-standard-error statistical tolerances |

## Directory layout

```
local-volatility/
  README.md, LEARN.md, COOKBOOK.md
  API_SPEC.md            # normative language-neutral contract for all ports
  docs/
    ARCHITECTURE.md      # component design, data flow, numerical decisions
    GITHUB_PAGES.md      # publishing guide for the docs site: https://ashjha0.github.io/local-volatility/
    diagrams/*.mmd       # raw Mermaid sources for the embedded diagrams
  data/
    implied_surface.csv  # SSVI-sampled arb-free surface: T in {0.25,0.5,1,2} x 15 k-nodes
    flat_surface.csv     # 20% flat on the same grid
    generate_data.py     # deterministic regeneration of both CSVs + golden.json
    golden/golden.json   # 15 cross-language golden cases
  python/
    src/localvol/        # reference implementation (src layout)
    tests/               # pytest suite incl. golden cases (90 tests)
    demo.py              # round-trip consistency demo
    requirements.txt
  cpp/
    CMakeLists.txt, include/localvol/*.hpp, src/*.cpp, tests/*.cpp, build.sh   (72 tests)
  rust/
    Cargo.toml, src/lib.rs, src/*.rs, src/bin/demo.rs, tests/*.rs             (73 tests)
  java/
    src/main/java/com/quant/localvol/*.java
    src/test/java/com/quant/localvol/*Test.java                                (70 tests)
    build.sh, test.sh, demo.sh
```

## Build, test and run — all four languages

All commands work on a fresh Linux clone; scripts are invoked with `bash` so
they do not depend on the executable bit.

### Python (reference)

```bash
cd python
PYTHONPATH=src python3 -m pytest -q    # test suite (incl. golden cases): 90 passed
PYTHONPATH=src python3 demo.py         # round-trip demo
```

Requires Python 3.11 with `numpy >= 1.24` and `pytest` (SciPy is used only
by one test, as an independent banded-solver cross-check; see
`python/requirements.txt`).

### C++ (C++17, CMake >= 3.16, GoogleTest, g++ 13)

```bash
cd cpp
bash build.sh    # cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j2
ctest --test-dir build --output-on-failure   # 72 tests
./build/demo                                 # demo binary (CMake target `demo`)
```

The build is warning-free under `-Wall -Wextra`; the test binary is
`./build/localvol_tests` and links `Threads::Threads` for the thread-safety
test.

### Rust (edition 2021)

```bash
cd rust
cargo build --release      # zero warnings
cargo test --release       # 73 tests
cargo run --release --bin demo
```

Dependencies: `rand 0.8`, `rand_distr 0.4` (runtime); `serde`,
`serde_json` (tests only, golden loader). `Cargo.lock` is committed.

### Java (21, JUnit 4 — no Maven/Gradle)

```bash
cd java
bash build.sh    # javac -Xlint:all everything into out/ (zero warnings)
bash test.sh     # JUnitCore on all *Test classes: 70 tests
bash demo.sh     # runs the demo main (data dir defaults to ../data)
```

The scripts expect JUnit 4 at `/usr/share/java/junit4.jar` and Hamcrest at
`/usr/share/java/hamcrest.jar` (Debian/Ubuntu `junit4` + `libhamcrest-java`
packages); edit the two variables at the top of `build.sh` / `test.sh` for
other locations.

## Demo output (Python reference; the three native demos print the same numbers)

```text
========================================================================
localvol demo — Dupire local volatility round-trip consistency
========================================================================
surface: 4 expiries x 15 strikes, calendar violations: 0
market : S0=100.00  r=2.00%  q=1.00%

Round-trip implied-vol error (PDE 200x200), basis points:
  T\K         80      90      95     100     105     110     120
  0.50      0.3     0.9     0.2     0.9     0.1     0.8     1.7
  1.00      0.8     1.0     0.8     1.0     0.6     0.1     0.2
  1.50      2.5     0.8     0.6     1.2     0.5     1.0     0.5
  2.00      2.7     1.2     1.1     1.2     0.4     0.2     0.4

max interior |vol error|: 2.73 bp at K=80, T=2.00  (target < 30 bp, |k| <= 0.30)
round-trip check: PASS
local-vol clamps: floor(1%) hit 0x, cap(500%) hit 0x

PDE vs MC (K=100, T=1): PDE=8.4131  MC=8.4508 +/- 0.0579  |diff|=0.65 SE (OK)
American put (PSOR)     : 7.5123  (European 7.4280, premium +0.0843)
Up-and-out call B=130 MC: 4.4993 +/- 0.0364 (vanilla 8.4131; barrier <= vanilla: OK)

done.
```

The PDE, American and round-trip numbers are identical in all four ports;
the MC lines differ per language because each uses its own fixed-seed
generator (they agree within the printed standard errors). The clamp
counters cover the whole 4 × 7 sweep and are exactly zero: the Dupire
stencil is clamped inside the quoted strike box, so the flat wing
extrapolation cannot trigger the 500% cap (an earlier revision let the
stencil straddle the wing kink, which produced 500% local vols *at the last
quoted strike* and a 16.4 bp round-trip error). On a smooth input surface
any non-zero counter now means the input carries genuine arbitrage.

## Golden values — cross-language testing

`data/golden/golden.json` holds 15 named cases with flat scalar
inputs/outputs and an **absolute** tolerance per case
(`|got - expect| <= tol`). Every language's test suite loads the file,
dispatches on the case-name prefix, runs its own implementation and asserts
agreement:

| Prefix | Engine | Reference |
|---|---|---|
| `flat_dupire_` | Dupire on `flat_surface.csv` | exactly 0.20 (tol 1e-6) |
| `pde_flat_` | European PDE, flat vol | closed-form Black-Scholes / Garman-Kohlhagen |
| `dupire_bundled_` | Dupire on `implied_surface.csv`, expiries placed 1e-3 off the pillars | Python reference vols (tol 1e-4 abs) |
| `pde_localvol_` | European PDE under the bundled Dupire vol | Python reference price (1e-3 rel, pre-multiplied) |
| `american_put_` | PSOR American put, flat vol | CRR binomial, 5000 steps |
| `mc_flat_`, `mc_localvol_`, `barrier_upout_` | Monte Carlo (statistical) | BS / PDE / analytic barrier, tolerance sized at 4 standard errors of the stated configuration (plus a residual-bias allowance for the barrier) |

Deterministic engines (surface, Dupire, PDE) must agree across languages to
tight absolute tolerances because the entire numerical path — spline system,
Thomas solver, finite-difference steps, wing clamp, Rannacher schedule,
upwind switch — is pinned down exactly in `API_SPEC.md`. MC cases are
statistical: each language uses its own fixed-seed normal generator
(deterministic for a fixed toolchain, not across toolchains), so tolerances
were sized at generation time to 4 standard errors. Regenerate everything
with `python3 data/generate_data.py` (fixed seed; validates the reference
against Black-Scholes, the flat-surface identity and a no-clamp sweep of the
quoted box before writing).

## Real-world usage notes

What a practitioner gets, precisely:

* **Units and conventions.** Spot is a price (or an FX rate, domestic per
  foreign); rates are continuously compounded; `T`, `t` and expiries are
  bare year fractions. There is **no day-count convention, calendar,
  settlement lag or date type** — converting dates to year fractions
  (ACT/365F, business time, …) is the caller's job and must be consistent
  between the surface CSV and the pricer inputs. Vols are Black implied vols
  in absolute terms (0.20 = 20%). `k` is forward log-moneyness
  `ln(K / F(T))` everywhere; the surface is carry-free and the pricers
  reintroduce carry through `Market`.
* **Model scope.** Continuous dividend yield / foreign rate only: no
  discrete cash dividends, no borrow curve, no term structure of rates
  (a single flat `r` and `q` per pricing call). European calls/puts,
  American puts and continuously-monitored up-and-out calls only; no
  Greeks from the PDE grid beyond what `PDEResult` exposes (the final value
  grid); no calibration — the surface is taken as given.
* **Parameter bounds that are validated** at every public entry point:
  finite everything; `spot > 0`; `strike >= 0` (`> 0` for American and for
  the implied-vol inverter); `expiry >= 0` (`> 0` for MC); `sigma >= 0`
  (`> 0` for flat-vol MC); strictly increasing strike and expiry nodes,
  strictly positive vols; `num_space` even and `>= 4`; `num_time >= 1`;
  `nsd > 0`; `omega` in `(0, 2)`; `tol` finite and `> 0`; `max_iter >= 1`;
  `n_paths >= 4` and even with antithetics (`>= 2` without); `n_steps >= 1`;
  user vol callables must return finite, non-negative sigma; CSV grids must
  be rectangular with no duplicate `(T, k)`; implied-vol prices must lie
  inside the static no-arbitrage bounds. Violations raise the idiomatic
  error; nothing propagates NaN silently.
* **What is reported rather than raised** (counters + warning/log):
  calendar arbitrage at the nodes, spline overshoot to `w <= 0` between
  nodes, Dupire floor/cap clamps, PSOR non-convergence, single-expiry
  surfaces. Read the counters after a pricing sweep; on a smooth surface
  they should all be zero.
* **Accuracy that is verified by the test suites** (each language): flat-vol
  PDE vs Black-Scholes to 1e-3 relative on 200×200; observed convergence
  order in [1.5, 2.5] (flat) and [0.8, 2.5] (local vol); non-oscillatory
  gamma; American put vs CRR-5000 within 1e-2; MC vs closed form / PDE
  within 3–4 SE with and without carry; barrier MC vs Reiner-Rubinstein
  within 4 SE + 0.02; round-trip implied-vol recovery < 5 bp on the demo
  grid; flat-surface Dupire identity to 1e-6; `T = 0` local vol continuous
  with `T -> 0+`. What is **not** verified: behaviour on real, noisy market
  quotes (the bundled surface is a smooth SSVI sample; a desk should fit an
  arbitrage-free parametric form per slice before feeding Dupire — see
  LEARN.md §3).
* **Numerical limits to know.** Local vol is discontinuous in `T` at every
  pillar (piecewise-constant forward variance); the finite-difference
  Dupire steps are fixed (`DK = 1e-3`, `DT = 1e-4`); the PDE grid is
  uniform with no local refinement; Rannacher uses two backward-Euler
  half-steps (price second order; Greeks at the kink retain an O(dt)
  component — four half-steps would restore it); upwinding at Péclet-
  violating nodes adds O(|mu| h / 2) numerical diffusion there; the
  Brownian-bridge weight uses the frozen per-step vol.
* **Threading.** `ImpliedVolSurface` is immutable; `DupireLocalVol.vol()`
  is safe to call from several threads on one shared object (atomic
  counters; totals exact, mid-sweep reads are snapshots). Pricer calls are
  independent and reentrant. The Python reference is single-threaded.
* **Performance** (measured on the 2-core review box, `-O2` / `--release`,
  C++ figures; Rust and Java are the same order): a 200×200 European PDE is
  201 Thomas solves of size 199 — 0.7 ms flat, ~5 ms under local vol (the
  ~40k Dupire evaluations dominate); Python takes ~50 ms / ~115 ms. The
  20k × 100 local-vol MC takes ~0.26 s natively and ~1.2 s in NumPy. Nothing
  is tuned for throughput; the priority is cross-language reproducibility.

## References

Works the code implements or the documentation relies on:

1. B. Dupire (1994), "Pricing with a Smile", *Risk* 7(1), 18–20. — Local-vol
   forward equation (LEARN §2.1).
2. E. Derman, I. Kani (1994), "Riding on a Smile", *Risk* 7(2), 32–39. —
   Tree formulation, referenced in LEARN §1.
3. J. Gatheral (2006), *The Volatility Surface: A Practitioner's Guide*,
   Wiley, ISBN 978-0-471-79251-2, Ch. 1 eq. (1.10). — Total-variance form
   of the Dupire formula implemented in `dupire.py` and its ports.
4. H. Berestycki, J. Busca, I. Florent (2002), "Asymptotics and calibration
   of local volatility models", *Quantitative Finance* 2(1), 61–69,
   doi:10.1088/1469-7688/2/1/305. — Short-time limit
   `sigma_loc(k, 0) = sigma_imp / (1 - k sigma_imp'/sigma_imp)` used for
   `vol(k, 0)`.
5. J. Gatheral, A. Jacquier (2014), "Arbitrage-free SVI volatility
   surfaces", *Quantitative Finance* 14(1), 59–71,
   doi:10.1080/14697688.2013.819986 (SSRN 2033323). — SSVI family and the
   no-butterfly condition `eta (1 + |rho|) <= 2` behind
   `data/generate_data.py`.
6. I. Gyöngy (1986), "Mimicking the one-dimensional marginal distributions
   of processes having an Itô differential", *Probability Theory and Related
   Fields* 71(4), 501–516, doi:10.1007/BF00699039. — Marginals theorem
   (LEARN §1).
7. R. Rannacher (1984), "Finite element solution of diffusion problems with
   irregular data", *Numerische Mathematik* 43(2), 309–327,
   doi:10.1007/BF01390130. — Implicit start-up steps.
8. M. B. Giles, R. Carter (2006), "Convergence analysis of Crank–Nicolson
   and Rannacher time-marching", *Journal of Computational Finance* 9(4),
   89–112, doi:10.21314/JCF.2006.152. — Analysis of the Rannacher start;
   this repository uses two half-steps rather than their four (see
   ARCHITECTURE §5).
9. D. Tavella, C. Randall (2000), *Pricing Financial Instruments: The Finite
   Difference Method*, Wiley, ISBN 978-0-471-19760-7. — Log-spot theta
   scheme, Dirichlet asymptotic boundaries, mesh Péclet condition and
   upwinding.
10. C. W. Cryer (1971), "The solution of a quadratic programming problem
    using systematic overrelaxation", *SIAM Journal on Control* 9(3),
    385–392, doi:10.1137/0309028. — Projected SOR for the American LCP.
11. P. Wilmott, J. Dewynne, S. Howison (1993), *Option Pricing: Mathematical
    Models and Computation*, Oxford Financial Press, ISBN 978-0-9522082-0-9,
    Ch. 9. — PSOR as applied to American options.
12. J. C. Cox, S. A. Ross, M. Rubinstein (1979), "Option pricing: a
    simplified approach", *Journal of Financial Economics* 7(3), 229–263,
    doi:10.1016/0304-405X(79)90015-1. — CRR binomial reference for the
    American golden case.
13. E. Reiner, M. Rubinstein (1991), "Breaking down the barriers", *Risk*
    4(8), 28–35; and E. G. Haug (2007), *The Complete Guide to Option
    Pricing Formulas*, 2nd ed., McGraw-Hill, ISBN 978-0-07-138997-6, §4.17.
    — Up-and-out call closed form (`generate_data.py`).
14. P. Glasserman (2004), *Monte Carlo Methods in Financial Engineering*,
    Springer, ISBN 978-0-387-00451-8, §4.2 (antithetics), §6.1 (Euler
    schemes), §6.4 (barrier crossings).
15. D. Beaglehole, P. Dybvig, G. Zhou (1997), "Going to extremes: correcting
    simulation bias in exotic option valuation", *Financial Analysts
    Journal* 53(1), 62–68, doi:10.2469/faj.v53.n1.2057. — Brownian-bridge
    crossing probability `exp(-2 (b - x0)(b - x1) / (sigma^2 dt))`.
16. M. B. Garman, S. W. Kohlhagen (1983), "Foreign currency option values",
    *Journal of International Money and Finance* 2(3), 231–237,
    doi:10.1016/S0261-5606(83)80001-1. — FX as `r = rd`, `q = rf`.
17. C. de Boor (1978), *A Practical Guide to Splines*, Springer, ISBN
    978-0-387-95366-3, Ch. IV. — Natural cubic spline in moment form.
18. M. Abramowitz, I. A. Stegun (1964), *Handbook of Mathematical
    Functions*, NBS Applied Mathematics Series 55, §7.1.14 (continued
    fraction for erfc). — `erfc` implementation in the Rust and Java ports.
19. P. S. Hagan, D. Kumar, A. S. Lesniewski, D. E. Woodward (2002),
    "Managing Smile Risk", *Wilmott Magazine*, September 2002, 84–108. —
    Critique of local-vol smile dynamics (LEARN §6).

## License

MIT License — see the `LICENSE` file (Copyright (c) 2026 Ashish Jha).
Provided "as is", without warranty of any kind; not investment advice.
