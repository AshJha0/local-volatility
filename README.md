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
is recovered to better than 30 bp in the interior (the reference achieves
about 16 bp; worst point is the deep put wing at T = 2).

## Feature matrix

| Feature | Notes |
|---|---|
| Implied surface `w(k, T)` | Natural cubic spline in log-moneyness per expiry, linear total variance in T; flat wings, flat-forward-variance below the first pillar, floored-slope linear extrapolation beyond the last |
| Calendar-arbitrage detection | Node-level `w` monotonicity check; violations counted and reported, never repaired or raised |
| Dupire local vol | Gatheral total-variance form, fixed-step central finite differences (`DK = 1e-3`, `DT = 1e-4`), clamped to [1%, 500%] with floor/cap counters |
| Black-Scholes / Garman-Kohlhagen | Price, delta, gamma, vega, bisection implied vol (100 halvings on [1e-9, 5]); equity `(r, q)` and FX `(rd, rf)` share every formula |
| PDE pricer | Log-spot theta scheme: Crank-Nicolson with Rannacher start (two backward-Euler half-steps), native Thomas solver, Dirichlet boundaries from discounted asymptotics, price read at the exact spot node |
| American put | PSOR (projected SOR) on the same theta-scheme systems; `omega = 1.5`, `tol = 1e-8`, warm-started; non-convergence warns, never throws |
| Monte Carlo | Log-Euler with start-of-step local-vol lookup, antithetic pairs (pair-mean standard error), fixed seeds |
| Barrier (up-and-out call) | Discrete monitoring plus optional Brownian-bridge survival weights removing the O(sqrt(dt)) monitoring bias |
| Edge cases | T = 0, sigma = 0, K = 0, negative rates, single-expiry surface (warns), strikes beyond the grid, clamp engagement on extreme skew |
| Cross-language goldens | 15 named cases covering every engine; MC cases use 4-standard-error statistical tolerances |

## Directory layout

```
03-local-volatility/
  README.md, LEARN.md, COOKBOOK.md
  API_SPEC.md            # normative language-neutral contract for all ports
  docs/
    ARCHITECTURE.md      # component design, data flow, numerical decisions
    diagrams/*.mmd       # raw Mermaid sources for the embedded diagrams
  data/
    implied_surface.csv  # SSVI-sampled arb-free surface: T in {0.25,0.5,1,2} x 15 k-nodes
    flat_surface.csv     # 20% flat on the same grid
    generate_data.py     # deterministic regeneration of both CSVs + golden.json
    golden/golden.json   # 15 cross-language golden cases
  python/
    src/localvol/        # reference implementation (src layout)
    tests/               # pytest suite incl. golden cases
    demo.py              # round-trip consistency demo
    requirements.txt
  cpp/
    CMakeLists.txt, include/localvol/*.hpp, src/*.cpp, tests/*.cpp, build.sh
  rust/
    Cargo.toml, src/lib.rs, src/*.rs, src/bin/demo.rs, tests/*.rs
  java/
    src/main/java/com/quant/localvol/*.java
    src/test/java/com/quant/localvol/*Test.java
    build.sh, test.sh, demo.sh
```

## Build, test and run — all four languages

### Python (reference)

```bash
cd python
PYTHONPATH=src pytest -q          # test suite (incl. golden cases)
PYTHONPATH=src python3 demo.py    # round-trip demo
```

Requires Python 3.11 with `numpy` (SciPy is used only by tests, as an
independent banded-solver cross-check).

### C++ (C++17, CMake, GoogleTest)

```bash
cd cpp
./build.sh    # cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j2
ctest --test-dir build --output-on-failure   # tests
./build/demo                                 # demo
```

### Rust (edition 2021)

```bash
cd rust
cargo build --release
cargo test
cargo run --release --bin demo
```

### Java (21, JUnit 4 — no Maven/Gradle)

```bash
cd java
./build.sh    # javac everything into out/
./test.sh     # JUnitCore on all *Test classes (JUnit at /usr/share/java/junit4.jar)
./demo.sh     # runs the demo main
```

## Demo output (Python reference, truncated)

```text
========================================================================
localvol demo — Dupire local volatility round-trip consistency
========================================================================
surface: 4 expiries x 15 strikes, calendar violations: 0
market : S0=100.00  r=2.00%  q=1.00%

Round-trip implied-vol error (PDE 200x200), basis points:
  T\K         80      90      95     100     105     110     120
  0.50      1.3     1.0     0.2     1.0     0.1     0.8     1.7
  1.00      5.9     2.1     1.2     1.2     0.7     0.1     0.2
  1.50     12.1     3.4     1.9     2.0     0.9     1.2     0.6
  2.00     16.4     6.2     4.2     3.1     1.5     0.9     0.2

max interior |vol error|: 16.39 bp at K=80, T=2.00  (target < 30 bp, |k| <= 0.30)
round-trip check: PASS
local-vol clamps: floor(1%) hit 0x, cap(500%) hit 278x

PDE vs MC (K=100, T=1): PDE=8.4124  MC=8.5165 +/- 0.0625  |diff|=1.66 SE (OK)
American put (PSOR)     : 7.5123  (European 7.4273, premium +0.0850)
Up-and-out call B=130 MC: 4.5031 +/- 0.0364 (vanilla 8.4124; barrier <= vanilla: OK)

done.
```

(The 278 cap hits are expected: the PDE grid spans 6 standard deviations, so
its far nodes query the surface well outside the quoted wings, where flat
extrapolation makes the Dupire denominator go non-positive. The clamps are
reported, and the affected nodes are far enough from the spot that the
interior round-trip error stays under 17 bp.)

## Golden values — cross-language testing

`data/golden/golden.json` holds 15 named cases with flat scalar
inputs/outputs and an **absolute** tolerance per case
(`|got - expect| <= tol`). Every language's test suite loads the file,
dispatches on the name prefix, runs its own implementation and asserts
agreement:

| Prefix | Engine | Reference |
|---|---|---|
| `flat_dupire_` | Dupire on `flat_surface.csv` | exactly 0.20 (tol 1e-6) |
| `pde_flat_` | European PDE, flat vol | closed-form Black-Scholes / Garman-Kohlhagen |
| `dupire_bundled_` | Dupire on `implied_surface.csv` | Python reference vols (tol 1e-4 abs) |
| `pde_localvol_` | European PDE under the bundled Dupire vol | Python reference price (1e-3 rel, pre-multiplied) |
| `american_put_` | PSOR American put, flat vol | CRR binomial, 5000 steps |
| `mc_flat_`, `mc_localvol_`, `barrier_upout_` | Monte Carlo (statistical) | BS / PDE / analytic barrier, tolerance sized at 4 standard errors of the stated configuration (plus a residual-bias allowance for the barrier) |

Deterministic engines (surface, Dupire, PDE) must agree across languages to
tight absolute tolerances because the entire numerical path — spline system,
Thomas solver, finite-difference steps, Rannacher schedule — is pinned down
exactly in `API_SPEC.md`. MC cases are statistical: each language uses its
own fixed-seed normal generator, so tolerances were sized at generation time
to 4 standard errors, making each run deterministic per language yet
comparable across languages. Regenerate everything with
`python3 data/generate_data.py` (fixed seed; validates the reference against
Black-Scholes and the flat-surface identity before writing).

## License

Educational use. No warranty; not investment advice.
