"""Regenerate the bundled datasets and cross-language golden values.

Run from anywhere:  python3 data/generate_data.py

Outputs (all deterministic, fixed seed):
  data/implied_surface.csv  — smooth arbitrage-free SSVI surface samples
  data/flat_surface.csv     — 20% flat surface on the same grid
  data/golden/golden.json   — ~15 named cross-language reference cases

The implied surface is a Gatheral–Jacquier SSVI (surface SVI) family: each
expiry slice is an SVI smile

    w(k, T) = theta_T / 2 * (1 + rho phi k + sqrt((phi k + rho)^2 + 1 - rho^2))

with theta_T = SIGMA0^2 * T and phi = ETA / (theta^GAMMA (1 + theta)^(1-GAMMA)).
With GAMMA = 1/2 and ETA (1 + |RHO|) <= 2 the surface is free of butterfly
and calendar arbitrage, and the slices are calendar-consistent by
construction (theta_T increasing).  Sampling it on a (T, k) grid gives a
realistic equity-skew surface that the interpolated ImpliedVolSurface can
round-trip through Dupire.

Before writing golden.json the script VALIDATES the reference
implementation: flat-vol PDE prices must match Black-Scholes to 1e-3
relative, and Dupire on the flat surface must return exactly 20% (to 1e-6)
— any failure aborts generation.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DATA_DIR.parent / "python" / "src"))

from localvol import (  # noqa: E402
    DupireLocalVol,
    ImpliedVolSurface,
    Market,
    bs_price,
    norm_cdf,
    price_american_put_pde,
    price_european_mc,
    price_european_pde,
    price_up_out_call_mc,
)

SEED = 20260827  # fixed seed for anything stochastic (MC tolerance sizing)

# ---------------------------------------------------------------- SSVI family
SIGMA0 = 0.20   # ATM vol level: theta_T = SIGMA0^2 * T
RHO = -0.5      # skew (equity-like: negative)
ETA = 0.7       # smile curvature; ETA * (1 + |RHO|) = 1.05 <= 2 (no butterfly)
GAMMA = 0.5     # phi exponent (calendar consistency for gamma in (0, 1/2])

EXPIRIES = [0.25, 0.5, 1.0, 2.0]
K_NODES = [round(k, 10) for k in np.linspace(-0.5, 0.5, 15)]


def ssvi_total_variance(k: float, T: float) -> float:
    """SSVI total variance w(k, T); see module docstring."""
    theta = SIGMA0 * SIGMA0 * T
    phi = ETA / (theta**GAMMA * (1.0 + theta) ** (1.0 - GAMMA))
    pk = phi * k
    return 0.5 * theta * (1.0 + RHO * pk + math.sqrt((pk + RHO) ** 2 + 1.0 - RHO * RHO))


def write_surface_csv(path: Path, iv_fn) -> None:
    lines = ["T,k,iv"]
    for T in EXPIRIES:
        for k in K_NODES:
            lines.append(f"{T},{k:.10f},{iv_fn(k, T):.10f}")
    path.write_text("\n".join(lines) + "\n")


def up_out_call_analytic(
    s: float, k: float, b: float, r: float, q: float, sigma: float, T: float
) -> float:
    """Continuous-barrier up-and-out call (zero rebate), K < B, S < B.

    Standard reflection-principle formula (Merton / Reiner-Rubinstein):
    with carry ``cb = r - q`` and ``mu = (cb - sigma^2/2)/sigma^2``,
    UOC = A - B + C - D in Haug's notation with phi = 1, eta = -1.
    """
    if not (k < b and s < b):
        raise ValueError("formula requires K < B and S < B")
    cb = r - q
    v = sigma * math.sqrt(T)
    mu = (cb - 0.5 * sigma * sigma) / (sigma * sigma)
    df_r = math.exp(-r * T)
    growth = math.exp((cb - r) * T)  # e^{-qT}

    def vanilla_term(x: float, phi: float) -> float:
        return phi * s * growth * norm_cdf(phi * x) - phi * k * df_r * norm_cdf(phi * (x - v))

    def reflect_term(y: float, eta: float) -> float:
        pw = (b / s) ** (2.0 * (mu + 1.0))
        pw2 = (b / s) ** (2.0 * mu)
        return s * growth * pw * norm_cdf(eta * y) - k * df_r * pw2 * norm_cdf(eta * (y - v))

    x1 = math.log(s / k) / v + (1.0 + mu) * v
    x2 = math.log(s / b) / v + (1.0 + mu) * v
    y1 = math.log(b * b / (s * k)) / v + (1.0 + mu) * v
    y2 = math.log(b / s) / v + (1.0 + mu) * v
    A = vanilla_term(x1, 1.0)
    B_ = vanilla_term(x2, 1.0)
    C_ = reflect_term(y1, -1.0)
    D_ = reflect_term(y2, -1.0)
    return A - B_ + C_ - D_


def crr_american_put(s: float, k: float, r: float, q: float, sigma: float, T: float, steps: int) -> float:
    """Cox-Ross-Rubinstein binomial American put (vectorised backward walk)."""
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp((r - q) * dt) - d) / (u - d)
    if not (0.0 < p < 1.0):
        raise ValueError(f"CRR probability out of (0,1): {p}")
    j = np.arange(steps + 1)
    st = s * u ** (2.0 * j - steps)  # terminal prices, ascending
    v = np.maximum(k - st, 0.0)
    for n in range(steps - 1, -1, -1):
        st = s * u ** (2.0 * np.arange(n + 1) - n)
        v = disc * (p * v[1:] + (1.0 - p) * v[:-1])
        v = np.maximum(v, k - st)
    return float(v[0])


def main() -> None:
    np.random.seed(SEED)  # nothing below draws from the legacy RNG, but pin it anyway
    golden_dir = DATA_DIR / "golden"
    golden_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------- CSV files
    write_surface_csv(
        DATA_DIR / "implied_surface.csv",
        lambda k, T: math.sqrt(ssvi_total_variance(k, T) / T),
    )
    write_surface_csv(DATA_DIR / "flat_surface.csv", lambda k, T: 0.20)
    print("wrote implied_surface.csv and flat_surface.csv")

    flat = ImpliedVolSurface.from_csv(DATA_DIR / "flat_surface.csv")
    bundled = ImpliedVolSurface.from_csv(DATA_DIR / "implied_surface.csv")
    assert flat.calendar_violations == 0 and bundled.calendar_violations == 0

    cases: list[dict] = []

    def add(name: str, inputs: dict, expect: dict, tol: float) -> None:
        cases.append({"name": name, "inputs": inputs, "expect": expect, "tol": tol})

    # ------------------------- 1-2: flat surface -> local vol == implied vol
    flat_lv = DupireLocalVol(flat)
    for name, k, T in [("flat_dupire_atm_T1", 0.0, 1.0), ("flat_dupire_wing_k03_T05", 0.3, 0.5)]:
        v = float(flat_lv.vol(k, T))
        assert abs(v - 0.20) < 1e-6, f"flat Dupire validation failed: {v}"
        add(name, {"k": k, "T": T}, {"local_vol": 0.20}, 1e-6)
    # broader validation sweep (not written to golden): whole grid
    for k in np.linspace(-0.45, 0.45, 7):
        for T in [0.1, 0.3, 0.75, 1.5, 2.5]:
            assert abs(float(flat_lv.vol(float(k), float(T))) - 0.20) < 1e-6
    print("validated: flat-surface Dupire == 0.20 to 1e-6 across the grid")

    # ------------------------- 3-5: flat-vol PDE vs Black-Scholes references
    pde_flat_specs = [
        ("pde_flat_call_equity", dict(s=100.0, k=100.0, r=0.05, q=0.02, sigma=0.2, T=1.0, call=1)),
        ("pde_flat_put_fx_gk", dict(s=1.10, k=1.05, r=0.03, q=0.01, sigma=0.10, T=0.5, call=0)),
        ("pde_flat_call_negrate", dict(s=100.0, k=110.0, r=-0.01, q=0.0, sigma=0.25, T=2.0, call=1)),
    ]
    for name, p in pde_flat_specs:
        mkt = Market(p["s"], p["r"], p["q"])
        exact = bs_price(p["s"], p["k"], p["r"], p["q"], p["sigma"], p["T"], bool(p["call"]))
        approx = price_european_pde(
            mkt, p["k"], p["T"], p["sigma"], is_call=bool(p["call"]), num_space=200, num_time=200
        )
        rel = abs(approx - exact) / exact
        assert rel < 1e-3, f"{name}: PDE vs BS rel error {rel:.2e} exceeds 1e-3"
        add(name, {**p, "num_space": 200, "num_time": 200}, {"price": round(exact, 10)}, round(1e-3 * exact, 12))
        print(f"validated: {name} PDE vs BS rel err {rel:.2e}")

    # --------- 6-10: Dupire local vol at 5 interior points, bundled surface.
    # Reference market for the bundled-surface cases: S0=100, r=q=0, so the
    # forward is 100 and k = ln(K/100) maps to familiar strike labels.
    # Linear-in-w time interpolation makes the forward variance dw/dT
    # piecewise constant, so sigma_loc jumps at every pillar; the golden
    # expiries therefore sit 1e-3 *off* the pillars (0.501, 1.001) so that
    # the DT = 1e-4 central stencil never straddles a jump and a port that
    # differentiates one-sidedly still reproduces the values.
    lv = DupireLocalVol(bundled)
    dupire_pts = [
        ("dupire_bundled_k80_T0501", 80.0, 0.501),
        ("dupire_bundled_k95_T1001", 95.0, 1.001),
        ("dupire_bundled_k100_T1001", 100.0, 1.001),
        ("dupire_bundled_k110_T15", 110.0, 1.5),
        ("dupire_bundled_k120_T075", 120.0, 0.75),
    ]
    lv.reset_counters()
    for name, strike, T in dupire_pts:
        k = math.log(strike / 100.0)
        for pillar in bundled.expiries:
            assert abs(T - pillar) > 5e-4, f"{name}: golden expiry {T} sits on pillar {pillar}"
        v = float(lv.vol(k, T))
        add(name, {"k": round(k, 12), "T": T}, {"local_vol": round(v, 10)}, 1e-4)
        print(f"golden {name}: k={k:+.6f} T={T} local_vol={v:.6f}")
    assert lv.floor_count == 0 and lv.cap_count == 0, "clamps fired on interior golden points"

    # Validation sweep (not written): with the wing-clamped stencil no clamp
    # may fire anywhere inside the quoted box, and the local vol must be
    # continuous across the last quoted strike.
    lv.reset_counters()
    for T in [0.1, 0.5, 0.75, 1.0, 2.0, 3.0]:
        lv.vol(np.arange(-0.5, 0.5 + 1e-9, 0.005), T)
        for edge in (bundled.k_min, bundled.k_max):
            jump = abs(float(lv.vol(edge - 1e-4, T)) - float(lv.vol(edge + 1e-4, T)))
            assert jump < 1e-3, f"local vol discontinuous at k={edge}, T={T}: {jump}"
    assert lv.floor_count == 0 and lv.cap_count == 0, (
        f"clamps fired inside the quoted box: {lv.violation_report}"
    )
    print("validated: no Dupire clamps inside the quoted box; continuous at the wings")

    # ------------------- 11: local-vol PDE price on the bundled surface.
    mkt0 = Market(100.0, 0.0, 0.0)
    sigma_ref = float(bundled.implied_vol(0.0, 1.0))
    pde_lv_price = price_european_pde(
        mkt0, 100.0, 1.0, lv.vol, is_call=True, num_space=200, num_time=200, sigma_ref=sigma_ref
    )
    # sanity: implied vol of that price should sit near the input ATM vol
    from localvol import implied_vol as bs_implied
    iv_back = bs_implied(pde_lv_price, 100.0, 100.0, 0.0, 0.0, 1.0, True)
    iv_in = float(bundled.implied_vol(0.0, 1.0))
    assert abs(iv_back - iv_in) < 30e-4, f"round-trip ATM error {abs(iv_back-iv_in)*1e4:.1f}bp"
    add(
        "pde_localvol_bundled_k100_T1",
        {"s": 100.0, "k": 100.0, "r": 0.0, "q": 0.0, "T": 1.0, "num_space": 200, "num_time": 200, "sigma_ref": round(sigma_ref, 10)},
        {"price": round(pde_lv_price, 10)},
        round(1e-3 * pde_lv_price, 12),
    )
    print(f"golden pde_localvol_bundled_k100_T1: price={pde_lv_price:.6f} (round-trip ATM err {abs(iv_back-iv_in)*1e4:.2f}bp)")

    # ------------------- 12: American put, flat vol, CRR-5000 reference.
    am = dict(s=100.0, k=100.0, r=0.05, q=0.0, sigma=0.2, T=1.0)
    crr = crr_american_put(am["s"], am["k"], am["r"], am["q"], am["sigma"], am["T"], 5000)
    psor = price_american_put_pde(
        Market(am["s"], am["r"], am["q"]), am["k"], am["T"], am["sigma"], num_space=200, num_time=200
    )
    diff = abs(psor - crr)
    tol_am = max(0.01, 2.0 * diff)
    assert diff < 0.05, f"American PSOR vs CRR diff {diff:.4f} too large"
    add(
        "american_put_flat_crr5000",
        {**am, "num_space": 200, "num_time": 200},
        {"price": round(crr, 10)},
        round(tol_am, 6),
    )
    print(f"golden american_put_flat_crr5000: CRR={crr:.6f} PSOR={psor:.6f} diff={diff:.5f} tol={tol_am:.4f}")

    # ------------------- 13: flat-vol MC vanilla vs BS (statistical).
    mc_p = dict(s=100.0, k=105.0, r=0.03, q=0.01, sigma=0.2, T=1.0)
    mc_mkt = Market(mc_p["s"], mc_p["r"], mc_p["q"])
    exact = bs_price(mc_p["s"], mc_p["k"], mc_p["r"], mc_p["q"], mc_p["sigma"], mc_p["T"], True)
    res = price_european_mc(mc_mkt, mc_p["k"], mc_p["T"], mc_p["sigma"], n_paths=20000, n_steps=100, seed=SEED)
    assert res.within(exact, 3.0), f"MC {res.price:.4f}±{res.stderr:.4f} vs BS {exact:.4f}"
    tol_mc = round(4.0 * res.stderr, 6)
    add(
        "mc_flat_call_vs_bs",
        {**mc_p, "n_paths": 20000, "n_steps": 100},
        {"price": round(exact, 10)},
        tol_mc,
    )
    print(f"golden mc_flat_call_vs_bs: BS={exact:.4f} MC={res.price:.4f} SE={res.stderr:.4f} tol=4SE={tol_mc}")

    # ------------------- 14: local-vol MC vs local-vol PDE (statistical).
    res_lv = price_european_mc(mkt0, 100.0, 1.0, lv.vol, n_paths=20000, n_steps=100, seed=SEED)
    assert res_lv.within(pde_lv_price, 3.0), (
        f"localvol MC {res_lv.price:.4f}±{res_lv.stderr:.4f} vs PDE {pde_lv_price:.4f}"
    )
    tol_lv = round(4.0 * res_lv.stderr, 6)
    add(
        "mc_localvol_bundled_k100_T1",
        {"s": 100.0, "k": 100.0, "r": 0.0, "q": 0.0, "T": 1.0, "n_paths": 20000, "n_steps": 100},
        {"price": round(pde_lv_price, 10)},
        tol_lv,
    )
    print(f"golden mc_localvol_bundled_k100_T1: PDE={pde_lv_price:.4f} MC={res_lv.price:.4f} tol=4SE={tol_lv}")

    # ------------------- 15: up-and-out barrier, flat vol, analytic reference.
    ba = dict(s=100.0, k=100.0, b=130.0, r=0.02, q=0.0, sigma=0.2, T=1.0)
    exact_uo = up_out_call_analytic(ba["s"], ba["k"], ba["b"], ba["r"], ba["q"], ba["sigma"], ba["T"])
    res_uo = price_up_out_call_mc(
        Market(ba["s"], ba["r"], ba["q"]), ba["k"], ba["b"], ba["T"], ba["sigma"],
        n_paths=100000, n_steps=200, seed=SEED, brownian_bridge=True,
    )
    bias = abs(res_uo.price - exact_uo)
    assert bias < 4.0 * res_uo.stderr + 0.02, (
        f"barrier MC {res_uo.price:.4f}±{res_uo.stderr:.4f} vs analytic {exact_uo:.4f}"
    )
    # Ports run 20k x 200 with the bridge: allow 4 SE at that size + residual bias.
    res_uo_small = price_up_out_call_mc(
        Market(ba["s"], ba["r"], ba["q"]), ba["k"], ba["b"], ba["T"], ba["sigma"],
        n_paths=20000, n_steps=200, seed=SEED, brownian_bridge=True,
    )
    tol_uo = round(4.0 * res_uo_small.stderr + 0.02, 6)
    add(
        "barrier_upout_flat_bb",
        {**ba, "n_paths": 20000, "n_steps": 200},
        {"price": round(exact_uo, 10)},
        tol_uo,
    )
    print(
        f"golden barrier_upout_flat_bb: analytic={exact_uo:.4f} "
        f"MC(100k)={res_uo.price:.4f}±{res_uo.stderr:.4f} tol={tol_uo}"
    )

    golden = {"cases": cases}
    (golden_dir / "golden.json").write_text(json.dumps(golden, indent=2) + "\n")
    print(f"wrote golden/golden.json with {len(cases)} cases")


if __name__ == "__main__":
    main()
