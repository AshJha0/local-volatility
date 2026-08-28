"""localvol demo: Dupire round-trip consistency check.

Pipeline: bundled SSVI implied surface -> Dupire local vol -> reprice
European vanillas by PDE (and spot-check by Monte Carlo) -> invert back to
implied vols -> report the error against the input surface in basis points.

Run:  cd python && PYTHONPATH=src python3 demo.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from localvol import (  # noqa: E402
    DupireLocalVol,
    ImpliedVolSurface,
    Market,
    implied_vol,
    price_american_put_pde,
    price_european_mc,
    price_european_pde,
    price_up_out_call_mc,
)

DATA = Path(__file__).resolve().parents[1] / "data"
INTERIOR_TARGET_BP = 30.0


def main() -> int:
    print("=" * 72)
    print("localvol demo — Dupire local volatility round-trip consistency")
    print("=" * 72)

    surface = ImpliedVolSurface.from_csv(DATA / "implied_surface.csv")
    lv = DupireLocalVol(surface)
    mkt = Market(spot=100.0, rate=0.02, dividend=0.01)  # equity-style carry
    print(
        f"surface: {surface.expiries.size} expiries x {surface.k_nodes.size} strikes, "
        f"calendar violations: {surface.calendar_violations}"
    )
    print(f"market : S0={mkt.spot:.2f}  r={mkt.rate:.2%}  q={mkt.dividend:.2%}\n")

    # ---- round trip: implied in -> local vol -> PDE price -> implied out
    expiries = [0.5, 1.0, 1.5, 2.0]
    strikes = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0]
    print("Round-trip implied-vol error (PDE 200x200), basis points:")
    print("  T\\K   " + "".join(f"{k:>8.0f}" for k in strikes))
    max_err_bp = 0.0
    max_at = (0.0, 0.0)
    for t in expiries:
        lv.reset_counters()
        row = [f"  {t:<5.2f}"]
        sigma_ref = float(surface.implied_vol(0.0, t))
        for strike in strikes:
            k = math.log(strike / mkt.forward(t))
            iv_in = float(surface.implied_vol(k, t))
            price = price_european_pde(
                mkt, strike, t, lv.vol, num_space=200, num_time=200, sigma_ref=sigma_ref
            )
            iv_out = implied_vol(price, mkt.spot, strike, mkt.rate, mkt.dividend, t, True)
            err_bp = abs(iv_out - iv_in) * 1e4
            row.append(f"{err_bp:>8.1f}")
            interior = abs(k) <= 0.30  # wings excluded per spec
            if interior and err_bp > max_err_bp:
                max_err_bp, max_at = err_bp, (strike, t)
        print("".join(row))
    print(f"\nmax interior |vol error|: {max_err_bp:.2f} bp at K={max_at[0]:.0f}, "
          f"T={max_at[1]:.2f}  (target < {INTERIOR_TARGET_BP:.0f} bp, |k| <= 0.30)")
    status = "PASS" if max_err_bp < INTERIOR_TARGET_BP else "FAIL"
    print(f"round-trip check: {status}")
    print(lv.violation_report)

    # ---- PDE vs MC cross-check at the ATM pillar
    t, strike = 1.0, 100.0
    pde = price_european_pde(mkt, strike, t, lv.vol, num_space=200, num_time=200,
                             sigma_ref=float(surface.implied_vol(0.0, t)))
    mc = price_european_mc(mkt, strike, t, lv.vol, n_paths=20000, n_steps=100, seed=42)
    print(f"\nPDE vs MC (K=100, T=1): PDE={pde:.4f}  MC={mc.price:.4f} "
          f"+/- {mc.stderr:.4f}  |diff|={abs(pde - mc.price) / mc.stderr:.2f} SE "
          f"({'OK' if mc.within(pde, 3.0) else 'OUTSIDE 3 SE'})")

    # ---- American and barrier flavours under the same local vol
    amer = price_american_put_pde(mkt, 100.0, 1.0, lv.vol, num_space=200, num_time=200,
                                  sigma_ref=float(surface.implied_vol(0.0, 1.0)))
    eur_put = price_european_pde(mkt, 100.0, 1.0, lv.vol, is_call=False,
                                 num_space=200, num_time=200,
                                 sigma_ref=float(surface.implied_vol(0.0, 1.0)))
    uo = price_up_out_call_mc(mkt, 100.0, 130.0, 1.0, lv.vol,
                              n_paths=20000, n_steps=100, seed=42, brownian_bridge=True)
    print(f"American put (PSOR)     : {amer:.4f}  (European {eur_put:.4f}, "
          f"premium {amer - eur_put:+.4f})")
    print(f"Up-and-out call B=130 MC: {uo.price:.4f} +/- {uo.stderr:.4f} "
          f"(vanilla {pde:.4f}; barrier <= vanilla: {'OK' if uo.price <= pde else 'VIOLATED'})")

    print("\ndone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
