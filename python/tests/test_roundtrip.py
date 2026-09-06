"""Round-trip consistency: implied surface -> Dupire -> PDE -> implied vols.

The full sweep lives in demo.py; here a representative interior subset is
asserted to stay under the 30 bp target (wings excluded per spec) and, more
tightly, under 5 bp — the wing-clamped Dupire stencil (CRIT-1) brought the
demo's worst interior cell from 16.4 bp to 2.7 bp.
"""

from __future__ import annotations

import math

import pytest

from localvol import Market, implied_vol, price_european_pde


@pytest.mark.parametrize("strike", [90.0, 100.0, 110.0])
@pytest.mark.parametrize("expiry", [0.5, 1.0])
def test_interior_round_trip_under_5bp(bundled_surface, bundled_localvol, strike, expiry):
    mkt = Market(100.0, 0.02, 0.01)  # equity-style carry exercises the forward logic
    k = math.log(strike / mkt.forward(expiry))
    iv_in = float(bundled_surface.implied_vol(k, expiry))
    price = price_european_pde(
        mkt, strike, expiry, bundled_localvol.vol,
        num_space=200, num_time=200,
        sigma_ref=float(bundled_surface.implied_vol(0.0, expiry)),
    )
    iv_out = implied_vol(price, mkt.spot, strike, mkt.rate, mkt.dividend, expiry, True)
    err_bp = abs(iv_out - iv_in) * 1e4
    assert err_bp < 30.0, f"K={strike} T={expiry}: {err_bp:.1f}bp"
    assert err_bp < 5.0, f"K={strike} T={expiry}: {err_bp:.2f}bp (post-CRIT-1 budget)"


def test_full_demo_grid_round_trip_under_5bp(bundled_surface, bundled_localvol):
    """The demo's 4 x 7 sweep, asserted: max interior error < 5 bp and no
    Dupire clamp fires anywhere on the PDE grids (r - q = 1%)."""
    mkt = Market(100.0, 0.02, 0.01)
    bundled_localvol.reset_counters()
    worst = 0.0
    for expiry in (0.5, 1.0, 1.5, 2.0):
        sigma_ref = float(bundled_surface.implied_vol(0.0, expiry))
        for strike in (80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0):
            k = math.log(strike / mkt.forward(expiry))
            iv_in = float(bundled_surface.implied_vol(k, expiry))
            price = price_european_pde(
                mkt, strike, expiry, bundled_localvol.vol, num_space=200, num_time=200, sigma_ref=sigma_ref
            )
            iv_out = implied_vol(price, mkt.spot, strike, mkt.rate, mkt.dividend, expiry, True)
            if abs(k) <= 0.30:
                worst = max(worst, abs(iv_out - iv_in) * 1e4)
    assert worst < 5.0, f"max interior error {worst:.2f} bp"
    assert worst == pytest.approx(2.73, abs=0.05)  # K=80, T=2 (documented value)
    assert bundled_localvol.floor_count == 0 and bundled_localvol.cap_count == 0
