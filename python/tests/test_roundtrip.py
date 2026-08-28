"""Round-trip consistency: implied surface -> Dupire -> PDE -> implied vols.

The full sweep lives in demo.py; here a representative interior subset is
asserted to stay under the 30bp target (wings excluded per spec).
"""

from __future__ import annotations

import math

import pytest

from localvol import Market, implied_vol, price_european_pde


@pytest.mark.parametrize("strike", [90.0, 100.0, 110.0])
@pytest.mark.parametrize("expiry", [0.5, 1.0])
def test_interior_round_trip_under_30bp(bundled_surface, bundled_localvol, strike, expiry):
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
