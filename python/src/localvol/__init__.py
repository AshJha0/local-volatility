"""localvol — Dupire local volatility: surface, PDE and Monte Carlo pricers.

Pipeline: implied-vol surface (total variance, spline-in-k / linear-in-T)
-> Dupire local vol (Gatheral total-variance formula, finite differences)
-> pricing via Crank-Nicolson/Rannacher PDE or log-Euler Monte Carlo,
with a round-trip consistency check recovering the input implied vols.
"""

from .black_scholes import (
    bs_delta,
    bs_gamma,
    bs_price,
    bs_vega,
    implied_vol,
    norm_cdf,
    norm_pdf,
)
from .dupire import CAP, DK, DT, FLOOR, DupireLocalVol
from .market import Market
from .mc import MCResult, price_european_mc, price_up_out_call_mc
from .pde import PDEResult, price_american_put_pde, price_european_pde
from .surface import CubicSpline1D, ImpliedVolSurface
from .tridiag import thomas_solve

__all__ = [
    "Market",
    "norm_cdf",
    "norm_pdf",
    "bs_price",
    "bs_delta",
    "bs_gamma",
    "bs_vega",
    "implied_vol",
    "thomas_solve",
    "CubicSpline1D",
    "ImpliedVolSurface",
    "DupireLocalVol",
    "DK",
    "DT",
    "FLOOR",
    "CAP",
    "PDEResult",
    "price_european_pde",
    "price_american_put_pde",
    "MCResult",
    "price_european_mc",
    "price_up_out_call_mc",
]

__version__ = "1.0.0"
