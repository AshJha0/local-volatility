"""Market description shared by all pricers.

A single dataclass covers both asset classes:

* **Equity**: ``rate`` is the risk-free rate ``r`` and ``dividend`` is the
  continuous dividend yield ``q``.
* **FX (Garman-Kohlhagen)**: ``rate`` is the domestic rate ``rd`` and
  ``dividend`` plays the role of the foreign rate ``rf``.  Under
  Garman-Kohlhagen the foreign rate enters every formula exactly where the
  dividend yield enters the equity formula, so one parametrisation serves
  both; use :meth:`Market.fx` to make the intent explicit at call sites.

The forward is ``F(T) = S * exp((rate - dividend) * T)`` in both cases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Market:
    """Spot / rates container.

    Attributes:
        spot: Spot price ``S0`` (equity price or FX rate, domestic per foreign).
            Must be strictly positive.
        rate: Continuously-compounded discount rate ``r`` (domestic rate ``rd``
            for FX).  May be negative (EUR/CHF style regimes are supported).
        dividend: Continuous dividend yield ``q`` (foreign rate ``rf`` for FX).
            May be negative.
    """

    spot: float
    rate: float
    dividend: float = 0.0

    def __post_init__(self) -> None:
        for name in ("spot", "rate", "dividend"):
            v = getattr(self, name)
            if not math.isfinite(v):
                raise ValueError(f"Market.{name} must be finite, got {v!r}")
        if self.spot <= 0.0:
            raise ValueError(f"Market.spot must be > 0, got {self.spot!r}")

    @classmethod
    def fx(cls, spot: float, rd: float, rf: float) -> "Market":
        """Garman-Kohlhagen market: domestic rate ``rd``, foreign rate ``rf``."""
        return cls(spot=spot, rate=rd, dividend=rf)

    def forward(self, expiry: float) -> float:
        """Forward ``F(T) = S0 * exp((r - q) T)``.  Requires ``expiry >= 0``."""
        if not math.isfinite(expiry) or expiry < 0.0:
            raise ValueError(f"expiry must be finite and >= 0, got {expiry!r}")
        return self.spot * math.exp((self.rate - self.dividend) * expiry)

    def log_forward(self, expiry: float) -> float:
        """``ln F(T)`` — convenient for log-moneyness lookups."""
        return math.log(self.spot) + (self.rate - self.dividend) * expiry
