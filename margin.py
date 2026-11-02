"""Margin and liquidation maths for USDT-margined perpetual futures.

Conventions
-----------
side      : +1 = long, -1 = short
qty       : position size in base units (e.g. BTC)
entry     : entry price
margin    : margin posted to the position (isolated) or wallet balance (cross)

Maintenance margin is tiered: bigger notional -> higher maintenance rate.
    maintenance(N) = N * rate_tier - cum_tier
where cum_tier makes the function continuous across tier boundaries (the
"maintenance amount" in exchange tier tables).

The DEFAULT_TIERS below are ILLUSTRATIVE. Replace them with the current table
from your exchange before quoting results.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# (notional upper bound in USDT, maintenance margin rate)
DEFAULT_TIERS = [
    (50_000, 0.004),
    (250_000, 0.005),
    (3_000_000, 0.01),
    (20_000_000, 0.025),
    (50_000_000, 0.05),
    (float("inf"), 0.10),
]


class TierTable:
    def __init__(self, tiers=DEFAULT_TIERS):
        self.caps = np.array([t[0] for t in tiers], dtype=float)
        self.rates = np.array([t[1] for t in tiers], dtype=float)
        cum = [0.0]
        for i in range(1, len(tiers)):
            cum.append(cum[-1] + self.caps[i - 1] * (self.rates[i] - self.rates[i - 1]))
        self.cum = np.array(cum)

    def index(self, notional):
        """Tier index for a notional (works on scalars or arrays)."""
        return np.searchsorted(self.caps, np.abs(notional), side="left")

    def maintenance(self, notional):
        i = self.index(notional)
        return np.abs(notional) * self.rates[i] - self.cum[i]


def liquidation_price(side, qty, entry, margin, tiers: TierTable | None = None, iters: int = 4):
    """Price at which equity == maintenance margin (isolated position).

    Solves  margin + side*qty*(P - entry) = maintenance(qty*P)  with the tier
    chosen at the liquidation price itself (fixed-point iteration, converges in
    1-2 steps in practice). Works on numpy arrays for qty/entry/margin.
    Returns 0 for a long that can never be liquidated.
    """
    tiers = tiers or TierTable()
    qty = np.asarray(qty, dtype=float)
    entry = np.asarray(entry, dtype=float)
    margin = np.asarray(margin, dtype=float)
    p = entry.copy() if entry.ndim else np.array(float(entry))
    for _ in range(iters):
        i = tiers.index(qty * p)
        r, c = tiers.rates[i], tiers.cum[i]
        p_long = (qty * entry - margin - c) / (qty * (1.0 - r))
        p_short = (qty * entry + margin + c) / (qty * (1.0 + r))
        p = p_long if side > 0 else p_short
        p = np.maximum(p, 1e-12)
    return np.where(p <= 1e-9, 0.0, p) if side > 0 else p


def bankruptcy_price(side, entry, leverage):
    """Price at which the position's margin is fully gone (equity == 0)."""
    return entry * (1 - side / leverage)


@dataclass
class IsolatedPosition:
    side: int
    notional: float
    entry: float
    leverage: float
    tiers: TierTable = None

    def __post_init__(self):
        self.tiers = self.tiers or TierTable()
        self.qty = self.notional / self.entry
        self.margin = self.notional / self.leverage
        if self.margin <= self.tiers.maintenance(self.notional):
            raise ValueError("Leverage too high: initial margin <= maintenance margin")

    def liq_price(self, margin: float | None = None) -> float:
        m = self.margin if margin is None else margin
        return float(liquidation_price(self.side, self.qty, self.entry, m, self.tiers))

    def distance_to_liq(self) -> float:
        """Adverse move (fraction of entry) that triggers liquidation."""
        return abs(self.liq_price() - self.entry) / self.entry

    def equity(self, price: float, funding_paid: float = 0.0) -> float:
        return self.margin - funding_paid + self.side * self.qty * (price - self.entry)

    def maintenance_margin(self, price: float) -> float:
        return float(self.tiers.maintenance(self.qty * price))


def cross_equity_and_mm(wallet, sides, qtys, entries, marks, tiers: TierTable | None = None):
    """Cross-margin account: equity = wallet + sum(unrealised PnL);
    maintenance = sum of per-position maintenance margins.
    Liquidation when equity <= maintenance."""
    tiers = tiers or TierTable()
    sides, qtys = np.asarray(sides, float), np.asarray(qtys, float)
    entries, marks = np.asarray(entries, float), np.asarray(marks, float)
    equity = wallet + np.sum(sides * qtys * (marks - entries))
    mm = np.sum(tiers.maintenance(qtys * marks))
    return float(equity), float(mm)
