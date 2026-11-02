"""Simulation engine.

Each simulated trade opens at the open of its entry bar and is held for
`hold_bars` bars. All entries are processed at once as (n_entries x hold_bars)
arrays, so thousands of entries run in seconds.

Per-bar logic
-------------
1. Funding settles at the bar open:  margin -= side * qty * open * funding_rate
2. Liquidation check uses the bar's WORST price for the position
   (low for longs, high for shorts), because wicks trigger liquidations:
       margin_t + side*qty*(worst - entry)  <=  maintenance(qty*worst)
3. On liquidation the engine closes at the trigger price, or at the bar open if
   the market gapped through it, then applies adverse `slippage`.
   shortfall = max(0, -equity at fill)  -> the loss the insurance fund absorbs.
   A second, pessimistic figure (shortfall_worst) assumes the close only
   completes at the bar's extreme price. Reality sits between the two.

Simplifications (state these in your README): no trading fees or liquidation
fees, no order-book depth (flat slippage instead), no auto-deleveraging, mark
price == candle price, funding applied on the notional at the bar open.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from margin import TierTable, liquidation_price


@dataclass
class IsoResult:
    entries: np.ndarray
    liquidated: np.ndarray      # bool per entry
    liq_bar: np.ndarray         # bar offset of liquidation, -1 if none
    shortfall: np.ndarray       # USDT lost beyond margin, closing at the trigger price (optimistic)
    shortfall_worst: np.ndarray # same, but closing at the bar's extreme price (pessimistic bound)
    pnl: np.ndarray             # USDT; liquidated positions lose their full margin
    funding_paid: np.ndarray    # USDT; positive = paid, negative = received
    notional: float
    margin: float


def _filter_entries(entries, n_bars: int, hold_bars: int) -> np.ndarray:
    entries = np.asarray(entries, dtype=int)
    entries = entries[(entries >= 0) & (entries + hold_bars <= n_bars)]
    if entries.size == 0:
        raise ValueError("No valid entries: data is shorter than the holding period")
    return entries


def simulate_isolated(df: pd.DataFrame, entries, side: int, leverage: float,
                      notional: float = 10_000.0, hold_bars: int = 720,
                      tiers: TierTable | None = None, slippage: float = 0.003) -> IsoResult:
    tiers = tiers or TierTable()
    entries = _filter_entries(entries, len(df), hold_bars)
    idx = entries[:, None] + np.arange(hold_bars)[None, :]

    op = df["open"].to_numpy()[idx]
    hi = df["high"].to_numpy()[idx]
    lo = df["low"].to_numpy()[idx]
    cl = df["close"].to_numpy()[idx]
    fr = df["funding_rate"].to_numpy()[idx].copy()
    fr[:, 0] = 0.0                                   # funding starts after entry bar

    entry_px = op[:, 0]
    qty = notional / entry_px
    margin0 = notional / leverage
    if margin0 <= tiers.maintenance(notional):
        raise ValueError(f"{leverage}x is too high for this tier table")

    cum_fund = np.cumsum(side * qty[:, None] * op * fr, axis=1)
    margin_t = margin0 - cum_fund

    worst = lo if side > 0 else hi
    equity = margin_t + side * qty[:, None] * (worst - entry_px[:, None])
    maint = tiers.maintenance(qty[:, None] * worst)
    liq = equity <= maint

    liquidated = liq.any(axis=1)
    k = np.where(liquidated, liq.argmax(axis=1), -1)

    shortfall = np.zeros(len(entries))
    shortfall_worst = np.zeros(len(entries))
    pnl = side * qty * (cl[:, -1] - entry_px) - cum_fund[:, -1]      # held to the end
    funding_paid = cum_fund[:, -1].copy()

    rows = np.where(liquidated)[0]
    if rows.size:
        kk = k[rows]
        m = margin_t[rows, kk]
        q, e = qty[rows], entry_px[rows]
        trigger = liquidation_price(side, q, e, m, tiers)
        open_k = op[rows, kk]
        fill = np.minimum(trigger, open_k) if side > 0 else np.maximum(trigger, open_k)
        fill = fill * (1 - slippage) if side > 0 else fill * (1 + slippage)
        equity_fill = m + side * q * (fill - e)
        shortfall[rows] = np.maximum(0.0, -equity_fill)
        # pessimistic bound: engine only manages to close at the bar's extreme
        fill_w = worst[rows, kk] * ((1 - slippage) if side > 0 else (1 + slippage))
        shortfall_worst[rows] = np.maximum(0.0, -(m + side * q * (fill_w - e)))
        pnl[rows] = -margin0                                        # user loses the margin
        funding_paid[rows] = cum_fund[rows, kk]

    return IsoResult(entries, liquidated, k, shortfall, shortfall_worst, pnl,
                     funding_paid, notional, margin0)


# --------------------------------------------------------------------------- #
# Multi-asset portfolio: isolated vs cross margin
# --------------------------------------------------------------------------- #
@dataclass
class PortfolioResult:
    entries: np.ndarray
    capital: float
    iso_any_liq: np.ndarray     # at least one leg liquidated
    iso_all_liq: np.ndarray     # every leg liquidated
    iso_pnl: np.ndarray
    iso_shortfall: np.ndarray
    cross_liq: np.ndarray       # whole account liquidated
    cross_pnl: np.ndarray
    cross_shortfall: np.ndarray


def simulate_portfolio(dfs: dict[str, pd.DataFrame], spec: list[tuple[str, int, float]],
                       entries, capital: float = 10_000.0, leverage: float = 5.0,
                       hold_bars: int = 720, tiers: TierTable | None = None,
                       slippage: float = 0.003) -> PortfolioResult:
    """spec = [(symbol, side, weight), ...] with weights summing to 1.
    Total notional = capital * leverage, split by weight.

    ISOLATED: each leg has its own margin (capital * weight) and is liquidated alone.
    CROSS:    one shared wallet (capital); the whole account is liquidated when
              total equity <= total maintenance margin. Detection uses each leg's
              worst price in the same bar (conservative, since a long's low and a
              short's high rarely coincide). Closing equity is the maintenance
              margin if the account was liquidated mid-bar, or the bar-open equity
              if the market gapped through, less slippage on closing notional.
    All dataframes must share the same index.
    """
    tiers = tiers or TierTable()
    symbols = [s for s, _, _ in spec]
    sides = np.array([sd for _, sd, _ in spec], dtype=float)
    weights = np.array([w for _, _, w in spec], dtype=float)
    ref_index = dfs[symbols[0]].index
    for s in symbols:
        if not dfs[s].index.equals(ref_index):
            raise ValueError("All symbols must share the same index")

    entries = _filter_entries(entries, len(ref_index), hold_bars)

    # ---- isolated: one independent simulation per leg -------------------- #
    legs = [simulate_isolated(dfs[s], entries, int(sd), leverage, capital * leverage * w,
                              hold_bars, tiers, slippage) for s, sd, w in spec]
    iso_any = np.any([r.liquidated for r in legs], axis=0)
    iso_all = np.all([r.liquidated for r in legs], axis=0)
    iso_pnl = np.sum([r.pnl for r in legs], axis=0)
    iso_short = np.sum([r.shortfall for r in legs], axis=0)

    # ---- cross: shared wallet -------------------------------------------- #
    idx = entries[:, None] + np.arange(hold_bars)[None, :]

    def stack(col):
        return np.stack([dfs[s][col].to_numpy()[idx] for s in symbols], axis=2)

    op, hi, lo, cl = stack("open"), stack("high"), stack("low"), stack("close")
    fr = stack("funding_rate").copy()
    fr[:, 0, :] = 0.0
    entry_px = op[:, 0, :]                                   # (m, a)
    qty = capital * leverage * weights / entry_px            # (m, a)

    fund = (sides * qty[:, None, :] * op * fr).sum(axis=2)   # (m, h) per-bar total
    cum_fund = np.cumsum(fund, axis=1)
    wallet_t = capital - cum_fund

    def equity_at(px):
        return wallet_t + (sides * qty[:, None, :] * (px - entry_px[:, None, :])).sum(axis=2)

    def maint_at(px):
        return tiers.maintenance(qty[:, None, :] * px).sum(axis=2)

    worst = np.where(sides > 0, lo, hi)
    liq = equity_at(worst) <= maint_at(worst)
    cross_liq = liq.any(axis=1)
    k = np.where(cross_liq, liq.argmax(axis=1), 0)
    r = np.arange(len(entries))

    eq_open = equity_at(op)[r, k]
    gap = eq_open <= maint_at(op)[r, k]
    closing_equity = np.where(gap, eq_open, maint_at(worst)[r, k])
    closing_notional = (qty * worst[r, k, :]).sum(axis=1)
    closing_equity = closing_equity - slippage * closing_notional
    cross_short = np.where(cross_liq, np.maximum(0.0, -closing_equity), 0.0)

    held_pnl = (sides * qty * (cl[:, -1, :] - entry_px)).sum(axis=1) - cum_fund[:, -1]
    cross_pnl = np.where(cross_liq, -capital, held_pnl)

    return PortfolioResult(entries, capital, iso_any, iso_all, iso_pnl, iso_short,
                           cross_liq, cross_pnl, cross_short)
