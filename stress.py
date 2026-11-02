"""Experiments that turn the simulation engine into resume-ready numbers.

1. liquidation_vs_leverage : % liquidated and shortfall by leverage and side
2. funding_drag            : what funding costs a long/short over a holding period
3. replay_windows          : replay named crash periods at several leverages
4. isolated_vs_cross       : same portfolio, two margin modes

Caveat: random entry times over a finite history give overlapping windows, so the
samples are correlated. Treat percentages as descriptive of this history, not as
precise probabilities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from margin import TierTable
from simulate import simulate_isolated, simulate_portfolio

CRASH_WINDOWS = {
    "Mar-2020 COVID crash": ("2020-03-08", "2020-03-20"),
    "May-2021 China ban crash": ("2021-05-10", "2021-05-24"),
    "Nov-2022 FTX collapse": ("2022-11-06", "2022-11-16"),
}


def bars_per_day(df: pd.DataFrame) -> int:
    step = df.index.to_series().diff().median()
    return int(round(pd.Timedelta(days=1) / step))


def sample_entries(n_bars: int, hold_bars: int, n_entries: int, seed: int = 42) -> np.ndarray:
    pool = np.arange(0, max(1, n_bars - hold_bars))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(pool, size=min(n_entries, pool.size), replace=False))


def liquidation_vs_leverage(df: pd.DataFrame, leverages=(2, 5, 10, 20, 50), sides=(1, -1),
                            hold_days: int = 30, n_entries: int = 3000, slippage: float = 0.003,
                            notional: float = 10_000.0, tiers: TierTable | None = None,
                            seed: int = 42) -> pd.DataFrame:
    hold = hold_days * bars_per_day(df)
    entries = sample_entries(len(df), hold, n_entries, seed)
    rows = []
    for side in sides:
        for lev in leverages:
            res = simulate_isolated(df, entries, side, lev, notional, hold, tiers, slippage)
            sf = res.shortfall / notional * 100
            sfw = res.shortfall_worst / notional * 100
            liq = res.liquidated
            rows.append({
                "side": "long" if side > 0 else "short",
                "leverage": lev,
                "hold_days": hold_days,
                "n_entries": len(res.entries),
                "pct_liquidated": liq.mean() * 100,
                "pct_liqs_with_shortfall": (sf[liq] > 0).mean() * 100 if liq.any() else 0.0,
                "avg_shortfall_pct_notional": sf[liq].mean() if liq.any() else 0.0,
                "avg_shortfall_worstfill_pct": sfw[liq].mean() if liq.any() else 0.0,
                "pct_liqs_shortfall_worstfill": (sfw[liq] > 0).mean() * 100 if liq.any() else 0.0,
                "worst_shortfall_worstfill_pct": sfw.max(),
                "avg_funding_pct_notional": res.funding_paid.mean() / notional * 100,
                "mean_pnl_pct_margin": res.pnl.mean() / res.margin * 100,
            })
    return pd.DataFrame(rows).round(3)


def shortfall_vs_slippage(df: pd.DataFrame, leverage: float = 20, side: int = 1,
                          slippages=(0.001, 0.003, 0.005, 0.01), hold_days: int = 30,
                          n_entries: int = 3000, notional: float = 10_000.0,
                          tiers: TierTable | None = None, seed: int = 42) -> pd.DataFrame:
    """How the insurance-fund shortfall grows as liquidation execution gets worse."""
    hold = hold_days * bars_per_day(df)
    entries = sample_entries(len(df), hold, n_entries, seed)
    rows = []
    for sl in slippages:
        res = simulate_isolated(df, entries, side, leverage, notional, hold, tiers, sl)
        liq = res.liquidated
        sf, sfw = res.shortfall / notional * 100, res.shortfall_worst / notional * 100
        rows.append({
            "slippage_pct": sl * 100, "leverage": leverage,
            "pct_liquidated": liq.mean() * 100,
            "avg_shortfall_trigger_fill_pct": sf[liq].mean() if liq.any() else 0.0,
            "avg_shortfall_worst_fill_pct": sfw[liq].mean() if liq.any() else 0.0,
            "pct_liqs_with_shortfall_trigger": (sf[liq] > 0).mean() * 100 if liq.any() else 0.0,
        })
    return pd.DataFrame(rows).round(3)


def funding_drag(df: pd.DataFrame, hold_days_list=(7, 30, 90)) -> pd.DataFrame:
    """Cumulative funding over rolling windows as % of (constant) notional.
    Positive = longs pay shorts."""
    bpd = bars_per_day(df)
    fr = df["funding_rate"]
    rows = []
    for d in hold_days_list:
        s = fr.rolling(d * bpd).sum().dropna() * 100
        rows.append({
            "hold_days": d,
            "mean_long_cost_pct": s.mean(),
            "median_long_cost_pct": s.median(),
            "pct_windows_longs_pay": (s > 0).mean() * 100,
            "p95_long_cost_pct": s.quantile(0.95),
            "p05_long_cost_pct": s.quantile(0.05),
        })
    return pd.DataFrame(rows).round(3)


def replay_windows(df: pd.DataFrame, windows: dict | None = None, leverages=(5, 10, 20, 50),
                   side: int = 1, slippage: float = 0.003, notional: float = 10_000.0,
                   tiers: TierTable | None = None) -> pd.DataFrame:
    """Open at the start of each window, hold to its end."""
    windows = windows or CRASH_WINDOWS
    rows = []
    for name, (start, end) in windows.items():
        i0 = df.index.searchsorted(pd.Timestamp(start, tz="UTC"))
        i1 = df.index.searchsorted(pd.Timestamp(end, tz="UTC"))
        if i0 >= len(df) or i1 <= i0 + 1 or i1 > len(df) or df.index[0] > pd.Timestamp(start, tz="UTC"):
            continue                                   # window not covered by the data
        seg = df.iloc[i0:i1]
        entry = seg["open"].iloc[0]
        adverse = (seg["low"].min() / entry - 1) if side > 0 else (entry / seg["high"].max() - 1) * -1
        for lev in leverages:
            res = simulate_isolated(df, [i0], side, lev, notional, i1 - i0, tiers, slippage)
            rows.append({
                "window": name, "side": "long" if side > 0 else "short", "leverage": lev,
                "max_adverse_move_pct": round(abs(adverse) * 100, 2) if adverse < 0 else 0.0,
                "liquidated": bool(res.liquidated[0]),
                "shortfall_pct_notional": round(res.shortfall[0] / notional * 100, 3),
            })
    return pd.DataFrame(rows)


def isolated_vs_cross(dfs: dict[str, pd.DataFrame], spec, leverage: float = 5.0,
                      capital: float = 10_000.0, hold_days: int = 30, n_entries: int = 3000,
                      slippage: float = 0.003, tiers: TierTable | None = None,
                      seed: int = 42) -> pd.DataFrame:
    ref = next(iter(dfs.values()))
    hold = hold_days * bars_per_day(ref)
    entries = sample_entries(len(ref), hold, n_entries, seed)
    r = simulate_portfolio(dfs, spec, entries, capital, leverage, hold, tiers, slippage)

    def row(mode, any_liq, all_liq, pnl, short):
        return {
            "margin_mode": mode, "leverage": leverage,
            "pct_any_leg_liquidated": any_liq.mean() * 100,
            "pct_account_wiped": all_liq.mean() * 100,
            "mean_pnl_pct_capital": pnl.mean() / capital * 100,
            "p05_pnl_pct_capital": np.percentile(pnl, 5) / capital * 100,
            "avg_shortfall_pct_capital": short.mean() / capital * 100,
            "worst_shortfall_pct_capital": short.max() / capital * 100,
        }

    out = [row("isolated", r.iso_any_liq, r.iso_all_liq, r.iso_pnl, r.iso_shortfall),
           row("cross", r.cross_liq, r.cross_liq, r.cross_pnl, r.cross_shortfall)]
    return pd.DataFrame(out).round(3)
