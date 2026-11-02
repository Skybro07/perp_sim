"""Run all experiments.

    python main.py --synthetic                 # offline demo on generated data
    python main.py --download                  # Bybit public data, cached in data_cache/
    python main.py --download --start 2020-01-01 --hold-days 30

Outputs: printed tables, CSVs and a chart in results/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data import DataDownloadError, load_or_download, make_synthetic
from stress import (funding_drag, isolated_vs_cross, liquidation_vs_leverage,
                    replay_windows, shortfall_vs_slippage)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", action="store_true", help="use generated data")
    src.add_argument("--download", action="store_true", help="use Bybit public data")
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--hold-days", type=int, default=30)
    ap.add_argument("--entries", type=int, default=3000)
    ap.add_argument("--slippage", type=float, default=0.003)
    args = ap.parse_args()

    if args.synthetic:
        dfs = make_synthetic(tuple(args.symbols))
    else:
        try:
            dfs = {s: load_or_download(s, start=args.start) for s in args.symbols}
        except DataDownloadError as exc:
            ap.error(str(exc))
        common = dfs[args.symbols[0]].index
        for s in args.symbols[1:]:
            common = common.intersection(dfs[s].index)
        dfs = {s: d.loc[common] for s, d in dfs.items()}   # align for the portfolio test

    out = Path("results")
    out.mkdir(exist_ok=True)
    base = args.symbols[0]
    pd.set_option("display.width", 200, "display.max_columns", 30)

    # 1. liquidation rate vs leverage
    table = liquidation_vs_leverage(dfs[base], hold_days=args.hold_days,
                                    n_entries=args.entries, slippage=args.slippage)
    table.to_csv(out / "liquidation_vs_leverage.csv", index=False)
    print(f"\n[1] Liquidation vs leverage ({base}, {args.hold_days}-day holds)\n", table)

    fig, ax = plt.subplots(figsize=(6, 4))
    for side, g in table.groupby("side"):
        ax.plot(g["leverage"], g["pct_liquidated"], marker="o", label=side)
    ax.set_xlabel("Leverage (x)")
    ax.set_ylabel("% of positions liquidated")
    ax.set_title(f"{base}: liquidation rate vs leverage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "liquidation_vs_leverage.png", dpi=150)

    # 2. funding drag
    fd = funding_drag(dfs[base])
    fd.to_csv(out / "funding_drag.csv", index=False)
    print(f"\n[2] Funding drag for longs ({base}), % of notional\n", fd)

    # 3. crash replay (only windows covered by the data)
    rp = replay_windows(dfs[base])
    rp.to_csv(out / "crash_replay.csv", index=False)
    print("\n[3] Crash replay (longs)\n", rp if len(rp) else "No crash window covered by this data.")

    # 3b. insurance-fund shortfall vs execution quality
    sv = shortfall_vs_slippage(dfs[base], hold_days=args.hold_days, n_entries=args.entries)
    sv.to_csv(out / "shortfall_vs_slippage.csv", index=False)
    print(f"\n[3b] Shortfall vs liquidation slippage ({base}, 20x long)\n", sv)

    # 4. isolated vs cross on a long/short portfolio
    if len(args.symbols) >= 2:
        spec = [(args.symbols[0], 1, 0.5), (args.symbols[1], -1, 0.5)]
        ic = isolated_vs_cross(dfs, spec, leverage=10, hold_days=args.hold_days,
                               n_entries=args.entries, slippage=args.slippage)
        ic.to_csv(out / "isolated_vs_cross.csv", index=False)
        print(f"\n[4] Isolated vs cross: long {args.symbols[0]} / short {args.symbols[1]}, 10x\n", ic)

    print(f"\nSaved tables and chart to {out.resolve()}")


if __name__ == "__main__":
    main()
