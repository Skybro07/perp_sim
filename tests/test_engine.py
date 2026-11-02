"""Run with `python tests/test_engine.py` (or pytest if installed)."""
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import requests

import data
from data import make_synthetic
from margin import IsolatedPosition, TierTable
from simulate import simulate_isolated, simulate_portfolio


def flat_df(n=20, price=100.0):
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open": price, "high": price, "low": price, "close": price,
                         "funding_rate": 0.0}, index=idx)


def test_tiers_continuous():
    t = TierTable()
    for cap in t.caps[:-1]:
        assert abs(t.maintenance(cap) - t.maintenance(cap + 1e-6)) < 1e-2


def test_liq_price_equity_equals_maintenance():
    t = TierTable()
    for side in (1, -1):
        for lev in (3, 10, 25):
            pos = IsolatedPosition(side, 10_000, 100.0, lev, t)
            p = pos.liq_price()
            assert abs(pos.equity(p) - pos.maintenance_margin(p)) < 1e-6, (side, lev)


def test_known_closed_form_long():
    # flat 0.4% maintenance: P = E*(1-1/L)/(1-mmr)
    pos = IsolatedPosition(1, 10_000, 100.0, 10)
    expected = 100 * (1 - 0.1) / (1 - 0.004)
    assert abs(pos.liq_price() - expected) < 1e-6


def test_higher_leverage_is_closer_to_liquidation():
    d = [IsolatedPosition(1, 10_000, 100.0, l).distance_to_liq() for l in (2, 5, 10, 20)]
    assert d == sorted(d, reverse=True)


def test_no_liquidation_on_flat_prices():
    r = simulate_isolated(flat_df(), [0], 1, 10, hold_bars=10)
    assert not r.liquidated[0] and abs(r.pnl[0]) < 1e-9


def test_wick_liquidation_without_shortfall():
    df = flat_df()
    df.loc[df.index[5], "low"] = 90.0          # wick below the ~90.36 liquidation price
    r = simulate_isolated(df, [0], 1, 10, hold_bars=10, slippage=0.0)
    assert r.liquidated[0] and r.liq_bar[0] == 5 and r.shortfall[0] == 0
    assert abs(r.pnl[0] + 1000) < 1e-9          # lost the whole margin


def test_gap_through_liquidation_creates_shortfall():
    df = flat_df()
    for c in ("open", "high", "low", "close"):
        df.loc[df.index[5]:, c] = 80.0          # gap down 20%
    r = simulate_isolated(df, [0], 1, 10, hold_bars=10, slippage=0.0)
    assert r.liquidated[0]
    assert abs(r.shortfall[0] - 1000) < 1e-6    # margin 1000, position lost 2000 -> 1000 beyond margin


def test_short_mirror():
    df = flat_df()
    df.loc[df.index[5], "high"] = 111.0
    r = simulate_isolated(df, [0], -1, 10, hold_bars=10, slippage=0.0)
    assert r.liquidated[0]


def test_funding_sign():
    df = flat_df()
    df["funding_rate"] = 0.001
    long_ = simulate_isolated(df, [0], 1, 5, hold_bars=10)
    short = simulate_isolated(df, [0], -1, 5, hold_bars=10)
    assert long_.funding_paid[0] > 0 > short.funding_paid[0]   # longs pay when rate > 0


def test_cross_equals_isolated_for_single_leg():
    dfs = make_synthetic(("BTCUSDT",), years=1, seed=3)
    entries = np.arange(0, 2000, 7)
    r = simulate_portfolio(dfs, [("BTCUSDT", 1, 1.0)], entries, 10_000, 20, 240)
    assert np.array_equal(r.iso_any_liq, r.cross_liq)


def test_liquidation_rate_rises_with_leverage():
    df = make_synthetic(("BTCUSDT",), years=2, seed=5)["BTCUSDT"]
    entries = np.arange(0, len(df) - 720, 11)
    rates = [simulate_isolated(df, entries, 1, l, hold_bars=720).liquidated.mean()
             for l in (2, 5, 10, 25, 50)]
    assert rates == sorted(rates) and rates[-1] > rates[0]


def test_cached_data_is_used_without_redownloading():
    with TemporaryDirectory() as cache_dir:
        cached = flat_df(2)
        cached.to_csv(Path(cache_dir) / "BTCUSDT_1h_bybit.csv")
        with patch("data.download_candles", side_effect=AssertionError("unexpected download")):
            result = data.load_or_download(
                "BTCUSDT", cache_dir=cache_dir, start="2024-01-01",
                end="2024-01-01T01:00:00Z",
            )
    assert len(result) == 2
    assert result.index.tz is not None


def test_cached_data_is_incrementally_updated():
    with TemporaryDirectory() as cache_dir:
        cached = flat_df(2)
        cached.to_csv(Path(cache_dir) / "BTCUSDT_1h_bybit.csv")
        new_candles = flat_df(2)
        new_candles.index = pd.date_range("2024-01-01T01:00:00Z", periods=2, freq="1h")
        with patch("data.download_candles", return_value=new_candles) as download:
            with patch("data.download_funding", return_value=pd.DataFrame(
                    columns=["time", "fundingRate"])):
                result = data.load_or_download(
                    "BTCUSDT", cache_dir=cache_dir, start="2024-01-01",
                    end="2024-01-01T02:00:00Z",
                )
        assert Path(cache_dir, "BTCUSDT_1h_bybit.csv").exists()
    assert len(result) == 3
    assert download.call_args.args[2] == "2024-01-01T01:00:00+00:00"


def test_cached_data_is_used_when_bybit_update_fails():
    with TemporaryDirectory() as cache_dir:
        cached = flat_df(2)
        cached.to_csv(Path(cache_dir) / "BTCUSDT_1h_bybit.csv")
        with patch("data.download_candles", side_effect=requests.ConnectionError("offline")):
            with patch("data.warnings.warn") as warn:
                result = data.load_or_download(
                    "BTCUSDT", cache_dir=cache_dir, start="2024-01-01",
                    end="2024-01-01T02:00:00Z",
                )
    assert len(result) == 2
    assert warn.call_args.args[1] is RuntimeWarning


def test_first_download_timeout_has_actionable_error():
    with TemporaryDirectory() as cache_dir:
        with patch("data.download_candles", side_effect=requests.ConnectTimeout("offline")):
            try:
                data.load_or_download("BTCUSDT", cache_dir=cache_dir)
            except data.DataDownloadError as exc:
                assert "no cached data is available" in str(exc)
                assert isinstance(exc.__cause__, requests.ConnectTimeout)
            else:
                raise AssertionError("Expected DataDownloadError when Bybit is unreachable")


def test_bybit_mark_price_candles_use_common_frame():
    timestamp = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000)
    row = [str(timestamp), "100", "101", "99", "100.5"]
    with patch("data._bybit_get", return_value=[row]) as get:
        result = data.download_candles(
            "BTCUSDT", start="2024-01-01", end="2024-01-01T01:00:00Z",
        )
    assert get.call_args.args[0] == "/v5/market/mark-price-kline"
    assert get.call_args.args[1]["category"] == "linear"
    assert get.call_args.args[1]["interval"] == "60"
    assert result.iloc[0]["close"] == 100.5


def test_bybit_funding_history_maps_to_common_funding_frame():
    timestamp = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000)
    row = {"fundingRateTimestamp": str(timestamp), "fundingRate": "0.0001"}
    with patch("data._bybit_get", return_value=[row]) as get:
        result = data.download_funding(
            "BTCUSDT", start="2024-01-01", end="2024-01-01T01:00:00Z",
        )
    assert get.call_args.args[0] == "/v5/market/funding/history"
    assert list(result.columns) == ["time", "fundingRate"]
    assert result.iloc[0]["fundingRate"] == 0.0001


def test_bybit_data_uses_bybit_cache_file():
    with TemporaryDirectory() as cache_dir:
        candles = flat_df(2).drop(columns="funding_rate")
        with patch("data.download_candles", return_value=candles) as download:
            with patch("data.download_funding", return_value=pd.DataFrame(
                    columns=["time", "fundingRate"])):
                data.load_or_download(
                    "BTCUSDT", cache_dir=cache_dir, start="2024-01-01",
                    end="2024-01-01T01:00:00Z",
                )
        assert download.call_args.args[0] == "BTCUSDT"
        assert Path(cache_dir, "BTCUSDT_1h_bybit.csv").exists()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok  ", fn.__name__)
    print(f"\n{len(fns)} tests passed")
