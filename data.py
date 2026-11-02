"""Data layer: download Bybit USDT-M futures data, cache it, or generate
synthetic data for testing.

Every price frame has a UTC DatetimeIndex (bar open time) and columns
open, high, low, close, funding_rate. `funding_rate` is the funding rate
settled at that bar's open (0 on bars with no funding event), so funding cost
per bar = side * qty * price * funding_rate.

If Bybit is unreachable, load compatible data with `load_csv_frame`, or use
the synthetic generator.
"""
from __future__ import annotations

import time
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import requests

BYBIT_BASE_URL = "https://api.bybit.com"


class DataDownloadError(RuntimeError):
    """Raised when remote data is unavailable and no local cache can be used."""


class DataProviderError(RuntimeError):
    """Raised when a data provider returns an API-level error."""


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def _bybit_get(path: str, params: dict, retries: int = 3):
    for attempt in range(retries):
        try:
            response = requests.get(BYBIT_BASE_URL + path, params=params, timeout=20)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or payload.get("retCode") != 0:
                message = payload.get("retMsg", "unexpected response") if isinstance(payload, dict) else (
                    "unexpected response"
                )
                raise DataProviderError(f"Bybit API error for {path}: {message}")
            result = payload.get("result")
            if not isinstance(result, dict) or not isinstance(result.get("list"), list):
                raise DataProviderError(f"Unexpected Bybit response for {path}")
            return result["list"]
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))


def _bybit_interval(interval: str) -> str:
    aliases = {
        "1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30",
        "1h": "60", "2h": "120", "4h": "240", "6h": "360", "12h": "720",
        "1d": "D", "1w": "W", "1M": "M",
    }
    value = aliases.get(interval, interval)
    if value not in {"1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M"}:
        raise ValueError(f"Unsupported Bybit kline interval: {interval}")
    return value


def _to_ms(ts) -> int:
    timestamp = pd.Timestamp(ts)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return int(timestamp.timestamp() * 1000)


def download_candles(symbol: str, interval: str = "1h", start: str = "2020-01-01",
                     end: str | None = None) -> pd.DataFrame:
    """Download Bybit mark-price klines, used for liquidation calculations."""
    path = "/v5/market/mark-price-kline"
    interval_param = _bybit_interval(interval)
    params = {"category": "linear", "symbol": symbol, "interval": interval_param, "limit": 1000}
    start_ms = _to_ms(start)
    end_ms = _to_ms(end) if end else int(time.time() * 1000)
    rows = []
    while start_ms < end_ms:
        batch = _bybit_get(path, {**params, "start": start_ms, "end": end_ms})
        if not batch:
            break
        rows.extend(batch)
        oldest_ms = min(int(row[0]) for row in batch)
        if oldest_ms >= end_ms:
            raise DataProviderError(f"Bybit returned non-advancing candle data for {symbol}")
        end_ms = oldest_ms - 1
        time.sleep(0.15)
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close"],
                            index=pd.DatetimeIndex([], tz="UTC", name="open_time"))
    df = pd.DataFrame(rows)
    if df.shape[1] < 5:
        raise DataProviderError(f"Unexpected Bybit candle response for {symbol}")
    df = df.iloc[:, :5]
    df.columns = ["open_time", "open", "high", "low", "close"]
    df["open_time"] = pd.to_datetime(
        pd.to_numeric(df["open_time"], errors="raise"), unit="ms", utc=True
    )
    df = df.set_index("open_time").astype(float)
    return df[~df.index.duplicated()].sort_index()


def download_funding(symbol: str, start: str = "2020-01-01",
                     end: str | None = None) -> pd.DataFrame:
    """Download historical Bybit funding rates."""
    path = "/v5/market/funding/history"
    params = {"category": "linear", "symbol": symbol, "limit": 200}
    start_ms = _to_ms(start)
    end_ms = _to_ms(end) if end else int(time.time() * 1000)
    rows = []
    while start_ms < end_ms:
        batch = _bybit_get(path, {**params, "startTime": start_ms, "endTime": end_ms})
        if not batch:
            break
        rows.extend(batch)
        time_key = "fundingRateTimestamp"
        oldest_ms = min(int(item[time_key]) for item in batch)
        if oldest_ms >= end_ms:
            raise DataProviderError(f"Bybit returned non-advancing funding data for {symbol}")
        end_ms = oldest_ms - 1
        time.sleep(0.15)
    if not rows:
        return pd.DataFrame(columns=["time", "fundingRate"])
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(
        pd.to_numeric(df["fundingRateTimestamp"], errors="raise"), unit="ms", utc=True
    )
    df["fundingRate"] = df["fundingRate"].astype(float)
    return df[["time", "fundingRate"]]


def merge_funding(candles: pd.DataFrame, funding: pd.DataFrame, freq: str = "1h") -> pd.DataFrame:
    """Attach each funding event to the candle whose open it falls in."""
    out = candles.copy()
    if funding.empty:
        out["funding_rate"] = 0.0
        return out
    f = funding.copy()
    f["bar"] = f["time"].dt.floor(freq)
    per_bar = f.groupby("bar")["fundingRate"].sum()
    out["funding_rate"] = per_bar.reindex(out.index).fillna(0.0)
    return out


def load_or_download(symbol: str, cache_dir: str = "data_cache", interval: str = "1h",
                     start: str = "2020-01-01", end: str | None = None) -> pd.DataFrame:
    cache = Path(cache_dir)
    if not cache.is_absolute():
        cache = Path(__file__).resolve().parent / cache
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{symbol}_{interval}_bybit.csv"
    requested_start = pd.Timestamp(start)
    requested_start = (requested_start.tz_localize("UTC") if requested_start.tzinfo is None
                       else requested_start.tz_convert("UTC"))
    requested_end = pd.Timestamp(end) if end else pd.Timestamp.now(tz="UTC")
    requested_end = (requested_end.tz_localize("UTC") if requested_end.tzinfo is None
                     else requested_end.tz_convert("UTC"))

    cached = load_csv_frame(path) if path.exists() else None
    updates = []
    try:
        if cached is None or cached.empty:
            candles = download_candles(symbol, interval, start, end)
            if candles.empty:
                raise DataProviderError(f"Bybit returned no candle data for {symbol}")
            funding = download_funding(symbol, start, end)
            updates.append(merge_funding(candles, funding, interval))
        else:
            first_cached = cached.index.min()
            last_cached = cached.index.max()

            if requested_start < first_cached:
                backfill_end = first_cached.isoformat()
                candles = download_candles(symbol, interval, start, backfill_end)
                if not candles.empty:
                    funding = download_funding(symbol, start, backfill_end)
                    updates.append(merge_funding(candles, funding, interval))

            # Start at the latest cached bar to refresh it if it is still forming.
            if requested_end > last_cached:
                update_start = last_cached.isoformat()
                candles = download_candles(symbol, interval, update_start, end)
                if not candles.empty:
                    funding = download_funding(symbol, update_start, end)
                    updates.append(merge_funding(candles, funding, interval))

        if updates:
            frames = ([cached] if cached is not None and not cached.empty else []) + updates
            result = pd.concat(frames).sort_index()
            result = result[~result.index.duplicated(keep="last")]
            temp_path = path.with_suffix(path.suffix + ".tmp")
            result.to_csv(temp_path)
            temp_path.replace(path)
            cached = result
    except (requests.RequestException, DataProviderError) as exc:
        if cached is None or cached.empty:
            raise DataDownloadError(
                f"Could not download {symbol} data from Bybit and no cached data is "
                f"available at {path}. Check your internet connection or use Synthetic data."
            ) from exc
        warnings.warn(
            f"Bybit data update failed for {symbol}; using cached data: {exc}",
            RuntimeWarning,
        )

    if cached is None or cached.empty:
        raise RuntimeError(f"No cached or downloaded candle data available for {symbol}")

    result = cached.loc[(cached.index >= requested_start) & (cached.index <= requested_end)].copy()
    if result.empty:
        raise RuntimeError(
            f"No cached or downloaded candle data for {symbol} between "
            f"{requested_start.isoformat()} and {requested_end.isoformat()}"
        )
    return result


def load_csv_frame(path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    required = {"open", "high", "low", "close", "funding_rate"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Data file {path} is missing required columns: {', '.join(sorted(missing))}")
    for column in required:
        df[column] = pd.to_numeric(df[column], errors="raise")
    return df[["open", "high", "low", "close", "funding_rate"]].sort_index()


# --------------------------------------------------------------------------- #
# Synthetic data (for tests, demos and offline development)
# --------------------------------------------------------------------------- #
def make_synthetic(symbols=("BTCUSDT", "ETHUSDT"), years: float = 3.0, seed: int = 7,
                   start_prices=None) -> dict[str, pd.DataFrame]:
    """Hourly OHLC with fat tails, volatility clustering, occasional crash/pump
    jumps, a shared market factor, and funding that follows momentum."""
    rng = np.random.default_rng(seed)
    n = int(years * 365 * 24)
    idx = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC")
    start_prices = start_prices or {"BTCUSDT": 30_000.0, "ETHUSDT": 2_000.0}
    betas = {"BTCUSDT": 1.0, "ETHUSDT": 1.2}

    # shared volatility regime (AR(1) in log-vol) and market factor
    logv = np.zeros(n)
    for t in range(1, n):
        logv[t] = 0.995 * logv[t - 1] + 0.05 * rng.standard_normal()
    sigma = 0.0055 * np.exp(logv - logv.mean())
    factor = sigma * rng.standard_t(4, n) / np.sqrt(2.0)
    # rare jumps: crashes more common than pumps
    jump = np.zeros(n)
    n_jumps = max(1, int(years * 6))
    for t in rng.choice(n, n_jumps, replace=False):
        jump[t] = -rng.uniform(0.06, 0.18) if rng.random() < 0.7 else rng.uniform(0.05, 0.12)

    out = {}
    for s in symbols:
        beta = betas.get(s, 1.0)
        idio = 0.5 * sigma * rng.standard_t(4, n) / np.sqrt(2.0)
        ret = beta * (factor + jump) + idio
        close = start_prices.get(s, 100.0) * np.exp(np.cumsum(ret))
        open_ = np.concatenate([[start_prices.get(s, 100.0)], close[:-1]])
        wick = sigma * 0.8
        high = np.maximum(open_, close) * np.exp(np.abs(rng.standard_normal(n)) * wick)
        low = np.minimum(open_, close) * np.exp(-np.abs(rng.standard_normal(n)) * wick)
        # a crash bar should print a deep wick
        low = np.where(jump < 0, np.minimum(low, close), low)
        df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)
        # funding every 8h: base 1bp + momentum component, clipped
        mom = pd.Series(close, index=idx).pct_change(72).fillna(0.0).to_numpy()
        rate = np.clip(0.0001 + 0.02 * mom, -0.0075, 0.0075)
        df["funding_rate"] = np.where(idx.hour % 8 == 0, rate, 0.0)
        out[s] = df
    return out
