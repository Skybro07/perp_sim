# Perpetual Futures Margin & Liquidation Simulator

Simulates USDT-margined perpetual futures positions on historical candles: tiered maintenance margin,
liquidation prices, funding payments, isolated vs cross margin, and insurance-fund shortfall under
liquidation slippage.

## Run

```bash
pip install -r requirements.txt
python tests/test_engine.py            # 18 sanity tests
python main.py --synthetic             # offline demo (numbers are NOT real-market results)
python main.py --download              # Bybit public data, BTCUSDT + ETHUSDT, cached in data_cache/
streamlit run app.py                   # dashboard; exchange data is cached and incrementally updated
```

The dashboard uses Bybit's public API and caches mark-price candles and historical funding rates
in the project's `data_cache/` folder. API availability depends on your network and region. If
Bybit is unreachable on a first run, select Synthetic (demo) or restore internet access and retry.
For manual data, build a frame with columns `open, high, low, close, funding_rate`
(UTC index) and load it with `data.load_csv_frame`. Keep mark-price candles for liquidation prices;
ordinary trade-price candles are a different series.

The Bybit integration uses its [mark-price kline](https://bybit-exchange.github.io/docs/v5/market/mark-kline)
and [funding history](https://bybit-exchange.github.io/docs/v5/market/history-fund-rate) public endpoints.

## Files

| File | Purpose |
|---|---|
| `data.py` | Bybit download + cache, funding merge, synthetic generator |
| `margin.py` | Tiered maintenance margin, liquidation price, cross equity |
| `simulate.py` | Vectorised engine: isolated positions, isolated-vs-cross portfolios |
| `stress.py` | Experiments: leverage sweep, funding drag, crash replay, slippage, margin modes |
| `main.py` | Runs everything, writes tables/chart to `results/` |
| `app.py` | Streamlit dashboard |

## Key formulas

- Initial margin = notional / leverage
- Maintenance margin = notional x rate_tier - cum_tier (tiered, continuous)
- Long liquidation price: P = (qE - margin - cum) / (q(1 - r)); short: P = (qE + margin + cum) / (q(1 + r))
- Funding per bar = side x qty x price x funding_rate (longs pay when rate > 0)
- Cross: liquidated when wallet + sum(unrealised PnL) <= sum(maintenance margins)

## Assumptions and limits (keep this section in your final README)

- The default tier table is illustrative. Replace `DEFAULT_TIERS` in `margin.py` with the exchange's current table.
- No trading or liquidation fees, no order-book depth (flat slippage), no auto-deleveraging.
- Mark price is taken as the candle price; liquidations are tested against the bar's low (longs) / high (shorts).
- Shortfall is reported two ways: closing at the trigger price (optimistic) and at the bar extreme (pessimistic).
  Real execution lies between them.
- Random entries over one history give overlapping, correlated windows. Percentages describe this history.
- Cross-margin detection uses each leg's worst price within the same bar, which is conservative.
- Validate: compare a few liquidation prices against the exchange's own calculator before trusting results.

## Filling your resume placeholders

After `python main.py --download`:

- `[N] positions`, `[Z]% liquidated at [leverage]`: `results/liquidation_vs_leverage.csv`
- funding cost: `results/funding_drag.csv`
- crash behaviour: `results/crash_replay.csv`
- shortfall under slippage: `results/shortfall_vs_slippage.csv`
- isolated vs cross: `results/isolated_vs_cross.csv`
