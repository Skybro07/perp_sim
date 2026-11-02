"""Streamlit dashboard:  streamlit run app.py"""
import streamlit as st

from data import DataDownloadError, load_or_download, make_synthetic
from margin import IsolatedPosition
from stress import isolated_vs_cross, liquidation_vs_leverage, shortfall_vs_slippage

st.set_page_config(page_title="Perp Margin & Liquidation Simulator", layout="wide")
st.title("Perpetual Futures Margin & Liquidation Simulator")

with st.sidebar:
    source = st.radio("Data", [
        "Synthetic (demo)",
        "Bybit (cached, updates automatically)",
    ])
    hold_days = st.slider("Holding period (days)", 3, 90, 30)
    slippage = st.slider("Liquidation slippage (%)", 0.0, 1.0, 0.3, 0.05) / 100
    n_entries = st.slider("Random entries", 500, 5000, 2000, 500)


@st.cache_data(show_spinner="Loading data...")
def get_data(src):
    if src.startswith("Synthetic"):
        return make_synthetic()
    return {s: load_or_download(s) for s in ("BTCUSDT", "ETHUSDT")}


try:
    dfs = get_data(source)
except DataDownloadError as exc:
    st.error(str(exc))
    st.info("To use the dashboard offline, select **Synthetic (demo)** in the Data menu.")
    st.stop()

base = st.selectbox("Asset", list(dfs))
df = dfs[base]
if source.startswith("Synthetic"):
    st.warning("Synthetic data: the numbers below demonstrate the tool, not real markets.")

tab1, tab2, tab3, tab4 = st.tabs(["Position calculator", "Liquidation vs leverage",
                                  "Shortfall vs slippage", "Isolated vs cross"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    side = 1 if c1.selectbox("Side", ["Long", "Short"]) == "Long" else -1
    notional = c2.number_input("Notional (USDT)", 100.0, 5_000_000.0, 10_000.0, 100.0)
    lev = c3.slider("Leverage", 1, 100, 10)
    entry = c4.number_input("Entry price", 1.0, 1_000_000.0, float(df["close"].iloc[-1]))
    try:
        pos = IsolatedPosition(side, notional, entry, lev)
        m1, m2, m3 = st.columns(3)
        m1.metric("Initial margin", f"{pos.margin:,.2f} USDT")
        m2.metric("Liquidation price", f"{pos.liq_price():,.2f}")
        m3.metric("Move to liquidation", f"{pos.distance_to_liq() * 100:.2f}%")
    except ValueError as e:
        st.error(str(e))

with tab2:
    table = liquidation_vs_leverage(df, hold_days=hold_days, n_entries=n_entries, slippage=slippage)
    st.dataframe(table, use_container_width=True)
    pivot = table.pivot(index="leverage", columns="side", values="pct_liquidated")
    st.line_chart(pivot)

with tab3:
    lev_s = st.slider("Leverage for shortfall test", 2, 50, 20)
    st.dataframe(shortfall_vs_slippage(df, leverage=lev_s, hold_days=hold_days,
                                       n_entries=n_entries), use_container_width=True)
    st.caption("'Trigger fill' assumes the engine closes at the liquidation price; "
               "'worst fill' assumes it only completes at the bar's extreme. Reality is in between.")

with tab4:
    if len(dfs) < 2:
        st.info("Needs two assets.")
    else:
        a, b = list(dfs)[:2]
        lev_p = st.slider("Portfolio leverage", 2, 30, 10)
        common = dfs[a].index.intersection(dfs[b].index)
        aligned = {a: dfs[a].loc[common], b: dfs[b].loc[common]}
        spec = [(a, 1, 0.5), (b, -1, 0.5)]
        st.write(f"Long {a} / short {b}, equal weight")
        st.dataframe(isolated_vs_cross(aligned, spec, leverage=lev_p, hold_days=hold_days,
                                       n_entries=n_entries, slippage=slippage),
                     use_container_width=True)
