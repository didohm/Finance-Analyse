"""FinScope — Financial Opportunity Analyzer.

Real-time market analysis dashboard: technical scoring, signals,
risk/return analytics, correlations, news and exports.
"""
from __future__ import annotations

import html
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import analysis as an
from data import (
    MAX_ASSETS,
    PERIODS,
    fetch_universe,
    get_secret,
    is_crypto,
    massive_news,
    parse_tickers,
)

st.set_page_config(
    page_title="FinScope — Financial Opportunity Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

MASSIVE_API_KEY = get_secret("MASSIVE_API_KEY")
DEFAULT_TICKERS = "AAPL, MSFT, GOOGL, AMZN, BTC-USD, ETH-USD"

# =========================
# STYLES
# =========================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --ink: #0f172a;
  --muted: #475569;
  --faint: #64748b;
  --surface: #ffffff;
  --surface-tint: #f8fafc;
  --border: #e2e8f0;
  --accent: #4f46e5;
}

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.stApp { background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%); }

.block-container { padding-top: 1.25rem; padding-bottom: 2rem; max-width: 1400px; }

h1, h2, h3 { letter-spacing: -0.02em; }

/* Accessible section headings */
.main h2 {
    font-size: 1.35rem;
    font-weight: 800;
    color: var(--ink);
    margin: 1.6rem 0 -0.4rem 0;
    padding-bottom: 0.45rem;
    border-bottom: 2px solid var(--border);
}
.main h3 { font-size: 1.05rem; font-weight: 700; color: var(--ink); }

.hero {
    background: linear-gradient(135deg, #0f172a 0%, #312e81 55%, #4f46e5 100%);
    color: #fff;
    padding: 1.75rem 2rem;
    border-radius: 20px;
    box-shadow: 0 18px 44px rgba(15, 23, 42, 0.16);
    margin-bottom: 1.25rem;
}
.hero h1 { margin: 0; font-size: clamp(1.5rem, 3.2vw, 2.15rem); font-weight: 800; }
.hero p { margin: 0.45rem 0 0; opacity: 0.88; font-size: 0.98rem; }
.hero .chips { margin-top: 0.9rem; display: flex; flex-wrap: wrap; gap: 0.45rem; }
.chip {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.22);
    border-radius: 999px;
    padding: 0.22rem 0.75rem;
    font-size: 0.78rem;
    font-weight: 600;
    backdrop-filter: blur(4px);
}

.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.15rem;
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
    transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
    height: 100%;
}
.card:hover {
    transform: translateY(-3px);
    box-shadow: 0 14px 32px rgba(15, 23, 42, 0.10);
    border-color: #cbd5e1;
}
.rank-badge {
    display: inline-block;
    padding: 0.26rem 0.7rem;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 700;
    border: 1px solid var(--border);
    background: var(--surface-tint);
    color: var(--muted);
    margin-bottom: 0.65rem;
}
.metric-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--faint);
    font-weight: 700;
}
.metric-price { font-size: 1.6rem; font-weight: 800; color: var(--ink); margin-top: 0.1rem; }
.metric-line { color: var(--muted); font-size: 0.88rem; margin-top: 0.28rem; }

.pill {
    display: inline-block;
    padding: 0.18rem 0.62rem;
    border-radius: 999px;
    font-size: 0.74rem;
    font-weight: 700;
    color: #fff;
    letter-spacing: 0.02em;
}

.up { color: #059669; font-weight: 700; }
.down { color: #dc2626; font-weight: 700; }

.ticker-chip {
    display: inline-block;
    background: #eef2ff;
    border: 1px solid #c7d2fe;
    color: #3730a3;
    border-radius: 999px;
    padding: 0.12rem 0.6rem;
    font-size: 0.75rem;
    font-weight: 700;
    margin: 0.12rem 0.18rem 0.12rem 0;
}

.news-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 0.95rem 1.1rem;
    margin-bottom: 0.7rem;
    transition: border-color .2s ease, box-shadow .2s ease;
}
.news-card:hover { border-color: #c7d2fe; box-shadow: 0 8px 20px rgba(15,23,42,.06); }
.news-card a { color: var(--ink); text-decoration: none; font-weight: 700; font-size: 1rem; }
.news-card a:hover { color: var(--accent); text-decoration: underline; }
.news-meta { color: var(--faint); font-size: 0.84rem; margin-top: 0.25rem; }

.footer {
    margin-top: 2.5rem;
    padding-top: 1.1rem;
    border-top: 1px solid var(--border);
    color: var(--faint);
    font-size: 0.86rem;
}

/* Interaction & accessibility */
div[data-testid="stButton"] > button:focus-visible,
div[data-testid="stDownloadButton"] > button:focus-visible,
a:focus-visible {
    outline: 3px solid var(--accent);
    outline-offset: 2px;
    border-radius: 8px;
}
div[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 0.9rem 1.05rem;
    box-shadow: 0 6px 16px rgba(15, 23, 42, 0.04);
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}
@media (max-width: 640px) {
    .hero { padding: 1.25rem 1.35rem; border-radius: 16px; }
    .block-container { padding-left: 1rem; padding-right: 1rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# HELPERS
# =========================
SOURCE_LABELS = {"massive": "Massive", "yahoo": "Yahoo Finance",
                 "historical_close": "last close"}


def fmt_price(value: float) -> str:
    if value is None or pd.isna(value):
        return "—"
    magnitude = abs(value)
    if magnitude >= 1:
        return f"${value:,.2f}"
    if magnitude >= 0.01:
        return f"${value:.4f}"
    return f"${value:.6f}"


def pct_html(value: float) -> str:
    if value is None or pd.isna(value):
        return '<span class="metric-line">n/a</span>'
    cls = "up" if value >= 0 else "down"
    arrow = "▲" if value >= 0 else "▼"
    return f'<span class="{cls}">{arrow} {value:+.2f}%</span>'


def signal_pill(signal: str) -> str:
    color = an.SIGNAL_COLORS.get(signal, "#64748b")
    return f'<span class="pill" style="background:{color};">{html.escape(signal)}</span>'


def ticker_chips(tickers: list[str]) -> str:
    return "".join(
        f'<span class="ticker-chip">{html.escape(t)}</span>' for t in tickers
    )


def empty_chart(message: str):
    st.info(message)


def preset_button(label: str, tickers_str: str):
    st.button(label, on_click=lambda: st.session_state.update(tickers_input=tickers_str),
              width="stretch")


# =========================
# SIDEBAR
# =========================
if "tickers_input" not in st.session_state:
    st.session_state.tickers_input = DEFAULT_TICKERS
if "period" not in st.session_state:
    st.session_state.period = "1Y"

with st.sidebar:
    st.markdown(
        """
        <div style="margin-bottom:1rem;">
          <div style="font-size:1.25rem;font-weight:800;color:#0f172a;">📊 FinScope</div>
          <div style="font-size:0.85rem;color:#64748b;">Financial Opportunity Analyzer</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tickers_input = st.text_area(
        "Tickers",
        key="tickers_input",
        help=f"Comma-separated symbols, up to {MAX_ASSETS}. Crypto uses the form BTC-USD.",
        height=96,
        placeholder="AAPL, MSFT, BTC-USD",
    )

    valid_tickers, invalid_tickers = parse_tickers(tickers_input or "")

    if valid_tickers:
        st.markdown(ticker_chips(valid_tickers), unsafe_allow_html=True)
        st.caption(f"{len(valid_tickers)} of max {MAX_ASSETS} assets selected")
    else:
        st.caption("Enter at least one ticker to begin.")

    if invalid_tickers:
        st.caption(
            "⚠️ Ignored invalid entries: " + ", ".join(map(html.escape, invalid_tickers))
        )

    st.markdown("**Quick presets**")
    p1, p2 = st.columns(2)
    with p1:
        preset_button("Tech", "AAPL, MSFT, GOOGL, AMZN, META")
        preset_button("Crypto", "BTC-USD, ETH-USD, SOL-USD, ADA-USD")
    with p2:
        preset_button("EV / Auto", "TSLA, RIVN, LCID, F, GM")
        preset_button("Finance", "JPM, V, MA, BAC, GS")

    period = st.select_slider(
        "Analysis period",
        options=list(PERIODS.keys()),
        key="period",
        help="Daily bars over this lookback window feed every indicator.",
    )

    refresh_btn = st.button("🔄 Refresh live prices", width="stretch")
    if refresh_btn:
        st.cache_data.clear()
        st.rerun()

    st.divider()
    if MASSIVE_API_KEY:
        st.caption("🟢 **Massive API** connected — institutional-grade data.")
    else:
        st.caption(
            "🟡 Using **Yahoo Finance** data. Set the `MASSIVE_API_KEY` "
            "secret for lower-latency market data."
        )
    st.caption("Prices are cached briefly (≈45 s) to stay fast and respect rate limits.")


# =========================
# HERO
# =========================
st.markdown(
    f"""
<div class="hero">
  <h1>Financial Opportunity Analyzer</h1>
  <p>Real-time prices • Technical scoring • Buy/sell signals • Risk analytics • News</p>
  <div class="chips">
    <span class="chip">📈 Period: {period}</span>
    <span class="chip">🗂️ Assets: {len(valid_tickers)}</span>
    <span class="chip">{'🟢 Massive API' if MASSIVE_API_KEY else '🟡 Yahoo Finance'}</span>
    <span class="chip">🕒 {datetime.now().strftime('%d %b %Y %H:%M')}</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

if not valid_tickers:
    st.info("👈 Enter some tickers in the sidebar (or pick a preset) to start analyzing.")
    st.stop()


# =========================
# FETCH + ANALYZE
# =========================
with st.spinner("Fetching market data…"):
    history_map, live_map, issues = fetch_universe(valid_tickers, period, MASSIVE_API_KEY)

failed = [t for t in valid_tickers if t not in history_map]
if failed:
    reasons = "; ".join(f"{t}: {issues.get(t, 'unavailable')}" for t in failed)
    st.warning(f"Could not fully load: {reasons}")

if not history_map:
    st.error(
        "No market data could be loaded for the selected tickers. "
        "Check your connection or try different symbols."
    )
    if st.button("Retry", type="primary"):
        st.cache_data.clear()
        st.rerun()
    st.stop()

metrics_df, closes_df = an.analyze(history_map, live_map)
if metrics_df.empty:
    st.error("Not enough historical data to compute the analysis. Try a longer period.")
    st.stop()


# =========================
# TOP OPPORTUNITIES
# =========================
st.markdown("## 🏆 Top opportunities")
top3 = metrics_df.head(3)
card_cols = st.columns(len(top3))
medals = ["🥇", "🥈", "🥉"]

for i, (_, row) in enumerate(top3.iterrows()):
    with card_cols[i]:
        st.markdown(
            f"""
<div class="card">
  <span class="rank-badge">{medals[i]} Rank #{int(row['Rank'])}</span>
  <div class="metric-label">{html.escape(str(row.name))}</div>
  <div class="metric-price">{fmt_price(row['Live_Price'])}</div>
  <div class="metric-line">vs prev close {pct_html(row['Variation_24h_%'])}</div>
  <div class="metric-line">RSI(14): <b>{row['RSI14']:.1f}</b> · Trend: <b>{row['Trend']}</b></div>
  <div class="metric-line">Score: <b>{row['Pro_Score']:.1f}/100</b></div>
  <div style="margin-top:0.55rem;">{signal_pill(row['Signal'])}</div>
</div>
""",
            unsafe_allow_html=True,
        )


# =========================
# SUMMARY METRICS
# =========================
st.markdown("## 📌 Summary")
best = metrics_df.iloc[0]
worst_risk_idx = metrics_df["Volatility_Annualized"].idxmax()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Assets analyzed", len(metrics_df),
          help="Number of assets successfully loaded and scored.")
k2.metric("Top score", f"{best.name}",
          f"{best['Pro_Score']:.1f} / 100",
          help="Highest composite score in the current selection.")
avg_ret = metrics_df["Return_Annualized"].mean() * 100
k3.metric("Avg annualized return", f"{avg_ret:.1f}%",
          help="Mean annualized return across the selection.")
avg_vol = metrics_df["Volatility_Annualized"].mean() * 100
k4.metric("Avg annualized risk", f"{avg_vol:.1f}%",
          help=f"Highest-risk asset: {worst_risk_idx}.")


# =========================
# COMPARISON TABLE
# =========================
st.markdown("## 📋 Detailed comparison")

display_df = metrics_df[
    [
        "Rank", "Live_Price", "Source", "Variation_24h_%", "RSI14", "Trend",
        "MACD_Hist", "Return_Annualized", "Volatility_Annualized",
        "Pro_Score", "Signal", "Confidence", "Suggested_Weight_%", "Max_Drawdown_%",
    ]
].reset_index()

display_df.columns = [
    "Asset", "Rank", "Price", "Source", "Δ vs Prev Close %", "RSI(14)", "Trend",
    "MACD Hist", "Annual Return", "Annual Risk", "Score", "Signal",
    "Confidence", "Suggested Weight %", "Max Drawdown %",
]


def color_signal(val: str) -> str:
    bg = an.SIGNAL_COLORS.get(val, "#64748b")
    weight = "bold" if val in {"STRONG BUY", "SELL"} else "normal"
    return f"background-color:{bg};color:white;font-weight:{weight};"


styled = (
    display_df.style.format(
        {
            "Price": lambda v: f"{v:,.2f}",
            "Δ vs Prev Close %": lambda v: "—" if pd.isna(v) else f"{v:+.2f}%",
            "RSI(14)": lambda v: "—" if pd.isna(v) else f"{v:.1f}",
            "MACD Hist": lambda v: "—" if pd.isna(v) else f"{v:.4f}",
            "Annual Return": lambda v: "—" if pd.isna(v) else f"{v:+.1%}",
            "Annual Risk": lambda v: "—" if pd.isna(v) else f"{v:.1%}",
            "Score": lambda v: f"{v:.1f}",
            "Suggested Weight %": lambda v: f"{v:.1f}%",
            "Max Drawdown %": lambda v: "—" if pd.isna(v) else f"{v:.1f}%",
        },
        na_rep="—",
    )
    .background_gradient(subset=["Score"], cmap="YlGn", vmin=0, vmax=100)
    .background_gradient(subset=["Annual Return"], cmap="RdYlGn")
    .background_gradient(subset=["Annual Risk"], cmap="YlOrRd")
    .map(color_signal, subset=["Signal"])
    .set_properties(**{"text-align": "center"})
    .set_properties(subset=["Asset"], **{"text-align": "left", "font-weight": "bold"})
)

st.dataframe(styled, width="stretch", hide_index=True, height=min(120 + 35 * len(display_df), 520))
st.caption(
    "Scores blend return, risk, momentum, RSI positioning and MACD strength. "
    f"Drawdown is measured over the selected {period} window."
)


# =========================
# RISK / RETURN MATRIX
# =========================
st.markdown("## ⚖️ Risk / Return matrix")

scatter_df = metrics_df.reset_index()
scatter_df["Return_%"] = scatter_df["Return_Annualized"] * 100
scatter_df["Risk_%"] = scatter_df["Volatility_Annualized"] * 100
scatter_df["Marker Size"] = scatter_df["Pro_Score"].clip(lower=10)

fig_scatter = px.scatter(
    scatter_df,
    x="Risk_%",
    y="Return_%",
    color="Pro_Score",
    size="Marker Size",
    hover_name="Asset",
    hover_data={
        "Risk_%": ":.1f",
        "Return_%": ":.1f",
        "Pro_Score": ":.1f",
        "Signal": True,
        "Marker Size": False,
    },
    color_continuous_scale="Viridis",
    size_max=34,
    labels={"Risk_%": "Risk — annualized volatility (%)", "Return_%": "Return — annualized (%)"},
)
mean_x = float(scatter_df["Risk_%"].mean())
mean_y = float(scatter_df["Return_%"].mean())
fig_scatter.add_hline(y=mean_y, line_dash="dot", line_color="#94a3b8",
                      annotation_text="Avg return", annotation_position="top right")
fig_scatter.add_vline(x=mean_x, line_dash="dot", line_color="#94a3b8",
                      annotation_text="Avg risk", annotation_position="bottom right")
fig_scatter.update_layout(
    height=500,
    template="plotly_white",
    margin=dict(l=20, r=20, t=30, b=20),
    coloraxis_colorbar=dict(title="Score"),
)
st.plotly_chart(fig_scatter, width="stretch", key="risk_return")


# =========================
# NORMALIZED PERFORMANCE
# =========================
st.markdown("## 📈 Relative performance (rebased)")

fig_line = go.Figure()
plotted_any = False
for col in closes_df.columns:
    s = closes_df[col].dropna()
    if len(s) < 2:
        continue
    norm = s / s.iloc[0] * 100
    plotted_any = True
    fig_line.add_trace(go.Scatter(
        x=norm.index, y=norm, mode="lines", name=col,
        hovertemplate=f"<b>{col}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.1f}}<extra></extra>",
    ))

if plotted_any:
    fig_line.add_hline(y=100, line_width=1, line_color="#cbd5e1")
    fig_line.update_layout(
        height=450,
        template="plotly_white",
        hovermode="x unified",
        yaxis_title="Rebased value (start = 100)",
        legend=dict(orientation="h"),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig_line, width="stretch", key="normalized")
else:
    empty_chart("Not enough overlapping history to compare performance.")
st.caption("Each asset starts at 100 at its own first available date, so recently listed assets remain comparable.")


# =========================
# CORRELATION HEATMAP
# =========================
st.markdown("## 🔗 Correlation matrix")
corr = closes_df.corr(min_periods=20)
if corr.shape[0] > 1 and not corr.dropna(how="all").empty:
    fig_corr = px.imshow(
        corr,
        zmin=-1, zmax=1,
        color_continuous_scale="RdBu_r",
        text_auto=".2f" if corr.shape[0] <= 8 else False,
        aspect="auto",
        labels=dict(color="Correlation"),
    )
    fig_corr.update_layout(
        height=max(380, 60 * corr.shape[0]),
        template="plotly_white",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig_corr, width="stretch", key="correlation")
    st.caption("Low pairwise correlation suggests stronger diversification when combined.")
else:
    empty_chart("Add at least two assets to compute correlations.")


# =========================
# SUGGESTED WEIGHTS
# =========================
st.markdown("## 🥧 Suggested portfolio weights")
weights_df = (
    metrics_df[["Suggested_Weight_%"]]
    .sort_values("Suggested_Weight_%")
)
fig_weights = px.bar(
    weights_df,
    x="Suggested_Weight_%",
    y=weights_df.index,
    orientation="h",
    text="Suggested_Weight_%",
    color_discrete_sequence=["#4f46e5"],
    labels={"Suggested_Weight_%": "Weight (%)", "index": ""},
)
fig_weights.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
fig_weights.update_layout(
    height=max(320, 42 * len(weights_df)),
    template="plotly_white",
    xaxis_title="Suggested allocation (%)",
    margin=dict(l=10, r=40, t=10, b=10),
)
st.plotly_chart(fig_weights, width="stretch", key="weights")
st.caption("Weights are proportional to each asset's composite score — informational only, not investment advice.")


# =========================
# DEEP DIVE
# =========================
st.markdown("## 🔍 Deep dive")
default_asset = metrics_df.index[0]
asset_order = list(metrics_df.index)
deep_asset = st.selectbox(
    "Asset",
    asset_order,
    index=asset_order.index(default_asset) if default_asset in asset_order else 0,
    help="Pick any analyzed asset for detailed charts.",
)

deep_close = closes_df[deep_asset]
deep_volume = history_map[deep_asset].get("Volume") if deep_asset in history_map else None
if deep_volume is not None:
    deep_volume = pd.to_numeric(deep_volume, errors="coerce").reindex(deep_close.index)

bb_mid, bb_up, bb_low = an.bollinger(deep_close)
macd_line, macd_sig, macd_hist = an.macd(deep_close)
rsi_series = an.rsi(deep_close)

tab_price, tab_macd, tab_rsi = st.tabs(["Price & Bollinger Bands", "MACD", "RSI"])

with tab_price:
    fig_price = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22],
        vertical_spacing=0.03,
    )
    fig_price.add_trace(go.Scatter(
        x=deep_close.index, y=deep_close, name="Close",
        line=dict(color="#4f46e5", width=2),
    ), row=1, col=1)
    fig_price.add_trace(go.Scatter(x=bb_mid.index, y=bb_mid, name="BB mid",
                                   line=dict(color="#94a3b8", dash="dot")), row=1, col=1)
    fig_price.add_trace(go.Scatter(
        x=bb_up.index, y=bb_up, name="BB upper",
        line=dict(color="#cbd5e1"), hoverinfo="skip",
    ), row=1, col=1)
    fig_price.add_trace(go.Scatter(
        x=bb_low.index, y=bb_low, name="BB lower", fill="tonexty",
        fillcolor="rgba(79,70,229,0.06)", line=dict(color="#cbd5e1"), hoverinfo="skip",
    ), row=1, col=1)
    if deep_volume is not None and deep_volume.notna().any():
        fig_price.add_trace(go.Bar(
            x=deep_close.index, y=deep_volume, name="Volume",
            marker_color="#c7d2fe", showlegend=False,
        ), row=2, col=1)
        fig_price.update_yaxes(title_text="Volume", row=2, col=1)
    fig_price.update_layout(
        template="plotly_white", height=480,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title_text="Price ($)",
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig_price, width="stretch", key=f"price_{deep_asset}")

with tab_macd:
    fig_macd = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             row_heights=[0.62, 0.38], vertical_spacing=0.05)
    fig_macd.add_trace(go.Scatter(x=macd_line.index, y=macd_line, name="MACD",
                                  line=dict(color="#4f46e5")), row=1, col=1)
    fig_macd.add_trace(go.Scatter(x=macd_sig.index, y=macd_sig, name="Signal",
                                  line=dict(color="#f59e0b")), row=1, col=1)
    fig_macd.add_trace(go.Bar(
        x=macd_hist.index, y=macd_hist, name="Histogram",
        marker_color=np.where(macd_hist >= 0, "#10b981", "#ef4444"),
    ), row=2, col=1)
    fig_macd.update_layout(
        template="plotly_white", height=430,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title_text="MACD",
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig_macd, width="stretch", key=f"macd_{deep_asset}")

with tab_rsi:
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=rsi_series.index, y=rsi_series, name="RSI(14)",
                                 line=dict(color="#4f46e5")))
    fig_rsi.add_hrect(y0=30, y1=70, fillcolor="rgba(16,185,129,0.07)",
                      line_width=0, annotation_text="Neutral zone",
                      annotation_font_size=11, annotation_font_color="#94a3b8")
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ef4444")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="#059669")
    fig_rsi.update_layout(
        template="plotly_white", height=400,
        yaxis_range=[0, 100],
        yaxis_title_text="RSI",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_rsi, width="stretch", key=f"rsi_{deep_asset}")

latest_rsi = rsi_series.dropna().iloc[-1] if rsi_series.notna().any() else None
if latest_rsi is not None:
    zone = "oversold" if latest_rsi < 30 else "overbought" if latest_rsi > 70 else "neutral"
    st.caption(f"Latest RSI(14) for {deep_asset}: **{latest_rsi:.1f}** ({zone}).")


# =========================
# SIGNAL DISTRIBUTION
# =========================
st.markdown("## 🚦 Signal distribution")
sig_counts = metrics_df["Signal"].value_counts()
sig_df = pd.DataFrame({"Signal": sig_counts.index.astype(str), "Count": sig_counts.values})
signal_order = [s for s in an.SIGNAL_ORDER if s in set(sig_df["Signal"])]
c_pie, c_legend = st.columns([3, 2])
with c_pie:
    fig_pie = px.pie(
        sig_df,
        names="Signal",
        values="Count",
        hole=0.58,
        color="Signal",
        color_discrete_map=an.SIGNAL_COLORS,
        category_orders={"Signal": signal_order},
    )
    fig_pie.update_traces(textinfo="percent", textfont_size=13)
    fig_pie.update_layout(
        height=360, template="plotly_white",
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    st.plotly_chart(fig_pie, width="stretch", key="signals")
with c_legend:
    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    for signal_name, count in sig_counts.items():
        color = an.SIGNAL_COLORS.get(signal_name, "#64748b")
        st.markdown(
            f"""
<div style="display:flex;justify-content:space-between;align-items:center;
            padding:0.42rem 0;border-bottom:1px solid #f1f5f9;">
  <span>{signal_pill(signal_name)}</span>
  <b>{count}</b>
</div>
""",
            unsafe_allow_html=True,
        )


# =========================
# NEWS
# =========================
st.markdown("## 📰 Latest news")

news_candidates = [
    t if not is_crypto(t) else t.rsplit("-", 1)[0]
    for t in metrics_df.index
]
news_symbol = st.selectbox("Show news for", news_candidates, index=0)

news_items = massive_news(news_symbol, MASSIVE_API_KEY, limit=5) if MASSIVE_API_KEY else []
if news_items:
    for item in news_items[:5]:
        title = item.get("title") or item.get("headline") or "Untitled"
        link = item.get("article_url") or item.get("url") or item.get("link") or "#"
        publisher_obj = item.get("publisher")
        if isinstance(publisher_obj, dict):
            publisher = publisher_obj.get("name", "Unknown source")
        else:
            publisher = publisher_obj or "Unknown source"
        published = item.get("published_utc") or item.get("published_at") or ""
        date_label = ""
        if published:
            try:
                date_label = " · " + datetime.fromisoformat(published.replace("Z", "+00:00")).strftime("%d %b %Y")
            except ValueError:
                date_label = ""

        st.markdown(
            f"""
<div class="news-card">
  <a href="{html.escape(link, quote=True)}" target="_blank" rel="noopener noreferrer">{html.escape(title)}</a>
  <div class="news-meta">{html.escape(str(publisher))}{date_label}</div>
</div>
""",
            unsafe_allow_html=True,
        )
else:
    st.info(
        "No news available here. News requires a Massive API key and coverage "
        "for the selected symbol."
    )


# =========================
# EXPORT
# =========================
st.markdown("## 📤 Export")
exp_col1, exp_col2, _ = st.columns([1, 1, 2])

with exp_col1:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        metrics_df.reset_index().to_excel(writer, sheet_name="Opportunities", index=False)
        closes_df.to_excel(writer, sheet_name="Historical_Close")
    st.download_button(
        "⬇️ Download Excel report",
        data=buffer.getvalue(),
        file_name=f"finscope_{datetime.now():%Y%m%d_%H%M}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )
with exp_col2:
    st.download_button(
        "⬇️ Download CSV",
        data=metrics_df.reset_index().to_csv(index=False).encode("utf-8-sig"),
        file_name=f"finscope_{datetime.now():%Y%m%d_%H%M}.csv",
        mime="text/csv",
        width="stretch",
    )


# =========================
# FOOTER
# =========================
sources_used = sorted({SOURCE_LABELS.get(str(s), str(s)) for s in metrics_df["Source"].dropna()})
st.markdown(
    f"""
<div class="footer">
  <b>FinScope</b> · Data: {' & '.join(sources_used) or 'historical closes'} · Built with Streamlit, Plotly &amp; pandas<br/>
  ⚠️ For research and education only — not financial advice. Past performance does not guarantee future results.
</div>
""",
    unsafe_allow_html=True,
)
