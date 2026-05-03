# app.py
import os
from io import BytesIO
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf


# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="FinTech Opportunities Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject Voiceflow Chatbot Widget
components.html(
    """
    <script type="text/javascript">
      if (!window.parent.document.getElementById('voiceflow-widget-script')) {
          const script = window.parent.document.createElement('script');
          script.id = 'voiceflow-widget-script';
          script.src = "https://cdn.voiceflow.com/widget-next/bundle.mjs";
          script.type = "text/javascript";
          script.onload = function() {
            window.parent.voiceflow.chat.load({
              verify: { projectID: '69f628ab952c4d555315ba3a' },
              url: 'https://general-runtime.voiceflow.com',
              versionID: 'production',
              voice: {
                url: "https://runtime-api.voiceflow.com"
              }
            });
          };
          window.parent.document.head.appendChild(script);
      }
    </script>
    """,
    height=0,
    width=0,
)

MASSIVE_BASE_URL = os.getenv("MASSIVE_BASE_URL", "https://api.massive.com").rstrip("/")

def _get_secret(name: str, default: str = "") -> str:
    try:
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return os.getenv(name, default).strip()

MASSIVE_API_KEY = _get_secret("MASSIVE_API_KEY")


# =========================
# UI STYLE
# =========================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    animation: fadeIn 0.6s ease-out;
}

.hero {
    background: linear-gradient(135deg, #0f172a 0%, #312e81 55%, #4f46e5 100%);
    color: white;
    padding: 2rem;
    border-radius: 24px;
    box-shadow: 0 20px 50px rgba(15, 23, 42, 0.18);
    margin-bottom: 1.2rem;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.hero:hover {
    transform: translateY(-2px);
    box-shadow: 0 25px 60px rgba(15, 23, 42, 0.25);
}

.hero h1 {
    margin: 0;
    font-size: 2.15rem;
    font-weight: 800;
    letter-spacing: -0.03em;
}

.hero p {
    margin: 0.5rem 0 0 0;
    opacity: 0.9;
}

.card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 22px;
    padding: 1.15rem;
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.05);
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    animation: fadeIn 0.5s ease-out backwards;
}

.card:nth-child(1) { animation-delay: 0.1s; }
.card:nth-child(2) { animation-delay: 0.2s; }
.card:nth-child(3) { animation-delay: 0.3s; }

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 35px rgba(15, 23, 42, 0.1);
    border-color: #cbd5e1;
}

div[data-testid="stButton"] > button {
    transition: all 0.2s ease-in-out;
}
div[data-testid="stButton"] > button:hover {
    transform: scale(1.02);
}
div[data-testid="stButton"] > button:active {
    transform: scale(0.98);
}

.small-badge {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    border: 1px solid #e2e8f0;
    background: #f8fafc;
    color: #334155;
    margin-bottom: 0.75rem;
}

.metric-title {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
    font-weight: 700;
}

.metric-value {
    font-size: 1.55rem;
    font-weight: 800;
    color: #0f172a;
    margin-top: 0.15rem;
}

.metric-sub {
    color: #64748b;
    font-size: 0.9rem;
}

.section-title {
    font-size: 1.25rem;
    font-weight: 800;
    color: #0f172a;
    margin: 1.3rem 0 0.8rem 0;
}

.footer {
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #e2e8f0;
    color: #64748b;
    font-size: 0.9rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# HELPERS
# =========================
def parse_tickers(text: str) -> list[str]:
    tickers = [t.strip().upper() for t in text.split(",")]
    tickers = [t for t in tickers if t]
    seen, out = set(), []
    for t in tickers:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def is_crypto_ticker(ticker: str) -> bool:
    # Example: BTC-USD, ETH-USD
    return "-" in ticker and ticker.endswith("-USD")


def split_crypto_ticker(ticker: str) -> tuple[str, str]:
    base, quote = ticker.split("-", 1)
    return base, quote


def strip_tz_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    try:
        if isinstance(out.index, pd.DatetimeIndex):
            if out.index.tz is not None:
                out.index = out.index.tz_convert(None)
    except Exception:
        pass
    return out


def request_massive(path: str, params: dict | None = None) -> dict | None:
    if not MASSIVE_API_KEY:
        return None

    url = f"{MASSIVE_BASE_URL}{path}"
    final_params = dict(params or {})
    final_params["apiKey"] = MASSIVE_API_KEY

    try:
        r = requests.get(url, params=final_params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def massive_stock_history(symbol: str, start: str, end: str, multiplier: int = 1, timespan: str = "day") -> pd.DataFrame:
    data = request_massive(
        f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start}/{end}",
        params={"adjusted": "true", "sort": "asc", "limit": "50000"},
    )
    if not data:
        return pd.DataFrame()

    results = data.get("results", [])
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    if "t" not in df.columns or "c" not in df.columns:
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(None)
    df = df.set_index("Date").sort_index()
    cols_map = {"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"}
    keep = [c for c in cols_map if c in df.columns]
    df = df[keep].rename(columns=cols_map)
    return df


def massive_stock_live(symbol: str) -> tuple[float | None, str]:
    data = request_massive(f"/v2/last/trade/{symbol}")
    if not data:
        return None, "no_api"

    res = data.get("results", {})
    price = res.get("p")
    if price is not None:
        return float(price), "massive"

    return None, "no_price"


def massive_crypto_live(symbol: str) -> tuple[float | None, str]:
    base, quote = split_crypto_ticker(symbol)
    data = request_massive(f"/v1/last/crypto/{base}/{quote}")
    if not data:
        return None, "no_api"

    last = data.get("last", {})
    price = last.get("price")
    if price is not None:
        return float(price), "massive"

    return None, "no_price"


@st.cache_data(ttl=300)
def yf_history(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    raw = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = strip_tz_index(raw)
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in raw.columns]
    if not cols:
        return pd.DataFrame()
    return raw[cols].copy()


def period_to_days(period: str) -> int:
    if period == "1d":
        return 1
    elif period == "5d":
        return 5
    elif period == "1mo":
        return 30
    elif period == "6mo":
        return 180
    elif period == "1y":
        return 365
    elif period == "5y":
        return 365 * 5
    return 365


def fetch_history(symbol: str, is_crypto: bool, period: str = "1y") -> pd.DataFrame:
    end = datetime.now(timezone.utc).date()
    start = (end - timedelta(days=period_to_days(period))).isoformat()
    end_str = end.isoformat()

    if not is_crypto:
        df = massive_stock_history(symbol, start, end_str)
        if not df.empty:
            return df

    # fallback / crypto path
    if is_crypto:
        df = yf_history(symbol, period=period, interval="1d" if period != "1d" else "15m")
        if not df.empty:
            return df

    # final fallback for stock too
    df = yf_history(symbol, period=period, interval="1d" if period != "1d" else "15m")
    return df


def fetch_live_price(symbol: str, is_crypto: bool) -> tuple[float | None, str]:
    if is_crypto:
        price, src = massive_crypto_live(symbol)
        if price is not None:
            return price, src
    else:
        price, src = massive_stock_live(symbol)
        if price is not None:
            return price, src

    # fallback to yfinance
    try:
        t = yf.Ticker(symbol)
        fast = getattr(t, "fast_info", None)
        if fast:
            last = fast.get("lastPrice") or fast.get("regularMarketPrice")
            if last is not None:
                return float(last), "yfinance_fast_info"

        hist = t.history(period="5d", interval="1d")
        if hist is not None and not hist.empty and "Close" in hist.columns:
            return float(hist["Close"].dropna().iloc[-1]), "yfinance_history"
    except Exception:
        pass

    return None, "unavailable"


def fetch_news(symbol: str) -> list[dict]:
    # Stocks news endpoint
    data = request_massive("/v2/reference/news", params={"ticker": symbol, "limit": 5, "order": "desc"})
    if not data:
        return []
    results = data.get("results", [])
    if isinstance(results, list):
        return results
    return []


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return line, sig, hist


def bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


def max_drawdown(series: pd.Series) -> float:
    roll_max = series.cummax()
    dd = (series / roll_max) - 1
    return float(dd.min())


def classify_signal(rsi_v, trend, macd_hist_v, price_vs_bb):
    if pd.isna(rsi_v) or pd.isna(macd_hist_v):
        return "WAIT"

    if rsi_v < 30 and trend == "Bullish" and macd_hist_v > 0 and price_vs_bb != "above":
        return "STRONG BUY"
    if rsi_v < 40 and trend == "Bullish":
        return "BUY"
    if rsi_v > 70 and (trend == "Bearish" or macd_hist_v < 0):
        return "SELL"
    if rsi_v > 65 and price_vs_bb == "above":
        return "REDUCE"
    return "HOLD"


def build_asset_table(history_map: dict[str, pd.DataFrame], live_map: dict[str, dict]) -> pd.DataFrame:
    closes = {}
    for symbol, df in history_map.items():
        if df is None or df.empty or "Close" not in df.columns:
            continue
        s = df["Close"].copy()
        s.index = pd.to_datetime(s.index)
        closes[symbol] = s

    close_df = pd.concat(closes, axis=1).sort_index().ffill().bfill()
    if close_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    live_rows = []
    for symbol in close_df.columns:
        last_close = float(close_df[symbol].dropna().iloc[-1])
        prev_close = float(close_df[symbol].dropna().iloc[-2]) if close_df[symbol].dropna().shape[0] > 1 else last_close

        live_price = live_map.get(symbol, {}).get("price")
        src = live_map.get(symbol, {}).get("source", "unknown")
        if live_price is None:
            live_price = last_close
            src = "historical_close"

        live_rows.append(
            {
                "Asset": symbol,
                "Live_Price": float(live_price),
                "Source": src,
                "Historical_Last": last_close,
                "Previous_Close": prev_close,
            }
        )

    df = pd.DataFrame(live_rows).set_index("Asset")

    df["Variation_24h_%"] = ((df["Live_Price"] - df["Previous_Close"]) / df["Previous_Close"]) * 100
    df["Return_Annualized"] = close_df.pct_change().mean() * 252
    df["Volatility_Annualized"] = close_df.pct_change().std() * np.sqrt(252)

    sma20 = close_df.rolling(20).mean().iloc[-1]
    sma50 = close_df.rolling(50).mean().iloc[-1]
    rsi14 = close_df.apply(rsi).iloc[-1]

    macd_line = {}
    macd_sig = {}
    macd_hist = {}
    bb_mid = {}
    bb_up = {}
    bb_low = {}

    for symbol in close_df.columns:
        ml, ms, mh = macd(close_df[symbol])
        mid, up, low = bollinger(close_df[symbol])
        macd_line[symbol] = ml.iloc[-1]
        macd_sig[symbol] = ms.iloc[-1]
        macd_hist[symbol] = mh.iloc[-1]
        bb_mid[symbol] = mid.iloc[-1]
        bb_up[symbol] = up.iloc[-1]
        bb_low[symbol] = low.iloc[-1]

    df["SMA20"] = sma20
    df["SMA50"] = sma50
    df["RSI14"] = rsi14
    df["MACD"] = pd.Series(macd_line)
    df["MACD_Signal"] = pd.Series(macd_sig)
    df["MACD_Hist"] = pd.Series(macd_hist)
    df["BB_Mid"] = pd.Series(bb_mid)
    df["BB_Upper"] = pd.Series(bb_up)
    df["BB_Lower"] = pd.Series(bb_low)

    df["Trend"] = np.where(df["SMA20"] > df["SMA50"], "Bullish", "Bearish")
    df["Momentum"] = df["SMA20"] / df["SMA50"]

    def bb_state(row):
        price = row["Live_Price"]
        if pd.isna(price) or pd.isna(row["BB_Upper"]) or pd.isna(row["BB_Lower"]):
            return "inside"
        if price > row["BB_Upper"]:
            return "above"
        if price < row["BB_Lower"]:
            return "below"
        return "inside"

    df["BB_State"] = df.apply(bb_state, axis=1)

    ret_rank = df["Return_Annualized"].rank(pct=True)
    risk_rank = 1 - df["Volatility_Annualized"].rank(pct=True)
    momentum_rank = df["Momentum"].rank(pct=True)

    rsi_quality = 1 - (df["RSI14"] - 50).abs() / 50
    rsi_quality = rsi_quality.clip(lower=0, upper=1)
    macd_rank = df["MACD_Hist"].rank(pct=True)

    df["Pro_Score"] = 100 * (
        0.28 * ret_rank.fillna(0.5)
        + 0.22 * risk_rank.fillna(0.5)
        + 0.18 * momentum_rank.fillna(0.5)
        + 0.16 * rsi_quality.fillna(0.5)
        + 0.16 * macd_rank.fillna(0.5)
    )

    df["Signal"] = df.apply(
        lambda row: classify_signal(row["RSI14"], row["Trend"], row["MACD_Hist"], row["BB_State"]),
        axis=1,
    )

    df["Confidence"] = pd.cut(
        df["Pro_Score"],
        bins=[-1, 40, 60, 75, 101],
        labels=["Low", "Medium", "High", "Very High"],
    )

    positive = df["Pro_Score"].clip(lower=0)
    df["Suggested_Weight_%"] = 100 * positive / positive.sum() if positive.sum() > 0 else 100 / len(df)

    df["Max_Drawdown_1Y_%"] = [
        max_drawdown(close_df[c]) * 100 if c in close_df.columns else np.nan
        for c in df.index
    ]

    df = df.sort_values("Pro_Score", ascending=False).copy()
    df.insert(0, "Rank", range(1, len(df) + 1))

    return df, close_df


def to_excel(analysis_df: pd.DataFrame, close_df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        analysis_df.reset_index().to_excel(writer, sheet_name="Opportunities", index=False)
        close_df.to_excel(writer, sheet_name="Historical_Close")
    return buffer.getvalue()


# =========================
# HEADER
# =========================
st.markdown(
    """
<div class="hero">
  <h1>Analyse & Détection d’Opportunités Financières</h1>
  <p>Real-time data • Scoring • Signals • Charts • Excel export</p>
</div>
""",
    unsafe_allow_html=True,
)

if "tickers_input" not in st.session_state:
    st.session_state.tickers_input = "AAPL, TSLA, MSFT, BTC-USD, ETH-USD"

def set_tickers(tickers_str):
    st.session_state.tickers_input = tickers_str

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("### ⚙️ Paramètres")

    tickers_input = st.text_area(
        "Tickers",
        key="tickers_input",
        help="Sépare les tickers par des virgules",
        height=90,
    )

    st.markdown("<p style='font-size: 0.85rem; color: #64748b; margin-bottom: 0.5rem;'>💡 Suggestions:</p>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.button("Tech", on_click=set_tickers, args=("AAPL, MSFT, GOOGL, AMZN, META",), use_container_width=True)
        st.button("Crypto", on_click=set_tickers, args=("BTC-USD, ETH-USD, SOL-USD, ADA-USD",), use_container_width=True)
    with c2:
        st.button("EV/Auto", on_click=set_tickers, args=("TSLA, RIVN, LCID, F, GM",), use_container_width=True)
        st.button("Finance", on_click=set_tickers, args=("JPM, V, MA, BAC, GS",), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    period = st.selectbox("Chart period", ["1d", "5d", "1mo", "6mo", "1y", "5y"], index=2)
    use_massive = st.toggle("Use Real Data", value=True)
    run_btn = st.button("🚀 Launch Analysis", use_container_width=True, type="primary")

# =========================
# MAIN
# =========================
if run_btn:
    tickers = parse_tickers(tickers_input)

    if not tickers:
        st.error("دخل على الأقل ticker واحد صحيح.")
        st.stop()

    history_map = {}
    live_map = {}

    with st.spinner("Fetching data..."):
        for t in tickers:
            crypto = is_crypto_ticker(t)

            # Historical data
            if use_massive and not crypto:
                hist = fetch_history(t, is_crypto=False, period=period)
            else:
                hist = fetch_history(t, is_crypto=crypto, period=period)

            if hist is None or hist.empty:
                st.warning(f"No historical data for {t}")
                continue

            history_map[t] = hist

            # Live data
            price, src = (None, "disabled")
            if use_massive:
                price, src = fetch_live_price(t, is_crypto=crypto)
            if price is None:
                price, src = fetch_live_price(t, is_crypto=crypto)

            live_map[t] = {"price": price, "source": src}

    if not history_map:
        st.error("Aucune donnée récupérée.")
        st.stop()

    analysis_df, close_df = build_asset_table(history_map, live_map)

    if analysis_df.empty:
        st.error("Impossible de construire l’analyse.")
        st.stop()

    st.success("Analysis completed successfully.")

    # =========================
    # TOP 3
    # =========================
    st.markdown('<div class="section-title">Top opportunities</div>', unsafe_allow_html=True)
    top3 = analysis_df.head(3)
    cols = st.columns(3)
    medals = ["🥇", "🥈", "🥉"]

    for i, (_, row) in enumerate(top3.iterrows()):
        with cols[i]:
            st.markdown(
                f"""
<div class="card">
  <div class="small-badge">{medals[i]} Rank #{int(row["Rank"])}</div>
  <div class="metric-title">{row.name}</div>
  <div class="metric-value">${row["Live_Price"]:,.2f}</div>
  <div class="metric-sub">24h: <b>{row["Variation_24h_%"]:+.2f}%</b></div>
  <div class="metric-sub">RSI(14): <b>{row["RSI14"]:.1f}</b> • Trend: <b>{row["Trend"]}</b></div>
  <div class="metric-sub">Signal: <b>{row["Signal"]}</b> • Score: <b>{row["Pro_Score"]:.1f}</b></div>
</div>
""",
                unsafe_allow_html=True,
            )

    # =========================
    # SUMMARY METRICS
    # =========================
    st.markdown('<div class="section-title">Summary</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Assets analyzed", len(analysis_df))
    with m2:
        best = analysis_df.iloc[0]
        st.metric("Best asset", best.name, f'{best["Pro_Score"]:.1f} score')
    with m3:
        st.metric("Average annual return", f'{analysis_df["Return_Annualized"].mean() * 100:.1f}%')
    with m4:
        st.metric("Average annual risk", f'{analysis_df["Volatility_Annualized"].mean() * 100:.1f}%')

    # =========================
    # TABLE
    # =========================
    st.markdown('<div class="section-title">Detailed comparison table</div>', unsafe_allow_html=True)

    display_df = analysis_df[
        [
            "Rank",
            "Live_Price",
            "Source",
            "Variation_24h_%",
            "RSI14",
            "Trend",
            "MACD_Hist",
            "Return_Annualized",
            "Volatility_Annualized",
            "Pro_Score",
            "Signal",
            "Confidence",
            "Suggested_Weight_%",
            "Max_Drawdown_1Y_%",
        ]
    ].reset_index()

    display_df.columns = [
        "Asset",
        "Rank",
        "Price",
        "Source",
        "24h %",
        "RSI(14)",
        "Trend",
        "MACD Hist",
        "Annual Return",
        "Annual Risk",
        "Score",
        "Signal",
        "Confidence",
        "Suggested Weight %",
        "1Y Max Drawdown %",
    ]

    def color_signals(val):
        if val == "STRONG BUY":
            return "background-color: #059669; color: white; font-weight: bold;"
        elif val == "BUY":
            return "background-color: #10b981; color: white;"
        elif val == "SELL":
            return "background-color: #ef4444; color: white; font-weight: bold;"
        elif val == "REDUCE":
            return "background-color: #f59e0b; color: white;"
        elif val == "WAIT":
            return "background-color: #64748b; color: white;"
        return "background-color: #3b82f6; color: white;"

    styled = (
        display_df.style.format(
            {
                "Price": "{:.2f}",
                "24h %": "{:+.2f}%",
                "RSI(14)": "{:.1f}",
                "MACD Hist": "{:.4f}",
                "Annual Return": "{:+.2%}",
                "Annual Risk": "{:.2%}",
                "Score": "{:.1f}",
                "Suggested Weight %": "{:.1f}%",
                "1Y Max Drawdown %": "{:.2f}%",
            }
        )
        .background_gradient(subset=["Score"], cmap="YlGn")
        .background_gradient(subset=["Annual Return"], cmap="RdYlGn")
        .background_gradient(subset=["Annual Risk"], cmap="YlOrRd")
        .map(color_signals, subset=["Signal"])
        .set_properties(**{"text-align": "center"})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # =========================
    # RISK/RETURN SCATTER
    # =========================
    st.markdown('<div class="section-title">Risk / Return matrix</div>', unsafe_allow_html=True)

    scatter_df = analysis_df.reset_index().copy()
    scatter_df["Return_%"] = scatter_df["Return_Annualized"] * 100
    scatter_df["Risk_%"] = scatter_df["Volatility_Annualized"] * 100
    scatter_df["Size"] = scatter_df["Pro_Score"].clip(lower=1)

    fig_scatter = px.scatter(
        scatter_df,
        x="Risk_%",
        y="Return_%",
        color="Pro_Score",
        size="Size",
        hover_name="Asset",
        hover_data={
            "Risk_%": ":.2f",
            "Return_%": ":.2f",
            "Pro_Score": ":.1f",
            "Signal": True,
            "Size": False,
        },
        color_continuous_scale="Viridis",
        size_max=36,
    )
    fig_scatter.update_layout(
        height=520,
        template="plotly_white",
        xaxis_title="Risk (Annualized Volatility %)",
        yaxis_title="Return (Annualized %)",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # =========================
    # NORMALIZED PRICE EVOLUTION
    # =========================
    st.markdown('<div class="section-title">Normalized price evolution</div>', unsafe_allow_html=True)

    norm = close_df / close_df.iloc[0] * 100
    fig_line = go.Figure()
    for col in norm.columns:
        fig_line.add_trace(
            go.Scatter(
                x=norm.index,
                y=norm[col],
                mode="lines",
                name=col,
                hovertemplate=f"<b>{col}</b><br>%{{x|%Y-%m-%d}}<br>Base 100 → %{{y:.2f}}<extra></extra>",
            )
        )
    fig_line.update_layout(
        height=460,
        template="plotly_white",
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Base 100",
        legend=dict(orientation="h"),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # =========================
    # DEEP DIVE TOP ASSET
    # =========================
    top_asset = analysis_df.index[0]
    top_close = close_df[top_asset]

    top_rsi = rsi(top_close)
    top_macd, top_signal, top_hist = macd(top_close)
    bb_mid, bb_upper, bb_lower = bollinger(top_close)

    st.markdown(f'<div class="section-title">Deep dive: {top_asset}</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=top_close.index, y=top_close, mode="lines", name="Close"))
        fig_price.add_trace(go.Scatter(x=bb_mid.index, y=bb_mid, mode="lines", name="BB Mid"))
        fig_price.add_trace(go.Scatter(x=bb_upper.index, y=bb_upper, mode="lines", name="BB Upper"))
        fig_price.add_trace(go.Scatter(x=bb_lower.index, y=bb_lower, mode="lines", name="BB Lower"))
        fig_price.update_layout(
            template="plotly_white",
            height=430,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Date",
            yaxis_title="Price",
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig_price, use_container_width=True)

    with c2:
        fig_mom = go.Figure()
        fig_mom.add_trace(go.Scatter(x=top_macd.index, y=top_macd, mode="lines", name="MACD"))
        fig_mom.add_trace(go.Scatter(x=top_signal.index, y=top_signal, mode="lines", name="Signal"))
        fig_mom.add_trace(go.Bar(x=top_hist.index, y=top_hist, name="Histogram"))
        fig_mom.add_trace(go.Scatter(x=top_rsi.index, y=top_rsi, mode="lines", name="RSI"))
        fig_mom.update_layout(
            template="plotly_white",
            height=430,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Date",
            yaxis_title="Indicator value",
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig_mom, use_container_width=True)

    # =========================
    # SIGNALS
    # =========================
    st.markdown('<div class="section-title">Signals</div>', unsafe_allow_html=True)
    sig_counts = analysis_df["Signal"].value_counts().reset_index()
    sig_counts.columns = ["Signal", "Count"]

    fig_pie = px.pie(sig_counts, names="Signal", values="Count", hole=0.55)
    fig_pie.update_layout(template="plotly_white", height=380, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_pie, use_container_width=True)

    # =========================
    # NEWS
    # =========================
    st.markdown('<div class="section-title">Latest news</div>', unsafe_allow_html=True)
    news_items = fetch_news(top_asset if not is_crypto_ticker(top_asset) else top_asset.replace("-USD", ""))

    if news_items:
        for item in news_items[:3]:
            title = item.get("title") or item.get("headline") or "No title"
            link = item.get("article_url") or item.get("url") or item.get("link") or "#"
            publisher = "Unknown"
            if isinstance(item.get("publisher"), dict):
                publisher = item["publisher"].get("name", "Unknown")
            else:
                publisher = item.get("publisher", "Unknown")

            st.markdown(
                f"""
<div class="card" style="margin-bottom:0.8rem;">
    <div style="font-weight:800;font-size:1.05rem;margin-bottom:0.3rem;">
        <a href="{link}" target="_blank" style="text-decoration:none;color:#0f172a;">{title}</a>
    </div>
    <div style="color:#64748b;font-size:0.9rem;">Source: <b>{publisher}</b></div>
</div>
""",
                unsafe_allow_html=True,
            )
    else:
        st.info("No stock news returned for the top asset.")

    # =========================
    # EXPORT
    # =========================
    st.markdown('<div class="section-title">Export</div>', unsafe_allow_html=True)
    excel_bytes = to_excel(analysis_df, close_df)
    st.download_button(
        "📥 Download Excel",
        data=excel_bytes,
        file_name=f"FinTech_Opportunities_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    st.markdown(
        """
<div class="footer">
    Built with Streamlit • Massive API • yfinance • Plotly
</div>
""",
        unsafe_allow_html=True,
    )

else:
    st.info("Mettez vos tickers à gauche puis cliquez sur Launch Analysis.")