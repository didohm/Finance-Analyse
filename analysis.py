"""Technical indicators, scoring and signal generation."""
from __future__ import annotations

import numpy as np
import pandas as pd

SIGNAL_ORDER = ["STRONG BUY", "BUY", "HOLD", "REDUCE", "SELL", "WAIT"]
SIGNAL_COLORS: dict[str, str] = {
    "STRONG BUY": "#059669",
    "BUY": "#10b981",
    "HOLD": "#3b82f6",
    "REDUCE": "#f59e0b",
    "SELL": "#ef4444",
    "WAIT": "#64748b",
}
TRADING_DAYS_PER_YEAR = {"stock": 252, "crypto": 365}


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing."""
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = series.rolling(window).mean()
    std = series.rolling(window).std()
    return mid, mid + num_std * std, mid - num_std * std


def max_drawdown(series: pd.Series) -> float:
    if series.dropna().empty:
        return float("nan")
    s = series.dropna()
    return float(((s / s.cummax()) - 1).min())


def classify_signal(rsi_v: float, trend: str, macd_hist_v: float, price_vs_bb: str) -> str:
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


def _bb_state(price: float, upper: float, lower: float) -> str:
    if any(pd.isna(v) for v in (price, upper, lower)):
        return "inside"
    if price > upper:
        return "above"
    if price < lower:
        return "below"
    return "inside"


def analyze(
    history_map: dict[str, pd.DataFrame],
    live_map: dict[str, dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the comparison table plus an aligned close-price matrix.

    Each asset's metrics are computed from its own history so assets with
    shorter listing histories are not distorted by cross-alignment.
    """
    per_asset: list[dict] = []
    closes: dict[str, pd.Series] = {}

    for symbol, df in history_map.items():
        if df is None or df.empty or "Close" not in df.columns:
            continue
        s = pd.to_numeric(df["Close"], errors="coerce").dropna()
        s.index = pd.to_datetime(s.index)
        s = s.sort_index()
        if len(s) < 2:
            continue
        closes[symbol] = s
        per_asset.append(_asset_metrics(symbol, s, live_map.get(symbol)))

    if not per_asset:
        return pd.DataFrame(), pd.DataFrame()

    metrics = pd.DataFrame(per_asset).set_index("Asset")
    metrics["Trend"] = np.where(
        metrics["SMA20"].isna() | metrics["SMA50"].isna(),
        "Neutral",
        np.where(metrics["SMA20"] > metrics["SMA50"], "Bullish", "Bearish"),
    )
    metrics["BB_State"] = [
        _bb_state(p, u, l)
        for p, u, l in zip(metrics["Live_Price"], metrics["BB_Upper"], metrics["BB_Lower"])
    ]
    metrics["Signal"] = [
        classify_signal(r_, t_, m_, b_)
        for r_, t_, m_, b_ in zip(metrics["RSI14"], metrics["Trend"],
                                  metrics["MACD_Hist"], metrics["BB_State"])
    ]

    # Cross-asset ranks (1 = best within the current selection)
    n = len(metrics)
    ret_rank = metrics["Return_Annualized"].rank(pct=True)
    risk_rank = 1 - metrics["Volatility_Annualized"].rank(pct=True)
    momentum_rank = metrics["Momentum"].rank(pct=True)
    macd_rank = metrics["MACD_Hist"].rank(pct=True)
    rsi_quality = (1 - (metrics["RSI14"] - 50).abs() / 50).clip(lower=0, upper=1)

    metrics["Pro_Score"] = 100 * (
        0.28 * ret_rank.fillna(0.5)
        + 0.22 * risk_rank.fillna(0.5)
        + 0.18 * momentum_rank.fillna(0.5)
        + 0.16 * rsi_quality.fillna(0.5)
        + 0.16 * macd_rank.fillna(0.5)
    )
    if n == 1:
        metrics["Pro_Score"] = 60.0

    metrics["Confidence"] = pd.Categorical(
        pd.cut(metrics["Pro_Score"], bins=[-1, 40, 60, 75, 101],
               labels=["Low", "Medium", "High", "Very High"]),
        categories=["Low", "Medium", "High", "Very High"],
        ordered=True,
    )

    positive = metrics["Pro_Score"].clip(lower=0)
    total = positive.sum()
    metrics["Suggested_Weight_%"] = (
        100 * positive / total if total > 0 else 100.0 / n
    )

    metrics = metrics.sort_values(["Pro_Score", "Volatility_Annualized"],
                                  ascending=[False, True])
    metrics.insert(0, "Rank", range(1, n + 1))

    closes_df = pd.DataFrame(closes).sort_index()
    return metrics, closes_df


def _asset_metrics(symbol: str, close: pd.Series, live: dict | None) -> dict:
    crypto_key = "crypto" if "-" in symbol else "stock"
    ann_factor = TRADING_DAYS_PER_YEAR[crypto_key]

    returns = close.pct_change().dropna()
    last_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2]) if len(close) > 1 else last_close

    price, source = (None, "historical_close")
    if live and live.get("price") is not None:
        price, source = float(live["price"]), str(live.get("source", ""))
    else:
        price = last_close

    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    _, bb_up, bb_low = bollinger(close)
    macd_line, _macd_sig, macd_hist = macd(close)

    return {
        "Asset": symbol,
        "Live_Price": price,
        "Source": source,
        "Previous_Close": prev_close,
        "Variation_24h_%": ((price - prev_close) / prev_close) * 100,
        "Return_Annualized": returns.mean() * ann_factor,
        "Volatility_Annualized": returns.std() * np.sqrt(ann_factor),
        "SMA20": sma20,
        "SMA50": sma50,
        "RSI14": rsi(close).iloc[-1],
        "MACD": macd_line.iloc[-1],
        "MACD_Hist": macd_hist.iloc[-1],
        "BB_Upper": bb_up.iloc[-1],
        "BB_Lower": bb_low.iloc[-1],
        "Momentum": (sma20 / sma50) if pd.notna(sma20) and pd.notna(sma50) else np.nan,
        "Max_Drawdown_%": max_drawdown(close) * 100,
        "History_Bars": int(len(close)),
    }
