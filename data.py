"""Data access layer: Massive API (primary) with yfinance fallback."""
from __future__ import annotations

import os
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import streamlit as st
import yfinance as yf

# Keep server logs clean: fetch failures are surfaced in the UI instead.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

TICKER_PATTERN = re.compile(r"^[A-Za-z0-9.\-=^]{1,15}$")
MAX_ASSETS = 12
_REQUEST_TIMEOUT = (3.05, 12)

# label -> number of calendar days of history
PERIODS: dict[str, int] = {
    "1M": 31,
    "3M": 92,
    "6M": 183,
    "YTD": -1,  # resolved at runtime
    "1Y": 365,
    "2Y": 730,
    "5Y": 1825,
}

# label -> yfinance period string
PERIOD_TO_YF: dict[str, str] = {
    "1M": "1mo",
    "3M": "3mo",
    "6M": "6mo",
    "YTD": "ytd",
    "1Y": "1y",
    "2Y": "2y",
    "5Y": "5y",
}


def get_secret(name: str, default: str = "") -> str:
    """Read a secret from Streamlit secrets or environment variables."""
    try:
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return os.getenv(name, default).strip()


def parse_tickers(text: str) -> tuple[list[str], list[str]]:
    """Split comma-separated input into (valid, invalid) ticker lists."""
    raw = [t.strip().upper() for t in text.replace(";", ",").split(",")]
    seen: set[str] = set()
    valid: list[str] = []
    invalid: list[str] = []
    for t in raw:
        if not t:
            continue
        if not TICKER_PATTERN.match(t):
            if t not in invalid:
                invalid.append(t)
            continue
        if t not in seen:
            seen.add(t)
            valid.append(t)
    return valid[:MAX_ASSETS], invalid


def is_crypto(ticker: str) -> bool:
    base, _, quote = ticker.rpartition("-")
    return bool(base) and quote in {"USD", "USDT"}


def _crypto_pair(ticker: str) -> tuple[str, str]:
    return tuple(ticker.rsplit("-", 1))


def _days_for_period(period: str) -> int:
    days = PERIODS.get(period, 365)
    if days == -1:  # YTD
        today = datetime.now(timezone.utc).date()
        return max((today - today.replace(month=1, day=1)).days, 5)
    return days


# -------------------------
# Massive API
# -------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def massive_history(
    symbol: str, start: str, end: str, api_key: str = ""
) -> pd.DataFrame | None:
    """Daily OHLCV aggregates from Massive (formerly Polygon-compatible)."""
    if not api_key:
        return None
    try:
        r = requests.get(
            f"https://api.massive.com/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}",
            params={"adjusted": "true", "sort": "asc", "limit": "50000", "apiKey": api_key},
            timeout=_REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        results = r.json().get("results") or []
    except Exception:
        return None

    if not results:
        return None
    df = pd.DataFrame(results)
    if "t" not in df.columns or "c" not in df.columns:
        return None

    df["Date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(None)
    df = df.set_index("Date").sort_index()
    rename = {"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"}
    cols = [c for c in rename if c in df.columns]
    return df[cols].rename(columns=rename)


@st.cache_data(ttl=45, show_spinner=False)
def massive_last_price(symbol: str, is_crypto_pair: bool, api_key: str = "") -> float | None:
    """Latest trade price for stocks or last crypto price, via Massive."""
    if not api_key:
        return None
    try:
        if is_crypto_pair:
            base, quote = _crypto_pair(symbol)
            url = f"https://api.massive.com/v1/last/crypto/{base}/{quote}"
            payload = requests.get(url, params={"apiKey": api_key},
                                   timeout=(3.05, 8)).json()
            price = (payload.get("last") or {}).get("price")
        else:
            url = f"https://api.massive.com/v2/last/trade/{symbol}"
            payload = requests.get(url, params={"apiKey": api_key},
                                   timeout=(3.05, 8)).json()
            price = (payload.get("results") or {}).get("p")
        return float(price) if price is not None else None
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def massive_news(symbol: str, api_key: str = "", limit: int = 6) -> list[dict]:
    """Recent news articles for a ticker via Massive."""
    if not api_key:
        return []
    try:
        r = requests.get(
            "https://api.massive.com/v2/reference/news",
            params={"ticker": symbol, "limit": limit, "order": "desc", "apiKey": api_key},
            timeout=_REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        results = r.json().get("results")
        return results if isinstance(results, list) else []
    except Exception:
        return []


# -------------------------
# yfinance fallback
# -------------------------
@st.cache_data(ttl=900, show_spinner=False)
def yf_history(symbol: str, period: str = "1y") -> pd.DataFrame | None:
    try:
        raw = yf.download(symbol, period=period, interval="1d",
                          progress=False, auto_adjust=True)
    except Exception:
        return None
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    if isinstance(raw.index, pd.DatetimeIndex) and raw.index.tz is not None:
        raw.index = raw.index.tz_convert(None)
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in raw.columns]
    if "Close" not in cols:
        return None
    out = raw[cols].apply(pd.to_numeric, errors="coerce").dropna(subset=["Close"])
    return out if not out.empty else None


@st.cache_data(ttl=60, show_spinner=False)
def yf_last_price(symbol: str) -> float | None:
    try:
        fast = getattr(yf.Ticker(symbol), "fast_info", None)
        price = fast.get("lastPrice") or fast.get("regularMarketPrice") if fast else None
        if price:
            return float(price)
    except Exception:
        pass
    try:
        hist = yf.Ticker(symbol).history(period="5d", interval="1d")
        if hist is not None and not hist.empty and "Close" in hist.columns:
            return float(hist["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return None


# -------------------------
# Unified fetching
# -------------------------
def fetch_history(symbol: str, crypto: bool, period: str, api_key: str = "") -> pd.DataFrame | None:
    """Daily history: Massive for stocks when available, yfinance otherwise."""
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=_days_for_period(period))

    df = None
    if not crypto and api_key:
        df = massive_history(symbol, start.isoformat(), end.isoformat(), api_key)
    if df is None or df.empty:
        df = yf_history(symbol, period=PERIOD_TO_YF[period])
    return df if df is not None and not df.empty else None


def fetch_live_price(symbol: str, crypto: bool, api_key: str = "") -> tuple[float | None, str]:
    """Live price: Massive first, then yfinance. Returns (price, source)."""
    price = massive_last_price(symbol, crypto, api_key)
    if price is not None:
        return price, "massive"
    price = yf_last_price(symbol)
    if price is not None:
        return price, "yahoo"
    return None, "unavailable"


def fetch_universe(
    tickers: list[str], period: str, api_key: str = ""
) -> tuple[dict[str, pd.DataFrame], dict[str, dict], dict[str, str]]:
    """Fetch history + live price for all tickers concurrently.

    Returns (history_map, live_map, issues) where issues maps ticker -> reason.
    """
    history_map: dict[str, pd.DataFrame] = {}
    live_map: dict[str, dict] = {}
    issues: dict[str, str] = {}

    def work(symbol: str):
        crypto = is_crypto(symbol)
        hist = fetch_history(symbol, crypto, period, api_key)
        price, src = fetch_live_price(symbol, crypto, api_key)
        return symbol, crypto, hist, price, src

    workers = min(8, max(1, len(tickers)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, t): t for t in tickers}
        for fut in as_completed(futures):
            symbol, _crypto, hist, price, src = fut.result()
            if hist is None:
                issues[symbol] = "no historical data found"
                continue
            history_map[symbol] = hist
            if price is not None:
                live_map[symbol] = {"price": price, "source": src}
            else:
                issues[symbol] = "live price unavailable (using last close)"

    return history_map, live_map, issues
