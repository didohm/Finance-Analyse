# 📊 FinScope — Financial Opportunity Analyzer

A production-ready Streamlit dashboard that ranks stocks and crypto by a composite
technical score, generates buy/sell signals, and visualizes risk, correlation,
momentum and news — all on **real market data**.

## Features

- **Real-time data** — Massive API when `MASSIVE_API_KEY` is configured, with
  automatic Yahoo Finance fallback (via `yfinance`). Live prices are cached for
  ~45 s; daily history for ~1 h.
- **Composite scoring** — blends annualized return, risk, momentum (SMA20/SMA50),
  RSI(14) positioning and MACD strength into a 0–100 score with confidence bands.
- **Signals** — STRONG BUY / BUY / HOLD / REDUCE / SELL / WAIT derived from RSI,
  trend, MACD histogram and Bollinger Band state. Consistent color coding across
  the table, cards and charts.
- **Analytics** — risk/return matrix with average quadrants, rebased relative
  performance, correlation heatmap, suggested portfolio weights.
- **Deep dive** — per-asset tabs: price + Bollinger Bands + volume, MACD,
  RSI with overbought/oversold zones.
- **News** — recent headlines for any analyzed asset.
- **Exports** — one-click Excel workbook (opportunities + historical closes) or CSV.

## Getting started

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens with a default watchlist; change tickers or the period in the
sidebar and everything updates automatically.

### Optional: Massive API

Add the key either as an environment variable or `.streamlit/secrets.toml`:

```toml
MASSIVE_API_KEY = "your_key_here"
```

Without it the app runs fully on Yahoo Finance data.

## Project structure

```
app.py        # UI: layout, styling, charts, exports
data.py       # Data layer: Massive + yfinance clients, caching, parallel fetching
analysis.py   # Indicators (RSI/MACD/Bollinger), scoring, signals
.streamlit/   # Theme configuration
.github/      # Keep-alive workflow for Streamlit Cloud
```

## Notes

- Up to 12 assets per analysis to keep responses fast.
- Assets with short listing histories are scored from their own data, so they
  remain comparable after rebasing.
- ⚠️ For research and education only — not financial advice.
