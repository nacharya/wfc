# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**wforecast (wfc)** is a stock prediction and portfolio management application. It pulls historical data from Yahoo Finance via `yfinance`, applies an ensemble of three ML models (Linear Regression, Gradient Boosting, Random Forest), summarizes news sentiment per ticker, and provides a 14-day (configurable) price forecast.

Three interfaces are provided:
- **CLI**: `app/wfc` (async Python script)
- **Web UI**: Streamlit app at `app/wfc_main.py`, served on port 9900
- **Notebook**: `app/wfc_doc.ipynb`

## Setup

```bash
cp app/dot_wfc.sample ~/.wfc   # create config
# edit ~/.wfc: set data_location, add portfolios and tickers
pip install -r requirements.txt
```

Date format throughout: **`YYYY-MM-DD`** (ISO 8601).

## Development Commands

### Local (no Docker)
```bash
make web        # streamlit run app/wfc_main.py --server.port 9900
make notebook   # jupyter notebook app/wfc_doc.ipynb
make cli        # prints CLI usage help

# CLI directly
cd app
./wfc list
./wfc ticker <portfolio> [list|add <TK>|del <TK>]
./wfc gather  <portfolio> <start_date> <end_date>
./wfc predict <ticker>    <YYYY-MM-DD | days>
./wfc news    [<ticker> | owned]
./wfc oa      [<days>]          # overall analysis, default 14 days
```

### Docker
```bash
make            # build image (wfc:latest)
make run        # run container on port 9900, mounts ./data
make stop       # stop and remove container
make shell      # exec into container
make logs       # view container logs
```

## Architecture

### Module Responsibilities

| File | Role |
|------|------|
| `wfc_config.py` | Loads `~/.wfc` JSON; manages `wfc_pfolio` (portfolio + ticker add/del) and `wfc_config` (owned positions, save) |
| `wfc_ds.py` | Data gathering (`StockDataGather`), reading/ML (`StockDataRead`), plotting (`StockDataPlot`), predict orchestration (`wfc_dpredict`) |
| `wfc_news.py` | `wfc_feed`: fetches news via `yf.Ticker.news`, scores with NLTK VADER, returns articles with title/summary/url/sentiment |
| `wfc_main.py` | Streamlit UI: Portfolio / Gather / Predict / News / Charts tabs |
| `app/wfc` | Async CLI dispatcher |
| `wfc_doc.ipynb` | Jupyter notebook: gather → visualize → predict → news → portfolio overview |
| `wfc_lstm.py` | Experimental LSTM (not integrated; requires separate TensorFlow install) |

### Data Flow

1. `gather` → `StockDataGather.gather()` downloads via `yf.Ticker.history()` → saves `data/<TICKER>/<TICKER>-stock-data.csv`
2. `predict` → `StockDataRead` loads CSV, adds `time_diff` feature → trains linear/GBR/RFR → `wfc_dpredict` returns dict with per-algo + ensemble results
3. `news` → `wfc_feed.get_news()` fetches `yf.Ticker.news` → VADER scores each headline → articles sorted by date
4. `oa` → iterates owned stocks → predict + news sentiment for each

### Prediction Return Format

`wfc_dpredict.predict()` returns a dict:
```python
{
  "linear":   {"price": 192.3, "rmse": 2.45, "r2": 0.987, "mae": 1.89, "label": "Linear Regression"},
  "gbr":      {"price": 195.1, ...},
  "rfr":      {"price": 193.9, ...},
  "ensemble": {"price": 193.8, "label": "Ensemble"},
}
```

### ML Features

`FEATURE_COLS = ['time_diff', 'Open', 'High', 'Low', 'Volume', 'MovAvg_5', 'MovAvg_10']`

`time_diff` is days since the first date in the dataset. Prediction works by taking the last test-set row and substituting the `time_diff` for the target future date.

### Config Format (`~/.wfc`)

```json
{
  "basename": "user",
  "data_location": "data/",
  "pfolios": [{"NAME": "P1", "TICKERS": ["AAPL"], "RELATED": [], "EXCHG": "NSDQ"}],
  "purchased": [{"ticker": "AAPL", "ticker_purchase": [{"2023-01-15": [50, 135.50]}]}]
}
```

`wfc_pfolio.tkrs` is a **reference** into `cfg_data`, so in-place `.append()`/`.remove()` is reflected automatically when `cfg.save()` is called.

### Streamlit Notes

- `matplotlib.use('Agg')` is set at the top of `wfc_ds.py`; plot methods return `Figure` objects for `st.pyplot(fig)`
- After any config mutation, call `save_config(cfg)` then `st.rerun()`
- `IMAGES_DIR` uses `__file__`-relative path so it works from any working directory

### News Article Fields

Each article in `wfc_feed.articles`:
`title`, `summary`, `url`, `publisher`, `published` (formatted string), `pos`, `neg`, `neu`, `compound`
