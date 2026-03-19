import warnings
warnings.filterwarnings("ignore")

import os
import shutil
import datetime as dtt
from datetime import timedelta

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import yfinance as yf

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from logzero import logger

DATE_FMT = "%Y-%m-%d"

# Features used for all ML models
FEATURE_COLS = ['time_diff', 'Open', 'High', 'Low', 'Volume', 'MovAvg_5', 'MovAvg_10']
TIME_DIFF_IDX = 0  # index of 'time_diff' in FEATURE_COLS

ALGO_KEYS = ["linear", "gbr", "rfr"]
ALGO_LABELS = {
    "linear": "Linear Regression",
    "gbr":    "Gradient Boosting",
    "rfr":    "Random Forest",
}


class StockNow:
    """Fetch the latest price for a ticker via yfinance."""

    def __init__(self, ticker):
        self.ticker = ticker

    def get(self, item="Close"):
        try:
            hist = yf.Ticker(self.ticker).history(period="2d")
            if hist.empty:
                return None
            return float(hist[item].iloc[-1])
        except Exception as e:
            logger.warning(f"StockNow.get({item}) failed for {self.ticker}: {e}")
            return None


class StockDataGather:
    """Download OHLCV data from Yahoo Finance and save to CSV."""

    def __init__(self, ticker, start_date, end_date, data_dir="data"):
        self.ticker = ticker
        self.start_date = start_date  # YYYY-MM-DD
        self.end_date = end_date      # YYYY-MM-DD
        self.data_dir = data_dir

    def _ticker_dir(self):
        return os.path.join(self.data_dir, self.ticker)

    def csv_path(self):
        return os.path.join(self._ticker_dir(), f"{self.ticker}-stock-data.csv")

    def gather(self, force=False):
        path = self.csv_path()
        if os.path.exists(path) and not force:
            logger.info(f"Already exists: {path}")
            return True
        os.makedirs(self._ticker_dir(), exist_ok=True)
        try:
            data = yf.Ticker(self.ticker).history(start=self.start_date, end=self.end_date)
            if data.empty:
                logger.error(f"No data returned for {self.ticker}")
                return False
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            for ma in [1, 5, 10]:
                data[f"MovAvg_{ma}"] = data['Close'].rolling(window=ma, center=False).mean()
            data.index.name = 'Date'
            data.to_csv(path)
            logger.info(f"Wrote {path} ({len(data)} rows)")
            return True
        except Exception as e:
            logger.error(f"Failed to gather {self.ticker}: {e}")
            return False


class StockDataRead:
    """Load gathered CSV data and train ML models for price prediction."""

    def __init__(self, ticker, data_dir="data"):
        self.ticker = ticker
        self._path = os.path.join(data_dir, ticker, f"{ticker}-stock-data.csv")
        df = pd.read_csv(self._path, parse_dates=['Date'])
        df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_convert(None)
        df.sort_values('Date', inplace=True)
        df.dropna(subset=['Close'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        self.min_date = df['Date'].min()
        self.max_date = df['Date'].max()
        df['time_diff'] = (df['Date'] - self.min_date).dt.days

        self.max_price = df['Close'].max()
        self.min_price = df['Close'].min()
        self.max_price_date = df.loc[df['Close'].idxmax(), 'Date'].strftime(DATE_FMT)
        self.min_price_date = df.loc[df['Close'].idxmin(), 'Date'].strftime(DATE_FMT)

        self.stock_data = df
        self.model_ = None
        self.last_X_ = None
        self.rmse = None
        self.r2 = None
        self.mae = None

    def csv_path(self):
        return self._path

    def rows(self):
        return len(self.stock_data)

    def first_row(self):
        return self.stock_data.iloc[0]

    def last_row(self):
        return self.stock_data.iloc[-1]

    def summary(self, predict_date_str=None):
        print(f"Ticker:     {self.ticker}")
        print(f"Data range: {self.min_date.date()} → {self.max_date.date()} ({self.rows()} trading days)")
        print(f"Max price:  ${self.max_price:.2f} on {self.max_price_date}")
        print(f"Min price:  ${self.min_price:.2f} on {self.min_price_date}")
        if predict_date_str:
            pd_date = pd.to_datetime(predict_date_str)
            delta = (pd_date - self.max_date).days
            print(f"Predicting: {predict_date_str} ({delta} days ahead)")

    def _data_set(self):
        df = self.stock_data.dropna(subset=FEATURE_COLS + ['Close'])
        return df[FEATURE_COLS].values, df['Close'].values

    def _score(self, y_test, y_pred):
        self.rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        self.r2   = float(r2_score(y_test, y_pred))
        self.mae  = float(mean_absolute_error(y_test, y_pred))

    def analyze(self, algo):
        """Train model. algo: 'linear' | 'gbr' | 'rfr'"""
        X, y = self._data_set()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        if algo == "linear":
            model = LinearRegression()
        elif algo == "gbr":
            model = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
        elif algo == "rfr":
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            raise ValueError(f"Unknown algorithm: {algo}")
        model.fit(X_train, y_train)
        self._score(y_test, model.predict(X_test))
        self.model_  = model
        self.last_X_ = X_test[-1].copy()

    def predict(self, future_date_str):
        if self.model_ is None:
            raise RuntimeError("Call analyze() before predict()")
        td = (pd.to_datetime(future_date_str) - self.min_date).days
        X_pred = self.last_X_.copy()
        X_pred[TIME_DIFF_IDX] = td
        return float(self.model_.predict([X_pred])[0])


class StockDataPlot:
    """Generate matplotlib figures from gathered CSV data."""

    def __init__(self, csv_path):
        df = pd.read_csv(csv_path, parse_dates=['Date'])
        df.sort_values('Date', inplace=True)
        df['daily_pct'] = df['Close'].pct_change()
        df['volatility'] = df['daily_pct'].rolling(10).std() * np.sqrt(10)
        self.data   = df.set_index('Date')
        self.ticker = os.path.basename(os.path.dirname(csv_path))

    def closing(self):
        sns.set_style("darkgrid")
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(12, 7), gridspec_kw={'height_ratios': [3, 1]}, sharex=True
        )
        ax_top.plot(self.data.index, self.data['Close'], label='Close', linewidth=1.5)
        for ma, ls in [('MovAvg_10', '-'), ('MovAvg_5', '--')]:
            if ma in self.data.columns:
                ax_top.plot(self.data.index, self.data[ma], label=ma, alpha=0.7, linewidth=1, linestyle=ls)
        ax_top.set_title(f'{self.ticker} — Closing Price')
        ax_top.set_ylabel('Price ($)')
        ax_top.legend()
        ax_bot.bar(self.data.index, self.data['Volume'], color='#1565c0', alpha=0.95)
        ax_bot.set_ylabel('Volume')
        plt.tight_layout()
        return fig

    def volume(self):
        sns.set_style("darkgrid")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(self.data.index, self.data['Volume'], color='#1565c0', alpha=0.95)
        ax.set_title(f'{self.ticker} — Volume')
        ax.set_ylabel('Volume')
        plt.tight_layout()
        return fig

    def daily(self):
        sns.set_style("darkgrid")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(self.data.index, self.data['daily_pct'] * 100,
                color='tomato', label='Daily % Change', linewidth=0.8)
        ax.plot(self.data.index, self.data['volatility'] * 100,
                color='seagreen', label='Volatility', linewidth=1.2)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_title(f'{self.ticker} — Daily % Change & Volatility')
        ax.set_ylabel('%')
        ax.legend()
        plt.tight_layout()
        return fig

    def show(self, cmd="closing"):
        methods = {"closing": self.closing, "volume": self.volume, "daily": self.daily}
        if cmd not in methods:
            logger.error(f"Unknown chart type: {cmd}")
            return
        fig = methods[cmd]()
        plt.show()
        plt.close(fig)


class wfc_dgather:
    def __init__(self, tkrs, begin_date, end_date):
        self.tkrs  = tkrs
        self.begin = begin_date
        self.end   = end_date

    def gather(self, data_dir="data"):
        for t in self.tkrs:
            StockDataGather(t, self.begin, self.end, data_dir).gather()


class wfc_dpredict:
    def __init__(self, cfg, tkr_name, future_date):
        self.data_dir   = cfg.data_location()
        self.tkr_name   = tkr_name
        self.future_date = future_date  # YYYY-MM-DD string

    def predict(self):
        """Return dict: algo -> {price, rmse, r2, mae, label} plus 'ensemble'."""
        results = {}
        sdr = StockDataRead(self.tkr_name, self.data_dir)
        for algo in ALGO_KEYS:
            try:
                sdr.analyze(algo)
                results[algo] = {
                    'price': round(sdr.predict(self.future_date), 2),
                    'rmse':  round(sdr.rmse, 4),
                    'r2':    round(sdr.r2, 4),
                    'mae':   round(sdr.mae, 4),
                    'label': ALGO_LABELS[algo],
                }
            except Exception as e:
                logger.warning(f"{algo} failed for {self.tkr_name}: {e}")
        if results:
            ensemble = sum(r['price'] for r in results.values()) / len(results)
            results['ensemble'] = {'price': round(ensemble, 2), 'label': 'Ensemble'}
        return results


class wfc_dclean:
    def __init__(self, cfg):
        self.data_dir = cfg.data_location()

    def execute(self):
        if os.path.isdir(self.data_dir):
            shutil.rmtree(self.data_dir, ignore_errors=True)
            logger.info(f"Cleaned {self.data_dir}")


class wfc_visual:
    def __init__(self, cfg, tkr_name):
        self.data_dir = cfg.data_location()
        self.tkr_name = tkr_name

    def now(self, item="Close"):
        return StockNow(self.tkr_name).get(item)

    def _csv_path(self):
        return os.path.join(self.data_dir, self.tkr_name, f"{self.tkr_name}-stock-data.csv")

    def execute(self, cmd):
        StockDataPlot(self._csv_path()).show(cmd)

    def plot(self, cmd):
        """Return matplotlib Figure for use in Streamlit or notebooks."""
        sdp = StockDataPlot(self._csv_path())
        methods = {"closing": sdp.closing, "volume": sdp.volume, "daily": sdp.daily}
        fn = methods.get(cmd)
        return fn() if fn else None
