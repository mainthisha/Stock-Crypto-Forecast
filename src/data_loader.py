"""Load OHLCV data: Yahoo Finance -> local CSV cache -> synthetic fallback."""
from pathlib import Path
import zlib
import numpy as np
import pandas as pd

import config


def _synthetic(ticker: str, start: str, end) -> pd.DataFrame:
    """Geometric-Brownian-motion price series with volatility regimes.
    Used only when there is no internet / no CSV, so the project always runs."""
    rng = np.random.default_rng(zlib.crc32(ticker.encode()))
    idx = pd.bdate_range(start=start, end=end or pd.Timestamp.today().normalize())
    n = len(idx)
    vol = 0.02 + 0.01 * np.sin(np.linspace(0, 12, n))
    ret = rng.normal(0.0004, vol)
    close = 100 * np.exp(np.cumsum(ret))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    volume = rng.integers(1_000_000, 5_000_000, n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close, "Volume": volume}, index=idx)


def load_data(ticker: str = config.DEFAULT_TICKER,
              start: str = config.START_DATE,
              end=config.END_DATE,
              source: str = "auto",
              csv_path: str = None) -> pd.DataFrame:
    """source: 'auto' | 'yahoo' | 'csv' | 'synthetic'"""
    cache = Path(config.DATA_DIR) / f"{ticker}.csv"
    df = None

    if source == "csv":
        df = pd.read_csv(csv_path or cache, index_col=0, parse_dates=True)
    elif source == "synthetic":
        df = _synthetic(ticker, start, end)
    else:  # auto / yahoo
        try:
            import yfinance as yf
            raw = yf.download(ticker, start=start, end=end,
                              auto_adjust=True, progress=False)
            if raw is not None and len(raw) > 200:
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
                print(f"[data] Downloaded {len(df)} rows for {ticker} from Yahoo Finance")
        except Exception as e:
            print(f"[data] Yahoo download failed: {e}")
        if df is None and source == "auto":
            if cache.exists():
                df = pd.read_csv(cache, index_col=0, parse_dates=True)
                print(f"[data] Using cached CSV {cache}")
            else:
                print("[data] No internet/CSV -> using SYNTHETIC data (demo only!)")
                df = _synthetic(ticker, start, end)
        elif df is None:
            raise RuntimeError("Could not download data from Yahoo Finance.")

    df = df.sort_index()
    df.index.name = "Date"
    Path(config.DATA_DIR).mkdir(exist_ok=True)
    if source != "csv":
        df.to_csv(cache)
    return df
