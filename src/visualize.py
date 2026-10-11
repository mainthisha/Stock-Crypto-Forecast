"""All matplotlib figures (saved as PNGs)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = ["tab:blue", "tab:red"]


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_history(df, ticker, path):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df.index, df["Close"], lw=1.2)
    ax.set(title=f"{ticker} closing price", xlabel="Date", ylabel="Price")
    ax.grid(alpha=.3); _save(fig, path)


def plot_predictions(dates, actual, preds: dict, ticker, path):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(dates, actual, label="Actual", color="black", lw=1.4)
    for (name, p), c in zip(preds.items(), COLORS):
        ax.plot(dates, p, label=f"{name} prediction", color=c, lw=1.1, alpha=.85)
    ax.set(title=f"{ticker}: actual vs predicted (test set)", xlabel="Date", ylabel="Price")
    ax.legend(); ax.grid(alpha=.3); _save(fig, path)


def plot_loss(histories: dict, ticker, path):
    fig, axes = plt.subplots(1, len(histories), figsize=(11, 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, h) in zip(axes, histories.items()):
        ax.plot(h["loss"], label="train"); ax.plot(h["val_loss"], label="validation")
        ax.set(title=f"{name} loss", xlabel="Epoch", ylabel="MSE")
        ax.legend(); ax.grid(alpha=.3)
    fig.suptitle(f"{ticker}: training curves"); _save(fig, path)


def plot_metric_bars(metrics_df: pd.DataFrame, ticker, path):
    cols = ["RMSE", "MAE", "MAPE(%)", "DirAcc(%)"]
    fig, axes = plt.subplots(1, len(cols), figsize=(12, 3.8))
    for ax, c in zip(axes, cols):
        ax.bar(metrics_df.index, metrics_df[c].astype(float), color=COLORS)
        ax.set_title(c); ax.grid(alpha=.3, axis="y")
    fig.suptitle(f"{ticker}: LSTM vs GRU"); _save(fig, path)


def plot_forecast(df, forecasts: dict, ticker, path, tail=120):
    fig, ax = plt.subplots(figsize=(11, 5))
    hist = df["Close"].iloc[-tail:]
    ax.plot(hist.index, hist.values, color="black", label="History")
    n = len(next(iter(forecasts.values())))
    future_idx = pd.bdate_range(df.index[-1] + pd.Timedelta(days=1), periods=n)
    for (name, f), c in zip(forecasts.items(), COLORS):
        ax.plot(future_idx, f, "--", color=c, label=f"{name} forecast")
    ax.set(title=f"{ticker}: {n}-day forecast", xlabel="Date", ylabel="Price")
    ax.legend(); ax.grid(alpha=.3); _save(fig, path)
