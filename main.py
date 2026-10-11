"""Run the full LSTM vs GRU comparison from the command line.

Examples
    python main.py                                  # BTC-USD, live data
    python main.py --ticker AAPL --epochs 30
    python main.py --ticker TSLA --source csv --csv data/TSLA.csv
    python main.py --source synthetic --epochs 5    # offline smoke test
"""
import argparse, json
import pandas as pd

import config
from src import data_loader, preprocessing, train, evaluate, visualize


def run(ticker, source="auto", epochs=config.EPOCHS, window=config.WINDOW,
        csv=None, forecast_days=config.FORECAST_DAYS, features=None, verbose=2):
    features = features or config.FEATURES
    out = config.RESULTS_DIR / ticker
    out.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(exist_ok=True)

    df = data_loader.load_data(ticker, source=source, csv_path=csv)
    print(f"[data] {len(df)} rows  {df.index[0].date()} -> {df.index[-1].date()}")
    visualize.plot_history(df, ticker, out / "price_history.png")

    ds = preprocessing.prepare(df, window=window, features=features)
    print(f"[prep] train={len(ds.X_train)}  test={len(ds.X_test)}  window={window}")

    rows, histories, preds, forecasts, actual = {}, {}, {}, {}, None
    for kind in ["LSTM", "GRU"]:
        print(f"\n=== Training {kind} ===")
        model, hist, secs = train.train_model(kind, ds, epochs=epochs, verbose=verbose)
        actual, pred = evaluate.predict_test(model, ds)
        m = evaluate.metrics(actual, pred)
        m["Params"] = int(model.count_params())
        m["Train time(s)"] = round(secs, 1)
        m["Epochs run"] = len(hist["loss"])
        rows[kind], histories[kind], preds[kind] = m, hist, pred
        forecasts[kind] = evaluate.forecast_future(model, ds, forecast_days)
        model.save(config.MODELS_DIR / f"{ticker}_{kind}.keras")

    mdf = pd.DataFrame(rows).T
    mdf.to_csv(out / "metrics.csv")
    pd.DataFrame({"Date": ds.test_dates[-len(actual):], "Actual": actual,
                  "LSTM": preds["LSTM"], "GRU": preds["GRU"]}
                 ).to_csv(out / "predictions.csv", index=False)

    dates = ds.test_dates[-len(actual):]
    visualize.plot_predictions(dates, actual, preds, ticker, out / "predictions.png")
    visualize.plot_loss(histories, ticker, out / "loss_curves.png")
    visualize.plot_metric_bars(mdf, ticker, out / "metric_comparison.png")
    visualize.plot_forecast(df, forecasts, ticker, out / "forecast.png")

    winner = mdf["RMSE"].astype(float).idxmin()
    summary = {"ticker": ticker, "best_model_by_RMSE": winner,
               "metrics": mdf.round(4).to_dict(orient="index")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + mdf.round(4).to_string())
    print(f"\nBest model by RMSE: {winner}\nOutputs saved in: {out}")
    return mdf


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=config.DEFAULT_TICKER)
    ap.add_argument("--source", default="auto", choices=["auto", "yahoo", "csv", "synthetic"])
    ap.add_argument("--csv", default=None)
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--window", type=int, default=config.WINDOW)
    ap.add_argument("--forecast-days", type=int, default=config.FORECAST_DAYS)
    ap.add_argument("--multivariate", action="store_true",
                    help="use Open/High/Low/Close/Volume instead of Close only")
    a = ap.parse_args()
    feats = ["Open", "High", "Low", "Close", "Volume"] if a.multivariate else None
    run(a.ticker, a.source, a.epochs, a.window, a.csv, a.forecast_days, feats)
