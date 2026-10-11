# 📈 Stock / Cryptocurrency Trend Forecasting Using LSTM and GRU — A Comparative Study

Complete, runnable deep-learning project. It downloads historical prices, trains an **LSTM** and a **GRU** with identical architecture, compares them on error metrics, training time and model size, and forecasts the next 30 days. Includes a **Streamlit dashboard**.

## Project structure
```
stock-crypto-lstm-gru/
├── main.py              # CLI: train + compare + save plots/metrics
├── app.py               # Streamlit dashboard
├── config.py            # all settings (tickers, window, epochs, units ...)
├── requirements.txt
├── src/
│   ├── data_loader.py   # Yahoo Finance -> CSV cache -> synthetic fallback
│   ├── preprocessing.py # scaling, sliding windows, train/test split
│   ├── models.py        # LSTM & GRU builders
│   ├── train.py         # training with early stopping
│   ├── evaluate.py      # RMSE, MAE, MAPE, R2, directional accuracy, forecast
│   └── visualize.py     # all charts
├── tests/test_pipeline.py
├── data/                # downloaded CSVs are cached here
├── models/              # trained .keras models are saved here
├── results/<TICKER>/    # metrics.csv, predictions.csv, PNG plots, summary.json
└── docs/REPORT_OUTLINE.md   # report / viva help
```

## Setup (Python 3.9 – 3.12)
```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        Mac/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python main.py                              # Bitcoin (BTC-USD), live data
python main.py --ticker AAPL --epochs 30
python main.py --ticker ETH-USD --window 90
python main.py --ticker RELIANCE.NS         # Indian stocks use .NS
python main.py --ticker TSLA --multivariate # Open/High/Low/Close/Volume
python main.py --source csv --csv data/my_prices.csv --ticker MYDATA
python main.py --source synthetic --epochs 3   # offline smoke test (fake data!)

streamlit run app.py                        # interactive dashboard
python -m pytest tests                      # unit tests
```
Your own CSV needs columns: `Date, Open, High, Low, Close, Volume`.

## Outputs (in `results/<TICKER>/`)
| File | Meaning |
|---|---|
| `metrics.csv` | RMSE, MAE, MAPE, R², directional accuracy, params, train time |
| `predictions.png` | actual vs LSTM vs GRU on the test set |
| `loss_curves.png` | train / validation loss |
| `metric_comparison.png` | bar chart comparison |
| `forecast.png` | 30-day future forecast |
| `predictions.csv`, `summary.json` | raw predictions and winner |

## Method
1. Daily OHLCV from Yahoo Finance (`yfinance`).
2. Min-Max scaling fitted **only on the training split** (no leakage).
3. 60-day sliding window → predict next-day Close.
4. Chronological 80/20 split (no shuffling).
5. Model: 2 stacked recurrent layers (64, 32) → Dropout 0.2 → Dense(16) → Dense(1). LSTM and GRU differ only in the cell.
6. Adam, MSE loss, EarlyStopping, ReduceLROnPlateau, fixed seed.
7. Metrics are computed in real price units after inverse scaling.

## Notes
- If there is no internet, `auto` mode uses a cached CSV, otherwise **synthetic demo data**. Do not report synthetic results — use real data for your submission.
- Change tickers, window, units and epochs in `config.py`.
- Results vary slightly between runs/machines. Run several tickers and discuss them in your report.
- Educational project — **not financial advice**.
