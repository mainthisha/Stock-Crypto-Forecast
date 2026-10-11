# Report / Viva Outline

**Title:** Stock Price / Cryptocurrency Trend Forecasting Using LSTM and GRU: A Comparative Study

## 1. Abstract
Compares LSTM and GRU recurrent networks for next-day closing-price prediction of stocks and cryptocurrencies, using RMSE, MAE, MAPE, R² and directional accuracy, plus training time and parameter count.

## 2. Problem statement
Financial time series are noisy and non-linear. Traditional models (ARIMA) assume linearity. Recurrent networks can learn long-range temporal dependencies.

## 3. Why LSTM / GRU
| | LSTM | GRU |
|---|---|---|
| Gates | input, forget, output (3) + cell state | update, reset (2) |
| Parameters | more | ~25% fewer |
| Training speed | slower | usually faster |
| Long dependencies | very strong | strong, sometimes equal |

## 4. Methodology (maps to the code)
1. Data collection – Yahoo Finance via `yfinance` (`src/data_loader.py`)
2. Pre-processing – Min-Max scaling fit on train only, 60-day sliding window (`src/preprocessing.py`)
3. Model – 2 stacked recurrent layers (64, 32) + Dropout 0.2 + Dense (`src/models.py`)
4. Training – Adam, MSE loss, early stopping, LR scheduling (`src/train.py`)
5. Evaluation – chronological 80/20 split, metrics on inverse-scaled prices (`src/evaluate.py`)
6. Forecast – recursive 30-day forecast
7. Visualisation – `src/visualize.py`, dashboard `app.py`

## 5. Experiments to report
- BTC-USD, ETH-USD, AAPL, TSLA (run `python main.py --ticker X` for each)
- Univariate (Close) vs multivariate (`--multivariate`)
- Window sizes 30 / 60 / 90 (`--window`)
- Collect each `results/<ticker>/metrics.csv` into one comparison table

## 6. Expected discussion points
- GRU often matches LSTM accuracy with fewer parameters and less training time
- Crypto is more volatile → higher error than large-cap stocks
- Low R²/DirAcc near 50% means price direction is hard to predict; the model mostly tracks the trend
- Limitations: no news/sentiment, EMH, recursive forecast error accumulation
- Future work: attention, Transformers, sentiment features, hyper-parameter tuning

## 7. Viva Q&A
- *Why scale only on training data?* Prevents data leakage.
- *Why not shuffle?* Time order matters.
- *Why RMSE and MAPE?* RMSE penalises large errors; MAPE is scale-independent.
- *Why does the forecast flatten over time?* Recursive prediction feeds on its own estimates.
