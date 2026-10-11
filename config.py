"""Central configuration for the LSTM vs GRU forecasting project."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

# Assets (Yahoo Finance symbols). Add your own here.
STOCKS = ["AAPL", "TSLA", "RELIANCE.NS", "TCS.NS"]
CRYPTOS = ["BTC-USD", "ETH-USD"]
DEFAULT_TICKER = "BTC-USD"

START_DATE = "2018-01-01"
END_DATE = None            # None = today

# Data / model hyper-parameters
WINDOW = 60                # look-back days fed to the network
TRAIN_RATIO = 0.80
VAL_SPLIT = 0.10           # fraction of training data used for validation
FEATURES = ["Close"]       # e.g. ["Open","High","Low","Close","Volume"]
TARGET = "Close"

UNITS = [64, 32]           # recurrent units in stacked layers
DROPOUT = 0.2
LEARNING_RATE = 1e-3
EPOCHS = 50
BATCH_SIZE = 32
PATIENCE = 8               # early stopping patience
FORECAST_DAYS = 30
SEED = 42
