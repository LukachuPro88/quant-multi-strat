from typing import Literal
from pathlib import Path

from pandas import Timestamp

# Project root & directory paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# Create the tickers
TICKERS = ["AAPL", "CAT", "KO"]

# Create the strategy datatype
StrategyName = Literal[
    "buyandhold",
    "momentum",
    "movingaverage",
    "knnmomentum",
    "meanreversion",
    "xgboost",
    "nn",
]

STRATEGIES: list[StrategyName] = [
    "buyandhold",
    "momentum",
    "movingaverage",
    "knnmomentum",
    "meanreversion",
    "xgboost",
    "nn",
]

# Create the default initial simulation hyperparameters
INITIAL_CAPITAL = 1_000
INITIAL_SHARES = 0
INITIAL_START_DATE = Timestamp("2020-01-01")

# Visual parameters
PLOT_FIGSIZE = (10, 5)
PLOT_DPI = 300

# Signals
SIGNAL_BUY = 1
SIGNAL_SELL = 0
SIGNAL_HOLD = 2