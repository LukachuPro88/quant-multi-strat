from typing import Literal

import pandas as pd

import src.config as config
import src.analysis.metrics as metrics
import src.strategies.buy_and_hold as bah
import src.strategies.mean_reversion as mr
import src.strategies.momentum as mom
import src.strategies.momentum_knn as knn
import src.strategies.moving_average as ma
import src.strategies.xgboost as xgb
import src.strategies.nn as nn


def backtest(
    df: pd.DataFrame,
    strategy: config.StrategyName = "buyandhold",
    capital: float = 1_000,
    shares: int = 0,
    start_date: pd.Timestamp = config.INITIAL_START_DATE,
) -> pd.DataFrame:
    """Handles backtesting for the strategies using a DatetimeIndex."""
    if df.empty:
        raise ValueError("Dataframe should not be empty.")

    # Make sure that the strategy name is always lowercase
    strategy_name = strategy.lower()

    match strategy_name:
        case "buyandhold":
            df_backtest = df[df.index >= start_date].copy()
            result = bah.buy_and_hold(df_backtest, capital, shares)

        case "momentum":
            df_backtest = df[df.index >= start_date].copy()
            result = mom.momentum(df_backtest, capital, shares)

        case "movingaverage":
            df_backtest = df[df.index >= start_date].copy()
            result = ma.moving_average(df_backtest, capital, shares)

        case "knnmomentum":
            df_backtest = df[df.index >= start_date].copy()
            result = knn.knn_momentum(df_backtest, capital, shares, start_date)

        case "meanreversion":
            result = mr.mean_reversion(df, capital, shares, start_date)

        case "xgboost":
            df_backtest = df[df.index >= start_date].copy()
            result = xgb.xgboost(df_backtest, capital, shares, start_date)

        case "nn":
            df_backtest = df[df.index >= start_date].copy()
            result = nn.nn(df_backtest, capital, shares, start_date)

        case _:
            raise ValueError("Invalid strategy given.")

    # Slice result to out-of-sample window via index
    result = result[result.index >= start_date].copy()

    metrics.print_metrics(
        strategy=strategy_name,
        final_capital=result["Cash"].iloc[-1],
        final_price=result["Open"].iloc[-1],
        final_shares=result["Shares"].iloc[-1],
        final_portfolio_value=result["Portfolio Value"].iloc[-1],
    )

    return result
