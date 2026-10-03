import math
import pandas as pd

import src.config as config


def mean_reversion(
    df: pd.DataFrame,
    capital: float,
    shares: int,
    start_date: pd.Timestamp,
    ma_range: int = 20,
    threshold: float = 2.0,
) -> pd.DataFrame:
    """Simulates a mean reversion trading strategy using Bollinger/Z-score bounds.

    The strategy calculates a rolling average and measures how much the
    current price varies from that average in standard deviations. If the price
    is sufficiently below the average, a buy signal is generated. If the price
    is sufficiently above the average, a sell signal is generated. Otherwise,
    the strategy holds its current position.

    Parameters
    ----------
    df : pd.DataFrame
        The input timeseries data. Must contain 'Open' and 'Close' columns and a
        DatetimeIndex.
    capital : float
        The initial cash amount available to deploy at the start of the
        simulation.
    shares : int
        The initial number of shares held at the start of the simulation.
    start_date : pd.Timestamp
        The starting date for backtesting simulation.
    ma_range : int, default=20
        The number of trading days used to calculate the rolling average.
    threshold : float, default=2.0
        The minimum number of standard deviations required to generate an action
        signal. A value of 2.0 represents 2 standard deviations.

    Returns
    -------
    pd.DataFrame
        The backtest execution history indexed by date.

        * Index : DatetimeIndex
            Trading dates corresponding to the active backtest window.
        * Open : float
            Opening stock price for the given date.
        * Cash : float
            Unallocated cash balance remaining in the portfolio.
        * Shares : int
            Total quantity of stock shares held.
        * Portfolio Value : float
            Total valuation of the portfolio (Cash + Shares * Open).

    Examples
    --------
    >>> import pandas as pd
    >>> dates = pd.date_range("2020-01-01", periods=10, freq="D")
    >>> prices = [100.0] * 8 + [80.0, 100.0]
    >>> df = pd.DataFrame({"Open": prices, "Close": prices}, index=dates)
    >>> mean_reversion(df, capital=1000.0, shares=0, start_date=dates[0], ma_range=3, threshold=1.0)  # doctest: +NORMALIZE_WHITESPACE
                 Open    Cash  Shares  Portfolio Value
    Date
    2020-01-10  100.0  1000.0       0           1000.0
    """
    if df.empty:
        raise ValueError("Dataframe should not be empty.")

    dates = pd.DatetimeIndex(df.index)
    rolling_average = df["Close"].rolling(ma_range).mean().shift(1)

    rolling_std = (
        df["Close"].shift(1).rolling(ma_range).apply(_get_standard_deviation, raw=False)
    )

    history: list[dict[str, float | int | pd.Timestamp]] = []

    for date, price, average, std in zip(
        dates,
        df["Open"],
        rolling_average,
        rolling_std,
    ):
        if date < start_date:
            continue

        if pd.isna(average) or pd.isna(std) or std == 0:
            continue

        price = float(price)
        variation = (price - average) / std

        # Generate signal using semantic constants
        if variation <= -threshold:
            signal = config.SIGNAL_BUY
        elif variation >= threshold:
            signal = config.SIGNAL_SELL
        else:
            signal = config.SIGNAL_HOLD

        # Portfolio accounting using semantic constants
        if signal == config.SIGNAL_BUY and shares == 0:
            shares = int(capital / price)
            capital -= price * shares

        elif signal == config.SIGNAL_SELL and shares > 0:
            capital += price * shares
            shares = 0

        portfolio_value = capital + (price * shares)

        history.append(
            {
                "Date": date,
                "Open": price,
                "Cash": capital,
                "Shares": shares,
                "Portfolio Value": portfolio_value,
            }
        )

    return pd.DataFrame(history).set_index("Date")


# --- Helper Functions ---


def _get_standard_deviation(series: pd.Series) -> float:
    """Calculates sample standard deviation for a given pandas Series."""
    mean = series.mean()
    sum_squared_diff = 0.0

    for value in series:
        sum_squared_diff += math.pow(value - mean, 2)

    variance = sum_squared_diff / (len(series) - 1)
    return math.sqrt(variance)
