import pandas as pd

import src.config as config


def moving_average(
    df: pd.DataFrame,
    capital: float,
    shares: int = 0,
    start_date: pd.Timestamp | None = None,
    ma_range: int = 5,
) -> pd.DataFrame:
    """Simulates a moving average crossover trading strategy.

    Calculates an n-day moving average of opening prices. If the current price
    exceeds the moving average, a buy signal is generated. If the current price
    falls below or equals the moving average, a sell signal is triggered.

    When buying, the strategy deploys all available cash capital to purchase whole
    shares. When selling, the entire position is liquidated.

    Parameters
    ----------
    df : pd.DataFrame
        The input timeseries data. Must contain an 'Open' column and a DatetimeIndex.
    capital : float
        The initial cash amount available to deploy at the start of the simulation.
    shares : int, default=0
        The initial number of shares held. Updated during execution.
    start_date : pd.Timestamp | None, default=None
        The starting date for backtesting. If None, uses the first date in `df`.
    ma_range : int, default=5
        The rolling window lookback range (in trading days) for the moving average.

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
    >>> dates = pd.date_range("2020-01-01", periods=7, freq="D")
    >>> prices = [100.0, 100.0, 100.0, 100.0, 100.0, 110.0, 90.0]
    >>> df = pd.DataFrame({"Open": prices}, index=dates)
    >>> moving_average(df, capital=1000.0, ma_range=5)  # doctest: +NORMALIZE_WHITESPACE
                 Open    Cash  Shares  Portfolio Value
    Date
    2020-01-06  110.0     0.0       9            990.0
    2020-01-07   90.0   810.0       0            810.0
    """
    if df.empty:
        raise ValueError("Dataframe should not be empty.")

    if start_date is None:
        start_date = pd.Timestamp(df.index[0])

    dates = pd.DatetimeIndex(df.index)
    history: list[dict[str, float | int | pd.Timestamp]] = []

    # Simulation loop
    for i, (date, price) in enumerate(zip(dates, df["Open"])):
        # Skip iteration if date is prior to start_date or window is insufficient
        if date < start_date or i < ma_range:
            continue

        price = float(price)
        ma = _calculate_ma(df, i, ma_range)

        # Generate trading signal based on price relative to moving average
        signal = config.SIGNAL_BUY if price > ma else config.SIGNAL_SELL

        # Execute signals using configured signal constants
        if signal == config.SIGNAL_BUY and shares == 0:
            shares = int(capital / price)
            capital -= price * shares

        elif signal == config.SIGNAL_SELL and shares > 0:
            capital += price * shares
            shares = 0

        portfolio_value = capital + (price * shares)

        # Record portfolio state accounting
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


def _calculate_ma(
    df: pd.DataFrame,
    today: int,
    window: int,
) -> float:
    """Calculate trailing simple moving average for the given row index."""
    return float(df["Open"].iloc[today - window : today].mean())
