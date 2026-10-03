import pandas as pd

import config


def momentum(
    df: pd.DataFrame,
    capital: float,
    shares: int = 0,
    start_date: pd.Timestamp | None = None,
    momentum_range: int = 5,
) -> pd.DataFrame:
    """Simulates a momentum trading strategy based on lookback returns.

    Calculates the percentage return over a rolling lookback window (`momentum_range`).
    If the return is positive, a buy signal is generated. If negative or zero,
    a sell signal is triggered.

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
    momentum_range : int, default=5
        The lookback window (in trading days) for return calculation.

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
    >>> prices = [100.0, 101.0, 102.0, 103.0, 104.0, 106.0, 105.0]
    >>> df = pd.DataFrame({"Open": prices}, index=dates)
    >>> momentum(df, capital=1000.0, momentum_range=5)  # doctest: +NORMALIZE_WHITESPACE
                 Open   Cash  Shares  Portfolio Value
    Date
    2020-01-06  106.0   54.0       9           1008.0
    2020-01-07  105.0   54.0       9            999.0
    """
    if df.empty:
        raise ValueError("Dataframe should not be empty.")

    if start_date is None:
        start_date = pd.Timestamp(df.index[0])

    dates = pd.DatetimeIndex(df.index)
    history: list[dict[str, float | int | pd.Timestamp]] = []

    for i, (date, price) in enumerate(zip(dates, df["Open"])):
        # Skip until start date and lookback window are reached
        if date < start_date or i < momentum_range:
            continue

        price = float(price)
        price_lookback = float(df["Open"].iloc[i - momentum_range])
        period_return = (price / price_lookback) - 1.0

        # Calculate signal based on return sign
        signal = config.SIGNAL_BUY if period_return > 0 else config.SIGNAL_SELL

        # Execute signals using configured signal constants
        if signal == config.SIGNAL_BUY and shares == 0:
            shares = int(capital / price)
            capital -= price * shares

        elif signal == config.SIGNAL_SELL and shares > 0:
            capital += price * shares
            shares = 0

        # Calculate total portfolio valuation
        portfolio_value = capital + (price * shares)

        # State accounting
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
