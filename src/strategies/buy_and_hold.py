import pandas as pd


def buy_and_hold(
    df: pd.DataFrame,
    capital: float,
    shares: int,
) -> pd.DataFrame:
    """Simulate a baseline Buy and Hold trading strategy.

    The strategy allocates available capital to purchase shares at the opening
    price on the first available trading day. The position is held throughout
    the entire timeseries and liquidated at the opening price of the final day.

    Parameters
    ----------
    df : pd.DataFrame
        The input timeseries data. Must contain an 'Open' column and a
        DatetimeIndex.
    capital : float
        The initial cash amount available to deploy at the start of the
        simulation.
    shares : int
        The initial number of shares held. Updated during the simulation.

    Returns
    -------
    pd.DataFrame
        The backtest execution history indexed by date.

        * Index : DatetimeIndex
            Trading dates corresponding to the dataset.
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
    >>> dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    >>> df = pd.DataFrame({"Open": [100.0, 110.0, 105.0]}, index=dates)
    >>> result = buy_and_hold(df, capital=1000.0, shares=0)
    >>> result["Shares"].tolist()
    [10, 10, 10]
    >>> result["Cash"].tolist()
    [0.0, 0.0, 0.0]
    >>> result["Portfolio Value"].tolist()
    [1000.0, 1100.0, 1050.0]
    """
    prices = df["Open"].astype(float)
    history: list[dict[str, float | int | pd.Timestamp]] = []

    # Execute initial purchase at the first available opening price
    start_price = float(prices.iloc[0])
    shares_bought = int(capital / start_price)
    capital -= shares_bought * start_price
    shares += shares_bought

    dates = pd.DatetimeIndex(df.index)

    # Calculate portfolio valuation across each trading session
    for date, price in zip(dates, prices):
        price = float(price)
        portfolio_value = capital + (shares * price)

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
