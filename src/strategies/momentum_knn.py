import numpy as np
import pandas as pd

import src.config as config


def knn_momentum(
    df: pd.DataFrame,
    capital: float,
    shares: int,
    start_date: pd.Timestamp,
    momentum_range: int = 5,
    knn_amount: int = 5,
    max_distance_threshold: float = 0.05,
) -> pd.DataFrame:
    """Simulates a momentum-based K-Nearest Neighbors (KNN) trading strategy.

    Calculates the n-day return and identifies the k most similar historical returns.
    Determines the trading signal based on the majority directional return of those
    neighbors on the subsequent day. If no close neighbors fall within the distance
    threshold, it defaults to standard momentum.

    When buying, the strategy deploys all available cash capital to purchase whole
    shares. When selling, the entire position is liquidated.

    Parameters
    ----------
    df : pd.DataFrame
        The input timeseries data. Must contain an 'Open' column and a DatetimeIndex.
    capital : float
        The initial cash amount available to deploy at start of simulation.
    shares : int
        The initial number of shares held at start of simulation.
    start_date : pd.Timestamp
        The starting date of the backtest simulation.
    momentum_range : int, default=5
        The lookback window (in trading days) for return calculation.
    knn_amount : int, default=5
        The number of nearest neighbors to evaluate.
    max_distance_threshold : float, default=0.05
        Maximum feature distance allowed for KNN classification before falling back
        to standard momentum.

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
    >>> dates = pd.date_range("2020-01-01", periods=30, freq="D")
    >>> prices = [100.0 + i * 0.5 for i in range(30)]
    >>> df = pd.DataFrame({"Open": prices}, index=dates)
    >>> knn_momentum(
    ...     df,
    ...     capital=1000.0,
    ...     shares=0,
    ...     start_date=dates[0],
    ...     momentum_range=5,
    ...     knn_amount=3,
    ... ) # doctest: +NORMALIZE_WHITESPACE
                    Open   Cash  Shares  Portfolio Value
    Date
    2020-01-26  112.5  100.0       8           1000.0
    2020-01-27  113.0  100.0       8           1004.0
    2020-01-28  113.5  100.0       8           1008.0
    2020-01-29  114.0  100.0       8           1012.0
    2020-01-30  114.5  100.0       8           1016.0
    """
    if df.empty:
        raise ValueError("Dataframe should not be empty.")

    if start_date is None:
        start_date = pd.Timestamp(df.index[0])

    dates = pd.DatetimeIndex(df.index)

    returns_n = df["Open"].pct_change(momentum_range).to_numpy()
    next_returns = df["Open"].pct_change(1).shift(-1).to_numpy()

    history: list[dict[str, float | int | pd.Timestamp]] = []

    for i, (date, price) in enumerate(zip(dates, df["Open"])):
        if date < start_date or i < momentum_range + 20:
            continue

        price = float(price)
        current_return = returns_n[i]

        # Calculate signal with KNN classification and standard momentum fallback
        signal = _get_knn_signal(
            current_return=current_return,
            hist_features=returns_n[momentum_range : i - 1],
            hist_targets=next_returns[momentum_range : i - 1],
            knn_amount=knn_amount,
            max_distance_threshold=max_distance_threshold,
        )

        # Execute signals using configured signal constants
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


def _get_knn_signal(
    current_return: float,
    hist_features: np.ndarray,
    hist_targets: np.ndarray,
    knn_amount: int,
    max_distance_threshold: float,
) -> int:
    """Determine signal via KNN historical return matching or momentum fallback."""
    # Default fallback: Standard Momentum signal
    fallback_signal = config.SIGNAL_BUY if current_return > 0 else config.SIGNAL_SELL

    valid_mask = ~np.isnan(hist_targets) & ~np.isnan(hist_features)
    if np.sum(valid_mask) < knn_amount:
        return fallback_signal

    valid_features = hist_features[valid_mask]
    valid_targets = hist_targets[valid_mask]

    distances = np.abs(valid_features - current_return)
    knn_indices = np.argsort(distances)[:knn_amount]
    closest_distances = distances[knn_indices]

    if np.mean(closest_distances) <= max_distance_threshold:
        positive_ratio = float(np.mean(valid_targets[knn_indices] > 0))
        return config.SIGNAL_BUY if positive_ratio > 0.5 else config.SIGNAL_SELL

    return fallback_signal
