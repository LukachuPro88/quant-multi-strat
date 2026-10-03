import pandas as pd
from xgboost import XGBClassifier

import src.config as config


def xgboost(
    df: pd.DataFrame,
    capital: float,
    shares: int,
    start_date: pd.Timestamp,
    confidence_threshold: float = 0.52,
) -> pd.DataFrame:
    """Executes an XGBoost classification trading strategy.

    Extracts rolling Z-scores and 1-day returns to train an XGBoost binary classifier
    on historical data. Predicts future direction on test set dates and generates a
    buy signal if the predicted probability exceeds `confidence_threshold`.

    Parameters
    ----------
    df : pd.DataFrame
        Market data with DatetimeIndex and price columns ('Open').
    capital : float
        Starting cash balance for the strategy.
    shares : int
        Initial number of shares held at start of simulation.
    start_date : pd.Timestamp
        The starting date for backtesting evaluation.
    confidence_threshold : float, default=0.52
        Probability threshold required to issue a buy signal.

    Returns
    -------
    pd.DataFrame
        The backtest execution history indexed by date.

        * Index : DatetimeIndex
            Trading dates corresponding to the active backtest evaluation period.
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
    >>> dates = pd.date_range("2020-01-01", periods=100, freq="D")
    >>> prices = [100.0 + (i % 5) * 2.0 for i in range(100)]
    >>> df = pd.DataFrame({"Open": prices}, index=dates)
    >>> xgboost(df, capital=1000.0, shares=0, start_date=dates[70])  # doctest: +NORMALIZE_WHITESPACE
                  Open    Cash  Shares  Portfolio Value
    Date
    2020-03-12   102.0  1000.0       0           1000.0
    2020-03-13   104.0  1000.0       0           1000.0
    2020-03-14   106.0  1000.0       0           1000.0
    2020-03-15   108.0  1000.0       0           1000.0
    2020-03-16   100.0  1000.0       0           1000.0
    """
    if df.empty:
        raise ValueError("Dataframe should not be empty.")

    df_prepared = _prepare_features(df)
    model = _train(df_prepared)

    _, X_test, _, _ = _split(df_prepared)
    test_dates = X_test.index

    history: list[dict[str, float | int | pd.Timestamp]] = []

    for idx in df_prepared.index:
        if idx < start_date or idx not in test_dates:
            continue

        row = df_prepared.loc[idx]
        price = float(row["Open"])
        current_features = row[["z_score", "return_1d"]]

        if not current_features.isna().any():
            signal = _get_ml_signal(
                model, current_features, confidence_threshold=confidence_threshold
            )

            # Execute trade signal using semantic constants
            if signal == config.SIGNAL_BUY and shares == 0:
                shares = int(capital // price)
                capital -= shares * price

            elif signal == config.SIGNAL_SELL and shares > 0:
                capital += shares * price
                shares = 0

        portfolio_value = capital + (shares * price)

        history.append(
            {
                "Date": idx,
                "Open": price,
                "Cash": capital,
                "Shares": shares,
                "Portfolio Value": portfolio_value,
            }
        )

    return pd.DataFrame(history).set_index("Date")


# --- Helper Functions ---


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates stationary features on dataframe with DatetimeIndex."""
    df_feat = df.copy()

    rolling_mean = df_feat["Open"].rolling(window=20).mean()
    rolling_std = df_feat["Open"].rolling(window=20).std()

    df_feat["z_score"] = (df_feat["Open"] - rolling_mean) / rolling_std
    df_feat["return_1d"] = df_feat["Open"].pct_change()

    return df_feat


def _get_ml_signal(
    model: XGBClassifier,
    current_features: pd.Series,
    confidence_threshold: float = 0.50,
) -> int:
    """Predicts a trading signal using probability thresholding."""
    X_today = pd.DataFrame([current_features])
    prob_up = float(model.predict_proba(X_today)[0, 1])

    if prob_up >= confidence_threshold:
        return config.SIGNAL_BUY
    else:
        return config.SIGNAL_SELL


def _train(df: pd.DataFrame) -> XGBClassifier:
    """Trains an XGBoost classifier on historical feature data."""
    bst = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        objective="binary:logistic",
        random_state=42,
    )

    X_train, _, y_train, _ = _split(df)
    bst.fit(X_train, y_train)

    return bst


def _split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits chronological market data into training and testing sets."""
    X = df[["z_score", "return_1d"]]
    y = (df["Open"].shift(-1) > df["Open"]).astype(float)

    valid = X.notna().all(axis=1) & df["Open"].shift(-1).notna()

    X = X.loc[valid]
    y = y.loc[valid]

    split = int(len(X) * 0.7)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    return X_train, X_test, y_train, y_test
