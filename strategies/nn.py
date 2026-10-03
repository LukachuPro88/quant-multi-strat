from collections.abc import Callable

import pandas as pd
import torch
import torch.nn as torch_nn
import torch.nn.functional as F

import config


class NN(torch_nn.Module):
    """Deep neural network classifier for stock return direction prediction."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.fc1 = torch_nn.Linear(in_features=in_features, out_features=10)
        self.fc2 = torch_nn.Linear(in_features=10, out_features=20)
        self.fc3 = torch_nn.Linear(in_features=20, out_features=out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs a forward pass through the network."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def nn(
    df: pd.DataFrame,
    capital: float,
    shares: int,
    start_date: pd.Timestamp,
    epochs: int = 1_000,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Executes a PyTorch Neural Network classification trading strategy.

    Extracts moving averages and return ratios as input features for training a
    multi-layer perceptron (MLP). Predicts future return direction on unseen test data
    and executes positions based on output probabilities exceeding `threshold`.

    Parameters
    ----------
    df : pd.DataFrame
        Market data with DatetimeIndex and price columns ('Open', 'High', 'Low', 'Close').
    capital : float
        Starting cash balance for the strategy.
    shares : int
        Initial number of shares held at the start of the simulation.
    start_date : pd.Timestamp
        The starting date for backtesting evaluation.
    epochs : int, default=1000
        Number of gradient descent iterations during model training.
    threshold : float, default=0.5
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
    >>> df = pd.DataFrame({"Open": prices, "High": prices, "Low": prices, "Close": prices}, index=dates)
    >>> nn(df, capital=1000.0, shares=0, start_date=dates[70], epochs=10)  # doctest: +NORMALIZE_WHITESPACE
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

    torch.manual_seed(1)

    X_train, X_test, y_train, _ = _split(df)
    X_train, X_test = _normalize(X_train, X_test)

    model = NN(
        in_features=10,
        out_features=1,
    )

    criterion = torch_nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.01,
        amsgrad=True,
    )

    X_train_tensor = torch.tensor(
        X_train.to_numpy(),
        dtype=torch.float32,
    )

    y_train_tensor = torch.tensor(
        y_train.to_numpy(),
        dtype=torch.float32,
    ).reshape(-1, 1)

    trained_model = _train(
        model=model,
        X=X_train_tensor,
        y=y_train_tensor,
        epochs=epochs,
        criterion=criterion,
        optimizer=optimizer,
    )

    X_test_tensor = torch.tensor(
        X_test.to_numpy(),
        dtype=torch.float32,
    )

    predictions = _predict(
        model=trained_model,
        X=X_test_tensor,
    )

    prices = df["Open"].shift(-1).loc[X_test.index]

    return _backtest(
        prices=prices,
        predictions=predictions,
        capital=capital,
        shares=shares,
        start_date=start_date,
        threshold=threshold,
    )


# --- Helper Functions ---


def _train(
    model: torch_nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> torch_nn.Module:
    """Trains the PyTorch model using full-batch optimization."""
    model.train()

    for _ in range(epochs):
        logits = model(X)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def _predict(
    model: torch_nn.Module,
    X: torch.Tensor,
) -> torch.Tensor:
    """Generates raw output logits from the trained model."""
    model.eval()

    with torch.no_grad():
        return model(X)


def _feature_engineering(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Creates input features and binary directional targets from market data."""
    features = pd.DataFrame(
        {
            "High": df["High"],
            "Open": df["Open"],
            "Low": df["Low"],
            "Close": df["Close"],
            "MA2": df["Close"].rolling(2).mean(),
            "MA5": df["Close"].rolling(5).mean(),
            "MA10": df["Close"].rolling(10).mean(),
            "MA20": df["Close"].rolling(20).mean(),
            "Return5": df["Close"].pct_change(5),
            "Return10": df["Close"].pct_change(10),
        },
        index=df.index,
    )

    target = (df["Open"].shift(-1) > df["Open"]).astype(float)

    return features, target


def _split(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """Splits chronological market data into training and testing sets."""
    X, y = _feature_engineering(df)

    valid = X.notna().all(axis=1) & df["Open"].shift(-1).notna()

    X = X.loc[valid]
    y = y.loc[valid]

    split = int(len(X) * 0.7)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    return X_train, X_test, y_train, y_test


def _normalize(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalizes feature matrices using training-set statistics."""
    mean = X_train.mean()
    std = X_train.std()

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test


def _backtest(
    prices: pd.Series,
    predictions: torch.Tensor,
    capital: float,
    shares: int,
    start_date: pd.Timestamp,
    threshold: float,
) -> pd.DataFrame:
    """Backtests network predictions using confidence-based position sizing."""
    history: list[dict[str, float | int | pd.Timestamp]] = []
    probabilities = torch.sigmoid(predictions).squeeze(1).numpy()

    for date, price, probability in zip(
        prices.index,
        prices,
        probabilities,
    ):
        if date < start_date:
            continue

        price = float(price)
        probability = float(probability)

        signal = config.SIGNAL_BUY if probability > threshold else config.SIGNAL_SELL

        portfolio_value = capital + (price * shares)

        if signal == config.SIGNAL_BUY:
            target_shares = int(portfolio_value / price)

            if target_shares > shares:
                amount = target_shares - shares
                cost = amount * price

                if cost <= capital:
                    shares += amount
                    capital -= cost

        elif signal == config.SIGNAL_SELL and shares > 0:
            capital += shares * price
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
