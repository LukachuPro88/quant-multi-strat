def print_metrics(**kwargs: float | str) -> None:
    """Prints formatted performance metrics after backtest execution.

    Parameters
    ----------
    **kwargs : float or str
        Arbitrary performance metrics to display where the key is the metric
        name and the value is the numerical result or string representation.
        Common key-value pairs include:

        sharpe_ratio : float
            The annualized Sharpe ratio of the portfolio.
        max_drawdown : float
            The maximum peak-to-trough decline expressed as a fraction.
        total_return : float
            The overall portfolio cumulative return.

    Examples
    --------
    >>> print_metrics(sharpe_ratio=1.45, total_return=5000)
    SHARPE RATIO: 1.45
    TOTAL RETURN: 5000
    """
    for name, value in kwargs.items():
        formatted_name = name.replace("_", " ").upper()
        print(f"{formatted_name}: {value}")
