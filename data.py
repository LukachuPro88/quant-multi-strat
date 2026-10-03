from pathlib import Path

import pandas as pd


def get_data(path: Path) -> pd.DataFrame:
    """Load the opening prices from a CSV data file.

    Parameters
    ----------
    path : pathlib.Path
        The path to load the data from.

    Returns
    -------
    pandas.DataFrame
        A single-column DataFrame containing the opening prices.

        * Index : datetime64[ns]
            The trading date, set as the DatetimeIndex.
        * Open : float
            The opening price of the stock, adjusted or unadjusted.
        * Close : float
            The closing price of the stock (only for NN).
        * High : float
            The highest price of the stock (only for NN).
        * Low : float
            The lowest price of the stock (only for NN).

    Raises
    ------
    FileNotFoundError
        If the file at the specified `path` does not exist.
    ValueError
        If the CSV file is corrupted or is missing the 'Date' or 'Open' columns.

    Examples
    --------
    >>> import pandas as pd
    >>> from pathlib import Path
    >>> get_data(Path("./data/AAPL.csv"))
                  Open
    Date
    1980-12-12  0.1002
    1980-12-15  0.0954
    """
    try:
        # Read only the necessary columns
        df = pd.read_csv(path, usecols=["Date", "Open", "Close", "High", "Low"])

        # Convert date into a date time index (stored as YYYY-MM-DD)
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)

        return df

    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {path} was not found.")
    except KeyError as e:
        raise ValueError(f"Required column missing from the CSV file: {e}")
    except Exception as e:
        raise ValueError(f"Failed to parse CSV file: {e}")
