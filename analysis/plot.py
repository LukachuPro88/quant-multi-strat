from typing import Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

PlotType = Literal["scatter", "line", "histogram"]

plot_types: list[PlotType] = ["scatter", "line", "histogram"]


def plot(
    x: Union[list[float], np.ndarray],
    y: Optional[Union[list[float], np.ndarray]] = None,
    plot_type: PlotType = "scatter",
    to_percent: bool = False,
    axes_point: float = -1.0,
    xlabel: str = "",
    ylabel: str = "",
) -> None:
    """Plot data and display it in an interactive GUI window.

    Parameters
    ----------
    x : array_like
        Data points for the X axis.
    y : array_like, optional
        Data points for the Y axis. Not required for histograms.
    plot_type : {"scatter", "line", "histogram"}, default="scatter"
        Type of plot to generate.
    to_percent : bool, default=False
        If True, multiply x and y values by 100.
    axes_point : float, default=-1.0
        Position where the reference axes cross. Negative values disable them.
    xlabel : str, default=""
        Label for the X axis.
    ylabel : str, default=""
        Label for the Y axis.

    Note
    ----
    The interactive GUI window blocks program execution until it is manually closed.
    This function is intended for interactive use.

    Raises
    ------
    ValueError
        If x is empty.
        If y is None for a non-histogram plot.
    """

    # Convert the input data to NumPy arrays for consistent processing.
    x_arr = np.asarray(x, dtype=float)

    if x_arr.size == 0:
        raise ValueError("Parameter x is empty or contains invalid data.")

    # Y data is required for scatter and line plots, but not for histograms.
    if y is None and plot_type != "histogram":
        raise ValueError(f"Parameter y is required when plot_type is '{plot_type}'.")

    y_arr = np.asarray(y, dtype=float) if y is not None else None

    # Convert the data to percentages when requested.
    if to_percent:
        x_arr *= 100.0

        if y_arr is not None:
            y_arr *= 100.0

    # Create a separate figure and axes for the plot.
    fig, ax = plt.subplots()

    # Draw the requested type of plot.
    if plot_type == "histogram":
        ax.hist(x_arr, bins=10)
    else:
        if y_arr is None:
            raise ValueError("Parameter y cannot be None for scatter or line plots.")

        if plot_type == "scatter":
            ax.scatter(x_arr, y_arr)
        elif plot_type == "line":
            ax.plot(x_arr, y_arr)

    # Add horizontal and vertical reference lines when enabled.
    if axes_point >= 0:
        ax.axhline(axes_point, color="gray", linestyle="--")
        ax.axvline(axes_point, color="gray", linestyle="--")

    # Add axis labels when they were provided.
    if xlabel:
        ax.set_xlabel(xlabel)

    if ylabel:
        ax.set_ylabel(ylabel)

    # Display the plot and release the figure after the window is closed.
    plt.show()
    plt.close(fig)
