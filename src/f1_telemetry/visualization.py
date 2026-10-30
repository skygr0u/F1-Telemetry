"""Plots for the race analysis. Each function returns a Matplotlib figure."""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure


def plot_pace_distribution(
    clean_laps: pd.DataFrame, pace: pd.DataFrame, title: str | None = None
) -> Figure:
    """
    Draw the distribution of clean lap times, one box per driver.

    Drivers appear in the order of ``pace``, so the fastest median is on the
    left. The box holds the middle half of the laps, the line inside it is the
    median, and the dots are unusual laps.

    Parameters
    ----------
    clean_laps : pandas.DataFrame
        Clean laps (from ``get_clean_laps`` after ``prepare_laps``) with the
        columns Driver and LapTimeSeconds.
    pace : pandas.DataFrame
        The table from ``driver_pace``. It gives the drivers to draw and
        their order.
    title : str, optional
        Title of the plot.

    Returns
    -------
    matplotlib.figure.Figure
        The figure. Showing or saving it is left to the caller.
    """
    drivers = pace["Driver"].tolist()
    times = [
        clean_laps.loc[clean_laps["Driver"] == driver, "LapTimeSeconds"]
        for driver in drivers
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(times, tick_labels=drivers, flierprops={"markersize": 3})
    ax.set_xlabel("Driver (fastest median on the left)")
    ax.set_ylabel("Lap time (s)")
    ax.grid(axis="y", alpha=0.3)
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    return fig
