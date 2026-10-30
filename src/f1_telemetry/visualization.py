"""Plots for the race analysis. Each function returns a Matplotlib figure."""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from f1_telemetry.cleaning import require_columns

# The colours used on F1 broadcasts; UNKNOWN is a neutral grey.
COMPOUND_COLORS = {
    "SOFT": "#e8002d",
    "MEDIUM": "#ffd12e",
    "HARD": "#f0f0f0",
    "INTERMEDIATE": "#39b54a",
    "WET": "#0067ff",
    "UNKNOWN": "#9e9e9e",
}
STRATEGY_PLOT_COLUMNS = ["Driver", "Compound", "FirstLap", "LastLap", "Laps"]


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


def plot_tyre_strategy(
    stints: pd.DataFrame, driver_order: list[str], title: str | None = None
) -> Figure:
    """
    Draw the tyre strategy of each driver as a row of coloured stints.

    Each stint is a bar from its first lap to its last lap, coloured by
    compound and marked with the first letter of the compound. The first
    driver of ``driver_order`` is drawn at the top.

    Parameters
    ----------
    stints : pandas.DataFrame
        The table from ``build_stints``.
    driver_order : list of str
        The drivers to draw, from top to bottom, for example in finishing
        order. Drivers that are not in the list are not drawn.
    title : str, optional
        Title of the plot.

    Returns
    -------
    matplotlib.figure.Figure
        The figure. Showing or saving it is left to the caller.

    Raises
    ------
    ValueError
        If a needed column is missing.
    """
    require_columns(stints, STRATEGY_PLOT_COLUMNS)
    row_of = {driver: row for row, driver in enumerate(driver_order)}

    fig, ax = plt.subplots(figsize=(10, 6))
    for driver, compound, first_lap, n_laps in zip(
        stints["Driver"].tolist(),
        stints["Compound"].tolist(),
        stints["FirstLap"].tolist(),
        stints["Laps"].tolist(),
    ):
        if driver not in row_of:
            continue
        row = row_of[driver]
        color = COMPOUND_COLORS.get(compound, COMPOUND_COLORS["UNKNOWN"])
        ax.barh(row, n_laps, left=first_lap - 1, color=color, edgecolor="black")
        ax.text(
            first_lap - 1 + n_laps / 2,
            row,
            compound[0],
            ha="center",
            va="center",
            fontsize=8,
        )

    ax.set_yticks(range(len(driver_order)), labels=driver_order)
    ax.invert_yaxis()  # first driver on top
    ax.set_xlim(0, stints["LastLap"].max())
    ax.set_xlabel("Lap")
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.3)

    present = [name for name in COMPOUND_COLORS if name in set(stints["Compound"])]
    handles = [
        Patch(facecolor=COMPOUND_COLORS[name], edgecolor="black", label=name.title())
        for name in present
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=len(handles),
    )
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    return fig
