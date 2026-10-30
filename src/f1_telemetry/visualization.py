"""Plots for the race analysis. Each function returns a Matplotlib figure."""

import matplotlib.pyplot as plt
import numpy as np
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


def plot_stint_fit(
    clean_laps: pd.DataFrame,
    fits: pd.DataFrame,
    driver: str,
    stint: int,
    title: str | None = None,
) -> Figure:
    """
    Draw the clean laps of one stint with the fitted line of lap time.

    Parameters
    ----------
    clean_laps : pandas.DataFrame
        Clean laps with the columns Driver, Stint, TyreLife and
        LapTimeSeconds.
    fits : pandas.DataFrame
        The table from ``stint_degradation``.
    driver : str
        Three-letter code of the driver, for example "LEC".
    stint : int
        Stint number of that driver.
    title : str, optional
        Title of the plot. By default it names the driver, the stint and the
        compound.

    Returns
    -------
    matplotlib.figure.Figure
        The figure. Showing or saving it is left to the caller.

    Raises
    ------
    ValueError
        If a needed column is missing, or the driver has no fitted stint with
        that number.
    """
    require_columns(clean_laps, ["Driver", "Stint", "TyreLife", "LapTimeSeconds"])
    require_columns(fits, ["Driver", "Stint", "Compound", "Slope", "Intercept"])
    points = clean_laps[
        (clean_laps["Driver"] == driver) & (clean_laps["Stint"] == stint)
    ]
    fit = fits[(fits["Driver"] == driver) & (fits["Stint"] == stint)]
    if fit.empty or points.empty:
        raise ValueError(f"There is no fitted stint {stint} for {driver}.")

    slope = fit["Slope"].to_numpy(dtype=float)[0]
    intercept = fit["Intercept"].to_numpy(dtype=float)[0]
    compound = str(fit["Compound"].iloc[0])
    color = COMPOUND_COLORS.get(compound, COMPOUND_COLORS["UNKNOWN"])
    ages = points["TyreLife"].to_numpy(dtype=float)
    line_ages = np.array([ages.min(), ages.max()])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(
        ages,
        points["LapTimeSeconds"],
        color=color,
        edgecolor="black",
        zorder=3,
        label="Clean laps",
    )
    ax.plot(
        line_ages,
        intercept + slope * line_ages,
        color="black",
        label=f"Fit: {slope:+.3f} s per lap of tyre age",
    )
    ax.set_xlabel("Tyre age (laps)")
    ax.set_ylabel("Lap time (s)")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(
        title if title is not None else f"{driver}, stint {stint} ({compound})"
    )
    fig.tight_layout()
    return fig


def plot_degradation_comparison(
    naive: pd.DataFrame, corrected: pd.DataFrame, title: str | None = None
) -> Figure:
    """
    Compare the naive and the fuel-corrected degradation of each compound.

    For each compound two bars are drawn: the median of the stint slopes (the
    naive estimate) and the slope of the fuel-corrected model with an error
    bar of two standard errors. Only compounds present in both tables are
    drawn, in the order of ``naive``.

    Parameters
    ----------
    naive : pandas.DataFrame
        The table from ``compound_degradation``.
    corrected : pandas.DataFrame
        The table from ``fuel_corrected_degradation``.
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
    require_columns(naive, ["Compound", "MedianSlope"])
    require_columns(corrected, ["Term", "Slope", "StdErr"])
    compounds = [name for name in naive["Compound"] if name in set(corrected["Term"])]
    naive_slopes = naive.set_index("Compound").loc[compounds, "MedianSlope"]
    model = corrected.set_index("Term").loc[compounds]

    positions = np.arange(len(compounds))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(
        positions - width / 2,
        naive_slopes.to_numpy(dtype=float),
        width,
        label="Naive: median of stint slopes",
    )
    ax.bar(
        positions + width / 2,
        model["Slope"].to_numpy(dtype=float),
        width,
        yerr=2 * model["StdErr"].to_numpy(dtype=float),
        capsize=4,
        label="Fuel-corrected model (error bar: 2 standard errors)",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(positions, labels=compounds)
    ax.set_xlabel("Compound")
    ax.set_ylabel("Degradation (s per lap of tyre age)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15))
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    return fig
