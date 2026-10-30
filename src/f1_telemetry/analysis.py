"""Analysis functions: race pace, tyre strategy, degradation and grid effect."""

import pandas as pd
from scipy import stats

from f1_telemetry.cleaning import get_clean_laps, require_columns

PACE_COLUMNS = ["Driver", "Team", "LapTimeSeconds"]
# A median of two or three laps is not a pace, so drivers below are left out.
MIN_CLEAN_LAPS = 5
STRATEGY_COLUMNS = ["Driver", "Stint", "Compound", "Laps"]
DEGRADATION_COLUMNS = ["Driver", "Stint", "Compound", "TyreLife", "LapTimeSeconds"]
# A line through fewer clean laps is too noisy to mean anything.
MIN_STINT_LAPS = 8


def driver_pace(laps: pd.DataFrame, min_laps: int = MIN_CLEAN_LAPS) -> pd.DataFrame:
    """
    Summarise the race pace of each driver on clean laps.

    The laps are filtered with ``get_clean_laps``, so in and out laps, the
    first lap, laps under yellow flags or safety car, deleted laps and laps
    without a time never count. The median is the main pace number because a
    single odd lap hardly moves it.

    Parameters
    ----------
    laps : pandas.DataFrame
        Lap table after ``prepare_laps`` (it needs ``LapTimeSeconds``).
    min_laps : int, optional
        Drivers with fewer clean laps are left out. Defaults to
        ``MIN_CLEAN_LAPS``.

    Returns
    -------
    pandas.DataFrame
        One row per driver, sorted from the fastest median, with the columns
        Driver, Team, CleanLaps, MedianTime, MeanTime, StdTime, FastestTime
        (all times in seconds) and GapToBest (the median minus the best
        median, in seconds).

    Raises
    ------
    ValueError
        If a needed column is missing, see ``get_clean_laps``.
    """
    require_columns(laps, PACE_COLUMNS)
    clean = get_clean_laps(laps)
    pace = clean.groupby("Driver", as_index=False).agg(
        Team=("Team", "first"),
        CleanLaps=("LapTimeSeconds", "count"),
        MedianTime=("LapTimeSeconds", "median"),
        MeanTime=("LapTimeSeconds", "mean"),
        StdTime=("LapTimeSeconds", "std"),
        FastestTime=("LapTimeSeconds", "min"),
    )
    pace = pace[pace["CleanLaps"] >= min_laps].copy()
    pace["GapToBest"] = pace["MedianTime"] - pace["MedianTime"].min()
    return pace.sort_values("MedianTime").reset_index(drop=True)


def strategy_summary(stints: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise the tyre strategy of each driver in one row.

    ``Stops`` counts every tyre change, so a free tyre change under a red
    flag counts like a normal pit stop.

    Parameters
    ----------
    stints : pandas.DataFrame
        The table from ``build_stints``.

    Returns
    -------
    pandas.DataFrame
        One row per driver, sorted by driver, with the columns Driver, Stops
        (number of stints minus one), Strategy (the compounds in race order,
        for example "SOFT > MEDIUM > SOFT") and Laps (total laps of all
        stints).

    Raises
    ------
    ValueError
        If a needed column is missing.
    """
    require_columns(stints, STRATEGY_COLUMNS)
    summary = (
        stints.sort_values(["Driver", "Stint"])
        .groupby("Driver", as_index=False)
        .agg(
            Stops=("Stint", "count"),
            Strategy=("Compound", " > ".join),
            Laps=("Laps", "sum"),
        )
    )
    summary["Stops"] = summary["Stops"] - 1
    return summary


def stint_degradation(
    laps: pd.DataFrame, min_laps: int = MIN_STINT_LAPS
) -> pd.DataFrame:
    """
    Fit a straight line of lap time against tyre age for each stint.

    Only clean laps are used. The slope is the degradation: the seconds a lap
    gets slower for each extra lap of tyre age. It is a naive estimate,
    because the car also gets faster as fuel burns, which hides part of the
    wear. The fit is meant for dry races only, so the caller should check
    ``is_wet_race`` first.

    Parameters
    ----------
    laps : pandas.DataFrame
        Lap table after ``prepare_laps`` (it needs ``LapTimeSeconds`` and
        ``TyreLife``).
    min_laps : int, optional
        Stints with fewer clean laps are left out. Defaults to
        ``MIN_STINT_LAPS``.

    Returns
    -------
    pandas.DataFrame
        One row per fitted stint with the columns Driver, Stint, Compound,
        Laps (clean laps used), Slope (seconds per lap of tyre age),
        Intercept (fitted lap time at tyre age zero, in seconds), RSquared
        (how much of the lap time variation the line explains) and StdErr
        (the uncertainty of the slope).

    Raises
    ------
    ValueError
        If a needed column is missing, see ``get_clean_laps``.
    """
    require_columns(laps, DEGRADATION_COLUMNS)
    clean = get_clean_laps(laps)

    rows = []
    for _, stint in clean.groupby(["Driver", "Stint"]):
        stint = stint.dropna(subset=["TyreLife"])
        if len(stint) < min_laps or stint["TyreLife"].nunique() < 2:
            continue
        slope, intercept, rvalue, _pvalue, stderr = stats.linregress(
            stint["TyreLife"], stint["LapTimeSeconds"]
        )
        rows.append(
            {
                "Driver": stint["Driver"].iloc[0],
                "Stint": stint["Stint"].iloc[0],
                "Compound": stint["Compound"].iloc[0],
                "Laps": len(stint),
                "Slope": slope,
                "Intercept": intercept,
                "RSquared": rvalue**2,  # type: ignore
                "StdErr": stderr,
            }
        )
    columns = [
        "Driver",
        "Stint",
        "Compound",
        "Laps",
        "Slope",
        "Intercept",
        "RSquared",
        "StdErr",
    ]
    return pd.DataFrame(rows, columns=columns)
