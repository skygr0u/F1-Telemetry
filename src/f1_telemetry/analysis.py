"""Analysis functions: race pace, tyre strategy, degradation and grid effect."""

import pandas as pd

from f1_telemetry.cleaning import get_clean_laps, require_columns

PACE_COLUMNS = ["Driver", "Team", "LapTimeSeconds"]
# A median of two or three laps is not a pace, so drivers below are left out.
MIN_CLEAN_LAPS = 5


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
