"""Analysis functions: race pace, tyre strategy, degradation and grid effect."""

import numpy as np
import pandas as pd
from scipy import stats

from f1_telemetry.cleaning import get_clean_laps, require_columns

PACE_COLUMNS = ["Driver", "Team", "LapTimeSeconds"]
# A median of two or three laps is not a pace, so drivers below are left out.
MIN_CLEAN_LAPS = 5
STRATEGY_COLUMNS = ["Driver", "Stint", "Compound", "Laps"]
DEGRADATION_COLUMNS = ["Driver", "Stint", "Compound", "TyreLife", "LapTimeSeconds"]
FUEL_MODEL_COLUMNS = [
    "Driver",
    "Compound",
    "TyreLife",
    "LapNumber",
    "LapTimeSeconds",
]
# A line through fewer clean laps is too noisy to mean anything.
MIN_STINT_LAPS = 8
# The order used to show compounds: dry from softest to hardest, then wet.
COMPOUND_ORDER = ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET", "UNKNOWN")
COMPOUND_RANK = {name: rank for rank, name in enumerate(COMPOUND_ORDER)}


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


def compound_degradation(degradation: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise the degradation slopes of the stints for each compound.

    The median slope is the main number because one noisy stint hardly moves
    it. The minimum and maximum show how much the stints differ. A compound
    that is not in ``COMPOUND_ORDER`` is listed last.

    Parameters
    ----------
    degradation : pandas.DataFrame
        The table from ``stint_degradation``.

    Returns
    -------
    pandas.DataFrame
        One row per compound, from the softest, with the columns Compound,
        Stints (number of fitted stints), MedianSlope, MeanSlope, MinSlope
        and MaxSlope (all slopes in seconds per lap of tyre age).

    Raises
    ------
    ValueError
        If the Compound or Slope column is missing.
    """
    require_columns(degradation, ["Compound", "Slope"])
    summary = degradation.groupby("Compound", as_index=False).agg(
        Stints=("Slope", "count"),
        MedianSlope=("Slope", "median"),
        MeanSlope=("Slope", "mean"),
        MinSlope=("Slope", "min"),
        MaxSlope=("Slope", "max"),
    )
    summary = summary.sort_values("Compound", key=lambda c: c.map(COMPOUND_RANK))
    return summary.reset_index(drop=True)


def fuel_corrected_degradation(
    laps: pd.DataFrame, min_laps: int = MIN_STINT_LAPS
) -> pd.DataFrame:
    """
    Estimate tyre degradation while allowing the car to get faster each lap.

    One least-squares model is fitted to all clean laps:
    lap time = driver base pace + compound offset + (tyre age x slope of the
    compound) + (lap number x L). ``L`` collects what makes a car faster as
    the race goes on at the same tyre age: fuel burn and a track that gets
    rubbered in. The data cannot split the two. Tyre age and lap number can be
    told apart because tyre age restarts at each stop. The model is meant for
    dry races, so the caller should check ``is_wet_race`` first.

    Parameters
    ----------
    laps : pandas.DataFrame
        Lap table after ``prepare_laps`` (it needs ``TyreLife``).
    min_laps : int, optional
        Compounds with fewer clean laps are left out of the model. Defaults
        to ``MIN_STINT_LAPS``.

    Returns
    -------
    pandas.DataFrame
        One row per compound and one row for the lap number term, with the
        columns Term (a compound name or "LapNumber"), Slope (seconds per lap
        of tyre age, or seconds per race lap for "LapNumber"), StdErr
        (uncertainty of the slope, assuming independent laps) and Laps
        (clean laps behind the estimate).

    Raises
    ------
    ValueError
        If a needed column is missing, or the laps cannot separate tyre age
        from lap number (too few laps, or one single stint).
    """
    require_columns(laps, FUEL_MODEL_COLUMNS)
    clean = get_clean_laps(laps).dropna(subset=["TyreLife"])
    counts = clean["Compound"].value_counts()
    kept = counts[counts >= min_laps].index
    clean = clean[clean["Compound"].isin(kept)]
    compounds = sorted(kept, key=lambda c: COMPOUND_RANK.get(c, len(COMPOUND_RANK)))
    if not compounds:
        raise ValueError(f"No compound has {min_laps} clean laps to fit a model.")

    # Design matrix: one row per lap, one column per term of the model.
    drivers = pd.get_dummies(clean["Driver"], dtype=float).to_numpy()
    offsets = pd.get_dummies(clean["Compound"], dtype=float)[compounds[1:]].to_numpy()
    tyre_age = clean["TyreLife"].to_numpy(dtype=float)
    is_compound = clean["Compound"].to_numpy()[:, None] == np.array(compounds)
    age_by_compound = tyre_age[:, None] * is_compound
    lap_number = clean["LapNumber"].to_numpy(dtype=float)[:, None]
    design = np.hstack([drivers, offsets, age_by_compound, lap_number])

    lap_time = clean["LapTimeSeconds"].to_numpy(dtype=float)
    coefficients, _, rank, _ = np.linalg.lstsq(design, lap_time, rcond=None)
    free = len(lap_time) - design.shape[1]
    if rank < design.shape[1] or free <= 0:
        raise ValueError(
            "These laps cannot separate tyre age from lap number: "
            "there are too few laps or only one stint."
        )

    residuals = lap_time - design @ coefficients
    variance = residuals @ residuals / free
    errors = np.sqrt(np.diag(variance * np.linalg.inv(design.T @ design)))

    first = drivers.shape[1] + offsets.shape[1]  # first tyre-age column
    laps_behind = [int(counts[c]) for c in compounds] + [len(clean)]
    return pd.DataFrame(
        {
            "Term": [*compounds, "LapNumber"],
            "Slope": coefficients[first:],
            "StdErr": errors[first:],
            "Laps": laps_behind,
        }
    )
