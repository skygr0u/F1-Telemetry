"""Clean lap data: time columns, validity flags, clean laps, stints."""

import pandas as pd

UNKNOWN_COMPOUND = "UNKNOWN"
# Different ways the raw data says "we do not know the tyre". A CSV snapshot
# turns the text "None" into a missing value, so we must accept all of them.
MISSING_COMPOUND_LABELS = ("", "NONE", UNKNOWN_COMPOUND)

REQUIRED_LAP_COLUMNS = ["LapTime", "LapNumber", "Stint", "Compound", "Deleted"]

CLEAN_LAP_COLUMNS = [
    "LapTime",
    "LapNumber",
    "PitInTime",
    "PitOutTime",
    "TrackStatus",
    "Deleted",
]

STINT_COLUMNS = [
    "Driver",
    "Stint",
    "LapNumber",
    "Compound",
    "TyreLife",
    "FreshTyre",
]

WET_COMPOUNDS = ("WET", "INTERMEDIATE")
# A race counts as wet when at least this share of its laps is on wet tyres.
WET_RACE_THRESHOLD = 0.10


def require_columns(laps: pd.DataFrame, columns: list[str]) -> None:
    """
    Check that a lap table has all the columns a function needs.

    Parameters
    ----------
    laps : pandas.DataFrame
        The table to check.
    columns : list of str
        Names of the columns that must be present.

    Raises
    ------
    ValueError
        If one or more of the columns are missing. The message names them.
    """
    missing = [column for column in columns if column not in laps.columns]
    if missing:
        raise ValueError(f"The lap table has no column(s): {', '.join(missing)}.")


def clean_compound(compound: pd.Series) -> pd.Series:
    """
    Give every tyre compound one clean, upper-case label.

    A missing value, the text "None" and "UNKNOWN" all mean that the tyre is
    not known, so they all become "UNKNOWN". A real label is kept ("soft"
    becomes "SOFT"). Missing values are replaced on purpose, because
    ``groupby`` silently drops them.

    Parameters
    ----------
    compound : pandas.Series
        The ``Compound`` column of a lap table.

    Returns
    -------
    pandas.Series
        A new series of upper-case text without missing values.
    """
    text = compound.fillna(UNKNOWN_COMPOUND).astype(str).str.strip().str.upper()
    return text.where(~text.isin(MISSING_COMPOUND_LABELS), UNKNOWN_COMPOUND)


def prepare_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of the lap table with clean types and one extra column.

    The changes are: ``LapNumber`` and ``Stint`` become integers,
    ``Compound`` is cleaned with ``clean_compound``, ``Deleted`` becomes a
    true/false column (only a real True counts as deleted), and
    ``LapTimeSeconds`` is added as a float.

    Parameters
    ----------
    laps : pandas.DataFrame
        Lap table from ``get_laps`` or from a snapshot read with
        ``load_dataframe`` (``LapTime`` must already be a timedelta).

    Returns
    -------
    pandas.DataFrame
        A new table. The input table is not changed.

    Raises
    ------
    ValueError
        If one of the columns in ``REQUIRED_LAP_COLUMNS`` is missing.
    """
    require_columns(laps, REQUIRED_LAP_COLUMNS)

    prepared = laps.copy()
    prepared["LapNumber"] = prepared["LapNumber"].astype(int)
    prepared["Stint"] = prepared["Stint"].astype(int)
    prepared["Compound"] = clean_compound(prepared["Compound"])
    prepared["Deleted"] = prepared["Deleted"].eq(True)
    prepared["LapTimeSeconds"] = prepared["LapTime"].dt.total_seconds()
    return prepared


def is_clean_lap(laps: pd.DataFrame) -> pd.Series:
    """
    Mark the laps that can be compared with each other.

    A lap is clean when it has a lap time, is not the first lap, is not an in
    lap or an out lap, was run under a clear track for the whole lap
    (``TrackStatus`` is exactly "1") and was not deleted.

    Parameters
    ----------
    laps : pandas.DataFrame
        Lap table from ``get_laps``, or from a snapshot read with
        ``load_dataframe`` (``TrackStatus`` must be text).

    Returns
    -------
    pandas.Series
        True for clean laps, False for the others. It has the same index as
        ``laps``.

    Raises
    ------
    ValueError
        If a needed column is missing, or ``TrackStatus`` holds numbers
        instead of text.
    """
    require_columns(laps, CLEAN_LAP_COLUMNS)
    if pd.api.types.is_numeric_dtype(laps["TrackStatus"]):
        raise ValueError(
            "TrackStatus must be text. Read snapshots with load_dataframe."
        )

    has_time = laps["LapTime"].notna()
    not_first_lap = laps["LapNumber"] != 1
    not_pit_lap = laps["PitInTime"].isna() & laps["PitOutTime"].isna()
    clear_track = laps["TrackStatus"].eq("1")
    not_deleted = ~laps["Deleted"].eq(True)
    return has_time & not_first_lap & not_pit_lap & clear_track & not_deleted


def get_clean_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the clean laps of a lap table.

    Parameters
    ----------
    laps : pandas.DataFrame
        Lap table, as for ``is_clean_lap``.

    Returns
    -------
    pandas.DataFrame
        A new table with the clean laps only. The index of the kept rows is
        not changed, so each row can be traced back to ``laps``.

    Raises
    ------
    ValueError
        If ``laps`` cannot be checked, see ``is_clean_lap``.
    """
    return laps[is_clean_lap(laps)].copy()


def build_stints(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise each stint (one set of tyres) of each driver in one row.

    All laps are used, clean or not, because a stint lasts from the out lap
    to the in lap. Laps are put in race order first, so "first" and "last"
    mean the start and the end of the stint.

    Parameters
    ----------
    laps : pandas.DataFrame
        Lap table after ``prepare_laps`` (``Stint`` and ``LapNumber`` must be
        integers and ``Compound`` must be clean).

    Returns
    -------
    pandas.DataFrame
        One row per driver and stint, sorted by driver and stint, with the
        columns Driver, Stint, Compound, FirstLap, LastLap, Laps,
        TyreLifeStart, TyreLifeEnd and FreshTyre. ``Laps`` is the number of
        laps in the stint. ``TyreLifeStart`` above 1 or ``FreshTyre`` False
        means the set had been used before the stint.

    Raises
    ------
    ValueError
        If a needed column is missing.
    """
    require_columns(laps, STINT_COLUMNS)
    stints = (
        laps.sort_values(["Driver", "LapNumber"])
        .groupby(["Driver", "Stint"], as_index=False)
        .agg(
            Compound=("Compound", "first"),
            FirstLap=("LapNumber", "min"),
            LastLap=("LapNumber", "max"),
            TyreLifeStart=("TyreLife", "first"),
            TyreLifeEnd=("TyreLife", "last"),
            FreshTyre=("FreshTyre", "first"),
        )
    )
    stints["Laps"] = stints["LastLap"] - stints["FirstLap"] + 1
    return stints[
        [
            "Driver",
            "Stint",
            "Compound",
            "FirstLap",
            "LastLap",
            "Laps",
            "TyreLifeStart",
            "TyreLifeEnd",
            "FreshTyre",
        ]
    ]


def wet_lap_share(laps: pd.DataFrame) -> float:
    """
    Compute the share of laps run on wet-weather tyres.

    Parameters
    ----------
    laps : pandas.DataFrame
        Lap table with a ``Compound`` column. Labels are cleaned with
        ``clean_compound`` first, so the table does not need to be prepared.

    Returns
    -------
    float
        A number between 0 and 1: laps on WET or INTERMEDIATE divided by all
        laps. It is 0.0 for an empty table.

    Raises
    ------
    ValueError
        If the ``Compound`` column is missing.
    """
    require_columns(laps, ["Compound"])
    if laps.empty:
        return 0.0
    compound = clean_compound(laps["Compound"])
    return float(compound.isin(WET_COMPOUNDS).mean())


def is_wet_race(laps: pd.DataFrame, threshold: float = WET_RACE_THRESHOLD) -> bool:
    """
    Tell whether a race was wet.

    A race is wet when the share of laps on WET or INTERMEDIATE tyres is at
    least ``threshold``.

    Parameters
    ----------
    laps : pandas.DataFrame
        Lap table with a ``Compound`` column.
    threshold : float, optional
        Share of wet laps from which the race counts as wet. Defaults to
        ``WET_RACE_THRESHOLD`` (10%).

    Returns
    -------
    bool
        True for a wet race, False for a dry one.

    Raises
    ------
    ValueError
        If the ``Compound`` column is missing.
    """
    return wet_lap_share(laps) >= threshold
