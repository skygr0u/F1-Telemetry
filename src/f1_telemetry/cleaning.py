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


def _require_columns(laps: pd.DataFrame, columns: list[str]) -> None:
    """Raise ValueError if the lap table lacks one of the columns."""
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
    _require_columns(laps, REQUIRED_LAP_COLUMNS)

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
    _require_columns(laps, CLEAN_LAP_COLUMNS)
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
