"""Clean lap data: time columns, validity flags, clean laps, stints."""

import pandas as pd

"""Prepare and filter lap tables so that laps can be compared with each other."""


UNKNOWN_COMPOUND = "UNKNOWN"
# Different ways the raw data says "we do not know the tyre". A CSV snapshot
# turns the text "None" into a missing value, so we must accept all of them.
MISSING_COMPOUND_LABELS = ("", "NONE", UNKNOWN_COMPOUND)

REQUIRED_LAP_COLUMNS = ["LapTime", "LapNumber", "Stint", "Compound", "Deleted"]


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
    missing = [column for column in REQUIRED_LAP_COLUMNS if column not in laps.columns]
    if missing:
        raise ValueError(f"The lap table has no column(s): {', '.join(missing)}.")

    prepared = laps.copy()
    prepared["LapNumber"] = prepared["LapNumber"].astype(int)
    prepared["Stint"] = prepared["Stint"].astype(int)
    prepared["Compound"] = clean_compound(prepared["Compound"])
    prepared["Deleted"] = prepared["Deleted"].eq(True)
    prepared["LapTimeSeconds"] = prepared["LapTime"].dt.total_seconds()
    return prepared
