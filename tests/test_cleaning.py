"""Tests for the lap cleaning functions, using tiny hand-made tables."""

import pandas as pd
import pytest

from f1_telemetry.cleaning import clean_compound, prepare_laps


def make_laps():
    """Three laps that show the problems found in the real data."""
    return pd.DataFrame(
        {
            "Driver": ["VER", "VER", "LEC"],
            "LapTime": pd.to_timedelta(pd.Series([95.5, None, 96.0]), unit="s"),
            "LapNumber": [1.0, 2.0, 3.0],
            "Stint": [1.0, 1.0, 2.0],
            "Compound": ["soft", None, "None"],
            "Deleted": [False, None, True],
        }
    )


def test_clean_compound_gives_one_label_for_unknown():
    compound = pd.Series(["soft", " Medium ", None, "None", "UNKNOWN", ""])
    result = clean_compound(compound)
    assert result.tolist() == [
        "SOFT",
        "MEDIUM",
        "UNKNOWN",
        "UNKNOWN",
        "UNKNOWN",
        "UNKNOWN",
    ]


def test_prepare_laps_fixes_types_and_adds_seconds():
    prepared = prepare_laps(make_laps())
    assert pd.api.types.is_integer_dtype(prepared["LapNumber"])
    assert pd.api.types.is_integer_dtype(prepared["Stint"])
    assert prepared["Deleted"].tolist() == [False, False, True]
    assert prepared["LapTimeSeconds"].iloc[0] == pytest.approx(95.5)
    assert pd.isna(prepared["LapTimeSeconds"].iloc[1])


def test_prepare_laps_does_not_change_the_input():
    laps = make_laps()
    prepare_laps(laps)
    assert laps["LapNumber"].dtype == float
    assert "LapTimeSeconds" not in laps.columns


def test_prepare_laps_missing_column_raises():
    laps = make_laps().drop(columns="Stint")
    with pytest.raises(ValueError, match="Stint"):
        prepare_laps(laps)
