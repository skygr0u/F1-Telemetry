"""Tests for the lap cleaning functions, using tiny hand-made tables."""

import pandas as pd
import pytest

from f1_telemetry.cleaning import (
    build_stints,
    clean_compound,
    get_clean_laps,
    is_clean_lap,
    is_wet_race,
    prepare_laps,
    wet_lap_share,
)


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


def make_rule_laps():
    """Eight laps: two are clean, each of the other six breaks one rule."""

    def seconds(values):
        return pd.to_timedelta(pd.Series(values, dtype=float), unit="s")

    return pd.DataFrame(
        {
            "LapNumber": [1, 2, 3, 4, 5, 6, 7, 8],
            "LapTime": seconds([95, 95, 95, 95, 95, 95, None, 95]),
            "PitInTime": seconds([None, None, 3000, None, None, None, None, None]),
            "PitOutTime": seconds([None, None, None, 3100, None, None, None, None]),
            "TrackStatus": ["1", "1", "1", "1", "12", "1", "1", "1"],
            "Deleted": [False, False, False, False, False, True, False, False],
        }
    )


def test_is_clean_lap_applies_each_rule():
    # Order: first lap, clean, in lap, out lap, yellow flag, deleted,
    # no lap time, clean.
    mask = is_clean_lap(make_rule_laps())
    assert mask.tolist() == [False, True, False, False, False, False, False, True]


def test_get_clean_laps_keeps_original_index():
    clean = get_clean_laps(make_rule_laps())
    assert clean.index.tolist() == [1, 7]


def test_is_clean_lap_rejects_numeric_track_status():
    laps = make_rule_laps()
    laps["TrackStatus"] = 1
    with pytest.raises(ValueError, match="TrackStatus"):
        is_clean_lap(laps)


def test_is_clean_lap_missing_column_raises():
    laps = make_rule_laps().drop(columns="PitInTime")
    with pytest.raises(ValueError, match="PitInTime"):
        is_clean_lap(laps)


def make_stint_laps():
    """Seven laps of two drivers, in shuffled order, with three stints."""
    return pd.DataFrame(
        {
            "Driver": ["VER", "VER", "VER", "VER", "VER", "LEC", "LEC"],
            "LapNumber": [3, 1, 2, 5, 4, 1, 2],
            "Stint": [1, 1, 1, 2, 2, 1, 1],
            "Compound": ["SOFT", "SOFT", "SOFT", "HARD", "HARD", "MEDIUM", "MEDIUM"],
            "TyreLife": [3.0, 1.0, 2.0, 6.0, 5.0, 1.0, 2.0],
            "FreshTyre": [True, True, True, False, False, True, True],
        }
    )


def test_build_stints_gives_one_row_per_driver_and_stint():
    stints = build_stints(make_stint_laps())
    assert stints["Driver"].tolist() == ["LEC", "VER", "VER"]
    assert stints["Stint"].tolist() == [1, 1, 2]
    assert stints["Compound"].tolist() == ["MEDIUM", "SOFT", "HARD"]
    assert stints["FirstLap"].tolist() == [1, 1, 4]
    assert stints["LastLap"].tolist() == [2, 3, 5]
    assert stints["Laps"].tolist() == [2, 3, 2]
    # The rows were shuffled, so these only work if the laps are sorted first.
    assert stints["TyreLifeStart"].tolist() == [1.0, 1.0, 5.0]
    assert stints["TyreLifeEnd"].tolist() == [2.0, 3.0, 6.0]
    assert stints["FreshTyre"].tolist() == [True, True, False]


def test_build_stints_missing_column_raises():
    laps = make_stint_laps().drop(columns="TyreLife")
    with pytest.raises(ValueError, match="TyreLife"):
        build_stints(laps)


def test_wet_lap_share_counts_wet_and_intermediate():
    laps = pd.DataFrame({"Compound": ["wet", "INTERMEDIATE", None, "SOFT"]})
    assert wet_lap_share(laps) == pytest.approx(0.5)


def test_is_wet_race_threshold_is_inclusive():
    # Exactly 1 lap in 10 (10%) on intermediates counts as wet.
    laps = pd.DataFrame({"Compound": ["INTERMEDIATE"] + ["SOFT"] * 9})
    assert is_wet_race(laps)


def test_is_wet_race_false_for_dry_race_and_strict_threshold():
    dry = pd.DataFrame({"Compound": ["SOFT"] * 10})
    assert not is_wet_race(dry)
    some_wet = pd.DataFrame({"Compound": ["WET"] * 2 + ["SOFT"] * 8})
    assert not is_wet_race(some_wet, threshold=0.5)


def test_wet_race_functions_handle_empty_table():
    laps = pd.DataFrame({"Compound": []})
    assert wet_lap_share(laps) == 0.0
    assert not is_wet_race(laps)


def test_wet_lap_share_missing_column_raises():
    with pytest.raises(ValueError, match="Compound"):
        wet_lap_share(pd.DataFrame({"Driver": ["VER"]}))
