"""Tests for the analysis functions, using tiny hand-made tables."""

import pandas as pd
import pytest

from f1_telemetry.analysis import driver_pace


def make_driver_laps(driver, team, seconds):
    """Columns of one driver's laps, numbered from lap 2 (lap 1 is never clean)."""
    n = len(seconds)
    return {
        "Driver": [driver] * n,
        "Team": [team] * n,
        "LapNumber": list(range(2, n + 2)),
        "LapTimeSeconds": list(seconds),
        "TrackStatus": ["1"] * n,
    }


def make_race():
    """VER (one lap under safety car), LEC, and HAM with only two laps."""
    ver = make_driver_laps("VER", "Red Bull", [200.0, 90.0, 91.0, 92.0, 93.0, 94.0])
    ver["TrackStatus"][0] = "4"  # the slow lap is under safety car
    lec = make_driver_laps("LEC", "Ferrari", [91.0, 91.0, 93.0, 95.0, 100.0])
    ham = make_driver_laps("HAM", "Mercedes", [90.0, 90.0])

    drivers = [ver, lec, ham]
    laps = pd.DataFrame({key: sum((d[key] for d in drivers), []) for key in ver})
    no_time = pd.to_timedelta(pd.Series([None] * len(laps), dtype=float), unit="s")
    laps["LapTime"] = pd.to_timedelta(laps["LapTimeSeconds"], unit="s")
    laps["PitInTime"] = no_time
    laps["PitOutTime"] = no_time
    laps["Deleted"] = False
    return laps


def test_driver_pace_uses_clean_laps_only_and_sorts_by_median():
    pace = driver_pace(make_race())
    assert pace["Driver"].tolist() == ["VER", "LEC"]
    assert pace["CleanLaps"].tolist() == [5, 5]
    assert pace["MedianTime"].tolist() == pytest.approx([92.0, 93.0])
    assert pace["MeanTime"].tolist() == pytest.approx([92.0, 94.0])
    assert pace["FastestTime"].tolist() == pytest.approx([90.0, 91.0])
    assert pace["GapToBest"].tolist() == pytest.approx([0.0, 1.0])
    assert pace["Team"].tolist() == ["Red Bull", "Ferrari"]


def test_driver_pace_leaves_out_drivers_with_few_clean_laps():
    assert "HAM" not in driver_pace(make_race())["Driver"].tolist()
    assert "HAM" in driver_pace(make_race(), min_laps=2)["Driver"].tolist()


def test_driver_pace_missing_column_raises():
    with pytest.raises(ValueError, match="Team"):
        driver_pace(make_race().drop(columns="Team"))
