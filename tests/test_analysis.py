"""Tests for the analysis functions, using tiny hand-made tables."""

import pandas as pd
import pytest

from f1_telemetry.analysis import (
    compound_degradation,
    driver_pace,
    fuel_corrected_degradation,
    stint_degradation,
    strategy_summary,
)


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


def make_stints():
    """Stint table of two drivers, with the rows in shuffled order."""
    return pd.DataFrame(
        {
            "Driver": ["VER", "LEC", "VER", "VER", "LEC"],
            "Stint": [2, 2, 1, 3, 1],
            "Compound": ["HARD", "HARD", "SOFT", "MEDIUM", "MEDIUM"],
            "Laps": [20, 37, 13, 24, 20],
        }
    )


def test_strategy_summary_counts_stops_and_orders_compounds():
    summary = strategy_summary(make_stints())
    assert summary["Driver"].tolist() == ["LEC", "VER"]
    assert summary["Stops"].tolist() == [1, 2]
    assert summary["Strategy"].tolist() == ["MEDIUM > HARD", "SOFT > HARD > MEDIUM"]
    assert summary["Laps"].tolist() == [57, 57]


def test_strategy_summary_missing_column_raises():
    with pytest.raises(ValueError, match="Compound"):
        strategy_summary(make_stints().drop(columns="Compound"))


def make_degradation_laps():
    """
    Three stints with exact straight lines of lap time against tyre age.

    VER stint 1: +0.10 s per lap of age (10 laps, one of them under safety
    car with a silly time). VER stint 2: only 5 laps. LEC stint 1: -0.05 s per
    lap of age (10 laps), like a car that gets faster as fuel burns.
    """
    plan = [
        ("VER", 1, "SOFT", 2, 10, 0.10),
        ("VER", 2, "HARD", 12, 5, 0.02),
        ("LEC", 1, "MEDIUM", 2, 10, -0.05),
    ]
    columns = {"Driver": [], "Stint": [], "Compound": [], "LapNumber": []}
    columns.update({"TyreLife": [], "LapTimeSeconds": []})
    for name, number, tyre, first_lap, n_laps, slope in plan:
        for i in range(n_laps):
            columns["Driver"].append(name)
            columns["Stint"].append(number)
            columns["Compound"].append(tyre)
            columns["LapNumber"].append(first_lap + i)
            columns["TyreLife"].append(float(i + 1))
            columns["LapTimeSeconds"].append(90.0 + slope * (i + 1))
    laps = pd.DataFrame(columns)

    # One lap of VER stint 1 is under safety car and very slow.
    slow = (laps["Driver"] == "VER") & (laps["Stint"] == 1) & (laps["TyreLife"] == 5.0)
    laps.loc[slow, "LapTimeSeconds"] = 200.0
    laps.loc[slow, "TrackStatus"] = "4"

    no_time = pd.to_timedelta(pd.Series([None] * len(laps), dtype=float), unit="s")
    laps["LapTime"] = pd.to_timedelta(laps["LapTimeSeconds"], unit="s")
    laps["PitInTime"] = no_time
    laps["PitOutTime"] = no_time
    laps["Deleted"] = False
    laps["TrackStatus"] = laps["TrackStatus"].fillna("1")
    return laps


def test_stint_degradation_recovers_slope_and_ignores_unclean_laps():
    fits = stint_degradation(make_degradation_laps())
    assert fits["Driver"].tolist() == ["LEC", "VER"]
    assert fits["Laps"].tolist() == [10, 9]  # VER lost the safety-car lap
    assert fits["Slope"].tolist() == pytest.approx([-0.05, 0.10])
    assert fits["Intercept"].tolist() == pytest.approx([90.0, 90.0])
    assert fits["RSquared"].tolist() == pytest.approx([1.0, 1.0])
    assert fits["Compound"].tolist() == ["MEDIUM", "SOFT"]


def test_stint_degradation_leaves_out_short_stints():
    assert len(stint_degradation(make_degradation_laps())) == 2
    assert len(stint_degradation(make_degradation_laps(), min_laps=5)) == 3


def test_stint_degradation_missing_column_raises():
    with pytest.raises(ValueError, match="TyreLife"):
        stint_degradation(make_degradation_laps().drop(columns="TyreLife"))


def make_slopes():
    """Slopes of five stints, with HARD first on purpose."""
    return pd.DataFrame(
        {
            "Compound": ["HARD", "SOFT", "SOFT", "MEDIUM", "SOFT"],
            "Slope": [0.02, 0.10, 0.14, 0.05, 0.30],
        }
    )


def test_compound_degradation_summarises_and_orders_compounds():
    summary = compound_degradation(make_slopes())
    assert summary["Compound"].tolist() == ["SOFT", "MEDIUM", "HARD"]
    assert summary["Stints"].tolist() == [3, 1, 1]
    assert summary["MedianSlope"].tolist() == pytest.approx([0.14, 0.05, 0.02])
    assert summary["MeanSlope"].tolist() == pytest.approx([0.18, 0.05, 0.02])
    assert summary["MinSlope"].tolist() == pytest.approx([0.10, 0.05, 0.02])
    assert summary["MaxSlope"].tolist() == pytest.approx([0.30, 0.05, 0.02])


def test_compound_degradation_handles_empty_table():
    empty = pd.DataFrame({"Compound": [], "Slope": []})
    assert compound_degradation(empty).empty


def test_compound_degradation_missing_column_raises():
    with pytest.raises(ValueError, match="Slope"):
        compound_degradation(make_slopes().drop(columns="Slope"))


def make_fuel_laps():
    """
    Two drivers, three stints each, built from an exact formula:
    lap time = driver base + compound offset + slope * tyre age - 0.05 * lap.
    SOFT slope 0.10, HARD slope 0.03. One VER lap is under safety car.
    """
    bases = {"VER": 90.0, "LEC": 91.0}
    plan = [
        ("SOFT", 2, 10, 0.10, 0.0),
        ("HARD", 12, 18, 0.03, 0.5),
        ("SOFT", 30, 10, 0.10, 0.0),
    ]
    names = ["Driver", "Compound", "LapNumber", "TyreLife", "LapTimeSeconds"]
    columns = {name: [] for name in names}
    for driver, base in bases.items():
        for compound, first_lap, n_laps, slope, offset in plan:
            for i in range(n_laps):
                lap_number = first_lap + i
                tyre_life = float(i + 1)
                columns["Driver"].append(driver)
                columns["Compound"].append(compound)
                columns["LapNumber"].append(lap_number)
                columns["TyreLife"].append(tyre_life)
                columns["LapTimeSeconds"].append(
                    base + offset + slope * tyre_life - 0.05 * lap_number
                )
    laps = pd.DataFrame(columns)

    slow = (laps["Driver"] == "VER") & (laps["LapNumber"] == 6)
    laps["TrackStatus"] = "1"
    laps.loc[slow, "TrackStatus"] = "4"
    laps.loc[slow, "LapTimeSeconds"] = 300.0

    no_time = pd.to_timedelta(pd.Series([None] * len(laps), dtype=float), unit="s")
    laps["LapTime"] = pd.to_timedelta(laps["LapTimeSeconds"], unit="s")
    laps["PitInTime"] = no_time
    laps["PitOutTime"] = no_time
    laps["Deleted"] = False
    return laps


def test_fuel_model_recovers_tyre_slopes_and_lap_number_effect():
    result = fuel_corrected_degradation(make_fuel_laps())
    assert result["Term"].tolist() == ["SOFT", "HARD", "LapNumber"]
    assert result["Slope"].tolist() == pytest.approx([0.10, 0.03, -0.05], abs=1e-6)
    assert result["Laps"].tolist() == [39, 36, 75]  # the safety-car lap is out


def test_fuel_model_leaves_out_compounds_with_few_laps():
    result = fuel_corrected_degradation(make_fuel_laps(), min_laps=37)
    assert result["Term"].tolist() == ["SOFT", "LapNumber"]


def test_fuel_model_cannot_separate_effects_in_one_stint():
    laps = make_fuel_laps()
    one_stint = laps[(laps["Driver"] == "LEC") & (laps["LapNumber"] <= 11)]
    with pytest.raises(ValueError, match="separate"):
        fuel_corrected_degradation(one_stint)


def test_fuel_model_missing_column_raises():
    with pytest.raises(ValueError, match="LapNumber"):
        fuel_corrected_degradation(make_fuel_laps().drop(columns="LapNumber"))
