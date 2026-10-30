"""Tests for the data loader that do not need the network."""

import numpy as np
import pandas as pd
import pytest

from f1_telemetry import data_loader
from f1_telemetry.data_loader import (
    SessionLoadError,
    load_dataframe,
    load_session,
    resolve_race,
    save_dataframe,
    snapshot_path,
)


@pytest.fixture
def fake_calendar(monkeypatch):
    """Replace the real season calendar with a small hand-made one."""
    calendar = pd.DataFrame(
        {
            "RoundNumber": [1, 4, 7, 14],
            "EventName": [
                "Bahrain Grand Prix",
                "Emilia Romagna Grand Prix",
                "Monaco Grand Prix",
                "Italian Grand Prix",
            ],
            "Country": ["Bahrain", "Italy", "Monaco", "Italy"],
            "Location": ["Sakhir", "Imola", "Monaco", "Monza"],
        }
    )
    monkeypatch.setattr(data_loader, "list_races", lambda year: calendar)


def test_snapshot_path_follows_naming_rule(tmp_path):
    path = snapshot_path(2022, "Bahrain", "r", "laps", data_dir=tmp_path)
    assert path == tmp_path / "2022_bahrain_R_laps.csv"


def test_roundtrip_restores_timedeltas_and_driver_number(tmp_path):
    laps = pd.DataFrame(
        {
            "Driver": ["VER", "LEC"],
            "DriverNumber": ["1", "16"],
            "LapTime": pd.to_timedelta(pd.Series([95.123, np.nan]), unit="s"),
        }
    )
    path = save_dataframe(laps, tmp_path / "sub" / "laps.csv")
    loaded = load_dataframe(path, timedelta_columns=["LapTime"])
    assert loaded["DriverNumber"].tolist() == ["1", "16"]
    assert loaded["LapTime"].iloc[0].total_seconds() == pytest.approx(95.123)
    assert pd.isna(loaded["LapTime"].iloc[1])


def test_load_dataframe_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataframe(tmp_path / "missing.csv")


def test_load_session_rejects_unknown_session_type():
    with pytest.raises(SessionLoadError):
        load_session(2022, "Bahrain", "XYZ")


def test_resolve_race_by_name_location_and_round(fake_calendar):
    assert resolve_race(2022, "Monaco") == 7
    assert resolve_race(2022, "BAHRAIN") == 1
    assert resolve_race(2022, "imola") == 4
    assert resolve_race(2022, 14) == 14


def test_resolve_race_unknown_name_raises(fake_calendar):
    with pytest.raises(SessionLoadError, match="No race matching"):
        resolve_race(2022, "Atlantis")


def test_resolve_race_ambiguous_name_raises(fake_calendar):
    with pytest.raises(SessionLoadError, match="several races"):
        resolve_race(2022, "Italy")


def test_resolve_race_unknown_round_raises(fake_calendar):
    with pytest.raises(SessionLoadError, match="does not exist"):
        resolve_race(2022, 99)


def test_resolve_race_accepts_numpy_integer(fake_calendar):
    round_number = resolve_race(2022, np.int64(7))  # type: ignore
    assert round_number == 7
    assert type(round_number) is int


def test_resolve_race_rejects_other_types(fake_calendar):
    with pytest.raises(SessionLoadError):
        resolve_race(2022, 3.5)  # type: ignore[arg-type]
    with pytest.raises(SessionLoadError):
        resolve_race(2022, None)  # type: ignore[arg-type]
