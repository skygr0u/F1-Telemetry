"""Load Formula 1 sessions through FastF1 and read or write CSV snapshots."""

from numbers import Integral
from pathlib import Path

import fastf1
import pandas as pd

# The project root is two folders above this file:
# <root>/src/f1_telemetry/data_loader.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "cache"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"

VALID_SESSION_TYPES = ("FP1", "FP2", "FP3", "Q", "SQ", "SS", "S", "R")

# Columns that FastF1 stores as timedeltas or datetimes. A CSV file stores them
# as text, so load_dataframe converts them back when asked.
LAP_TIMEDELTA_COLUMNS = [
    "Time",
    "LapTime",
    "LapStartTime",
    "PitOutTime",
    "PitInTime",
    "Sector1Time",
    "Sector2Time",
    "Sector3Time",
    "Sector1SessionTime",
    "Sector2SessionTime",
    "Sector3SessionTime",
]
LAP_DATETIME_COLUMNS = ["LapStartDate"]
RESULT_TIMEDELTA_COLUMNS = ["Q1", "Q2", "Q3", "Time"]


class SessionLoadError(Exception):
    """Raised when a session or a race calendar cannot be loaded."""


def setup_cache(cache_dir: str | Path = DEFAULT_CACHE_DIR) -> Path:
    """
    Create the cache folder and enable the FastF1 cache.

    The cache stores downloaded data on disk, so a session is downloaded only
    once and later runs are fast.

    Parameters
    ----------
    cache_dir : str or Path, optional
        Folder used for the cache. Defaults to ``cache/`` in the project root.

    Returns
    -------
    Path
        The cache folder.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    return cache_dir


def resolve_race(year: int, race: str | int) -> int:
    """
    Find the round number of a race from a name, location, country or round.

    FastF1 guesses the closest race name, so a typo can silently load a
    different race (for example 'Atlantis' became the Australian Grand Prix).
    This function accepts only one unambiguous match from the season calendar.

    Parameters
    ----------
    year : int
        Season, for example 2022.
    race : str or int
        Part of the race name, location or country (any letter case), or the
        round number as an integer (a NumPy integer is also accepted).

    Returns
    -------
    int
        The round number of the race.

    Raises
    ------
    SessionLoadError
        If ``race`` is neither a name nor a round number, nothing matches,
        several races match, or the round number does not exist.
    """
    if isinstance(race, Integral) and not isinstance(race, bool):
        round_number = int(race)
        if round_number in list_races(year)["RoundNumber"].tolist():
            return round_number
        raise SessionLoadError(f"Round {round_number} does not exist in {year}.")

    if not isinstance(race, str):
        raise SessionLoadError(
            f"The race must be a name or a round number, not {type(race).__name__}."
        )
    query = race.strip().lower()
    if not query:
        raise SessionLoadError("The race name is empty.")

    calendar = list_races(year)
    mask = pd.Series(False, index=calendar.index)
    for column in ["EventName", "Location", "Country"]:
        mask |= calendar[column].str.lower().str.contains(query, regex=False, na=False)
    matches = calendar[mask]

    if len(matches) == 1:
        return int(matches["RoundNumber"].iloc[0])
    if matches.empty:
        raise SessionLoadError(
            f"No race matching '{race}' in {year}. "
            f"Use list_races({year}) to see the calendar."
        )
    names = ", ".join(matches["EventName"])
    raise SessionLoadError(
        f"'{race}' matches several races in {year}: {names}. Be more specific."
    )


def load_session(
    year: int, grand_prix: str | int, session_type: str, messages: bool = True
):
    """
    Load one session of a Grand Prix weekend with lap and result data.

    The race is looked up in the season calendar with ``resolve_race`` and
    loaded by round number, so FastF1 never has to guess a name. Telemetry
    and weather are skipped because the analysis does not need them and they
    are the slowest part to download. Race control messages are kept by
    default, because the ``Deleted`` and ``DeletedReason`` columns of the laps
    are only filled when they are loaded.

    Parameters
    ----------
    year : int
        Season, for example 2022.
    grand_prix : str or int
        Part of the race name, location or country (for example "Bahrain" or
        "Monza"), or the round number.
    session_type : str
        One of FP1, FP2, FP3, Q, SQ, SS, S, R.
    messages : bool, optional
        Whether to load race control messages. Defaults to True.

    Returns
    -------
    fastf1.core.Session
        The loaded FastF1 session.

    Raises
    ------
    SessionLoadError
        If the session type is unknown, the race cannot be identified, the
        session does not exist, or the data cannot be downloaded.
    """
    session_type = session_type.upper()
    if session_type not in VALID_SESSION_TYPES:
        raise SessionLoadError(
            f"Unknown session type '{session_type}'. "
            f"Use one of: {', '.join(VALID_SESSION_TYPES)}."
        )
    round_number = resolve_race(year, grand_prix)
    try:
        session = fastf1.get_session(year, round_number, session_type)
        session.load(laps=True, telemetry=False, weather=False, messages=messages)
    except ValueError as exc:
        raise SessionLoadError(
            f"No {session_type} session found for {year} round "
            f"{round_number} ('{grand_prix}')."
        ) from exc
    except Exception as exc:  # network errors and FastF1 internal errors
        raise SessionLoadError(
            f"Could not load {year} '{grand_prix}' ({session_type}): {exc}"
        ) from exc
    return session


def list_races(year: int) -> pd.DataFrame:
    """
    List the Grand Prix of a season, so that any race can be chosen.

    Parameters
    ----------
    year : int
        Season, for example 2022.

    Returns
    -------
    pandas.DataFrame
        One row per race, with the columns RoundNumber, EventName, Country,
        Location, EventDate and EventFormat.

    Raises
    ------
    SessionLoadError
        If the calendar cannot be loaded.
    """
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
    except Exception as exc:
        raise SessionLoadError(f"Could not load the {year} calendar: {exc}") from exc
    columns = [
        "RoundNumber",
        "EventName",
        "Country",
        "Location",
        "EventDate",
        "EventFormat",
    ]
    return schedule[columns].reset_index(drop=True)


def get_laps(session) -> pd.DataFrame:
    """
    Return the laps of a loaded session as a plain DataFrame.

    Parameters
    ----------
    session : fastf1.core.Session
        A loaded FastF1 session.

    Returns
    -------
    pandas.DataFrame
        One row per driver per lap, as a new DataFrame.
    """
    return pd.DataFrame(session.laps).reset_index(drop=True)


def get_results(session) -> pd.DataFrame:
    """
    Return the official results of a loaded session as a plain DataFrame.

    Parameters
    ----------
    session : fastf1.core.Session
        A loaded FastF1 session.

    Returns
    -------
    pandas.DataFrame
        One row per driver, with grid position and final position.
    """
    return pd.DataFrame(session.results).reset_index(drop=True)


def snapshot_path(
    year: int,
    grand_prix: str,
    session_type: str,
    kind: str,
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> Path:
    """
    Build the path of a CSV snapshot, e.g. ``2022_bahrain_R_laps.csv``.

    Parameters
    ----------
    year : int
        Season.
    grand_prix : str
        Name of the Grand Prix.
    session_type : str
        Session code, for example "R".
    kind : str
        What the file contains, for example "laps" or "results".
    data_dir : str or Path, optional
        Folder for snapshots. Defaults to ``data/raw/``.

    Returns
    -------
    Path
        The full path of the snapshot file.
    """
    slug = grand_prix.strip().lower().replace(" ", "_")
    return Path(data_dir) / f"{year}_{slug}_{session_type.upper()}_{kind}.csv"


def save_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    """
    Save a DataFrame as CSV, creating missing folders.

    Parameters
    ----------
    df : pandas.DataFrame
        The table to save.
    path : str or Path
        Destination file.

    Returns
    -------
    Path
        The path of the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_dataframe(
    path: str | Path,
    timedelta_columns: list[str] | None = None,
    datetime_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Read a CSV snapshot and restore time columns.

    Parameters
    ----------
    path : str or Path
        CSV file to read.
    timedelta_columns : list of str, optional
        Columns to convert back to timedeltas, for example
        ``LAP_TIMEDELTA_COLUMNS``. Names that are not in the file are ignored.
    datetime_columns : list of str, optional
        Columns to convert back to datetimes.

    Returns
    -------
    pandas.DataFrame
        The table, with ``DriverNumber`` kept as text as in FastF1.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Snapshot not found: {path}. Create it with save_dataframe first."
        )
    df = pd.read_csv(path, dtype={"DriverNumber": str})
    for column in timedelta_columns or []:
        if column in df.columns:
            df[column] = pd.to_timedelta(df[column])
    for column in datetime_columns or []:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column])
    return df
