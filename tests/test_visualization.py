"""Tests for the plotting functions."""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

from f1_telemetry.visualization import (
    COMPOUND_COLORS,
    plot_pace_distribution,
    plot_tyre_strategy,
)


def make_inputs():
    """Two drivers; the pace table lists LEC first on purpose."""
    clean = pd.DataFrame(
        {
            "Driver": ["VER", "VER", "LEC", "LEC"],
            "LapTimeSeconds": [90.0, 91.0, 92.0, 93.0],
        }
    )
    pace = pd.DataFrame({"Driver": ["LEC", "VER"]})
    return clean, pace


def test_pace_plot_returns_a_figure_with_one_axes():
    fig = plot_pace_distribution(*make_inputs())
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    plt.close(fig)


def test_pace_plot_follows_pace_order_and_has_labels():
    fig = plot_pace_distribution(*make_inputs(), title="Test race")
    ax = fig.axes[0]
    assert [label.get_text() for label in ax.get_xticklabels()] == ["LEC", "VER"]
    assert ax.get_title() == "Test race"
    assert ax.get_ylabel() == "Lap time (s)"
    plt.close(fig)


def make_stints():
    """Three stints: LEC has two (MEDIUM then HARD), VER has one (SOFT)."""
    return pd.DataFrame(
        {
            "Driver": ["LEC", "LEC", "VER"],
            "Stint": [1, 2, 1],
            "Compound": ["MEDIUM", "HARD", "SOFT"],
            "FirstLap": [1, 21, 1],
            "LastLap": [20, 57, 57],
            "Laps": [20, 37, 57],
        }
    )


def test_strategy_plot_follows_driver_order_with_first_driver_on_top():
    fig = plot_tyre_strategy(make_stints(), ["VER", "LEC"], title="Test race")
    ax = fig.axes[0]
    assert [label.get_text() for label in ax.get_yticklabels()] == ["VER", "LEC"]
    assert ax.yaxis_inverted()
    assert ax.get_title() == "Test race"
    plt.close(fig)


def test_strategy_plot_draws_only_the_listed_drivers():
    fig = plot_tyre_strategy(make_stints(), ["LEC"])
    assert len(fig.axes[0].patches) == 2  # the two stints of LEC
    plt.close(fig)


def test_strategy_plot_colours_bars_by_compound():
    fig = plot_tyre_strategy(make_stints(), ["LEC", "VER"])
    colors = [patch.get_facecolor() for patch in fig.axes[0].patches]
    assert to_rgba(COMPOUND_COLORS["SOFT"]) in colors
    assert to_rgba(COMPOUND_COLORS["HARD"]) in colors
    plt.close(fig)


def test_strategy_plot_missing_column_raises():
    import pytest

    with pytest.raises(ValueError, match="Laps"):
        plot_tyre_strategy(make_stints().drop(columns="Laps"), ["LEC"])
