"""Tests for the plotting functions."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

from f1_telemetry.visualization import (
    COMPOUND_COLORS,
    plot_degradation_comparison,
    plot_pace_distribution,
    plot_stint_fit,
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


def make_fit_inputs():
    """VER stint 1 has four clean laps and a fitted line; LEC has two laps."""
    clean = pd.DataFrame(
        {
            "Driver": ["VER"] * 4 + ["LEC"] * 2,
            "Stint": [1] * 6,
            "TyreLife": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0],
            "LapTimeSeconds": [90.1, 90.2, 90.3, 90.4, 91.0, 91.1],
        }
    )
    fits = pd.DataFrame(
        {
            "Driver": ["VER"],
            "Stint": [1],
            "Compound": ["SOFT"],
            "Slope": [0.1],
            "Intercept": [90.0],
        }
    )
    return clean, fits


def test_stint_fit_plot_draws_the_laps_and_the_fitted_line():
    clean, fits = make_fit_inputs()
    fig = plot_stint_fit(clean, fits, "VER", 1)
    ax = fig.axes[0]
    points = np.asarray(ax.collections[0].get_offsets())
    line_y = np.asarray(ax.lines[0].get_ydata())
    assert len(points) == 4  # only VER's four laps
    assert line_y.tolist() == pytest.approx([90.1, 90.4])
    assert "VER" in ax.get_title()
    plt.close(fig)


def test_stint_fit_plot_for_missing_stint_raises():
    clean, fits = make_fit_inputs()
    with pytest.raises(ValueError, match="no fitted stint"):
        plot_stint_fit(clean, fits, "LEC", 1)


def make_comparison_inputs():
    """MEDIUM is missing from the model, LapNumber is not a compound."""
    naive = pd.DataFrame(
        {
            "Compound": ["SOFT", "MEDIUM", "HARD"],
            "MedianSlope": [0.05, 0.03, 0.01],
        }
    )
    corrected = pd.DataFrame(
        {
            "Term": ["SOFT", "HARD", "LapNumber"],
            "Slope": [0.10, 0.04, -0.05],
            "StdErr": [0.01, 0.01, 0.002],
        }
    )
    return naive, corrected


def test_comparison_plot_draws_two_bars_for_each_common_compound():
    fig = plot_degradation_comparison(*make_comparison_inputs())
    ax = fig.axes[0]
    assert [label.get_text() for label in ax.get_xticklabels()] == ["SOFT", "HARD"]
    assert len(ax.patches) == 4
    plt.close(fig)


def test_comparison_plot_missing_column_raises():
    naive, corrected = make_comparison_inputs()
    with pytest.raises(ValueError, match="StdErr"):
        plot_degradation_comparison(naive, corrected.drop(columns="StdErr"))
