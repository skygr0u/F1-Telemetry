"""Tests for the plotting functions."""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from f1_telemetry.visualization import plot_pace_distribution


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
