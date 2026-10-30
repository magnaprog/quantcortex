from __future__ import annotations

import matplotlib as mpl

from quantcortex.backtest.metrics.plotting import (
    BRIGHT_COLORS,
    contrasting_text_color,
    plot_style_context,
    return_diverging_colormap,
)


def test_plot_style_context_applies_and_restores_settings():
    original_font_size = mpl.rcParams["font.size"]
    original_font_family = mpl.rcParams["font.family"]
    original_math_fontset = mpl.rcParams["mathtext.fontset"]
    original_cycle = list(mpl.rcParams["axes.prop_cycle"])

    with plot_style_context("paper"):
        assert mpl.rcParams["font.size"] == 8.0
        assert mpl.rcParams["font.family"] == ["serif"]
        assert mpl.rcParams["font.serif"] == ["DejaVu Serif"]
        assert mpl.rcParams["mathtext.fontset"] == "stix"
        assert mpl.rcParams["pdf.fonttype"] == 42
        assert [entry["color"] for entry in mpl.rcParams["axes.prop_cycle"]] == list(
            BRIGHT_COLORS
        )

    assert mpl.rcParams["font.size"] == original_font_size
    assert mpl.rcParams["font.family"] == original_font_family
    assert mpl.rcParams["mathtext.fontset"] == original_math_fontset
    assert list(mpl.rcParams["axes.prop_cycle"]) == original_cycle


def test_contrasting_text_color_handles_light_and_dark_backgrounds():
    assert contrasting_text_color((1.0, 1.0, 1.0, 1.0)) == "black"
    assert contrasting_text_color((0.0, 0.0, 0.0, 1.0)) == "white"


def test_return_colormap_has_neutral_center_and_distinct_endpoints():
    colormap = return_diverging_colormap()
    low = colormap(0.0)
    center = colormap(0.5)
    high = colormap(1.0)

    assert min(center[:3]) > 0.9
    assert low[:3] != high[:3]
