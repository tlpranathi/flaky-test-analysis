import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

# CHART COLOURS
# Fixed roles so every chart in the project reads the same way:
# Flaky is always orange, Non-Flaky is always blue.

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"

NON_FLAKY_COLOR = BLUE
FLAKY_COLOR = ORANGE

# Categorical order for model / ratio series (never cycled)

SERIES_COLORS = [BLUE, ORANGE, AQUA, YELLOW, "#e87ba4"]

# Sequential blue ramp (light -> dark) for magnitude

BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]

# 1. APPLY STYLE

def apply_style():
    """Set matplotlib defaults: light surface, recessive grid and axes."""

    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "xtick.labelcolor": TEXT_SECONDARY,
        "ytick.labelcolor": TEXT_SECONDARY,
        "legend.frameon": False,
        "legend.labelcolor": TEXT_SECONDARY,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 10,
        "lines.linewidth": 2,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })

# 2. SAVE FIGURE

def save_figure(fig, path):
    """Save a figure and close it."""

    fig.savefig(path)
    plt.close(fig)

    print("Chart saved to:", path)
