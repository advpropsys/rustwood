"""Shared publication-quality matplotlib style (ICML/ICLR-grade).

One consistent type scale, colorblind-safe palette (Okabe–Ito), clean spines, serif
body with LaTeX-style math, 300-dpi output. Import and call ``apply()`` at the top of
every plotting script so all figures match.
"""
import os

import matplotlib as mpl

# Okabe–Ito colorblind-safe qualitative palette, assigned consistently per system.
PALETTE = {
    "rustwood (B300)": "#D55E00",   # vermillion ("rust")
    "rustwood+TE":     "#E69F00",   # orange
    "rustwood-CPU":    "#9467bd",   # muted purple (CPU inference path)
    "XGBoost-GPU":     "#0072B2",   # blue
    "LightGBM-GPU":    "#009E73",   # bluish green
    "LightGBM-CPU":    "#009E73",
    "baseline (CPU)":    "#CC79A7",   # reddish purple
    "baseline (CPU)": "#CC79A7",
}


def color(name: str) -> str:
    """Palette lookup tolerant of label variants (case-insensitive substring)."""
    key = name.lower()
    table = [
        ("rustwood+te", "#E69F00"),
        ("rustwood-cpu", "#9467bd"),
        ("rustwood", "#D55E00"),
        ("gboost", "#D55E00"),
        ("xgboost", "#0072B2"),
        ("lightgbm", "#009E73"),
        ("baseline", "#CC79A7"),
    ]
    for token, c in table:
        if token in key:
            return c
    return "#444444"


def _register_inter():
    """Make sure matplotlib can find Inter (from ~/.fonts or the system)."""
    from matplotlib import font_manager
    if any("inter" == f.name.lower() for f in font_manager.fontManager.ttflist):
        return
    for d in (os.path.expanduser("~/.fonts"), "/usr/share/fonts"):
        if os.path.isdir(d):
            for p in font_manager.findSystemFonts(d):
                if "inter" in os.path.basename(p).lower():
                    try:
                        font_manager.fontManager.addfont(p)
                    except Exception:
                        pass


def apply():
    _register_inter()
    mpl.rcParams.update({
        # output
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "figure.constrained_layout.use": True,
        # fonts — Inter (clean modern sans), DejaVu Sans for math
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "DejaVu Sans", "Helvetica", "Arial"],
        "mathtext.fontset": "dejavusans",
        "font.size": 11,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 11.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "legend.title_fontsize": 10,
        # axes / spines
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        # grid
        "axes.grid": True,
        "grid.color": "0.85",
        "grid.linewidth": 0.6,
        "grid.alpha": 1.0,
        # lines / markers
        "lines.linewidth": 1.9,
        "lines.markersize": 5.5,
        "lines.markeredgewidth": 0.0,
        # legend
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "0.8",
        "legend.borderpad": 0.5,
        "legend.handlelength": 1.6,
        # ticks
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
    })
