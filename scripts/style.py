"""Shared plotting style — clean, modern, editorial (Inter, minimal chrome).

Design: a single warm "hero" accent for rustwood among cool muted baselines; almost no
chrome (no tick marks, faint y-grid only, no box); left-aligned title + gray subtitle;
Inter throughout; generous whitespace; 300-dpi output. Import, call ``apply()``, and use
``title(ax, ...)`` / ``finish(ax)`` for consistent framing.
"""
import os

import matplotlib as mpl

# Editorial palette: rustwood is the only warm tone, so it reads as the protagonist;
# baselines are cool and muted so they recede without being hidden.
PALETTE = {
    "rustwood": "#E8613C",      # terracotta / "rust"
    "rustwood+TE": "#F1A340",   # warm amber
    "rustwood-CPU": "#7C6BAE",  # muted violet (CPU path)
    "rustwood-fast": "#F1A340",
    "xgboost": "#33506B",       # deep slate blue
    "lightgbm": "#5FA08C",      # muted teal
    "baseline": "#B58DB6",        # muted mauve
}
INK = "#15202B"      # title / emphasis
MUTED = "#5B6573"    # labels, ticks, subtitles
FAINT = "#9AA3AF"    # captions
GRID = "#ECEEF1"


def color(name: str) -> str:
    key = name.lower()
    for token, c in (
        ("rustwood+te", PALETTE["rustwood+TE"]),
        ("rustwood-cpu", PALETTE["rustwood-CPU"]),
        ("fast", PALETTE["rustwood-fast"]),
        ("xgboost", PALETTE["xgboost"]),  # before "rustwood": "xgboost" contains "gboost"
        ("rustwood", PALETTE["rustwood"]),
        ("xgb", PALETTE["xgboost"]),
        ("lightgbm", PALETTE["lightgbm"]),
        ("baseline", PALETTE["baseline"]),
    ):
        if token in key:
            return c
    return "#8A929E"


def _register_inter():
    from matplotlib import font_manager
    if any(f.name == "Inter" for f in font_manager.fontManager.ttflist):
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
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.28,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "figure.constrained_layout.use": True,
        # type
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "DejaVu Sans", "Helvetica", "Arial"],
        "mathtext.fontset": "dejavusans",
        "font.size": 11,
        "text.color": INK,
        "axes.titlesize": 13.5,
        "axes.titleweight": "semibold",
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "axes.titlepad": 18,
        "axes.labelsize": 10.5,
        "axes.labelcolor": MUTED,
        "axes.labelpad": 7,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        # minimal chrome — no box, no tick marks
        "axes.edgecolor": "#D7DBE0",
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": True,
        "axes.axisbelow": True,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.pad": 6,
        "ytick.major.pad": 6,
        # faint horizontal grid only
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 1.1,
        # lines / legend
        "lines.linewidth": 2.2,
        "lines.markersize": 5,
        "lines.markeredgewidth": 0.0,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "legend.handlelength": 1.3,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.3,
        "legend.labelspacing": 0.35,
    })


def title(ax, main, subtitle=None):
    """Left-aligned bold title with an optional gray subtitle above the axes."""
    ax.set_title(main)
    if subtitle:
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=9.5, color=MUTED)


def legend(ax, **kw):
    """Frameless legend with sensible defaults."""
    kw.setdefault("frameon", False)
    kw.setdefault("loc", "best")
    return ax.legend(**kw)
