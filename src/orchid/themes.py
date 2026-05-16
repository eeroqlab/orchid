"""Theme definitions for the DashPlotter.

Each theme is a flat dict consumed by two layers kept in lock-step:

1. DOM chrome — ``#lp-root`` carries a ``theme-<key>`` className.  The
   matching block in ``assets/style.css`` sets CSS custom properties
   (``--surface``, ``--panel``, ``--accent``, ``--trace-*``, …).

2. Plotly figures — ``plotly_template(theme)`` returns a layout dict
   merged into each figure when it is built or re-themed.

To add a new theme: add an entry here **and** a matching
``.theme-<key> { … }`` block in ``assets/style.css``.
"""

from __future__ import annotations

THEMES: dict[str, dict] = {
    "orchid": {
        "name": "Orchid",
        "sub": "Default · violet",
        "mode": "light",
        "surface": "#f2f0f7",
        "panel": "#ffffff",
        "panel_border": "#ddd8ea",
        "hairline": "#ddd8ea",
        "ink": "#1a1628",
        "ink_mute": "#5c5470",
        "ink_faint": "#9b96ab",
        "accent": "#6d28d9",
        "accent_soft": "rgba(109, 40, 217, 0.10)",
        "on_accent": "#ffffff",
        "ok": "#16a34a",
        "err": "#dc2626",
        "traces": ["#4878d0", "#ee854a", "#6acc64"],
        "colorscale": [
            [0.00, "#f2f0f7"],
            [0.25, "#c4b5fd"],
            [0.50, "#8b5cf6"],
            [0.75, "#6d28d9"],
            [1.00, "#2e1065"],
        ],
        "plot_bg": "#ffffff",
    },
    "t1000": {
        "name": "T1000",
        "sub": "Braun homage · warm cream",
        "mode": "light",
        "surface": "#e8e2d4",
        "panel": "#f1ecde",
        "panel_border": "#d2cab8",
        "hairline": "#c8c0ac",
        "ink": "#1f1b15",
        "ink_mute": "#5a5142",
        "ink_faint": "#8b8170",
        "accent": "#cc5500",
        "accent_soft": "rgba(204, 85, 0, 0.12)",
        "on_accent": "#fff",
        "ok": "#4a6b2a",
        "err": "#a83a1f",
        "traces": ["#1f1b15", "#cc5500", "#4a6b2a"],
        "colorscale": [
            [0.00, "#f1ecde"],
            [0.25, "#e0c48c"],
            [0.50, "#cc5500"],
            [0.78, "#781e0a"],
            [1.00, "#1f1b15"],
        ],
        "plot_bg": "#fafaf6",
    },
    "vitsoe": {
        "name": "Vitsœ",
        "sub": "Office calm · oxblood accent",
        "mode": "light",
        "surface": "#f4f0e8",
        "panel": "#f4f0e8",
        "panel_border": "transparent",
        "hairline": "#d8d2c4",
        "ink": "#1a1714",
        "ink_mute": "#6b6457",
        "ink_faint": "#9a9384",
        "accent": "#8a2a1f",
        "accent_soft": "rgba(138, 42, 31, 0.10)",
        "on_accent": "#fff",
        "ok": "#3f5a2a",
        "err": "#8a2a1f",
        "traces": ["#1a1714", "#8a2a1f", "#6b6457"],
        "colorscale": [
            [0.00, "#f4f0e8"],
            [0.25, "#e6d2be"],
            [0.50, "#c88c6e"],
            [0.78, "#8a2a1f"],
            [1.00, "#28120c"],
        ],
        "plot_bg": "#fafaf6",
    },
    "modern": {
        "name": "Functional",
        "sub": "Modern neutral · green accent",
        "mode": "light",
        "surface": "#f6f6f4",
        "panel": "#ffffff",
        "panel_border": "#e5e4e0",
        "hairline": "#e5e4e0",
        "ink": "#16181a",
        "ink_mute": "#5e6166",
        "ink_faint": "#9aa0a6",
        "accent": "#006a52",
        "accent_soft": "rgba(0, 106, 82, 0.10)",
        "on_accent": "#fff",
        "ok": "#006a52",
        "err": "#b3261e",
        "traces": ["#16181a", "#006a52", "#7a4cc6"],
        "colorscale": [
            [0.00, "#ffffff"],
            [0.25, "#c8e4da"],
            [0.50, "#50aa8c"],
            [0.78, "#006a52"],
            [1.00, "#0a2820"],
        ],
        "plot_bg": "#ffffff",
    },
    "console": {
        "name": "Console",
        "sub": "Dark lab · amber accent",
        "mode": "dark",
        "surface": "#16140f",
        "panel": "#1f1c16",
        "panel_border": "#2e2a22",
        "hairline": "#2e2a22",
        "ink": "#e8e0cf",
        "ink_mute": "#a39a87",
        "ink_faint": "#6b6557",
        "accent": "#e0922f",
        "accent_soft": "rgba(224, 146, 47, 0.14)",
        "on_accent": "#1a1612",
        "ok": "#8ec07c",
        "err": "#e07b5a",
        "traces": ["#e8e0cf", "#e0922f", "#8ec07c"],
        "colorscale": [
            [0.00, "#16140f"],
            [0.25, "#3c2814"],
            [0.50, "#a05a1e"],
            [0.78, "#e0922f"],
            [1.00, "#fce8b4"],
        ],
        "plot_bg": "#0e0c08",
    },
}


def plotly_template(theme: dict) -> dict:
    """Return a Plotly layout dict that themes any figure."""
    grid = "rgba(255,255,255,0.06)" if theme["mode"] == "dark" else "rgba(0,0,0,0.06)"
    axis = dict(
        showgrid=True,
        gridcolor=grid,
        zeroline=False,
        linecolor=theme["ink"],
        linewidth=1,
        mirror=False,
        ticks="outside",
        ticklen=4,
        tickwidth=1,
        tickcolor=theme["ink"],
        tickfont=dict(color=theme["ink_mute"], size=11),
        title=dict(font=dict(color=theme["ink"], size=11), standoff=8),
    )
    return dict(
        paper_bgcolor=theme["panel"],
        plot_bgcolor=theme["plot_bg"],
        font=dict(
            family='"Helvetica Neue", Helvetica, Arial, sans-serif',
            color=theme["ink"],
            size=11,
        ),
        colorway=theme["traces"],
        margin=dict(l=64, r=18, t=18, b=46),
        xaxis=axis,
        yaxis=axis,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=1,
            xanchor="right",
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=theme["ink_mute"], size=10),
        ),
        hoverlabel=dict(
            bgcolor=theme["panel"],
            bordercolor=theme["hairline"],
            font=dict(color=theme["ink"], family='"Helvetica Neue", Arial, sans-serif'),
        ),
        modebar=dict(
            bgcolor="rgba(0,0,0,0)",
            color=theme["ink_mute"],
            activecolor=theme["accent"],
        ),
    )
