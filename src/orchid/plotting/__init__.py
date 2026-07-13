"""Live plotting for experiments — backend-agnostic base with swappable servers."""

from ._spec import PlotSpec, PostResult
from ._base import PlotterBase
from ._dash import DashPlotter, LivePlotter
from ._browse import BrowseApp

__all__ = [
    "PlotSpec",
    "PostResult",
    "PlotterBase",
    "DashPlotter",
    "LivePlotter",
    "BrowseApp",
]

# Apply the default Plotly theme when this module is first imported.
try:
    from ..utils import apply_theme as _apply_default_theme
    _apply_default_theme()
except Exception:
    pass
