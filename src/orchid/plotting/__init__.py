"""Live plotting for experiments — backend-agnostic base with swappable servers."""

from ._spec import EventLineConfig, PlotSpec, PostResult
from ._base import PlotterBase
from ._dash import DashPlotter, LivePlotter, _lp_line_trace_info, _lp_has_rail, _lp_rail_children, _lp_header
from ._browse import BrowseApp

__all__ = [
    "EventLineConfig",
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
