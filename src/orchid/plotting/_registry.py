"""Shared singletons used by both DashPlotter (_dash.py) and BrowseApp (_browse.py).

Kept in a separate module so neither file needs to import from the other,
which would create a circular dependency.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._dash import DashPlotter
    from ._browse import BrowseApp

import numpy as _np

# ── Port registries ────────────────────────────────────────────────────────────
# Each maps port → live server instance so _start_server() can auto-stop any
# previous server on the same port before binding.

_port_registry:   dict[int, "DashPlotter"] = {}
_browse_registry: dict[int, "BrowseApp"]   = {}


# ── JSON encoder ───────────────────────────────────────────────────────────────

def _json_encode(x):
    """JSON ``default`` handler: converts numpy arrays and scalars to Python types."""
    if isinstance(x, _np.ndarray):
        return x.tolist()
    if isinstance(x, _np.generic):   # np.float32, np.int64, etc.
        return x.item()
    raise TypeError(f"Object of type {type(x).__name__} is not JSON serializable")
