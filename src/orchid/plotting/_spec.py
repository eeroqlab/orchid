"""Plotting data-classes and pure helper functions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..controller import DataKind


@dataclass
class EventLineConfig:
    """Visual properties for parameter-change event markers on time-series plots.

    Parameters
    ----------
    color : str
        Line and label font color. Any CSS/plotly color string.
    width : int
        Line width in pixels.
    dash : str
        Line style: ``"solid"``, ``"dot"``, ``"dash"``, ``"longdash"``, ``"dashdot"``.
    font_size : int
        Label font size in points.
    bgcolor : str
        Label box background color. Use ``rgba(r,g,b,a)`` for transparency.
    bordercolor : str
        Label box border color.
    borderwidth : int
        Label box border width in pixels.
    borderpad : int
        Padding in pixels between the label text and the box border.

    Examples
    --------
    >>> EventLineConfig(color="#444444", dash="dot", width=2)
    >>> EventLineConfig(color="#2255cc", bgcolor="rgba(255,255,255,0.0)")  # no box
    """

    color: str = "#444444"
    width: int = 2
    dash: str = "dash"
    font_size: int = 15
    bgcolor: str = "rgba(255,255,255,0.85)"
    bordercolor: str = "#000000"
    borderwidth: int = 1
    borderpad: int = 3


@dataclass
class PlotSpec:
    """Describes one subplot in a LivePlotter.

    Parameters
    ----------
    x : str or array-like
        Sweep parameter name (str) or fixed axis values (array).
        For line / heatmap / monitor: a string parameter name or ``"_time"``.
        For **live_trace**: a fixed array of axis values (e.g. frequencies).
            The plot type is auto-detected when ``x`` is an array.
    y : str or list of str or array-like
        For line plots: readout name, or a list of readout names to overlay.
        For heatmaps: outer sweep parameter name.
        For **trace_heatmap**: a fixed array of axis values (e.g. frequencies).
            The plot type is auto-detected when ``y`` is an array.
    z : str, optional
        Readout name for the color axis.
        Required for ``"heatmap"`` and ``"trace_heatmap"``. Ignored for line.
    y_col : int, str, list[int | str], or None
        Column selector for TRACE or IMAGE readouts on ``"line"`` and
        ``"live_trace"`` plots.

        - ``None``: not used (SCALAR readouts) or use whole array (TRACE
          ``live_trace``).
        - ``int``: zero-based channel index.
        - ``str``: channel name, resolved via ``readout.contains``.
        - ``list``: multiple channels → one trace per channel on the same
          subplot.

        For TRACE shape ``(N,)``: ``y_col`` indexes elements → scalar per
        point.
        For IMAGE shape ``(N_channels, N_pts)``: ``y_col`` indexes the first
        (channel) axis → 1D array per point for ``live_trace``, scalar per
        point for ``line``.
    z_col : int or str or None
        Column selector for IMAGE readouts on ``"heatmap"`` and
        ``"trace_heatmap"`` plots (color axis). Same int/str semantics as
        ``y_col``. Ignored for SCALAR and TRACE readouts.
    plot_type : str
        ``"line"``, ``"heatmap"``, ``"live_trace"``, ``"trace_heatmap"``,
        or ``"auto"`` (default).

        ``"auto"`` resolves using DataKind and context:

        ============================  ==========================
        condition                     resolved type
        ============================  ==========================
        ``x`` is array                ``live_trace``
        ``y`` is array                ``trace_heatmap``
        ``z`` set, Procedure ≥2D      ``heatmap``
        ``y_col`` set or SCALAR ``y`` ``line``
        ============================  ==========================

    update_every : str or None
        ``"point"``, ``"sweep"``, or ``"plane"``.
        ``None`` (default) — auto-determined from the resolved plot type:
        ``live_trace`` / ``trace_heatmap`` → ``"point"``;
        ``heatmap`` → ``"sweep"`` (3D → ``"plane"``);
        ``line`` → ``"point"``.
    update_func : callable, optional
        Custom update function ``(fig_dict, index, data) -> None``.
        Overrides built-in update logic when set.

    Examples
    --------
    Line — 1D sweep, SCALAR readout::

        PlotSpec(x="Vgt", y="lockin_X")

    Line — 1D sweep, lock-in returning [X, Y], both on same subplot::

        PlotSpec(x="Vgt", y="lockin_XY", y_col=["X", "Y"])

    Line — monitor, lock-in X vs time::

        PlotSpec(x="_time", y="lockin_XY", y_col="X")

    Heatmap — 2D sweep::

        PlotSpec(x="fac", y="Vgt", z="lockin_X")

    Live trace — VNA magnitude waveform refreshing each gate step::

        PlotSpec(x=freqs, y="vna", y_col="mag")

    Trace heatmap — accumulate VNA magnitude vs gate voltage::

        PlotSpec(x="Vgt", y=freqs, z="vna", z_col="mag")

    Trace heatmap — plain TRACE readout (no z_col needed)::

        PlotSpec(x="Vgt", y=freqs, z="mag")
    """

    x: str | np.ndarray | list
    y: str | list[str] | np.ndarray | list
    z: str | None = None
    y_col: int | str | list[int | str] | None = None
    z_col: int | str | None = None
    plot_type: str = "auto"
    update_every: str | None = None
    update_func: Callable | None = None
    colorscale: str | list | None = None  # heatmap / trace_heatmap only


def _y_list(spec: PlotSpec) -> list[str]:
    """Normalise PlotSpec.y to a list of readout name strings.

    Returns an empty list when y is an array (trace_heatmap axis values).
    """
    if isinstance(spec.y, np.ndarray):
        return []
    if isinstance(spec.y, list):
        if spec.y and isinstance(spec.y[0], str):
            return spec.y
        return []
    return [spec.y]


def _is_array(v) -> bool:
    """True when v is a numeric array-like axis specification.

    A list of strings is a list of readout names, not axis values, so it
    returns False.  Only numpy arrays and lists of numbers return True.
    """
    if isinstance(v, np.ndarray):
        return True
    if isinstance(v, list) and v and not isinstance(v[0], str):
        return True
    return False


def _sweep_ctrl_names(sweep) -> list[str]:
    """Controller names for a Sweep or MultiSweep."""
    if hasattr(sweep, 'controllers'):
        return [c.name for c in sweep.controllers]
    return [sweep.controller.name]


def _find_sweep_by_ctrl(proc, name: str):
    """Return the sweep whose (first) controller matches *name*, or None."""
    for s in getattr(proc, 'sweeps', []):
        if name in _sweep_ctrl_names(s):
            return s
    return None


def _resolve_plot_type(spec: PlotSpec, proc) -> str:
    """Resolve ``"auto"`` to a concrete plot type."""
    if spec.plot_type != 'auto':
        return spec.plot_type
    if _is_array(spec.x):
        return 'live_trace'
    if _is_array(spec.y):
        return 'trace_heatmap'
    if spec.z and hasattr(proc, 'sweeps') and len(proc.sweeps) >= 2:
        return 'heatmap'
    return 'line'


def _default_update_every(resolved_type: str, proc) -> str:
    """Choose a sensible default ``update_every`` from the resolved plot type."""
    if resolved_type in ('live_trace', 'trace_heatmap'):
        return 'point'
    if resolved_type == 'heatmap':
        return 'plane' if len(getattr(proc, 'sweeps', [])) >= 3 else 'sweep'
    return 'point'


def _resolve_col(col, readout, label: str = 'col') -> int | None:
    """Resolve a column specifier (int / str / None) to an integer index or None."""
    if col is None:
        return None
    if isinstance(col, int):
        return col
    if isinstance(col, str):
        if isinstance(readout.contains, list):
            try:
                return readout.contains.index(col)
            except ValueError:
                raise ValueError(
                    f"{label}='{col}' not found in readout.contains="
                    f"{readout.contains} for readout '{readout.name}'"
                )
        raise ValueError(
            f"{label}='{col}' is a string but readout '{readout.name}' "
            "has no contains list"
        )
    raise TypeError(f"{label} must be int, str, or None, got {type(col)}")


def _normalize_y_col(y_col, readout) -> list[int] | None:
    """Normalize ``y_col`` to ``list[int]`` or ``None``.

    Returns ``None`` for SCALAR readouts, or when ``y_col`` is ``None``
    (whole-array live_trace, direct SCALAR read).  Returns a list with one
    element for single-column extraction; multi-element for multi-trace.
    """
    if readout is None or readout.kind == DataKind.SCALAR:
        return None
    if y_col is None:
        return None
    cols = y_col if isinstance(y_col, list) else [y_col]
    return [_resolve_col(c, readout, 'y_col') for c in cols]


def _deep_merge(target: dict, source: dict) -> dict:
    """Recursively merge *source* into *target* in-place. Returns *target*."""
    for key, val in source.items():
        if isinstance(val, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], val)
        else:
            target[key] = val
    return target


_EV_PALETTE = [
    "#4878d0", "#ee854a", "#6acc64", "#d65f5f",
    "#956cb4", "#8c613c", "#d5bb67", "#82c6e2",
]


def _format_elapsed_display(elapsed_s: float) -> str:
    """Format elapsed seconds as ``MM:SS`` or ``H:MM:SS``."""
    h = int(elapsed_s // 3600)
    m = int((elapsed_s % 3600) // 60)
    s = int(elapsed_s % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _format_eta(seconds: float) -> str:
    """Format a duration as ``MM m SS s`` / ``H h MM m`` / ``SS s``."""
    if seconds < 60:
        return f"{int(seconds)} s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h} h {m:02d} m"
    return f"{m:02d} m {s:02d} s"
