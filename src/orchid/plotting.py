"""Live plotting for experiments — backend-agnostic base with swappable servers."""

from __future__ import annotations

import abc
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .controller import DataKind


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
    z_col : int or str or None
        Column selector for IMAGE or TRACE readouts used as ``z`` (or ``y``
        for ``live_trace``).
        - ``None`` (default): use the whole array (valid for TRACE readouts).
        - ``int``: column index.
        - ``str``: column name, resolved via ``readout.contains``.
        Ignored for SCALAR readouts.
    plot_type : str
        ``"line"``, ``"heatmap"``, ``"live_trace"``, ``"trace_heatmap"``,
        or ``"auto"`` (default).
        ``"auto"`` infers the type from the types of ``x`` and ``y``:

        ============  ============  ====================
        ``x``         ``y``         resolved type
        ============  ============  ====================
        str           str, ndim=1   line
        str           str, ndim≥2   heatmap
        array         str           live_trace
        str           array         trace_heatmap
        ============  ============  ====================

    update_every : str
        ``"point"`` — update after every measurement point (use for
        ``live_trace`` / ``trace_heatmap`` with a 1D sweep).
        ``"sweep"`` — update after each inner sweep completes (default).
        ``"plane"`` — update after each 2D plane completes.
    update_func : callable, optional
        Custom update function ``(fig_dict, index, data) -> None``.
        Overrides built-in update logic when set.

    Examples
    --------
    Line plot (1D sweep)::

        PlotSpec(x="Vgt", y="lockin_X")

    Heatmap (2D sweep)::

        PlotSpec(x="fac", y="Vgt", z="lockin_X")

    Live trace — current VNA trace refreshing each power step::

        PlotSpec(x=freqs, y="S21", z_col="mag", update_every="point")

    Trace heatmap — accumulate VNA traces vs power::

        PlotSpec(x="vna_power", y=freqs, z="S21", z_col="mag",
                 update_every="point")

    Trace heatmap from a plain TRACE readout (no z_col needed)::

        PlotSpec(x="vna_power", y=freqs, z="mag", update_every="point")
    """

    x: str | np.ndarray | list
    y: str | list[str] | np.ndarray | list
    z: str | None = None
    z_col: int | str | None = None
    plot_type: str = "auto"
    update_every: str = "sweep"
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
    """True when v is an array-like axis specification (not a string)."""
    return isinstance(v, (np.ndarray, list)) and not isinstance(v, str)


# ══════════════════════════════════════════════════════════════════════
#  PlotterBase — all data logic, no server code
# ══════════════════════════════════════════════════════════════════════


class PlotterBase(abc.ABC):
    """Abstract base class for live plotters.

    Holds all figure-building and data-update logic. Subclasses implement
    ``_start_server()``, ``stop()``, and ``is_running`` to provide the
    display backend (Dash, Taipy, etc.).

    Parameters
    ----------
    plots : list of PlotSpec
        Subplot specifications.
    height : int
        Figure height in pixels per subplot row.
    width : int
        Figure width in pixels.
    open_browser : bool
        If True, automatically open the plot when the server starts.
    event_line : EventLineConfig, optional
        Visual style for parameter-change event markers.
    max_display_pts : int
        Maximum points shown on line plots. For monitors this is the
        rolling-window size; for sweep plots the buffer is sized to the
        inner sweep length (always fits).
    """

    def __init__(
        self,
        plots: list[PlotSpec],
        height: int = 350,
        width: int = 700,
        open_browser: bool = True,
        event_line: EventLineConfig | None = None,
        max_display_pts: int = 5000,
    ):
        self.specs = plots
        self.height_per_plot = height
        self.width = width
        self.open_browser = open_browser
        self.event_line = event_line if event_line is not None else EventLineConfig()
        self.max_display_pts = max_display_pts

        # Internal state — populated by setup()
        self._fig_dict: dict | None = None
        self._proc = None
        self._resolved_types: list[str] = []
        self._sweep_data: dict[int, dict] = {}
        self._trace_offsets: list[int] = []
        self._stopped = False
        self._t0: float | None = None
        self._event_timestamps: list[float] = []

    # ── Abstract interface (implement in each backend) ─────────────────

    @abc.abstractmethod
    def _start_server(self) -> None:
        """Start the display server. Called once from setup()."""

    @abc.abstractmethod
    def stop(self, _silent: bool = False) -> None:
        """Stop the server and free resources."""

    @property
    @abc.abstractmethod
    def is_running(self) -> bool:
        """True if the server is currently running."""

    # ── Override hook ──────────────────────────────────────────────────

    def on_data_changed(self) -> None:
        """Called after every write to ``_fig_dict``.

        Override in subclasses to implement push (Taipy) or poll-version
        increment (Dash). The default implementation is a no-op.
        """

    # ── Lifecycle ──────────────────────────────────────────────────────

    def setup(self, proc) -> None:
        """Initialize figure and start server. Called once before each experiment."""
        self._proc = proc
        self._fig_dict = None
        self._resolved_types = []
        self._sweep_data = {}
        self._trace_offsets = []
        self._stopped = False
        self._t0 = None
        self._event_timestamps = []

        n = len(self.specs)

        # Resolve "auto" plot types
        self._resolved_types = []
        for spec in self.specs:
            if spec.plot_type == "auto":
                if _is_array(spec.x):
                    self._resolved_types.append("live_trace")
                elif _is_array(spec.y):
                    self._resolved_types.append("trace_heatmap")
                elif spec.z and hasattr(proc, "sweeps") and len(proc.sweeps) >= 2:
                    self._resolved_types.append("heatmap")
                else:
                    self._resolved_types.append("line")
            else:
                self._resolved_types.append(spec.plot_type)

        # Validate readout names
        if hasattr(proc, "bench"):
            registered = set(proc.bench.readouts.keys())
            for spec, ptype in zip(self.specs, self._resolved_types):
                if ptype == "heatmap":
                    readout_names = [spec.z] if spec.z else []
                elif ptype == "live_trace":
                    readout_names = [spec.y] if isinstance(spec.y, str) else []
                elif ptype == "trace_heatmap":
                    readout_names = [spec.z] if spec.z else []
                else:
                    readout_names = [y for y in _y_list(spec) if y != "_time"]
                for name in readout_names:
                    if name not in registered:
                        raise ValueError(
                            f"PlotSpec readout {name!r} is not registered. "
                            f"Registered: {sorted(registered)}"
                        )

        self._fig_dict = self.build_figure_dict(proc, n)

        # Start server only once — reuse across experiments so the browser
        # doesn't reconnect mid-session and briefly show stale data.
        if not self.is_running:
            self._start_server()

    def update_point(self, index: tuple, data: dict, sweep_values: dict) -> None:
        """Called by the runner after every measurement point."""
        self.dispatch("point", index, data, sweep_values)

    def update_sweep(self, outer_index: tuple, data: dict, sweep_values: dict) -> None:
        """Called by the runner after each inner sweep completes."""
        self.dispatch("sweep", outer_index, data, sweep_values)

    def update_plane(self, outer_index: tuple, data: dict, sweep_values: dict) -> None:
        """Called by the runner after each 2D plane completes."""
        self.dispatch("plane", outer_index, data, sweep_values)

    def dispatch(self, event: str, index, data, sweep_values) -> None:
        """Route update to each subplot whose ``update_every`` matches ``event``."""
        if self._fig_dict is None:
            return

        changed = False
        for i, (spec, ptype) in enumerate(zip(self.specs, self._resolved_types)):
            if spec.update_every != event:
                continue

            if spec.update_func is not None:
                spec.update_func(self._fig_dict, index, data)
                changed = True
                continue

            if ptype in ("heatmap", "trace_heatmap"):
                if spec.z not in data:
                    continue
            elif ptype == "live_trace":
                if not isinstance(spec.y, str) or spec.y not in data:
                    continue
            else:
                if not any(y in data for y in _y_list(spec)):
                    continue

            if ptype == "line":
                self.update_line(i, spec, data, sweep_values)
                changed = True
            elif ptype == "heatmap":
                self.update_heatmap(i, spec, index, data)
                changed = True
            elif ptype == "live_trace":
                self.update_live_trace(i, spec, data)
                changed = True
            elif ptype == "trace_heatmap":
                self.update_trace_heatmap(i, spec, index, data)
                changed = True

        if changed:
            self.on_data_changed()

    def update_monitor(self, sample_idx: int, data: dict, timestamp: float) -> None:
        """Update plots for monitoring mode. Called by the runner each sample.

        Uses a pre-allocated rolling numpy buffer of size ``max_display_pts``.
        When the buffer is full, the oldest sample is dropped (O(n) numpy
        shift in C — fast in practice) and the new sample is placed at the end.
        """
        if self._fig_dict is None:
            return

        if self._t0 is None:
            self._t0 = timestamp

        elapsed = timestamp - self._t0

        for i, (spec, ptype) in enumerate(zip(self.specs, self._resolved_types)):
            if spec.update_func is not None:
                spec.update_func(self._fig_dict, sample_idx, data)
                continue

            y_names = _y_list(spec)
            if not any(y in data for y in y_names):
                continue

            if ptype == "line":
                states = self._sweep_data[i]

                if spec.x == "_time":
                    x_val, unit = self.format_elapsed(elapsed)
                    axis_key = f"xaxis{i + 1}" if i > 0 else "xaxis"
                    current_label = self._fig_dict["layout"].get(axis_key, {}).get("title", {})
                    new_label = f"Time ({unit})"
                    if current_label.get("text") != new_label:
                        if axis_key not in self._fig_dict["layout"]:
                            self._fig_dict["layout"][axis_key] = {}
                        self._fig_dict["layout"][axis_key]["title"] = {"text": new_label}
                        self._fig_dict["layout"][axis_key]["autorange"] = True
                        for state in states:
                            n = state["_n"]
                            if n > 0 and state.get("_unit") != unit:
                                divisor = self.unit_divisor(unit)
                                state["x"][:n] = (state["_raw_t"][:n] - self._t0) / divisor
                            state["_unit"] = unit
                        if self._event_timestamps:
                            divisor = self.unit_divisor(unit)
                            xref = "x" if i == 0 else f"x{i + 1}"
                            new_xs = [
                                (t - self._t0) / divisor for t in self._event_timestamps
                            ]
                            layout = self._fig_dict["layout"]
                            ev_idx = 0
                            for shape in layout.get("shapes", []):
                                if shape.get("xref") == xref and ev_idx < len(new_xs):
                                    shape["x0"] = new_xs[ev_idx]
                                    shape["x1"] = new_xs[ev_idx]
                                    ev_idx += 1
                            ev_idx = 0
                            for ann in layout.get("annotations", []):
                                if ann.get("xref") == xref and ev_idx < len(new_xs):
                                    ann["x"] = new_xs[ev_idx]
                                    ev_idx += 1
                else:
                    x_val = data.get(spec.x, sample_idx)

                x_float = float(x_val)

                for j, (y_name, state) in enumerate(zip(y_names, states)):
                    if y_name not in data:
                        continue
                    y_float = float(data[y_name])
                    n = state["_n"]
                    cap = state["_cap"]
                    trace_idx = self._trace_offsets[i] + j

                    if n < cap:
                        state["x"][n] = x_float
                        state["y"][n] = y_float
                        if spec.x == "_time":
                            state["_raw_t"][n] = timestamp
                        state["_n"] = n + 1
                    else:
                        state["x"][:-1] = state["x"][1:]
                        state["y"][:-1] = state["y"][1:]
                        if spec.x == "_time":
                            state["_raw_t"][:-1] = state["_raw_t"][1:]
                            state["_raw_t"][-1] = timestamp
                        state["x"][-1] = x_float
                        state["y"][-1] = y_float

                    display_n = state["_n"]
                    self._fig_dict["data"][trace_idx]["x"] = state["x"][:display_n]
                    self._fig_dict["data"][trace_idx]["y"] = state["y"][:display_n]

        self.on_data_changed()

    def notify_event(self, timestamp: float, param: str, value) -> None:
        """Mark a parameter change on all time-series subplots.

        Called automatically by the runner when ``bench["param"] = value``
        is executed during a monitor run. Draws a vertical dashed line
        and a label on every subplot whose x-axis is ``"_time"``.
        """
        if self._fig_dict is None or self._t0 is None:
            return

        elapsed = timestamp - self._t0
        x_val, unit = self.format_elapsed(elapsed)

        label = f"{param}={value:.4g}" if isinstance(value, (int, float)) else f"{param}={value}"

        layout = self._fig_dict["layout"]
        if "shapes" not in layout:
            layout["shapes"] = []
        if "annotations" not in layout:
            layout["annotations"] = []

        self._event_timestamps.append(timestamp)

        for i, spec in enumerate(self.specs):
            if spec.x != "_time":
                continue

            xref = "x" if i == 0 else f"x{i + 1}"
            y_axis_key = "yaxis" if i == 0 else f"yaxis{i + 1}"
            domain = layout.get(y_axis_key, {}).get("domain", [0.0, 1.0])
            y0, y1 = domain[0], domain[1]

            layout["shapes"].append({
                "type": "line",
                "xref": xref,
                "yref": "paper",
                "x0": x_val,
                "x1": x_val,
                "y0": y0,
                "y1": y1,
                "line": {
                    "color": self.event_line.color,
                    "width": self.event_line.width,
                    "dash": self.event_line.dash,
                },
            })
            layout["annotations"].append({
                "xref": xref,
                "yref": "paper",
                "x": x_val,
                "y": (y0 + y1) / 2,
                "text": label,
                "showarrow": False,
                "textangle": -90,
                "font": {"size": self.event_line.font_size, "color": self.event_line.color},
                "xanchor": "center",
                "yanchor": "middle",
                "bgcolor": self.event_line.bgcolor,
                "bordercolor": self.event_line.bordercolor,
                "borderwidth": self.event_line.borderwidth,
                "borderpad": self.event_line.borderpad,
            })

        self.on_data_changed()

    def finalize(self) -> None:
        """Called once after experiment completes.

        Server keeps running so zoom/pan state is preserved, but stops
        refreshing so the browser doesn't reset the view on every poll.
        """
        self._stopped = True

    # ── Figure construction ────────────────────────────────────────────

    def build_figure_dict(self, proc, n: int) -> dict:
        """Build the plotly figure as a plain dict.

        Plain dict (not ``go.Figure``) means Jupyter will never auto-display
        it via ``_repr_html_``.
        """
        from plotly.subplots import make_subplots

        subplot_types = [
            {"type": "heatmap"} if pt in ("heatmap", "trace_heatmap") else {"type": "xy"}
            for pt in self._resolved_types
        ]

        fig = make_subplots(
            rows=n, cols=1,
            specs=[[st] for st in subplot_types],
            vertical_spacing=0.08 if n > 1 else 0.0,
        )
        fig.update_layout(height=self.height_per_plot * n, width=self.width)

        import plotly.graph_objects as go

        trace_idx = 0
        for i, (spec, ptype) in enumerate(zip(self.specs, self._resolved_types)):
            row = i + 1
            self._trace_offsets.append(trace_idx)

            if ptype == "line":
                y_names = _y_list(spec)
                for y_name in y_names:
                    fig.add_trace(
                        go.Scatter(x=[], y=[], mode="lines+markers", name=y_name),
                        row=row, col=1,
                    )
                    trace_idx += 1
                fig.update_xaxes(title_text=spec.x, row=row, col=1)
                fig.update_yaxes(
                    title_text=" / ".join(y_names) if len(y_names) > 1 else y_names[0],
                    row=row, col=1,
                )
                cap = proc.sweeps[-1].length if hasattr(proc, "sweeps") and proc.sweeps else self.max_display_pts
                states = []
                for _ in y_names:
                    st: dict = {
                        "x": np.empty(cap, dtype=np.float64),
                        "y": np.empty(cap, dtype=np.float64),
                        "_n": 0,
                        "_cap": cap,
                    }
                    if spec.x == "_time":
                        st["_raw_t"] = np.empty(cap, dtype=np.float64)
                        st["_unit"] = None
                    states.append(st)
                self._sweep_data[i] = states

            elif ptype == "heatmap":
                x_sweep = next(
                    (s for s in proc.sweeps if s.controller.name == spec.x),
                    proc.sweeps[-1],
                )
                y_sweep = next(
                    (s for s in proc.sweeps if s.controller.name == spec.y),
                    proc.sweeps[0] if len(proc.sweeps) >= 2 else None,
                )
                x_vals = x_sweep.values
                y_vals = y_sweep.values if y_sweep is not None else np.array([0])
                z = np.full((len(y_vals), len(x_vals)), np.nan)
                fig.add_trace(
                    go.Heatmap(
                        z=z.tolist(), x=x_vals.tolist(), y=y_vals.tolist(),
                        **({"colorscale": spec.colorscale} if spec.colorscale else {}),
                        name=spec.z, colorbar=dict(title=spec.z),
                    ),
                    row=row, col=1,
                )
                fig.update_xaxes(title_text=spec.x, row=row, col=1)
                fig.update_yaxes(title_text=spec.y, row=row, col=1)
                self._sweep_data[i] = [{"z": z}]
                trace_idx += 1

            elif ptype == "live_trace":
                x_arr = np.asarray(spec.x, dtype=np.float64)
                n_pts = len(x_arr)
                y_label = spec.y if isinstance(spec.y, str) else "value"
                fig.add_trace(
                    go.Scatter(x=x_arr.tolist(), y=[np.nan] * n_pts, mode="lines", name=y_label),
                    row=row, col=1,
                )
                fig.update_xaxes(title_text="", row=row, col=1)
                fig.update_yaxes(title_text=y_label, row=row, col=1)
                self._sweep_data[i] = [{"y": np.full(n_pts, np.nan)}]
                trace_idx += 1

            elif ptype == "trace_heatmap":
                y_arr = np.asarray(spec.y, dtype=np.float64)
                n_freq = len(y_arr)
                x_sweep = next(
                    (s for s in proc.sweeps if s.controller.name == spec.x),
                    proc.sweeps[0],
                )
                x_vals = x_sweep.values
                n_steps = len(x_vals)
                z = np.full((n_freq, n_steps), np.nan)
                fig.add_trace(
                    go.Heatmap(
                        z=z.tolist(), x=x_vals.tolist(), y=y_arr.tolist(),
                        **({"colorscale": spec.colorscale} if spec.colorscale else {}),
                        name=spec.z, colorbar=dict(title=spec.z),
                    ),
                    row=row, col=1,
                )
                fig.update_xaxes(title_text=spec.x, row=row, col=1)
                fig.update_yaxes(title_text="", row=row, col=1)
                self._sweep_data[i] = [{"z": z}]
                trace_idx += 1

        # Fix colorbar positions in multi-subplot layouts.
        # Plotly defaults to y=0.5, len=1.0 (full figure height) so every
        # colorbar spans the whole figure. Pin each to its own subplot domain.
        for i, (spec, ptype) in enumerate(zip(self.specs, self._resolved_types)):
            if ptype not in ("heatmap", "trace_heatmap"):
                continue
            row = i + 1
            axis_key = "yaxis" if row == 1 else f"yaxis{row}"
            domain = getattr(fig.layout, axis_key).domain
            y0, y1 = domain[0], domain[1]
            fig.data[self._trace_offsets[i]].colorbar.update(
                y=(y0 + y1) / 2, len=y1 - y0, yanchor="middle",
            )

        return fig.to_dict()

    # ── Data update helpers ────────────────────────────────────────────

    def update_line(self, spec_idx: int, spec: PlotSpec, data: dict, sweep_values: dict) -> None:
        """Update line traces using pre-allocated numpy buffers.

        x can be a sweep parameter name or a readout name.
        Sweep parameter as x: resets on each new inner sweep (O(1) pointer reset).
        Readout as x: accumulates all points across the full experiment.
        """
        y_names = _y_list(spec)
        states = self._sweep_data[spec_idx]
        x_from_sweep = spec.x in sweep_values
        x_val = sweep_values.get(spec.x) if x_from_sweep else data.get(spec.x)

        for j, (y_name, state) in enumerate(zip(y_names, states)):
            if y_name not in data:
                continue
            y_val = data[y_name]
            trace_idx = self._trace_offsets[spec_idx] + j

            if isinstance(x_val, np.ndarray):
                # Sweep-level update: full row arrives at once (SWEEPWISE)
                length = len(x_val)
                y_arr = y_val if isinstance(y_val, np.ndarray) else np.full(length, float(y_val))
                if length > state["_cap"]:
                    state["x"] = np.empty(length, dtype=np.float64)
                    state["y"] = np.empty(length, dtype=np.float64)
                    state["_cap"] = length
                state["x"][:length] = x_val
                state["y"][:length] = y_arr
                state["_n"] = length
            else:
                x_float = float(x_val) if x_val is not None else float(state["_n"])
                n = state["_n"]
                # Reset on new inner sweep — detected by x going back to start
                if x_from_sweep and n >= 2 and x_float <= state["x"][0]:
                    state["_n"] = 0
                    n = 0
                if n < state["_cap"]:
                    state["x"][n] = x_float
                    state["y"][n] = float(y_val)
                    state["_n"] = n + 1

            n = state["_n"]
            self._fig_dict["data"][trace_idx]["x"] = state["x"][:n]
            self._fig_dict["data"][trace_idx]["y"] = state["y"][:n]

    def update_heatmap(self, spec_idx: int, spec: PlotSpec, index: tuple, data: dict) -> None:
        """Fill one row or one cell of the heatmap z-matrix."""
        state = self._sweep_data[spec_idx][0]
        if spec.z not in data or len(index) == 0:
            return
        z_val = data[spec.z]
        row_idx = index[0]
        if isinstance(z_val, np.ndarray) and z_val.ndim >= 1:
            state["z"][row_idx, :] = z_val
        else:
            col_idx = index[-1] if len(index) >= 2 else 0
            state["z"][row_idx, col_idx] = z_val
        self._fig_dict["data"][self._trace_offsets[spec_idx]]["z"] = state["z"].tolist()

    def update_live_trace(self, spec_idx: int, spec: PlotSpec, data: dict) -> None:
        """Overwrite the scatter trace with the current readout values."""
        readout_name = spec.y
        if not isinstance(readout_name, str) or readout_name not in data:
            return
        raw = np.asarray(data[readout_name])
        if hasattr(self._proc, "bench"):
            readout = self._proc.bench.readouts[readout_name]
            y_vals = self.extract_col(raw, spec.z_col, readout)
        else:
            y_vals = (raw[:, spec.z_col] if isinstance(spec.z_col, int) else raw).astype(np.float64)
        self._fig_dict["data"][self._trace_offsets[spec_idx]]["y"] = y_vals.tolist()

    def update_trace_heatmap(self, spec_idx: int, spec: PlotSpec, index: tuple, data: dict) -> None:
        """Fill one column of the trace heatmap (one sweep step = one column)."""
        if not spec.z or spec.z not in data or len(index) == 0:
            return
        raw = np.asarray(data[spec.z])
        if hasattr(self._proc, "bench"):
            readout = self._proc.bench.readouts[spec.z]
            col_data = self.extract_col(raw, spec.z_col, readout)
        else:
            col_data = (raw[:, spec.z_col] if isinstance(spec.z_col, int) else raw).astype(np.float64)
        state = self._sweep_data[spec_idx][0]
        state["z"][:, index[0]] = col_data
        self._fig_dict["data"][self._trace_offsets[spec_idx]]["z"] = state["z"].tolist()

    def resolve_col(self, z_col, readout) -> int | None:
        """Resolve ``z_col`` to an integer index, or None (use whole array).

        Rules:
          TRACE kind + z_col=None  → None (use whole 1-D array)
          TRACE kind + z_col given → warning, ignored, return None
          IMAGE kind + z_col=None  → 0 with a warning
          IMAGE kind + int         → used directly
          IMAGE kind + str         → looked up in ``readout.contains``
        """
        if readout.kind == DataKind.TRACE:
            if z_col is not None:
                print(f"Warning: z_col ignored for TRACE readout '{readout.name}'")
            return None
        # IMAGE kind
        if z_col is None:
            print(f"Warning: z_col not set for IMAGE readout '{readout.name}', using column 0")
            return 0
        if isinstance(z_col, int):
            return z_col
        if isinstance(z_col, str):
            if isinstance(readout.contains, list):
                try:
                    return readout.contains.index(z_col)
                except ValueError:
                    raise ValueError(
                        f"z_col='{z_col}' not found in readout.contains={readout.contains}"
                    )
            raise ValueError(
                f"z_col='{z_col}' is a string but readout '{readout.name}' has no contains list"
            )
        raise TypeError(f"z_col must be int, str, or None, got {type(z_col)}")

    def extract_col(self, raw: np.ndarray, z_col, readout) -> np.ndarray:
        """Extract the relevant channel from raw readout data."""
        col = self.resolve_col(z_col, readout)
        if col is None:
            return raw.astype(np.float64)
        return raw[col].astype(np.float64)

    @staticmethod
    def format_elapsed(elapsed_seconds: float) -> tuple[float, str]:
        """Convert elapsed seconds to a scaled value and unit string.

        Returns ``(value, unit)`` where unit auto-scales:
        < 120 s  → seconds (``"s"``)
        < 7200 s → minutes (``"min"``)
        otherwise → hours (``"hr"``)
        """
        if elapsed_seconds < 120:
            return elapsed_seconds, "s"
        elif elapsed_seconds < 7200:
            return elapsed_seconds / 60, "min"
        else:
            return elapsed_seconds / 3600, "hr"

    @staticmethod
    def unit_divisor(unit: str) -> float:
        """Seconds per display unit — used to rescale raw timestamps."""
        if unit == "min":
            return 60.0
        elif unit == "hr":
            return 3600.0
        return 1.0


# ══════════════════════════════════════════════════════════════════════
#  DashPlotter — Dash/Werkzeug backend (poll-based)
# ══════════════════════════════════════════════════════════════════════


class DashPlotter(PlotterBase):
    """Live plotting via a Dash app served in a separate browser window.

    Creates a local Dash server on a background thread and opens it in
    the default browser. The browser polls for updates every
    ``update_interval`` milliseconds.

    Parameters
    ----------
    plots : list of PlotSpec
        Subplot specifications.
    port : int
        Port for the Dash server. Default 8050.
    height : int
        Figure height in pixels per subplot row. Default 350.
    width : int
        Figure width in pixels. Default 700.
    open_browser : bool
        If True, automatically open the plot in the default browser.
    update_interval : int
        Dash polling interval in milliseconds. Lower = faster updates
        but more CPU. Default 500 ms.
    event_line : EventLineConfig, optional
        Visual style for parameter-change markers on monitor plots.
    max_display_pts : int
        Maximum points shown on line plots (rolling window for monitors).
        Default 5000.

    Examples
    --------
    >>> plotter = DashPlotter([PlotSpec(x="Vgt", y="lockin_X")])
    >>> runner.run(proc, plotter=plotter)

    >>> plotter = DashPlotter([
    ...     PlotSpec(x="Vgt", y="lockin_X"),
    ...     PlotSpec(x="Vgt", y="lockin_Y"),
    ... ])
    """

    def __init__(
        self,
        plots: list[PlotSpec],
        port: int = 8050,
        height: int = 350,
        width: int = 700,
        open_browser: bool = True,
        update_interval: int = 500,
        event_line: EventLineConfig | None = None,
        max_display_pts: int = 5000,
    ):
        super().__init__(
            plots=plots,
            height=height,
            width=width,
            open_browser=open_browser,
            event_line=event_line,
            max_display_pts=max_display_pts,
        )
        self.port = port
        self.update_interval = update_interval

        # Poll-version counter — Dash reads these in the refresh() callback
        self._data_version = 0
        self._last_sent_version = -1

        self._server_thread: threading.Thread | None = None
        self._wsgi_server = None
        self._dash_app = None

    def setup(self, proc) -> None:
        # Reset poll counters before calling base setup (which builds the figure)
        self._data_version = 0
        self._last_sent_version = -1
        super().setup(proc)

    # ── PlotterBase interface ──────────────────────────────────────────

    def on_data_changed(self) -> None:
        """Increment the poll-version counter so Dash sends the next update."""
        self._data_version += 1

    @property
    def is_running(self) -> bool:
        """True if the Dash server thread is alive."""
        return self._server_thread is not None and self._server_thread.is_alive()

    def _start_server(self) -> None:
        """Launch Dash app on a background daemon thread."""
        from dash import Dash, dcc, html
        from dash.dependencies import Input, Output
        import logging

        # Suppress Dash/Flask/Werkzeug startup logs
        for logger_name in ("werkzeug", "dash", "dash.dash", "flask", "flask.app"):
            logging.getLogger(logger_name).setLevel(logging.ERROR)

        app = Dash(__name__, update_title=None)
        app.title = self._proc.name if self._proc else "Orchid Live Plot"
        app.logger.setLevel(logging.ERROR)
        self._dash_app = app

        plotter = self

        # Callable layout: re-evaluated on every fresh browser load so the
        # browser always gets the current figure after setup() replaces _fig_dict.
        def serve_layout():
            return html.Div([
                dcc.Graph(id="live-graph", figure=plotter._fig_dict or {}),
                dcc.Interval(id="interval", interval=self.update_interval, n_intervals=0),
            ])

        app.layout = serve_layout

        @app.callback(
            Output("live-graph", "figure"),
            Input("interval", "n_intervals"),
        )
        def refresh(n):
            # Stop updating once the experiment is done so zoom/pan is preserved
            if plotter._stopped or plotter._data_version == plotter._last_sent_version:
                from dash import no_update
                return no_update
            plotter._last_sent_version = plotter._data_version
            return plotter._fig_dict

        # Suppress the Flask "Serving Flask app" CLI banner
        try:
            import flask.cli
            flask.cli.show_server_banner = lambda *args, **kwargs: None
        except (ImportError, AttributeError):
            pass

        from werkzeug.serving import make_server

        # Bind socket on the calling thread — fails immediately if port is in use
        srv = make_server("127.0.0.1", self.port, app.server)
        self._wsgi_server = srv

        self._server_thread = threading.Thread(target=srv.serve_forever, daemon=True)
        self._server_thread.start()

        time.sleep(0.5)

        if self.open_browser:
            import webbrowser
            webbrowser.open(f"http://localhost:{self.port}")

        print(f"Live plot server started at http://localhost:{self.port}")

    def stop(self, _silent: bool = False) -> None:
        """Stop the Dash server and free the port.

        After calling ``stop()``, the browser page shows a connection error
        and the port becomes available for a new plotter.
        """
        self._stopped = True
        if self._wsgi_server is not None:
            self._wsgi_server.shutdown()
            self._wsgi_server.server_close()
            self._wsgi_server = None
        self._server_thread = None
        self._dash_app = None
        if not _silent:
            print("Live plot server stopped.")


# Backward-compatible alias — existing code using LivePlotter(...) keeps working
LivePlotter = DashPlotter


# ---------------------------------------------------------------------------
# TaipyPlotter — Taipy GUI backend (push-based via broadcast_callback)
# ---------------------------------------------------------------------------

class TaipyPlotter(PlotterBase):
    """Live plotting backend using Taipy GUI.

    Uses Taipy's ``broadcast_callback`` to push figure updates directly to
    every connected browser tab — no polling required.

    Requires ``pip install taipy-gui>=3.1``.

    Parameters
    ----------
    plots:
        List of :class:`PlotSpec` descriptors (same as :class:`DashPlotter`).
    port:
        TCP port for the Taipy GUI server (default ``5000``).
    host:
        Hostname / IP to bind (default ``"localhost"``).
    height, width:
        Figure dimensions in pixels.
    open_browser:
        Open a browser tab automatically when the server starts.
    event_line:
        Optional :class:`EventLineConfig` — draws a vertical marker when
        :py:meth:`notify_event` is called.
    max_display_pts:
        Down-sample display to at most this many points per trace.
    """

    def __init__(
        self,
        plots: "list[PlotSpec]",
        port: int = 5000,
        host: str = "localhost",
        height: int = 350,
        width: int = 700,
        open_browser: bool = True,
        event_line: "EventLineConfig | None" = None,
        max_display_pts: int = 5_000,
    ) -> None:
        super().__init__(
            plots=plots,
            height=height,
            width=width,
            open_browser=open_browser,
            event_line=event_line,
            max_display_pts=max_display_pts,
        )
        self.port = port
        self.host = host
        self._gui: "taipy.gui.Gui | None" = None  # type: ignore[name-defined]
        self._gui_thread: "threading.Thread | None" = None

    # ------------------------------------------------------------------
    # PlotterBase abstract interface
    # ------------------------------------------------------------------

    def _start_server(self) -> None:
        """Launch a Taipy GUI server in a daemon thread."""
        try:
            from taipy.gui import Gui  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "TaipyPlotter requires 'taipy-gui>=3.1'. "
                "Install it with:  pip install taipy-gui"
            ) from exc

        page = "<|chart|figure={figure}|>"
        self._gui = Gui(page)

        # Take a snapshot so the GUI thread's initial state is stable.
        fig_copy = copy.deepcopy(self._fig_dict)

        self._gui_thread = threading.Thread(
            target=self._gui.run,
            kwargs=dict(
                host=self.host,
                port=self.port,
                use_reloader=False,
                dark_mode=False,
                debug=False,
                # Pass the initial figure as a page-level variable.
                figure=fig_copy,
            ),
            daemon=True,
        )
        self._gui_thread.start()
        self._wait_for_startup()

        url = f"http://{self.host}:{self.port}"
        if self.open_browser:
            import webbrowser
            webbrowser.open(url)
        print(f"Taipy plot server started — {url}")

    def stop(self, _silent: bool = False) -> None:
        """Shut down the Taipy GUI server."""
        self._stopped = True
        if self._gui is not None:
            try:
                self._gui.stop()
            except Exception:
                pass
            self._gui = None
        self._gui_thread = None
        if not _silent:
            print("Taipy plot server stopped.")

    @property
    def is_running(self) -> bool:
        """``True`` while the Taipy GUI daemon thread is alive."""
        return self._gui_thread is not None and self._gui_thread.is_alive()

    # ------------------------------------------------------------------
    # Push updates via broadcast_callback
    # ------------------------------------------------------------------

    def on_data_changed(self) -> None:
        """Push the current figure to all connected browser clients.

        Called by :class:`PlotterBase` after every data mutation.  We
        deep-copy ``_fig_dict`` here so the GUI thread always reads a
        stable snapshot while the experiment thread continues writing.
        """
        if self._gui is None:
            return
        try:
            from taipy.gui import broadcast_callback  # noqa: PLC0415
        except ImportError:
            return

        fig_snapshot = copy.deepcopy(self._fig_dict)

        def _set_figure(state: object) -> None:
            state.figure = fig_snapshot  # type: ignore[attr-defined]

        broadcast_callback(self._gui, _set_figure)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _wait_for_startup(self, timeout: float = 15.0) -> None:
        """Block until the HTTP server responds or *timeout* seconds elapse."""
        import urllib.request  # noqa: PLC0415

        url = f"http://{self.host}:{self.port}/"
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                urllib.request.urlopen(url, timeout=0.5)
                return
            except Exception:
                time.sleep(0.2)
        print(
            f"Warning: Taipy server may not have started yet on port {self.port}. "
            "Check for errors above."
        )


# Apply the default Plotly theme when this module is first imported.
# Users can override it at any time by calling apply_theme() from orchid.utils.
try:
    from .utils import apply_theme as _apply_default_theme
    _apply_default_theme()
except Exception:
    pass
