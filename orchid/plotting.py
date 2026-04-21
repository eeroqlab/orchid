"""Live plotting for experiments using a Dash app in a separate browser window."""

from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class EventLineConfig:
    """Visual properties for parameter-change event markers on time-series plots.

    Parameters
    ----------
    color : str
        Line and label color. Any CSS/plotly color string.
    width : int
        Line width in pixels.
    dash : str
        Line style: ``"solid"``, ``"dot"``, ``"dash"``, ``"longdash"``, ``"dashdot"``.
    font_size : int
        Label font size in points.

    Examples
    --------
    >>> EventLineConfig(color="rgba(0,150,255,0.8)", dash="dot", width=2)
    """

    color: str = "rgba(255,80,80,0.7)"
    width: int = 1
    dash: str = "dash"
    font_size: int = 9


@dataclass
class PlotSpec:
    """Describes one subplot in a LivePlotter.

    Parameters
    ----------
    x : str
        Sweep parameter name for the x-axis.
        For line plots: the horizontal axis.
        For heatmaps: the horizontal axis (typically the inner sweep).
        For monitors: use "_time" for timestamps.
    y : str
        For line plots: readout name for the vertical axis.
        For heatmaps: sweep parameter name for the vertical axis
        (typically the outer sweep).
    z : str, optional
        Readout name for the color axis (heatmaps only).
        Required when plot_type is "heatmap". Ignored for line plots.
    plot_type : str
        "line", "heatmap", or "auto".
        "auto" infers from experiment dimensionality:
        1D -> line, 2D -> heatmap.
    update_every : str
        Controls how often the plot refreshes. Independent of write_mode.
        "point" — update after every measurement point.
        "sweep" — update after each inner sweep completes (default).
        "plane" — update after each 2D plane completes.
    update_func : callable, optional
        Custom update function with signature (fig_dict, index, data) -> None.
        If provided, overrides the default plot update logic.
        ``fig_dict`` is the raw plotly figure dictionary, ``index`` is the
        current sweep index tuple, ``data`` is a dict of readout name -> value(s).

    Examples
    --------
    Line plot (1D sweep)::

        PlotSpec(x="Vgt", y="lockin_X")

    Heatmap (2D sweep)::

        PlotSpec(x="fac", y="Vgt", z="lockin_X")

    Auto-detect (uses y as readout for line, requires z for heatmap)::

        PlotSpec(x="Vgt", y="lockin_X")             # 1D -> line
        PlotSpec(x="fac", y="Vgt", z="lockin_X")    # 2D -> heatmap
    """

    x: str
    y: str | list[str]          # one readout name, or a list to overlay multiple traces
    z: str | None = None
    plot_type: str = "auto"
    update_every: str = "sweep"
    update_func: Callable | None = None
    colorscale: str | list | None = None  # heatmap only; None = use template default


def _y_list(spec: PlotSpec) -> list[str]:
    """Normalise PlotSpec.y to a list of readout names."""
    return spec.y if isinstance(spec.y, list) else [spec.y]


class LivePlotter:
    """Live plotting via a Dash app served in a separate browser window.

    Creates a local Dash server on a background thread and opens it
    in the default browser. The plots update in real time as data is
    acquired by the ExperimentRunner.

    Parameters
    ----------
    plots : list of PlotSpec
        Subplot specifications.
    port : int
        Port for the Dash server.
    height : int
        Figure height in pixels per subplot row.
    width : int
        Figure width in pixels.
    open_browser : bool
        If True, automatically open the plot in the default browser.
    update_interval : int
        Dash polling interval in milliseconds. Lower = faster updates
        but more CPU. Default 500ms.
    max_display_pts : int
        Maximum points shown on line plots. For sweep plots the buffer is
        sized exactly to the inner sweep length (always fits, no cap).
        For monitors, once this limit is reached the oldest sample is
        dropped on each new point (rolling window). Default 5000.

    Examples
    --------
    >>> plotter = LivePlotter([PlotSpec(x="Vgt", y="lockin_X")])
    >>> runner.run(proc, plotter=plotter)

    >>> plotter = LivePlotter([
    ...     PlotSpec(x="Vgt", y="lockin_X"),
    ...     PlotSpec(x="Vgt", y="lockin_Y"),
    ... ])
    >>> runner.run(proc, plotter=plotter)
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
        self.specs = plots
        self.port = port
        self.height_per_plot = height
        self.width = width
        self.open_browser = open_browser
        self.update_interval = update_interval
        self.event_line = event_line if event_line is not None else EventLineConfig()
        self.max_display_pts = max_display_pts

        # Internal state — all plain dicts/lists, no plotly objects
        self._fig_dict: dict | None = None
        self._proc = None
        self._resolved_types: list[str] = []
        self._sweep_data: dict[int, dict] = {}
        # First trace index in _fig_dict["data"] for each spec (>1 when multi-y)
        self._trace_offsets: list[int] = []
        self._server_thread: threading.Thread | None = None
        self._wsgi_server = None
        self._dash_app = None
        self._data_version = 0
        self._last_sent_version = -1
        self._stopped = False

    # ── Lifecycle (called by the runner) ──────────────────────────────

    def setup(self, proc) -> None:
        """Initialize figure and start Dash server. Called once before experiment."""
        # Reset all state — but keep the server running if it's already up.
        # Restarting the server causes the browser to reconnect mid-session
        # and briefly show stale data before the first callback fires.
        self._proc = proc
        self._fig_dict = None
        self._resolved_types = []
        self._sweep_data = {}
        self._trace_offsets = []
        self._data_version = 0
        self._last_sent_version = -1
        self._stopped = False
        self._t0: float | None = None  # first timestamp for monitor mode
        self._event_timestamps: list[float] = []  # raw timestamps for each notify_event call

        n = len(self.specs)

        # Resolve auto plot types
        # "auto" uses heatmap only if z is provided AND ndim >= 2,
        # otherwise defaults to line.
        self._resolved_types = []
        for spec in self.specs:
            if spec.plot_type == "auto":
                if spec.z and hasattr(proc, "sweeps") and len(proc.sweeps) >= 2:
                    self._resolved_types.append("heatmap")
                else:
                    self._resolved_types.append("line")
            else:
                self._resolved_types.append(spec.plot_type)

        # Validate y readout names for line plots only.
        # For heatmaps, y is the outer sweep parameter (not a readout) — only z is.
        if hasattr(proc, "context"):
            registered = set(proc.context.readouts.keys())
            for spec, ptype in zip(self.specs, self._resolved_types):
                if ptype == "heatmap":
                    readout_names = [spec.z] if spec.z else []
                else:
                    readout_names = [y for y in _y_list(spec) if y != "_time"]
                for name in readout_names:
                    if name not in registered:
                        raise ValueError(
                            f"PlotSpec {'z' if ptype == 'heatmap' else 'y'}={name!r} "
                            f"is not a registered readout. Add it to your procedure's "
                            f"readouts list. Registered: {sorted(registered)}"
                        )

        # Build figure as a plain dict (never a go.Figure, so Jupyter
        # won't auto-display it via _repr_html_).
        self._fig_dict = self._build_figure_dict(proc, n)

        # Start the server only once. If it's already running (reuse across
        # experiments), the browser will receive the new figure on the next
        # callback poll — no reconnection, no stale data flash.
        if not self.is_running:
            self._start_dash()

    def update_point(self, index: tuple, data: dict, sweep_values: dict) -> None:
        """Called by the runner after every measurement point."""
        self._dispatch("point", index, data, sweep_values)

    def update_sweep(self, outer_index: tuple, data: dict, sweep_values: dict) -> None:
        """Called by the runner after each inner sweep completes."""
        self._dispatch("sweep", outer_index, data, sweep_values)

    def update_plane(self, outer_index: tuple, data: dict, sweep_values: dict) -> None:
        """Called by the runner after each 2D plane completes."""
        self._dispatch("plane", outer_index, data, sweep_values)

    def _dispatch(self, event: str, index, data, sweep_values) -> None:
        """Route update to each subplot if its update_every matches the event."""
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

            # Check that at least one relevant readout is present in data
            if ptype == "heatmap":
                if spec.z not in data:
                    continue
            else:
                if not any(y in data for y in _y_list(spec)):
                    continue

            if ptype == "line":
                self._update_line(i, spec, data, sweep_values)
                changed = True
            elif ptype == "heatmap":
                self._update_heatmap(i, spec, index, data)
                changed = True

        if changed:
            self._data_version += 1

    def update_monitor(self, sample_idx: int, data: dict, timestamp: float) -> None:
        """Update plots for monitoring mode. Called by the runner.

        Uses a pre-allocated rolling numpy buffer of size max_display_pts.
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

                # x value and axis label are shared across all y traces in this spec
                if spec.x == "_time":
                    x_val, unit = self._format_elapsed(elapsed)
                    axis_key = f"xaxis{i + 1}" if i > 0 else "xaxis"
                    current_label = self._fig_dict["layout"].get(axis_key, {}).get("title", {})
                    new_label = f"Time ({unit})"
                    if current_label.get("text") != new_label:
                        if axis_key not in self._fig_dict["layout"]:
                            self._fig_dict["layout"][axis_key] = {}
                        self._fig_dict["layout"][axis_key]["title"] = {"text": new_label}
                        # Force Plotly.js to discard any cached range and refit
                        self._fig_dict["layout"][axis_key]["autorange"] = True
                        # Rescale existing x values when unit changes
                        for state in states:
                            n = state["_n"]
                            if n > 0 and state.get("_unit") != unit:
                                divisor = self._unit_divisor(unit)
                                state["x"][:n] = (state["_raw_t"][:n] - self._t0) / divisor
                            state["_unit"] = unit
                        # Rescale event-line shapes and annotations for this subplot
                        if self._event_timestamps:
                            divisor = self._unit_divisor(unit)
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
                        # Rolling window: shift buffer left by 1 (numpy C-level copy)
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

        self._data_version += 1

    @staticmethod
    def _format_elapsed(elapsed_seconds: float) -> tuple[float, str]:
        """Convert elapsed seconds to a value and unit string.

        Returns (value, unit) where unit auto-scales:
        < 120s  → seconds ("s")
        < 120min → minutes ("min")
        otherwise → hours ("hr")
        """
        if elapsed_seconds < 120:
            return elapsed_seconds, "s"
        elif elapsed_seconds < 7200:
            return elapsed_seconds / 60, "min"
        else:
            return elapsed_seconds / 3600, "hr"

    @staticmethod
    def _unit_divisor(unit: str) -> float:
        if unit == "min":
            return 60.0
        elif unit == "hr":
            return 3600.0
        return 1.0

    @property
    def is_running(self) -> bool:
        """True if the Dash server is currently running."""
        return (
            self._server_thread is not None
            and self._server_thread.is_alive()
        )

    def notify_event(self, timestamp: float, param: str, value) -> None:
        """Mark a parameter change on all time-series subplots.

        Called automatically by the runner when ``ctx["param"] = value``
        is executed during a monitor run. Draws a vertical dashed line
        and a label on every subplot whose x-axis is ``"_time"``.
        """
        if self._fig_dict is None or self._t0 is None:
            return

        elapsed = timestamp - self._t0
        x_val, unit = self._format_elapsed(elapsed)

        label = f"{param}={value:.4g}" if isinstance(value, (int, float)) else f"{param}={value}"

        layout = self._fig_dict["layout"]
        if "shapes" not in layout:
            layout["shapes"] = []
        if "annotations" not in layout:
            layout["annotations"] = []

        # Record the raw timestamp so we can rescale this line if the unit changes later
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
                "y": y1,
                "text": label,
                "showarrow": False,
                "textangle": -90,
                "font": {"size": self.event_line.font_size, "color": self.event_line.color},
                "xanchor": "left",
                "yanchor": "top",
            })

        self._data_version += 1

    def finalize(self) -> None:
        """Called once after experiment completes. Server keeps running
        but stops refreshing so zoom/pan state is preserved."""
        self._stopped = True

    def stop(self, _silent: bool = False) -> None:
        """Stop the Dash server and free the port.

        After calling stop(), the browser page will show a connection
        error. The port becomes available for a new LivePlotter.
        """
        self._stopped = True
        if self._wsgi_server is not None:
            self._wsgi_server.shutdown()   # blocks until serve_forever() exits
            self._wsgi_server.server_close()
            self._wsgi_server = None
        self._server_thread = None
        self._dash_app = None
        if not _silent:
            print("Live plot server stopped.")

    # ── Figure construction (plain dicts only) ────────────────────────

    def _build_figure_dict(self, proc, n: int) -> dict:
        """Build the plotly figure as a plain dict. No go.Figure objects."""
        from plotly.subplots import make_subplots

        subplot_types = []
        for pt in self._resolved_types:
            subplot_types.append(
                {"type": "heatmap"} if pt == "heatmap" else {"type": "xy"}
            )

        fig = make_subplots(
            rows=n, cols=1,
            specs=[[st] for st in subplot_types],
            vertical_spacing=0.08 if n > 1 else 0.0,
        )
        fig.update_layout(
            height=self.height_per_plot * n,
            width=self.width,
        )

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

                # Pre-allocate numpy buffers — one state dict per y name.
                # Sweep procedures: bounded by inner sweep length (known upfront).
                # Monitor procedures: capped at max_display_pts (rolling window).
                if hasattr(proc, "sweeps") and proc.sweeps:
                    cap = proc.sweeps[-1].length
                else:
                    cap = self.max_display_pts

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
                # Find sweep objects matching spec.x (heatmap x-axis)
                # and spec.y (heatmap y-axis)
                x_sweep = next(
                    (s for s in proc.sweeps if s.parameter.name == spec.x),
                    proc.sweeps[-1],
                )
                y_sweep = next(
                    (s for s in proc.sweeps if s.parameter.name == spec.y),
                    proc.sweeps[0] if len(proc.sweeps) >= 2 else None,
                )

                x_vals = x_sweep.values
                y_vals = y_sweep.values if y_sweep is not None else np.array([0])

                z = np.full((len(y_vals), len(x_vals)), np.nan)
                fig.add_trace(
                    go.Heatmap(
                        z=z.tolist(),
                        x=x_vals.tolist(),
                        y=y_vals.tolist(),
                        **({"colorscale": spec.colorscale} if spec.colorscale else {}),
                        name=spec.z,
                        colorbar=dict(title=spec.z),
                    ),
                    row=row, col=1,
                )
                fig.update_xaxes(title_text=spec.x, row=row, col=1)
                fig.update_yaxes(title_text=spec.y, row=row, col=1)
                self._sweep_data[i] = [{"z": z}]
                trace_idx += 1

        # Fix colorbar positions for heatmap traces.
        # Plotly defaults to y=0.5, len=1.0 (full figure height), so in a
        # multi-subplot layout every colorbar spans the entire figure.
        # Read each subplot's yaxis.domain and pin the colorbar to it.
        for i, (spec, ptype) in enumerate(zip(self.specs, self._resolved_types)):
            if ptype != "heatmap":
                continue
            row = i + 1
            axis_key = "yaxis" if row == 1 else f"yaxis{row}"
            domain = getattr(fig.layout, axis_key).domain   # (y0, y1) in [0,1]
            y0, y1 = domain[0], domain[1]
            fig.data[self._trace_offsets[i]].colorbar.update(
                y=(y0 + y1) / 2,
                len=y1 - y0,
                yanchor="middle",
            )

        # Convert to plain dict — this is the key: a plain dict has no
        # _repr_html_ so Jupyter will never auto-display it.
        return fig.to_dict()

    # ── Dash server ───────────────────────────────────────────────────

    def _start_dash(self):
        """Launch Dash app on a background daemon thread."""
        from dash import Dash, dcc, html
        from dash.dependencies import Input, Output
        import logging

        # Suppress all Dash/Flask/Werkzeug startup logs
        for logger_name in ("werkzeug", "dash", "dash.dash", "flask", "flask.app"):
            logging.getLogger(logger_name).setLevel(logging.ERROR)

        app = Dash(__name__, update_title=None)
        app.title = self._proc.name if self._proc else "Orchid Live Plot"
        app.logger.setLevel(logging.ERROR)
        self._dash_app = app

        plotter = self

        # Callable layout: evaluated on every fresh browser load, so the
        # browser always gets the current figure even after setup() replaces
        # _fig_dict for a new experiment.
        def serve_layout():
            return html.Div([
                dcc.Graph(id="live-graph", figure=plotter._fig_dict or {}),
                dcc.Interval(
                    id="interval",
                    interval=self.update_interval,
                    n_intervals=0,
                ),
            ])

        app.layout = serve_layout

        @app.callback(
            Output("live-graph", "figure"),
            Input("interval", "n_intervals"),
        )
        def refresh(n):
            # Stop updating once the experiment is done — this prevents
            # the browser from resetting zoom/pan on every poll.
            if plotter._stopped or plotter._data_version == plotter._last_sent_version:
                from dash import no_update
                return no_update
            plotter._last_sent_version = plotter._data_version
            return plotter._fig_dict

        # Suppress the Flask "Serving Flask app" banner by disabling
        # Werkzeug's CLI banner at the source.
        try:
            import flask.cli
            flask.cli.show_server_banner = lambda *args, **kwargs: None
        except (ImportError, AttributeError):
            pass

        from werkzeug.serving import make_server

        # Bind the socket on the main thread — fails immediately if port
        # is already in use, before we even start the background thread.
        srv = make_server("127.0.0.1", self.port, app.server)
        self._wsgi_server = srv

        def run_server():
            srv.serve_forever()

        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()

        # Give the server a moment to start
        time.sleep(0.5)

        if self.open_browser:
            import webbrowser
            webbrowser.open(f"http://localhost:{self.port}")

        print(f"Live plot server started at http://localhost:{self.port}")

    # ── Internal update methods ───────────────────────────────────────

    def _update_line(self, spec_idx, spec, data, sweep_values):
        """Update line traces using pre-allocated numpy buffers.

        x can be a sweep parameter name or a readout name.

        Sweep parameter as x: shows only the current inner sweep row —
        resets (O(1) pointer reset) when a new inner sweep starts.

        Readout as x: accumulates all points across the full experiment
        (useful for readout-vs-readout plots, e.g. Lissajous).

        Supports multiple y readouts overlaid on one subplot.
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
                # Reset on new inner sweep (sweep-parameter x only); O(1) pointer reset
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

    def _update_heatmap(self, spec_idx, spec, index, data):
        """Fill one row or one cell of the heatmap."""
        state = self._sweep_data[spec_idx][0]
        z_key = spec.z  # readout name for the color values

        if z_key not in data:
            return
        z_val = data[z_key]

        if len(index) == 0:
            return

        row_idx = index[0] if len(index) >= 1 else 0

        if isinstance(z_val, np.ndarray) and z_val.ndim >= 1:
            state["z"][row_idx, :] = z_val
        else:
            col_idx = index[-1] if len(index) >= 2 else 0
            state["z"][row_idx, col_idx] = z_val

        self._fig_dict["data"][self._trace_offsets[spec_idx]]["z"] = state["z"].tolist()


# Apply the default theme when plotting is first imported.
# Users can override it at any time by calling apply_theme() from orchid.utils.
try:
    from .utils import apply_theme as _apply_default_theme
    _apply_default_theme()
except Exception:
    pass
