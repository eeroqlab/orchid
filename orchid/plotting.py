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
    y: str
    z: str | None = None
    plot_type: str = "auto"
    update_every: str = "sweep"
    update_func: Callable | None = None


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
    ):
        self.specs = plots
        self.port = port
        self.height_per_plot = height
        self.width = width
        self.open_browser = open_browser
        self.update_interval = update_interval
        self.event_line = event_line if event_line is not None else EventLineConfig()

        # Internal state — all plain dicts/lists, no plotly objects
        self._fig_dict: dict | None = None
        self._proc = None
        self._resolved_types: list[str] = []
        self._sweep_data: dict[int, dict] = {}
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
        self._data_version = 0
        self._last_sent_version = -1
        self._stopped = False
        self._t0: float | None = None  # first timestamp for monitor mode

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

            # Check that the relevant readout is present in data
            readout_key = spec.z if (ptype == "heatmap" and spec.z) else spec.y
            if readout_key not in data:
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
        """Update plots for monitoring mode. Called by the runner."""
        if self._fig_dict is None:
            return

        # Track start time for relative timestamps
        if self._t0 is None:
            self._t0 = timestamp

        elapsed = timestamp - self._t0

        for i, (spec, ptype) in enumerate(zip(self.specs, self._resolved_types)):
            if spec.update_func is not None:
                spec.update_func(self._fig_dict, sample_idx, data)
                continue

            if spec.y not in data:
                continue

            if ptype == "line":
                state = self._sweep_data[i]

                if spec.x == "_time":
                    x_val, unit = self._format_elapsed(elapsed)
                    # Update axis label if unit changed
                    axis_key = f"xaxis{i + 1}" if i > 0 else "xaxis"
                    current_label = self._fig_dict["layout"].get(axis_key, {}).get("title", {})
                    new_label = f"Time ({unit})"
                    if current_label.get("text") != new_label:
                        if axis_key not in self._fig_dict["layout"]:
                            self._fig_dict["layout"][axis_key] = {}
                        self._fig_dict["layout"][axis_key]["title"] = {"text": new_label}
                        # Rescale existing x values when unit changes
                        if state["x"] and state.get("_unit") != unit:
                            divisor = self._unit_divisor(unit)
                            state["x"] = [(t - self._t0) / divisor
                                          for t in state["_raw_t"]]
                        state["_unit"] = unit
                else:
                    x_val = data.get(spec.x, sample_idx)

                y_val = data[spec.y]

                state["x"].append(float(x_val))
                state["y"].append(float(y_val))
                # Keep raw timestamps for rescaling
                if spec.x == "_time":
                    if "_raw_t" not in state:
                        state["_raw_t"] = []
                    state["_raw_t"].append(timestamp)

                self._fig_dict["data"][i]["x"] = list(state["x"])
                self._fig_dict["data"][i]["y"] = list(state["y"])

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

        # Build subplot titles
        titles = []
        for s, pt in zip(self.specs, self._resolved_types):
            if pt == "heatmap":
                titles.append(f"{s.z} ({s.y} vs {s.x})")
            else:
                titles.append(f"{s.y} vs {s.x}")

        fig = make_subplots(
            rows=n, cols=1,
            subplot_titles=titles,
            specs=[[st] for st in subplot_types],
            vertical_spacing=0.12 if n > 1 else 0.0,
        )
        fig.update_layout(
            height=self.height_per_plot * n,
            width=self.width,
        )

        import plotly.graph_objects as go

        for i, (spec, ptype) in enumerate(zip(self.specs, self._resolved_types)):
            row = i + 1

            if ptype == "line":
                fig.add_trace(
                    go.Scatter(x=[], y=[], mode="lines+markers", name=spec.y),
                    row=row, col=1,
                )
                fig.update_xaxes(title_text=spec.x, row=row, col=1)
                fig.update_yaxes(title_text=spec.y, row=row, col=1)
                self._sweep_data[i] = {"x": [], "y": []}

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
                        colorscale="Viridis",
                        name=spec.z,
                        colorbar=dict(title=spec.z),
                    ),
                    row=row, col=1,
                )
                fig.update_xaxes(title_text=spec.x, row=row, col=1)
                fig.update_yaxes(title_text=spec.y, row=row, col=1)
                self._sweep_data[i] = {"z": z}

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
        """Update a line trace.

        x can be a sweep parameter name or a readout name.

        Sweep parameter as x: shows only the current inner sweep row —
        resets when a new inner sweep starts (x wraps back to start).

        Readout as x: accumulates all points across the full experiment
        (useful for readout-vs-readout plots, e.g. Lissajous).
        """
        state = self._sweep_data[spec_idx]

        # x can be a sweep parameter or a readout
        x_from_sweep = spec.x in sweep_values
        x_val = sweep_values.get(spec.x) if x_from_sweep else data.get(spec.x)
        y_val = data[spec.y]

        if isinstance(x_val, np.ndarray):
            # Sweep-level update: full row at once
            state["x"] = x_val.tolist()
            state["y"] = y_val.tolist() if isinstance(y_val, np.ndarray) else [float(y_val)]
        else:
            x_float = float(x_val) if x_val is not None else len(state["x"])
            # Only reset on new inner sweep when x is a sweep parameter;
            # readout-vs-readout plots accumulate all points.
            if x_from_sweep and len(state["x"]) >= 2 and x_float <= state["x"][0]:
                state["x"] = []
                state["y"] = []
            state["x"].append(x_float)
            state["y"].append(float(y_val))

        self._fig_dict["data"][spec_idx]["x"] = list(state["x"])
        self._fig_dict["data"][spec_idx]["y"] = list(state["y"])

    def _update_heatmap(self, spec_idx, spec, index, data):
        """Fill one row or one cell of the heatmap."""
        state = self._sweep_data[spec_idx]
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

        self._fig_dict["data"][spec_idx]["z"] = state["z"].tolist()
