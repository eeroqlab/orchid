"""PlotterBase — backend-agnostic figure building and data update logic."""
from __future__ import annotations

import abc
import warnings

import numpy as np

from ..controller import DataKind
from ._spec import (
    PlotSpec,
    EventLineConfig,
    _y_list,
    _is_array,
    _find_sweep_by_ctrl,
    _resolve_plot_type,
    _default_update_every,
    _normalize_y_col,
    _resolve_col,
    _format_elapsed_display,
    _format_eta,
)


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
        open_browser: bool = False,
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
        self._resolved_update_every: list[str] = []
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

    def set_run_info(self, data_dir, experiment_id: str | None = None) -> None:
        """Called by the runner once the data directory is known.

        Override in subclasses to display path / ID in the UI.
        The default is a no-op.
        """

    def show_analysis(self, results: list) -> None:
        """Overlay post-experiment analysis results on the live plot.

        Call this after ``runner.run()`` completes and your analysis process
        returns its results. Accepts a list of :class:`PostResult` objects.
        The default implementation is a no-op; :class:`DashPlotter` overrides it.
        """

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
        self._resolved_update_every = []
        self._sweep_data = {}
        self._trace_offsets = []
        self._stopped = False
        self._t0 = None
        self._event_timestamps = []

        n = len(self.specs)

        # Resolve plot types and update_every defaults
        for spec in self.specs:
            ptype = _resolve_plot_type(spec, proc)
            self._resolved_types.append(ptype)
            uev = spec.update_every if spec.update_every is not None else _default_update_every(ptype, proc)
            self._resolved_update_every.append(uev)

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
            if self._resolved_update_every[i] != event:
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
                # line: check all state readout names
                states = self._sweep_data.get(i, [])
                if not any(s["_readout"] in data for s in states):
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

            if ptype == "live_trace":
                rname = spec.y if isinstance(spec.y, str) else None
                if rname and rname in data:
                    self.update_live_trace(i, spec, data)
                continue

            if ptype != "line":
                continue

            states = self._sweep_data[i]
            if not any(s["_readout"] in data for s in states):
                continue

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

            for j, state in enumerate(states):
                rname = state["_readout"]
                col = state["_col"]
                if rname not in data:
                    continue
                raw = data[rname]
                if col is None:
                    y_float = float(raw)
                else:
                    arr = np.asarray(raw)
                    sub = arr[col]
                    y_float = float(sub) if sub.ndim == 0 else float(sub.mean())
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
            if not isinstance(spec.x, str) or spec.x != "_time":
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

    # ── Theme hook (override in subclasses) ────────────────────────────

    def _theme_layout(self, fig) -> None:
        """Apply theme styling to *fig* (a ``go.Figure``) before serialisation.

        Called at the end of :meth:`build_figure_dict` while the figure is
        still a live object.  The default is a no-op; :class:`DashPlotter`
        overrides it to inject colour-scheme tokens from :mod:`.themes`.
        """

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
                # Build (readout_name, col_index, trace_label) for each trace
                trace_specs_line = []
                if isinstance(spec.y, list) and spec.y and isinstance(spec.y[0], str):
                    # Multiple named readouts — legacy multi-trace mode
                    for rname in spec.y:
                        trace_specs_line.append((rname, None, rname))
                else:
                    rname = spec.y if isinstance(spec.y, str) else str(spec.y)
                    readout = (
                        proc.bench.readouts.get(rname)
                        if hasattr(proc, "bench") and isinstance(spec.y, str)
                        else None
                    )
                    resolved_cols = _normalize_y_col(spec.y_col, readout)
                    if resolved_cols is None:
                        trace_specs_line.append((rname, None, rname))
                    else:
                        orig_list = spec.y_col if isinstance(spec.y_col, list) else [spec.y_col]
                        for col, orig in zip(resolved_cols, orig_list):
                            label = orig if isinstance(orig, str) else f"{rname}[{col}]"
                            trace_specs_line.append((rname, col, label))

                for _rname, _col, _label in trace_specs_line:
                    fig.add_trace(
                        go.Scatter(x=[], y=[], mode="lines+markers", name=_label),
                        row=row, col=1,
                    )
                    trace_idx += 1
                x_label = spec.x if isinstance(spec.x, str) else ""
                fig.update_xaxes(title_text=x_label, row=row, col=1)
                fig.update_yaxes(
                    title_text=" / ".join(t[2] for t in trace_specs_line),
                    row=row, col=1,
                )
                is_monitor = not hasattr(proc, "sweeps") or not proc.sweeps
                if is_monitor:
                    cap = self.max_display_pts
                else:
                    x_sweep = _find_sweep_by_ctrl(proc, spec.x) if isinstance(spec.x, str) else None
                    cap = x_sweep.length if x_sweep is not None else proc.sweeps[-1].length
                states = []
                for _rname, _col, _label in trace_specs_line:
                    st: dict = {
                        "x": np.empty(cap, dtype=np.float64),
                        "y": np.empty(cap, dtype=np.float64),
                        "_n": 0,
                        "_cap": cap,
                        "_readout": _rname,
                        "_col": _col,
                    }
                    if spec.x == "_time":
                        st["_raw_t"] = np.empty(cap, dtype=np.float64)
                        st["_unit"] = None
                    states.append(st)
                self._sweep_data[i] = states

            elif ptype == "heatmap":
                x_sweep = _find_sweep_by_ctrl(proc, spec.x) if isinstance(spec.x, str) else None
                if x_sweep is None and hasattr(proc, 'sweeps') and proc.sweeps:
                    x_sweep = proc.sweeps[-1]
                y_sweep = _find_sweep_by_ctrl(proc, spec.y) if isinstance(spec.y, str) else None
                if y_sweep is None and hasattr(proc, 'sweeps') and len(proc.sweeps) >= 2:
                    y_sweep = proc.sweeps[0]
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
                rname_lt = spec.y if isinstance(spec.y, str) else "value"
                readout_lt = (
                    proc.bench.readouts.get(rname_lt)
                    if hasattr(proc, "bench") and isinstance(spec.y, str)
                    else None
                )
                resolved_lt = _normalize_y_col(spec.y_col, readout_lt)
                if resolved_lt is None:
                    col_specs_lt = [(None, rname_lt)]
                else:
                    orig_list_lt = spec.y_col if isinstance(spec.y_col, list) else [spec.y_col]
                    col_specs_lt = []
                    for col, orig in zip(resolved_lt, orig_list_lt):
                        label = orig if isinstance(orig, str) else f"{rname_lt}[{col}]"
                        col_specs_lt.append((col, label))
                for _col_lt, _label_lt in col_specs_lt:
                    fig.add_trace(
                        go.Scatter(x=x_arr.tolist(), y=[np.nan] * n_pts, mode="lines", name=_label_lt),
                        row=row, col=1,
                    )
                    trace_idx += 1
                fig.update_xaxes(title_text="", row=row, col=1)
                fig.update_yaxes(
                    title_text=" / ".join(t[1] for t in col_specs_lt),
                    row=row, col=1,
                )
                self._sweep_data[i] = [
                    {"y": np.full(n_pts, np.nan), "_col": _col_lt}
                    for _col_lt, _ in col_specs_lt
                ]

            elif ptype == "trace_heatmap":
                y_arr = np.asarray(spec.y, dtype=np.float64)
                n_freq = len(y_arr)
                x_sweep = _find_sweep_by_ctrl(proc, spec.x) if isinstance(spec.x, str) else None
                if x_sweep is None and hasattr(proc, 'sweeps') and proc.sweeps:
                    x_sweep = proc.sweeps[0]
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

        self._theme_layout(fig)
        return fig.to_dict()

    # ── Data update helpers ────────────────────────────────────────────

    def update_line(self, spec_idx: int, spec: PlotSpec, data: dict, sweep_values: dict) -> None:
        """Update line traces using pre-allocated numpy buffers.

        x can be a sweep parameter name or a readout name.
        Sweep parameter as x: resets on each new inner sweep (O(1) pointer reset).
        Readout as x: accumulates all points across the full experiment.
        """
        states = self._sweep_data[spec_idx]
        x_from_sweep = isinstance(spec.x, str) and spec.x in sweep_values
        x_val = sweep_values.get(spec.x) if x_from_sweep else data.get(spec.x)

        for j, state in enumerate(states):
            rname = state["_readout"]
            col = state["_col"]
            if rname not in data:
                continue
            raw = data[rname]
            trace_idx = self._trace_offsets[spec_idx] + j

            if isinstance(x_val, np.ndarray):
                # Sweep-level update: full row arrives at once
                length = len(x_val)
                if col is None:
                    y_arr = raw if isinstance(raw, np.ndarray) else np.full(length, float(raw))
                else:
                    arr = np.asarray(raw)
                    sub = arr[col]
                    y_arr = sub if (sub.ndim == 1 and len(sub) == length) else np.full(length, float(sub.flat[0]))
                y_arr = np.asarray(y_arr, dtype=np.float64)
                if length > state["_cap"]:
                    state["x"] = np.empty(length, dtype=np.float64)
                    state["y"] = np.empty(length, dtype=np.float64)
                    state["_cap"] = length
                state["x"][:length] = x_val
                state["y"][:length] = y_arr[:length]
                state["_n"] = length
            else:
                x_float = float(x_val) if x_val is not None else float(state["_n"])
                n = state["_n"]
                # Reset on new inner sweep — detected by x going back to start
                if x_from_sweep and n >= 2 and x_float <= state["x"][0]:
                    state["_n"] = 0
                    n = 0
                if col is None:
                    y_float = float(raw)
                else:
                    arr = np.asarray(raw)
                    sub = arr[col]
                    y_float = float(sub) if sub.ndim == 0 else float(sub.mean())
                if n < state["_cap"]:
                    state["x"][n] = x_float
                    state["y"][n] = y_float
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
        """Overwrite the live-trace scatter(s) with the current readout values."""
        rname = spec.y
        if not isinstance(rname, str) or rname not in data:
            return
        raw = np.asarray(data[rname])
        states = self._sweep_data[spec_idx]
        for j, state in enumerate(states):
            col = state["_col"]
            if col is None:
                y_vals = raw.astype(np.float64)
            else:
                y_vals = np.asarray(raw[col], dtype=np.float64)
            self._fig_dict["data"][self._trace_offsets[spec_idx] + j]["y"] = y_vals.tolist()

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

        TRACE + z_col given  → warning, ignored, returns None.
        IMAGE + z_col=None   → warning, defaults to 0.
        IMAGE + int/str      → resolved via ``_resolve_col``.
        """
        if readout.kind == DataKind.TRACE:
            if z_col is not None:
                warnings.warn(
                    f"z_col ignored for TRACE readout '{readout.name}'",
                    stacklevel=2,
                )
            return None
        # IMAGE kind
        if z_col is None:
            warnings.warn(
                f"z_col not set for IMAGE readout '{readout.name}', defaulting to column 0",
                stacklevel=2,
            )
            return 0
        return _resolve_col(z_col, readout, 'z_col')

    def extract_col(self, raw: np.ndarray, z_col, readout) -> np.ndarray:
        """Extract the relevant channel from raw readout data."""
        col = self.resolve_col(z_col, readout)
        if col is None:
            return raw.astype(np.float64)
        return np.asarray(raw[col], dtype=np.float64)

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
