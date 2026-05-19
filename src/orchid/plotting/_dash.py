"""DashPlotter — Dash/Werkzeug backend (poll-based) and UI helpers."""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from ._spec import (
    PlotSpec,
    EventLineConfig,
    _deep_merge,
    _EV_PALETTE,
    _format_elapsed_display,
    _format_eta,
)
from ._base import PlotterBase


# ══════════════════════════════════════════════════════════════════════
#  DashPlotter UI helpers  (Dash imports are deferred to call time)
# ══════════════════════════════════════════════════════════════════════


def _lp_line_trace_info(plotter) -> list[tuple[str, str]]:
    """Return ``[(trace_name, css_color_var)]`` for line-plot traces.

    Heatmap traces don't consume colorway slots, so we skip them when
    computing the scatter index that maps to ``--trace-N`` CSS variables.
    """
    if plotter._fig_dict is None or not plotter._resolved_types:
        return []
    result = []
    scatter_idx = 0
    strip_idx = getattr(plotter, '_strip_trace_idx', None)
    analysis_start = getattr(plotter, '_analysis_trace_start_idx', None)
    n_data = strip_idx if strip_idx is not None else len(plotter._fig_dict["data"])
    if analysis_start is not None:
        n_data = min(n_data, analysis_start)
    for i, ptype in enumerate(plotter._resolved_types):
        offset = plotter._trace_offsets[i]
        n_next = (plotter._trace_offsets[i + 1]
                  if i + 1 < len(plotter._trace_offsets)
                  else n_data)
        n = n_next - offset
        if ptype in ("heatmap", "trace_heatmap"):
            pass  # heatmaps don't consume the scatter colorway
        elif ptype == "line":
            for j in range(n):
                tidx = offset + j
                name = plotter._fig_dict["data"][tidx].get("name", f"trace {j}")
                css_color = f"var(--trace-{scatter_idx % 3})"
                result.append((name, css_color))
                scatter_idx += 1
        else:
            scatter_idx += n  # live_trace — consumes slots but not shown in rail
    return result


def _lp_has_rail(plotter) -> bool:
    """True when at least one rail section has content to show."""
    if plotter._proc is None:
        return False
    has_sweeps = bool(getattr(plotter._proc, "sweeps", None))
    has_readouts = bool(plotter.rail_readouts)
    has_traces = bool(_lp_line_trace_info(plotter))
    has_instruments = bool(plotter.instrument_info)
    has_events = bool(getattr(plotter, '_events', None))
    has_analysis = bool(getattr(plotter, '_analysis_results', None))
    return has_sweeps or has_readouts or has_traces or has_instruments or has_events or has_analysis


def _lp_rail_children(plotter) -> list:
    """Build the full children list for the ``#lp-rail`` div.

    Called on every refresh tick so sweep values and readout values
    stay current.  Static groups (traces, instruments) are rebuilt each
    time but produce the same VDOM nodes, so Dash's diff avoids extra
    DOM mutations.
    """
    from dash import html

    proc = plotter._proc
    children = []

    # ── Sweep group ───────────────────────────────────────────────────
    sweeps = getattr(proc, "sweeps", None) or []
    if sweeps:
        kv_rows = []
        sv = plotter._last_sweep_values
        n_axes = len(sweeps)

        for ax_idx, sweep in enumerate(sweeps):
            # Gather per-controller value arrays
            if hasattr(sweep, "all_values"):  # MultiSweep
                ctrl_arrays = list(zip(sweep.controllers, sweep.all_values))
            else:
                ctrl_arrays = [(sweep.controller, sweep.values)]

            for ctrl, arr in ctrl_arrays:
                unit = getattr(ctrl, "unit", None) or ""
                mn, mx = float(arr.min()), float(arr.max())

                # Range row
                kv_rows.append(html.Div([
                    html.Span(f"{ctrl.name} Range", className="lp-kv-k"),
                    html.Span(f"{mn:.4g} → {mx:.4g}{' ' + unit if unit else ''}",
                              className="lp-kv-v"),
                ], className="lp-kv"))

        # Points row  (e.g. "56 × 40")
        pts_str = " × ".join(str(s.length) for s in sweeps)
        kv_rows.append(html.Div([
            html.Span("Points", className="lp-kv-k"),
            html.Span(pts_str, className="lp-kv-v"),
        ], className="lp-kv"))

        # ETA row
        settle = getattr(proc, "settle_time", 0.0) or 0.0
        if settle > 0:
            total_pts = 1
            for s in sweeps:
                total_pts *= s.length
            eta_str = _format_eta(total_pts * settle)
            kv_rows.append(html.Div([
                html.Span("ETA", className="lp-kv-k"),
                html.Span(eta_str, className="lp-kv-v"),
            ], className="lp-kv"))

        children.append(html.Div([
            html.Div([
                html.Span("Sweep", className="lp-group-title"),
                html.Span(f"{n_axes} {'axis' if n_axes == 1 else 'axes'}",
                          className="lp-group-right"),
            ], className="lp-group-head"),
            *kv_rows,
        ]))

    # ── Readouts group ────────────────────────────────────────────────
    if plotter.rail_readouts:
        kv_rows = []
        for name in plotter.rail_readouts:
            val = plotter._last_rail_data.get(name)
            if val is None:
                val_str = "—"
            elif isinstance(val, float):
                val_str = f"{val:.4g}"
            else:
                val_str = str(val)
            unit = ""
            if hasattr(proc, "bench"):
                ro = proc.bench.readouts.get(name)
                if ro:
                    unit = getattr(ro, "unit", None) or ""
            display = f"{val_str} {unit}".strip() if unit else val_str
            kv_rows.append(html.Div([
                html.Span(name, className="lp-kv-k"),
                html.Span(display, className="lp-kv-v"),
            ], className="lp-kv"))
        children.append(html.Div([
            html.Div(html.Span("Readouts", className="lp-group-title"),
                     className="lp-group-head"),
            *kv_rows,
        ]))

    # ── Traces group (line plots only) ────────────────────────────────
    traces = _lp_line_trace_info(plotter)
    if traces:
        rows = []
        for name, css_color in traces:
            rows.append(html.Div([
                html.Span(style={
                    "background": css_color,
                    "display": "inline-block",
                    "width": "24px", "height": "3px",
                    "borderRadius": "2px", "flexShrink": "0",
                }, className="lp-swatch"),
                html.Span(name, className="lp-trace-name"),
            ], className="lp-trace-row"))
        children.append(html.Div([
            html.Div(html.Span("Traces", className="lp-group-title"),
                     className="lp-group-head"),
            *rows,
        ]))

    # ── Instruments group ─────────────────────────────────────────────
    if plotter.instrument_info:
        rows = []
        for name, detail in plotter.instrument_info.items():
            rows.append(html.Div([
                html.Span(className="lp-dot lp-dot-ok"),
                html.Div([
                    html.Div(name, className="lp-inst-name"),
                    html.Div(str(detail), className="lp-inst-detail"),
                ], className="lp-inst-meta"),
            ], className="lp-inst-row"))
        children.append(html.Div([
            html.Div(html.Span("Instruments", className="lp-group-title"),
                     className="lp-group-head"),
            *rows,
        ]))

    # ── Analysis group ────────────────────────────────────────────────
    analysis_results = getattr(plotter, '_analysis_results', None) or []
    for ri, pr in enumerate(analysis_results):
        color = pr.color or _EV_PALETTE[ri % len(_EV_PALETTE)]
        section_rows = []

        # VLines
        if pr.vlines:
            section_rows.append(html.Div("VLines", className="lp-analysis-sub"))
            for vl in pr.vlines:
                section_rows.append(html.Div([
                    html.Span(vl["name"], className="lp-kv-k"),
                    html.Span(f"x = {vl['x']:.4g}", className="lp-kv-v"),
                ], className="lp-kv lp-analysis-item"))

        # HLines
        if pr.hlines:
            section_rows.append(html.Div("HLines", className="lp-analysis-sub"))
            for hl in pr.hlines:
                section_rows.append(html.Div([
                    html.Span(hl["name"], className="lp-kv-k"),
                    html.Span(f"y = {hl['y']:.4g}", className="lp-kv-v"),
                ], className="lp-kv lp-analysis-item"))

        # Points
        if pr.points:
            section_rows.append(html.Div([
                html.Span("Points", className="lp-analysis-sub", style={"flex": "1"}),
                html.Span(f"({len(pr.points)})", className="lp-group-right"),
            ], style={"display": "flex", "alignItems": "baseline"}))
            for pi, pt in enumerate(pr.points):
                section_rows.append(html.Div([
                    html.Div([
                        html.Span(str(pi), className="lp-analysis-idx", style={"color": color}),
                        html.Span(pt["name"], className="lp-kv-k",
                                  style={"maxWidth": "none", "flex": "1"}),
                    ], style={"display": "flex", "alignItems": "baseline",
                              "gap": "5px", "flex": "1"}),
                    html.Span(f"{pt['x']:.4g},  {pt['y']:.4g}", className="lp-kv-v"),
                ], className="lp-kv lp-analysis-item"))

        # Boxes
        if pr.boxes:
            section_rows.append(html.Div([
                html.Span("Boxes", className="lp-analysis-sub", style={"flex": "1"}),
                html.Span(f"({len(pr.boxes)})", className="lp-group-right"),
            ], style={"display": "flex", "alignItems": "baseline"}))
            for bi, box in enumerate(pr.boxes):
                section_rows.append(html.Div([
                    html.Div([
                        html.Span(str(bi), className="lp-analysis-idx", style={"color": color}),
                        html.Span(box["name"], className="lp-kv-k",
                                  style={"maxWidth": "none", "flex": "1"}),
                    ], style={"display": "flex", "alignItems": "baseline",
                              "gap": "5px", "flex": "1"}),
                    html.Div([
                        html.Div(f"x  {box['x0']:.4g} → {box['x1']:.4g}", className="lp-kv-v"),
                        html.Div(f"y  {box['y0']:.4g} → {box['y1']:.4g}", className="lp-kv-v"),
                    ]),
                ], className="lp-kv lp-analysis-item"))

        # railpanel KV rows
        if pr.railpanel:
            for k, v in pr.railpanel.items():
                if isinstance(v, float):
                    v_str = f"{v:.4g}"
                else:
                    v_str = str(v)
                section_rows.append(html.Div([
                    html.Span(k, className="lp-kv-k"),
                    html.Span(v_str, className="lp-kv-v"),
                ], className="lp-kv lp-analysis-item"))

        sub_badge = f"[sub {pr.subplot}]" if pr.subplot > 0 else ""
        children.append(html.Div([
            html.Div([
                html.Span(style={"background": color, "borderRadius": "50%",
                                 "width": "8px", "height": "8px",
                                 "display": "inline-block", "flexShrink": "0"},
                          className="lp-analysis-dot"),
                html.Span(pr.name, className="lp-group-title",
                          style={"marginLeft": "6px", "flex": "1"}),
                html.Span(sub_badge, className="lp-group-right"),
            ], className="lp-group-head"),
            *section_rows,
        ]))

    # ── Events group ──────────────────────────────────────────────────
    if getattr(plotter, '_events', None):
        selected_set = set(plotter._event_selection or [])
        n_sel = len(selected_set)
        rows = []
        for ev in reversed(plotter._events):
            is_sel = ev["id"] in selected_set
            color = ev.get("color", "#888")
            is_toggle = ev.get("is_toggle", False)
            subtitle = "state" if is_toggle else "setpoint"
            val_cls = "lp-ev-val"
            if is_toggle:
                val_cls += " lp-ev-val-on" if ev["value"] else " lp-ev-val-off"
            rows.append(html.Div(
                id={"type": "lp-ev-row", "id": ev["id"]},
                className="lp-ev-row" + (" lp-ev-row-sel" if is_sel else ""),
                n_clicks=0,
                children=[
                    html.Span(style={"background": color}, className="lp-ev-colorbar"),
                    html.Span("✕" if is_sel else "", className="lp-ev-check" + (" lp-ev-check-sel" if is_sel else "")),
                    html.Span(ev["t_label"], className="lp-ev-t"),
                    html.Div([
                        html.Div(ev["param"],  className="lp-ev-param"),
                        html.Div(subtitle,     className="lp-ev-subtitle"),
                    ], className="lp-ev-center"),
                    html.Span(ev["val_label"], className=val_cls),
                ],
            ))
        children.append(html.Div([
            html.Div([
                html.Span("Events", className="lp-group-title"),
                html.Span(str(len(plotter._events)), className="lp-group-right"),
            ], className="lp-group-head"),
            html.Div(
                className="lp-ev-selbar" + ("" if n_sel else " lp-ev-selbar-empty"),
                children=[
                    html.Span([
                        html.Span(str(n_sel), className="lp-ev-sel-count"),
                        " Selected",
                    ], className="lp-ev-selbar-label"),
                    html.Button("Clear · Esc", id="lp-ev-clear",
                                className="lp-btn lp-btn-ghost lp-btn-sm", n_clicks=0),
                ],
            ),
            html.Div(rows, className="lp-ev-rows"),
        ]))

    return children


def _lp_header(proc_name: str, theme_name: str, plotter) -> object:
    """Build the ``<header>`` bar for the DashPlotter page."""
    from ._themes import THEMES
    from dash import dcc, html

    # Appearance options: one entry per theme
    options = []
    for key, td in THEMES.items():
        traces = td.get("traces", ["#888"])
        swatches = [
            html.Span(style={
                "backgroundColor": traces[i] if i < len(traces) else "#ccc",
                "display": "inline-block",
                "width": "14px", "height": "28px",
            })
            for i in range(3)
        ]
        label = html.Div([
            html.Span(swatches, style={
                "display": "inline-flex",
                "border": "1px solid rgba(128,128,128,0.2)",
            }),
            html.Div([
                html.Div(td["name"], className="lp-theme-name"),
                html.Div(td.get("sub", ""), className="lp-theme-sub"),
            ]),
        ], className="lp-theme-option")
        options.append({"label": label, "value": key})

    # Mini swatches for the summary trigger
    cur_td = THEMES.get(theme_name, THEMES["orchid"])
    cur_traces = cur_td.get("traces", ["#888"])
    summary_swatches = [
        html.Span(style={
            "backgroundColor": cur_traces[i] if i < len(cur_traces) else "#ccc",
            "display": "inline-block", "width": "8px", "height": "12px",
        })
        for i in range(3)
    ]

    return html.Div(className="lp-header", children=[
        # Brand mark
        html.Div(className="lp-brand", children=[
            html.Div(className="lp-mark"),
            html.Div([
                html.Div("orchid", className="lp-brand-name"),
                html.Div("Live Plot", className="lp-brand-sub"),
            ]),
        ]),
        html.Div(className="lp-divider"),
        html.Div(className="lp-exp-block", children=[
            html.Span(proc_name, className="lp-exp-name"),
            html.Div(id="lp-data-info", className="lp-data-info"),
        ]),
        html.Div(className="lp-spacer"),
        # Elapsed + status
        html.Div(className="lp-meta", children=[
            html.Div([
                html.Div("Elapsed", className="lp-label"),
                html.Span("--:--", id="lp-elapsed", className="lp-mono"),
            ]),
            html.Div(className="lp-status", children=[
                html.Span(id="lp-dot", className="lp-dot lp-dot-pulse"),
                html.Span("Running", id="lp-status-text", className="lp-status-text"),
            ]),
        ]),
        # Stop button — rendered in monitor and sweep modes
        *([ html.Button(
            "⏹ Stop" if not plotter._stopped else "✓ Stopped",
            id="lp-stop-btn",
            className="lp-btn lp-btn-stop" + (" lp-btn-stop-done" if plotter._stopped else ""),
            n_clicks=0,
            disabled=plotter._stopped,
        )] if (plotter._is_monitor or plotter._is_sweep) else [
            html.Div(id="lp-stop-btn", style={"display": "none"}, n_clicks=0),
        ]),
        # Appearance dropdown
        html.Details(className="lp-appearance", children=[
            html.Summary([
                html.Span("Appearance", className="lp-appearance-current"),
                html.Span(
                    summary_swatches,
                    className="lp-appearance-swatches",
                    style={"marginLeft": "8px"},
                ),
                html.Span(className="lp-chev"),
            ]),
            html.Div(className="lp-appearance-panel", children=[
                html.Div("Theme", className="lp-appearance-heading"),
                dcc.RadioItems(
                    id="lp-theme-radio",
                    options=options,
                    value=theme_name,
                    labelClassName="lp-theme-option-label",
                    inputStyle={"display": "none"},
                ),
            ]),
        ]),
        # Snapshot
        html.Div(className="lp-acq", children=[
            html.Button("Snapshot", id="lp-snapshot", className="lp-btn lp-btn-ghost",
                        n_clicks=0),
        ]),
    ])


# ══════════════════════════════════════════════════════════════════════
#  Static save / load helpers
# ══════════════════════════════════════════════════════════════════════

import numpy as _np

# Port → DashPlotter instance.  Lets _start_server auto-stop any previous
# plotter on the same port so DashPlotter.load() / DashPlotter() can be
# called repeatedly in a notebook without "Address already in use" errors.
_port_registry: dict[int, "DashPlotter"] = {}


def _json_encode(x):
    """JSON ``default`` handler: converts numpy arrays and scalars to Python types."""
    if isinstance(x, _np.ndarray):
        return x.tolist()
    if isinstance(x, _np.generic):   # np.float32, np.int64, etc.
        return x.item()
    raise TypeError(f"Object of type {type(x).__name__} is not JSON serializable")


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
        Dash polling interval in milliseconds. Default 500 ms.
    event_line : EventLineConfig, optional
        Visual style for parameter-change markers on monitor plots.
    max_display_pts : int
        Maximum points shown on line plots (rolling window for monitors).
        Default 5000.
    theme : str
        Initial UI theme. One of ``"orchid"`` (default), ``"t1000"``,
        ``"vitsoe"``, ``"modern"``, ``"console"``.  Switchable live via
        the Appearance dropdown in the browser.
    rail_readouts : list of str, optional
        Readout names whose scalar values appear in the control rail.
        Must be ``DataKind.SCALAR`` readouts registered on the bench.
    instrument_info : dict, optional
        ``{name: detail}`` annotations for the Instruments rail group,
        e.g. ``{"SR830": "GPIB::8", "Keithley": "USB0::..."}``.

    Examples
    --------
    >>> plotter = DashPlotter([PlotSpec(x="Vgt", y="lockin_X")])
    >>> runner.run(proc, plotter=plotter)

    >>> plotter = DashPlotter(
    ...     [PlotSpec(x="Vgt", y="lockin_X")],
    ...     theme="console",
    ...     rail_readouts=["T_mc"],
    ...     instrument_info={"SR830": "GPIB::8"},
    ... )
    """

    def __init__(
        self,
        plots: list[PlotSpec],
        port: int = 8050,
        height: int = 350,
        width: int = 700,
        open_browser: bool = False,
        update_interval: int = 500,
        event_line: EventLineConfig | None = None,
        max_display_pts: int = 5000,
        theme: str = "orchid",
        rail_readouts: list[str] | None = None,
        instrument_info: dict[str, str] | None = None,
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
        self.rail_readouts: list[str] = list(rail_readouts) if rail_readouts else []
        self.instrument_info: dict[str, str] = dict(instrument_info) if instrument_info else {}

        self._current_theme: str = theme
        self._last_rail_data: dict = {}
        self._last_sweep_values: dict = {}
        self._start_time: float | None = None
        self._data_dir: str | None = None
        self._experiment_id: str | None = None
        self._events: list[dict] = []
        self._event_selection: list = []
        self._strip_trace_idx: int | None = None
        self._param_colors: dict[str, str] = {}
        self._analysis_results: list = []
        self._analysis_trace_start_idx: int | None = None
        self._stop_fn = None       # set by runner via set_stop_callback()
        self._is_monitor: bool = False
        self._is_sweep: bool = False
        self._prepared: bool = False  # set by prepare(); consumed by runner

        # Poll-version counter — Dash reads these in the refresh() callback
        self._data_version = 0
        self._last_sent_version = -1

        self._server_thread: threading.Thread | None = None
        self._wsgi_server = None
        self._dash_app = None

    def setup(self, proc) -> None:
        # Reset all per-experiment state before base setup (which builds the figure)
        self._data_version = 0
        self._last_sent_version = -1
        self._last_rail_data = {}
        self._last_sweep_values = {}
        self._start_time = time.time()
        self._events = []
        self._event_selection = []
        self._strip_trace_idx = None
        self._param_colors = {}
        self._analysis_results = []
        self._analysis_trace_start_idx = None
        self._stop_fn = None
        self._is_monitor = hasattr(proc, 'interval') and not hasattr(proc, 'sweeps')
        self._is_sweep = hasattr(proc, 'sweeps') and bool(proc.sweeps)
        self._prepared = False
        super().setup(proc)

    # ── Rail data caching ──────────────────────────────────────────────

    def dispatch(self, event: str, index, data: dict, sweep_values: dict) -> None:
        """Cache sweep / readout values for the rail, then route to base."""
        if sweep_values:
            self._last_sweep_values = dict(sweep_values)
        if self.rail_readouts and data:
            for name in self.rail_readouts:
                if name in data:
                    self._last_rail_data[name] = data[name]
        super().dispatch(event, index, data, sweep_values)

    def update_monitor(self, sample_idx: int, data: dict, timestamp: float) -> None:
        """Cache readout values for the rail; detect time-unit changes for strip."""
        if self.rail_readouts and data:
            for name in self.rail_readouts:
                if name in data:
                    self._last_rail_data[name] = data[name]
        old_unit = self._time_unit_from_state()
        super().update_monitor(sample_idx, data, timestamp)
        new_unit = self._time_unit_from_state()
        if (new_unit is not None and new_unit != old_unit
                and self._strip_trace_idx is not None
                and self._t0 is not None and self._events):
            self._rescale_strip_for_unit(new_unit)

    def _time_unit_from_state(self) -> str | None:
        """Return current time unit from first _time spec's state dict."""
        for i, spec in enumerate(self.specs):
            if isinstance(spec.x, str) and spec.x == "_time" and i in self._sweep_data:
                states = self._sweep_data[i]
                if states:
                    return states[0].get("_unit")
        return None

    def _rescale_strip_for_unit(self, unit: str) -> None:
        """Rescale strip trace x values and _events t_elapsed to the new unit."""
        divisor = self.unit_divisor(unit)
        for ev in self._events:
            ev["t_elapsed"] = (ev["t_abs"] - self._t0) / divisor
        trace = self._fig_dict["data"][self._strip_trace_idx]
        trace["x"] = [ev["t_elapsed"] for ev in self._events]
        if self._event_selection:
            self._apply_event_selection(self._event_selection)

    # ── Theme ──────────────────────────────────────────────────────────

    def _theme_layout(self, fig) -> None:
        """Apply the current colour theme to the freshly built figure.

        Also sets ``autosize=True`` so the figure fills its CSS container
        (the `.lp-panel` flex cell) rather than using the fixed
        ``height_per_plot * n`` value set in ``build_figure_dict``.
        """
        from ._themes import THEMES, plotly_template
        theme = THEMES.get(self._current_theme, THEMES["orchid"])
        tpl = plotly_template(theme)
        axis_style = {k: v for k, v in tpl.pop("xaxis", {}).items() if k != "title"}
        y_axis_style = {k: v for k, v in tpl.pop("yaxis", {}).items() if k != "title"}
        # Clear the explicit pixel height/width set in build_figure_dict so the
        # figure fills its CSS container (.lp-panel) instead of overflowing it.
        fig.update_layout(autosize=True, height=None, width=None, **tpl)
        if axis_style:
            fig.update_xaxes(**axis_style)
        if y_axis_style:
            fig.update_yaxes(**y_axis_style)

        # Monitor mode: compress data axes into bottom 88 % and add diamond strip
        is_monitor = not hasattr(self._proc, 'sweeps') or not self._proc.sweeps
        if is_monitor:
            import plotly.graph_objects as go
            for attr_name in dir(fig.layout):
                if attr_name.startswith("yaxis"):
                    axis = getattr(fig.layout, attr_name)
                    if axis.domain:
                        d = list(axis.domain)
                        axis.domain = [d[0] * 0.88, d[1] * 0.88]
            # yaxis indices 1..n_subplots are used by make_subplots; strip gets n+1
            n_subplots = len(self.specs)
            strip_num = n_subplots + 1
            strip_key = f"yaxis{strip_num}"
            strip_ref = f"y{strip_num}"
            fig.update_layout(**{strip_key: dict(
                domain=[0.91, 1.0], visible=False, range=[-0.1, 1.1],
                anchor="x",
            )})
            self._strip_trace_idx = len(fig.data)
            fig.add_trace(go.Scatter(
                x=[], y=[], mode="markers",
                yaxis=strip_ref,
                marker=dict(symbol="diamond", size=10,
                            color=[], opacity=[], line=dict(width=0)),
                customdata=[],
                hovertemplate="<b>%{customdata[0]}</b>=%{customdata[1]}<br>t=%{customdata[2]}<extra></extra>",
                showlegend=False, name="_ev_strip",
            ))

    def _retheme_fig_dict(self, theme_name: str) -> None:
        """Apply a new theme to ``_fig_dict`` in-place (no go.Figure needed)."""
        if self._fig_dict is None:
            return
        from ._themes import THEMES, plotly_template
        theme = THEMES.get(theme_name, THEMES["orchid"])
        tpl = plotly_template(theme)
        # Strip axis sub-dicts; merge them separately to preserve title text
        axis_style = {k: v for k, v in tpl.pop("xaxis", {}).items() if k != "title"}
        y_axis_style = {k: v for k, v in tpl.pop("yaxis", {}).items() if k != "title"}
        layout = self._fig_dict["layout"]
        layout.update(tpl)
        layout["autosize"] = True
        layout.pop("height", None)
        layout.pop("width", None)
        for key in list(layout.keys()):
            if key.startswith("xaxis") and isinstance(layout[key], dict):
                _deep_merge(layout[key], axis_style)
            elif key.startswith("yaxis") and isinstance(layout[key], dict):
                _deep_merge(layout[key], y_axis_style)
        self._current_theme = theme_name

    def set_run_info(self, data_dir, experiment_id: str | None = None) -> None:
        """Store data path for display in the header; triggers a UI refresh.

        Pass ``data_dir=None`` (write_mode=NONE) to show a "not saving" badge.
        """
        self._data_dir = "" if data_dir is None else str(data_dir)
        self._experiment_id = experiment_id
        self._data_version += 1

    def notify_event(self, timestamp: float, param: str, value) -> None:
        """Record event and update strip marker. No hairlines on the main plot."""
        if self._fig_dict is None or self._t0 is None:
            return

        # Keep _event_timestamps in sync so base-class unit-rescaling still works
        self._event_timestamps.append(timestamp)

        elapsed = timestamp - self._t0
        x_val, _unit = self.format_elapsed(elapsed)
        is_toggle = isinstance(value, bool)
        if is_toggle:
            val_label = "ON" if value else "OFF"
        elif isinstance(value, (int, float)):
            val_label = f"{value:.4g}"
        else:
            val_label = str(value)
        t_label = _format_elapsed_display(elapsed)

        if param not in self._param_colors:
            self._param_colors[param] = _EV_PALETTE[len(self._param_colors) % len(_EV_PALETTE)]
        color = self._param_colors[param]

        ev = {
            "id": len(self._events),
            "t_abs": timestamp,
            "t_elapsed": x_val,
            "param": param,
            "value": value,
            "t_label": t_label,
            "val_label": val_label,
            "is_toggle": is_toggle,
            "color": color,
        }
        self._events.append(ev)

        if self._strip_trace_idx is not None:
            trace = self._fig_dict["data"][self._strip_trace_idx]
            _ml = lambda v: list(v) if isinstance(v, list) else []
            xs = _ml(trace.get("x")) + [x_val]
            ys = _ml(trace.get("y")) + [0.5]
            cd = _ml(trace.get("customdata")) + [[param, val_label, t_label]]
            colors = _ml(trace.get("marker", {}).get("color")) + [color]
            sizes = _ml(trace.get("marker", {}).get("size")) + [10]
            opacities = _ml(trace.get("marker", {}).get("opacity")) + [0.55]
            trace["x"] = xs
            trace["y"] = ys
            trace["customdata"] = cd
            trace["marker"] = {**trace.get("marker", {}),
                               "color": colors, "size": sizes, "opacity": opacities}
        self.on_data_changed()

    def _apply_event_selection(self, selected_ids: list) -> None:
        """Update strip marker sizes/opacities and vertical guide shapes."""
        if self._fig_dict is None or self._strip_trace_idx is None:
            return
        selected_set = set(selected_ids)

        trace = self._fig_dict["data"][self._strip_trace_idx]
        n = len(trace.get("x") or [])
        sizes =     [13 if self._events[i]["id"] in selected_set else 10  for i in range(n)]
        opacities = [1.0 if self._events[i]["id"] in selected_set else 0.55 for i in range(n)]
        trace["marker"] = {**trace.get("marker", {}), "size": sizes, "opacity": opacities}

        layout = self._fig_dict["layout"]
        shapes = [s for s in layout.get("shapes", []) if not s.get("_ev_guide")]
        for ev in self._events:
            if ev["id"] not in selected_set:
                continue
            # Draw one guide per _time subplot, each confined to its own axis.
            for i, spec in enumerate(self.specs):
                if not isinstance(spec.x, str) or spec.x != "_time":
                    continue
                xref  = "x" if i == 0 else f"x{i + 1}"
                yaxis = "y" if i == 0 else f"y{i + 1}"
                shapes.append({
                    "_ev_guide": True,
                    "type": "line",
                    "xref": xref, "yref": f"{yaxis} domain",
                    "x0": ev["t_elapsed"], "x1": ev["t_elapsed"],
                    "y0": 0.0, "y1": 1.0,
                    "line": {"color": ev["color"], "width": 1.5},
                    "opacity": 0.6,
                })
        layout["shapes"] = shapes
        self._data_version += 1

    def show_analysis(self, results: list) -> None:
        """Overlay analysis traces, lines, points, and boxes on the live plot.

        Parameters
        ----------
        results : list of PostResult
            One or more analysis results. Each is rendered as its own rail
            section. Call after ``runner.run()`` completes.
        """
        self._analysis_results = list(results)
        if self._fig_dict is None:
            return

        from ._themes import THEMES
        _theme = THEMES.get(self._current_theme, THEMES["orchid"])
        _chip_bg   = _theme["panel"]
        _chip_ink  = _theme["ink"]
        _chip_mute = _theme["ink_faint"]

        layout = self._fig_dict["layout"]

        # Remove any previous analysis elements
        layout["shapes"] = [s for s in layout.get("shapes", []) if not s.get("_analysis")]
        layout["annotations"] = [a for a in layout.get("annotations", []) if not a.get("_analysis")]
        # Remove previous analysis traces
        if self._analysis_trace_start_idx is not None:
            self._fig_dict["data"] = self._fig_dict["data"][:self._analysis_trace_start_idx]

        self._analysis_trace_start_idx = len(self._fig_dict["data"])
        shapes = list(layout.get("shapes", []))
        annotations = list(layout.get("annotations", []))

        def _xref(sub: int) -> str:
            return "x" if sub == 0 else f"x{sub + 1}"

        def _yref(sub: int) -> str:
            return "y" if sub == 0 else f"y{sub + 1}"

        def _ydomain(sub: int) -> list:
            key = "yaxis" if sub == 0 else f"yaxis{sub + 1}"
            return list(layout.get(key, {}).get("domain", [0.0, 1.0]))

        def _xdomain(sub: int) -> list:
            key = "xaxis" if sub == 0 else f"xaxis{sub + 1}"
            return list(layout.get(key, {}).get("domain", [0.0, 1.0]))

        for ri, pr in enumerate(results):
            color = pr.color or _EV_PALETTE[ri % len(_EV_PALETTE)]
            default_sub = pr.subplot

            # ── Traces ────────────────────────────────────────────────
            for t in (pr.traces or []):
                sub = t.get("subplot", default_sub)
                self._fig_dict["data"].append({
                    "type": "scatter",
                    "x": list(t["x"]),
                    "y": list(t["y"]),
                    "name": t.get("name", pr.name),
                    "mode": t.get("mode", "lines"),
                    "xaxis": _xref(sub),
                    "yaxis": _yref(sub),
                    "line": {
                        "color": t.get("color", color),
                        "width": t.get("width", 2),
                        "dash":  t.get("dash", "dot"),
                    },
                    "showlegend": True,
                    "_analysis": True,
                })

            # ── VLines ────────────────────────────────────────────────
            for vl in (pr.vlines or []):
                sub = vl.get("subplot", default_sub)
                yd = _ydomain(sub)
                shapes.append({
                    "_analysis": True,
                    "type": "line",
                    "xref": _xref(sub), "yref": "paper",
                    "x0": vl["x"], "x1": vl["x"],
                    "y0": yd[0], "y1": yd[1],
                    "line": {"color": color, "width": 1.5, "dash": "dash"},
                })
                chip_text = (
                    f"<span style='font-size:9px;color:{_chip_mute};"
                    f"text-transform:uppercase;letter-spacing:0.06em'>"
                    f"{vl['name']}</span><br>"
                    f"<b style='color:{_chip_ink}'>x = {vl['x']:.4g}</b>"
                )
                annotations.append({
                    "_analysis": True,
                    "xref": _xref(sub), "yref": "paper",
                    "x": vl["x"], "y": yd[1],
                    "text": chip_text,
                    "showarrow": True,
                    "arrowhead": 0, "arrowwidth": 1,
                    "arrowcolor": color, "arrowside": "none",
                    "ax": 0, "ay": 4, "axref": "pixel", "ayref": "pixel",
                    "xanchor": "center", "yanchor": "top",
                    "bgcolor": _chip_bg,
                    "bordercolor": color, "borderwidth": 1.5, "borderpad": 5,
                    "align": "left",
                    "font": {"family": "ui-monospace, monospace", "size": 10,
                             "color": _chip_ink},
                })

            # ── HLines ────────────────────────────────────────────────
            for hl in (pr.hlines or []):
                sub = hl.get("subplot", default_sub)
                xd = _xdomain(sub)
                shapes.append({
                    "_analysis": True,
                    "type": "line",
                    "xref": "paper", "yref": _yref(sub),
                    "x0": xd[0], "x1": xd[1],
                    "y0": hl["y"], "y1": hl["y"],
                    "line": {"color": color, "width": 1.5, "dash": "dash"},
                })
                chip_text = (
                    f"<span style='font-size:9px;color:{_chip_mute};"
                    f"text-transform:uppercase;letter-spacing:0.06em'>"
                    f"{hl['name']}</span><br>"
                    f"<b style='color:{_chip_ink}'>y = {hl['y']:.4g}</b>"
                )
                annotations.append({
                    "_analysis": True,
                    "xref": "paper", "yref": _yref(sub),
                    "x": xd[0], "y": hl["y"],
                    "text": chip_text,
                    "showarrow": True,
                    "arrowhead": 0, "arrowwidth": 1,
                    "arrowcolor": color, "arrowside": "none",
                    "ax": 6, "ay": 0, "axref": "pixel", "ayref": "pixel",
                    "xanchor": "left", "yanchor": "middle",
                    "bgcolor": _chip_bg,
                    "bordercolor": color, "borderwidth": 1.5, "borderpad": 5,
                    "align": "left",
                    "font": {"family": "ui-monospace, monospace", "size": 10,
                             "color": _chip_ink},
                })

            # ── Points (grouped by subplot into one trace each) ───────
            from collections import defaultdict
            pts_by_sub: dict[int, list] = defaultdict(list)
            for pi, pt in enumerate(pr.points or []):
                sub = pt.get("subplot", default_sub)
                pts_by_sub[sub].append((pi, pt))
            for sub, pt_list in pts_by_sub.items():
                xs = [p["x"] for _, p in pt_list]
                ys = [p["y"] for _, p in pt_list]
                labels = [str(i) for i, _ in pt_list]
                self._fig_dict["data"].append({
                    "type": "scatter",
                    "x": xs, "y": ys,
                    "mode": "markers+text",
                    "text": labels,
                    "textposition": "top center",
                    "name": f"{pr.name} points",
                    "xaxis": _xref(sub), "yaxis": _yref(sub),
                    "marker": {"color": color, "size": 10, "symbol": "circle"},
                    "showlegend": True,
                    "_analysis": True,
                })

            # ── Boxes ────────────────────────────────────────────────
            for bi, box in enumerate(pr.boxes or []):
                sub = box.get("subplot", default_sub)
                shapes.append({
                    "_analysis": True,
                    "type": "rect",
                    "xref": _xref(sub), "yref": _yref(sub),
                    "x0": box["x0"], "x1": box["x1"],
                    "y0": box["y0"], "y1": box["y1"],
                    "line": {"color": color, "width": 1.5},
                    "fillcolor": color,
                    "opacity": 0.15,
                })
                annotations.append({
                    "_analysis": True,
                    "xref": _xref(sub), "yref": _yref(sub),
                    "x": (box["x0"] + box["x1"]) / 2,
                    "y": (box["y0"] + box["y1"]) / 2,
                    "text": str(bi),
                    "showarrow": False,
                    "font": {"size": 11, "color": color},
                })

        layout["shapes"] = shapes
        layout["annotations"] = annotations
        self.on_data_changed()

    # ── PlotterBase interface ──────────────────────────────────────────

    def prepare(self, proc) -> None:
        """Set up the plotter before calling ``runner.run()`` or ``runner.run_monitor()``.

        Builds the figure, starts the Dash server, and opens the browser
        immediately — before the experiment begins.  The runner will detect
        that setup has already been done and skip its own ``setup()`` call,
        but will still reset the elapsed timer to the actual experiment start.

        Example::

            plotter.prepare(proc)        # browser opens here
            # ... instrument warmup ...
            runner.run_monitor(proc, plotter=plotter)
        """
        self.setup(proc)       # resets state, builds figure, starts server
        self._prepared = True  # must be set after setup() which clears it

    def _mark_start(self) -> None:
        """Reset the elapsed timer to now.  Called by the runner at the true
        experiment start, overriding the time set during setup() / prepare()."""
        self._start_time = time.time()
        self._final_elapsed = None

    def set_stop_callback(self, fn) -> None:
        """Register a callable that halts the running experiment.

        Called by the runner before the monitor loop starts so the Stop button
        in the browser can signal the measurement loop to exit.
        """
        self._stop_fn = fn

    def finalize(self) -> None:
        """Freeze the plot: latch elapsed time and stop live updates."""
        self._final_elapsed = _format_elapsed_display(
            time.time() - self._start_time
        ) if self._start_time is not None else "--:--"
        super().finalize()

    # ── Config save / load ────────────────────────────────────────────

    def save(self, data_dir) -> None:
        """Save plotter configuration and figure to *data_dir*.

        Writes two files:

        ``plotter_config.yaml``
            Plotter constructor args, ``PlotSpec`` list, and internal
            bookkeeping needed to restore the rail and header at load time.

        ``figure.json.gz``
            Complete Plotly figure dict (layout + all trace data) compressed
            with gzip.  Self-contained: no zarr dependency at load time.
        """
        import dataclasses, gzip, json, yaml

        el = self.event_line
        config = {
            "version": 2,
            "meta": {
                "proc_name": self._proc.name if self._proc else "unknown",
                "elapsed":   self._final_elapsed,
            },
            "specs": [s.to_dict() for s in self.specs],
            "plotter": {
                "port":            self.port,
                "update_interval": self.update_interval,
                "theme":           self._current_theme,
                "open_browser":    self.open_browser,
                "max_display_pts": self.max_display_pts,
                "rail_readouts":   list(self.rail_readouts),
                "instrument_info": dict(self.instrument_info),
                **({"event_line": dataclasses.asdict(el)} if el is not None else {}),
            },
            "internal": {
                "resolved_types":           list(self._resolved_types),
                "trace_offsets":            list(self._trace_offsets),
                "strip_trace_idx":          self._strip_trace_idx,
                "analysis_trace_start_idx": self._analysis_trace_start_idx,
                "is_monitor":               self._is_monitor,
                "monitor_interval":         getattr(self._proc, "interval", None),
                "time_unit":                self._time_unit_from_state(),
            },
        }

        data_dir = Path(data_dir)
        (data_dir / "plotter_config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
        )
        if self._fig_dict is not None:
            payload = json.dumps(self._fig_dict, default=_json_encode).encode()
            with gzip.open(data_dir / "figure.json.gz", "wb") as fh:
                fh.write(payload)

    @classmethod
    def load(cls, data_dir, *, port: int | None = 8052) -> "DashPlotter":
        """Load a saved plotter configuration and open the browser.

        Reads ``plotter_config.yaml`` and ``figure.json.gz`` from *data_dir*
        and starts the Dash server.  The plot is shown in a frozen (Done)
        state — identical to how it looked at the end of the original run.

        Parameters
        ----------
        data_dir : str or Path
            Directory containing ``plotter_config.yaml`` and ``figure.json.gz``.
        port : int, optional
            Override the port saved in the config.  Useful when the original
            port is already occupied by a running experiment::

                # live experiment on 8050, browse an old run in parallel
                old = DashPlotter.load("run_2024_001", port=8051)

        Example::

            from orchid import DashPlotter

            plotter = DashPlotter.load("/path/to/data_dir")
            # browser opens with the saved figure
        """
        import gzip, json, yaml
        from ._spec import EventLineConfig

        data_dir = Path(data_dir)
        config   = yaml.safe_load((data_dir / "plotter_config.yaml").read_text())
        with gzip.open(data_dir / "figure.json.gz", "rb") as fh:
            fig_dict = json.loads(fh.read())

        # Reconstruct plotter from saved args
        pa = dict(config["plotter"])
        if port is not None:
            pa["port"] = port
        if "event_line" in pa and pa["event_line"] is not None:
            pa["event_line"] = EventLineConfig(**pa["event_line"])
        specs   = [PlotSpec.from_dict(s) for s in config["specs"]]
        plotter = cls(plots=specs, **pa)

        # Restore internal bookkeeping so the rail / strip work correctly
        internal = config["internal"]
        plotter._resolved_types           = internal["resolved_types"]
        plotter._trace_offsets            = internal["trace_offsets"]
        plotter._strip_trace_idx          = internal.get("strip_trace_idx")
        plotter._analysis_trace_start_idx = internal.get("analysis_trace_start_idx")
        plotter._is_monitor               = internal.get("is_monitor", False)
        plotter._final_elapsed            = config["meta"].get("elapsed")
        plotter._fig_dict                 = fig_dict
        plotter._stopped                  = True

        # Minimal proc stub so the header shows the original experiment name
        proc_name = config.get("meta", {}).get("proc_name", "Orchid")
        plotter._proc = type("_ProcStub", (), {"name": proc_name})()

        plotter._start_server()
        return plotter

    def on_data_changed(self) -> None:
        """Increment the poll-version counter so Dash sends the next update."""
        self._data_version += 1

    @property
    def is_running(self) -> bool:
        """True if the Dash server thread is alive."""
        return self._server_thread is not None and self._server_thread.is_alive()

    def _start_server(self) -> None:
        """Launch Dash app on a background daemon thread."""
        import logging
        from dash import Dash, dcc, html, no_update, ctx
        from dash.dependencies import Input, Output, State, ALL

        for logger_name in ("werkzeug", "dash", "dash.dash", "flask", "flask.app"):
            logging.getLogger(logger_name).setLevel(logging.ERROR)

        assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
        app = Dash(
            __name__,
            update_title=None,
            assets_folder=assets_dir,
            suppress_callback_exceptions=True,
            external_scripts=[
                "https://html2canvas.hertzen.com/dist/html2canvas.min.js"
            ],
        )
        app.title = self._proc.name if self._proc else "Orchid Live Plot"
        app.logger.setLevel(logging.ERROR)
        self._dash_app = app

        plotter = self

        # Callable layout — re-evaluated on every fresh browser load so that a
        # new tab always sees the current figure and theme after setup() runs.
        def serve_layout():
            theme_name = plotter._current_theme
            proc_name = plotter._proc.name if plotter._proc else "Orchid"
            has_rail = _lp_has_rail(plotter)
            return html.Div(
                id="lp-root",
                className=f"theme-{theme_name}",
                children=[
                    _lp_header(proc_name, theme_name, plotter),
                    html.Div(
                        className="lp-body",
                        children=[
                            html.Div(
                                className="lp-plots",
                                children=[html.Div(
                                    className="lp-panel",
                                    children=[html.Div(
                                        className="lp-graph",
                                        children=[dcc.Graph(
                                            id="live-graph",
                                            figure=plotter._fig_dict or {},
                                            config={"responsive": True,
                                                    "displayModeBar": True},
                                            style={"height": "100%"},
                                        )],
                                    )],
                                )],
                            ),
                            # Rail — always in DOM so callback IDs resolve;
                            # lp-rail class only added when content exists.
                            html.Div(
                                id="lp-rail",
                                className="lp-rail" if has_rail else "",
                                children=_lp_rail_children(plotter) if has_rail else [],
                            ),
                        ],
                    ),
                    dcc.Interval(id="interval", interval=plotter.update_interval,
                                 n_intervals=0),
                    html.Div(id="lp-snap-dummy", style={"display": "none"}),
                    dcc.Store(id="lp-ev-selected", data=[]),
                    dcc.Store(id="lp-ev-anchor",   data=None),
                    dcc.Store(id="lp-ev-mods",     data={"shift": False, "ctrl": False}),
                    dcc.Store(id="lp-ev-kbd-init", data=""),
                ],
            )

        app.layout = serve_layout

        # ── Main refresh callback ──────────────────────────────────────
        @app.callback(
            Output("live-graph", "figure"),
            Output("lp-elapsed", "children"),
            Output("lp-dot", "className"),
            Output("lp-status-text", "children"),
            Output("lp-rail", "children"),
            Output("lp-data-info", "children"),
            Output("lp-stop-btn", "children"),
            Output("lp-stop-btn", "disabled"),
            Output("lp-stop-btn", "className"),
            Input("interval", "n_intervals"),
            Input("lp-theme-radio", "value"),
        )
        def refresh(n, theme_name):
            # Retheme when theme radio changes
            if theme_name != plotter._current_theme:
                plotter._retheme_fig_dict(theme_name)
                fig_out = plotter._fig_dict
                plotter._last_sent_version = plotter._data_version
            elif plotter._data_version != plotter._last_sent_version:
                plotter._last_sent_version = plotter._data_version
                fig_out = plotter._fig_dict
            else:
                fig_out = no_update

            # Elapsed time — latch to final value once stopped
            if plotter._stopped and plotter._final_elapsed is not None:
                elapsed_str = plotter._final_elapsed
            elif plotter._start_time is not None:
                elapsed_str = _format_elapsed_display(time.time() - plotter._start_time)
            else:
                elapsed_str = "--:--"

            # Status dot + text
            if plotter._stopped:
                dot_cls = "lp-dot lp-dot-idle"
                status_text = "Done"
            else:
                dot_cls = "lp-dot lp-dot-pulse"
                status_text = "Running"

            # Rail children (sweep values and readout values update each tick)
            has_rail = _lp_has_rail(plotter)
            rail_children = _lp_rail_children(plotter) if has_rail else []

            # Data path info — split into parent dir and experiment leaf
            import os as _os
            from dash import html as _html
            p = plotter._data_dir
            if p is None:
                # set_run_info not yet called — show nothing
                data_info = []
            elif p == "":
                # write_mode=NONE — explicitly not saving
                data_info = [_html.Span("not saving", className="lp-data-nosave")]
            else:
                home = _os.path.expanduser("~")
                parent = _os.path.dirname(p)
                leaf = _os.path.basename(p)
                parent_abbr = ("~" + parent[len(home):] if parent.startswith(home) else parent) + "/"
                data_info = [
                    _html.Span(parent_abbr, className="lp-data-dir"),
                    _html.Span(leaf, className="lp-data-id"),
                ]

            # Stop button state
            stopped = plotter._stopped
            stop_label = "✓ Stopped" if stopped else "⏹ Stop"
            stop_cls = "lp-btn lp-btn-stop" + (" lp-btn-stop-done" if stopped else "")

            return (fig_out, elapsed_str, dot_cls, status_text, rail_children,
                    data_info, stop_label, stopped, stop_cls)

        # ── Theme class update ─────────────────────────────────────────
        @app.callback(
            Output("lp-root", "className"),
            Input("lp-theme-radio", "value"),
            prevent_initial_call=True,
        )
        def update_theme_class(theme_name):
            return f"theme-{theme_name}"

        # ── Stop button ────────────────────────────────────────────────
        @app.callback(
            Output("lp-stop-btn", "children", allow_duplicate=True),
            Output("lp-stop-btn", "disabled", allow_duplicate=True),
            Output("lp-stop-btn", "className", allow_duplicate=True),
            Input("lp-stop-btn", "n_clicks"),
            prevent_initial_call=True,
        )
        def on_stop(n_clicks):
            if not n_clicks:
                return no_update, no_update, no_update
            # Signal the measurement loop to exit
            if plotter._stop_fn is not None:
                plotter._stop_fn()
            # Freeze the plot immediately (latch elapsed, set _stopped)
            plotter.finalize()
            return "✓ Stopped", True, "lp-btn lp-btn-stop lp-btn-stop-done"

        # ── Snapshot — client-side full-page capture via html2canvas ──
        app.clientside_callback(
            """
            function(n) {
                if (!n) return '';
                var root = document.getElementById('lp-root');
                var panel = root.querySelector('.lp-appearance-panel');
                if (panel) panel.style.display = 'none';
                html2canvas(root, {scale: 2, useCORS: true, logging: false})
                    .then(function(canvas) {
                        if (panel) panel.style.display = '';
                        var a = document.createElement('a');
                        a.download = 'orchid_snapshot.png';
                        a.href = canvas.toDataURL('image/png');
                        a.click();
                    });
                return '';
            }
            """,
            Output("lp-snap-dummy", "children"),
            Input("lp-snapshot", "n_clicks"),
            prevent_initial_call=True,
        )

        # ── Keyboard init + modifier tracking ─────────────────────────
        app.clientside_callback(
            """
            function(_) {
                if (window.__lp_ev_inited) return window.dash_clientside.no_update;
                window.__lp_ev_inited = true;
                window.__lp_mods = {shift: false, ctrl: false};
                var upd = function(e) {
                    window.__lp_mods.shift = e.shiftKey;
                    window.__lp_mods.ctrl  = e.ctrlKey || e.metaKey;
                };
                document.addEventListener('keydown', upd);
                document.addEventListener('keyup',   upd);
                document.addEventListener('keydown', function(e) {
                    if (e.key !== 'Escape') return;
                    var ae = document.activeElement;
                    if (ae && (ae.tagName === 'INPUT' || ae.tagName === 'TEXTAREA')) return;
                    var btn = document.getElementById('lp-ev-clear');
                    if (btn) btn.click();
                });
                return '';
            }
            """,
            Output("lp-ev-kbd-init", "data"),
            Input("lp-ev-kbd-init", "data"),
        )

        app.clientside_callback(
            """
            function(clickData, rowClicks) {
                var m = window.__lp_mods || {shift: false, ctrl: false};
                return {shift: !!m.shift, ctrl: !!m.ctrl};
            }
            """,
            Output("lp-ev-mods", "data"),
            Input("live-graph", "clickData"),
            Input({"type": "lp-ev-row", "id": ALL}, "n_clicks"),
        )

        # ── Event selection ────────────────────────────────────────────
        @app.callback(
            Output("lp-ev-selected", "data"),
            Output("lp-ev-anchor",   "data"),
            Input("live-graph", "clickData"),
            Input({"type": "lp-ev-row", "id": ALL}, "n_clicks"),
            Input("lp-ev-clear", "n_clicks"),
            State("lp-ev-selected", "data"),
            State("lp-ev-anchor",   "data"),
            State("lp-ev-mods",     "data"),
            prevent_initial_call=True,
        )
        def _update_ev_selection(click_data, row_clicks, clear_n, selected, anchor, mods):
            trig = ctx.triggered_id
            if trig == "lp-ev-clear" and (clear_n or 0) > 0:
                plotter._event_selection = []
                plotter._apply_event_selection([])
                return [], None

            selected = list(selected or [])
            shift = bool((mods or {}).get("shift"))
            ctrl  = bool((mods or {}).get("ctrl"))

            clicked_id = None
            if trig == "live-graph" and click_data:
                for pt in click_data.get("points", []):
                    if pt.get("curveNumber") == plotter._strip_trace_idx:
                        x_clicked = pt.get("x")
                        for ev in plotter._events:
                            if abs(ev["t_elapsed"] - x_clicked) < 1e-9:
                                clicked_id = ev["id"]
                                break
            elif isinstance(trig, dict) and trig.get("type") == "lp-ev-row":
                if row_clicks is None or all((c or 0) == 0 for c in row_clicks):
                    return no_update, no_update
                clicked_id = trig.get("id")

            if clicked_id is None:
                return no_update, no_update

            if shift and anchor is not None and anchor != clicked_id and isinstance(trig, dict):
                ids = [e["id"] for e in plotter._events]
                try:
                    ii, jj = ids.index(anchor), ids.index(clicked_id)
                except ValueError:
                    ii, jj = -1, -1
                if ii >= 0 and jj >= 0:
                    lo, hi = min(ii, jj), max(ii, jj)
                    rng = ids[lo:hi + 1]
                    seen = set(selected); merged = list(selected)
                    for r in rng:
                        if r not in seen:
                            merged.append(r); seen.add(r)
                    plotter._event_selection = merged
                    plotter._apply_event_selection(merged)
                    return merged, clicked_id

            if clicked_id in selected:
                new_sel = [x for x in selected if x != clicked_id]
            else:
                new_sel = selected + [clicked_id]
            plotter._event_selection = new_sel
            plotter._apply_event_selection(new_sel)
            return new_sel, clicked_id

        @app.callback(
            Output("live-graph", "figure", allow_duplicate=True),
            Input("lp-ev-selected", "data"),
            prevent_initial_call=True,
        )
        def _ev_selection_to_figure(selected):
            # Sync version so refresh doesn't overwrite this update on next tick
            plotter._last_sent_version = plotter._data_version
            return plotter._fig_dict or no_update

        # Suppress the Flask "Serving Flask app" CLI banner
        try:
            import flask.cli
            flask.cli.show_server_banner = lambda *args, **kwargs: None
        except (ImportError, AttributeError):
            pass

        from werkzeug.serving import make_server

        # Auto-stop any previous plotter occupying this port so repeated
        # DashPlotter.load() / DashPlotter() calls in a notebook just work.
        prev = _port_registry.get(self.port)
        if prev is not None and prev is not self:
            prev.stop(_silent=True)

        # Bind socket on the calling thread so port errors surface immediately
        srv = make_server("127.0.0.1", self.port, app.server)
        self._wsgi_server = srv
        _port_registry[self.port] = self

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
        _port_registry.pop(self.port, None)
        if not _silent:
            print("Live plot server stopped.")


# Backward-compatible alias — existing code using LivePlotter(...) keeps working
LivePlotter = DashPlotter
